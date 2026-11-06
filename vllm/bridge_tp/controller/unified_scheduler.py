# SPDX-License-Identifier: Apache-2.0
"""Unified scheduling (user 11-06): few observable rules, migration fixes the rest.

Admission keeps two rules: an engine queue limit and a prompt that fits. Every
active instance is then ranked by one fullness, the largest of three shares of
its own limit, whatever its TP degree:

- running slots: requests in the engine plus routed ones it has not reported,
  over that instance's ``max_num_seqs``;
- KV: used plus unallocated prompt blocks (with this request), over its blocks;
- speed: the exponentially smoothed inter-token interval over ``ITL_REFERENCE_S``.

No output cap is read as a final length and no cost model is fitted: a request
that grows too large for where it was placed is migration's job. Migration has
two triggers on the same fullness: KV (safety, immediate) and pressure (benefit,
sustained, with hysteresis and cooldown). It moves the request with the longest
current context to the least full instance that holds it.

All values here are initial values, not tuned (plan 11-06).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

ITL_REFERENCE_S = 0.035
ROUTE_TAU_S = 5.0
MIGRATE_TAU_S = 10.0
GUARD_BLOCKS = 125
ENGINE_QUEUE_LIMIT = 8
KV_TRIGGER = 0.9
PRESSURE_TRIGGER = 0.9
PRESSURE_SUSTAIN_S = 10.0
PRESSURE_GAP = 0.3
TARGET_AFTER_MAX = 0.7
COOLDOWN_S = 15.0


@dataclass
class _Smoothed:
    updated_at: float
    route: float
    migrate: float


class SmoothedItl:
    """Two exponentially weighted inter-token intervals per instance.

    new = a * latest + (1 - a) * previous, with a = 1 - exp(-dt / tau) from the
    real time since the last update. The latest value is the median of the
    intervals that arrived since then. An idle instance decays towards zero; one
    with running requests but no new token keeps its value (a stall is handled
    by migration and gap metrics, not by calling it idle).
    """

    def __init__(self, route_tau_s=ROUTE_TAU_S, migrate_tau_s=MIGRATE_TAU_S):
        if route_tau_s <= 0 or migrate_tau_s <= 0:
            raise ValueError("smoothing time constants must be positive")
        self.taus = (route_tau_s, migrate_tau_s)
        self._values: dict[str, _Smoothed] = {}

    def update(self, pool_id, now, intervals, active):
        intervals = [v for v in intervals if math.isfinite(v) and v >= 0]
        if intervals:
            latest = statistics.median(intervals)
        elif active == 0:
            latest = 0.0
        else:
            latest = None
        current = self._values.get(pool_id)
        if current is None:
            if latest is not None:
                self._values[pool_id] = _Smoothed(now, latest, latest)
            return
        dt = now - current.updated_at
        if dt <= 0:
            return
        current.updated_at = now
        if latest is None:
            return
        route_a, migrate_a = (1 - math.exp(-dt / tau) for tau in self.taus)
        current.route = route_a * latest + (1 - route_a) * current.route
        current.migrate = migrate_a * latest + (1 - migrate_a) * current.migrate

    def value(self, pool_id, kind="route"):
        current = self._values.get(pool_id)
        if current is None:
            return None
        return current.route if kind == "route" else current.migrate

    def snapshot(self):
        return {
            k: dict(updated_at=v.updated_at, route_s=v.route, migrate_s=v.migrate)
            for k, v in self._values.items()
        }


def used_blocks(reading):
    """Allocated plus not yet allocated prompt blocks, or None without evidence."""
    if (
        reading.total_blocks is None
        or reading.total_blocks <= 0
        or reading.free_blocks is None
        or reading.free_blocks < 0
    ):
        return None
    unallocated = reading.unallocated_prompt_blocks
    unallocated = unallocated if type(unallocated) is int and unallocated > 0 else 0
    return reading.total_blocks - reading.free_blocks + unallocated


def fullness(reading, slots_max, itl_s, *, extra_requests=0, extra_blocks=0):
    """The three shares and their maximum; None components count as zero."""
    if type(slots_max) is not int or slots_max <= 0:
        raise ValueError("instance slot limit must be a positive integer")
    requests = reading.engine_requests if type(reading.engine_requests) is int else 0
    slots = (requests + extra_requests) / slots_max
    used = used_blocks(reading)
    kv = None if used is None else (used + extra_blocks) / reading.total_blocks
    speed = None if itl_s is None else itl_s / ITL_REFERENCE_S
    return dict(
        slots=slots,
        kv=kv,
        speed=speed,
        value=max(v for v in (slots, kv, speed, 0.0) if v is not None),
    )


def prompt_room(reading, pending_blocks, guard_blocks=GUARD_BLOCKS):
    """Free blocks a new prompt may take, after commitments and the guard."""
    if reading.free_blocks is None or reading.block_size <= 0:
        return None
    unallocated = reading.unallocated_prompt_blocks
    unallocated = unallocated if type(unallocated) is int and unallocated > 0 else 0
    return reading.free_blocks - unallocated - pending_blocks - guard_blocks


def rank_admission(rows):
    """Least full first; ties by engine queue, then instance id."""
    return sorted(
        rows,
        key=lambda r: (r["fullness"]["value"], r["waiting"] or 0, r["pool_id"]),
    )


@dataclass
class MigrationState:
    """Pressure-trigger timers and per-instance cooldowns."""

    pressure_since: dict = field(default_factory=dict)
    cooldown_until: dict = field(default_factory=dict)

    def cooling(self, pool_id, now):
        return now < self.cooldown_until.get(pool_id, -math.inf)

    def started(self, source, target, now):
        for pool_id in (source, target):
            self.cooldown_until[pool_id] = now + COOLDOWN_S
        self.pressure_since.pop(source, None)


def migration_trigger(
    source_full, source_kv_exhausting, best_target_full, state, source_id, now
):
    """("KV" | "PRESSURE" | None, reason). Advances the pressure timer."""
    kv = source_full["kv"]
    if (kv is not None and kv >= KV_TRIGGER) or source_kv_exhausting:
        state.pressure_since.pop(source_id, None)
        return "KV", "source KV at the trigger or about to run out"
    if source_full["value"] < PRESSURE_TRIGGER:
        state.pressure_since.pop(source_id, None)
        return None, "source below the pressure trigger"
    since = state.pressure_since.setdefault(source_id, now)
    if now - since < PRESSURE_SUSTAIN_S:
        return None, "pressure not sustained yet"
    if best_target_full is None:
        return None, "no target fullness"
    if source_full["value"] - best_target_full["value"] < PRESSURE_GAP:
        return None, "target not clearly less full"
    if state.cooling(source_id, now):
        return None, "source cooling down"
    return "PRESSURE", "sustained pressure above the least full target"
