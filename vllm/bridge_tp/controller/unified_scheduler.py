# SPDX-License-Identifier: Apache-2.0
"""Unified scheduling (user 11-06): few observable rules, migration fixes the rest.

Admission keeps two rules: an engine queue limit and a prompt that fits. Every
active instance is then ranked by one fullness, the largest of three shares of
its own limit, whatever its TP degree:

- running slots: requests in the engine plus routed ones it has not reported,
  over that instance's ``max_num_seqs``;
- KV: used plus unallocated prompt blocks (with this request), over its blocks;
- speed: the exponentially smoothed mean inter-token interval over
  ``ITL_REFERENCE_S``.

No output cap is read as a final length and no cost model is fitted: a request
that grows too large for where it was placed is migration's job. Migration has
two triggers on the same fullness: KV (safety, immediate) and pressure (benefit,
sustained, with hysteresis and cooldown). It moves the request with the longest
current context to the least full instance that holds it.

All values here are initial values, not tuned (plan 11-06).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Mean inter-token interval, as TPOT measures it, kept below the strict 35ms
# line so routing backs off before an instance crosses it (user 11-06). The
# earlier 28ms was against the median interval, which hid prefill stalls.
ITL_REFERENCE_S = 0.032
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


# Token rate standing in for an instance first seen idle, so its value is zero.
_IDLE_TOKENS = 1e-9


@dataclass
class _Smoothed:
    updated_at: float
    # Per kind: smoothed stream-seconds per second and tokens per second.
    seconds: list
    tokens: list


class SmoothedItl:
    """Exponentially smoothed mean inter-token interval per instance.

    Each update brings the decoding stream-seconds and the tokens since the
    previous one; their ratio is the mean interval, the quantity TPOT averages,
    so prefill stalls count instead of being filtered out. Both are smoothed as
    rates, new = a * latest + (1 - a) * previous with a = 1 - exp(-dt / tau)
    from the real elapsed time, and the value is their ratio. A stall with no
    new token raises it while it lasts; an instance with no stream at all
    decays towards zero.
    """

    def __init__(self, route_tau_s=ROUTE_TAU_S, migrate_tau_s=MIGRATE_TAU_S):
        if route_tau_s <= 0 or migrate_tau_s <= 0:
            raise ValueError("smoothing time constants must be positive")
        self.taus = (route_tau_s, migrate_tau_s)
        self._values: dict[str, _Smoothed] = {}

    def update(self, pool_id, now, seconds, tokens, active):
        if not (math.isfinite(seconds) and seconds >= 0) or tokens < 0:
            raise ValueError("invalid decode progress")
        current = self._values.get(pool_id)
        if current is None:
            # The first call only starts the clock: rates need an elapsed time.
            self._values[pool_id] = _Smoothed(now, [None] * 2, [None] * 2)
            return
        dt = now - current.updated_at
        if dt <= 0:
            return
        current.updated_at = now
        rate = (seconds / dt, tokens / dt)
        if current.tokens[0] is None:
            if tokens > 0:
                current.seconds, current.tokens = [rate[0]] * 2, [rate[1]] * 2
            elif active == 0:
                current.seconds, current.tokens = [0.0] * 2, [_IDLE_TOKENS] * 2
            return
        for i, tau in enumerate(self.taus):
            a = 1 - math.exp(-dt / tau)
            current.seconds[i] = a * rate[0] + (1 - a) * current.seconds[i]
            if active or tokens:
                current.tokens[i] = a * rate[1] + (1 - a) * current.tokens[i]

    def value(self, pool_id, kind="route"):
        current = self._values.get(pool_id)
        if current is None:
            return None
        i = 0 if kind == "route" else 1
        tokens = current.tokens[i]
        return current.seconds[i] / tokens if tokens else None

    def snapshot(self):
        return {
            k: dict(
                updated_at=v.updated_at,
                route_s=self.value(k, "route"),
                migrate_s=self.value(k, "migrate"),
            )
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


def fullness(
    reading,
    slots_max,
    itl_s,
    *,
    extra_requests=0,
    extra_blocks=0,
    itl_reference_s=ITL_REFERENCE_S,
):
    """The three shares and their maximum; None components count as zero."""
    if type(slots_max) is not int or slots_max <= 0:
        raise ValueError("instance slot limit must be a positive integer")
    requests = reading.engine_requests if type(reading.engine_requests) is int else 0
    slots = (requests + extra_requests) / slots_max
    used = used_blocks(reading)
    kv = None if used is None else (used + extra_blocks) / reading.total_blocks
    if not itl_reference_s > 0:
        raise ValueError("ITL reference must be positive")
    speed = None if itl_s is None else itl_s / itl_reference_s
    return dict(
        slots=slots,
        kv=kv,
        speed=speed,
        value=max(v for v in (slots, kv, speed, 0.0) if v is not None),
    )


def prompt_room(reading, pending_blocks, guard_blocks=GUARD_BLOCKS):
    """Free blocks a new prompt or a migrated request may take, after
    commitments (prompts not yet allocated, reserved moves) and the guard.
    """
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
