# SPDX-License-Identifier: Apache-2.0
"""CPU-only pressure feedback, admission, and shared future commitments.

Thresholds are explicit draft configuration, not fitted net-benefit gates.
One coordinator serializes admission and migration against this ledger. Physical
free blocks never include pending prompt tokens or unallocated reservations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace

from .bounded_length import BoundedLength

VERSION = "pressure-balanced-v1-draft"


@dataclass(frozen=True)
class PressureConfig:
    max_age_s: float = 2.0
    min_decode_samples: int = 20
    spare_ratio: float = 0.8
    enter_ratio: float = 1.2
    leave_ratio: float = 1.0
    imbalance_enter: float = 0.4
    imbalance_leave: float = 0.2
    sustain_s: float = 1.0
    max_tick_gap_s: float = 0.5
    cooldown_s: float = 2.0
    # Both prompt/prefill boundaries must be supplied from pilot calibration.
    prompt_boundary: int = 0
    prefill_pending_budget: int = 0
    prefill_waiting_budget: int = 0
    capacity_guard_blocks: int = 0
    survival_horizon: int = 256
    min_survival_lower: float = 0.5
    length_resolution: int = 512

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("configuration must be finite and nonnegative")
        if not (
            0 < self.spare_ratio < self.leave_ratio < self.enter_ratio
            and self.imbalance_leave < self.imbalance_enter
            and 0 <= self.min_survival_lower <= 1
            and self.max_age_s > 0
            and self.max_tick_gap_s > 0
            and self.length_resolution > 0
            and self.min_decode_samples > 0
            and self.prompt_boundary > 0
            and self.prefill_pending_budget > 0
            and self.prefill_waiting_budget > 0
            and self.capacity_guard_blocks > 0
        ):
            raise ValueError(
                "explicit calibrated/draft budgets and hysteresis required"
            )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class PoolReading:
    pool_id: str
    sampled_at: float
    published_at: float
    free_blocks: int | None
    block_size: int
    context_limit: int
    decode_ratio: float | None
    decode_samples: int
    decode_sampled_at: float | None
    # IDLE must be established by a current empty scheduler report, not samples=0.
    idle: bool
    pending_prompt_tokens: int | None
    waiting: int | None
    growth_blocks_s: float | None
    guard_time_s: float | None = None
    urgency: float | None = None
    native_itl_p95_s: float | None = None
    slow_interval_fraction: float | None = None
    actual_decode_requests: int | None = None
    actual_decode_context: int | None = None
    prefill_batch_tokens: int | None = None
    prefill_batch_s: float | None = None


@dataclass(frozen=True)
class Pressure:
    decode: float | None
    prefill: float | None
    capacity: float | None
    zone: str
    missing: tuple[str, ...]


def classify(pool: PoolReading, cfg: PressureConfig, now: float) -> Pressure:
    missing = []
    if not all(
        math.isfinite(v) and 0 <= now - v <= cfg.max_age_s
        for v in (pool.sampled_at, pool.published_at)
    ):
        return Pressure(None, None, None, "UNKNOWN", ("pool timestamp",))
    if pool.block_size <= 0 or pool.context_limit <= 0:
        return Pressure(None, None, None, "UNKNOWN", ("KV geometry",))
    decode = pool.decode_ratio
    if pool.idle and pool.waiting == 0 and pool.pending_prompt_tokens == 0:
        decode = 0.0
    elif (
        decode is None
        or not math.isfinite(decode)
        or decode < 0
        or pool.decode_samples < cfg.min_decode_samples
        or pool.decode_sampled_at is None
        or not 0 <= now - pool.decode_sampled_at <= cfg.max_age_s
    ):
        decode = None
        missing.append("decode service")
    if (
        pool.pending_prompt_tokens is None
        or pool.waiting is None
        or pool.pending_prompt_tokens < 0
        or pool.waiting < 0
    ):
        prefill = None
        missing.append("prefill service")
    else:
        prefill = max(
            pool.pending_prompt_tokens / cfg.prefill_pending_budget,
            pool.waiting / cfg.prefill_waiting_budget,
        )
    if pool.free_blocks is None or pool.free_blocks < 0:
        capacity = None
        missing.append("physical free blocks")
    else:
        capacity = cfg.capacity_guard_blocks / max(1, pool.free_blocks)
    values = (decode, prefill, capacity)
    if any(v is not None and v >= cfg.enter_ratio for v in values):
        zone = "OVERLOAD"
    elif missing:
        zone = "UNKNOWN"
    elif max(values) <= cfg.spare_ratio:
        zone = "SPARE"
    else:
        zone = "CRITICAL"
    return Pressure(decode, prefill, capacity, zone, tuple(missing))


@dataclass
class Reservation:
    key: str
    request_id: str
    pool_id: str
    blocks: int
    decode_ratio: float
    prefill_tokens: int
    kind: str
    allocated_blocks: int = 0
    service_observed: bool = False
    prefill_observed: bool = False
    state: str = "ACTIVE"


class ReservationLedger:
    """Shared admission/migration commitments, serialized by the coordinator.

    Commit alone does not free a promise: release only after scheduler evidence
    accounts for its allocation and service. Cancel/failure after a possible
    START require cleanup evidence. Keys are never reused.
    """

    def __init__(self) -> None:
        self.entries: dict[str, Reservation] = {}

    def pending(self, pool_id: str) -> tuple[int, float, int]:
        rows = [
            r
            for r in self.entries.values()
            if r.pool_id == pool_id and r.state in {"ACTIVE", "COMMITTED"}
        ]
        return (
            sum(r.blocks - r.allocated_blocks for r in rows),
            sum(r.decode_ratio for r in rows if not r.service_observed),
            sum(r.prefill_tokens for r in rows if not r.prefill_observed),
        )

    def channel_available(self, pool_id: str) -> bool:
        return not any(
            r.pool_id == pool_id and r.kind == "migration" and r.state == "ACTIVE"
            for r in self.entries.values()
        )

    def reserve(self, value: Reservation, pool: PoolReading) -> None:
        if value.key in self.entries:
            raise ValueError("reservation key already used")
        if any(
            r.request_id == value.request_id
            and r.state in {"ACTIVE", "COMMITTED"}
            and not (
                value.kind == "migration"
                and r.kind == "admission"
                and r.pool_id != value.pool_id
                and r.state == "COMMITTED"
            )
            for r in self.entries.values()
        ):
            raise ValueError("request already reserved")
        if (
            value.kind not in {"admission", "migration"}
            or value.blocks < 0
            or not math.isfinite(value.decode_ratio)
            or value.decode_ratio < 0
            or value.prefill_tokens < 0
            or value.pool_id != pool.pool_id
            or value.state != "ACTIVE"
            or value.allocated_blocks
            or value.service_observed
            or value.prefill_observed
        ):
            raise ValueError("invalid new reservation")
        if value.kind == "migration" and not self.channel_available(value.pool_id):
            raise ValueError("migration channel occupied")
        blocks, _, _ = self.pending(pool.pool_id)
        if pool.free_blocks is None or value.blocks + blocks > pool.free_blocks:
            raise ValueError("target cannot hold shared commitments")
        self.entries[value.key] = replace(value)

    def observe(
        self,
        key: str,
        *,
        allocated_blocks: int,
        service_observed: bool,
        prefill_observed: bool | None = None,
    ) -> None:
        """Call only with matching, published scheduler allocation evidence."""
        row = self.entries[key]
        if row.state not in {"ACTIVE", "COMMITTED"}:
            raise ValueError("reservation already settled")
        if not row.allocated_blocks <= allocated_blocks <= row.blocks:
            raise ValueError("allocation acknowledgement must be monotonic")
        if row.service_observed and not service_observed:
            raise ValueError("service acknowledgement cannot regress")
        if (
            prefill_observed is not None
            and row.prefill_observed
            and not prefill_observed
        ):
            raise ValueError("prefill acknowledgement cannot regress")
        row.allocated_blocks = allocated_blocks
        row.service_observed = service_observed
        if prefill_observed is not None:
            row.prefill_observed = prefill_observed
        elif service_observed:
            row.prefill_observed = True
        if (
            row.state == "COMMITTED"
            and allocated_blocks == row.blocks
            and service_observed
        ):
            row.state = "ACCOUNTED"

    def retire_admission(self, key: str, outcome: str, *, evidence: bool) -> None:
        row = self.entries[key]
        if (
            row.kind != "admission"
            or outcome not in {"COMPLETED", "MOVED"}
            or not evidence
        ):
            raise ValueError("admission retirement requires matching release evidence")
        if row.state == outcome:
            return
        if row.state not in {"ACTIVE", "COMMITTED", "ACCOUNTED"}:
            raise ValueError("conflicting admission retirement")
        row.state = outcome

    def complete(self, key: str, *, evidence: bool) -> None:
        row = self.entries[key]
        if not evidence or row.state not in {"COMMITTED", "ACCOUNTED", "COMPLETED"}:
            raise ValueError(
                "completion requires confirmed owner-stream end and release"
            )
        row.state = "COMPLETED"

    def settle(self, key: str, outcome: str, *, evidence: bool) -> None:
        row = self.entries[key]
        if outcome not in {"COMMITTED", "CANCELLED", "FAILED"} or not evidence:
            raise ValueError("matching ownership/release or cleanup evidence required")
        if row.state == outcome or (
            row.state == "ACCOUNTED" and outcome == "COMMITTED"
        ):
            return
        if row.state != "ACTIVE":
            raise ValueError("conflicting terminal outcome")
        row.state = outcome
        if (
            outcome == "COMMITTED"
            and row.allocated_blocks == row.blocks
            and row.service_observed
        ):
            row.state = "ACCOUNTED"

    def snapshot(self) -> list[dict]:
        return [asdict(r) for r in self.entries.values()]


@dataclass(frozen=True)
class Candidate:
    request_id: str
    owner: str
    sampled_at: float
    published_at: float
    generated: int
    prompt_tokens: int
    max_output_tokens: int
    # Observed unique releasable blocks; None for unverified/shared ownership.
    releasable_blocks: int | None
    owned_blocks: int | None
    transfer_bytes: int | None
    remaining: BoundedLength | None
    prediction_status: str
    source_decode_relief: float | None
    target_decode_cost: float | None
    estimated_release_s: float | None
    finished: bool = False
    migrating: bool = False
    pmf_reference: str | None = None
    prediction_age_s: float | None = None
    prediction_lag: int | None = None
    action_available: bool = True
    inventory_reference: str | None = None

    def audit(self) -> dict:
        return {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "remaining"
        }


@dataclass(frozen=True)
class PressureDecision:
    action: str
    reason: str
    request_id: str | None
    source: Pressure
    target: Pressure
    candidates: tuple[dict, ...] = ()
    capacity_path: bool = False
    required_release_blocks: int = 0
    version: str = VERSION
    config_sha256: str = ""


class PressurePolicy:
    def __init__(self, cfg: PressureConfig, ledger: ReservationLedger) -> None:
        self.cfg, self.ledger = cfg, ledger
        self._since: float | None = None
        self._last_tick: float | None = None
        self._last_start = -math.inf
        self._latched = False

    def note_started(self, now: float) -> None:
        self._last_start = now
        self._since = None
        self._latched = False

    def decide(
        self,
        now: float,
        source: PoolReading,
        target: PoolReading,
        candidates: list[Candidate],
    ) -> PressureDecision:
        cfg = self.cfg
        s, t = classify(source, cfg, now), classify(target, cfg, now)
        if self._last_tick is not None and now < self._last_tick:
            raise ValueError("decision time regressed")
        if self._last_tick is not None and now - self._last_tick > cfg.max_tick_gap_s:
            self._since = None
        self._last_tick = now
        if len({c.request_id for c in candidates}) != len(candidates):
            raise ValueError("duplicate candidate identity")
        emergency = (
            s.capacity is not None
            and s.capacity >= 1
            and source.growth_blocks_s is not None
            and math.isfinite(source.growth_blocks_s)
            and source.growth_blocks_s > 0
        )
        required = (
            max(1, cfg.capacity_guard_blocks - source.free_blocks) if emergency else 0
        )

        def decision(reason, selected=None, rows=()):
            if not rows:
                rows = tuple(
                    {
                        "request": c.audit(),
                        "remaining": None,
                        "rejections": ["trigger not satisfied: " + reason],
                    }
                    for c in candidates
                )
            return PressureDecision(
                "START_SHADOW" if selected else "STAY",
                reason,
                selected,
                s,
                t,
                tuple(rows),
                emergency,
                required,
                config_sha256=cfg.sha256,
            )

        if not self.ledger.channel_available(target.pool_id):
            self._since = None
            return decision("migration channel occupied")
        if not emergency:
            active_source = [v for v in (s.decode, s.prefill) if v is not None]
            active_target = [v for v in (t.decode, t.prefill) if v is not None]
            threshold = cfg.leave_ratio if self._latched else cfg.enter_ratio
            gap = cfg.imbalance_leave if self._latched else cfg.imbalance_enter
            needed = (
                not s.missing
                and t.zone == "SPARE"
                and max(active_source) >= threshold
                and max(active_source) - max(active_target) >= gap
            )
            if not needed:
                self._since = None
                self._latched = False
                return decision("source relief or target spare evidence absent")
            self._latched = True
            if self._since is None:
                self._since = now
            if now - self._since < cfg.sustain_s:
                return decision("imbalance not sustained")
            if now - self._last_start < cfg.cooldown_s:
                return decision("migration cooldown")
        # Capacity bypasses persistence/cooldown, never physical feasibility.
        if t.capacity is None or "pool timestamp" in t.missing:
            return decision("target physical capacity evidence absent")
        pending_blocks, pending_decode, pending_prefill = self.ledger.pending(
            target.pool_id
        )
        feasible, rows = [], []
        for c in candidates:
            reasons = []
            if not c.action_available:
                reasons.append("no bound source stream/action adapter")
            cap = max(0, min(8192, c.max_output_tokens) - c.generated)
            required_blocks = math.ceil(
                (c.prompt_tokens + min(8192, c.max_output_tokens)) / target.block_size
            )
            if c.finished or c.migrating or c.owner != source.pool_id:
                reasons.append("not active local owner")
            if any(
                row.request_id == c.request_id
                and row.state in {"ACTIVE", "COMMITTED"}
                and not (
                    row.kind == "admission"
                    and row.pool_id == source.pool_id
                    and row.state == "COMMITTED"
                )
                for row in self.ledger.entries.values()
            ):
                reasons.append("request has an unsettled commitment")
            if (
                any(
                    type(v) is not int or v < 0
                    for v in (c.generated, c.prompt_tokens, c.max_output_tokens)
                )
                or cap == 0
            ):
                reasons.append("stop contract exhausted/invalid")
            if not all(
                math.isfinite(v) and 0 <= now - v <= cfg.max_age_s
                for v in (c.sampled_at, c.published_at)
            ):
                reasons.append("inventory stale/future")
            if c.prompt_tokens + min(8192, c.max_output_tokens) > target.context_limit:
                reasons.append("target context infeasible")
            if c.prompt_tokens + min(8192, c.max_output_tokens) > source.context_limit:
                reasons.append("source stop contract was not context feasible")
            if pending_blocks + required_blocks > target.free_blocks:
                reasons.append("target physical commitments exceed free blocks")
            if (
                c.owned_blocks is None
                or c.releasable_blocks is None
                or c.owned_blocks <= 0
                or not 0 < c.releasable_blocks <= c.owned_blocks
            ):
                reasons.append("physical release unknown/empty")
            if c.transfer_bytes is None or c.transfer_bytes < 0:
                reasons.append("transfer size unknown")
            needed_release = required
            if (
                emergency
                and c.estimated_release_s is not None
                and math.isfinite(c.estimated_release_s)
                and c.estimated_release_s >= 0
            ):
                needed_release = max(
                    required,
                    math.ceil(
                        cfg.capacity_guard_blocks
                        - source.free_blocks
                        + source.growth_blocks_s * c.estimated_release_s
                    ),
                )
            if emergency and (c.releasable_blocks or 0) < needed_release:
                reasons.append("insufficient capacity relief")
            # guard is an early warning; do not reject merely for U>1 or
            # release later than guard. Physical exhaustion is separate.
            if emergency and (
                c.estimated_release_s is None
                or not math.isfinite(c.estimated_release_s)
                or c.estimated_release_s < 0
                or c.estimated_release_s * source.growth_blocks_s >= source.free_blocks
            ):
                reasons.append("cannot establish release before physical exhaustion")
            remaining = c.remaining
            valid_prediction = (
                c.prediction_status == "AVAILABLE"
                and remaining
                and remaining.cap == cap
            )
            summary = (
                remaining.summary(cfg.survival_horizon) if valid_prediction else None
            )
            if not valid_prediction and not emergency:
                reasons.append("prediction missing/stale/cap mismatch")
            if summary and summary["p_gt_bounds"][0] < cfg.min_survival_lower:
                reasons.append("insufficient sustained work probability")
            costs = (c.source_decode_relief, c.target_decode_cost)
            valid_costs = all(
                v is not None and math.isfinite(v) and v >= 0 for v in costs
            )
            if not emergency:
                if not valid_costs or s.decode is None or t.decode is None:
                    reasons.append("post-move service estimate missing")
                elif (
                    t.decode + pending_decode + c.target_decode_cost > cfg.leave_ratio
                    or t.decode + pending_decode + c.target_decode_cost
                    > max(0, s.decode - c.source_decode_relief) + cfg.imbalance_enter
                    or (target.pending_prompt_tokens + pending_prefill)
                    / cfg.prefill_pending_budget
                    > cfg.spare_ratio
                ):
                    reasons.append("post-move overload/reverse imbalance")
            elif (
                t.prefill is None
                or t.decode is None
                or not valid_costs
                or t.decode + pending_decode + c.target_decode_cost > cfg.enter_ratio
                or t.prefill + pending_prefill / cfg.prefill_pending_budget
                > cfg.enter_ratio
            ):
                reasons.append("emergency target service budget unavailable")
            rows.append(
                {
                    "request": c.audit(),
                    "remaining": summary,
                    "rejections": reasons,
                    "target_required_blocks": required_blocks,
                    "capacity_required_release_blocks": needed_release,
                    "prediction_fallback": bool(not valid_prediction and emergency),
                }
            )
            if not reasons:
                feasible.append((c, summary))
        if not feasible:
            return decision("no feasible candidate", rows=rows)
        # Compare to the longest candidate, then treat differences below the
        # predictor's configured resolution as one tier (no boundary artifacts).
        predicted = [(c, summary) for c, summary in feasible if summary]
        if predicted:
            longest = max(summary["mean"] for _, summary in predicted)
            tier = [
                c
                for c, summary in predicted
                if longest - summary["mean"] <= cfg.length_resolution
            ]
        else:
            tier = [c for c, _ in feasible]
        selected = min(
            tier, key=lambda c: (-c.releasable_blocks, c.transfer_bytes, c.request_id)
        )
        return decision(
            "capacity relief"
            if emergency
            else "sustained pressure; longest tier, release, bytes, id",
            selected.request_id,
            rows,
        )


@dataclass(frozen=True)
class Admission:
    request_id: str
    pool_id: str
    max_output_tokens: int
    reservation_key: str


def admit(
    request_id: str,
    prompt_tokens: int,
    requested_output: int,
    pools: tuple[PoolReading, PoolReading],
    cfg: PressureConfig,
    ledger: ReservationLedger,
    now: float,
    key: str,
    decode_costs: dict[str, float],
    allowed_pool_id: str | None = None,
) -> Admission:
    """Fix a feasible total stop cap and reserve before dispatching a request.

    Callers must submit the returned max_output_tokens unchanged on both pools.
    Pressure may override the prompt preference; overload/missing pools queue.
    """
    if prompt_tokens < 0 or requested_output <= 0:
        raise ValueError("invalid request lengths")
    output_cap = min(8192, requested_output)
    preferred = pools[1] if prompt_tokens >= cfg.prompt_boundary else pools[0]
    eligible = []
    for pool in pools:
        if allowed_pool_id is not None and pool.pool_id != allowed_pool_id:
            continue
        p = classify(pool, cfg, now)
        if p.missing or p.zone == "OVERLOAD":
            continue
        blocks, decode, pending = ledger.pending(pool.pool_id)
        cost = decode_costs.get(pool.pool_id)
        if cost is None or not math.isfinite(cost) or cost < 0:
            continue
        capped = min(output_cap, pool.context_limit - prompt_tokens)
        required = math.ceil((prompt_tokens + capped) / pool.block_size)
        if (
            capped <= 0
            or required + blocks > pool.free_blocks
            or p.decode + decode + cost > cfg.leave_ratio
            or (
                (pool.pending_prompt_tokens + pending + prompt_tokens)
                / cfg.prefill_pending_budget
                > cfg.enter_ratio
                and not (pool.idle and pending == 0 and pool.pending_prompt_tokens == 0)
            )
        ):
            continue
        eligible.append((pool, capped, required, cost, p))
    if not eligible:
        raise ValueError("queue: no feasible admission budget")
    # Keep prompt preference unless it is critical and the alternative is spare.
    eligible.sort(
        key=lambda row: (row[4].zone != "SPARE", row[0].pool_id != preferred.pool_id)
    )
    pool, capped, blocks, cost, _ = eligible[0]
    ledger.reserve(
        Reservation(
            key, request_id, pool.pool_id, blocks, cost, prompt_tokens, "admission"
        ),
        pool,
    )
    return Admission(request_id, pool.pool_id, capped, key)
