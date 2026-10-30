# SPDX-License-Identifier: Apache-2.0
"""CPU-only routing, pressure feedback, and migration commitments.

Thresholds are explicit draft configuration, not fitted net-benefit gates.
One coordinator serializes routing and migration bookkeeping. New ingress
records carry no KV or decode promise. Physical free blocks come from allocation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace

from . import pressure_workload as workload
from .bounded_length import BoundedLength
from .pressure_relief import RoutingRelief

VERSION = "pressure-balanced-v2-draft"


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
    # Explicit draft boundaries; backlog and one scheduled step are distinct.
    prompt_boundary: int = 0
    very_long_prompt_tokens: int = 4096
    prefill_pending_budget: int = 0
    # Complete queued prompts have a separate, bounded backlog budget.
    prefill_backlog_budget: int = 8192
    prefill_waiting_budget: int = 0
    capacity_guard_blocks: int = 0
    # Opt-in KV-release pilot only; existing modes retain zero lookahead.
    capacity_lookahead_s: float = 0.0
    capacity_only: bool = False
    # Opt-in urgent-KV policy: target service/remaining-work are preferences,
    # not vetoes. Physical release/deadline and ownership checks remain.
    kv_emergency_priority: bool = False
    health_service: bool = False
    # Route from passive health/KV evidence, without fixed prompt cutoffs.
    pressure_routing: bool = False
    workload_pressure: bool = False
    routing_relief: bool = False
    health_min_efficiency: float = 0.9
    health_target_load: float = 0.8
    survival_horizon: int = 256
    min_survival_lower: float = 0.5
    length_resolution: int = 512
    shadow_cancel: bool = False
    cancel_sustain_s: float = 3.0
    cancel_recover_eta: float = 1.0

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
            and self.very_long_prompt_tokens >= self.prompt_boundary
            and self.prefill_pending_budget > 0
            and self.prefill_backlog_budget > 0
            and self.prefill_waiting_budget > 0
            and self.capacity_guard_blocks > 0
            and type(self.capacity_only) is bool
            and type(self.kv_emergency_priority) is bool
            and type(self.health_service) is bool
            and type(self.pressure_routing) is bool
            and (not self.pressure_routing or self.health_service)
            and type(self.workload_pressure) is bool
            and type(self.routing_relief) is bool
            and (not self.routing_relief or self.workload_pressure)
            and (
                not self.workload_pressure
                or (self.health_service and self.sustain_s > 0)
            )
            and type(self.shadow_cancel) is bool
            and self.cancel_sustain_s > 0
            and (
                not self.shadow_cancel
                or (
                    self.health_service
                    and self.cancel_recover_eta > self.health_min_efficiency
                )
            )
            and 0 < self.health_min_efficiency <= 1
            and 0 < self.health_target_load < 1
            and not (self.health_service and self.capacity_only)
            and (not self.capacity_only or self.capacity_lookahead_s > 0)
        ):
            raise ValueError(
                "explicit calibrated/draft budgets and hysteresis required"
            )

    @property
    def sha256(self) -> str:
        values = asdict(self)
        # Preserve frozen historical policy identities when the fix is disabled.
        if not self.kv_emergency_priority:
            values.pop("kv_emergency_priority")
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


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
    unallocated_prompt_blocks: int | None = None
    health_status: str = "UNKNOWN"
    health_eta: float | None = None
    health_load: float | None = None
    health_sampled_at: float | None = None
    health_mixed_window: bool = False
    health_reason: str = "not observed"
    workload: dict | None = None
    service_evidence: dict | None = None
    prefill_generation: int | None = None


@dataclass(frozen=True)
class Pressure:
    decode: float | None
    prefill: float | None
    capacity: float | None
    zone: str
    missing: tuple[str, ...]


def health_valid(pool, cfg, now):
    return (
        pool.health_status in {"SUPPORTED", "IDLE"}
        and pool.health_eta is not None
        and math.isfinite(pool.health_eta)
        and pool.health_eta >= 0
        and pool.health_load is not None
        and math.isfinite(pool.health_load)
        and 0 <= pool.health_load <= 1
        and pool.health_sampled_at is not None
        and 0 <= now - pool.health_sampled_at <= 12
        and all(
            math.isfinite(t) and 0 <= now - t <= cfg.max_age_s
            for t in (pool.sampled_at, pool.published_at)
        )
    )


def health_spare(pool, cfg, now):
    current = classify(pool, cfg, now)
    if cfg.workload_pressure:
        return (
            workload.valid(pool, cfg, now)
            and pool.workload["load"] <= cfg.health_target_load
            and pool.pending_prompt_tokens == pool.waiting == 0
            and current.decode is not None
            and current.decode <= cfg.leave_ratio
            and current.prefill is not None
            and current.prefill <= cfg.spare_ratio
            and current.capacity is not None
            and current.capacity < 1
        )
    return (
        health_valid(pool, cfg, now)
        and pool.health_eta >= cfg.health_min_efficiency
        and pool.health_load <= cfg.health_target_load
        and pool.pending_prompt_tokens == pool.waiting == 0
        and current.decode is not None
        and current.decode <= cfg.leave_ratio
    )


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
        step = pool.prefill_batch_tokens
        if step is None and pool.pending_prompt_tokens == 0:
            step = 0
        if step is None or step < 0:
            missing.append("prefill step")
        prefill = max(
            pool.pending_prompt_tokens / cfg.prefill_backlog_budget,
            step / cfg.prefill_pending_budget if step is not None and step >= 0 else 0,
            pool.waiting / cfg.prefill_waiting_budget,
        )
    if cfg.workload_pressure and workload.valid(pool, cfg, now):
        decode = max(decode or 0, pool.workload["load"])
        prefill = pool.workload["prefill_s"] / cfg.sustain_s
        missing = [
            v
            for v in missing
            if v not in {"decode service", "prefill service", "prefill step"}
        ]
    if pool.free_blocks is None or pool.free_blocks < 0:
        capacity = None
        missing.append("physical free blocks")
    else:
        capacity = cfg.capacity_guard_blocks / max(1, pool.free_blocks)
        if (
            cfg.capacity_lookahead_s > 0
            and pool.growth_blocks_s is not None
            and math.isfinite(pool.growth_blocks_s)
            and pool.growth_blocks_s >= 0
            and type(pool.unallocated_prompt_blocks) is int
            and pool.unallocated_prompt_blocks >= 0
        ):
            capacity = (
                cfg.capacity_guard_blocks
                + pool.unallocated_prompt_blocks
                + math.ceil(
                    pool.growth_blocks_s
                    * (cfg.capacity_lookahead_s + now - pool.sampled_at)
                )
            ) / max(1, pool.free_blocks)
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
    """Migration commitments plus zero-resource ingress observation records.

    Commit alone does not free a promise: release only after scheduler evidence
    accounts for its allocation and service. Cancel/failure after a possible
    START require cleanup evidence. Keys are never reused.
    """

    def __init__(self) -> None:
        self.entries: dict[str, Reservation] = {}

    def pending(self, pool_id: str) -> tuple[int, float, int]:
        rows = [
            r
            for r in tuple(self.entries.values())
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
            for r in tuple(self.entries.values())
        )

    def track_ingress(
        self, key: str, request_id: str, pool_id: str, prompt_tokens: int
    ) -> None:
        """Track identity/unpublished prompt demand without reserving capacity."""
        if key in self.entries or any(
            row.request_id == request_id for row in tuple(self.entries.values())
        ):
            raise ValueError("ingress identity already tracked")
        if prompt_tokens < 0:
            raise ValueError("invalid prompt length")
        self.entries[key] = Reservation(
            key, request_id, pool_id, 0, 0.0, prompt_tokens, "admission"
        )

    def unobserved_ingress(self, pool_id: str) -> tuple[int, int]:
        rows = [
            row
            for row in tuple(self.entries.values())
            if row.pool_id == pool_id
            and row.kind == "admission"
            and row.state in {"ACTIVE", "COMMITTED"}
            and not row.prefill_observed
        ]
        return len(rows), sum(row.prefill_tokens for row in rows)

    def routing_reading(self, pool: PoolReading) -> PoolReading:
        """Include dispatched ingress until it appears in scheduler inventory."""
        count, tokens = self.unobserved_ingress(pool.pool_id)
        value = pool.workload
        if value and value.get("status") == "OBSERVED":
            extra = sum(
                workload.slot_cost(value["profile"], row.prefill_tokens)[0]
                for row in tuple(self.entries.values())
                if row.pool_id == pool.pool_id
                and row.kind == "admission"
                and row.state in {"ACTIVE", "COMMITTED"}
                and not row.prefill_observed
            )
            value = dict(
                value,
                load=value["load"] + extra,
                prefill_s=value["prefill_s"]
                + tokens * value["profile"]["prefill_s_per_token"],
            )
        return replace(
            pool,
            workload=value,
            unallocated_prompt_blocks=(
                pool.unallocated_prompt_blocks
                + sum(
                    math.ceil(row.prefill_tokens / pool.block_size)
                    for row in tuple(self.entries.values())
                    if row.pool_id == pool.pool_id
                    and row.kind == "admission"
                    and row.state in {"ACTIVE", "COMMITTED"}
                    and not row.prefill_observed
                )
                if pool.unallocated_prompt_blocks is not None and pool.block_size > 0
                else pool.unallocated_prompt_blocks
            ),
            idle=pool.idle and count == 0,
            pending_prompt_tokens=pool.pending_prompt_tokens + tokens
            if pool.pending_prompt_tokens is not None
            else None,
            waiting=pool.waiting + count if pool.waiting is not None else None,
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
            for r in tuple(self.entries.values())
        ):
            raise ValueError("request already reserved")
        if (
            value.kind != "migration"
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
        return [asdict(r) for r in tuple(self.entries.values())]


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
    target_health_after_supported: bool = False

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
    service_path: str = "legacy"


class PressurePolicy:
    def __init__(self, cfg: PressureConfig, ledger: ReservationLedger) -> None:
        self.cfg, self.ledger = cfg, ledger
        self._since: float | None = None
        self._last_tick: float | None = None
        self._last_start = -math.inf
        self._latched = False
        self._retry_after: dict[str, float] = {}
        self._service_confirmation = None
        self.relief = RoutingRelief()

    def observe_relief(self, now, source):
        if not self.cfg.routing_relief:
            return
        source = self.ledger.routing_reading(source)
        quality_bad = (
            source.decode_ratio is not None
            and math.isfinite(source.decode_ratio)
            and source.decode_ratio >= self.cfg.enter_ratio
            and source.decode_samples >= self.cfg.min_decode_samples
            and source.decode_sampled_at is not None
            and 0 <= now - source.decode_sampled_at <= self.cfg.max_age_s
        )
        if self.relief.update(
            source,
            classify(source, self.cfg, now),
            self.ledger,
            self.cfg,
            now,
            quality_bad,
        ):
            self._since = None
            self._service_confirmation = None

    def _confirm_new_service_intervals(self, source, now, required_windows):
        """Require disjoint, newly observed bad windows before service moves."""
        evidence = source.service_evidence or {}
        count = evidence.get("interval_count")
        slow = evidence.get("slow_interval_count")
        window = evidence.get("window_s")
        budget = evidence.get("slow_budget")
        valid = (
            type(count) is int
            and type(slow) is int
            and 0 <= slow <= count
            and isinstance(window, (int, float))
            and math.isfinite(window)
            and window > 0
            and isinstance(budget, (int, float))
            and math.isfinite(budget)
            and 0 < budget < 1
        )
        if not valid:
            self._service_confirmation = None
            return False
        period = max(self.cfg.sustain_s, window)
        previous = self._service_confirmation
        if previous is None:
            self._service_confirmation = (now, count, slow, 0, window, budget)
            return False
        since, old_count, old_slow, passed, old_window, old_budget = previous
        if (
            count < old_count
            or slow < old_slow
            or (window, budget) != (old_window, old_budget)
        ):
            self._service_confirmation = (now, count, slow, 0, window, budget)
            return False
        if now - since >= period:
            samples, bad = count - old_count, slow - old_slow
            passed = (
                passed + 1
                if samples >= self.cfg.min_decode_samples
                and 0 <= bad <= samples
                and bad / samples >= budget * self.cfg.enter_ratio
                else 0
            )
            self._service_confirmation = (now, count, slow, passed, window, budget)
        return passed >= required_windows

    def note_cancelled(self, request_id: str, now: float) -> None:
        self._retry_after[request_id] = now + self.cfg.cooldown_s
        self.note_started(now)

    def note_started(self, now: float) -> None:
        self._last_start = now
        self._since = None
        self._latched = False
        self._service_confirmation = None

    def decide(
        self,
        now: float,
        source: PoolReading,
        target: PoolReading,
        candidates: list[Candidate],
    ) -> PressureDecision:
        cfg = self.cfg
        if self._last_tick is not None and now < self._last_tick:
            raise ValueError("decision time regressed")
        self.observe_relief(now, source)
        source = self.ledger.routing_reading(source)
        target = self.ledger.routing_reading(target)
        s, t = classify(source, cfg, now), classify(target, cfg, now)
        allowed_tick_gap = cfg.max_age_s if cfg.health_service else cfg.max_tick_gap_s
        if self._last_tick is not None and now - self._last_tick > allowed_tick_gap:
            self._since = None
            self._service_confirmation = None
            self.relief.recovery_since = None
        self._last_tick = now
        if len({c.request_id for c in candidates}) != len(candidates):
            raise ValueError("duplicate candidate identity")
        emergency = (
            s.capacity is not None
            and s.capacity >= 1
            and (
                cfg.capacity_lookahead_s > 0
                or (
                    source.growth_blocks_s is not None
                    and math.isfinite(source.growth_blocks_s)
                    and source.growth_blocks_s > 0
                )
            )
        )
        if emergency:
            self._service_confirmation = None
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
                service_path=("workload" if cfg.workload_pressure else "health")
                if cfg.health_service and not emergency
                else "legacy",
            )

        if not self.ledger.channel_available(target.pool_id):
            self._since = None
            self._service_confirmation = None
            return decision("migration channel occupied")
        if not emergency:
            if cfg.routing_relief and (
                not self.relief.active or not self.relief.confirming
            ):
                self._since = None
                self._service_confirmation = None
                return decision("router relief not settled; " + self.relief.phase)
            if cfg.routing_relief and source.workload["load"] < cfg.leave_ratio:
                self._since = None
                self._service_confirmation = None
                return decision("source workload recovered after router relief")
            if cfg.capacity_only:
                self._since = None
                return decision("KV-only pilot; capacity warning absent")
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
            if cfg.health_service:
                needed = (
                    health_valid(source, cfg, now)
                    and source.health_status == "SUPPORTED"
                    and source.health_eta < cfg.health_min_efficiency
                    and health_spare(target, cfg, now)
                )
            if cfg.workload_pressure:
                quality_bad = (
                    source.decode_ratio is not None
                    and math.isfinite(source.decode_ratio)
                    and source.decode_ratio >= cfg.enter_ratio
                    and source.decode_samples >= cfg.min_decode_samples
                    and source.decode_sampled_at is not None
                    and 0 <= now - source.decode_sampled_at <= cfg.max_age_s
                )
                needed = (
                    workload.valid(source, cfg, now)
                    and health_spare(target, cfg, now)
                    and quality_bad
                )
            if not needed:
                self._since = None
                self._latched = False
                self._service_confirmation = None
                return decision("source relief or target spare evidence absent")
            self._latched = True
            if self._since is None:
                self._since = now
            if cfg.workload_pressure:
                busy = (
                    source.workload["load"] >= cfg.leave_ratio
                    or source.workload["prefill_s"] / cfg.sustain_s >= cfg.leave_ratio
                )
                # Over-efficient workload needs quality confirmation too. At low
                # workload require an extra independent window, not an old gap
                # surviving in overlapping snapshots. KV bypasses this entirely.
                if not self._confirm_new_service_intervals(
                    source, now, 1 if busy else 2
                ):
                    return decision("awaiting independent service degradation windows")
            if now - self._since < cfg.sustain_s:
                return decision("imbalance not sustained")
            if now - self._last_start < cfg.cooldown_s:
                return decision("migration cooldown")
        # Capacity bypasses persistence/cooldown, never physical feasibility.
        kv_priority = emergency and cfg.kv_emergency_priority
        if t.capacity is None or "pool timestamp" in t.missing:
            return decision("target physical capacity evidence absent")
        pending_blocks, pending_decode, _ = self.ledger.pending(target.pool_id)
        feasible, rows = [], []
        for c in candidates:
            reasons = []
            soft_warnings = []
            if now < self._retry_after.get(c.request_id, -math.inf):
                reasons.append("cancelled request cooling down")
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
            if (
                c.transfer_bytes is None
                or not math.isfinite(c.transfer_bytes)
                or c.transfer_bytes < 0
            ):
                reasons.append("transfer size unknown")
            needed_release = required
            growth_known = (
                source.growth_blocks_s is not None
                and math.isfinite(source.growth_blocks_s)
                and source.growth_blocks_s >= 0
            )
            extra_prompt = (
                source.unallocated_prompt_blocks if cfg.capacity_lookahead_s > 0 else 0
            )
            projection_known = (
                growth_known and type(extra_prompt) is int and extra_prompt >= 0
            )
            age = now - source.sampled_at if cfg.capacity_lookahead_s > 0 else 0
            if (
                emergency
                and projection_known
                and c.estimated_release_s is not None
                and math.isfinite(c.estimated_release_s)
                and c.estimated_release_s >= 0
            ):
                needed_release = max(
                    required,
                    math.ceil(
                        cfg.capacity_guard_blocks
                        - source.free_blocks
                        + extra_prompt
                        + source.growth_blocks_s * (c.estimated_release_s + age)
                    ),
                )
            if emergency and (c.releasable_blocks or 0) < needed_release:
                reasons.append("insufficient capacity relief")
            # guard is an early warning; do not reject merely for U>1 or
            # release later than guard. Physical exhaustion is separate.
            if emergency and (
                not projection_known
                or c.estimated_release_s is None
                or not math.isfinite(c.estimated_release_s)
                or c.estimated_release_s < 0
                or extra_prompt + (c.estimated_release_s + age) * source.growth_blocks_s
                >= source.free_blocks
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
            if not emergency and (
                c.estimated_release_s is None
                or not math.isfinite(c.estimated_release_s)
                or c.estimated_release_s < 0
            ):
                reasons.append("release time estimate unavailable")
            if emergency and not valid_prediction and cap <= cfg.survival_horizon:
                # A missing predictor is not evidence that a short stop budget
                # can survive transport. Keep emergency fallback explicit.
                (soft_warnings if kv_priority else reasons).append(
                    "stop budget cannot cover migration horizon"
                )
            if summary and summary["p_gt_bounds"][0] < cfg.min_survival_lower:
                (soft_warnings if kv_priority else reasons).append(
                    "insufficient sustained work probability"
                )
            costs = (c.source_decode_relief, c.target_decode_cost)
            valid_costs = all(
                v is not None and math.isfinite(v) and v >= 0 for v in costs
            )
            if not emergency:
                if cfg.health_service:
                    if (
                        (
                            not cfg.workload_pressure
                            and not c.target_health_after_supported
                        )
                        or not valid_costs
                        or t.decode is None
                        or t.prefill is None
                        or t.decode + pending_decode + c.target_decode_cost
                        > cfg.leave_ratio
                        or target.pending_prompt_tokens
                        or target.waiting
                    ):
                        reasons.append(
                            "post-move health support/service budget missing"
                        )
                elif not valid_costs or s.decode is None or t.decode is None:
                    reasons.append("post-move service estimate missing")
                elif (
                    t.decode + pending_decode + c.target_decode_cost > cfg.leave_ratio
                    or t.decode + pending_decode + c.target_decode_cost
                    > max(0, s.decode - c.source_decode_relief) + cfg.imbalance_enter
                    or target.pending_prompt_tokens / cfg.prefill_backlog_budget
                    > cfg.spare_ratio
                ):
                    reasons.append("post-move overload/reverse imbalance")
            elif kv_priority:
                # The ledger must still be able to track a finite target cost.
                if (
                    c.target_decode_cost is None
                    or not math.isfinite(c.target_decode_cost)
                    or c.target_decode_cost < 0
                ):
                    reasons.append("target commitment cost unknown/invalid")
                if (
                    t.prefill is None
                    or t.decode is None
                    or not valid_costs
                    or t.decode + pending_decode + c.target_decode_cost
                    > cfg.enter_ratio
                    or t.prefill > cfg.enter_ratio
                ):
                    soft_warnings.append("emergency target service budget unavailable")
            elif (
                t.prefill is None
                or t.decode is None
                or not valid_costs
                or t.decode + pending_decode + c.target_decode_cost > cfg.enter_ratio
                or t.prefill > cfg.enter_ratio
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
            if kv_priority:
                rows[-1]["kv_priority_override"] = True
                rows[-1]["soft_warnings"] = soft_warnings
            if not reasons:
                feasible.append((c, summary))
        if not feasible:
            return decision("no feasible candidate", rows=rows)
        predicted = [(c, summary) for c, summary in feasible if summary]
        if emergency:
            # Known adequate survival takes precedence over the explicit
            # emergency-only missing-prediction fallback. All candidates have
            # already passed candidate-specific release size/deadline checks.
            if kv_priority:
                durable = [
                    c
                    for c, summary in predicted
                    if summary["p_gt_bounds"][0] >= cfg.min_survival_lower
                ]
                tier = durable or [c for c, _ in feasible]
            else:
                tier = (
                    [c for c, _ in predicted] if predicted else [c for c, _ in feasible]
                )

            def rank(c):
                return (
                    c.estimated_release_s,
                    c.transfer_bytes,
                    -c.releasable_blocks,
                    c.request_id,
                )

            policy = (
                "KV priority: durable preference, release time, bytes, release, id"
                if kv_priority
                else "KV: survival evidence, release time, bytes, release, id"
            )
        else:
            # Service relief should persist. Predictions within the configured
            # resolution form one tier; KV size is not the service tie-breaker.
            longest = max(summary["mean"] for _, summary in predicted)
            tier = [
                c
                for c, summary in predicted
                if longest - summary["mean"] <= cfg.length_resolution
            ]

            def rank(c):
                return (
                    -c.source_decode_relief,
                    c.target_decode_cost,
                    c.estimated_release_s,
                    c.transfer_bytes,
                    c.request_id,
                )

            policy = (
                "service: longest tier, source relief, target cost, "
                "release time, bytes, id"
            )
        selected = min(tier, key=rank)
        eligible_ids = {c.request_id for c in tier}
        by_id = {c.request_id: c for c in candidates}
        for row in rows:
            c = by_id[row["request"]["request_id"]]
            row["selection_policy"] = policy
            row["selection_eligible"] = c.request_id in eligible_ids
            row["selection_key"] = (
                list(rank(c)) if c.request_id in eligible_ids else None
            )
            row["selected"] = c.request_id == selected.request_id
        return decision(
            "capacity relief" if emergency else "sustained pressure; " + policy,
            selected.request_id,
            rows,
        )


@dataclass(frozen=True)
class Admission:
    request_id: str
    pool_id: str
    max_output_tokens: int
    # Identity/visibility receipt only; no capacity is reserved for ingress.
    reservation_key: str
    routing_reason: str = ""
    reservation_basis: str = "none"
    routing_capacity: dict | None = None
    routing_pressure: dict | None = None


def prompt_fit(pool, prompt_tokens, ledger, cfg, now):
    """Route using physical blocks minus only unpublished/unallocated demand."""
    if (
        pool.block_size <= 0
        or pool.free_blocks is None
        or pool.free_blocks < 0
        or not all(
            math.isfinite(t) and 0 <= now - t <= cfg.max_age_s
            for t in (pool.sampled_at, pool.published_at)
        )
    ):
        return dict(status="UNKNOWN", required_blocks=None, available_blocks=None)
    required = math.ceil(prompt_tokens / pool.block_size)
    unpublished = sum(
        math.ceil(row.prefill_tokens / pool.block_size)
        for row in ledger.entries.values()
        if row.pool_id == pool.pool_id
        and row.kind == "admission"
        and row.state in {"ACTIVE", "COMMITTED"}
        and not row.prefill_observed
    )
    pending_blocks = ledger.pending(pool.pool_id)[0]
    native = pool.unallocated_prompt_blocks
    known = type(native) is int and native >= 0
    # Missing native demand still permits NO_FIT from an upper bound on free,
    # but never permits declaring FIT. Allocated native blocks are already out.
    available = max(
        0, pool.free_blocks - unpublished - pending_blocks - (native if known else 0)
    )
    status = "NO_FIT" if available < required else "FIT" if known else "UNKNOWN"
    return dict(
        status=status,
        required_blocks=required,
        available_blocks=available,
        native_unallocated_blocks=native,
        unpublished_prompt_blocks=unpublished,
        migration_pending_blocks=pending_blocks,
        prefix_cache_credit=0,
    )


class RoutingError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("no route with a feasible context/stop contract")


def admit(
    request_id: str,
    prompt_tokens: int,
    requested_output: int,
    pools: tuple[PoolReading, PoolReading],
    cfg: PressureConfig,
    ledger: ReservationLedger,
    now: float,
    key: str,
    allowed_pool_id: str | None = None,
    routing_strategy: str = "pressure",
    routing_weights: tuple[float, float] = (1.0, 1.0),
    routing_relief: dict | None = None,
) -> Admission:
    """Choose an instance; leave queueing and KV allocation to vLLM.

    Only the unchanged stop cap/context contract constrains eligible routes.
    Prompt fit chooses between instances but never gates ordinary ingress.
    Fixed pilot routes override the usual preference.
    """
    if (
        type(prompt_tokens) is not int
        or prompt_tokens < 0
        or type(requested_output) is not int
        or requested_output <= 0
    ):
        raise ValueError("invalid request lengths")
    output_cap = min(8192, requested_output)
    if routing_strategy not in {"pressure", "weighted-round-robin", "least-inflight"}:
        raise ValueError("unknown routing strategy")
    if len(routing_weights) != 2 or any(
        not math.isfinite(w) or w <= 0 for w in routing_weights
    ):
        raise ValueError("routing weights must be positive and finite")
    source = ledger.routing_reading(pools[0])
    target = ledger.routing_reading(pools[1])
    source_pressure = classify(source, cfg, now)
    prefer_target = prompt_tokens >= cfg.very_long_prompt_tokens or (
        prompt_tokens >= cfg.prompt_boundary
        and source_pressure.zone in {"CRITICAL", "OVERLOAD"}
    )
    if cfg.health_service:
        source_needs_relief = (
            source_pressure.capacity is not None and source_pressure.capacity >= 1
        ) or (
            health_valid(source, cfg, now)
            and source.health_eta < cfg.health_min_efficiency
        )
        prefer_target = health_spare(target, cfg, now) and (
            prompt_tokens >= cfg.very_long_prompt_tokens
            or (prompt_tokens >= cfg.prompt_boundary and source_needs_relief)
        )
        if cfg.pressure_routing:
            prefer_target = source_needs_relief and health_spare(target, cfg, now)
    preferred = pools[1] if prefer_target else pools[0]
    projected = None
    routing_reason = (
        "very long known prompt"
        if prefer_target and prompt_tokens >= cfg.very_long_prompt_tokens
        else "long prompt and TP1 pressure"
        if prefer_target
        else "prefer TP1"
    )
    if cfg.pressure_routing:
        routing_reason = "TP1 pressure and TP4 spare" if prefer_target else "prefer TP1"
    if cfg.workload_pressure and routing_strategy == "pressure":
        projected = {}
        for reading in (source, target):
            pending_blocks, pending_decode, _ = ledger.pending(reading.pool_id)
            service = workload.projected(
                reading, prompt_tokens, cfg, now, pending_decode
            )
            capacity = classify(reading, cfg, now).capacity
            if capacity is not None and reading.free_blocks is not None:
                capacity += (
                    pending_blocks + math.ceil(prompt_tokens / reading.block_size)
                ) / max(1, reading.free_blocks)
            projected[reading.pool_id] = dict(
                service=service,
                capacity=capacity,
                score=max(service, capacity or 0) if service is not None else None,
            )
        a, b = (projected[pool.pool_id]["score"] for pool in pools)
        ac, bc = (projected[pool.pool_id]["capacity"] for pool in pools)
        if (
            a is not None
            and b is not None
            and ac is not None
            and bc is not None
            and ac >= 1 > bc
        ):
            preferred = pools[1]
            routing_reason = "TP1 projected KV warning; TP4 physical headroom"
        elif a is not None and a <= cfg.leave_ratio:
            preferred = pools[0]
            routing_reason = "TP1 projected workload and KV within efficient budget"
        elif b is not None and (a is None or b < a):
            preferred = pools[1]
            routing_reason = "lower projected pressure on TP4"
        elif a is not None:
            preferred = pools[0]
            routing_reason = "lower projected pressure on TP1"
        else:
            counts = [
                sum(
                    row.pool_id == pool.pool_id
                    and row.kind == "admission"
                    and row.state in {"ACTIVE", "COMMITTED", "ACCOUNTED"}
                    for row in ledger.entries.values()
                )
                for pool in pools
            ]
            index = min(range(2), key=lambda i: (counts[i] + 1) / routing_weights[i])
            preferred = pools[index]
            routing_reason = "telemetry unavailable; normalized least-inflight fallback"
    if routing_strategy != "pressure":
        counts = [
            sum(
                row.pool_id == pool.pool_id
                and row.kind == "admission"
                and (
                    routing_strategy == "weighted-round-robin"
                    or row.state in {"ACTIVE", "COMMITTED", "ACCOUNTED"}
                )
                for row in ledger.entries.values()
            )
            for pool in pools
        ]
        index = min(range(2), key=lambda i: (counts[i] + 1) / routing_weights[i])
        preferred = pools[index]
        routing_reason = routing_strategy
    eligible, diagnostics = [], {}
    for pool in pools:
        if allowed_pool_id is not None and pool.pool_id != allowed_pool_id:
            continue
        reasons = []
        if pool.context_limit <= 0 or prompt_tokens + output_cap > pool.context_limit:
            reasons.append("context/stop budget exhausted")
        diagnostics[pool.pool_id] = dict(reasons=reasons)
        if not reasons:
            eligible.append(pool)
    if not eligible:
        raise RoutingError(diagnostics)
    capacity = {
        pool.pool_id: prompt_fit(pool, prompt_tokens, ledger, cfg, now)
        for pool in pools
    }
    other_fit = [pool for pool in eligible if capacity[pool.pool_id]["status"] == "FIT"]
    if (
        allowed_pool_id is None
        and capacity[preferred.pool_id]["status"] == "NO_FIT"
        and other_fit
    ):
        preferred = other_fit[0]
        routing_reason = "other pool cannot fit prompt; prompt capacity fallback"
    elif all(capacity[pool.pool_id]["status"] == "NO_FIT" for pool in eligible):
        routing_reason += "; neither pool fits prompt now; native queue"
    eligible.sort(key=lambda pool: pool.pool_id != preferred.pool_id)
    # Prefer relief only with fresh post-arrival target budget AND physical fit.
    # Never force a context-infeasible target, and never gate external ingress.
    if (
        cfg.routing_relief
        and routing_strategy == "pressure"
        and allowed_pool_id is None
        and routing_relief
        and routing_relief.get("active")
        and projected
        and pools[1] in eligible
        and health_spare(target, cfg, now)
        and capacity[pools[1].pool_id]["status"] == "FIT"
        and projected[pools[1].pool_id]["score"] is not None
        and projected[pools[1].pool_id]["score"] <= cfg.leave_ratio
        and projected[pools[1].pool_id]["capacity"] is not None
        and projected[pools[1].pool_id]["capacity"] < 1
    ):
        preferred = pools[1]
        eligible.sort(key=lambda pool: pool.pool_id != preferred.pool_id)
        routing_reason = "router relief epoch; TP4 projected budget and prompt fit"
    pool = eligible[0]
    if allowed_pool_id is not None:
        routing_reason = "fixed pilot route"
    elif pool.pool_id != preferred.pool_id:
        routing_reason += "; preferred pool context infeasible"
    ledger.track_ingress(key, request_id, pool.pool_id, prompt_tokens)
    return Admission(
        request_id,
        pool.pool_id,
        output_cap,
        key,
        routing_reason,
        routing_capacity=capacity,
        routing_pressure=projected,
    )
