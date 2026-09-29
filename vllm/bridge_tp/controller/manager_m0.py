# SPDX-License-Identifier: Apache-2.0
"""Read-only runtime snapshots and decisions for the M0 migration manager.

M0 observes the existing controller.  It never calls the action adapter and
does not change request ownership.  Missing evidence stays missing so replay
cannot accidentally turn an absent metric into a safe zero.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeSnapshot:
    """One request, source pool, target pool, and migration progress at a tick.

    All times are Unix seconds, lengths are tokens, and rates are bytes/s.
    ``None`` means the source trace did not establish that value.
    """

    unix_s: float
    migration_id: str | None
    request_id: str | None
    state: str
    generated_tokens: int | None
    current_context_tokens: int | None = None
    request_age_s: float | None = None
    deadline_unix_s: float | None = None
    source_sampled_unix_s: float | None = None
    target_sampled_unix_s: float | None = None
    source_free_kv_tokens: int | None = None
    source_guard_free_kv_tokens: int | None = None
    source_pool_growth_tokens_s: float | None = None
    source_pool_sustained_growth_tokens_s: float | None = None
    source_running: int | None = None
    source_waiting: int | None = None
    target_free_kv_tokens: int | None = None
    target_kv_usage_frac: float | None = None
    target_running: int | None = None
    target_waiting: int | None = None
    target_p99_tpot_s: float | None = None
    target_tpot_samples: int | None = None
    history_total_bytes: int | None = None
    history_resident_bytes: int | None = None
    delta_lag_tokens: int | None = None
    all_ranks_history_resident: bool | None = None
    all_ranks_armed: bool | None = None
    current_rate_bytes_s: float | None = None
    expected_remaining_tokens: float | None = None
    capacity_pressure: bool | None = None
    channel_available: bool | None = None

    def to_json(self) -> dict[str, Any]:
        """Return an audit-ready snapshot with explicit unknown values."""
        return asdict(self)

    def freshness_errors(self, max_age_s: float) -> list[str]:
        """Identify missing, future, or stale pool samples."""
        errors = []
        for name, sampled in (
            ("source", self.source_sampled_unix_s),
            ("target", self.target_sampled_unix_s),
        ):
            if sampled is None or not math.isfinite(sampled):
                errors.append(f"{name} sample missing")
            elif not 0 <= self.unix_s - sampled <= max_age_s:
                errors.append(f"{name} sample stale or in the future")
        return errors


@dataclass(frozen=True)
class M0Proposal:
    """Inputs proposed by the current policy, never executed by M0."""

    start: bool | None = None
    rate_bytes_s: float | None = None
    commit: bool | None = None
    cancel: bool | None = None
    origin: str = "existing_controller"


@dataclass(frozen=True)
class M0Decision:
    """One reproducible, advisory decision and its evidence gap."""

    action: str
    reason: str
    missing: tuple[str, ...] = ()
    proposed_rate_bytes_s: float | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize the advisory action."""
        return asdict(self)


class MigrationManagerM0:
    """Validate policy proposals against observable M0 state."""

    def __init__(self, max_sample_age_s: float = 2.0) -> None:
        if max_sample_age_s <= 0:
            raise ValueError("max_sample_age_s must be positive")
        self.max_sample_age_s = max_sample_age_s

    def decide(
        self, snapshot: RuntimeSnapshot, proposal: M0Proposal
    ) -> M0Decision:
        """Choose an advisory action without mutating any runtime state."""
        stale = snapshot.freshness_errors(self.max_sample_age_s)
        if snapshot.state == "LOCAL":
            missing = list(stale)
            if snapshot.generated_tokens is None:
                missing.append("generated_tokens")
            if snapshot.target_free_kv_tokens is None:
                missing.append("target_free_kv_tokens")
            if snapshot.channel_available is None:
                missing.append("channel_available")
            if proposal.start is None:
                missing.append("start proposal")
            if missing:
                return M0Decision(
                    "WOULD_WAIT", "start evidence incomplete", tuple(missing)
                )
            if snapshot.channel_available is False:
                return M0Decision("WOULD_WAIT", "migration channel occupied")
            if snapshot.target_free_kv_tokens <= 0:
                return M0Decision("WOULD_WAIT", "target KV capacity unavailable")
            if proposal.start:
                return M0Decision("WOULD_START", "existing policy proposes start")
            return M0Decision("WOULD_WAIT", "existing policy keeps TP1")

        if snapshot.state in {"SHADOW", "READY_NOT_COMMITTED"}:
            if proposal.cancel:
                if stale:
                    return M0Decision(
                        "WOULD_WAIT", "cancel evidence is stale", tuple(stale)
                    )
                if snapshot.capacity_pressure is True:
                    return M0Decision(
                        "WOULD_WAIT", "source capacity pressure blocks cancel"
                    )
                return M0Decision("WOULD_CANCEL", "existing policy proposes cancel")
            if proposal.commit:
                missing = list(stale)
                if snapshot.all_ranks_history_resident is not True:
                    missing.append("all_ranks_history_resident")
                if snapshot.all_ranks_armed is not True:
                    missing.append("all_ranks_armed")
                if snapshot.delta_lag_tokens is None or snapshot.delta_lag_tokens > 16:
                    missing.append("delta_lag_tokens<=16")
                if missing:
                    return M0Decision(
                        "WOULD_WAIT", "commit evidence incomplete", tuple(missing)
                    )
                return M0Decision("WOULD_COMMIT", "commit proposal is ready")
            if proposal.rate_bytes_s is not None:
                if stale:
                    return M0Decision(
                        "WOULD_WAIT", "rate evidence is stale", tuple(stale)
                    )
                if snapshot.target_p99_tpot_s is None:
                    return M0Decision(
                        "WOULD_WAIT",
                        "target TPOT evidence unavailable",
                        ("target_p99_tpot_s",),
                    )
                if not math.isfinite(proposal.rate_bytes_s):
                    return M0Decision("WOULD_WAIT", "rate proposal is not finite")
                if proposal.rate_bytes_s < 0:
                    return M0Decision("WOULD_WAIT", "rate proposal is negative")
                if snapshot.current_rate_bytes_s != proposal.rate_bytes_s:
                    return M0Decision(
                        "WOULD_SET_RATE",
                        "existing controller proposes a rate change",
                        proposed_rate_bytes_s=proposal.rate_bytes_s,
                    )
            missing = () if proposal.rate_bytes_s is not None else ("rate proposal",)
            return M0Decision("WOULD_WAIT", "no M0 action proposed", missing)

        return M0Decision("WOULD_WAIT", f"state {snapshot.state} has no M0 action")


class ChannelRegistry:
    """Track one active migration per target without creating communicators."""

    def __init__(self) -> None:
        self._active_by_target: dict[str, str] = {}

    def available(self, target_id: str, migration_id: str | None = None) -> bool:
        """Return whether this target can admit a new active session."""
        active = self._active_by_target.get(target_id)
        return active is None or active == migration_id

    def observe_active(self, target_id: str, migration_id: str) -> None:
        """Record a session already observed by the runtime."""
        if not self.available(target_id, migration_id):
            raise ValueError(f"target {target_id} already has an active migration")
        self._active_by_target[target_id] = migration_id

    def observe_idle(self, target_id: str, migration_id: str) -> None:
        """Release only the matching session from the read-only registry."""
        if self._active_by_target.get(target_id) == migration_id:
            del self._active_by_target[target_id]


class M0ExecutorAdapter:
    """Publish advisory actions to an audit sink without runtime actuation."""

    def __init__(self, audit_sink: Callable[[dict[str, Any]], None]) -> None:
        self._audit_sink = audit_sink

    def publish(
        self,
        tick: int,
        snapshot: RuntimeSnapshot,
        proposal: M0Proposal,
        decision: M0Decision,
    ) -> None:
        """Write one M0 record; deliberately has no ActionAdapter reference."""
        self._audit_sink(
            {
                "kind": "manager_m0_shadow",
                "tick": tick,
                "snapshot": snapshot.to_json(),
                "proposal": asdict(proposal),
                "decision": decision.to_json(),
            }
        )


def snapshot_from_telemetry(
    row: dict[str, Any], *, migration_id: str | None = None,
    request_id: str | None = None, channel_available: bool | None = None,
    expected_remaining_tokens: float | None = None,
    current_context_tokens: int | None = None,
    request_age_s: float | None = None,
) -> RuntimeSnapshot:
    """Normalize an existing Phase 9 telemetry record for live use or replay."""
    source = row.get("tp1") or {}
    target = row.get("tp4") or {}
    capacity = row.get("capacity_signal") or {}

    def free_tokens(pool: dict[str, Any]) -> int | None:
        blocks = pool.get("free_kv_blocks")
        size = pool.get("block_size")
        return int(blocks) * int(size) if blocks is not None and size else None

    p99 = target.get("p99_tpot_s")
    tpot_samples = target.get("tpot_samples")
    return RuntimeSnapshot(
        unix_s=float(row["unix_s"]),
        migration_id=migration_id,
        request_id=request_id,
        state=str(row["state"]),
        generated_tokens=row.get("output_tokens"),
        current_context_tokens=current_context_tokens,
        request_age_s=request_age_s,
        source_sampled_unix_s=source.get("sampled_unix_s"),
        target_sampled_unix_s=target.get("sampled_unix_s"),
        source_free_kv_tokens=free_tokens(source),
        source_guard_free_kv_tokens=(
            capacity.get("guard_free_kv_tokens")
            if capacity.get("guard_free_kv_tokens", 0) > 0
            else None
        ),
        source_pool_growth_tokens_s=(
            capacity.get("decline_rate_tokens_s")
            if capacity.get("transition") not in {"WARMUP", "DISABLED", None}
            else None
        ),
        source_pool_sustained_growth_tokens_s=(
            capacity.get("sustained_decline_rate_tokens_s")
            if capacity.get("transition") not in {"WARMUP", "DISABLED", None}
            else None
        ),
        source_running=source.get("num_running"),
        source_waiting=source.get("num_waiting"),
        target_free_kv_tokens=free_tokens(target),
        target_kv_usage_frac=target.get("kv_usage_frac"),
        target_running=target.get("num_running"),
        target_waiting=target.get("num_waiting"),
        target_p99_tpot_s=(
            float(p99)
            if p99 and p99 > 0 and (tpot_samples is None or tpot_samples > 0)
            else None
        ),
        target_tpot_samples=tpot_samples,
        current_rate_bytes_s=row.get("rate_bytes_s"),
        expected_remaining_tokens=expected_remaining_tokens,
        capacity_pressure=capacity.get("active"),
        channel_available=channel_available,
    )


class RuntimeStateCollector:
    """Normalize the current Phase 9 tick into the manager input contract."""

    def collect(
        self,
        telemetry: dict[str, Any],
        *,
        migration_id: str | None = None,
        request_id: str | None = None,
        channel_available: bool | None = None,
        expected_remaining_tokens: float | None = None,
        current_context_tokens: int | None = None,
        request_age_s: float | None = None,
        progress: dict[str, Any] | None = None,
    ) -> RuntimeSnapshot:
        """Collect only fields present in this tick, retaining unknowns."""
        snapshot = snapshot_from_telemetry(
            telemetry,
            migration_id=migration_id,
            request_id=request_id,
            channel_available=channel_available,
            expected_remaining_tokens=expected_remaining_tokens,
            current_context_tokens=current_context_tokens,
            request_age_s=request_age_s,
        )
        if not progress:
            return snapshot
        allowed = {
            "history_total_bytes",
            "history_resident_bytes",
            "delta_lag_tokens",
            "all_ranks_history_resident",
            "all_ranks_armed",
        }
        return replace(
            snapshot,
            **{key: value for key, value in progress.items() if key in allowed},
        )
