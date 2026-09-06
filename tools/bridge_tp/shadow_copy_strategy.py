# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""C-series-calibrated model for Shadow-only KV transfer policies."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from typing import Literal


Strategy = Literal["history_backfill", "new_kv_only"]


@dataclass(frozen=True)
class InterferenceCell:
    """One paired C2 target-interference observation."""

    load_band: str
    repetition: int
    target_rate_gib_s: float
    effective_rate_gib_s: float
    target_load_frac: float
    baseline_mean_tpot_s: float
    copy_mean_tpot_s: float
    baseline_p99_tpot_s: float
    copy_p99_tpot_s: float
    baseline_p99_itl_s: float
    copy_p99_itl_s: float

    @property
    def mean_tpot_penalty_s(self) -> float:
        return self.copy_mean_tpot_s - self.baseline_mean_tpot_s

    @property
    def p99_tpot_penalty_s(self) -> float:
        return self.copy_p99_tpot_s - self.baseline_p99_tpot_s

    @property
    def p99_itl_penalty_s(self) -> float:
        return self.copy_p99_itl_s - self.baseline_p99_itl_s


@dataclass(frozen=True)
class ShadowTransferInputs:
    """Inputs shared by the two Shadow-only transfer policies."""

    history_tokens: int
    remaining_tokens: int
    block_size: int
    kv_bytes_per_token: int
    source_load_frac: float
    source_tpot_s: float
    interference: InterferenceCell

    def validate(self) -> None:
        if self.history_tokens <= 0 or self.remaining_tokens <= 0:
            raise ValueError("history_tokens and remaining_tokens must be positive")
        if self.block_size <= 0 or self.kv_bytes_per_token <= 0:
            raise ValueError("block_size and KV bytes must be positive")
        if self.source_tpot_s <= 0:
            raise ValueError("source_tpot_s must be positive")
        if self.interference.effective_rate_gib_s <= 0:
            raise ValueError("effective copy rate must be positive")
        if self.interference.copy_mean_tpot_s <= 0:
            raise ValueError("copy mean TPOT must be positive")


@dataclass(frozen=True)
class ShadowTransferResult:
    """Result for one policy over the remaining source-generation window."""

    strategy: Strategy
    observation_window_s: float
    copy_active_time_s: float
    copy_duty_cycle: float
    bytes_sent: int
    history_bytes_sent: int
    new_kv_bytes_sent: int
    history_backlog_end_bytes: int
    history_backlog_end_tokens: float
    maximum_new_kv_lag_s: float
    takeover_ready: bool
    takeover_ready_time_s: float | None
    source_finishes_before_ready: bool
    estimated_affected_target_tokens: float
    estimated_target_delay_s: float
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def kv_bytes_per_token(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    dtype_bytes: int,
) -> int:
    """Return aggregate K+V bytes for one token before TP sharding."""
    values = (num_layers, num_kv_heads, head_size, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError("KV geometry values must be positive")
    return 2 * num_layers * num_kv_heads * head_size * dtype_bytes


def interpolate_tpot(
    load_frac: float,
    load_knots: list[float],
    tpot_knots_s: list[float],
) -> float:
    """Linearly interpolate inside a frozen monotone C1 TPOT curve."""
    if len(load_knots) != len(tpot_knots_s) or not load_knots:
        raise ValueError("invalid TPOT knot arrays")
    if load_frac < load_knots[0] or load_frac > load_knots[-1]:
        raise ValueError("source load is outside the C1 support")
    right = bisect_right(load_knots, load_frac)
    if right == 0:
        return tpot_knots_s[0]
    if right == len(load_knots):
        return tpot_knots_s[-1]
    left = right - 1
    x0, x1 = load_knots[left], load_knots[right]
    y0, y1 = tpot_knots_s[left], tpot_knots_s[right]
    if x1 == x0:
        return max(y0, y1)
    weight = (load_frac - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


def _target_impact(
    active_time_s: float, cell: InterferenceCell
) -> tuple[float, float]:
    affected = active_time_s / cell.copy_mean_tpot_s
    return affected, affected * cell.mean_tpot_penalty_s


def simulate_history_backfill(
    inputs: ShadowTransferInputs,
) -> ShadowTransferResult:
    """Simulate continuous newest-first history backfill plus new KV."""
    inputs.validate()
    copy_rate = inputs.interference.effective_rate_gib_s * 1024**3
    new_kv_rate = inputs.kv_bytes_per_token / inputs.source_tpot_s
    history_bytes = inputs.history_tokens * inputs.kv_bytes_per_token
    window = inputs.remaining_tokens * inputs.source_tpot_s
    drain_rate = copy_rate - new_kv_rate

    catch_up = None
    if drain_rate > 0:
        catch_up = history_bytes / drain_rate
    ready = catch_up is not None and catch_up <= window
    active_time = catch_up if ready else window
    available_bytes = history_bytes + new_kv_rate * active_time
    bytes_sent = min(copy_rate * active_time, available_bytes)
    new_bytes = min(new_kv_rate * active_time, bytes_sent)
    history_sent = max(0.0, bytes_sent - new_bytes)
    backlog = max(0.0, history_bytes - history_sent)
    affected, target_delay = _target_impact(active_time, inputs.interference)

    if ready:
        outcome = "TAKEOVER_READY"
    elif drain_rate <= 0:
        outcome = "NEW_KV_RATE_EXCEEDS_COPY_RATE"
    else:
        outcome = "SOURCE_FINISHED_BEFORE_HISTORY_CAUGHT_UP"
    return ShadowTransferResult(
        strategy="history_backfill",
        observation_window_s=window,
        copy_active_time_s=active_time,
        copy_duty_cycle=active_time / window,
        bytes_sent=round(bytes_sent),
        history_bytes_sent=round(history_sent),
        new_kv_bytes_sent=round(new_bytes),
        history_backlog_end_bytes=round(backlog),
        history_backlog_end_tokens=backlog / inputs.kv_bytes_per_token,
        maximum_new_kv_lag_s=inputs.kv_bytes_per_token / copy_rate,
        takeover_ready=ready,
        takeover_ready_time_s=catch_up if ready else None,
        source_finishes_before_ready=not ready,
        estimated_affected_target_tokens=affected,
        estimated_target_delay_s=target_delay,
        outcome=outcome,
    )


def simulate_new_kv_only(
    inputs: ShadowTransferInputs,
) -> ShadowTransferResult:
    """Simulate per-decode-step new-KV mirroring without history transfer."""
    inputs.validate()
    copy_rate = inputs.interference.effective_rate_gib_s * 1024**3
    window = inputs.remaining_tokens * inputs.source_tpot_s
    new_bytes = inputs.remaining_tokens * inputs.kv_bytes_per_token
    active_time = new_bytes / copy_rate
    duty_cycle = min(1.0, active_time / window)
    transfer_lag = inputs.kv_bytes_per_token / copy_rate
    affected, target_delay = _target_impact(active_time, inputs.interference)
    history_bytes = inputs.history_tokens * inputs.kv_bytes_per_token
    stable = active_time <= window
    return ShadowTransferResult(
        strategy="new_kv_only",
        observation_window_s=window,
        copy_active_time_s=active_time,
        copy_duty_cycle=duty_cycle,
        bytes_sent=new_bytes,
        history_bytes_sent=0,
        new_kv_bytes_sent=new_bytes,
        history_backlog_end_bytes=history_bytes,
        history_backlog_end_tokens=float(inputs.history_tokens),
        maximum_new_kv_lag_s=transfer_lag,
        takeover_ready=False,
        takeover_ready_time_s=None,
        source_finishes_before_ready=True,
        estimated_affected_target_tokens=affected,
        estimated_target_delay_s=target_delay,
        outcome=(
            "NEW_KV_SYNCHRONIZED_HISTORY_UNMOVED"
            if stable
            else "NEW_KV_RATE_EXCEEDS_COPY_RATE"
        ),
    )


def compare_shadow_transfers(
    inputs: ShadowTransferInputs,
) -> dict[str, object]:
    """Compare Shadow transport overhead and readiness separately."""
    history = simulate_history_backfill(inputs)
    new_only = simulate_new_kv_only(inputs)
    return {
        "inputs": {
            "history_tokens": inputs.history_tokens,
            "remaining_tokens": inputs.remaining_tokens,
            "block_size": inputs.block_size,
            "kv_bytes_per_token": inputs.kv_bytes_per_token,
            "source_load_frac": inputs.source_load_frac,
            "source_tpot_s": inputs.source_tpot_s,
            **asdict(inputs.interference),
            "mean_tpot_penalty_s": inputs.interference.mean_tpot_penalty_s,
            "p99_tpot_penalty_s": inputs.interference.p99_tpot_penalty_s,
            "p99_itl_penalty_s": inputs.interference.p99_itl_penalty_s,
        },
        "history_backfill": history.to_dict(),
        "new_kv_only": new_only.to_dict(),
        "lower_bytes": (
            history.strategy
            if history.bytes_sent < new_only.bytes_sent
            else new_only.strategy
        ),
        "lower_estimated_interference": (
            history.strategy
            if history.estimated_target_delay_s
            < new_only.estimated_target_delay_s
            else new_only.strategy
        ),
        "takeover_ready": history.strategy if history.takeover_ready else "none",
    }
