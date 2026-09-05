# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Performance model for BridgeTP Shadow KV-copy alternatives.

This module deliberately models the transfer schedule without changing the
production takeover path.  It compares a complete, newest-history-first
backfill against a new-KV-only bridge that leaves historical attention on the
source.  The latter is not a standalone takeover for a full-context model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Literal


Strategy = Literal["history_backfill", "new_kv_bridge"]


@dataclass(frozen=True)
class ShadowCopyInputs:
    """Inputs shared by both Shadow-copy strategies."""

    history_tokens: int
    remaining_tokens: int
    block_size: int
    kv_bytes_per_token: int
    copy_rate_bytes_s: float
    source_tpot_s: float
    target_tpot_s: float
    remote_attention_penalty_s: float
    remote_attention_bytes_per_token: int
    bridge_start_tokens: int = 1

    def validate(self) -> None:
        """Validate model inputs.

        Raises:
            ValueError: If a size, rate, or latency is outside its domain.
        """
        integer_fields = {
            "history_tokens": self.history_tokens,
            "remaining_tokens": self.remaining_tokens,
            "block_size": self.block_size,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "remote_attention_bytes_per_token": (
                self.remote_attention_bytes_per_token
            ),
            "bridge_start_tokens": self.bridge_start_tokens,
        }
        for name, value in integer_fields.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.history_tokens == 0:
            raise ValueError("history_tokens must be positive")
        if self.remaining_tokens == 0:
            raise ValueError("remaining_tokens must be positive")
        if self.block_size == 0 or self.kv_bytes_per_token == 0:
            raise ValueError("block_size and kv_bytes_per_token must be positive")
        if self.bridge_start_tokens == 0:
            raise ValueError("bridge_start_tokens must be positive")
        if self.bridge_start_tokens > self.remaining_tokens:
            raise ValueError("bridge_start_tokens exceeds remaining_tokens")
        if self.copy_rate_bytes_s <= 0:
            raise ValueError("copy_rate_bytes_s must be positive")
        if self.source_tpot_s <= 0 or self.target_tpot_s <= 0:
            raise ValueError("TPOT values must be positive")
        if self.remote_attention_penalty_s < 0:
            raise ValueError("remote_attention_penalty_s must be nonnegative")


@dataclass(frozen=True)
class ShadowCopyResult:
    """One strategy result for a fixed request and system condition."""

    strategy: Strategy
    completion_time_s: float
    source_release_time_s: float
    baseline_tp1_completion_s: float
    latency_gain_vs_tp1_s: float
    shadow_ready_time_s: float | None
    source_tokens_before_switch: int
    target_tokens_after_switch: int
    history_bytes_sent: int
    delta_bytes_sent: int
    remote_attention_bytes_sent: int
    total_network_bytes: int
    history_blocks_completed: int
    history_block_order: tuple[tuple[int, int], ...]
    standalone_takeover: bool
    source_needed_after_switch: bool
    outcome: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/CSV-friendly representation."""
        result = asdict(self)
        result["history_block_order"] = ";".join(
            f"[{start},{end})" for start, end in self.history_block_order
        )
        return result


@dataclass
class _TransferItem:
    kind: Literal["history", "delta"]
    start_token: int
    end_token: int
    remaining_bytes: float
    original_bytes: int


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


def remote_attention_bytes_per_token(
    *, num_layers: int, hidden_size: int, dtype_bytes: int
) -> int:
    """Estimate aggregate query plus attention-output bytes per token.

    This intentionally excludes protocol headers.  A real remote-attention
    implementation must replace the estimate with measured wire bytes.
    """
    values = (num_layers, hidden_size, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError("remote-attention geometry values must be positive")
    return 2 * num_layers * hidden_size * dtype_bytes


def newest_first_history_blocks(
    history_tokens: int, block_size: int
) -> tuple[tuple[int, int], ...]:
    """Return logical history ranges from the Shadow boundary toward token 0."""
    if history_tokens <= 0 or block_size <= 0:
        raise ValueError("history_tokens and block_size must be positive")
    ranges = []
    end = history_tokens
    while end > 0:
        start = max(0, end - block_size)
        ranges.append((start, end))
        end = start
    return tuple(ranges)


def _newest_first_items(inputs: ShadowCopyInputs) -> deque[_TransferItem]:
    return deque(
        _TransferItem(
            kind="history",
            start_token=start,
            end_token=end,
            remaining_bytes=(end - start) * inputs.kv_bytes_per_token,
            original_bytes=(end - start) * inputs.kv_bytes_per_token,
        )
        for start, end in newest_first_history_blocks(
            inputs.history_tokens, inputs.block_size
        )
    )


def simulate_history_backfill(inputs: ShadowCopyInputs) -> ShadowCopyResult:
    """Simulate newest-history-first backfill with delta priority.

    History is transferred in logical blocks starting at the Shadow boundary.
    A block already on the wire is not preempted, but every newly generated KV
    delta takes priority before the next history block.  Takeover is allowed
    only when all history and all produced deltas have been acknowledged.
    """
    inputs.validate()
    history = _newest_first_items(inputs)
    deltas: deque[_TransferItem] = deque()
    current: _TransferItem | None = None
    now = 0.0
    generated = 0
    next_token_time = inputs.source_tpot_s
    history_bytes_sent = 0.0
    delta_bytes_sent = 0.0
    completed_history: list[tuple[int, int]] = []
    epsilon = 1e-12

    while generated < inputs.remaining_tokens:
        if current is None:
            if deltas:
                current = deltas.popleft()
            elif history:
                current = history.popleft()
            else:
                break

        finish_time = now + current.remaining_bytes / inputs.copy_rate_bytes_s
        item_finished = finish_time <= next_token_time
        event_time = finish_time if item_finished else next_token_time
        transmitted = (
            current.remaining_bytes
            if item_finished
            else max(0.0, event_time - now) * inputs.copy_rate_bytes_s
        )
        current.remaining_bytes -= transmitted
        if current.kind == "history":
            history_bytes_sent += transmitted
        else:
            delta_bytes_sent += transmitted
        now = event_time

        token_arrived = next_token_time <= now + epsilon
        if item_finished:
            if current.kind == "history":
                completed_history.append(
                    (current.start_token, current.end_token)
                )
            current = None
        if token_arrived:
            start = inputs.history_tokens + generated
            generated += 1
            deltas.append(
                _TransferItem(
                    kind="delta",
                    start_token=start,
                    end_token=start + 1,
                    remaining_bytes=inputs.kv_bytes_per_token,
                    original_bytes=inputs.kv_bytes_per_token,
                )
            )
            next_token_time += inputs.source_tpot_s

        if current is None and not history and not deltas:
            target_tokens = inputs.remaining_tokens - generated
            completion = now + target_tokens * inputs.target_tpot_s
            baseline = inputs.remaining_tokens * inputs.source_tpot_s
            return ShadowCopyResult(
                strategy="history_backfill",
                completion_time_s=completion,
                source_release_time_s=now,
                baseline_tp1_completion_s=baseline,
                latency_gain_vs_tp1_s=baseline - completion,
                shadow_ready_time_s=now,
                source_tokens_before_switch=generated,
                target_tokens_after_switch=target_tokens,
                history_bytes_sent=round(history_bytes_sent),
                delta_bytes_sent=round(delta_bytes_sent),
                remote_attention_bytes_sent=0,
                total_network_bytes=round(history_bytes_sent + delta_bytes_sent),
                history_blocks_completed=len(completed_history),
                history_block_order=tuple(completed_history),
                standalone_takeover=True,
                source_needed_after_switch=False,
                outcome="TAKEOVER",
            )

    baseline = inputs.remaining_tokens * inputs.source_tpot_s
    return ShadowCopyResult(
        strategy="history_backfill",
        completion_time_s=baseline,
        source_release_time_s=baseline,
        baseline_tp1_completion_s=baseline,
        latency_gain_vs_tp1_s=0.0,
        shadow_ready_time_s=None,
        source_tokens_before_switch=inputs.remaining_tokens,
        target_tokens_after_switch=0,
        history_bytes_sent=round(history_bytes_sent),
        delta_bytes_sent=round(delta_bytes_sent),
        remote_attention_bytes_sent=0,
        total_network_bytes=round(history_bytes_sent + delta_bytes_sent),
        history_blocks_completed=len(completed_history),
        history_block_order=tuple(completed_history),
        standalone_takeover=False,
        source_needed_after_switch=False,
        outcome="SOURCE_FINISHED_BEFORE_TAKEOVER",
    )


def simulate_new_kv_bridge(inputs: ShadowCopyInputs) -> ShadowCopyResult:
    """Simulate new-KV-only startup followed by remote-attention Bridge.

    TP1 generates and transfers a small number of new token KV records before
    TP4 becomes the compute endpoint.  Historical KV remains on TP1, so TP1 is
    still required for remote attention until the request ends.
    """
    inputs.validate()
    now = 0.0
    generated = 0
    acknowledged = 0
    next_token_time = inputs.source_tpot_s
    pending_bytes = 0.0
    transmitted = 0.0
    epsilon = 1e-12

    def source_finished() -> ShadowCopyResult:
        baseline = inputs.remaining_tokens * inputs.source_tpot_s
        delta_bytes = round(transmitted)
        return ShadowCopyResult(
            strategy="new_kv_bridge",
            completion_time_s=baseline,
            source_release_time_s=baseline,
            baseline_tp1_completion_s=baseline,
            latency_gain_vs_tp1_s=0.0,
            shadow_ready_time_s=None,
            source_tokens_before_switch=inputs.remaining_tokens,
            target_tokens_after_switch=0,
            history_bytes_sent=0,
            delta_bytes_sent=delta_bytes,
            remote_attention_bytes_sent=0,
            total_network_bytes=delta_bytes,
            history_blocks_completed=0,
            history_block_order=(),
            standalone_takeover=False,
            source_needed_after_switch=False,
            outcome="SOURCE_FINISHED_BEFORE_BRIDGE",
        )

    while acknowledged < inputs.bridge_start_tokens:
        if pending_bytes <= epsilon:
            now = next_token_time
            generated += 1
            pending_bytes += inputs.kv_bytes_per_token
            next_token_time += inputs.source_tpot_s
            if generated == inputs.remaining_tokens:
                return source_finished()

        finish_time = now + pending_bytes / inputs.copy_rate_bytes_s
        if generated < inputs.remaining_tokens and next_token_time < finish_time:
            sent = (next_token_time - now) * inputs.copy_rate_bytes_s
            pending_bytes -= sent
            transmitted += sent
            now = next_token_time
            generated += 1
            pending_bytes += inputs.kv_bytes_per_token
            next_token_time += inputs.source_tpot_s
            if generated == inputs.remaining_tokens:
                return source_finished()
            continue

        transmitted += pending_bytes
        now = finish_time
        acknowledged = generated
        pending_bytes = 0.0

    bridge_tokens = inputs.remaining_tokens - generated
    bridge_tpot = inputs.target_tpot_s + inputs.remote_attention_penalty_s
    completion = now + bridge_tokens * bridge_tpot
    baseline = inputs.remaining_tokens * inputs.source_tpot_s
    remote_bytes = bridge_tokens * inputs.remote_attention_bytes_per_token
    delta_bytes = generated * inputs.kv_bytes_per_token
    return ShadowCopyResult(
        strategy="new_kv_bridge",
        completion_time_s=completion,
        source_release_time_s=completion,
        baseline_tp1_completion_s=baseline,
        latency_gain_vs_tp1_s=baseline - completion,
        shadow_ready_time_s=now,
        source_tokens_before_switch=generated,
        target_tokens_after_switch=bridge_tokens,
        history_bytes_sent=0,
        delta_bytes_sent=delta_bytes,
        remote_attention_bytes_sent=remote_bytes,
        total_network_bytes=delta_bytes + remote_bytes,
        history_blocks_completed=0,
        history_block_order=(),
        standalone_takeover=False,
        source_needed_after_switch=True,
        outcome="BRIDGE_TO_REQUEST_END",
    )


def compare_shadow_strategies(
    inputs: ShadowCopyInputs,
) -> dict[str, object]:
    """Run both strategies and report objective-specific winners."""
    history = simulate_history_backfill(inputs)
    bridge = simulate_new_kv_bridge(inputs)

    def winner(a: float, b: float) -> str:
        if abs(a - b) <= 1e-12:
            return "tie"
        return history.strategy if a < b else bridge.strategy

    bridge_tokens = bridge.target_tokens_after_switch
    break_even_ms: float | None = None
    if bridge_tokens > 0:
        base_bridge_time = (
            bridge.shadow_ready_time_s or 0.0
        ) + bridge_tokens * inputs.target_tpot_s
        break_even_ms = (
            (history.completion_time_s - base_bridge_time) / bridge_tokens * 1000
        )

    return {
        "inputs": asdict(inputs),
        "history_backfill": history.to_dict(),
        "new_kv_bridge": bridge.to_dict(),
        "latency_winner": winner(
            history.completion_time_s, bridge.completion_time_s
        ),
        "source_release_winner": winner(
            history.source_release_time_s, bridge.source_release_time_s
        ),
        "network_bytes_winner": winner(
            history.total_network_bytes, bridge.total_network_bytes
        ),
        "new_kv_bridge_remote_penalty_break_even_ms": break_even_ms,
    }
