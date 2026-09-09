# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure scheduling contract for Shadow KV transfer experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ShadowStrategy = Literal["S_NEW", "S_NEW_OLD"]
EpisodeOutcome = Literal["CANCEL", "COMMIT"]
TransferKind = Literal["NEW", "HISTORY"]
TransferPhase = Literal["SHADOW", "BRIDGE"]


@dataclass(frozen=True)
class TransferUnit:
    """One ordered KV payload in a logical decode/migration step."""

    phase: TransferPhase
    step: int
    kind: TransferKind
    token_start: int
    token_end: int

    @property
    def tokens(self) -> int:
        return self.token_end - self.token_start

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowTransferPlan:
    strategy: ShadowStrategy
    outcome: EpisodeOutcome
    history_tokens: int
    shadow_steps: int
    block_size: int
    history_blocks_per_shadow_step: int
    shadow_units: tuple[TransferUnit, ...]
    bridge_units: tuple[TransferUnit, ...]
    history_tokens_copied_in_shadow: int
    bridge_entry_history_backlog_tokens: int

    @property
    def bridge_steps(self) -> int:
        return len({unit.step for unit in self.bridge_units})

    def validate(self) -> None:
        if self.strategy not in ("S_NEW", "S_NEW_OLD"):
            raise ValueError(f"unknown Shadow strategy {self.strategy!r}")
        if self.outcome not in ("CANCEL", "COMMIT"):
            raise ValueError(f"unknown episode outcome {self.outcome!r}")
        if self.history_tokens <= 0 or self.shadow_steps <= 0:
            raise ValueError("history tokens and Shadow steps must be positive")
        if self.block_size <= 0 or self.history_blocks_per_shadow_step <= 0:
            raise ValueError("block sizes and per-step history budget must be positive")
        if self.history_tokens % self.block_size:
            raise ValueError("history boundary must align to a complete KV block")

        shadow_new = [unit for unit in self.shadow_units if unit.kind == "NEW"]
        shadow_history = [unit for unit in self.shadow_units if unit.kind == "HISTORY"]
        if len(shadow_new) != self.shadow_steps:
            raise ValueError("every Shadow step must transfer exactly one new KV token")
        if self.strategy == "S_NEW" and shadow_history:
            raise ValueError("S_NEW cannot transfer history during Shadow")
        if self.history_tokens_copied_in_shadow != sum(
            unit.tokens for unit in shadow_history
        ):
            raise ValueError("Shadow history accounting does not match the schedule")
        if (
            self.bridge_entry_history_backlog_tokens
            != self.history_tokens - self.history_tokens_copied_in_shadow
        ):
            raise ValueError("Bridge entry backlog does not match Shadow progress")

        for step in range(self.shadow_steps):
            units = [unit for unit in self.shadow_units if unit.step == step]
            if not units or units[0].kind != "NEW":
                raise ValueError("new KV must have priority in every Shadow step")
        _validate_reverse_history_order(shadow_history, self.history_tokens)

        if self.outcome == "CANCEL" and self.bridge_units:
            raise ValueError("cancelled Shadow cannot contain Bridge transfers")
        if self.outcome == "COMMIT":
            bridge_history = [
                unit for unit in self.bridge_units if unit.kind == "HISTORY"
            ]
            if sum(unit.tokens for unit in bridge_history) != (
                self.bridge_entry_history_backlog_tokens
            ):
                raise ValueError("Bridge does not drain the complete history backlog")
            for step in range(self.bridge_steps):
                units = [unit for unit in self.bridge_units if unit.step == step]
                if len(units) != 2 or [unit.kind for unit in units] != [
                    "NEW",
                    "HISTORY",
                ]:
                    raise ValueError("Bridge steps must place new KV before history")
            _validate_reverse_history_order(
                bridge_history,
                self.bridge_entry_history_backlog_tokens,
            )


def reverse_history_blocks(
    history_tokens: int,
    block_size: int,
) -> list[tuple[int, int]]:
    """Return complete historical blocks from the Shadow boundary toward zero."""
    if history_tokens <= 0 or block_size <= 0:
        raise ValueError("history tokens and block size must be positive")
    if history_tokens % block_size:
        raise ValueError("history boundary must align to a complete KV block")
    return [(end - block_size, end) for end in range(history_tokens, 0, -block_size)]


def _validate_reverse_history_order(
    units: list[TransferUnit],
    expected_end: int,
) -> None:
    cursor = expected_end
    for unit in units:
        if unit.token_end != cursor:
            raise ValueError("history is not copied continuously from the boundary")
        if unit.tokens <= 0:
            raise ValueError("history transfer has an empty token range")
        cursor = unit.token_start


def build_shadow_transfer_plan(
    *,
    strategy: ShadowStrategy,
    outcome: EpisodeOutcome,
    history_tokens: int,
    shadow_steps: int,
    block_size: int = 16,
    history_blocks_per_shadow_step: int = 1,
) -> ShadowTransferPlan:
    """Build the causal new-first schedule shared by both compared strategies."""
    if strategy not in ("S_NEW", "S_NEW_OLD"):
        raise ValueError(f"unknown Shadow strategy {strategy!r}")
    if outcome not in ("CANCEL", "COMMIT"):
        raise ValueError(f"unknown episode outcome {outcome!r}")
    if shadow_steps <= 0 or history_blocks_per_shadow_step <= 0:
        raise ValueError("Shadow steps and history budget must be positive")

    remaining = reverse_history_blocks(history_tokens, block_size)
    shadow_units: list[TransferUnit] = []
    for step in range(shadow_steps):
        token = history_tokens + step
        shadow_units.append(TransferUnit("SHADOW", step, "NEW", token, token + 1))
        if strategy == "S_NEW_OLD":
            for _ in range(history_blocks_per_shadow_step):
                if not remaining:
                    break
                start, end = remaining.pop(0)
                shadow_units.append(TransferUnit("SHADOW", step, "HISTORY", start, end))

    copied = history_tokens - sum(end - start for start, end in remaining)
    bridge_units: list[TransferUnit] = []
    if outcome == "COMMIT":
        for step, (start, end) in enumerate(remaining):
            token = history_tokens + shadow_steps + step
            bridge_units.extend(
                [
                    TransferUnit("BRIDGE", step, "NEW", token, token + 1),
                    TransferUnit("BRIDGE", step, "HISTORY", start, end),
                ]
            )

    plan = ShadowTransferPlan(
        strategy=strategy,
        outcome=outcome,
        history_tokens=history_tokens,
        shadow_steps=shadow_steps,
        block_size=block_size,
        history_blocks_per_shadow_step=history_blocks_per_shadow_step,
        shadow_units=tuple(shadow_units),
        bridge_units=tuple(bridge_units),
        history_tokens_copied_in_shadow=copied,
        bridge_entry_history_backlog_tokens=sum(
            end - start for start, end in remaining
        ),
    )
    plan.validate()
    return plan


def group_units_by_step(
    units: tuple[TransferUnit, ...],
) -> list[list[TransferUnit]]:
    """Preserve phase-local step order without using future information."""
    grouped: list[list[TransferUnit]] = []
    for unit in units:
        if not grouped or grouped[-1][0].step != unit.step:
            grouped.append([unit])
        else:
            grouped[-1].append(unit)
    return grouped
