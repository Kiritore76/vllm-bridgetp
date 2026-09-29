# SPDX-License-Identifier: Apache-2.0
"""Choose a bounded future commit boundary before admitting the TP4 request.

The dormant target is sized for one immutable output boundary.  M3 may delay
that boundary before target admission, but must never move it afterwards.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class M3CommitConfig:
    handoff_s: float
    gain_margin_s: float
    defer_tokens: int = 64
    safety_margin_s: float = 2.0
    min_target_owned_tokens: int = 64

    def validate(self) -> None:
        if not all(
            math.isfinite(value) and value >= 0
            for value in (self.handoff_s, self.gain_margin_s, self.safety_margin_s)
        ):
            raise ValueError("M3 time estimates must be finite and non-negative")
        if self.defer_tokens <= 0 or self.min_target_owned_tokens <= 0:
            raise ValueError("M3 token margins must be positive")


@dataclass(frozen=True)
class M3CommitDecision:
    action: str
    candidate_output_tokens: int
    reason: str
    expected_gain_s: float | None = None
    expected_remaining_tokens: float | None = None
    source_time_to_guard_s: float | None = None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class M3CommitController:
    def __init__(self, config: M3CommitConfig) -> None:
        config.validate()
        self.config = config

    def plan_candidate(
        self,
        *,
        output_tokens: int,
        base_candidate: int,
        max_output_tokens: int,
        expected_remaining_tokens: float | None,
        source_tpot_s: float | None,
        target_tpot_s: float | None,
        target_waiting: int | None,
        source_time_to_guard_s: float | None,
        capacity_emergency: bool,
    ) -> M3CommitDecision:
        """Defer at most once; an admitted target cannot be safely rebased."""
        cfg = self.config
        if not output_tokens < base_candidate < max_output_tokens:
            raise ValueError("M3 base candidate is outside the remaining output")

        def earliest(reason: str, gain: float | None = None) -> M3CommitDecision:
            return M3CommitDecision(
                "COMMIT_EARLIEST", base_candidate, reason, gain,
                expected_remaining_tokens, source_time_to_guard_s,
            )

        if capacity_emergency:
            return earliest("source capacity requires earliest safe boundary")
        if (
            expected_remaining_tokens is None
            or source_tpot_s is None
            or target_tpot_s is None
            or target_waiting is None
            or source_time_to_guard_s is None
        ):
            return earliest("benefit or source safety evidence unavailable")
        if not all(
            math.isfinite(value) and value >= 0
            for value in (
                expected_remaining_tokens, source_tpot_s, target_tpot_s,
                source_time_to_guard_s,
            )
        ) or source_tpot_s == 0 or target_tpot_s == 0 or target_waiting < 0:
            return earliest("benefit evidence invalid")

        remaining = min(
            expected_remaining_tokens, max_output_tokens - output_tokens
        )
        # The queue term is deliberately conservative and auditable. It is
        # only a first-order estimate until an A100 target queue model exists.
        queue_s = target_waiting * target_tpot_s * cfg.min_target_owned_tokens
        gain = (
            remaining * (source_tpot_s - target_tpot_s)
            - cfg.handoff_s - queue_s
        )
        if gain > cfg.gain_margin_s:
            return earliest("estimated migration gain exceeds margin", gain)
        delayed = base_candidate + cfg.defer_tokens
        if delayed > max_output_tokens - cfg.min_target_owned_tokens:
            return earliest("not enough target-owned output to defer", gain)
        latest_safe_wait_s = (
            (delayed - output_tokens) * source_tpot_s
            + cfg.handoff_s + cfg.safety_margin_s
        )
        if source_time_to_guard_s <= latest_safe_wait_s:
            return earliest("source guard cannot support deferral", gain)
        return M3CommitDecision(
            "DEFER", delayed, "low estimated gain while source is safe",
            gain, expected_remaining_tokens, source_time_to_guard_s,
        )
