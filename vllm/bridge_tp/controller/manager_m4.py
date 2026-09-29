# SPDX-License-Identifier: Apache-2.0
"""Conservative pre-freeze cancellation of a nearly finished Shadow request."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .events import SourceRequestView
from .predictor import SurvivalTable


@dataclass(frozen=True)
class M4CancelDecision:
    action: str
    reason: str
    output_tokens: int
    max_output_tokens: int
    remaining_budget_tokens: int
    expected_remaining_tokens: float | None = None
    probability_beyond_window: float | None = None
    survivors: int | None = None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class M4CancelConfig:
    short_window_tokens: int = 64
    maximum_probability_beyond_window: float = 0.1
    minimum_survivors: int = 100
    minimum_source_headroom_tokens: int = 64


class M4CancelController:
    def __init__(self, config: M4CancelConfig = M4CancelConfig()) -> None:
        if config.short_window_tokens <= 0:
            raise ValueError("M4 short window must be positive")
        if not 0 <= config.maximum_probability_beyond_window < 1:
            raise ValueError("M4 probability limit must be in [0, 1)")
        if config.minimum_survivors <= 0:
            raise ValueError("M4 survivor minimum must be positive")
        if config.minimum_source_headroom_tokens < 0:
            raise ValueError("M4 headroom minimum cannot be negative")
        self.config = config

    def decide(
        self,
        request: SourceRequestView,
        table: SurvivalTable,
        *,
        max_output_tokens: int,
        ignore_eos: bool,
        source_free_kv_tokens: int,
        source_guard_free_kv_tokens: int,
        source_capacity_pressure: bool,
        freeze_started: bool,
    ) -> M4CancelDecision:
        """Cancel only when finishing on TP1 is likely and capacity is safe."""
        budget = max(0, max_output_tokens - request.output_tokens)
        def stay(
            reason: str,
            *,
            expected_remaining_tokens: float | None = None,
            probability_beyond_window: float | None = None,
            survivors: int | None = None,
        ) -> M4CancelDecision:
            return M4CancelDecision(
                "KEEP_SHADOW", reason, request.output_tokens,
                max_output_tokens, budget, expected_remaining_tokens,
                probability_beyond_window, survivors,
            )
        if freeze_started:
            return stay("source freeze has begun")
        if budget == 0:
            return stay("source completion is handled by the request future")
        headroom = source_free_kv_tokens - source_guard_free_kv_tokens
        if source_capacity_pressure or headroom < max(
            budget, self.config.minimum_source_headroom_tokens
        ):
            return stay("TP1 capacity is not safe for continued ownership")
        if budget <= self.config.short_window_tokens:
            return M4CancelDecision(
                "CANCEL_SHADOW", "output cap is within the short window",
                request.output_tokens, max_output_tokens, budget,
            )
        if ignore_eos or not table.in_support(request.output_tokens):
            return stay("EOS prediction is unavailable")
        survivors = table.n_survivors(request.output_tokens)
        if survivors < self.config.minimum_survivors:
            return stay("too few remaining-length observations", survivors=survivors)
        expected = table.expected_remaining(request.output_tokens)
        probability = table.p_remaining_gt(
            request.output_tokens, self.config.short_window_tokens
        )
        if (
            expected <= self.config.short_window_tokens
            and probability <= self.config.maximum_probability_beyond_window
        ):
            return M4CancelDecision(
                "CANCEL_SHADOW", "likely EOS within the short window",
                request.output_tokens, max_output_tokens, budget,
                expected, probability, survivors,
            )
        return stay(
            "remaining output is not confidently short",
            expected_remaining_tokens=expected,
            probability_beyond_window=probability,
            survivors=survivors,
        )
