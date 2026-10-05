#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Experiment-only M1 gate that waits for a fresh M5 predictor update."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class M1PredictorRefreshGate:
    """Compare immediate admission with one predictor-refresh wait.

    This gate only suppresses M1's safe START_SHADOW decision. It never
    overrides a physical M1 refusal or starts an unsafe Shadow.
    """

    action: str
    baseline_position: int | None = None
    released: bool = False

    def __post_init__(self) -> None:
        if self.action not in {"NOW", "WAIT"}:
            raise ValueError("experimental action must be NOW or WAIT")

    def decide(
        self, *, m1_action: str, m5_row: dict[str, Any] | None,
        source_time_to_guard_s: float | None,
        estimated_preparation_s: float | None,
        source_release_tail_s: float | None,
    ) -> tuple[bool, str | None]:
        """Return whether M1's decision may actuate and an audit reason."""
        available = bool(m5_row and m5_row.get("status") == "AVAILABLE")
        position = (m5_row.get("prediction_output_tokens")
                    if available else None)
        fresh_position = isinstance(position, int) and position >= 0
        guard_s = source_time_to_guard_s
        prepare_s = estimated_preparation_s
        tail_s = source_release_tail_s
        urgent = (
            m1_action == "START_SHADOW"
            and isinstance(guard_s, (int, float)) and math.isfinite(guard_s)
            and isinstance(prepare_s, (int, float))
            and math.isfinite(prepare_s)
            and isinstance(tail_s, (int, float)) and math.isfinite(tail_s)
            and guard_s <= prepare_s + tail_s
        )
        if urgent:
            self.released = True
            return True, "CAPACITY_SAFETY_RELEASE"
        if self.action == "NOW":
            if m1_action != "START_SHADOW":
                return False, None
            return ((True, "NOW_M5_AVAILABLE") if fresh_position
                    else (False, "NOW_M5_UNAVAILABLE"))

        if self.released:
            return m1_action == "START_SHADOW", None
        if self.baseline_position is None:
            if m1_action != "START_SHADOW":
                return False, None
            if not fresh_position:
                return False, "WAIT_M5_UNAVAILABLE"
            self.baseline_position = position
            return False, "WAIT_ARMED"

        if fresh_position and position > self.baseline_position:
            self.released = True
            return m1_action == "START_SHADOW", "WAIT_RECHECK_FRESH_M5"

        return False, "WAIT_FOR_FRESH_M5" if m1_action == "START_SHADOW" else None
