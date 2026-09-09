# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure contracts for Bridge remote-attention/copy interference experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

MODES = ("ATTENTION_ONLY", "COPY_ONLY", "JOINT")


@dataclass(frozen=True)
class JointInterferenceCase:
    """One factorial cell shared by the three Bridge treatment modes."""

    context_tokens: int
    remote_fraction: float
    target_load_repeats: int
    copy_bytes_per_rank_step: int

    def validate(self) -> None:
        if self.context_tokens <= 0:
            raise ValueError("context tokens must be positive")
        if not 0 < self.remote_fraction <= 1:
            raise ValueError("remote fraction must be in (0, 1]")
        if self.target_load_repeats <= 0:
            raise ValueError("G3-J requires a non-zero TP4 target load")
        if self.copy_bytes_per_rank_step <= 0:
            raise ValueError("G3-J requires a non-zero Bridge copy payload")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_cases(
    *,
    context_tokens: Iterable[int],
    remote_fractions: Iterable[float],
    target_load_repeats: Iterable[int],
    copy_bytes_per_rank_step: Iterable[int],
) -> list[JointInterferenceCase]:
    cases = [
        JointInterferenceCase(context, fraction, load, copy_bytes)
        for context in context_tokens
        for fraction in remote_fractions
        for load in target_load_repeats
        for copy_bytes in copy_bytes_per_rank_step
    ]
    if not cases:
        raise ValueError("no G3-J interference cases requested")
    for case in cases:
        case.validate()
    return cases


def factorial_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one interaction row per cell from the three measured modes."""
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["cell_index"]), {})[str(row["mode"])] = row

    summaries = []
    for cell_index, by_mode in sorted(grouped.items()):
        if set(by_mode) != set(MODES):
            raise ValueError(f"cell {cell_index} does not contain all G3-J modes")
        attention = by_mode["ATTENTION_ONLY"]
        copy = by_mode["COPY_ONLY"]
        joint = by_mode["JOINT"]
        interaction = (
            float(joint["target_signed_slowdown_frac"])
            - float(attention["target_signed_slowdown_frac"])
            - float(copy["target_signed_slowdown_frac"])
        )
        summaries.append(
            {
                "cell_index": cell_index,
                "context_tokens": joint["context_tokens"],
                "remote_fraction": joint["remote_fraction"],
                "target_load_repeats": joint["target_load_repeats"],
                "copy_bytes_per_rank_step": joint["copy_bytes_per_rank_step"],
                "attention_target_signed_slowdown_frac": attention[
                    "target_signed_slowdown_frac"
                ],
                "copy_target_signed_slowdown_frac": copy[
                    "target_signed_slowdown_frac"
                ],
                "joint_target_signed_slowdown_frac": joint[
                    "target_signed_slowdown_frac"
                ],
                "joint_interaction_slowdown_frac": interaction,
                "attention_bridge_path_p50_ms": attention["bridge_path_p50_ms"],
                "copy_bridge_path_p50_ms": copy["bridge_path_p50_ms"],
                "joint_bridge_path_p50_ms": joint["bridge_path_p50_ms"],
            }
        )
    return summaries


def validate_results(
    rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    *,
    expected_cells: int,
    measured_steps: int,
) -> dict[str, Any]:
    errors = []
    expected_rows = expected_cells * len(MODES)
    expected_step_rows = expected_rows * measured_steps
    if len(rows) != expected_rows:
        errors.append(f"recorded {len(rows)} rows, expected {expected_rows}")
    if len(step_rows) != expected_step_rows:
        errors.append(
            f"recorded {len(step_rows)} step rows, expected {expected_step_rows}"
        )

    modes_by_cell: dict[int, set[str]] = {}
    steps_by_arm: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for step in step_rows:
        key = (int(step["cell_index"]), str(step["mode"]))
        steps_by_arm.setdefault(key, []).append(step)
    for row in rows:
        label = f"cell {row.get('cell_index')} {row.get('mode')}"
        key = (int(row["cell_index"]), str(row["mode"]))
        modes_by_cell.setdefault(key[0], set()).add(key[1])
        if row.get("status") != "PASS":
            errors.append(f"{label}: treatment failed")
        if not row.get("copy_verified"):
            errors.append(f"{label}: copy payload verification failed")
        if row.get("staged_remote_kv_bytes") != row.get(
            "expected_staged_remote_kv_bytes"
        ):
            errors.append(f"{label}: staged remote KV byte count mismatch")
        if row.get("mode") in ("ATTENTION_ONLY", "JOINT"):
            if not row.get("attention_finite"):
                errors.append(f"{label}: remote attention is not finite")
            if not row.get("attention_accurate"):
                errors.append(f"{label}: remote attention exceeded tolerance")
        numeric = (
            "target_signed_slowdown_frac",
            "target_harm_ms",
            "bridge_path_p50_ms",
        )
        if not all(math.isfinite(float(row[name])) for name in numeric):
            errors.append(f"{label}: non-finite timing output")
        arm_steps = steps_by_arm.get(key, [])
        if len(arm_steps) != measured_steps:
            errors.append(
                f"{label}: recorded {len(arm_steps)} steps, "
                f"expected {measured_steps}"
            )
            continue
        step_fields = (
            "target_control_before_ms",
            "target_control_after_ms",
            "target_control_ms",
            "target_treatment_ms",
            "target_delta_ms",
            "target_harm_ms",
            "target_slowdown_frac",
            "bridge_path_ms",
        )
        if not all(
            math.isfinite(float(step[name]))
            for step in arm_steps
            for name in step_fields
        ):
            errors.append(f"{label}: raw step telemetry is non-finite")
            continue
        control = sum(float(step["target_control_ms"]) for step in arm_steps)
        delta = sum(float(step["target_delta_ms"]) for step in arm_steps)
        harm = sum(float(step["target_harm_ms"]) for step in arm_steps)
        if control <= 0:
            errors.append(f"{label}: raw control duration is not positive")
        elif not math.isclose(
            delta / control,
            float(row["target_signed_slowdown_frac"]),
            abs_tol=1e-12,
        ):
            errors.append(f"{label}: raw slowdown does not reproduce summary")
        if not math.isclose(harm, float(row["target_harm_ms"]), abs_tol=1e-9):
            errors.append(f"{label}: raw harm does not reproduce summary")

    for cell_index, modes in modes_by_cell.items():
        if modes != set(MODES):
            errors.append(f"cell {cell_index}: missing factorial treatment")

    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "expected_cells": expected_cells,
        "expected_rows": expected_rows,
        "recorded_rows": len(rows),
        "expected_step_rows": expected_step_rows,
        "recorded_step_rows": len(step_rows),
        "errors": errors,
    }
