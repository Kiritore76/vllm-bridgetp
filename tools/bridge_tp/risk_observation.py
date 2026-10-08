#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build passive, predecision capacity observations for later risk audits."""

from __future__ import annotations

import math
from typing import Any


def build_risk_observation(
    *, tick: int, snapshot: dict[str, Any],
    m5_row: dict[str, Any] | None,
    natural_m1: dict[str, Any], applied_m1: dict[str, Any],
    initial_rate: dict[str, Any] | None,
    assigned_action: str | None,
) -> dict[str, Any]:
    """Keep model probabilities separate from observed pool capacity risk."""
    free = snapshot.get("source_free_kv_tokens")
    guard = snapshot.get("source_guard_free_kv_tokens")
    pending = snapshot.get("source_prefill_pending_kv_tokens")
    growth = snapshot.get("source_decode_growth_tokens_s")
    prefill_growth = snapshot.get("source_prefill_growth_tokens_s")
    if isinstance(growth, (int, float)) and prefill_growth is not None:
        growth = (
            growth + prefill_growth
            if isinstance(prefill_growth, (int, float))
            and math.isfinite(prefill_growth) and prefill_growth >= 0
            else None
        )
    headroom = (
        free - guard
        if all(isinstance(value, int) for value in (free, guard))
        else None
    )
    point_time_to_guard = None
    if (headroom is not None and isinstance(growth, (int, float))
            and math.isfinite(growth) and growth > 0):
        point_time_to_guard = max(0.0, headroom) / growth
    m5 = m5_row or {}
    return {
        "kind": "manager_risk_observation_shadow",
        "format_version": 2,
        "capacity_model": (
            "allocated_kv_plus_scheduled_growth"
            if prefill_growth is not None
            else "allocated_kv_plus_decode_growth"
        ),
        "prefill_capacity_policy": "OBSERVATION_ONLY",
        "tick": tick,
        "unix_s": snapshot.get("unix_s"),
        "request_id": snapshot.get("request_id"),
        "migration_id": snapshot.get("migration_id"),
        "generated_tokens": snapshot.get("generated_tokens"),
        "current_context_tokens": snapshot.get("current_context_tokens"),
        "source_free_kv_tokens": free,
        "source_guard_free_kv_tokens": guard,
        "source_prefill_pending_kv_tokens": pending,
        "source_safe_headroom_tokens": headroom,
        "source_decode_growth_tokens_s": snapshot.get(
            "source_decode_growth_tokens_s"),
        "source_prefill_growth_tokens_s": prefill_growth,
        "source_estimated_kv_growth_tokens_s": growth,
        "source_pool_sustained_growth_tokens_s": snapshot.get(
            "source_pool_sustained_growth_tokens_s"),
        "point_time_to_guard_s": point_time_to_guard,
        "source_running": snapshot.get("source_running"),
        "source_waiting": snapshot.get("source_waiting"),
        "target_free_kv_tokens": snapshot.get("target_free_kv_tokens"),
        "target_kv_usage_frac": snapshot.get("target_kv_usage_frac"),
        "target_running": snapshot.get("target_running"),
        "target_waiting": snapshot.get("target_waiting"),
        "target_p99_tpot_s": snapshot.get("target_p99_tpot_s"),
        "channel_available": snapshot.get("channel_available"),
        "m5_status": m5.get("status", "DISABLED"),
        "m5_prediction_output_tokens": m5.get("prediction_output_tokens"),
        "m5_age_s": m5.get("age_s"),
        "m5_headroom_probability_bounds": m5.get(
            "p_remaining_gt_headroom_runtime_bounds"),
        "m5_long_window_tokens": m5.get("long_window_tokens"),
        "m5_long_probability_bounds": m5.get(
            "p_remaining_gt_long_window_runtime_bounds"),
        "m5_probability_is_pool_oom_risk": False,
        "pool_guard_probability_status": "UNCALIBRATED",
        "initial_rate_profile": (initial_rate or {}).get("profile"),
        "initial_rate_bytes_s": (initial_rate or {}).get("rate_bytes_s"),
        "estimated_preparation_s": natural_m1.get(
            "estimated_preparation_s"),
        "source_release_tail_s": natural_m1.get("source_release_tail_s"),
        "natural_m1_action": natural_m1.get("action"),
        "natural_m1_reason": natural_m1.get("reason"),
        "applied_m1_action": applied_m1.get("action"),
        "applied_m1_reason": applied_m1.get("reason"),
        "assigned_experiment_action": assigned_action,
    }
