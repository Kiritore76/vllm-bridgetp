# SPDX-License-Identifier: Apache-2.0
"""Apply a versioned continuous SLO discount to completed request output."""

from __future__ import annotations

import math
from typing import Any

SOFT_POLICY = "WORST_SLO_RATIO_POWER_V1"
SOFT_WINDOW_POLICY = "WINDOW_TOKENS_FULL_REQUEST_SOFT_SLO_DRAIN_V1"


def quality_contract(config: dict[str, Any]) -> tuple[str, float]:
    """Validate the optional output objective independently of hard SLO flags."""
    policy = config.get("output_quality_policy", "HARD_SLO")
    if policy == "HARD_SLO":
        return policy, 1.0
    if policy != SOFT_POLICY:
        raise ValueError("unknown output quality policy")
    beta = config.get("output_quality_beta")
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(beta)
        or beta <= 0
    ):
        raise ValueError("output quality beta must be positive and finite")
    if config.get("max_visible_interval_policy") != "DIAGNOSTIC_ONLY":
        raise ValueError("soft output requires diagnostic max visible interval")
    rate = config.get("primary_max_slow_interval_rate")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate <= 0
    ):
        raise ValueError("soft output requires a positive slow interval limit")
    return policy, float(beta)


def _ratio(value: Any, limit: Any) -> float:
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in (value, limit)
    ):
        raise ValueError("missing or invalid output quality evidence")
    if value < 0 or limit <= 0:
        raise ValueError("invalid output quality value or limit")
    return value / limit


def request_quality(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Return severity and weight; incomplete service has zero output value.

    Missing evidence on completed service raises rather than inventing a zero.
    The worst normalized ratio avoids multiplying correlated latency penalties.
    """
    policy, beta = quality_contract(config)
    if policy != SOFT_POLICY:
        raise ValueError("request quality requires an explicit soft policy")
    if row["status"] != "COMPLETED":
        return {"quality_weight": 0.0, "quality_severity": None, "quality_ratios": {}}
    ratios = {
        "TTFT": _ratio(row.get("ttft_ms"), row.get("ttft_limit_ms")),
        "SLOW_INTERVAL_RATE": _ratio(
            row.get("slow_interval_rate"), config["primary_max_slow_interval_rate"]
        ),
    }
    mean = row.get("mean_tpot_ms")
    if mean is None:
        if row["output_tokens"] > 1:
            raise ValueError("missing mean TPOT for completed multi-token stream")
    else:
        ratios["MEAN_TPOT"] = _ratio(mean, config["primary_tpot_mean_ms"])
    if row["pool"] == "anchor":
        ratios["HANDOFF"] = _ratio(row.get("handoff_ms"), config["max_handoff_ms"])
    severity = max(1.0, *ratios.values())
    return {
        "quality_weight": severity**-beta,
        "quality_severity": severity,
        "quality_ratios": ratios,
    }
