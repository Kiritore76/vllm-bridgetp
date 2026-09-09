# SPDX-License-Identifier: Apache-2.0
"""Pure analysis helpers for paired online Shadow strategy experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


WINDOWS = ("PRE_SHADOW", "SHADOW", "BRIDGE", "POST_COMMIT")


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def classify_interval(
    unix_s: float,
    *,
    shadow_start_unix_s: float,
    bridge_start_unix_s: float,
    committed_unix_s: float,
) -> str:
    if not shadow_start_unix_s < bridge_start_unix_s <= committed_unix_s:
        raise ValueError("online migration timestamps are not ordered")
    if unix_s < shadow_start_unix_s:
        return "PRE_SHADOW"
    if unix_s < bridge_start_unix_s:
        return "SHADOW"
    if unix_s < committed_unix_s:
        return "BRIDGE"
    return "POST_COMMIT"


def summarize_background_windows(
    results: Iterable[dict[str, Any]],
    *,
    shadow_start_unix_s: float,
    bridge_start_unix_s: float,
    committed_unix_s: float,
) -> dict[str, dict[str, float | int | None]]:
    intervals: dict[str, list[float]] = {window: [] for window in WINDOWS}
    jobs: dict[str, set[str]] = {window: set() for window in WINDOWS}
    for result in results:
        if result.get("pool") != "target" or result.get("status") != "COMPLETED":
            continue
        job_id = str(result.get("job_id"))
        times = [float(value) for value in result.get("token_times_unix_s", [])]
        for previous, current in zip(times, times[1:]):
            previous_window = classify_interval(
                previous,
                shadow_start_unix_s=shadow_start_unix_s,
                bridge_start_unix_s=bridge_start_unix_s,
                committed_unix_s=committed_unix_s,
            )
            window = classify_interval(
                current,
                shadow_start_unix_s=shadow_start_unix_s,
                bridge_start_unix_s=bridge_start_unix_s,
                committed_unix_s=committed_unix_s,
            )
            if previous_window != window:
                continue
            intervals[window].append((current - previous) * 1000)
            jobs[window].add(job_id)
    return {
        window: {
            "samples": len(intervals[window]),
            "jobs": len(jobs[window]),
            "tpot_p50_ms": percentile(intervals[window], 0.50),
            "tpot_p95_ms": percentile(intervals[window], 0.95),
            "tpot_p99_ms": percentile(intervals[window], 0.99),
        }
        for window in WINDOWS
    }


def validate_strategy_timing(
    strategy: str,
    *,
    shadow_start_unix_s: float,
    bridge_start_unix_s: float,
    history_start_unix_s: float,
    tolerance_s: float = 0.050,
) -> list[str]:
    errors: list[str] = []
    if strategy == "S_NEW":
        if history_start_unix_s + tolerance_s < bridge_start_unix_s:
            errors.append("S_NEW started history transfer during Shadow")
    elif strategy == "S_NEW_OLD":
        if history_start_unix_s > shadow_start_unix_s + tolerance_s:
            errors.append("S_NEW_OLD did not start history transfer at Shadow")
    else:
        errors.append(f"unknown Shadow strategy: {strategy!r}")
    return errors
