# SPDX-License-Identifier: Apache-2.0
"""Freeze whole-pool SLO v6 GoodOutput at H; drain is diagnostic only."""

from __future__ import annotations

import math


def score_horizon(
    slo: dict,
    background: dict,
    source: dict,
    target: dict,
    proxy: dict,
    horizon_s: float,
) -> dict:
    """Count completed, SLO-valid requests by H and zero other service results.

    Missing/corrupt evidence is a technical exclusion, never a service zero.
    All timestamps come from client streams; origin is the earliest actual
    request arrival (same convention as the existing fixed-window pilot).
    """
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("H must be positive and finite")
    excluded = {
        "eligible": False,
        "goodoutput_tokens_s": None,
        "failure_kind": "TECHNICAL_FAILURE",
    }
    if not slo.get("computable") or slo.get("errors"):
        return {**excluded, "errors": slo.get("errors") or ["missing SLO audit"]}
    records = {r["job_id"]: r for r in background.get("results", [])}
    anchor_id = str(proxy.get("external_request_id", "anchor"))
    records[anchor_id] = {
        "request_started_unix_s": source.get("request_started_unix_s"),
        "request_ended_unix_s": (target or source).get("completed_unix_s"),
        "token_times_unix_s": [x.get("unix_s") for x in proxy.get("emitted", [])],
        "finish_reason": (target or source).get("finish_reason"),
    }
    rows = slo.get("request_rows", [])
    if not rows or set(records) != {r["request_id"] for r in rows}:
        return {**excluded, "errors": ["request universe mismatch"]}
    starts = [r.get("request_started_unix_s") for r in records.values()]
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in starts):
        return {**excluded, "errors": ["missing arrival timestamp"]}
    origin = min(starts)
    cutoff = origin + horizon_s
    scored = []
    for row in rows:
        record = records[row["request_id"]]
        end = record.get("request_ended_unix_s")
        times = record.get("token_times_unix_s")
        if (
            not isinstance(end, (int, float))
            or not math.isfinite(end)
            or not isinstance(times, list)
            or any(
                not isinstance(t, (int, float)) or not math.isfinite(t) for t in times
            )
            or any(b < a for a, b in zip(times, times[1:]))
        ):
            return {**excluded, "errors": ["missing/corrupt stream boundary"]}
        in_window = sum(origin <= t <= cutoff for t in times)
        complete = row["status"] == "COMPLETED" and end <= cutoff
        good = in_window if complete and row.get("slo_success") else 0
        scored.append(
            {
                "request_id": row["request_id"],
                "pool": row["pool"],
                "completed_by_H": complete,
                "output_tokens_within_H": in_window,
                "good_tokens": good,
                "service_status": row["status"],
                "finish_reason": record.get("finish_reason"),
                "right_censored": record.get("finish_reason") == "length",
                "slo_failure_reasons": row.get("failure_reasons"),
            }
        )
    good = sum(r["good_tokens"] for r in scored)
    return {
        "eligible": True,
        "errors": [],
        "failure_kind": None,
        "origin_unix_s": origin,
        "cutoff_unix_s": cutoff,
        "horizon_s": horizon_s,
        "good_output_tokens": good,
        "goodoutput_tokens_s": good / horizon_s,
        "request_rows": scored,
        "unfinished_or_failed_at_H": sum(not r["completed_by_H"] for r in scored),
    }
