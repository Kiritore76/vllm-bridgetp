# SPDX-License-Identifier: Apache-2.0
"""Passive workload pressure from measured efficient working points.

This is a capacity proxy, not a counterfactual goodoutput or latency predictor.
No future arrivals, final output lengths, or probing requests are inputs.
"""

import math


def build_profiles(reference, evidence, efficient_fraction=0.9):
    """Choose the smallest measured Nd reaching the selected fraction of peak G."""
    if not 0 < efficient_fraction <= 1:
        raise ValueError("invalid efficient fraction")
    profiles = {}
    for pool, label in (("tp1", "source"), ("tp4", "target")):
        cells = [
            c
            for c in reference["cells"]
            if c["pool"] == label and c["health_status"] == "REFERENCE"
        ]
        capacities = []
        for cb in sorted({c["context_bin"] for c in cells}):
            group = [c for c in cells if c["context_bin"] == cb]
            peak = max(c["window_goodoutput"]["median"] for c in group)
            eligible = [
                c
                for c in group
                if c["window_goodoutput"]["median"] >= efficient_fraction * peak
            ]
            capacities.append([cb, min(c["nd"] for c in eligible)])
        prefill = [
            c["engine_ms"]["p95"] / 1000 / c["prefill_tokens_range"]["min"]
            for c in evidence["service_lookup"]
            if c["pool"] == label
            and c["runs"] >= 2
            and c["samples"] >= 5
            # Tiny tails include a nearly fixed decode-step cost and
            # cannot be linearly scaled into full-prompt demand.
            and c["prefill_tokens_range"]["min"] >= 256
        ]
        if not capacities or not prefill:
            raise ValueError("measured workload/prefill support missing")
        profiles[pool] = dict(
            capacities=capacities,
            prefill_s_per_token=max(prefill),
            efficient_fraction=efficient_fraction,
            basis="smallest repeated Nd within selected fraction of measured peak G; "
            "prefill uses repeated local P95 per-token envelope",
            maximum_capacity_claim=False,
        )
    return profiles


def slot_cost(profile, context):
    """Interpolate request cost inside supported context-bin centers only.

    Outside the measured context range, use the smallest measured Nd capacity
    as an explicitly conservative fallback; never call it an eta observation.
    """
    if not math.isfinite(context) or context < 0:
        raise ValueError("invalid workload context")
    points = [(cb * 2048 + 1024, 1 / nd) for cb, nd in profile["capacities"]]
    if context < points[0][0] or context > points[-1][0]:
        return max(y for _, y in points), "conservative outside-context fallback"
    for x, y in points:
        if context == x:
            return y, "measured context center"
    for (left, a), (right, b) in zip(points, points[1:]):
        if left < context < right:
            return a + (b - a) * (context - left) / (
                right - left
            ), "interpolated slot cost"
    return points[0][1], "measured context center"


def observe(profile, inventory, now):
    if not 0 <= now - inventory["sampled_at"] <= 2:
        return dict(status="UNKNOWN", reason="native inventory stale/future")
    load = pending = 0
    fallback = 0
    for row in inventory["requests"]:
        prompt, computed = row["prompt_tokens"], row["computed_tokens"]
        if any(type(v) is not int or v < 0 for v in (prompt, computed)):
            return dict(status="UNKNOWN", reason="invalid request shape")
        cost, basis = slot_cost(profile, max(prompt, computed))
        fallback += "fallback" in basis
        load += cost
        pending += max(0, prompt - computed)
    return dict(
        status="OBSERVED",
        reason="fresh passive native workload",
        sampled_at=inventory["sampled_at"],
        load=load,
        prefill_s=pending * profile["prefill_s_per_token"],
        requests=len(inventory["requests"]),
        fallback_requests=fallback,
        profile=profile,
    )


def valid(pool, cfg, now):
    value = pool.workload
    return bool(
        value
        and value.get("status") == "OBSERVED"
        and all(
            math.isfinite(t) and 0 <= now - t <= cfg.max_age_s
            for t in (pool.sampled_at, pool.published_at, value["sampled_at"])
        )
        and all(
            math.isfinite(value[k]) and value[k] >= 0 for k in ("load", "prefill_s")
        )
    )


def projected(pool, prompt_tokens, cfg, now, pending_decode=0):
    if not valid(pool, cfg, now):
        return None
    value = pool.workload
    unit, _ = slot_cost(value["profile"], prompt_tokens)
    return max(
        value["load"] + pending_decode + unit,
        (value["prefill_s"] + prompt_tokens * value["profile"]["prefill_s_per_token"])
        / cfg.sustain_s,
    )
