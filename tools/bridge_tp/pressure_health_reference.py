# SPDX-License-Identifier: Apache-2.0
"""Offline health lookup and independent physical-KV warning, without START."""

import math
from collections import deque


class NativeDecodeGrowth:
    """Past-only generated-token growth; do not extrapolate prefill bursts."""

    def __init__(self, window_s=2.0, minimum_span_s=0.5):
        self.window_s, self.minimum_span_s = window_s, minimum_span_s
        self.history = deque()
        self.generation = None
        self.rate = None

    def update(self, inventory):
        generation = inventory["generation"]
        if generation == self.generation:
            return self.rate
        stamp = inventory["sampled_at"]
        if self.history and stamp < self.history[-1][0]:
            raise ValueError("native generation time regressed")
        self.generation = generation
        while self.history and stamp - self.history[0][0] > self.window_s:
            self.history.popleft()
        current = {r["request_id"]: r["generated"] for r in inventory["requests"]}
        self.history.append((stamp, current))
        span = stamp - self.history[0][0]
        self.rate = None
        if span >= self.minimum_span_s:
            produced = 0
            for (_, before), (_, after) in zip(self.history, list(self.history)[1:]):
                for key in before.keys() & after.keys():
                    if after[key] < before[key]:
                        raise ValueError("native generated tokens regressed")
                    produced += after[key] - before[key]
            self.rate = produced / span / inventory["block_size"]
        return self.rate


def build_reference(cells, windows, *, provenance, release_budget_s, diagnostic_shapes):
    """Export repeated measured cells, retaining diagnostic and tail evidence."""
    result = []
    for cell in cells:
        if (
            cell["version"] != "current"
            or cell["prefill_class"] != "none"
            or cell["support"] != "REPEATED_WORKING_POINT"
        ):
            continue
        key = (cell["pool"], cell["nd"], cell["context_bin"])
        observations = [
            w
            for w in windows
            if (
                w["version"] == "current"
                and w["width_s"] == 10
                and w["trusted_shape"]
                and w["role"] != "migration_holdout"
                and w["prefill_class"] == "none"
                and (w["pool"], w["decode_requests"], w["mean_context_bin_2048"]) == key
            )
        ]
        if len({w["run"] for w in observations}) != cell["run_count"]:
            raise ValueError("window/cell provenance differs")
        diagnostic = key in diagnostic_shapes
        result.append(
            dict(
                pool=key[0],
                nd=key[1],
                context_bin=key[2],
                context_tokens=[key[2] * 2048, (key[2] + 1) * 2048 - 1],
                runs=cell["run_count"],
                seconds=cell["seconds"],
                window_goodoutput=cell["causal_good"],
                raw_output=cell["raw"],
                health_status="DIAGNOSTIC_ONLY" if diagnostic else "REFERENCE",
                health_reason="unresolved target delivery anomaly"
                if diagnostic
                else "repeated empirical working point; not a capacity ceiling",
                windows=len(observations),
                quality_below95_windows=sum(
                    w["causal_quality_retention"] < 0.95 for w in observations
                ),
                minimum_window_goodoutput_s=min(
                    w["causal_window_itl_goodoutput_s"] for w in observations
                ),
                run_points=cell["runpoints"],
            )
        )
    if len({(c["pool"], c["nd"], c["context_bin"]) for c in result}) != len(result):
        raise ValueError("duplicate reference shape")
    return dict(
        schema="PRESSURE_HEALTH_REFERENCE_V1",
        deployment_ready=False,
        context_bin_tokens=2048,
        window_s=10,
        itl_s=0.1,
        slow_fraction=0.01,
        curve="closed-window ITL discounted output, per-run then cross-run median",
        no_interpolation_or_extrapolation=True,
        no_admission_reservation=True,
        unknown_health_never_suppresses_KV_warning=True,
        upper_capacity_boundaries="UNKNOWN; working points are not maximum batch",
        provenance=provenance,
        cells=result,
        kv=dict(
            guard_tokens=2000,
            release_budget_s=release_budget_s,
            budget_basis="historical START-to-source-release observed maximum",
            budget_is_guaranteed_upper_bound=False,
            warning=(
                "free <= guard + unallocated_prompt_blocks + "
                "ceil(decode_growth * (observation_age + release))"
            ),
            low_free_trigger_independent_of_positive_growth=True,
            candidate_physical_and_service_checks_still_required=True,
        ),
    )


def healthy_lookup(reference, pool, nd, mean_context):
    """Use only repeated exact Nd/context cells; never infer unseen capacity."""
    if (
        type(nd) is not int
        or nd <= 0
        or not isinstance(mean_context, (float, int))
        or not math.isfinite(mean_context)
        or mean_context < 0
    ):
        return dict(status="UNKNOWN", reason="invalid shape")
    context_bin = int(mean_context // reference["context_bin_tokens"])
    matched = [
        c
        for c in reference["cells"]
        if (c["pool"] == pool and c["nd"] == nd and c["context_bin"] == context_bin)
    ]
    if len(matched) != 1:
        return dict(status="UNKNOWN", reason="outside repeated shape support")
    cell = matched[0]
    if cell["health_status"] != "REFERENCE":
        return dict(status="UNKNOWN", reason=cell["health_reason"], cell=cell)
    return dict(
        status="SUPPORTED",
        cell=cell,
        reference_goodoutput_s=cell["window_goodoutput"]["median"],
    )


def health_efficiency(reference, pool, nd, mean_context, goodoutput_s):
    """Compare a closed observed window with the same measured workload shape."""
    if (
        not isinstance(goodoutput_s, (float, int))
        or not math.isfinite(goodoutput_s)
        or goodoutput_s < 0
    ):
        return dict(status="UNKNOWN", reason="invalid observed goodoutput", eta=None)
    result = healthy_lookup(reference, pool, nd, mean_context)
    denominator = result.get("reference_goodoutput_s")
    result["eta"] = goodoutput_s / denominator if denominator else None
    return result


def unallocated_prompt_blocks(requests, block_size):
    """Forecast only visible prompt blocks not already in the native ownership."""
    if type(block_size) is not int or block_size <= 0:
        return None
    total = 0
    for row in requests:
        if any(
            type(row.get(k)) is not int or row[k] < 0
            for k in ("prompt_tokens", "owned_blocks")
        ):
            return None
        total += max(
            0, math.ceil(row["prompt_tokens"] / block_size) - row["owned_blocks"]
        )
    return total


def kv_warning(
    *,
    free_blocks,
    block_size,
    sampled_at,
    published_at,
    now,
    growth_blocks_s,
    release_budget_s,
    guard_tokens=2000,
    max_age_s=2.0,
    pending_prompt_blocks=0,
):
    """Warn before the guard can be crossed during source-release latency.

    This is a past-observation extrapolation, not a future consumption bound.
    An unknown health lookup cannot suppress KV relief. Ordinary ingress has
    no reservation here. A warning asks for candidate evaluation, never START.
    """
    fresh = all(
        v is not None and math.isfinite(v) and 0 <= now - v <= max_age_s
        for v in (sampled_at, published_at)
    )
    if (
        not fresh
        or type(free_blocks) is not int
        or free_blocks < 0
        or type(block_size) is not int
        or block_size <= 0
        or not math.isfinite(guard_tokens)
        or guard_tokens <= 0
    ):
        return dict(
            status="UNKNOWN",
            needs_relief=False,
            reason="fresh physical KV evidence required",
            action="NONE",
        )
    guard = math.ceil(guard_tokens / block_size)
    known = (
        all(
            v is not None and math.isfinite(v) and v >= 0
            for v in (growth_blocks_s, release_budget_s)
        )
        and type(pending_prompt_blocks) is int
        and pending_prompt_blocks >= 0
    )
    loss = (
        pending_prompt_blocks
        + math.ceil(growth_blocks_s * (now - sampled_at + release_budget_s))
        if known
        else None
    )
    projected = free_blocks - loss if known else None
    trigger = (
        "LOW_FREE_KV"
        if free_blocks <= guard
        else "GUARD_CROSSING_BEFORE_SOURCE_RELEASE"
        if known and projected <= guard
        else None
    )
    required = max(1, guard - projected) if trigger and known else None
    return dict(
        status="WARNING" if trigger else "CLEAR" if known else "UNKNOWN",
        needs_relief=trigger is not None,
        trigger=trigger,
        action="EVALUATE_CAPACITY_CANDIDATES" if trigger else "NONE",
        free_blocks=free_blocks,
        guard_blocks=guard,
        warning_blocks=guard + loss if known else guard,
        growth_blocks_s=growth_blocks_s,
        pending_prompt_blocks=pending_prompt_blocks,
        release_budget_s=release_budget_s,
        observation_age_s=now - sampled_at,
        projected_consumption_blocks=loss,
        projected_free_at_release=projected,
        required_release_blocks=required,
        release_before_exhaustion_established=projected > 0 if known else None,
        forecast_known=known,
        priority="KV_RELIEF" if trigger else None,
        note="empirical release budget; no future demand or ending is assumed",
    )


def capacity_candidate(
    warning,
    *,
    releasable_blocks,
    target_free_blocks,
    target_pending_blocks,
    target_required_blocks,
    target_evidence_fresh,
):
    """Check observed source release and target promises; no service/owner proof."""
    reasons = []
    if not warning.get("needs_relief"):
        reasons.append("capacity warning absent")
    if not warning.get("forecast_known"):
        reasons.append("source growth/release budget unknown")
    elif not warning["release_before_exhaustion_established"]:
        reasons.append("release before physical exhaustion not established")
    required = warning.get("required_release_blocks")
    if (
        type(releasable_blocks) is not int
        or releasable_blocks <= 0
        or required is None
        or releasable_blocks < required
    ):
        reasons.append("insufficient or unknown uniquely releasable KV")
    if not target_evidence_fresh:
        reasons.append("target physical evidence stale/unknown")
    if any(
        type(v) is not int or v < 0
        for v in (
            target_free_blocks,
            target_pending_blocks,
            target_required_blocks,
        )
    ):
        reasons.append("target commitment evidence invalid")
    elif target_required_blocks + target_pending_blocks > target_free_blocks:
        reasons.append("target physical commitments exceed free blocks")
    return dict(
        physical_candidate=not reasons,
        rejections=reasons,
        action="NO_START_IN_OFFLINE_REPLAY",
        remaining_checks=[
            "active owner/source stream",
            "context/stop contract",
            "target service budget",
            "migration channel/ledger",
            "candidate work survives transfer",
        ],
    )
