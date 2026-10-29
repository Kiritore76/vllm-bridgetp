# SPDX-License-Identifier: Apache-2.0
"""Matched workloads and honest metrics for the existing pressure controller."""

import copy
import hashlib
import json
import math
import statistics

from tools.bridge_tp import qwen3_pressure_observation as observation
from tools.bridge_tp.run_pressure_comparison import percentile, result_metrics


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def recipes(prior, repetitions, include_natural=True):
    """Freeze historical natural lengths before either comparison arm runs.

    Fixed stops make workload counts equal across TP arithmetic differences.
    These are declared client stops, not hidden labels provided to the policy.
    This benchmark measures system behavior, not natural-EOS prediction accuracy.
    """
    result = observation.read(prior / "result.json")
    if (
        result.get("status") != "COMPLETED_OBSERVATION_ONLY"
        or result.get("total_profile_attempts") != 31
    ):
        raise ValueError("requires the completed four-scene observation batch")
    if repetitions not in (1, 2):
        raise ValueError("one or two bounded repetitions required")
    output = []
    order = [
        "kv_pressure",
        "short_burst_recovery",
        "natural_long_pressure",
        "target_busy",
    ]
    for repetition in range(1, repetitions + 1):
        for name in order:
            values = copy.deepcopy(observation.read(prior / name / "arrivals.json"))
            measured = observation.read(prior / name / "observation_summary.json")
            lengths = {r["job_id"]: r["tokens"] for r in measured["request_quality"]}
            for job in values:
                request = job["request"]
                if not request.get("ignore_eos", False):
                    request["max_tokens"] = lengths[job["id"]]
                request.update(ignore_eos=True, seed=20261029 + repetition)
                job["stop_scope"] = "matched fixed-length system benchmark"
                if (
                    not 0 < request["max_tokens"] <= 8192
                    or len(request["prompt"]) + request["max_tokens"] > 8192
                ):
                    raise ValueError(
                        "matched workload violates source context contract"
                    )
            output.append(
                dict(
                    name=f"{name}-r{repetition}",
                    scenario=name,
                    repetition=repetition,
                    fixed_output=True,
                    jobs=values,
                    input_sha256=digest(values),
                )
            )
    if include_natural:
        natural = copy.deepcopy(
            observation.read(prior / "natural_long_pressure/arrivals.json")
        )
        for job in natural:
            if job["request"].get("ignore_eos") is not False:
                raise ValueError(
                    "natural EOS supplement must retain the original stop rule"
                )
            job["request"]["seed"] = 20261030
        output.append(
            dict(
                name="natural-eos-supplement-r1",
                scenario="natural_eos_supplement",
                repetition=1,
                fixed_output=False,
                jobs=natural,
                input_sha256=digest(natural),
            )
        )
    return output


def config(values):
    return dict(
        arms=["stay", "migrate"],
        recipes=values,
        maximum_migrations=0,
        policy_overrides={},
        pool_comparison=True,
        routing_weights=[1.0, 1.0],
        natural_eos=False,
        max_dispatch_lag_s=1.0,
        horizon_s=300.0,
    )


def metrics(run_dir, jobs, horizon=300.0, natural_eos=False):
    rows = observation.read(run_dir / "pressure_request_results.json")
    audit = [
        json.loads(line)
        for line in (run_dir / "pressure_audit.jsonl").read_text().splitlines()
    ]
    final = [r for r in audit if r.get("kind") == "pressure_final"][-1]
    if final["outcome"] not in {"COMPLETED", "COMPLETED_PENDING_OWNER_CLEANUP"}:
        raise ValueError("controller did not finish workload")
    origin = next(
        r["unix_s"] for r in audit if r.get("kind") == "pressure_clock_origin"
    )
    values = result_metrics(rows, jobs, origin, 1.0, natural_eos)
    by_id = {r["job_id"]: r for r in rows}
    raw = soft = hard = 0
    buckets = set()
    intervals = []
    for value in values:
        frac, gap = value["slow_fraction"], value["max_itl_s"]
        weight = min(1.0, 0.01 / frac if frac else 1.0, 1.0 / gap if gap > 1 else 1.0)
        passed = frac <= 0.01 and gap <= 1.0
        emitted = by_id[value["job_id"]]["response"]["emitted"]
        times = [
            r["unix_s"] - origin for r in emitted if 0 <= r["unix_s"] - origin < horizon
        ]
        raw += len(times)
        soft += len(times) * weight
        hard += len(times) * passed
        buckets.update(int(t // 10) for t in times)
        transport = by_id[value["job_id"]]["source_transport"]
        intervals.append(
            (
                max(0.0, transport["request_started_unix_s"] - origin),
                min(horizon, value["completion_s"]),
            )
        )
        value["slo_pass"] = passed
    covered = 0.0
    last_end = 0.0
    for left, right in sorted(intervals):
        if right > max(left, last_end):
            covered += right - max(left, last_end)
            last_end = right
    small = [r for r in values if r["output_tokens"] <= 128]
    completed = [r for r in values if 0 <= r["completion_s"] < horizon]
    starts = [r for r in audit if r.get("kind") == "pressure_actual_start"]
    releases = [
        r for r in audit if r.get("kind") == "pressure_source_release_confirmed"
    ]
    summary = dict(
        requests=len(values),
        output_tokens=sum(r["output_tokens"] for r in values),
        slo_pass=sum(r["slo_pass"] for r in values),
        ttft_p95_s=percentile([r["ttft_s"] for r in values], 0.95),
        e2e_mean_s=statistics.mean(r["e2e_s"] for r in values),
        e2e_p95_s=percentile([r["e2e_s"] for r in values], 0.95),
        short_ttft_p95_s=percentile([r["ttft_s"] for r in small], 0.95),
        short_e2e_p95_s=percentile([r["e2e_s"] for r in small], 0.95),
        max_itl_s=max(r["max_itl_s"] for r in values),
        maximum_handoff_s=max((r["handoff_stall_s"] or 0 for r in values), default=0.0),
        workload_makespan_s=max(r["completion_s"] for r in values),
        fixed_arrivals_valid=all(r["arrival_valid"] for r in values),
        migration_starts=len(starts),
        source_releases=len(releases),
        committed_requests=sum(r["migrated"] for r in values),
        channel_reuses=sum(r.get("kind") == "pressure_channel_reused" for r in audit),
        cancellations_requested=sum(
            r.get("kind") == "pressure_cancel_requested" for r in audit
        ),
        window_s=horizon,
        window_raw_tokens_s=raw / horizon,
        window_soft_goodoutput_s=soft / horizon,
        window_hard_token_goodput_s=hard / horizon,
        window_request_goodput_s=sum(r["slo_pass"] for r in completed) / horizon,
        inflight_coverage_fraction=covered / horizon,
        ten_second_buckets_with_output=len(buckets),
        load_coverage_pass=covered / horizon >= 0.95
        and len(buckets) == math.ceil(horizon / 10),
        note=(
            "all failures retained; incomplete load coverage "
            "is not peak throughput evidence"
        ),
    )
    return summary, values, audit


def compare(stay, move):
    same = (
        stay["requests"] == move["requests"]
        and stay["output_tokens"] == move["output_tokens"]
    )
    fields = [
        "window_soft_goodoutput_s",
        "window_hard_token_goodput_s",
        "e2e_mean_s",
        "e2e_p95_s",
        "short_ttft_p95_s",
        "short_e2e_p95_s",
        "max_itl_s",
        "workload_makespan_s",
    ]
    result = {}
    for key in fields:
        a, b = stay[key], move[key]
        result[key] = dict(
            stay=a,
            migrate=b,
            change_percent=(b / a - 1) * 100 if a and b is not None else None,
        )
    return dict(
        matched_output_work=same,
        arrivals_valid=stay["fixed_arrivals_valid"] and move["fixed_arrivals_valid"],
        actual_start_observed=move["migration_starts"] > 0,
        actual_migration_observed=move["committed_requests"] > 0
        and move["source_releases"] > 0,
        slo_pass=dict(stay=stay["slo_pass"], migrate=move["slo_pass"]),
        metrics=result,
        load_coverage_pass=stay["load_coverage_pass"] and move["load_coverage_pass"],
        note=(
            "positive rate changes improve throughput; "
            "negative time changes improve latency; no cherry-picking"
        ),
    )
