# SPDX-License-Identifier: Apache-2.0
"""Frozen 300-second arrivals and full-drain maximum-gap goodoutput scoring."""

import math

from tools.bridge_tp import pressure_kv_pilot as kv

METRIC = "ITL100MS_SLOW1PCT_MAXGAP1S_MIN_WEIGHT_V1"


def frozen_arrivals(repetition):
    rows = [
        dict(
            id=f"r{repetition}-target-background-{i}",
            at_s=i * 80,
            route="tp4",
            prompt_tokens=2048,
            max_tokens=6144,
        )
        for i in range(4)
    ]
    rows += [
        dict(
            id=f"r{repetition}-source-background-{i}",
            at_s=5 + i * 0.02,
            route="tp1",
            prompt_tokens=4096,
            max_tokens=3072,
        )
        for i in range(5)
    ]
    rows += [
        dict(
            id=f"r{repetition}-source-arrival-{i}",
            at_s=110 + i * 24,
            route="tp1",
            prompt_tokens=4096,
            max_tokens=3072,
        )
        for i in range(8)
    ]
    return sorted(rows, key=lambda r: r["at_s"])


def validate(config, inputs):
    from tools.bridge_tp.run_pressure_comparison import arrivals

    if (
        config["schema"] != "PRESSURE_SUSTAINED_V1"
        or config["arms"] != ["stay", "migrate"]
        or config["maximum_migrations"] != 1
        or config["policy_overrides"]
        != dict(capacity_only=True, capacity_lookahead_s=kv.RELEASE_BUDGET_S)
        or config["horizon_s"] != 300
        or config["metric"] != METRIC
        or config["maximum_gap_s"] != 1
        or config["itl_s"] != 0.1
        or config["slow_fraction"] != 0.01
        or config["max_dispatch_lag_s"] != 0.5
        or config["minimum_remaining_at_start"] != 1024
        or len(config["recipes"]) != 2
        or {r["repetition"] for r in config["recipes"]} != {1, 2}
        or len({r["name"] for r in config["recipes"]}) != 2
    ):
        raise ValueError("sustained pilot differs from the frozen four-run plan")
    for recipe in config["recipes"]:
        if recipe["arrivals"] != frozen_arrivals(recipe["repetition"]):
            raise ValueError("sustained arrival schedule differs")
        jobs = arrivals(inputs, recipe)
        if len(jobs) != 17 or any(j["at_s"] >= 300 for j in jobs):
            raise ValueError("require 17 arrivals before the fixed cutoff")


def valid_output(row, request, natural_eos=False):
    """Require complete indexed output; an ordinary EOS may precede the cap."""
    emitted = row["response"]["emitted"]
    count, cap = len(emitted), request["max_tokens"]
    valid_length = count == cap
    if natural_eos:
        transport = row.get("source_transport") or {}
        reason = transport.get("finish_reason")
        valid_length = (
            row.get("source_done") is True
            and 0 < count <= cap
            and (
                reason == "stop"
                or (reason == "length" and count == cap)
                or (reason == "abort" and row["response"]["committed"])
            )
        )
    return (
        not row["source_error"]
        and valid_length
        and [t["index"] for t in emitted] == list(range(count))
    )


def score(rows, config):
    """Count only window tokens; quality uses every completed visible interval."""
    horizon = config["horizon_s"]
    origin = config.get("origin_unix_s")
    if origin is None:
        origin = min(r["source_transport"]["request_started_unix_s"] for r in rows)
    records, bins, active = [], set(), []
    pools = dict(tp1=0.0, tp4=0.0)
    for row in rows:
        emitted = row["response"]["emitted"]
        if not valid_output(row, row["request"], config.get("natural_eos", False)):
            raise ValueError("incomplete request cannot be scored as a service zero")
        times = [t["unix_s"] for t in emitted]
        start = row["source_transport"]["request_started_unix_s"]
        if (
            any(not math.isfinite(t) for t in times)
            or times != sorted(times)
            or times[0] < start
        ):
            raise ValueError("invalid visible token times")
        gaps = [b - a for a, b in zip(times, times[1:])]
        slow = sum(g > config["itl_s"] for g in gaps)
        fraction = slow / len(gaps) if gaps else 0
        old_weight = min(1, config["slow_fraction"] / fraction) if fraction else 1
        max_gap = max(gaps, default=0)
        gap_weight = min(1, config["maximum_gap_s"] / max_gap) if max_gap else 1
        weight = min(old_weight, gap_weight)
        slo_itl_pass = fraction <= config["slow_fraction"]
        slo_gap_pass = max_gap <= config["maximum_gap_s"]
        slo_pass = slo_itl_pass and slo_gap_pass
        selected = [t for t in emitted if origin <= t["unix_s"] < origin + horizon]
        for token in selected:
            bins.add(int((token["unix_s"] - origin) // 10))
            pool = (
                "tp4"
                if row["response"]["committed"] and token["origin"] == "target"
                else row["initial_pool"]
            )
            pools[pool] += weight
        active.append((max(origin, start), min(origin + horizon, times[-1])))
        records.append(
            dict(
                request_id=row["job_id"],
                initial_pool=row["initial_pool"],
                migrated=row["response"]["committed"],
                window_tokens=len(selected),
                drain_tokens=len(emitted) - len(selected),
                ttft_s=times[0] - start,
                e2e_s=times[-1] - start,
                tpot_s=(times[-1] - times[0]) / len(gaps) if gaps else None,
                intervals=len(gaps),
                slow_intervals=slow,
                slow_fraction=fraction,
                maximum_gap_s=max_gap,
                slo_itl_pass=slo_itl_pass,
                slo_maximum_gap_pass=slo_gap_pass,
                slo_pass=slo_pass,
                in_arrival_cohort=origin <= start < origin + horizon,
                itl_weight=old_weight,
                maximum_gap_weight=gap_weight,
                final_weight=weight,
                old_good_tokens=len(selected) * old_weight,
                good_tokens=len(selected) * weight,
                completed_offset_s=times[-1] - origin,
            )
        )
    end, busy = origin, 0.0
    for a, b in sorted(active):
        busy += max(0, b - max(a, end))
        end = max(end, b)
    raw = sum(r["window_tokens"] for r in records)
    good = sum(r["good_tokens"] for r in records)
    cohort = [r for r in records if r["in_arrival_cohort"]]

    def attainment(group):
        passed = sum(r["slo_pass"] for r in group)
        return dict(
            requests=len(group),
            passed=passed,
            rate=passed / len(group) if group else None,
        )

    slo = dict(
        definition="slow_interval_fraction<=limit AND maximum_gap<=limit",
        quality_scope="full request including drain; TTFT/queue excluded",
        cohort="actual request start in fixed measurement window",
        itl_s=config["itl_s"],
        slow_fraction=config["slow_fraction"],
        maximum_gap_s=config["maximum_gap_s"],
        all_requests=attainment(cohort),
        decode_requests=attainment([r for r in cohort if r["intervals"] > 0]),
        single_token_requests=attainment([r for r in cohort if r["intervals"] == 0]),
        by_initial_pool={
            pool: attainment([r for r in cohort if r["initial_pool"] == pool])
            for pool in pools
        },
        excluded_late_requests=len(records) - len(cohort),
    )
    if abs(sum(pools.values()) - good) > 1e-6:
        raise ValueError("pool goodoutput accounting mismatch")
    return dict(
        metric=config["metric"],
        horizon_s=horizon,
        origin_unix_s=origin,
        quality_scope="full completed request; drain gaps included; TTFT excluded",
        drain_tokens_counted=False,
        raw_window_tokens=raw,
        raw_rate=raw / horizon,
        old_itl_G=sum(r["old_good_tokens"] for r in records) / horizon,
        good_tokens=good,
        goodoutput_tokens_s=good / horizon,
        slo_attainment=slo,
        # Completion goodput counts only completions before cutoff; token
        # goodput counts only window tokens. Neither credits drain production.
        slo_request_goodput=sum(
            r["slo_pass"] and r["completed_offset_s"] < horizon for r in cohort
        )
        / horizon,
        slo_token_goodput=sum(r["window_tokens"] for r in records if r["slo_pass"])
        / horizon,
        actual_pool_G={p: v / horizon for p, v in pools.items()},
        inflight_fraction=busy / horizon,
        output_10s_bins=len(bins),
        expected_output_10s_bins=math.ceil(horizon / 10),
        load_coverage="COVERED"
        if busy / horizon >= 0.95 and len(bins) == math.ceil(horizon / 10)
        else "COVERAGE_GAP",
        last_output_offset_s=max(r["completed_offset_s"] for r in records),
        request_rows=records,
        note="measured output and quality; actual migration coverage is separate",
    )


def queue_summary(audit, origin, horizon):
    result = {}
    for pool in ("tp1", "tp4"):
        samples = [
            r
            for r in audit
            if r.get("kind") == "pressure_metrics_sample"
            and r["pool"] == pool
            and r.get("metrics") is not None
            and origin <= r["unix_s"] < origin + horizon
        ]
        by_bin = []
        for left in range(0, int(horizon), 30):
            subset = [
                r["metrics"]
                for r in samples
                if left <= r["unix_s"] - origin < left + 30
            ]
            by_bin.append(
                dict(
                    left_s=left,
                    samples=len(subset),
                    mean_waiting=sum(r["num_waiting"] for r in subset) / len(subset)
                    if subset
                    else None,
                    max_waiting=max((r["num_waiting"] for r in subset), default=None),
                )
            )
        result[pool] = dict(
            samples=len(samples),
            bins_30s=by_bin,
            final_window_sample=samples[-1]["metrics"] if samples else None,
        )
    return result
