# SPDX-License-Identifier: Apache-2.0
"""Frozen four-mode workloads and the existing A100 pressure score on TP2."""

import copy
import random
import statistics
from pathlib import Path

from tools.bridge_tp import pressure_sustained as sustained
from tools.bridge_tp import qwen3_pressure_observation as observation
from tools.bridge_tp import qwen3_stay_migrate as matched
from tools.bridge_tp.run_pressure_comparison import percentile

ARMS = ["weighted", "least", "stay", "migrate"]


def supplement(prior):
    """Read the immutable partial batch; only missing cells may run again."""
    prior = Path(prior)
    old = observation.read(prior / "result.json")
    cfg = observation.read(prior / "comparison_recipe.json")
    identity = observation.read(prior / "identity.json")
    from tools.bridge_tp.qwen3_online_validation import CHECKPOINT_SHA

    if (
        old.get("status") != "TECHNICAL_FAILURE"
        or cfg.get("schema") != "QWEN3_FOUR_MODE_FIXED_V1"
        or cfg.get("arms") != ARMS
        or len(cfg.get("recipes", [])) != 3
        or identity.get("checkpoint_sha256") != CHECKPOINT_SHA
        or observation.read(prior / "comparison_policy.json")
        != observation.read(observation.ASSETS / "policy.json")
    ):
        raise ValueError("supplement requires matching frozen four-mode contract")
    if (
        cfg["routing_weights"]
        != measured_weights(
            observation.read(observation.ASSETS / "health_reference.json")
        )[0]
    ):
        raise ValueError("supplement weights changed")
    expected = set()
    for recipe in cfg["recipes"]:
        jobs = recipe["jobs"]
        if (
            len(jobs) != 75
            or matched.digest(jobs) != recipe["input_sha256"]
            or observation.read(prior / (recipe["name"] + "_arrivals.json")) != jobs
        ):
            raise ValueError("supplement workload identity changed")
        expected.update(recipe["name"] + "-" + a for a in ARMS)
    inherited, evaluations = [], []
    for run in old["runs"]:
        if not observation.read(prior / run["name"] / "owned_shutdown_receipt.json")[
            "all_owned_groups_ended"
        ]:
            raise ValueError("prior owned process cleanup was not proven")
        if run["status"] != "COMPLETED":
            continue
        if run["name"] not in expected:
            raise ValueError("unknown inherited cell")
        row = copy.deepcopy(run)
        row["provenance"] = dict(
            batch=str(prior), revision=identity["revision"], reused=True
        )
        inherited.append(row)
        score = observation.read(prior / run["name"] / "sustained_goodoutput.json")
        if (
            abs(
                score["goodoutput_tokens_s"]
                - row["summary"]["window_soft_goodoutput_s"]
            )
            > 1e-8
        ):
            raise ValueError("inherited score differs from completed ledger")
        evaluations.append(dict(arm=run["arm"], score=score))
    done = {r["name"] for r in inherited}
    if len(done) != len(inherited) or not done < expected:
        raise ValueError("invalid or already complete supplement")
    return cfg, inherited, evaluations, expected - done, old["total_profile_attempts"]


def measured_weights(reference):
    if reference.get("target_tp_size") != 2:
        raise ValueError("four-mode weights require TP2 reference")
    selected, weights = [], []
    for pool in ("source", "target"):
        peaks = []
        for context_bin in (1, 2):
            cells = [
                c
                for c in reference["cells"]
                if c["pool"] == pool
                and c["context_bin"] == context_bin
                and c["health_status"] == "REFERENCE"
                and c["runs"] >= 2
            ]
            if not cells:
                raise ValueError("missing repeated common-context weight cell")
            peak = max(cells, key=lambda c: c["window_goodoutput"]["median"])
            peaks.append(peak["window_goodoutput"]["median"])
            selected.append(copy.deepcopy(peak))
        weights.append(statistics.median(peaks))
    if not all(w > 0 for w in weights):
        raise ValueError("nonpositive measured weight")
    return weights, selected


def config(prior, inputs, reference):
    # Reuse audited source prompts and one previously measured coding task.
    original = matched.recipes(prior, 1, include_natural=False)
    natural = next(r for r in original if r["scenario"] == "natural_long_pressure")
    code = copy.deepcopy(natural["jobs"][0]["request"])
    short = next(r for r in original if r["scenario"] == "short_burst_recovery")
    short = copy.deepcopy(
        next(j["request"] for j in short["jobs"] if j["request"]["max_tokens"] == 128)
    )
    templates = []
    for key, cap in (("2048", 5120), ("6144", 1024), ("2048", 2048)):
        payload = copy.deepcopy(inputs["requests"][key])
        payload.pop("request_id", None)
        payload["max_tokens"] = cap
        templates.append(payload)
    templates += [short, code]
    recipes = []
    for repetition in (1, 2, 3):
        seed = 20261029 + repetition
        order = list(range(5)) * 15
        random.Random(seed).shuffle(order)
        jobs = []
        for index, template in enumerate(order):
            request = copy.deepcopy(templates[template])
            request.update(
                ignore_eos=True, seed=seed, stream=True, return_token_ids=True
            )
            if len(request["prompt"]) + request["max_tokens"] > 8192:
                raise ValueError("four-mode workload exceeds shared context")
            jobs.append(
                dict(
                    id=f"four-r{repetition}-request-{index}",
                    at_s=index * 4,
                    route=None,
                    request=request,
                    stop_scope="fixed-length system comparison",
                    template=template,
                )
            )
        recipes.append(
            dict(
                name=f"four-mixed-r{repetition}",
                scenario="four_mixed",
                repetition=repetition,
                fixed_output=True,
                jobs=jobs,
                input_sha256=matched.digest(jobs),
            )
        )
    weights, cells = measured_weights(reference)
    return dict(
        schema="QWEN3_FOUR_MODE_FIXED_V1",
        arms=ARMS,
        recipes=recipes,
        maximum_migrations=0,
        policy_overrides={},
        pool_comparison=True,
        routing_weights=weights,
        weight_cells=cells,
        weight_basis="median of repeated peak G in common bins 1/2; "
        "capacity proxy, not sustainable capacity",
        horizon_s=300.0,
        max_dispatch_lag_s=1.0,
        natural_eos=False,
        metric=sustained.METRIC,
        itl_s=0.1,
        slow_fraction=0.01,
        maximum_gap_s=1.0,
        arrivals_per_second=0.25,
        note="75 arrivals at 0..296s; five templates repeated 15 times; "
        "three seeds; no online policy or load tuning",
    )


def distribution(values):
    values = [v for v in values if v is not None]
    return {
        k: percentile(values, q)
        for k, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1))
    }


def request_group(rows):
    return dict(
        requests=len(rows),
        passed=sum(r["slo_pass"] for r in rows),
        **{
            key: distribution([r[key] for r in rows])
            for key in ("ttft_s", "tpot_s", "e2e_s", "slow_fraction", "maximum_gap_s")
        },
    )


def full_score(rows, cfg, origin, audit):
    score = sustained.score(rows, dict(cfg, origin_unix_s=origin))
    by_id = {r["job_id"]: r for r in rows}
    for r in score["request_rows"]:
        raw = by_id[r["request_id"]]
        r.update(
            prompt_tokens=len(raw["request"]["prompt"]),
            output_tokens=len(raw["response"]["emitted"]),
        )
    cohort = [r for r in score["request_rows"] if r["in_arrival_cohort"]]
    score["latency"] = request_group(cohort)
    score["groups"] = dict(
        initial_pool={
            p: request_group([r for r in cohort if r["initial_pool"] == p])
            for p in ("tp1", "tp4")
        },
        prompt={
            name: request_group(
                [r for r in cohort if low <= r["prompt_tokens"] <= high]
            )
            for name, low, high in (
                ("<=2k", 0, 2048),
                ("2k-4k", 2049, 4096),
                (">4k", 4097, 100000),
            )
        },
        output={
            name: request_group(
                [r for r in cohort if low <= r["output_tokens"] <= high]
            )
            for name, low, high in (
                ("<=512", 0, 512),
                ("513-2048", 513, 2048),
                (">2048", 2049, 100000),
            )
        },
    )
    raw_pools = dict(tp1=0, tp4=0)
    gaps = []
    for row in rows:
        emitted = row["response"]["emitted"]
        gaps.extend(b["unix_s"] - a["unix_s"] for a, b in zip(emitted, emitted[1:]))
        for token in emitted:
            if origin <= token["unix_s"] < origin + cfg["horizon_s"]:
                pool = (
                    "tp4"
                    if row["response"]["committed"] and token["origin"] == "target"
                    else row["initial_pool"]
                )
                raw_pools[pool] += 1
    score["actual_pool_window_tokens"] = raw_pools
    score["token_itl_s"] = distribution(gaps)
    score["tail_interval_counts"] = {
        str(ms): sum(g > ms / 1000 for g in gaps) for ms in (100, 200, 500, 1000)
    }
    score["queue_observations"] = sustained.queue_summary(
        audit, origin, cfg["horizon_s"]
    )
    score["pool_labels"] = dict(tp1="TP1", tp4="TP2 (legacy field name)")
    return score


def comparisons(ledger):
    output = []
    for pair in dict.fromkeys(r["pair"] for r in ledger):
        arms = {
            r["arm"]: r
            for r in ledger
            if r["pair"] == pair and r["status"] == "COMPLETED"
        }
        if set(arms) != set(ARMS):
            continue
        first = arms["stay"]
        output.append(
            dict(
                pair=pair,
                arrivals_valid=all(
                    r["summary"]["fixed_arrivals_valid"] for r in arms.values()
                ),
                matched_output_work=all(
                    r["summary"]["output_tokens"] == first["summary"]["output_tokens"]
                    and r["input_sha256"] == first["input_sha256"]
                    for r in arms.values()
                ),
                all_load_covered=all(
                    r["summary"]["load_coverage_pass"] for r in arms.values()
                ),
                comparisons={
                    arm: matched.compare(r["summary"], arms["migrate"]["summary"])
                    for arm, r in arms.items()
                    if arm != "migrate"
                },
            )
        )
    return output


def aggregate(evaluations):
    output = {}
    for arm in ARMS:
        scores = [e["score"] for e in evaluations if e["arm"] == arm]
        if not scores:
            continue
        rows = [r for s in scores for r in s["request_rows"] if r["in_arrival_cohort"]]
        duration = sum(s["horizon_s"] for s in scores)
        values = [s["goodoutput_tokens_s"] for s in scores]
        output[arm] = dict(
            completed_runs=len(scores),
            denominator_s=duration,
            raw_rate=sum(s["raw_window_tokens"] for s in scores) / duration,
            soft_G=sum(s["good_tokens"] for s in scores) / duration,
            soft_G_per_run=values,
            sample_sd_G=statistics.stdev(values) if len(values) > 1 else None,
            old_itl_G=sum(s["old_itl_G"] * s["horizon_s"] for s in scores) / duration,
            hard_token_goodput=sum(
                s["slo_token_goodput"] * s["horizon_s"] for s in scores
            )
            / duration,
            hard_request_goodput=sum(
                s["slo_request_goodput"] * s["horizon_s"] for s in scores
            )
            / duration,
            all_requests=request_group(rows),
            decode=request_group([r for r in rows if r["intervals"]]),
            single_token=request_group([r for r in rows if not r["intervals"]]),
            all_load_covered=all(s["load_coverage"] == "COVERED" for s in scores),
            pool_labels=dict(tp1="TP1", tp4="TP2 (legacy field name)"),
            note="pooled request quantiles; completed runs only; "
            "performance acceptance is separate",
        )
    return output
