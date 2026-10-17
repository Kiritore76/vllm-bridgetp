# SPDX-License-Identifier: Apache-2.0
"""Frozen KV-release experiment checks; no forced migration or ingress gate."""

import math

RELEASE_BUDGET_S = 16.86410117149353


def validate(config, inputs):
    from tools.bridge_tp.run_pressure_comparison import arrivals

    if (
        config["schema"] != "PRESSURE_KV_RELEASE_V1"
        or config["arms"] != ["stay", "migrate"]
        or config["maximum_migrations"] != 1
        or config["policy_overrides"]
        != dict(capacity_only=True, capacity_lookahead_s=RELEASE_BUDGET_S)
        or config["itl_s"] != 0.1
        or config["slow_fraction"] != 0.01
        or config["max_dispatch_lag_s"] != 1.0
        or config["minimum_remaining_at_start"] != 1024
        or len(config["recipes"]) != 2
        or {r["repetition"] for r in config["recipes"]} != {1, 2}
        or len({r["name"] for r in config["recipes"]}) != 2
    ):
        raise ValueError("KV-release differs from the frozen four-run plan")
    for recipe in config["recipes"]:
        jobs = arrivals(inputs, recipe)
        expected = [("tp4", 0, 2048, 6144)] + [
            ("tp1", 5 + i * 0.02, 4096, 3072) for i in range(5)
        ]
        actual = [
            (
                j["route"],
                j["at_s"],
                len(j["request"]["prompt"]),
                j["request"]["max_tokens"],
            )
            for j in jobs
        ]
        if actual != expected or len({j["id"] for j in jobs}) != 6:
            raise ValueError("KV-release fixed arrivals differ")


def capacity_shape(inventories, jobs, guard_blocks=125):
    """Check the experiment can stress KV and one offload can relieve it."""
    source, target = inventories
    if source["block_size"] != 16 or target["block_size"] != 16:
        raise ValueError("KV pilot requires the frozen 16-token block geometry")
    source_jobs = [j["request"] for j in jobs if j["route"] == "tp1"]
    target_jobs = [j["request"] for j in jobs if j["route"] == "tp4"]

    def blocks(req):
        return math.ceil((len(req["prompt"]) + req["max_tokens"]) / 16)

    full = sum(map(blocks, source_jobs))
    initial = sum(math.ceil(len(j["prompt"]) / 16) for j in source_jobs)
    moved = max(map(blocks, source_jobs))
    target_full = sum(map(blocks, target_jobs)) + moved
    valid = (
        initial + guard_blocks <= source["free_blocks"] < full
        and full - moved + guard_blocks <= source["free_blocks"]
        and target_full + guard_blocks <= target["free_blocks"]
    )
    return dict(
        status="VALID" if valid else "INVALID_EXPERIMENT_SHAPE",
        source_free_blocks=source["free_blocks"],
        target_free_blocks=target["free_blocks"],
        source_initial_prompt_blocks=initial,
        source_full_stop_blocks=full,
        source_after_one_offload_blocks=full - moved,
        target_full_stop_with_offload_blocks=target_full,
        guard_blocks=guard_blocks,
        note="experiment feasibility only; ordinary ingress remains ungated",
    )


def assess(audit, metrics, arm, minimum_remaining):
    starts = [r for r in audit if r.get("kind") == "pressure_actual_start"]
    releases = [
        r for r in audit if r.get("kind") == "pressure_source_release_confirmed"
    ]
    decisions = [r for r in audit if r.get("execution") == "RESERVED"]
    selected = None
    if decisions:
        decision = decisions[0]
        selected = next(
            c
            for c in decision["decision"]["candidates"]
            if c["request"]["request_id"] == decision["decision"]["request_id"]
        )
    remaining = (
        min(8192, selected["request"]["max_output_tokens"])
        - selected["request"]["generated"]
        if selected
        else None
    )
    start = starts[0]["unix_s"] if starts else None
    counters = {}
    for pool in ("tp1", "tp4"):
        rows = [
            r
            for r in audit
            if r.get("kind") == "pressure_metrics_sample"
            and r["pool"] == pool
            and r.get("metrics") is not None
        ]
        values = [r["metrics"]["preemptions_total"] for r in rows]
        before = [
            r["metrics"]["preemptions_total"]
            for r in rows
            if start is not None and r["unix_s"] <= start
        ]
        counters[pool] = dict(
            total=max(values) if values else None,
            before_start=max(before) if before else None,
        )
    covered = (
        len(starts) == len(releases) == len(decisions) == 1
        and decisions[0]["decision"]["capacity_path"]
        and remaining >= minimum_remaining
        and counters["tp1"]["before_start"] == 0
        and sum(m["migrated"] for m in metrics) == 1
        and all(m["arrival_valid"] for m in metrics)
    )
    return dict(
        coverage="STAY_CONTROL"
        if arm == "stay"
        else "KV_RELEASE_OBSERVED"
        if covered
        else "COVERAGE_GAP",
        remaining_at_start=remaining,
        start_to_source_release_s=(
            releases[0]["receipt"]["released_unix_ns"] / 1e9 - start
            if starts and releases
            else None
        ),
        source_reading_at_decision=decisions[0]["source_reading"]
        if decisions
        else None,
        selected_candidate=selected,
        preemptions=counters,
        actual_released_blocks=None,
        note="receipt proves free-return; net free change is not released block count",
    )
