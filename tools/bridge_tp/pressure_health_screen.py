# SPDX-License-Identifier: Apache-2.0
"""Small service-path function screen; no forced START or net-benefit claim."""

from tools.bridge_tp.pressure_kv_pilot import RELEASE_BUDGET_S
from tools.bridge_tp.pressure_sustained import METRIC


def arrivals(kind):
    count = 1 if kind == "spare" else 24
    rows = [
        dict(
            id=f"{kind}-target-background-{i}",
            at_s=i * 0.02,
            route="tp4",
            prompt_tokens=2048,
            max_tokens=6144 if kind == "spare" else 4096,
        )
        for i in range(count)
    ]
    rows += [
        dict(
            id=f"{kind}-source-background-{i}",
            at_s=5 + i * 0.02,
            route="tp1",
            prompt_tokens=2048,
            max_tokens=4096,
        )
        for i in range(3)
    ]
    rows += [
        dict(
            id=f"{kind}-source-prefill-{i}",
            at_s=20 + i * 2,
            route="tp1",
            prompt_tokens=2048,
            max_tokens=1,
        )
        for i in range(30)
    ]
    return sorted(rows, key=lambda r: r["at_s"])


def validate(config, inputs):
    from tools.bridge_tp.run_pressure_comparison import arrivals as build

    if (
        config["schema"] != "PRESSURE_HEALTH_SCREEN_V1"
        or config["policy_overrides"].get("health_service") is not True
        or config["policy_overrides"].get("sustain_s") != 10
        or config["policy_overrides"].get("health_min_efficiency") != 0.9
        or config["policy_overrides"].get("health_target_load") != 0.8
        or config["policy_overrides"].get("capacity_only") is not False
        or config["policy_overrides"].get("capacity_lookahead_s") != RELEASE_BUDGET_S
        or config["arms"] != ["stay", "migrate"]
        or config["maximum_migrations"] != 1
        or config["horizon_s"] != 300
        or config["metric"] != METRIC
        or config["maximum_gap_s"] != 1
        or config["itl_s"] != 0.1
        or config["slow_fraction"] != 0.01
        or config["max_dispatch_lag_s"] != 0.5
        or config["minimum_remaining_at_start"] != 1024
        or len(config["recipes"]) != 2
        or {(r["kind"], r["repetition"]) for r in config["recipes"]}
        != {("spare", 1), ("busy", 2)}
    ):
        raise ValueError("health function screen differs from frozen plan")
    for recipe in config["recipes"]:
        if recipe["arrivals"] != arrivals(recipe["kind"]):
            raise ValueError("health screen arrivals differ")
        build(inputs, recipe)


def sustained_arrivals(repetition):
    """Continue arrivals through the cutoff, preserving the proven first phase."""
    rows = [
        dict(
            id=f"r{repetition}-target-background-{i}",
            at_s=i * 90,
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
            prompt_tokens=2048,
            max_tokens=4096,
        )
        for i in range(3)
    ]
    rows += [
        dict(
            id=f"r{repetition}-source-arrival-{i}",
            at_s=110 + i * 55,
            route="tp1",
            prompt_tokens=2048,
            max_tokens=4096,
        )
        for i in range(4)
    ]
    rows += [
        dict(
            id=f"r{repetition}-source-prefill-{i}",
            at_s=20 + i * 2,
            route="tp1",
            prompt_tokens=2048,
            max_tokens=1,
        )
        for i in range(140)
    ]
    return sorted(rows, key=lambda r: r["at_s"])


def validate_sustained(config, inputs):
    from tools.bridge_tp.run_pressure_comparison import arrivals as build

    if (
        config["schema"] != "PRESSURE_HEALTH_SUSTAINED_V1"
        or config["arms"] != ["stay", "migrate"]
        or config["maximum_migrations"] != 1
        or config["horizon_s"] != 300
        or config["metric"] != METRIC
        or config["maximum_gap_s"] != 1
        or config["itl_s"] != 0.1
        or config["slow_fraction"] != 0.01
        or config["max_dispatch_lag_s"] != 0.5
        or config["minimum_remaining_at_start"] != 1024
        or config["policy_overrides"].get("health_service") is not True
        or config["policy_overrides"].get("capacity_only") is not False
        or config["policy_overrides"].get("health_min_efficiency") != 0.9
        or config["policy_overrides"].get("health_target_load") != 0.8
        or config["policy_overrides"].get("sustain_s") != 10
        or config["policy_overrides"].get("capacity_lookahead_s") != RELEASE_BUDGET_S
        or [(r["name"], r["kind"], r["repetition"]) for r in config["recipes"]]
        != [(f"health-sustained-r{i}", "spare", i) for i in (1, 2)]
    ):
        raise ValueError("health sustained plan differs from frozen four-run protocol")
    for recipe in config["recipes"]:
        if recipe["arrivals"] != sustained_arrivals(recipe["repetition"]):
            raise ValueError("health sustained arrivals differ")
        jobs = build(inputs, recipe)
        if len(jobs) != 151 or jobs[-1]["at_s"] != 298:
            raise ValueError("require151 arrivals through298s, then full drain")


def assess(audit, metrics, kind, arm):
    starts = [r for r in audit if r.get("kind") == "pressure_actual_start"]
    releases = [
        r for r in audit if r.get("kind") == "pressure_source_release_confirmed"
    ]
    reserved = [r for r in audit if r.get("execution") == "RESERVED"]
    decisions = [r for r in audit if r.get("kind") == "pressure_decision"]
    busy = [
        r
        for r in decisions
        if r["source_reading"].get("health_status") == "SUPPORTED"
        and r["source_reading"]["health_eta"] < 0.9
        and r["target_reading"].get("health_status") == "SUPPORTED"
        and (
            r["target_reading"]["health_load"] > 0.8
            or r["target_reading"]["health_eta"] < 0.9
        )
    ]
    service_moves = [
        r
        for r in reserved
        if r["decision"].get("service_path") == "health"
        and not r["decision"]["capacity_path"]
    ]
    remaining = None
    if service_moves:
        decision = service_moves[0]["decision"]
        selected = next(
            (
                c["request"]
                for c in decision.get("candidates", [])
                if c["request"]["request_id"] == decision.get("request_id")
            ),
            None,
        )
        if selected:
            remaining = min(8192, selected["max_output_tokens"]) - selected["generated"]
    if len(starts) > 1 or (kind == "busy" and starts):
        raise RuntimeError("busy-target/one-move safety contract violated")
    if arm == "stay":
        coverage = "STAY_CONTROL"
    elif kind == "busy":
        coverage = (
            "BUSY_TARGET_BLOCK_OBSERVED" if busy and not starts else "COVERAGE_GAP"
        )
    else:
        coverage = (
            "SERVICE_MOVE_OBSERVED"
            if (
                len(starts) == len(releases) == len(service_moves) == 1
                and sum(m["migrated"] for m in metrics) == 1
                and remaining is not None
                and remaining >= 1024
            )
            else "COVERAGE_GAP"
        )
    return dict(
        coverage=coverage,
        service_moves=len(service_moves),
        busy_target_block_samples=len(busy),
        actual_starts=len(starts),
        remaining_at_start=remaining,
        note="function coverage only; no repeated-benefit or throughput claim",
    )
