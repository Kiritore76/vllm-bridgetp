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
        or config["policy_overrides"].get("sustain_s") != 3
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
        config["schema"] != "PRESSURE_HEALTH_SUSTAINED_V2"
        or config["arms"] != ["stay", "migrate"]
        or config["maximum_migrations"] != 0
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
        or config["policy_overrides"].get("sustain_s") != 3
        or config["policy_overrides"].get("capacity_lookahead_s") != RELEASE_BUDGET_S
        or [(r["name"], r["kind"], r["repetition"]) for r in config["recipes"]]
        != (
            [(f"health-sustained-r{i}", "spare", i) for i in (1, 2)]
            + [
                ("health-light-r1", "light", 1),
                ("health-busy-r1", "busy", 1),
                ("health-kv-r2", "kv", 2),
            ]
        )
    ):
        raise ValueError("health repeated migration matrix differs from frozen plan")
    for recipe in config["recipes"]:
        expected = (
            sustained_arrivals(recipe["repetition"])
            if recipe["kind"] == "spare"
            else matrix_arrivals(recipe["kind"])
        )
        if recipe["arrivals"] != expected:
            raise ValueError("health sustained arrivals differ")
        jobs = build(inputs, recipe)
        if recipe["kind"] == "spare" and (len(jobs) != 151 or jobs[-1]["at_s"] != 298):
            raise ValueError("require151 arrivals through298s, then full drain")


def select_kv_test(config, *, cancel_smoke=False):
    """Select from the already validated full matrix without mutating its inputs."""
    import copy

    selected = copy.deepcopy(config)
    selected["recipes"] = [r for r in selected["recipes"] if r["kind"] == "kv"]
    if len(selected["recipes"]) != 1:
        raise ValueError("require the single frozen two-wave KV recipe")
    selected["selection"] = "cancel-smoke" if cancel_smoke else "health-kv-only"
    selected["test_only_cancel_smoke"] = cancel_smoke
    if cancel_smoke:
        selected["arms"] = ["migrate"]
    return selected


def inject_cancel_smoke(sequence, inventory, now, audit):
    """Exercise cleanup, never force START or change the online pressure policy."""
    import json

    s = sequence.sessions[0]
    if (
        len(sequence.sessions) != 1
        or s.record is None
        or s.settled
        or s.cancel_future is not None
        or s.target_future is None
        or s.target_future.done()
        or not s.reversible()
        or not 0 <= now - inventory["sampled_at"] <= 2
    ):
        return False
    prefix = f"cmpl-bridgetp-phase9-target-{s.run_dir.name}-0-"
    targets = [r for r in inventory["requests"] if r["request_id"].startswith(prefix)]
    if len(targets) != 1 or targets[0]["generated"] != 0:
        return False
    for rank in range(4):
        path = s.run_dir / "initial_stage_receipts" / f"tp_rank_{rank}.json"
        if not path.is_file():
            return False
        row = json.loads(path.read_text())
        if row.get("migration_id") != s.migration_id or row.get("status") != "STAGED":
            raise ValueError("cancel smoke stage receipt identity differs")
    if not s.reversible():
        return False  # Source may have selected its boundary during the reads.
    audit.write(
        dict(
            kind="pressure_test_cancel_injected",
            migration_id=s.migration_id,
            request_id=s.record.request_id,
            test_only=True,
            natural_cancellation_validated=False,
            target_request_id=targets[0]["request_id"],
        )
    )
    s.cancel("TEST_ONLY admitted pre-freeze shadow cleanup")
    return True


def assess_cancel_smoke(audit, run_dir):
    """Require confirmed cancellation AND later actual migration on the same run."""
    import json

    injected = [v for v in audit if v.get("kind") == "pressure_test_cancel_injected"]
    cancelled = [v for v in audit if v.get("kind") == "pressure_cancel_confirmed"]
    releases = [
        v for v in audit if v.get("kind") == "pressure_source_release_confirmed"
    ]
    observed = False
    if len(injected) == len(cancelled) == 1:
        first, done = injected[0], cancelled[0]
        if first["migration_id"] != done["migration_id"]:
            raise ValueError("test cancellation identity differs")
        native = json.loads(
            (run_dir / "sessions/migration-0001/target_abort_receipt.json").read_text()
        )
        if native.get("target_request_id") != first["target_request_id"]:
            raise ValueError("test target abort identity differs")
        later = any(
            v["unix_s"] > done["unix_s"] and v["migration_id"] != done["migration_id"]
            for v in releases
        )
        reused = any(
            v.get("kind") == "pressure_channel_reused"
            and v.get("previous_migration_id") == first["migration_id"]
            and v.get("previous_outcome") == "CANCELLED"
            for v in audit
        )
        observed = later and reused
    return dict(
        coverage="CANCEL_AND_REUSE_OBSERVED" if observed else "COVERAGE_GAP",
        injected=len(injected),
        confirmed_cancelled=len(cancelled),
        later_releases=len(releases),
        test_only=True,
        natural_cancellation_validated=False,
        benefit_comparison=False,
    )


def matrix_arrivals(kind):
    if kind == "busy":
        return arrivals("busy")
    if kind == "light":
        return [
            dict(
                id="light-target",
                at_s=0,
                route="tp4",
                prompt_tokens=2048,
                max_tokens=6144,
            ),
            dict(
                id="light-source",
                at_s=5,
                route="tp1",
                prompt_tokens=2048,
                max_tokens=4096,
            ),
        ]
    if kind == "kv":
        rows = [
            dict(
                id="kv-target", at_s=0, route="tp4", prompt_tokens=2048, max_tokens=6144
            )
        ]
        rows += [
            dict(
                id=f"kv-wave-{wave}-source-{i}",
                at_s=start + i * 0.02,
                route="tp1",
                prompt_tokens=4096,
                max_tokens=3072,
            )
            for wave, start in enumerate((5, 170))
            for i in range(5)
        ]
        return rows
    raise ValueError("unknown migration matrix kind")


def assess_repeated(audit, metrics, kind, arm):
    starts = [r for r in audit if r.get("kind") == "pressure_actual_start"]
    releases = [
        r for r in audit if r.get("kind") == "pressure_source_release_confirmed"
    ]
    reserved = [r for r in audit if r.get("execution") == "RESERVED"]
    paths = [r["decision"]["capacity_path"] for r in reserved]
    if arm == "stay" and (starts or reserved):
        raise RuntimeError("STAY control started migration")
    # KV relief may legitimately bypass the health busy-target gate; only a
    # service-path move violates this scenario's busy-target safety contract.
    if kind == "busy" and any(not value for value in paths):
        raise RuntimeError("health service migration into busy target")
    if kind == "light" and starts:
        raise RuntimeError("light source unexpectedly migrated")
    if len({r["migration_id"] for r in starts}) != len(starts):
        raise RuntimeError("duplicate migration identity")
    service = sum(not value for value in paths)
    kv = sum(paths)
    busy = sum(
        r.get("kind") == "pressure_decision"
        and r.get("source_reading", {}).get("health_status") == "SUPPORTED"
        and r["source_reading"]["health_eta"] < 0.9
        and r.get("target_reading", {}).get("health_status") == "SUPPORTED"
        and (
            r["target_reading"]["health_load"] > 0.8
            or r["target_reading"]["health_eta"] < 0.9
        )
        for r in audit
    )
    complete = len(starts) == len(releases) == sum(m["migrated"] for m in metrics)
    release_ids = {r["migration_id"] for r in releases}
    if not release_ids <= {r["migration_id"] for r in starts} or (
        len(release_ids) != len(releases)
    ):
        raise RuntimeError("migration START/release identity sets differ")
    covered = complete and (
        (kind == "spare" and service >= 2)
        or (kind == "kv" and kv >= 2)
        or (kind == "busy" and busy > 0 and not starts)
        or (kind == "light" and not starts)
    )
    return dict(
        coverage="STAY_CONTROL"
        if arm == "stay"
        else "REPEATED_SCENARIO_OBSERVED"
        if covered
        else "COVERAGE_GAP",
        actual_starts=len(starts),
        source_releases=len(releases),
        service_moves=service,
        kv_moves=kv,
        selected_requests=[
            dict(
                migration_path="kv" if row["decision"]["capacity_path"] else "health",
                request_id=row["decision"]["request_id"],
                candidates=row["decision"].get("candidates", []),
            )
            for row in reserved
        ],
        busy_target_block_samples=busy,
        note="missing repeated START is a coverage gap, never force a migration",
    )


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
        if r["decision"].get("service_path") in {"health", "workload"}
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
