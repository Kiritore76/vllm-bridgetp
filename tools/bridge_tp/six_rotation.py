# SPDX-License-Identifier: Apache-2.0
"""Registered six-card rotation v2 and first static baseline evidence."""

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.bridge_tp import qwen3_phase_shift as phase
from tools.bridge_tp import qwen3_stay_migrate as matched

SCENARIO = "mixed_rotation_six_v2"
INPUT_SHA256 = "128e6a69f8f9b30d6b38edc7a4b2533d0b22331ab5bc01decaef27a5b1c6e1ec"
REQUESTS = 36371
OUTPUT_TOKENS = 5574784
HORIZON_S = 3900
TIMEOUT_S = 5400

SIMPLE_SCENARIO = "mixed_simple_six_v1"
RETURN_SCENARIO = "mixed_return_six_v1"
GUARD_SCENARIO = "mixed_guard_six_v1"


def registration(scenario):
    if scenario == GUARD_SCENARIO:
        return dict(
            input_sha256="96a46bc77137643862dfde118c323b4adb6903821a34c4e26579c826c04cde16",
            requests=16385,
            output_tokens=2589184,
            horizon_s=2400,
            controller_timeout_s=5400,
            max_conversions=8,
        )
    if scenario == SCENARIO:
        return dict(
            input_sha256=INPUT_SHA256,
            requests=REQUESTS,
            output_tokens=OUTPUT_TOKENS,
            horizon_s=HORIZON_S,
            controller_timeout_s=TIMEOUT_S,
            max_conversions=12,
        )
    if scenario == SIMPLE_SCENARIO:
        return dict(
            input_sha256="db268e9bba9c906a67e96fce644786550020d6568fd819a01425e537abae3387",
            requests=30731,
            output_tokens=4748032,
            horizon_s=3420,
            controller_timeout_s=5400,
            max_conversions=8,
        )
    if scenario == RETURN_SCENARIO:
        return dict(
            input_sha256="e181e33bde6e3d63c328392cc2dfd6732d415da3f38d102198ac0b369535060c",
            requests=11049,
            output_tokens=1704576,
            horizon_s=1860,
            controller_timeout_s=5400,
            max_conversions=8,
        )
    raise ValueError("unregistered six-card comparison scenario")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate_recipe(config):
    """Bind actual token IDs, budgets, phases and arrivals before any launch."""
    recipes = config["recipes"]
    if len(recipes) != 1:
        raise ValueError("six-card comparison needs one registered recipe")
    recipe = recipes[0]
    scenario = recipe["scenario"]
    registered = registration(scenario)
    if (
        recipe["scenario"] != scenario
        or recipe["arrival_seed"] != 6
        or recipe["phases"] != [list(p) for p in phase.SCENARIOS[scenario]["phases"]]
        or config["horizon_s"] != registered["horizon_s"]
        or recipe["input_sha256"] != registered["input_sha256"]
        or matched.digest(recipe["jobs"]) != registered["input_sha256"]
        or len(recipe["jobs"]) != registered["requests"]
        or recipe["planned_output_tokens"] != registered["output_tokens"]
    ):
        raise ValueError("six-card comparison input/phase contract changed")
    return recipe


def plan(inputs, scenario=SCENARIO, arm="static_L1AC"):
    registered = registration(scenario)
    if arm not in {"static_L1AC", "auto_from_L1AC"}:
        raise ValueError("unregistered comparison arm")
    config = phase.config(inputs, scenario)
    recipe = validate_recipe(config)
    counts = Counter((j["workload_phase"], j["workload_role"]) for j in recipe["jobs"])
    return dict(
        schema="QWEN3_SIX_ROTATION_V2"
        if scenario == SCENARIO
        else "QWEN3_SIX_GUARD_V1"
        if scenario == GUARD_SCENARIO
        else "QWEN3_SIX_RETURN_V1"
        if scenario == RETURN_SCENARIO
        else "QWEN3_SIX_SIMPLE_V1",
        scenario=scenario,
        input_sha256=registered["input_sha256"],
        requests=registered["requests"],
        output_tokens=registered["output_tokens"],
        long_requests=sum(
            n for (_, role), n in counts.items() if role != "short_probe"
        ),
        horizon_s=registered["horizon_s"],
        controller_timeout_s=registered["controller_timeout_s"],
        seed=6,
        phases=[
            dict(
                start_s=a,
                end_s=b,
                short_rate=short_rate,
                long_rate=long_rate,
                name=name,
                short_requests=counts[name, "short_probe"],
                long_requests=counts[name, "long_input_output"],
            )
            for a, b, short_rate, long_rate, name in phase.SCENARIOS[scenario]["phases"]
        ],
        comparison_arms=["static_L1AC", "static_L2C", "static_L3", "auto_from_L1AC"],
        dispatched_arms=[arm],
        prestart=False,
        migration_coverage_required=False,
        scope=arm + "; no GPU result or performance claim",
    )


def configure_plan(current, *, recipe, unlimited=False, assist=False, prestart=False):
    """Fail closed on unfair service settings; allow honest zero migration."""
    scenario = recipe["scenario"]
    registered = registration(scenario)
    argv = current["controller_argv"]
    if (
        recipe["scenario"] != scenario
        or recipe["input_sha256"] != registered["input_sha256"]
        or current["scheduler"] != "unified"
        or current.get("expected_context_limits") != (32768, 32768)
        or int(argv[argv.index("--max-seconds") + 1])
        != registered["controller_timeout_s"]
        or int(argv[argv.index("--hold-unstarted") + 1]) != 8
        or current["tp2_max_num_seqs"] != 16
    ):
        raise ValueError("six-card comparison service contract changed")
    conversion = current.get("conversion")
    if conversion and (
        # CPU-side pre-start of the replacements is an explicit choice of the
        # run (user 11-10) and is recorded with the contract below.
        bool(conversion.get("prestart")) != prestart
        or bool(conversion.get("assist")) != (assist if unlimited else False)
        or conversion.get("evacuate")
        or conversion.get("max_conversions")
        != (0 if unlimited else registered["max_conversions"])
    ):
        raise ValueError("six-card comparison conversion budget/options differ")
    rows = current["instances"]["instances"]
    if (
        not conversion
        and sum(r["tp_degree"] == 2 for r in rows) == 2
        and current.get("receiver_stage") != 3
    ):
        raise ValueError("static L2C must enable both migration receivers")
    current["migration_coverage_required"] = False
    write(
        current["run_dir"].parent / "rotation_contract.json",
        dict(
            scenario=scenario,
            input_sha256=registered["input_sha256"],
            requests=registered["requests"],
            output_tokens=registered["output_tokens"],
            horizon_s=registered["horizon_s"],
            controller_timeout_s=registered["controller_timeout_s"],
            migration_coverage_required=False,
            automatic_conversion=bool(conversion),
            max_conversions=conversion.get("max_conversions") if conversion else None,
            assist=bool(conversion and conversion.get("assist")),
            prestart=bool(conversion and conversion.get("prestart")),
            instances=rows,
        ),
    )


def return_coverage(steps):
    """A completed A merge must be followed by a completed A split."""
    merged = False
    for step in steps:
        plan = step["plan"]
        if plan["state"] != "COMPLETED" or plan.get("bundle_id") != "A":
            continue
        if plan["target_bundle_mode"] == "TP1" and merged:
            return True
        if plan["target_bundle_mode"] == "TP2":
            merged = True
    return False


def pulse_coverage(records, origin):
    """Require a live L2C observer through the pulse and its recovery window."""
    rows = [r for r in records if 1500 <= r["evaluated_unix_s"] - origin < 1620]
    times = [r["evaluated_unix_s"] - origin for r in rows]
    observing = bool(rows) and all(
        r.get("policy") in ("SIX_SERVICE_BUDGET_V1", "SIX_GAP_RULES_V1")
        and r["signals"]["layout"] == "L2C"
        and r.get("decision") == "KEEP"
        and not r.get("proposal")
        and not r.get("prepared")
        and not r.get("tilt")
        and r.get("candidates")
        and all(c["budget"]["supported"] for c in r["candidates"].values())
        for r in rows
    )
    continuous = bool(times) and (
        times[0] <= 1505
        and times[-1] >= 1615
        and all(0 < b - a <= 5 for a, b in zip(times, times[1:]))
    )
    return dict(
        passed=bool(observing and continuous),
        samples=len(rows),
        start_s=times[0] if times else None,
        end_s=times[-1] if times else None,
        live_supported_l2c_keep=bool(observing),
        continuous_observation=bool(continuous),
        window_s=[1500, 1620],
    )


def guard_round_trip(steps, origin):
    """Cover the intended long merge and sustained return, not an earlier split."""
    merged = False
    for step in steps:
        plan = step["plan"]
        if plan["state"] != "COMPLETED" or plan.get("bundle_id") != "A":
            continue
        triggered = step["triggered_unix_s"] - origin
        completed = step["completed_unix_s"] - origin
        if (
            plan["target_bundle_mode"] == "TP2"
            and 240 <= triggered < 1200
            and completed < 1500
        ):
            merged = True
        if (
            merged
            and plan["target_bundle_mode"] == "TP1"
            and 1620 <= triggered < 2220
            and completed < 2400
        ):
            return True
    return False


def assess(directory, scenario=SCENARIO, *, automatic=False):
    """Audit the first static L1AC run without treating SLO failures as success."""
    registered = registration(scenario)
    directory = Path(directory)
    value = dict(passed=False, performance_validated=False, errors=[])
    value["cleanup_confirmed"] = False
    try:
        shutdowns = list(directory.rglob("owned_shutdown_receipt.json"))
        value["cleanup_confirmed"] = (
            len(shutdowns) == 1
            and read(shutdowns[0]).get("all_owned_groups_ended") is True
        )
        result = read(directory / "result.json")
        expected_status = (
            "COMPLETED_CONVERSION_SEQUENCE" if automatic else "COMPLETED_STATIC_LAYOUT"
        )
        if result.get("status") != expected_status or (
            not automatic and result.get("layout") != "L1AC"
        ):
            raise ValueError(
                result.get("error") or "registered comparison did not complete"
            )
        contracts = list(directory.rglob("rotation_contract.json"))
        if len(contracts) != 1 or not value["cleanup_confirmed"]:
            raise ValueError("missing unique contract or owned shutdown")
        scene = contracts[0].parent
        contract = read(contracts[0])
        if (
            contract["input_sha256"] != registered["input_sha256"]
            or contract["automatic_conversion"] != automatic
        ):
            raise ValueError("input hash changed or first static arm converted")
        runs = result["runs"]
        if len(runs) != 1 or runs[0]["status"] != "COMPLETED":
            raise ValueError("expected one completed baseline")
        recipe = validate_recipe(read(directory / "comparison_recipe.json"))
        if runs[0]["input_sha256"] != recipe["input_sha256"]:
            raise ValueError("executed input hash changed")
        summary = read(scene / "comparison_metrics.json")["summary"]
        if (
            summary["requests"] != registered["requests"]
            or summary["output_tokens"] != registered["output_tokens"]
            or summary["window_s"] != registered["horizon_s"]
        ):
            raise ValueError("request/output/window or six-card mechanism differs")
        proof_name = (
            "conversion_sequence_acceptance.json"
            if automatic
            else "six_card_acceptance.json"
        )
        proof = read(scene / proof_name)
        if not proof.get("passed"):
            raise ValueError("mechanism evidence failed")
        empty = read(scene / "native_empty_receipt.json")
        expected = {r["instance_id"]: r for r in contract["instances"]}
        if automatic:
            control = read(scene / "controller/layout_control.json")
            expected = {
                r["instance_id"]: r
                for r in control["instances"]
                if r["lifecycle"] == "ACTIVE"
            }
            status = read(scene / "controller/layout_status.json")["instances"]
            if any(status.get(k, {}).get("lifecycle") != "ACTIVE" for k in expected):
                raise ValueError("final layout lacks controller ACK")
        if (
            not expected
            or set(empty) != set(expected)
            or any(
                r["running"] or r["waiting"] or r["requests"] for r in empty.values()
            )
        ):
            raise ValueError("native inventories did not empty")
        for name, row in expected.items():
            if (
                read(scene / (name + "_trace_finalization.json")).get("status")
                != "FINALIZED"
            ):
                raise ValueError("missing finalized current engine trace")
            trace = read(scene / (name + "_trace_acceptance.json"))
            if trace.get("samples", 0) <= 0 or trace.get("ranks") != row["tp_degree"]:
                raise ValueError("invalid current engine trace")
        export = read(scene / "controller/pressure_result_export_receipt.json")
        if (
            export.get("status") != "WRITTEN"
            or export.get("requests") != registered["requests"]
        ):
            raise ValueError("missing bounded result export proof")
        from tools.bridge_tp.pressure_cpu import load

        value["pause_timing"] = load("pause_diagnostics").verify(
            scene / "controller/pressure_timing.jsonl"
        )
        if automatic:
            sequence = read(scene / "conversion_sequence_result.json")
            steps = sequence.get("steps", [])
            limits = [
                json.loads(line)
                for line in (scene / "conversion_events.jsonl").read_text().splitlines()
                if line.strip()
            ]
            value["automatic_coverage"] = dict(
                attempts=len(steps),
                completed=sum(s["plan"]["state"] == "COMPLETED" for s in steps),
                safely_aborted=sum(s["plan"]["state"] == "ABORTED" for s in steps),
                conversion_observed=any(
                    s["plan"]["state"] == "COMPLETED" for s in steps
                ),
                budget_limit_reached=any(e["kind"] == "DECIDER_LIMIT" for e in limits),
                actual_migration_coverage=proof.get("actual_migration_coverage", False),
            )
        value.update(
            passed=True,
            arm="auto_from_L1AC" if automatic else "static_L1AC",
            input_sha256=registered["input_sha256"],
            summary=summary,
            fixed_arrivals_valid=summary["fixed_arrivals_valid"],
            scope="complete baseline and cleanup; SLO/CPU/dispatch review follows",
        )
        if automatic and scenario in {RETURN_SCENARIO, GUARD_SCENARIO}:
            value["mechanism_complete"] = True
            covered = return_coverage(steps)
            value["automatic_coverage"]["a_round_trip_completed"] = covered
            if not covered:
                value["passed"] = False
                value["errors"].append("A merge/split round trip not covered")
            if scenario == GUARD_SCENARIO:
                origin = read(scene / "controller/layout_status.json")[
                    "workload_origin_unix_s"
                ]
                timed = guard_round_trip(steps, origin)
                value["automatic_coverage"]["timed_a_round_trip_completed"] = timed
                if not timed:
                    value["passed"] = False
                    value["errors"].append("intended sustained round trip not covered")
                records = [
                    json.loads(line)
                    for line in (scene / "controller/conversion_decider.jsonl")
                    .read_text()
                    .splitlines()
                    if line.strip()
                ]
                value["pulse_coverage"] = pulse_coverage(records, origin)
                if not value["pulse_coverage"]["passed"]:
                    value["passed"] = False
                    value["errors"].append("short pulse KEEP observation not covered")
                if value["automatic_coverage"]["safely_aborted"]:
                    value["passed"] = False
                    value["errors"].append("guard scenario conversion aborted")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        value["errors"].append(repr(error))
    write(directory.parent / "rotation_acceptance.json", value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "assess"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument(
        "--arm", choices=("static_L1AC", "auto_from_L1AC"), default="static_L1AC"
    )
    parser.add_argument(
        "--scenario", choices=phase.SIX_COMPARISON_SCENARIOS, default=SCENARIO
    )
    args = parser.parse_args(argv)
    if args.action == "plan":
        value = plan(read(args.inputs), args.scenario, args.arm)
        write(args.directory / "rotation_plan.json", value)
    else:
        value = assess(
            args.directory, args.scenario, automatic=args.arm == "auto_from_L1AC"
        )
    print(json.dumps(value, indent=2))
    return 0 if args.action == "plan" or value["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
