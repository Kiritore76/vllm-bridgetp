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


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate_recipe(config):
    """Bind actual token IDs, budgets, phases and arrivals before any launch."""
    recipes = config["recipes"]
    if len(recipes) != 1:
        raise ValueError("rotation v2 needs one registered recipe")
    recipe = recipes[0]
    if (
        recipe["scenario"] != SCENARIO
        or recipe["arrival_seed"] != 6
        or recipe["phases"] != [list(p) for p in phase.SCENARIOS[SCENARIO]["phases"]]
        or config["horizon_s"] != HORIZON_S
        or recipe["input_sha256"] != INPUT_SHA256
        or matched.digest(recipe["jobs"]) != INPUT_SHA256
        or len(recipe["jobs"]) != REQUESTS
        or recipe["planned_output_tokens"] != OUTPUT_TOKENS
    ):
        raise ValueError("rotation v2 input/phase contract changed")
    return recipe


def plan(inputs):
    config = phase.config(inputs, SCENARIO)
    recipe = validate_recipe(config)
    counts = Counter((j["workload_phase"], j["workload_role"]) for j in recipe["jobs"])
    return dict(
        schema="QWEN3_SIX_ROTATION_V2",
        scenario=SCENARIO,
        input_sha256=INPUT_SHA256,
        requests=REQUESTS,
        output_tokens=OUTPUT_TOKENS,
        long_requests=sum(
            n for (_, role), n in counts.items() if role != "short_probe"
        ),
        horizon_s=HORIZON_S,
        controller_timeout_s=TIMEOUT_S,
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
            for a, b, short_rate, long_rate, name in phase.SCENARIOS[SCENARIO]["phases"]
        ],
        comparison_arms=["static_L1AC", "static_L2C", "static_L3", "auto_from_L1AC"],
        dispatched_arms=["static_L1AC"],
        prestart=False,
        migration_coverage_required=False,
        scope="first static baseline; no GPU result or performance claim",
    )


def configure_plan(current, *, recipe):
    """Fail closed on unfair service settings; allow honest zero migration."""
    argv = current["controller_argv"]
    if (
        recipe["scenario"] != SCENARIO
        or recipe["input_sha256"] != INPUT_SHA256
        or current["scheduler"] != "unified"
        or current.get("expected_context_limits") != (32768, 32768)
        or int(argv[argv.index("--max-seconds") + 1]) != TIMEOUT_S
        or int(argv[argv.index("--hold-unstarted") + 1]) != 8
        or current["tp2_max_num_seqs"] != 16
    ):
        raise ValueError("rotation v2 service contract changed")
    conversion = current.get("conversion")
    if conversion and (
        conversion.get("prestart")
        or conversion.get("assist")
        or conversion.get("evacuate")
        or conversion.get("max_conversions") != 12
    ):
        raise ValueError("rotation v2 conversion budget/options differ")
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
            scenario=SCENARIO,
            input_sha256=INPUT_SHA256,
            requests=REQUESTS,
            output_tokens=OUTPUT_TOKENS,
            horizon_s=HORIZON_S,
            controller_timeout_s=TIMEOUT_S,
            migration_coverage_required=False,
            automatic_conversion=bool(conversion),
            instances=rows,
        ),
    )


def assess(directory):
    """Audit the first static L1AC run without treating SLO failures as success."""
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
        if (
            result.get("status") != "COMPLETED_STATIC_LAYOUT"
            or result.get("layout") != "L1AC"
        ):
            raise ValueError(result.get("error") or "first arm did not complete L1AC")
        contracts = list(directory.rglob("rotation_contract.json"))
        if len(contracts) != 1 or not value["cleanup_confirmed"]:
            raise ValueError("missing unique contract or owned shutdown")
        scene = contracts[0].parent
        contract = read(contracts[0])
        if contract["input_sha256"] != INPUT_SHA256 or contract["automatic_conversion"]:
            raise ValueError("input hash changed or first static arm converted")
        runs = result["runs"]
        if len(runs) != 1 or runs[0]["status"] != "COMPLETED":
            raise ValueError("expected one completed baseline")
        recipe = validate_recipe(read(directory / "comparison_recipe.json"))
        if runs[0]["input_sha256"] != recipe["input_sha256"]:
            raise ValueError("executed input hash changed")
        summary = read(scene / "comparison_metrics.json")["summary"]
        if (
            summary["requests"] != REQUESTS
            or summary["output_tokens"] != OUTPUT_TOKENS
            or summary["window_s"] != HORIZON_S
            or not read(scene / "six_card_acceptance.json")["passed"]
        ):
            raise ValueError("request/output/window or six-card mechanism differs")
        empty = read(scene / "native_empty_receipt.json")
        if len(empty) != 5 or any(
            r["running"] or r["waiting"] or r["requests"] for r in empty.values()
        ):
            raise ValueError("native inventories did not empty")
        value.update(
            passed=True,
            arm="static_L1AC",
            input_sha256=INPUT_SHA256,
            summary=summary,
            fixed_arrivals_valid=summary["fixed_arrivals_valid"],
            scope="complete baseline and cleanup; SLO/CPU/dispatch review follows",
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        value["errors"].append(repr(error))
    write(directory.parent / "rotation_acceptance.json", value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "assess"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args(argv)
    if args.action == "plan":
        value = plan(read(args.inputs))
        write(args.directory / "rotation_plan.json", value)
    else:
        value = assess(args.directory)
    print(json.dumps(value, indent=2))
    return 0 if args.action == "plan" or value["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
