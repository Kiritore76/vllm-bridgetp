# SPDX-License-Identifier: Apache-2.0
"""Registered independent remaining-edge checks; no forced migration coverage."""

import argparse
import json
from pathlib import Path

from tools.bridge_tp import qwen3_long_burst_pair as burst
from tools.bridge_tp import six_conversion as six
from tools.bridge_tp.pressure_cpu import load

EDGES = ("L0-L1AC", "L1AC-L0", "L2C-L3", "L3-L2C")
TRIGGER_S = 60


def read(path):
    return json.loads(Path(path).read_text())


def recipe(inputs):
    return next(
        r for r in burst.config(inputs)["recipes"] if r["scenario"] == "deep_long_burst"
    )


def plan(inputs):
    registered = recipe(inputs)
    return dict(
        schema="QWEN3_SIX_REMAINING_EDGES_V1",
        execution="four independent trials, serial; no automatic retries",
        edges=list(EDGES),
        trigger_s=TRIGGER_S,
        requests=len(registered["jobs"]),
        output_tokens=registered["planned_output_tokens"],
        input_sha256=registered["input_sha256"],
        horizon_s=300,
        long_arrivals_s=[90, 92, 94, 96, 98, 100],
        scope="conversion mechanism; migration observed separately; "
        "not loaded-drain, automatic-direction or performance acceptance",
    )


def assess_case(directory, edge, registered):
    """Persist failure evidence and independently check cleanup for continuation."""
    directory = Path(directory)
    shutdowns = list((directory / "trial").rglob("owned_shutdown_receipt.json"))
    try:
        cleanup = (
            len(shutdowns) == 1
            and read(shutdowns[0]).get("all_owned_groups_ended") is True
        )
    except (OSError, ValueError, TypeError, AttributeError):
        cleanup = False
    value = dict(edge=edge, passed=False, cleanup_confirmed=cleanup, errors=[])
    try:
        if edge not in EDGES:
            raise ValueError("unexpected edge")
        if not cleanup:
            raise ValueError("owned cleanup not confirmed")
        if int((directory / "runner_exit_code.txt").read_text()) != 0:
            raise ValueError("runner did not exit zero")
        root = directory / "trial"
        result = read(root / "result.json")
        if result.get("status") != "COMPLETED_CONVERSION" or result.get("edge") != edge:
            raise ValueError("runner result edge/status mismatch")
        paths = list(root.rglob("conversion_sequence_acceptance.json"))
        if len(paths) != 1:
            raise ValueError("need exactly one conversion acceptance")
        scene = paths[0].parent
        proof = read(paths[0])
        if proof != six.assess(scene, registered["requests"]):
            raise ValueError("offline mechanism acceptance differs")
        if not proof["passed"] or not proof["mechanism_passed"]:
            raise ValueError("conversion mechanism failed")
        sequence = read(scene / "conversion_sequence_result.json")
        steps = sequence["steps"]
        before, after, bundle, mode = six.EDGES[edge]
        if len(steps) != 1:
            raise ValueError("need exactly one conversion step")
        step = steps[0]
        if (step["request"]["bundle"], step["request"]["target_mode"]) != (
            bundle,
            mode,
        ):
            raise ValueError("wrong conversion bundle or mode")
        if step["layout"] != six.PATH[after] or proof.get("late_dispatches") != []:
            raise ValueError("wrong final layout or retired-instance dispatch")
        runs = result["runs"]
        if len(runs) != 1 or runs[0]["input_sha256"] != registered["input_sha256"]:
            raise ValueError("input plan mismatch")
        summary = read(scene / "comparison_metrics.json")["summary"]
        if not (
            summary["requests"] == registered["requests"]
            and summary["output_tokens"] == registered["output_tokens"]
            and summary["fixed_arrivals_valid"]
            and summary["load_coverage_pass"]
        ):
            raise ValueError("incomplete workload or invalid arrivals")
        requests = read(scene / "controller/pressure_request_results.json")
        if (
            len(requests) != registered["requests"]
            or len({r["job_id"] for r in requests}) != len(requests)
            or not all(r["source_done"] and r["source_error"] is None for r in requests)
        ):
            raise ValueError("request ledger incomplete or errored")
        empty = read(scene / "native_empty_receipt.json")
        if not empty or not all(
            r["running"] == r["waiting"] == 0
            and not r["requests"]
            and r["writer_error"] is None
            for r in empty.values()
        ):
            raise ValueError("native inventory not empty")
        audit = [
            json.loads(s)
            for s in (scene / "controller/pressure_audit.jsonl")
            .read_text()
            .splitlines()
            if s.strip()
        ]
        cfg = read(root / "comparison_policy.json")
        cancellation = load("pressure_cancel").assess_unified_cancellations(
            audit, cfg["max_age_s"]
        )
        if not cancellation["passed"]:
            raise ValueError("unified recovery cancellation evidence failed")
        value.update(
            passed=True,
            initial_layout=before,
            final_layout=step["layout"],
            mechanism_passed=True,
            actual_migration_coverage=proof["actual_migration_coverage"],
            actual_committed_transfers=proof["actual_committed_transfers"],
            migration_coverage_required=False,
            loaded_drain_coverage=proof["loaded_drain_coverage"],
            phases=proof["phases"],
            summary=summary,
            cancellation_acceptance=cancellation,
        )
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        value["errors"].append(f"{type(error).__name__}: {error}")
    return value


def aggregate(directory):
    cases = {}
    for edge in EDGES:
        path = Path(directory) / edge / "case_acceptance.json"
        try:
            cases[edge] = read(path)
        except (OSError, ValueError):
            cases[edge] = dict(
                edge=edge,
                passed=False,
                errors=["not attempted or acceptance unreadable"],
            )
    return dict(
        passed=all(
            isinstance(v, dict)
            and v.get("passed") is True
            and v.get("cleanup_confirmed") is True
            and v.get("edge") == k
            for k, v in cases.items()
        ),
        expected_edges=list(EDGES),
        cases=cases,
        scope="independent serial mechanism checks; inspect actual migration "
        "coverage and service costs per edge; no all-edge performance claim",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "assess", "aggregate"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--edge", choices=EDGES)
    args = parser.parse_args()
    if args.action == "plan":
        value, name = plan(read(args.inputs)), "batch_plan.json"
    elif args.action == "assess":
        value = assess_case(
            args.directory, args.edge, read(args.directory.parent / "batch_plan.json")
        )
        name = "case_acceptance.json"
    else:
        value, name = aggregate(args.directory), "batch_acceptance.json"
    (args.directory / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))
    return 0 if value.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
