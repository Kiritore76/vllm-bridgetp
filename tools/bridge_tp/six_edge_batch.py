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


def first_target_order(scene):
    """Require model validation before topology publication and formal warmup."""
    scene = Path(scene)
    events = [
        json.loads(s)
        for s in (scene / "conversion_events.jsonl").read_text().splitlines()
        if s.strip()
    ]
    deferred = [e for e in events if e["kind"] == "CHANNEL_TOPOLOGY_DEFERRED"]
    if len(deferred) != 1:
        raise ValueError("first target needs one deferred topology event")
    target = deferred[0]["instance"]
    verified = [
        e for e in events if e["kind"] == "VERIFIED" and e.get("instance") == target
    ]
    published = [
        e
        for e in events
        if e["kind"] == "CHANNEL_TOPOLOGY_PUBLISHED" and e.get("instance") == target
    ]
    routes = [e for e in events if e["kind"] == "ROUTE_PUBLISHED"]
    if not (
        len(verified) == len(published) == len(routes) == 1
        and deferred[0]["unix_s"]
        <= verified[0]["unix_s"]
        <= published[0]["unix_s"]
        <= routes[0]["unix_s"]
    ):
        raise ValueError("first target model/topology/route ordering differs")
    root = scene / "controller/migration" / target
    channels = read(root / "gpu_channel_plan.json")
    if len(channels) != 4 or len({c["generation"] for c in channels}) != 4:
        raise ValueError("first target requires four distinct source channels")
    receipts = six.pc.preconnect_receipts(root)
    for c in channels:
        names = [f"source-{c['source']}-{target}.json"] + [
            f"tp_rank_{r}-{c['source']}.json" for r in range(2)
        ]
        for name in names:
            v = receipts.get(name, {})
            if not (
                v.get("status") == "WARMUP_COMPLETE"
                and v.get("channel_generation") == c["generation"]
                and verified[0]["unix_s"]
                <= v.get("completed_unix_s", 0)
                <= routes[0]["unix_s"]
            ):
                raise ValueError(
                    "first target formal warmup missing, stale or after route: " + name
                )
    return dict(
        passed=True,
        target=target,
        formal_warmup_receipts=12,
        model_verified_unix_s=verified[0]["unix_s"],
        topology_published_unix_s=published[0]["unix_s"],
        route_unix_s=routes[0]["unix_s"],
    )


def assess_case(directory, edge, registered, *, require_first_target_order=False):
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
    try:
        result = read(directory / "trial/result.json")
    except (OSError, ValueError):
        result = {}
    try:
        launch = read(directory / "trial/owned_launch_receipt.json")
    except (OSError, ValueError):
        launch = {}
    if not isinstance(result, dict):
        result = {}
    not_started = (
        isinstance(launch, dict)
        and launch.get("schema") == "BRIDGETP_OWNED_LAUNCH_V1"
        and launch.get("state") == "NOT_STARTED"
        and result.get("status") in {"TECHNICAL_FAILURE", "INTERRUPTED"}
        and result.get("runs") == []
        and result.get("total_profile_attempts") == 0
        and not shutdowns
        and not list((directory / "trial").rglob("owned_processes.json"))
        and not list((directory / "trial").rglob("effective_argv.json"))
    )
    value = dict(
        edge=edge,
        passed=False,
        cleanup_confirmed=cleanup,
        launch_state=launch.get("state") if isinstance(launch, dict) else None,
        launch_not_started=not_started,
        continuation_safe=cleanup or not_started,
        runner_error=result.get("error"),
        errors=[],
    )
    try:
        if edge not in EDGES:
            raise ValueError("unexpected edge")
        if int((directory / "runner_exit_code.txt").read_text()) != 0:
            raise ValueError(result.get("error") or "runner did not exit zero")
        if not cleanup:
            raise ValueError("owned cleanup not confirmed")
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
        if require_first_target_order:
            value["first_target_order"] = first_target_order(scene)
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


def aggregate(directory, edges=EDGES):
    if not edges or len(set(edges)) != len(edges) or not set(edges) <= set(EDGES):
        raise ValueError("invalid requested edge set")
    cases = {}
    for edge in edges:
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
        expected_edges=list(edges),
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
    parser.add_argument("--edges", nargs="+", choices=EDGES, default=list(EDGES))
    parser.add_argument("--require-first-target-order", action="store_true")
    args = parser.parse_args()
    if args.action == "plan":
        value, name = plan(read(args.inputs)), "batch_plan.json"
        value["edges"] = list(args.edges)
        if len(args.edges) != len(EDGES):
            value["execution"] = (
                f"{len(args.edges)} independent trial(s), serial; no automatic retries"
            )
    elif args.action == "assess":
        value = assess_case(
            args.directory,
            args.edge,
            read(args.directory.parent / "batch_plan.json"),
            require_first_target_order=args.require_first_target_order,
        )
        name = "case_acceptance.json"
    else:
        value, name = aggregate(args.directory, args.edges), "batch_acceptance.json"
    (args.directory / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))
    return 0 if value.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
