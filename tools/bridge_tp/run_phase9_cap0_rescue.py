#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one non-counted CAP-0 Rescue reachability bring-up."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402
from tools.bridge_tp import run_phase9_cap0_noop as scenario_runner  # noqa: E402
from tools.bridge_tp.run_phase9_capacity_background import (  # noqa: E402
    load_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--survival-table", type=Path, required=True)
    parser.add_argument("--guard-file", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-survival-sha256", required=True)
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--expected-guard", type=int, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--tp1-blocks", type=int, required=True)
    parser.add_argument("--tp4-blocks", type=int, required=True)
    parser.add_argument("--anchor-max-tokens", type=int, default=8000)
    parser.add_argument("--tp1-gpu", default="0")
    parser.add_argument("--tp4-gpus", default="1,2,3,4")
    parser.add_argument("--tp1-port", type=int, default=8001)
    parser.add_argument("--tp4-port", type=int, default=8200)
    parser.add_argument("--snapshot-port", type=int, default=29800)
    parser.add_argument("--delta-port", type=int, default=29900)
    parser.add_argument("--delivery-port", type=int, default=30000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--server-start-timeout-s", type=float, default=900)
    parser.add_argument("--run-timeout-s", type=float, default=1800)
    parser.add_argument("--stager-timeout-s", type=float, default=1800)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def prompt_tokens(request: dict[str, Any]) -> int | None:
    prompt = request.get("prompt")
    if isinstance(prompt, list) and all(isinstance(item, int) for item in prompt):
        return len(prompt)
    return None


def validate_rescue_pressure(
    manifest: dict[str, Any],
    *,
    anchor_max_tokens: int,
    tp1_total_tokens: int,
    tp4_total_tokens: int,
    max_model_len: int,
    max_target_waiting: int = 4,
    expected_scenario: str = "CAP-0 Rescue reachability bring-up",
) -> dict[str, Any]:
    if manifest.get("scenario") != expected_scenario:
        raise ValueError(f"manifest scenario differs from {expected_scenario!r}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Rescue manifest requires jobs")
    source = [job for job in jobs if job.get("pool") == "source"]
    target = [job for job in jobs if job.get("pool") == "target"]
    if len(source) < 4:
        raise ValueError("Rescue requires at least four source pressure jobs")
    if len(target) <= max_target_waiting + 1:
        raise ValueError("too few target jobs to exceed the waiting guard")

    source_demand = anchor_max_tokens + sum(
        int(job["request"]["max_tokens"]) for job in source
    )
    if source_demand < int(tp1_total_tokens * 1.10):
        raise ValueError("source demand is below the 110% TP1 pressure floor")
    target_starts = [float(job["start_after_s"]) for job in target]
    source_starts = [float(job["start_after_s"]) for job in source]
    if min(target_starts) > min(source_starts):
        raise ValueError("target pressure must start before source pressure")
    if max(target_starts) - min(target_starts) > 1.0:
        raise ValueError("target arrivals must form a burst within one second")

    target_demand = 0
    for job in target:
        request = job["request"]
        exact_prompt = prompt_tokens(request)
        if exact_prompt is None:
            raise ValueError(
                f"target job {job['job_id']} must use exact prompt token IDs"
            )
        context = exact_prompt + int(request["max_tokens"])
        if context > max_model_len:
            raise ValueError(f"target job {job['job_id']} exceeds max model length")
        target_demand += context
    if target_demand >= tp4_total_tokens:
        raise ValueError("Rescue target demand must fit after the finite burst drains")
    if target_demand < int(tp4_total_tokens * 0.50):
        raise ValueError("Rescue target demand is below the 50% contention floor")
    return {
        "source_jobs": len(source),
        "target_jobs": len(target),
        "source_output_demand_tokens": source_demand,
        "target_context_demand_tokens": target_demand,
        "tp1_total_tokens": tp1_total_tokens,
        "tp4_total_tokens": tp4_total_tokens,
        "target_to_capacity_frac": target_demand / tp4_total_tokens,
    }


def load_audit(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def receipt_evidence(controller_dir: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    required = {
        "session": controller_dir / "session_manifest.json",
        "staging": controller_dir / "staging_manifest.json",
        "takeover": controller_dir / "takeover_state.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        errors.append(f"missing migration evidence: {', '.join(missing)}")
        return {"receiver_ranks": [], "exact_readback": []}, errors
    session = common.read_json(required["session"])
    staging = common.read_json(required["staging"])
    takeover = common.read_json(required["takeover"])
    migration_id = str(session.get("migration_id"))
    source_request_id = str(session.get("source_request_id"))
    sender_dir = controller_dir / "stage_delivery_receipts"
    receiver_root = controller_dir / "receiver_receipts"
    target_dirs = (
        sorted(path for path in receiver_root.iterdir() if path.is_dir())
        if receiver_root.is_dir()
        else []
    )
    if len(target_dirs) != 1:
        errors.append(f"expected one receiver target directory, got {len(target_dirs)}")
        return {"receiver_ranks": [], "exact_readback": []}, errors

    target_request_id = target_dirs[0].name
    receiver_ranks: list[int] = []
    readbacks: list[bool] = []
    for rank in range(4):
        sender_path = sender_dir / f"tp_rank_{rank}.json"
        receiver_path = target_dirs[0] / f"tp_rank_{rank}.json"
        if not sender_path.is_file() or not receiver_path.is_file():
            errors.append(f"rank {rank} sender/receiver receipt is missing")
            continue
        sender = common.read_json(sender_path)
        receiver = common.read_json(receiver_path)
        if sender.get("status") != "READY":
            errors.append(f"rank {rank} sender is not READY")
        if receiver.get("status") != "OWNERSHIP_COMMITTED":
            errors.append(f"rank {rank} receiver ownership is not committed")
        if sender.get("migration_id") != migration_id:
            errors.append(f"rank {rank} sender migration ID differs")
        if receiver.get("migration_id") != migration_id:
            errors.append(f"rank {rank} receiver migration ID differs")
        if receiver.get("source_request_id") != source_request_id:
            errors.append(f"rank {rank} receiver source request differs")
        if sender.get("target_request_id") != target_request_id:
            errors.append(f"rank {rank} sender target request differs")
        if receiver.get("target_request_id") != target_request_id:
            errors.append(f"rank {rank} receiver target request differs")
        if sender.get("payload_sha256") != receiver.get("payload_sha256"):
            errors.append(f"rank {rank} payload digest differs")
        if int(sender.get("payload_bytes", -1)) != int(
            receiver.get("payload_bytes", -2)
        ):
            errors.append(f"rank {rank} payload byte count differs")
        exact = receiver.get("exact_readback") is True
        if not exact:
            errors.append(f"rank {rank} exact readback failed")
        else:
            receiver_ranks.append(rank)
        readbacks.append(exact)

    if staging.get("migration_id") != migration_id:
        errors.append("staging migration ID differs from the session")
    if staging.get("source_request_id") != source_request_id:
        errors.append("staging source request differs from the session")
    if takeover.get("migration_id") != migration_id:
        errors.append("takeover migration ID differs from the session")
    if takeover.get("source_request_id") != source_request_id:
        errors.append("takeover source request differs from the session")
    if takeover.get("target_request_id") != target_request_id:
        errors.append("takeover target request differs from receiver receipts")
    return {
        "migration_id": migration_id,
        "source_request_id": source_request_id,
        "target_request_id": target_request_id,
        "receiver_ranks": receiver_ranks,
        "exact_readback": readbacks,
    }, errors


def accept_rescue(
    controller_dir: Path,
    background_dir: Path,
    expected_jobs: int,
    expected_anchor_tokens: int,
    max_target_kv_usage_frac: float = 0.85,
    max_target_waiting: int = 4,
) -> dict[str, Any]:
    background = common.read_json(background_dir / "background_summary.json")
    rows = load_audit(controller_dir / "phase9_audit.jsonl")
    capacity = [
        row for row in rows if row.get("kind") == "capacity_pilot_decision"
    ]
    blocked = [
        (index, row)
        for index, row in enumerate(capacity)
        if row.get("signal", {}).get("active")
        and row.get("action") == "STAY"
        and (
            float(row.get("target_kv_usage_frac", 0.0))
            > max_target_kv_usage_frac
            or int(row.get("target_waiting", 0)) > max_target_waiting
        )
    ]
    starts = [
        (index, row)
        for index, row in enumerate(capacity)
        if row.get("action") == "START_SHADOW"
    ]
    transitions = [row for row in rows if row.get("kind") == "transition"]
    transition_states = [str(row.get("to")) for row in transitions]
    ends = [row for row in rows if row.get("kind") == "run_end"]
    commits = [row for row in rows if row.get("kind") == "commit"]
    fatal_kinds = {
        "abandon",
        "action_error",
        "commit_refused",
        "invariant_violation",
        "rollback_failed",
    }
    fatal = [row for row in rows if row.get("kind") in fatal_kinds]
    proxy = common.read_json(controller_dir / "response_proxy_stats.json")
    takeover = common.read_json(controller_dir / "takeover_state.json")
    receipts, receipt_errors = receipt_evidence(controller_dir)

    errors: list[str] = []
    if background.get("jobs") != expected_jobs:
        errors.append("background job count differs from the manifest")
    if background.get("completed") != expected_jobs or background.get("failed") != 0:
        errors.append("background workload did not complete cleanly")
    if not blocked:
        errors.append("target was not initially blocked while source signal was active")
    if len(starts) != 1:
        errors.append(f"expected one START_SHADOW decision, got {len(starts)}")
    if blocked and starts and blocked[0][0] >= starts[0][0]:
        errors.append("START_SHADOW did not follow an earlier guarded STAY")
    if starts:
        start = starts[0][1]
        if not start.get("signal", {}).get("active"):
            errors.append("START_SHADOW did not use an active capacity signal")
        if float(start.get("target_kv_usage_frac", 1.0)) > max_target_kv_usage_frac:
            errors.append("START_SHADOW exceeded the target KV guard")
        if int(start.get("target_waiting", max_target_waiting + 1)) > (
            max_target_waiting
        ):
            errors.append("START_SHADOW exceeded the target waiting guard")
    required_states = ["SHADOW", "HANDOFF", "TAKEOVER"]
    positions = [
        transition_states.index(state) if state in transition_states else -1
        for state in required_states
    ]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        errors.append(f"illegal or incomplete migration path: {transition_states!r}")
    if len(ends) != 1 or ends[0].get("final_state") != "TAKEOVER":
        errors.append(f"unexpected run_end records: {ends!r}")
    elif ends[0].get("trigger_path") != "CAPACITY_PILOT":
        errors.append("terminal run did not use CAPACITY_PILOT")
    elif sorted(ends[0].get("ranks_ready", [])) != [0, 1, 2, 3]:
        errors.append("terminal run does not record four ready ranks")
    if len(commits) != 1:
        errors.append(f"expected one commit audit record, got {len(commits)}")
    if fatal:
        errors.append(f"observed fatal controller events: {fatal!r}")
    if takeover.get("state") != "COMMITTED":
        errors.append("takeover state is not COMMITTED")
    if takeover.get("source_abort_dispatched") is not True:
        errors.append("source abort was not dispatched on commit")

    token_ids = proxy.get("token_ids")
    emitted = proxy.get("emitted")
    source_tokens = int(proxy.get("source_origin_tokens", 0))
    target_tokens = int(proxy.get("target_origin_tokens", 0))
    if proxy.get("committed") is not True:
        errors.append("response proxy did not commit")
    if proxy.get("emitted_tokens") != expected_anchor_tokens:
        errors.append("unified response length differs from the anchor budget")
    if source_tokens <= 0 or target_tokens <= 0:
        errors.append("unified response is not composed from both owners")
    if proxy.get("cutover_index") != source_tokens:
        errors.append("proxy cutover does not equal the source-owned prefix")
    if not isinstance(token_ids, list) or len(token_ids) != expected_anchor_tokens:
        errors.append("proxy token IDs are missing or incomplete")
    if not isinstance(emitted, list) or [
        item.get("index") for item in emitted
    ] != list(range(expected_anchor_tokens)):
        errors.append("proxy emitted indices are not contiguous")
    unified_path = controller_dir / "unified_response.jsonl"
    unified = load_audit(unified_path) if unified_path.is_file() else []
    if [row.get("index") for row in unified] != list(range(expected_anchor_tokens)):
        errors.append("unified JSONL indices are not contiguous")
    if [row.get("token_id") for row in unified] != list(token_ids or []):
        errors.append("unified JSONL tokens differ from proxy state")

    background_ids = {
        result.get("response_id") for result in background.get("results", [])
    }
    if receipts.get("source_request_id") in background_ids:
        errors.append("a background request was selected as the anchor")
    if receipts.get("target_request_id") in background_ids:
        errors.append("the migration target collided with a background request")
    errors.extend(receipt_errors)
    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "background_completed": background.get("completed"),
        "background_failed": background.get("failed"),
        "capacity_decisions": len(capacity),
        "blocked_stay_decisions_before_rescue": len(blocked),
        "start_shadow_decisions": len(starts),
        "start_shadow_target_kv_usage_frac": (
            starts[0][1].get("target_kv_usage_frac") if starts else None
        ),
        "start_shadow_target_waiting": (
            starts[0][1].get("target_waiting") if starts else None
        ),
        "transition_states": transition_states,
        "trigger_path": ends[0].get("trigger_path") if len(ends) == 1 else None,
        "final_state": ends[0].get("final_state") if len(ends) == 1 else None,
        "takeover_state": takeover.get("state"),
        "source_abort_dispatched": takeover.get("source_abort_dispatched"),
        "receiver_ranks": receipts.get("receiver_ranks"),
        "exact_readback": receipts.get("exact_readback"),
        "source_origin_tokens": source_tokens,
        "target_origin_tokens": target_tokens,
        "anchor_emitted_tokens": proxy.get("emitted_tokens"),
        "handoff_stall_s": proxy.get("handoff_stall_s"),
        "errors": errors,
    }


def validate_inputs(
    args: argparse.Namespace,
    *,
    expected_scenario: str = "CAP-0 Rescue reachability bring-up",
) -> tuple[str, int, dict[str, Any]]:
    if os.name == "nt":
        raise RuntimeError("CAP-0 Rescue runner requires Linux")
    for path in (
        args.python_bin,
        args.model_path,
        args.manifest,
        args.survival_table,
        args.guard_file,
        common.CONFIG_TEMPLATE,
        common.SOURCE_REQUEST,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.tp1_blocks <= 0 or args.tp4_blocks <= 0:
        raise ValueError("KV block counts must be positive")
    revision = common.git("rev-parse", "HEAD")
    expected = common.git("rev-parse", args.expected_revision)
    if revision != expected:
        raise RuntimeError(f"HEAD {revision} differs from expected {expected}")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--"]
    ).returncode != 0:
        raise RuntimeError("tracked working-tree changes are present")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--cached", "--quiet", "HEAD", "--"]
    ).returncode != 0:
        raise RuntimeError("staged changes are present")
    if common.sha256(args.manifest) != args.expected_manifest_sha256:
        raise RuntimeError("manifest SHA-256 differs from expected")
    if common.sha256(args.survival_table) != args.expected_survival_sha256:
        raise RuntimeError("survival SHA-256 differs from expected")
    if common.sha256(args.guard_file) != args.expected_guard_sha256:
        raise RuntimeError("frozen guard SHA-256 differs from expected")
    guard = int(args.guard_file.read_text(encoding="utf-8").strip())
    if guard != args.expected_guard:
        raise RuntimeError(f"frozen guard {guard} differs from expected")
    pressure = validate_rescue_pressure(
        load_manifest(args.manifest),
        anchor_max_tokens=args.anchor_max_tokens,
        tp1_total_tokens=args.tp1_blocks * 16,
        tp4_total_tokens=args.tp4_blocks * 16,
        max_model_len=args.max_model_len,
        expected_scenario=expected_scenario,
    )
    if args.out_root.exists():
        raise FileExistsError(f"refusing to reuse output root {args.out_root}")
    return revision, guard, pressure


def main() -> None:
    args = parse_args()
    revision, guard, pressure = validate_inputs(args)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "scenario": "CAP-0 Rescue reachability bring-up",
                    "revision": revision,
                    "guard_free_kv_tokens": guard,
                    "manifest_sha256": common.sha256(args.manifest),
                    "pressure_preflight": pressure,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    scenario_runner.run(
        args,
        revision,
        guard,
        pressure,
        phase="bringup",
        scenario="rescue",
        scenario_title="CAP-0 Rescue reachability bring-up",
        provenance_status="BRINGUP_NOT_REPORTABLE",
        platform_note=(
            "CAP-0 RESCUE REACHABILITY BRING-UP; automated 5x A100 PCIe; "
            "not reportable"
        ),
        success_status="BRINGUP_COMPLETE",
        success_marker="RESCUE_BRINGUP_COMPLETE",
        acceptance_fn=accept_rescue,
        allow_clean_stager_exit=True,
    )


if __name__ == "__main__":
    main()
