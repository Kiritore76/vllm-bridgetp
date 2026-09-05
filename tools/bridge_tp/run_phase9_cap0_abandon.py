#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one non-counted CAP-0 Safe-abandon bring-up."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402
from tools.bridge_tp import run_phase9_cap0_noop as scenario_runner  # noqa: E402
from tools.bridge_tp.run_phase9_cap0_rescue import (  # noqa: E402
    load_audit,
    prompt_tokens,
)
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


def validate_abandon_pressure(
    manifest: dict[str, Any],
    *,
    anchor_max_tokens: int,
    tp1_total_tokens: int,
    tp4_total_tokens: int,
    max_model_len: int,
) -> dict[str, Any]:
    if manifest.get("scenario") != "CAP-0 Safe abandon bring-up":
        raise ValueError("manifest is not labeled CAP-0 Safe abandon")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Safe-abandon manifest requires jobs")
    source = [job for job in jobs if job.get("pool") == "source"]
    target = [job for job in jobs if job.get("pool") == "target"]
    if len(source) != 4:
        raise ValueError("Safe abandon requires exactly four source jobs")
    if not 1 <= len(target) <= 4:
        raise ValueError("Safe abandon requires one to four target jobs")

    source_demand = anchor_max_tokens + sum(
        int(job["request"]["max_tokens"]) for job in source
    )
    source_fraction = source_demand / tp1_total_tokens
    if not 0.75 <= source_fraction <= 0.90:
        raise ValueError("source demand must stay inside the recovery window")
    source_starts = [float(job["start_after_s"]) for job in source]
    if max(source_starts) - min(source_starts) > 1.0:
        raise ValueError("source arrivals must form a burst within one second")

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
    target_fraction = target_demand / tp4_total_tokens
    if target_fraction >= 0.10:
        raise ValueError("target load is too high for the abandon control case")
    return {
        "source_jobs": len(source),
        "target_jobs": len(target),
        "source_output_demand_tokens": source_demand,
        "source_to_capacity_frac": source_fraction,
        "target_context_demand_tokens": target_demand,
        "target_to_capacity_frac": target_fraction,
        "tp1_total_tokens": tp1_total_tokens,
        "tp4_total_tokens": tp4_total_tokens,
    }


def accept_abandon(
    controller_dir: Path,
    background_dir: Path,
    expected_jobs: int,
    expected_anchor_tokens: int,
    max_target_kv_usage_frac: float = 0.85,
    max_target_waiting: int = 4,
    cleanup_wait_timeout_s: float = 0.0,
) -> dict[str, Any]:
    errors: list[str] = []
    cleanup_names = (
        "cleanup_request.json",
        "source_cleanup_receipt.json",
        "stager_cleanup_receipt.json",
    )
    deadline = time.monotonic() + cleanup_wait_timeout_s
    while cleanup_wait_timeout_s > 0 and any(
        not (controller_dir / name).is_file() for name in cleanup_names
    ):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)

    def required_json(name: str) -> dict[str, Any]:
        path = controller_dir / name
        if not path.is_file():
            errors.append(f"missing required abandon artifact: {name}")
            return {}
        try:
            return common.read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid abandon artifact {name}: {error}")
            return {}

    background = common.read_json(background_dir / "background_summary.json")
    rows = load_audit(controller_dir / "phase9_audit.jsonl")
    capacity = [
        row for row in rows if row.get("kind") == "capacity_pilot_decision"
    ]
    starts = [row for row in capacity if row.get("action") == "START_SHADOW"]
    transitions = [row for row in rows if row.get("kind") == "transition"]
    states = [str(row.get("to")) for row in transitions]
    shadow_clears = [
        row
        for row in rows
        if row.get("kind") == "telemetry"
        and row.get("state") == "SHADOW"
        and row.get("capacity_signal", {}).get("transition") == "CLEAR"
    ]
    abandons = [row for row in rows if row.get("kind") == "abandon"]
    cleanup_events = [
        row for row in rows if row.get("kind") == "cleanup_complete"
    ]
    ends = [row for row in rows if row.get("kind") == "run_end"]
    commits = [row for row in rows if row.get("kind") == "commit"]
    fatal_kinds = {
        "action_error",
        "commit_refused",
        "invariant_violation",
        "rollback_failed",
    }
    fatal = [row for row in rows if row.get("kind") in fatal_kinds]
    proxy = required_json("response_proxy_stats.json")
    cleanup = required_json("cleanup_request.json")
    takeover = required_json("takeover_state.json")
    source_cleanup = required_json("source_cleanup_receipt.json")
    stager_cleanup = required_json("stager_cleanup_receipt.json")

    if background.get("jobs") != expected_jobs:
        errors.append("background job count differs from the manifest")
    if background.get("completed") != expected_jobs or background.get("failed") != 0:
        errors.append("background workload did not complete cleanly")
    if len(starts) != 1:
        errors.append(f"expected one START_SHADOW decision, got {len(starts)}")
    if starts:
        start = starts[0]
        if not start.get("signal", {}).get("active"):
            errors.append("START_SHADOW did not use an active capacity signal")
        if float(start.get("target_kv_usage_frac", 1.0)) > max_target_kv_usage_frac:
            errors.append("START_SHADOW exceeded the target KV guard")
        if int(start.get("target_waiting", max_target_waiting + 1)) > (
            max_target_waiting
        ):
            errors.append("START_SHADOW exceeded the target waiting guard")
    if states != ["SHADOW", "CANCELLED"]:
        errors.append(f"unexpected abandon migration path: {states!r}")
    if not shadow_clears:
        errors.append("no causal capacity CLEAR was observed during SHADOW")
    if len(abandons) != 1 or "headroom recovered" not in str(
        abandons[0].get("reason", "") if abandons else ""
    ):
        errors.append("missing capacity-headroom abandon event")
    if len(cleanup_events) != 1:
        errors.append(
            f"expected one completed cleanup action, got {len(cleanup_events)}"
        )
    elif (
        cleanup_events[0].get("state") != "CANCELLED"
        or cleanup_events[0].get("source_abort_dispatched") is not False
    ):
        errors.append("completed cleanup action was not non-destructive")
    if len(ends) != 1 or ends[0].get("final_state") != "CANCELLED":
        errors.append(f"unexpected run_end records: {ends!r}")
    elif ends[0].get("trigger_path") != "CAPACITY_PILOT":
        errors.append("terminal run did not use CAPACITY_PILOT")
    if commits:
        errors.append("Safe abandon emitted a commit record")
    if fatal:
        errors.append(f"observed fatal controller events: {fatal!r}")

    reason = str(cleanup.get("reason", ""))
    if cleanup.get("abort_source") is not False:
        errors.append("cleanup requested source abort")
    if "headroom recovered" not in reason:
        errors.append("cleanup reason is not source headroom recovery")
    if takeover.get("state") != "CANCELLED":
        errors.append("takeover state is not CANCELLED")
    if takeover.get("source_abort_dispatched") is not False:
        errors.append("takeover state reports source abort")
    if takeover.get("source_continues_on_tp1") is not True:
        errors.append("takeover state does not preserve TP1 ownership")
    if source_cleanup.get("status") != "CLEANED":
        errors.append("source mirror cleanup is incomplete")
    if stager_cleanup.get("status") != "CLEANED":
        errors.append("stager cleanup is incomplete")

    emitted = proxy.get("emitted")
    token_ids = proxy.get("token_ids")
    source_tokens = int(proxy.get("source_origin_tokens", 0))
    target_tokens = int(proxy.get("target_origin_tokens", 0))
    if proxy.get("committed") is not False:
        errors.append("response proxy unexpectedly committed")
    if proxy.get("emitted_tokens") != expected_anchor_tokens:
        errors.append("unified response length differs from the anchor budget")
    if source_tokens != expected_anchor_tokens or target_tokens != 0:
        errors.append("client-visible output was not exclusively owned by TP1")
    if not isinstance(token_ids, list) or len(token_ids) != expected_anchor_tokens:
        errors.append("proxy token IDs are missing or incomplete")
    if not isinstance(emitted, list) or [
        item.get("index") for item in emitted
    ] != list(range(expected_anchor_tokens)):
        errors.append("proxy emitted indices are not contiguous")
    if isinstance(emitted, list) and any(
        item.get("origin") != "source" for item in emitted
    ):
        errors.append("proxy emitted a non-source token after abandon")
    unified_path = controller_dir / "unified_response.jsonl"
    unified = load_audit(unified_path) if unified_path.is_file() else []
    if [row.get("index") for row in unified] != list(range(expected_anchor_tokens)):
        errors.append("unified JSONL indices are not contiguous")
    if [row.get("token_id") for row in unified] != list(token_ids or []):
        errors.append("unified JSONL tokens differ from proxy state")

    forbidden = [
        path.name
        for path in (
            controller_dir / "cutover_manifest.json",
            controller_dir / "staging_manifest.json",
            controller_dir / "target_request.json",
            controller_dir / "target_response.json",
        )
        if path.exists()
    ]
    if forbidden:
        errors.append(f"post-cutover artifacts exist: {forbidden!r}")
    receiver_root = controller_dir / "receiver_receipts"
    receiver_receipts = (
        list(receiver_root.glob("*/tp_rank_*.json"))
        if receiver_root.is_dir()
        else []
    )
    if receiver_receipts:
        errors.append("receiver receipts exist despite pre-cutover abandon")

    session = required_json("session_manifest.json")
    background_ids = {
        result.get("response_id") for result in background.get("results", [])
    }
    if session.get("source_request_id") in background_ids:
        errors.append("a background request was selected as the anchor")
    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "background_completed": background.get("completed"),
        "background_failed": background.get("failed"),
        "capacity_decisions": len(capacity),
        "start_shadow_decisions": len(starts),
        "start_shadow_target_kv_usage_frac": (
            starts[0].get("target_kv_usage_frac") if starts else None
        ),
        "start_shadow_target_waiting": (
            starts[0].get("target_waiting") if starts else None
        ),
        "transition_states": states,
        "shadow_capacity_clear_samples": len(shadow_clears),
        "cleanup_complete_events": len(cleanup_events),
        "abandon_reason": abandons[0].get("reason") if len(abandons) == 1 else None,
        "trigger_path": ends[0].get("trigger_path") if len(ends) == 1 else None,
        "final_state": ends[0].get("final_state") if len(ends) == 1 else None,
        "takeover_state": takeover.get("state"),
        "source_abort_dispatched": takeover.get("source_abort_dispatched"),
        "source_continues_on_tp1": takeover.get("source_continues_on_tp1"),
        "source_cleanup_status": source_cleanup.get("status"),
        "stager_cleanup_status": stager_cleanup.get("status"),
        "source_origin_tokens": source_tokens,
        "target_origin_tokens": target_tokens,
        "anchor_emitted_tokens": proxy.get("emitted_tokens"),
        "forbidden_artifacts": forbidden,
        "receiver_receipt_count": len(receiver_receipts),
        "errors": errors,
    }


def validate_inputs(args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    if os.name == "nt":
        raise RuntimeError("CAP-0 Safe-abandon runner requires Linux")
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
    pressure = validate_abandon_pressure(
        load_manifest(args.manifest),
        anchor_max_tokens=args.anchor_max_tokens,
        tp1_total_tokens=args.tp1_blocks * 16,
        tp4_total_tokens=args.tp4_blocks * 16,
        max_model_len=args.max_model_len,
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
                    "scenario": "CAP-0 Safe abandon bring-up",
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
        scenario="abandon",
        scenario_title="CAP-0 Safe abandon bring-up",
        provenance_status="BRINGUP_NOT_REPORTABLE",
        platform_note=(
            "CAP-0 SAFE ABANDON BRING-UP; automated 5x A100 PCIe; "
            "not reportable"
        ),
        success_status="BRINGUP_COMPLETE",
        success_marker="ABANDON_BRINGUP_COMPLETE",
        acceptance_fn=partial(
            accept_abandon,
            cleanup_wait_timeout_s=30.0,
        ),
        allow_clean_stager_exit=True,
    )


if __name__ == "__main__":
    main()
