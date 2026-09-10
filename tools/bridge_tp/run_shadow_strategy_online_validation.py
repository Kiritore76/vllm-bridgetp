#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run paired Shadow-policy or Bridge/Shadow-only live TP1/TP4 experiments.

This experiment uses the real Phase 8 KV export, TCP staging, TP4 restore,
atomic takeover, and unified response proxy.  It measures target-request TPOT
in paired pre-Shadow/Shadow/Bridge/post-commit windows.  The current online
decode path still waits for complete history before TP4 continuation, so this
runner does not claim online remote-attention execution.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402
from tools.bridge_tp import run_phase9_cap0_noop as scenario_runner  # noqa: E402
from tools.bridge_tp import run_phase9_cap0_rescue as rescue  # noqa: E402
from tools.bridge_tp.run_phase9_capacity_background import (  # noqa: E402
    load_manifest,
)
from vllm.bridge_tp.online_shadow_strategy_protocol import (  # noqa: E402
    percentile,
    summarize_background_windows,
    validate_strategy_timing,
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
    parser.add_argument("--phase", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--strategy-order", nargs=2, default=["S_NEW", "S_NEW_OLD"])
    parser.add_argument(
        "--bridge-only",
        action="store_true",
        help="run only the original S_NEW Bridge system on this branch",
    )
    parser.add_argument(
        "--architecture-comparison",
        action="store_true",
        help=(
            "pair the original Bridge path (S_NEW) with direct Shadow-only "
            "takeover (S_NEW_OLD) instead of comparing copy policies alone"
        ),
    )
    parser.add_argument(
        "--shadow-only-only",
        action="store_true",
        help="run only the Shadow-only system on this branch",
    )
    parser.add_argument(
        "--gpu-resident-shadow",
        action="store_true",
        help=(
            "reserve TP4 blocks at Shadow start and inject history/deltas "
            "before direct takeover"
        ),
    )
    parser.add_argument("--slo-tpot-ms", type=float, default=50.0)
    parser.add_argument("--slo-ttft-ms", type=float, default=1000.0)
    parser.add_argument("--slo-e2e-ms", type=float, default=60000.0)
    parser.add_argument("--slo-handoff-ms", type=float, default=1000.0)
    parser.add_argument("--trigger-output-tokens", type=int, default=128)
    parser.add_argument("--cutover-output-tokens", type=int, default=160)
    parser.add_argument("--anchor-max-tokens", type=int, default=1024)
    parser.add_argument("--minimum-ready-target-jobs", type=int, default=2)
    parser.add_argument("--background-lead-s", type=float, default=2.0)
    parser.add_argument("--minimum-window-samples", type=int, default=4)
    parser.add_argument(
        "--fixed-rate-gib-s",
        type=float,
        default=None,
        help=(
            "Pin aggregate migration bandwidth to this value. Zero means "
            "unlimited. Omit to retain the adaptive Phase 9 rate controller."
        ),
    )
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
    parser.add_argument("--run-timeout-s", type=float, default=2400)
    parser.add_argument("--stager-timeout-s", type=float, default=1800)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    if os.name == "nt":
        raise RuntimeError("online Shadow validation requires Linux and five GPUs")
    if args.phase == "formal" and args.repetitions < 3:
        raise ValueError("formal online validation requires at least three runs")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    selected_modes = sum(
        bool(value)
        for value in (
            args.bridge_only,
            args.architecture_comparison,
            args.shadow_only_only,
        )
    )
    if selected_modes > 1:
        raise ValueError(
            "select at most one of Bridge-only, architecture comparison, "
            "or Shadow-only"
        )
    if args.gpu_resident_shadow and not args.shadow_only_only:
        raise ValueError("GPU-resident Shadow currently requires --shadow-only-only")
    if set(args.strategy_order) != {"S_NEW", "S_NEW_OLD"}:
        raise ValueError("strategy order must contain S_NEW and S_NEW_OLD once")
    if not 0 < args.trigger_output_tokens < args.cutover_output_tokens:
        raise ValueError("trigger/cutover boundaries are invalid")
    if args.anchor_max_tokens <= args.cutover_output_tokens + 64:
        raise ValueError("anchor must leave at least 64 target-owned tokens")
    if args.minimum_ready_target_jobs <= 0 or args.minimum_window_samples <= 0:
        raise ValueError("online sample thresholds must be positive")
    if min(
        args.slo_tpot_ms,
        args.slo_ttft_ms,
        args.slo_e2e_ms,
        args.slo_handoff_ms,
    ) <= 0:
        raise ValueError("SLO thresholds must be positive")
    if args.fixed_rate_gib_s is not None and args.fixed_rate_gib_s < 0:
        raise ValueError("fixed migration rate cannot be negative")
    if not args.python_bin.is_file():
        raise FileNotFoundError(f"Python executable is missing: {args.python_bin}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"model path is missing: {args.model_path}")
    required_files = {
        "manifest": args.manifest,
        "survival table": args.survival_table,
        "guard file": args.guard_file,
        "controller template": common.CONFIG_TEMPLATE,
        "source request": common.SOURCE_REQUEST,
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
    revision = common.git("rev-parse", "HEAD")
    expected = common.git("rev-parse", args.expected_revision)
    if revision != expected:
        raise RuntimeError(f"HEAD {revision} differs from expected {expected}")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--"]
    ).returncode != 0:
        raise RuntimeError("tracked working-tree changes are present")
    expected_hashes = (
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (args.survival_table, args.expected_survival_sha256, "survival table"),
        (args.guard_file, args.expected_guard_sha256, "guard file"),
    )
    for path, expected_sha, label in expected_hashes:
        if common.sha256(path) != expected_sha:
            raise RuntimeError(f"{label} SHA-256 differs from expected")
    guard = int(args.guard_file.read_text(encoding="utf-8").strip())
    if guard != args.expected_guard:
        raise RuntimeError(f"frozen guard {guard} differs from expected")
    manifest = load_manifest(args.manifest)
    jobs = manifest["jobs"]
    if any(job.get("pool") != "target" for job in jobs):
        raise ValueError("online Shadow manifest must contain target jobs only")
    if len(jobs) < args.minimum_ready_target_jobs:
        raise ValueError("manifest has too few target jobs for the readiness gate")
    for job in jobs:
        prompt = job["request"].get("prompt")
        if not isinstance(prompt, list) or not all(isinstance(x, int) for x in prompt):
            raise ValueError("online target jobs require exact prompt token IDs")
        if len(prompt) + int(job["request"]["max_tokens"]) > args.max_model_len:
            raise ValueError(f"target job {job['job_id']} exceeds max model length")
    if args.out_root.exists():
        raise FileExistsError(f"refusing to reuse output root {args.out_root}")
    return revision, guard, {
        "target_jobs": len(jobs),
        "trigger_output_tokens": args.trigger_output_tokens,
        "cutover_output_tokens": args.cutover_output_tokens,
        "shadow_window_output_tokens": (
            args.cutover_output_tokens - args.trigger_output_tokens
        ),
        "minimum_ready_target_jobs": args.minimum_ready_target_jobs,
        "fixed_rate_gib_s": args.fixed_rate_gib_s,
    }


def build_controller_config_overrides(
    *,
    trigger_output_tokens: int,
    cutover_output_tokens: int,
    fixed_rate_gib_s: float | None,
) -> dict[str, Any]:
    """Keep the controller's configured Shadow window equal to the CLI design."""
    overrides: dict[str, Any] = {
        "handoff_output_tokens": cutover_output_tokens - trigger_output_tokens
    }
    if fixed_rate_gib_s is not None:
        fixed_rate_bytes_s = fixed_rate_gib_s * 1024**3
        overrides["rate"] = {
            "b_min_bytes_s": fixed_rate_bytes_s,
            "b_max_bytes_s": fixed_rate_bytes_s,
            "b_start_bytes_s": fixed_rate_bytes_s,
            "b_hard_max_bytes_s": fixed_rate_bytes_s,
        }
    return overrides


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_slo(
    results: list[dict[str, Any]],
    *,
    tpot_ms: float,
    ttft_ms: float,
    e2e_ms: float,
) -> dict[str, Any]:
    completed = [row for row in results if row.get("status") == "COMPLETED"]
    intervals: list[float] = []
    for row in completed:
        times = [float(value) for value in row.get("token_times_unix_s", [])]
        intervals.extend(
            (current - previous) * 1000
            for previous, current in zip(times, times[1:])
        )
    violating_intervals = sum(value > tpot_ms for value in intervals)
    return {
        "thresholds": {
            "tpot_ms": tpot_ms,
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
        },
        "completed_requests": len(completed),
        "token_intervals": len(intervals),
        "tpot_interval_violations": violating_intervals,
        "tpot_interval_violation_rate": (
            violating_intervals / len(intervals) if intervals else None
        ),
        "request_p99_tpot_violations": sum(
            float(row.get("tpot_p99_ms", float("inf"))) > tpot_ms
            for row in completed
        ),
        "ttft_violations": sum(
            float(row.get("ttft_ms", float("inf"))) > ttft_ms
            for row in completed
        ),
        "e2e_violations": sum(
            float(row.get("e2e_ms", float("inf"))) > e2e_ms
            for row in completed
        ),
    }


def write_measurements(out_root: Path, runs: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        acceptance = run["acceptance"]
        row: dict[str, Any] = {
            "repetition": run["repetition"],
            "strategy": run["strategy"],
            "architecture": run.get("architecture", "BRIDGE"),
            "status": acceptance["status"],
            "shadow_duration_ms": acceptance["shadow_duration_ms"],
            "bridge_to_commit_ms": acceptance["bridge_to_commit_ms"],
            "final_sync_to_commit_ms": acceptance.get(
                "final_sync_to_commit_ms"
            ),
            "handoff_stall_ms": acceptance["handoff_stall_ms"],
            "source_origin_tokens": acceptance["source_origin_tokens"],
            "target_origin_tokens": acceptance["target_origin_tokens"],
            "fixed_rate_gib_s": acceptance.get("fixed_rate_gib_s"),
            "history_payload_bytes": acceptance.get("history_payload_bytes"),
            "history_observed_aggregate_gib_s": acceptance.get(
                "history_observed_aggregate_gib_s"
            ),
            "history_max_stage_ms": acceptance.get("history_max_stage_ms"),
            "history_ready_before_freeze_ms": acceptance.get(
                "history_ready_before_freeze_ms"
            ),
            "history_gpu_ready_before_freeze_ms": acceptance.get(
                "history_gpu_ready_before_freeze_ms"
            ),
            "gpu_resident_shadow": acceptance.get("gpu_resident_shadow"),
            "gpu_history_block_acks": acceptance.get("gpu_history_block_acks"),
            "gpu_delta_acks": acceptance.get("gpu_delta_acks"),
            "anchor_tpot_p50_ms": acceptance.get("anchor_tpot", {}).get(
                "p50_ms"
            ),
            "anchor_tpot_p95_ms": acceptance.get("anchor_tpot", {}).get(
                "p95_ms"
            ),
            "anchor_tpot_p99_ms": acceptance.get("anchor_tpot", {}).get(
                "p99_ms"
            ),
            "output_throughput_tokens_s": acceptance.get("workload", {}).get(
                "output_throughput_tokens_s"
            ),
            "slo_tpot_interval_violation_rate": acceptance.get("slo", {}).get(
                "tpot_interval_violation_rate"
            ),
            "slo_ttft_violations": acceptance.get("slo", {}).get(
                "ttft_violations"
            ),
            "slo_e2e_violations": acceptance.get("slo", {}).get(
                "e2e_violations"
            ),
            "slo_handoff_violation": acceptance.get("slo", {}).get(
                "handoff_violation"
            ),
        }
        for window, metrics in acceptance["target_tpot_windows"].items():
            prefix = window.lower()
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    if not rows:
        return
    with (out_root / "measurements.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    paired: list[dict[str, Any]] = []
    repetitions = sorted({int(row["repetition"]) for row in rows})
    for repetition in repetitions:
        architecture_comparison = any(
            row.get("architecture") == "SHADOW_ONLY" for row in rows
        )
        label_field = "architecture" if architecture_comparison else "strategy"
        by_strategy = {
            str(row[label_field]): row
            for row in rows
            if int(row["repetition"]) == repetition
        }
        expected_labels = (
            {"BRIDGE", "SHADOW_ONLY"}
            if architecture_comparison
            else {"S_NEW", "S_NEW_OLD"}
        )
        if set(by_strategy) != expected_labels:
            continue
        new = by_strategy["BRIDGE" if architecture_comparison else "S_NEW"]
        old = by_strategy[
            "SHADOW_ONLY" if architecture_comparison else "S_NEW_OLD"
        ]
        required = (
            "bridge_to_commit_ms",
            "handoff_stall_ms",
            "shadow_tpot_p99_ms",
            "bridge_tpot_p99_ms",
        )
        if any(new.get(key) is None or old.get(key) is None for key in required):
            continue
        paired.append(
            {
                "repetition": repetition,
                "bridge_ms_saved_by_history_precopy": (
                    float(new["bridge_to_commit_ms"])
                    - float(old["bridge_to_commit_ms"])
                ),
                "handoff_stall_ms_saved_by_history_precopy": (
                    float(new["handoff_stall_ms"])
                    - float(old["handoff_stall_ms"])
                ),
                "shadow_target_p99_ms_extra_from_history_precopy": (
                    float(old["shadow_tpot_p99_ms"])
                    - float(new["shadow_tpot_p99_ms"])
                ),
                "bridge_target_p99_ms_delta_history_precopy": (
                    float(old["bridge_tpot_p99_ms"])
                    - float(new["bridge_tpot_p99_ms"])
                ),
            }
            if not architecture_comparison
            else {
                "repetition": repetition,
                "final_sync_ms_saved_by_shadow_only": (
                    float(new["bridge_to_commit_ms"])
                    - float(old["bridge_to_commit_ms"])
                ),
                "handoff_stall_ms_saved_by_shadow_only": (
                    float(new["handoff_stall_ms"])
                    - float(old["handoff_stall_ms"])
                ),
                "shadow_target_p99_ms_delta_shadow_only": (
                    float(old["shadow_tpot_p99_ms"])
                    - float(new["shadow_tpot_p99_ms"])
                ),
                "final_sync_target_p99_ms_delta_shadow_only": (
                    float(old["bridge_tpot_p99_ms"])
                    - float(new["bridge_tpot_p99_ms"])
                ),
            }
        )
    if paired:
        with (out_root / "paired_comparisons.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)


def accept_online(
    controller_dir: Path,
    background_dir: Path,
    expected_jobs: int,
    expected_anchor_tokens: int,
    *,
    strategy: str,
    minimum_window_samples: int,
    fixed_rate_gib_s: float | None = None,
    handoff_mode: str = "bridge",
    slo_tpot_ms: float = 50.0,
    slo_ttft_ms: float = 1000.0,
    slo_e2e_ms: float = 60000.0,
    slo_handoff_ms: float = 1000.0,
) -> dict[str, Any]:
    background = common.read_json(background_dir / "background_summary.json")
    session = common.read_json(controller_dir / "session_manifest.json")
    cutover = common.read_json(controller_dir / "cutover_manifest.json")
    staging = common.read_json(controller_dir / "staging_manifest.json")
    takeover = common.read_json(controller_dir / "takeover_state.json")
    proxy = common.read_json(controller_dir / "response_proxy_stats.json")
    audit = _load_rows(controller_dir / "phase9_audit.jsonl")
    end_rows = [row for row in audit if row.get("kind") == "run_end"]
    transitions = [row.get("to") for row in audit if row.get("kind") == "transition"]
    receipts, receipt_errors = rescue.receipt_evidence(controller_dir)
    initial_stage_receipts = [
        common.read_json(path)
        for path in sorted((controller_dir / "initial_stage_receipts").glob("*.json"))
    ]
    gpu_initial_receipts = [
        common.read_json(path)
        for path in sorted((controller_dir / "gpu_initial_receipts").glob("*.json"))
    ]

    shadow_start = float(session["shadow_started_unix_s"])
    bridge_start = float(cutover["updated_unix_s"])
    committed = float(takeover["updated_unix_s"])
    if strategy == "S_NEW":
        history_start = float(
            common.read_json(controller_dir / "history_transfer_start.json")[
                "started_unix_s"
            ]
        )
    else:
        history_start = float(session["history_transfer_started_unix_s"])
    windows = summarize_background_windows(
        background.get("results", []),
        shadow_start_unix_s=shadow_start,
        bridge_start_unix_s=bridge_start,
        committed_unix_s=committed,
    )

    errors: list[str] = []
    if background.get("jobs") != expected_jobs:
        errors.append("background job count differs from manifest")
    if background.get("completed") != expected_jobs or background.get("failed") != 0:
        errors.append("target background workload did not complete")
    if session.get("shadow_strategy") != strategy:
        errors.append("session recorded the wrong Shadow strategy")
    if staging.get("shadow_strategy") != strategy:
        errors.append("staging recorded the wrong Shadow strategy")
    expected_order = "TOKEN_ASCENDING_FROM_REQUEST_START"
    if session.get("history_copy_order") != expected_order:
        errors.append("history KV was not recorded as head-first token order")
    errors.extend(
        validate_strategy_timing(
            strategy,
            shadow_start_unix_s=shadow_start,
            bridge_start_unix_s=bridge_start,
            history_start_unix_s=history_start,
        )
    )
    expected_transitions = (
        ["SHADOW", "TAKEOVER"]
        if handoff_mode == "shadow-only"
        else ["SHADOW", "HANDOFF", "TAKEOVER"]
    )
    if transitions[-len(expected_transitions):] != expected_transitions:
        errors.append(f"unexpected migration transitions: {transitions!r}")
    history_completed = [
        float(row.get("completed_unix_s", float("inf")))
        for row in initial_stage_receipts
    ]
    if handoff_mode == "shadow-only" and (
        len(history_completed) != 4
        or any(value > bridge_start for value in history_completed)
    ):
        errors.append(
            "Shadow-only history did not finish staging on all ranks before "
            "the source freeze boundary"
        )
    history_ready_before_freeze_ms = (
        (bridge_start - max(history_completed)) * 1000
        if len(history_completed) == 4
        else None
    )
    gpu_resident_shadow = staging.get("gpu_resident_shadow") is True
    gpu_history_completed = [
        float(row.get("completed_unix_s", float("inf")))
        for row in gpu_initial_receipts
    ]
    if gpu_resident_shadow and (
        len(gpu_history_completed) != 4
        or any(value > bridge_start for value in gpu_history_completed)
        or not all(row.get("exact_readback") is True for row in gpu_initial_receipts)
    ):
        errors.append(
            "initial history was not GPU-resident on all ranks before cutover"
        )
    if gpu_resident_shadow:
        initial_end = int(session["num_computed_tokens"])
        final_end = int(cutover["num_computed_tokens"])
        expected_blocks = int(session["num_blocks"])
        for rank in range(4):
            block_paths = sorted(
                (
                    controller_dir / "gpu_block_receipts" / f"tp_rank_{rank}"
                ).glob("*.json")
            )
            blocks = [common.read_json(path) for path in block_paths]
            if (
                len(blocks) != expected_blocks
                or [int(row.get("logical_block", -1)) for row in blocks]
                != list(range(expected_blocks))
                or not all(row.get("exact_readback") is True for row in blocks)
            ):
                errors.append(f"TP4 rank {rank} history block ACKs are incomplete")
            delta_paths = sorted(
                (
                    controller_dir / "gpu_delta_receipts" / f"tp_rank_{rank}"
                ).glob("*.json")
            )
            expected_start = initial_end
            for path in delta_paths:
                delta = common.read_json(path)
                start = int(delta.get("start_token", -1))
                end = int(delta.get("end_token", -1))
                if (
                    start != expected_start
                    or end <= start
                    or delta.get("exact_readback") is not True
                ):
                    errors.append(f"TP4 rank {rank} delta ACK coverage is invalid")
                    break
                expected_start = end
            if expected_start != final_end:
                errors.append(f"TP4 rank {rank} final GPU watermark is incomplete")
    history_gpu_ready_before_freeze_ms = (
        (bridge_start - max(gpu_history_completed)) * 1000
        if len(gpu_history_completed) == 4
        else None
    )
    if len(end_rows) != 1 or end_rows[0].get("final_state") != "TAKEOVER":
        errors.append("controller did not finish in TAKEOVER")
    elif end_rows[0].get("trigger_path") != "DIAGNOSTIC_FIXED_BOUNDARY":
        errors.append("controller did not use the fixed experimental boundary")
    if takeover.get("state") != "COMMITTED":
        errors.append("takeover state is not COMMITTED")
    if proxy.get("committed") is not True:
        errors.append("unified response proxy did not commit")
    if proxy.get("emitted_tokens") != expected_anchor_tokens:
        errors.append("unified response length differs from anchor budget")
    if int(proxy.get("source_origin_tokens", 0)) <= 0 or int(
        proxy.get("target_origin_tokens", 0)
    ) <= 0:
        errors.append("unified response does not contain tokens from both owners")
    if proxy.get("handoff_stall_s") is None:
        errors.append("unified response did not record a handoff stall")
    emitted = proxy.get("emitted", [])
    if [row.get("index") for row in emitted] != list(range(expected_anchor_tokens)):
        errors.append("unified response indices are not contiguous")
    for window in ("PRE_SHADOW", "SHADOW", "BRIDGE"):
        if int(windows[window]["samples"]) < minimum_window_samples:
            errors.append(
                f"{window} has {windows[window]['samples']} TPOT samples, "
                f"requires {minimum_window_samples}"
            )
    errors.extend(receipt_errors)
    observed_rates = [
        float(row["rate_gib_s"])
        for row in audit
        if row.get("kind") == "rate" and row.get("rate_gib_s") is not None
    ]
    if fixed_rate_gib_s is not None:
        if not observed_rates:
            errors.append("controller did not record fixed-rate actuation")
        elif any(
            abs(value - fixed_rate_gib_s) > 1e-9 for value in observed_rates
        ):
            errors.append("controller deviated from the requested fixed rate")
    slo = summarize_slo(
        background.get("results", []),
        tpot_ms=slo_tpot_ms,
        ttft_ms=slo_ttft_ms,
        e2e_ms=slo_e2e_ms,
    )
    handoff_stall_ms = (
        float(proxy["handoff_stall_s"]) * 1000
        if proxy.get("handoff_stall_s") is not None
        else None
    )
    slo["handoff_ms"] = handoff_stall_ms
    slo["handoff_threshold_ms"] = slo_handoff_ms
    slo["handoff_violation"] = (
        handoff_stall_ms is None or handoff_stall_ms > slo_handoff_ms
    )
    emitted_times = [
        float(row["unix_s"])
        for row in proxy.get("emitted", [])
        if row.get("unix_s") is not None
    ]
    anchor_intervals = [
        (current - previous) * 1000
        for previous, current in zip(emitted_times, emitted_times[1:])
    ]
    workload_start = float(background.get("start_unix_s", 0.0))
    workload_end = float(background.get("end_unix_s", workload_start))
    workload_seconds = max(0.0, workload_end - workload_start)
    workload_tokens = sum(
        int(row.get("output_tokens", 0))
        for row in background.get("results", [])
        if row.get("status") == "COMPLETED"
    )
    bridge_to_commit_ms = (committed - bridge_start) * 1000
    reported_windows = dict(windows)
    if handoff_mode == "shadow-only":
        # Preserve the legacy BRIDGE key for old result readers while naming
        # the actual Shadow-only interval accurately for new analyses.
        reported_windows["FINAL_SYNC"] = dict(windows["BRIDGE"])
    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "evidence_class": (
            "ONLINE_VLLM_GPU_RESIDENT_SHADOW_TAKEOVER"
            if gpu_resident_shadow
            else "ONLINE_VLLM_SHADOW_ONLY_TAKEOVER"
            if handoff_mode == "shadow-only"
            else "ONLINE_VLLM_PHASE8_STRATEGY_COMPARISON"
        ),
        "evidence_boundary": (
            "Real vLLM TP1/TP4 export, incremental CPU relay into reserved "
            "TP4 GPU blocks, four-rank watermark, atomic takeover, unified "
            "response, and target-request TPOT. TP4 performs no migrated-"
            "request forward before commit; online remote attention is not "
            "executed."
            if gpu_resident_shadow
            else "Real vLLM TP1/TP4 export, transfer, restore, takeover, "
            "unified response, and target-request TPOT. The Shadow-only "
            "variant skips the controller Bridge/Handoff state, but stages "
            "history on CPU and restores TP4 after the final source freeze. "
            "The Bridge baseline defers history until that boundary. Online "
            "remote attention is not executed."
        ),
        "strategy": strategy,
        "handoff_mode": handoff_mode,
        "fixed_rate_gib_s": fixed_rate_gib_s,
        "observed_controller_rates_gib_s": sorted(set(observed_rates)),
        "history_payload_bytes": sum(
            int(row.get("payload_bytes", 0)) for row in initial_stage_receipts
        ),
        "history_observed_aggregate_gib_s": sum(
            float(row.get("observed_gib_s", 0.0))
            for row in initial_stage_receipts
        ),
        "history_max_stage_ms": max(
            (float(row.get("stage_ms", 0.0)) for row in initial_stage_receipts),
            default=0.0,
        ),
        "history_ready_before_freeze_ms": history_ready_before_freeze_ms,
        "history_gpu_ready_before_freeze_ms": history_gpu_ready_before_freeze_ms,
        "gpu_resident_shadow": gpu_resident_shadow,
        "gpu_history_block_acks": sum(
            1
            for _ in (controller_dir / "gpu_block_receipts").glob("**/*.json")
        ),
        "gpu_delta_acks": sum(
            1
            for _ in (controller_dir / "gpu_delta_receipts").glob("**/*.json")
        ),
        "history_copy_order": session.get("history_copy_order"),
        "target_jobs_completed": background.get("completed"),
        "shadow_duration_ms": (bridge_start - shadow_start) * 1000,
        "bridge_to_commit_ms": bridge_to_commit_ms,
        "final_sync_to_commit_ms": (
            bridge_to_commit_ms if handoff_mode == "shadow-only" else None
        ),
        "history_transfer_started_unix_s": history_start,
        "history_transfer_phase": session.get("history_transfer_phase"),
        "handoff_stall_ms": handoff_stall_ms,
        "slo": slo,
        "anchor_tpot": {
            "samples": len(anchor_intervals),
            "p50_ms": percentile(anchor_intervals, 0.50),
            "p95_ms": percentile(anchor_intervals, 0.95),
            "p99_ms": percentile(anchor_intervals, 0.99),
        },
        "workload": {
            "wall_time_s": workload_seconds,
            "output_tokens": workload_tokens,
            "output_throughput_tokens_s": (
                workload_tokens / workload_seconds if workload_seconds else None
            ),
            "request_throughput_s": (
                background.get("completed", 0) / workload_seconds
                if workload_seconds
                else None
            ),
        },
        "source_origin_tokens": proxy.get("source_origin_tokens"),
        "target_origin_tokens": proxy.get("target_origin_tokens"),
        "receiver_ranks": receipts.get("receiver_ranks"),
        "exact_readback": receipts.get("exact_readback"),
        "target_tpot_windows": reported_windows,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    revision, guard, pressure = validate_inputs(args)
    contract = {
        "format_version": 1,
        "phase": args.phase,
        "revision": revision,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": common.sha256(args.manifest),
        "survival_table_sha256": common.sha256(args.survival_table),
        "guard_file_sha256": common.sha256(args.guard_file),
        "guard_free_kv_tokens": guard,
        "strategies": args.strategy_order,
        "bridge_only": args.bridge_only,
        "architecture_comparison": args.architecture_comparison,
        "shadow_only_only": args.shadow_only_only,
        "gpu_resident_shadow": args.gpu_resident_shadow,
        "repetitions": args.repetitions,
        "fixed_rate_gib_s": args.fixed_rate_gib_s,
        "slo_thresholds": {
            "tpot_ms": args.slo_tpot_ms,
            "ttft_ms": args.slo_ttft_ms,
            "e2e_ms": args.slo_e2e_ms,
            "handoff_ms": args.slo_handoff_ms,
        },
        "pressure": pressure,
        "evidence_boundary": "online Phase 8 takeover without remote attention",
    }
    if args.validate_only:
        print(json.dumps({"status": "VALID", "contract": contract}, indent=2))
        return

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=False)
    common.write_json(out_root / "contract.json", contract)
    batch: dict[str, Any] = {
        "format_version": 1,
        "status": "RUNNING",
        "started_unix_s": time.time(),
        "contract": contract,
        "runs": [],
    }
    common.write_json(out_root / "batch_status.json", batch)
    try:
        for repetition in range(1, args.repetitions + 1):
            variants = (
                [("BRIDGE", "S_NEW", "bridge")]
                if args.bridge_only
                else (
                [
                    ("BRIDGE", "S_NEW", "bridge"),
                    ("SHADOW_ONLY", "S_NEW_OLD", "shadow-only"),
                ]
                if args.architecture_comparison
                else [("SHADOW_ONLY", "S_NEW_OLD", "shadow-only")]
                if args.shadow_only_only
                else [
                    (strategy, strategy, "bridge")
                    for strategy in args.strategy_order
                ]
                )
            )
            if repetition % 2 == 0:
                variants.reverse()
            for architecture, strategy, handoff_mode in variants:
                label = f"r{repetition:02d}_{architecture.lower()}"
                rep_args = copy.copy(args)
                rep_args.out_root = out_root / label
                run_id = f"{out_root.name}-{label}"

                def acceptance(
                    controller_dir: Path,
                    background_dir: Path,
                    expected_jobs: int,
                    expected_anchor_tokens: int,
                    selected: str = strategy,
                    selected_handoff: str = handoff_mode,
                ) -> dict[str, Any]:
                    return accept_online(
                        controller_dir,
                        background_dir,
                        expected_jobs,
                        expected_anchor_tokens,
                        strategy=selected,
                        minimum_window_samples=args.minimum_window_samples,
                        fixed_rate_gib_s=args.fixed_rate_gib_s,
                        handoff_mode=selected_handoff,
                        slo_tpot_ms=args.slo_tpot_ms,
                        slo_ttft_ms=args.slo_ttft_ms,
                        slo_e2e_ms=args.slo_e2e_ms,
                        slo_handoff_ms=args.slo_handoff_ms,
                    )

                source_env_overrides = {"BRIDGETP_SHADOW_STRATEGY": strategy}
                controller_config_overrides = build_controller_config_overrides(
                    trigger_output_tokens=args.trigger_output_tokens,
                    cutover_output_tokens=args.cutover_output_tokens,
                    fixed_rate_gib_s=args.fixed_rate_gib_s,
                )
                if args.fixed_rate_gib_s is not None:
                    source_env_overrides["BRIDGETP_STREAM_RATE_GIB_S"] = str(
                        args.fixed_rate_gib_s
                    )

                result = scenario_runner.run(
                    rep_args,
                    revision,
                    guard,
                    pressure,
                    phase="bringup" if args.phase == "smoke" else "formal",
                    repetition=repetition,
                    run_id=run_id,
                    scenario="shadow_online",
                    scenario_title=(
                        f"Online {architecture} architecture comparison"
                        if args.architecture_comparison
                        else f"Online Shadow strategy {strategy}"
                    ),
                    provenance_status=(
                        "SMOKE_NOT_REPORTABLE"
                        if args.phase == "smoke"
                        else "FORMAL_REPETITION"
                    ),
                    platform_note=(
                        f"ONLINE {architecture} / {strategy}; real Phase 8 "
                        "takeover; "
                        "no online remote attention"
                    ),
                    success_status="PASS",
                    success_marker="PASS",
                    acceptance_fn=acceptance,
                    allow_clean_stager_exit=True,
                    source_env_overrides=source_env_overrides,
                    controller_config_overrides=controller_config_overrides,
                    controller_extra_args=[
                        "--diagnostic-trigger-output-tokens",
                        str(args.trigger_output_tokens),
                        "--diagnostic-cutover-output-tokens",
                        str(args.cutover_output_tokens),
                        "--handoff-mode",
                        handoff_mode,
                    ]
                    + (["--gpu-resident-shadow"] if args.gpu_resident_shadow else []),
                    background_before_controller=True,
                    background_lead_s=args.background_lead_s,
                    background_ready_jobs=args.minimum_ready_target_jobs,
                )
                batch["runs"].append(
                    {
                        "repetition": repetition,
                        "strategy": strategy,
                        "architecture": architecture,
                        "handoff_mode": handoff_mode,
                        "status": result["status"],
                        "root": str(rep_args.out_root.resolve()),
                        "acceptance": result["acceptance"],
                    }
                )
                common.write_json(out_root / "batch_status.json", batch)

        errors = [
            f"r{row['repetition']:02d} "
            f"{row.get('architecture', row['strategy'])} failed"
            for row in batch["runs"]
            if row.get("status") != "PASS"
            or row.get("acceptance", {}).get("status") != "PASS"
        ]
        final = {
            "format_version": 1,
            "status": "PASS" if not errors else "FAIL",
            "phase": args.phase,
            "expected_runs": args.repetitions * (
                1 if (args.bridge_only or args.shadow_only_only) else 2
            ),
            "recorded_runs": len(batch["runs"]),
            "runs": batch["runs"],
            "errors": errors,
        }
        write_measurements(out_root, batch["runs"])
        common.write_json(out_root / "acceptance.json", final)
        if errors:
            raise RuntimeError("; ".join(errors))
        batch["status"] = "COMPLETE"
        batch["ended_unix_s"] = time.time()
        common.write_json(out_root / "batch_status.json", batch)
        print(f"ONLINE_SHADOW_{args.phase.upper()}_COMPLETE: {out_root}")
    except BaseException as error:
        batch["status"] = "FAILED"
        batch["ended_unix_s"] = time.time()
        batch["error"] = f"{type(error).__name__}: {error}"
        common.write_json(out_root / "batch_status.json", batch)
        raise


if __name__ == "__main__":
    main()
