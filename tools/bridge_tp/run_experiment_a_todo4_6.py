#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the formal Experiment-A TODO 4, TODO 5, or TODO 6 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402

STATIC = REPO / "tools" / "bridge_tp" / "run_experiment_a_static.py"
ONLINE = REPO / "tools" / "bridge_tp" / "run_shadow_strategy_online_validation.py"

MODES = ("ALWAYS_TP1", "ALWAYS_TP4", "STOP_AND_COPY", "SHADOW_ONLY")
TODO4_ORDERS = (
    ("ALWAYS_TP1", "STOP_AND_COPY", "ALWAYS_TP4", "SHADOW_ONLY"),
    ("SHADOW_ONLY", "ALWAYS_TP4", "STOP_AND_COPY", "ALWAYS_TP1"),
    ("ALWAYS_TP4", "SHADOW_ONLY", "ALWAYS_TP1", "STOP_AND_COPY"),
)


@dataclass(frozen=True)
class Cell:
    stage: str
    repetition: int
    output_tokens: int
    mode: str
    trigger_tokens: int
    commit_tokens: int | None

    @property
    def label(self) -> str:
        commit = "none" if self.commit_tokens is None else f"o{self.commit_tokens}"
        return (
            f"r{self.repetition:02d}_out{self.output_tokens}_"
            f"{self.mode.lower()}_{commit}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("todo4", "todo5", "todo6"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--survival-table", type=Path, required=True)
    parser.add_argument("--guard-file", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-survival-sha256", required=True)
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--expected-guard", type=int, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--tp1-blocks", type=int, required=True)
    parser.add_argument("--tp4-blocks", type=int, required=True)
    parser.add_argument("--anchor-prompt-tokens", type=int, default=2048)
    parser.add_argument("--tp1-gpu", default="0")
    parser.add_argument("--tp4-gpus", default="1,2,3,4")
    parser.add_argument("--tp1-port", type=int, default=8001)
    parser.add_argument("--tp4-port", type=int, default=8200)
    parser.add_argument("--snapshot-port", type=int, default=29800)
    parser.add_argument("--delta-port", type=int, default=29900)
    parser.add_argument("--delivery-port", type=int, default=30000)
    parser.add_argument("--gpu-direct-base-port", type=int, default=30400)
    parser.add_argument("--ready-notification-port", type=int, default=30500)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--server-start-timeout-s", type=float, default=900)
    parser.add_argument("--run-timeout-s", type=float, default=7200)
    parser.add_argument("--stager-timeout-s", type=float, default=3600)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotate(values: tuple[Any, ...], offset: int) -> tuple[Any, ...]:
    index = offset % len(values)
    return values[index:] + values[:index]


def design(stage: str) -> list[Cell]:
    cells: list[Cell] = []
    if stage == "todo4":
        for repetition, order in enumerate(TODO4_ORDERS, start=1):
            for mode in order:
                commit = (
                    None
                    if mode.startswith("ALWAYS_")
                    else 64
                    if mode == "STOP_AND_COPY"
                    else 256
                )
                cells.append(
                    Cell(stage, repetition, 1024, mode, 64, commit)
                )
        return cells

    if stage == "todo5":
        outputs = (256, 1024, 4096)
        for repetition in range(1, 6):
            for output_tokens in rotate(outputs, repetition - 1):
                commit_tokens = 128 if output_tokens == 256 else 256
                for mode in rotate(MODES, repetition + output_tokens):
                    commit = (
                        None
                        if mode.startswith("ALWAYS_")
                        else 64
                        if mode == "STOP_AND_COPY"
                        else commit_tokens
                    )
                    cells.append(
                        Cell(
                            stage,
                            repetition,
                            output_tokens,
                            mode,
                            64,
                            commit,
                        )
                    )
        return cells

    for repetition in range(1, 6):
        outputs = rotate((1024, 4096), repetition - 1)
        modes = (
            ("STOP_AND_COPY", "SHADOW_ONLY")
            if repetition % 2
            else ("SHADOW_ONLY", "STOP_AND_COPY")
        )
        for output_tokens in outputs:
            for mode in modes:
                cells.append(
                    Cell(stage, repetition, output_tokens, mode, 64, 256)
                )
    return cells


def validate(args: argparse.Namespace, cells: list[Cell]) -> str:
    if os.name == "nt":
        raise RuntimeError("Experiment-A formal matrices require Linux and five GPUs")
    if not args.python_bin.is_file():
        raise FileNotFoundError(f"Python executable is missing: {args.python_bin}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"model path is missing: {args.model_path}")
    for label, path, expected in (
        ("survival table", args.survival_table, args.expected_survival_sha256),
        ("guard file", args.guard_file, args.expected_guard_sha256),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
        if sha256(path) != expected:
            raise RuntimeError(f"{label} SHA-256 differs from expected")
    if int(args.guard_file.read_text(encoding="utf-8").strip()) != args.expected_guard:
        raise RuntimeError("guard value differs from expected")
    revision = common.git("rev-parse", "HEAD")
    expected = common.git("rev-parse", args.expected_revision)
    if revision != expected:
        raise RuntimeError(f"HEAD {revision} differs from expected {expected}")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--"]
    ).returncode:
        raise RuntimeError("tracked working-tree changes are present")
    if args.out_root.exists():
        raise FileExistsError(f"refusing to reuse output root {args.out_root}")
    if max(cell.output_tokens for cell in cells) + args.anchor_prompt_tokens > (
        args.max_model_len
    ):
        raise ValueError("anchor prompt plus output exceeds max model length")
    expected_rows = {"todo4": 12, "todo5": 60, "todo6": 20}[args.stage]
    if len(cells) != expected_rows:
        raise AssertionError("internal Experiment-A design row count mismatch")
    return revision


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode:
        raise RuntimeError(
            f"child exited with code {returncode}; see {log_path.resolve()}"
        )


def shared_server_args(args: argparse.Namespace, cell: Cell) -> list[str]:
    return [
        "--python-bin",
        str(args.python_bin),
        "--model-path",
        str(args.model_path),
        "--expected-revision",
        args.expected_revision,
        "--anchor-prompt-tokens",
        str(args.anchor_prompt_tokens),
        "--anchor-max-tokens",
        str(cell.output_tokens),
        "--tp1-gpu",
        args.tp1_gpu,
        "--tp4-gpus",
        args.tp4_gpus,
        "--tp1-port",
        str(args.tp1_port),
        "--tp4-port",
        str(args.tp4_port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--server-start-timeout-s",
        str(args.server_start_timeout_s),
        "--run-timeout-s",
        str(args.run_timeout_s),
    ]


def online_command(
    args: argparse.Namespace,
    cell: Cell,
    root: Path,
    manifest: Path,
    manifest_sha: str,
) -> list[str]:
    assert cell.commit_tokens is not None
    selector = (
        "--stop-and-copy-only"
        if cell.mode == "STOP_AND_COPY"
        else "--shadow-only-only"
    )
    trigger = (
        cell.commit_tokens
        if cell.mode == "STOP_AND_COPY"
        else cell.trigger_tokens
    )
    # Stop freezes at trigger.  A later placeholder boundary satisfies the
    # generic online runner's strict trigger < cutover input contract.
    cutover = (
        min(cell.output_tokens - 65, cell.commit_tokens + 128)
        if cell.mode == "STOP_AND_COPY"
        else cell.commit_tokens
    )
    bridge = (trigger + cutover) // 2
    command = [
        str(args.python_bin),
        str(ONLINE),
        "--phase",
        "formal",
        "--managed-formal-subrun",
        "--repetitions",
        "1",
        "--out-root",
        str(root),
        selector,
        "--gpu-resident-shadow",
        "--gpu-direct-history",
        "--gpu-direct-base-port",
        str(args.gpu_direct_base_port),
        "--ready-sync-mode",
        "STREAM_EVENT",
        "--ready-notification-mode",
        "UDP",
        "--ready-notification-port",
        str(args.ready_notification_port),
        "--ready-latch-poll-ms",
        "5",
        "--manifest",
        str(manifest),
        "--expected-manifest-sha256",
        manifest_sha,
        "--survival-table",
        str(args.survival_table),
        "--guard-file",
        str(args.guard_file),
        "--expected-survival-sha256",
        args.expected_survival_sha256,
        "--expected-guard-sha256",
        args.expected_guard_sha256,
        "--expected-guard",
        str(args.expected_guard),
        "--tp1-blocks",
        str(args.tp1_blocks),
        "--tp4-blocks",
        str(args.tp4_blocks),
        "--trigger-output-tokens",
        str(trigger),
        "--bridge-output-tokens",
        str(bridge),
        "--cutover-output-tokens",
        str(cutover),
        "--minimum-ready-target-jobs",
        "1",
        "--minimum-window-samples",
        "0",
        "--background-lead-s",
        "0",
        "--snapshot-port",
        str(args.snapshot_port),
        "--delta-port",
        str(args.delta_port),
        "--delivery-port",
        str(args.delivery_port),
        "--stager-timeout-s",
        str(args.stager_timeout_s),
        *shared_server_args(args, cell),
    ]
    if cell.mode == "SHADOW_ONLY":
        command.extend(
            [
                "--gpu-direct-delta",
                "--gpu-direct-delta-batch-tokens",
                "16",
                "--gpu-direct-delta-flush-ms",
                "0",
                "--deferred-comm-destroy",
            ]
        )
    return command


def static_command(args: argparse.Namespace, cell: Cell, root: Path) -> list[str]:
    return [
        str(args.python_bin),
        str(STATIC),
        "--out-root",
        str(root),
        "--repetitions",
        "1",
        "--mode",
        cell.mode.lower(),
        *shared_server_args(args, cell),
    ]


def extract_row(cell: Cell, root: Path) -> dict[str, Any]:
    if cell.mode.startswith("ALWAYS_"):
        acceptance = common.read_json(root / "acceptance.json")
        result = acceptance["runs"][0]
        detailed = common.read_json(next(root.glob("r01_*/result.json")))
        token_rows = detailed.get("token_rows", [])
        intervals = [
            (right["monotonic_ns"] - left["monotonic_ns"]) / 1e6
            for left, right in zip(token_rows, token_rows[1:])
        ]
        row = {
            "status": result["status"],
            "ttft_ms": result.get("ttft_ms"),
            "e2e_ms": result.get("e2e_ms"),
            "itl_p50_ms": result.get("tpot_p50_ms"),
            "itl_p95_ms": result.get("tpot_p95_ms"),
            "itl_p99_ms": result.get("tpot_p99_ms"),
            "max_itl_ms": result.get("max_itl_ms"),
            "tpot_p95_ms": result.get("tpot_p95_ms"),
            "tpot_p99_ms": result.get("tpot_p99_ms"),
            "token_interval_violation_rate": (
                sum(value > 50.0 for value in intervals) / len(intervals)
                if intervals
                else 0.0
            ),
            "handoff_stall_ms": None,
            "final_sync_to_commit_ms": None,
            "freeze_to_all_rank_ready_ms": None,
            "rank_ready_skew_ms": None,
            "source_origin_tokens": (
                result.get("output_tokens") if cell.mode == "ALWAYS_TP1" else 0
            ),
            "target_origin_tokens": (
                result.get("output_tokens") if cell.mode == "ALWAYS_TP4" else 0
            ),
            "source_kv_release_after_commit_ms": None,
            "gpu_direct_delta_batches": None,
            "gpu_direct_delta_logical_submissions": None,
            "gpu_direct_delta_coalesced_submissions": None,
            "final_delta_drain_ms": None,
            "cutover_hook_to_delta_drain_ms": None,
            "mechanism_valid": result["status"] == "PASS",
            "measurement_valid": result["status"] == "PASS",
            "slo_success": None,
        }
    else:
        acceptance = common.read_json(root / "acceptance.json")
        run = acceptance["runs"][0]["acceptance"]
        row = {
            "status": run["status"],
            "ttft_ms": run.get("anchor_ttft_ms"),
            "e2e_ms": run.get("anchor_e2e_ms"),
            "itl_p50_ms": run.get("anchor_tpot", {}).get("p50_ms"),
            "itl_p95_ms": run.get("anchor_tpot", {}).get("p95_ms"),
            "itl_p99_ms": run.get("anchor_tpot", {}).get("p99_ms"),
            "max_itl_ms": run.get("anchor_tpot", {}).get("max_ms"),
            "tpot_p95_ms": run.get("anchor_tpot", {}).get("p95_ms"),
            "tpot_p99_ms": run.get("anchor_tpot", {}).get("p99_ms"),
            "token_interval_violation_rate": run.get("anchor_slo", {}).get(
                "itl_violation_rate"
            ),
            "handoff_stall_ms": run.get("handoff_stall_ms"),
            "final_sync_to_commit_ms": run.get("final_sync_to_commit_ms"),
            "freeze_to_all_rank_ready_ms": run.get(
                "freeze_to_all_rank_ready_ms"
            ),
            "rank_ready_skew_ms": run.get("rank_ready_skew_ms"),
            "source_origin_tokens": run.get("source_origin_tokens"),
            "target_origin_tokens": run.get("target_origin_tokens"),
            "source_kv_release_after_commit_ms": run.get(
                "source_kv_release_after_commit_ms"
            ),
            "gpu_direct_delta_batches": run.get("gpu_direct_delta_batches"),
            "gpu_direct_delta_logical_submissions": run.get(
                "gpu_direct_delta_logical_submissions"
            ),
            "gpu_direct_delta_coalesced_submissions": run.get(
                "gpu_direct_delta_coalesced_submissions"
            ),
            "final_delta_drain_ms": run.get("final_delta_drain_ms"),
            "cutover_hook_to_delta_drain_ms": run.get(
                "cutover_hook_to_delta_drain_ms"
            ),
            "mechanism_valid": run["status"] == "PASS" and not run.get("errors"),
            "measurement_valid": run.get("anchor_e2e_ms") is not None,
            "slo_success": run.get("anchor_slo", {}).get("success"),
        }
    return {
        "stage": cell.stage.upper(),
        "repetition": cell.repetition,
        "output_tokens": cell.output_tokens,
        "mode": cell.mode,
        "trigger_tokens": cell.trigger_tokens if cell.mode == "SHADOW_ONLY" else None,
        "commit_tokens": cell.commit_tokens,
        **row,
        "root": str(root.resolve()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(row["output_tokens"], row["mode"]) for row in rows})
    for output_tokens, mode in keys:
        group = [
            row
            for row in rows
            if row["output_tokens"] == output_tokens and row["mode"] == mode
        ]
        summary: dict[str, Any] = {
            "output_tokens": output_tokens,
            "mode": mode,
            "runs": len(group),
            "passes": sum(row["status"] == "PASS" for row in group),
        }
        for metric in (
            "e2e_ms",
            "handoff_stall_ms",
            "itl_p95_ms",
            "itl_p99_ms",
            "tpot_p95_ms",
            "tpot_p99_ms",
        ):
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None
            ]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else None
            summary[f"{metric}_median"] = statistics.median(values) if values else None
        output.append(summary)
    return output


def paired(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (int(row["repetition"]), int(row["output_tokens"])), {}
        )[str(row["mode"])] = row
    for (repetition, output_tokens), modes in sorted(groups.items()):
        if stage == "todo5" and "ALWAYS_TP1" in modes:
            baseline = float(modes["ALWAYS_TP1"]["e2e_ms"])
            for mode in MODES:
                if mode in modes:
                    output.append(
                        {
                            "repetition": repetition,
                            "output_tokens": output_tokens,
                            "mode": mode,
                            "net_gain_vs_always_tp1_ms": (
                                baseline - float(modes[mode]["e2e_ms"])
                            ),
                        }
                    )
        if stage == "todo6" and {"STOP_AND_COPY", "SHADOW_ONLY"} <= modes.keys():
            stop = modes["STOP_AND_COPY"]
            shadow = modes["SHADOW_ONLY"]
            output.append(
                {
                    "repetition": repetition,
                    "output_tokens": output_tokens,
                    "commit_tokens": 256,
                    "shadow_e2e_advantage_ms": (
                        float(stop["e2e_ms"]) - float(shadow["e2e_ms"])
                    ),
                    "handoff_ms_hidden_by_shadow": (
                        float(stop["handoff_stall_ms"])
                        - float(shadow["handoff_stall_ms"])
                    ),
                }
            )
    return output


def output_name(stage: str) -> str:
    return {
        "todo4": "post_optimization_four_mode_summary.csv",
        "todo5": "a1_output_length_results.csv",
        "todo6": "a1_same_commit_results.csv",
    }[stage]


def main() -> None:
    args = parse_args()
    cells = design(args.stage)
    revision = validate(args, cells)
    contract = {
        "format_version": 1,
        "status": "VALID",
        "stage": args.stage.upper(),
        "revision": revision,
        "expected_rows": len(cells),
        "fixed_mechanism": {
            "ready_sync_mode": "STREAM_EVENT",
            "ready_notification_mode": "UDP",
            "ready_latch_poll_ms": 5.0,
            "shadow_deferred_comm_destroy": True,
            "gpu_direct_history": True,
            "gpu_direct_delta": True,
            "delta_batch_tokens": 16,
            "delta_flush_ms": 0.0,
        },
        "cells": [asdict(cell) for cell in cells],
    }
    if args.validate_only:
        print(json.dumps(contract, indent=2))
        return

    args.out_root.mkdir(parents=True, exist_ok=False)
    common.write_json(args.out_root / "resolved_config.json", contract)
    (args.out_root / "git_commit.txt").write_text(revision + "\n", encoding="utf-8")
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "diff", "HEAD", "--"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (args.out_root / "dirty.patch").write_text(dirty, encoding="utf-8")
    common.write_json(
        args.out_root / "environment.json",
        {
            "format_version": 1,
            "python": sys.version,
            "platform": platform.platform(),
            "model_path": str(args.model_path.resolve()),
            "tp1_gpu": args.tp1_gpu,
            "tp4_gpus": args.tp4_gpus,
        },
    )
    manifest = args.out_root / "workload_manifest.json"
    common.write_json(
        manifest,
        {
            "format_version": 1,
            "note": "one-token TP4 warm-up; no measured background cohort",
            "jobs": [
                {
                    "job_id": "target_warmup",
                    "pool": "target",
                    "start_after_s": 0,
                    "request": {
                        "model": "bridgetp-model",
                        "prompt": [100] * 16,
                        "max_tokens": 1,
                        "ignore_eos": True,
                    },
                }
            ],
        },
    )
    manifest_sha = sha256(manifest)
    rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for index, cell in enumerate(cells, start=1):
            root = args.out_root / "runs" / cell.label
            root.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"===== [{index}/{len(cells)}] {cell.label} =====",
                flush=True,
            )
            command = (
                static_command(args, cell, root)
                if cell.mode.startswith("ALWAYS_")
                else online_command(args, cell, root, manifest, manifest_sha)
            )
            run_logged(command, args.out_root / f"{cell.label}.console.txt")
            row = extract_row(cell, root)
            if row["status"] != "PASS":
                raise RuntimeError(f"{cell.label} did not pass acceptance")
            if cell.mode in {"STOP_AND_COPY", "SHADOW_ONLY"} and (
                int(row["source_origin_tokens"]) != int(cell.commit_tokens)
            ):
                raise RuntimeError(
                    f"{cell.label} froze at {row['source_origin_tokens']} tokens, "
                    f"expected {cell.commit_tokens}"
                )
            rows.append(row)
            common.write_json(
                args.out_root / "progress.json",
                {
                    "status": "RUNNING",
                    "completed": len(rows),
                    "expected": len(cells),
                    "last_row": row,
                },
            )

        write_csv(args.out_root / output_name(args.stage), rows)
        write_csv(args.out_root / "aggregate_summary.csv", aggregate(rows))
        paired_rows = paired(rows, args.stage)
        if paired_rows:
            write_csv(args.out_root / "paired_comparisons.csv", paired_rows)
        acceptance = {
            "format_version": 1,
            "status": "PASS",
            "stage": args.stage.upper(),
            "expected_rows": len(cells),
            "recorded_rows": len(rows),
            "mechanism_valid_rows": sum(row["mechanism_valid"] for row in rows),
            "measurement_valid_rows": sum(
                row["measurement_valid"] for row in rows
            ),
            "started_unix_s": started,
            "ended_unix_s": time.time(),
            "errors": [],
        }
        common.write_json(args.out_root / "acceptance.json", acceptance)
        print(f"EXPERIMENT_A_{args.stage.upper()}_COMPLETE: {args.out_root}")
    except BaseException as error:
        common.write_json(
            args.out_root / "failure.json",
            {
                "status": "FAILED",
                "completed_rows": len(rows),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        if rows:
            write_csv(args.out_root / "partial_results.csv", rows)
        raise


if __name__ == "__main__":
    main()
