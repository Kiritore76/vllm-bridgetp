#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run Section-7's three-round, four-mode Experiment-A smoke gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402
from vllm.bridge_tp.experiment_timeline import merge_parts  # noqa: E402

STATIC = REPO / "tools" / "bridge_tp" / "run_experiment_a_static.py"
ONLINE = REPO / "tools" / "bridge_tp" / "run_shadow_strategy_online_validation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--anchor-prompt-tokens", type=int, default=2048)
    parser.add_argument("--anchor-max-tokens", type=int, default=1024)
    parser.add_argument("--trigger-output-tokens", type=int, default=64)
    # O160 left only 96 decode tokens for a 2048-token history and missed
    # four-rank GPU residency by 3.36 s on the Section-7 reference host.
    parser.add_argument("--cutover-output-tokens", type=int, default=256)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(args: argparse.Namespace) -> str:
    if os.name == "nt":
        raise RuntimeError("four-mode smoke requires Linux and five GPUs")
    if args.repetitions != 3:
        raise ValueError("Section-7 smoke requires exactly three paired repetitions")
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
    if revision != common.git("rev-parse", args.expected_revision):
        raise RuntimeError("HEAD differs from expected revision")
    if (
        subprocess.run(
            ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--"]
        ).returncode
        != 0
    ):
        raise RuntimeError("tracked working-tree changes are present")
    if args.out_root.exists():
        raise FileExistsError(f"refusing to reuse output root {args.out_root}")
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


def shared_server_args(args: argparse.Namespace) -> list[str]:
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
        str(args.anchor_max_tokens),
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


def online_args(
    args: argparse.Namespace, manifest: Path, manifest_sha: str
) -> list[str]:
    return [
        "--phase",
        "smoke",
        "--repetitions",
        "1",
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
        str(args.trigger_output_tokens),
        "--cutover-output-tokens",
        str(args.cutover_output_tokens),
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
        *shared_server_args(args),
    ]


def extract_row(mode: str, repetition: int, root: Path) -> dict[str, Any]:
    if mode.startswith("ALWAYS_"):
        result_path = next(root.glob("r01_*/result.json"))
        result = common.read_json(result_path)
        return {
            "repetition": repetition,
            "mode": mode,
            "status": result["status"],
            "ttft_ms": result["ttft_ms"],
            "tpot_p50_ms": result["tpot_p50_ms"],
            "tpot_p95_ms": result["tpot_p95_ms"],
            "tpot_p99_ms": result["tpot_p99_ms"],
            "e2e_ms": result["e2e_ms"],
            "handoff_stall_ms": None,
            "final_sync_to_commit_ms": None,
            "source_origin_tokens": result["output_tokens"]
            if mode == "ALWAYS_TP1"
            else 0,
            "target_origin_tokens": result["output_tokens"]
            if mode == "ALWAYS_TP4"
            else 0,
            "root": str(root.resolve()),
        }
    acceptance = common.read_json(root / "acceptance.json")
    run = acceptance["runs"][0]["acceptance"]
    return {
        "repetition": repetition,
        "mode": mode,
        "status": run["status"],
        "ttft_ms": None,
        "tpot_p50_ms": run.get("anchor_tpot", {}).get("p50_ms"),
        "tpot_p95_ms": run.get("anchor_tpot", {}).get("p95_ms"),
        "tpot_p99_ms": run.get("anchor_tpot", {}).get("p99_ms"),
        "e2e_ms": None,
        "handoff_stall_ms": run.get("handoff_stall_ms"),
        "final_sync_to_commit_ms": run.get("final_sync_to_commit_ms"),
        "source_origin_tokens": run.get("source_origin_tokens"),
        "target_origin_tokens": run.get("target_origin_tokens"),
        "root": str(root.resolve()),
    }


def main() -> None:
    args = parse_args()
    revision = validate(args)
    contract = {
        "format_version": 1,
        "gate": "SECTION_7_FOUR_MODE_SMOKE",
        "status": "VALID",
        "revision": revision,
        "modes": ["ALWAYS_TP1", "ALWAYS_TP4", "STOP_AND_COPY", "SHADOW_ONLY"],
        "repetitions": 3,
        "anchor_prompt_tokens": args.anchor_prompt_tokens,
        "anchor_max_tokens": args.anchor_max_tokens,
        "trigger_output_tokens": args.trigger_output_tokens,
        "shadow_cutover_output_tokens": args.cutover_output_tokens,
        "scope": "Stop after this smoke; A1-A5 are not launched",
    }
    if args.validate_only:
        print(json.dumps(contract, indent=2))
        return

    args.out_root.mkdir(parents=True, exist_ok=False)
    common.write_json(args.out_root / "resolved_contract.json", contract)
    manifest = args.out_root / "target_warmup_manifest.json"
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
    base_orders = [
        ["ALWAYS_TP1", "STOP_AND_COPY", "ALWAYS_TP4", "SHADOW_ONLY"],
        ["SHADOW_ONLY", "ALWAYS_TP4", "STOP_AND_COPY", "ALWAYS_TP1"],
        ["ALWAYS_TP4", "SHADOW_ONLY", "ALWAYS_TP1", "STOP_AND_COPY"],
    ]
    try:
        for repetition, modes in enumerate(base_orders, start=1):
            for mode in modes:
                root = args.out_root / f"r{repetition:02d}_{mode.lower()}"
                print(f"===== repetition={repetition} mode={mode} =====", flush=True)
                if mode.startswith("ALWAYS_"):
                    command = [
                        str(args.python_bin),
                        str(STATIC),
                        "--out-root",
                        str(root),
                        "--repetitions",
                        "1",
                        "--mode",
                        mode.lower(),
                        *shared_server_args(args),
                    ]
                else:
                    selector = (
                        "--stop-and-copy-only"
                        if mode == "STOP_AND_COPY"
                        else "--shadow-only-only"
                    )
                    command = [
                        str(args.python_bin),
                        str(ONLINE),
                        "--out-root",
                        str(root),
                        selector,
                        "--gpu-resident-shadow",
                        *online_args(args, manifest, manifest_sha),
                    ]
                run_logged(
                    command,
                    args.out_root / f"r{repetition:02d}_{mode.lower()}.console.txt",
                )
                if not mode.startswith("ALWAYS_"):
                    online_run = next(root.glob("r01_*/controller"))
                    timeline = merge_parts(online_run)
                    if timeline["status"] != "PASS":
                        raise RuntimeError(
                            f"timeline validation failed for r{repetition:02d} {mode}"
                        )
                row = extract_row(mode, repetition, root)
                rows.append(row)
                common.write_json(
                    args.out_root / "progress.json",
                    {"status": "RUNNING", "completed": len(rows), "rows": rows},
                )

        with (args.out_root / "four_mode_measurements.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        errors = [
            f"r{row['repetition']:02d} {row['mode']} failed"
            for row in rows
            if row["status"] != "PASS"
        ]
        acceptance = {
            "format_version": 1,
            "status": "PASS" if not errors and len(rows) == 12 else "FAIL",
            "expected_rows": 12,
            "recorded_rows": len(rows),
            "modes": contract["modes"],
            "repetitions": 3,
            "errors": errors,
        }
        common.write_json(args.out_root / "acceptance.json", acceptance)
        if acceptance["status"] != "PASS":
            raise RuntimeError("; ".join(errors) or "four-mode row count mismatch")
        print(f"EXPERIMENT_A_FOUR_MODE_SMOKE_COMPLETE: {args.out_root}")
    except BaseException as error:
        common.write_json(
            args.out_root / "failure.json",
            {"status": "FAILED", "error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    main()
