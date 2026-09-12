#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the Always-TP1 and Always-TP4 Experiment-A smoke baselines."""

from __future__ import annotations

import argparse
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
from vllm.bridge_tp.controller.online_io import post_streaming_completion  # noqa: E402
from vllm.bridge_tp.experiment_timeline import emit_event, merge_parts  # noqa: E402
from vllm.bridge_tp.online_shadow_strategy_protocol import percentile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("always_tp1", "always_tp4", "both"),
        default="both",
    )
    parser.add_argument("--anchor-prompt-tokens", type=int, default=2048)
    parser.add_argument("--anchor-max-tokens", type=int, default=1024)
    parser.add_argument("--tp1-gpu", default="0")
    parser.add_argument("--tp4-gpus", default="1,2,3,4")
    parser.add_argument("--tp1-port", type=int, default=8001)
    parser.add_argument("--tp4-port", type=int, default=8200)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--server-start-timeout-s", type=float, default=900)
    parser.add_argument("--run-timeout-s", type=float, default=2400)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> str:
    if os.name == "nt":
        raise RuntimeError("static Experiment-A baselines require Linux and five GPUs")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.anchor_prompt_tokens + args.anchor_max_tokens > args.max_model_len:
        raise ValueError("anchor prompt plus output exceeds max model length")
    if not args.model_path.exists():
        raise FileNotFoundError(f"model path is missing: {args.model_path}")
    if not args.python_bin.is_file():
        raise FileNotFoundError(f"Python executable is missing: {args.python_bin}")
    revision = common.git("rev-parse", "HEAD")
    expected = common.git("rev-parse", args.expected_revision)
    if revision != expected:
        raise RuntimeError(f"HEAD {revision} differs from expected {expected}")
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


def run_one(args: argparse.Namespace, root: Path, mode: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("BRIDGETP_")
    }
    env["OMP_NUM_THREADS"] = "1"
    processes: list[common.ManagedProcess] = []
    started_ns = time.time_ns()
    try:
        target = common.start_process(
            "target TP4",
            common.server_command(args, 4, args.tp4_port),
            env | {"CUDA_VISIBLE_DEVICES": args.tp4_gpus},
            root / "target_tp4.log",
        )
        processes.append(target)
        common.wait_healthy(
            f"http://127.0.0.1:{args.tp4_port}",
            target,
            args.server_start_timeout_s,
        )
        source = common.start_process(
            "source TP1",
            common.server_command(args, 1, args.tp1_port),
            env | {"CUDA_VISIBLE_DEVICES": args.tp1_gpu},
            root / "source_tp1.log",
        )
        processes.append(source)
        common.wait_healthy(
            f"http://127.0.0.1:{args.tp1_port}",
            source,
            args.server_start_timeout_s,
        )
        post_streaming_completion(
            f"http://127.0.0.1:{args.tp4_port}",
            {
                "model": "bridgetp-model",
                "prompt": [100] * 16,
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": True,
                "return_token_ids": True,
                "ignore_eos": True,
                "request_id": f"experiment-a-warmup-{root.name}",
            },
            args.run_timeout_s,
            lambda _index, _token_id, _unix_s: None,
        )
        payload = {
            "model": "bridgetp-model",
            "prompt": [100] * args.anchor_prompt_tokens,
            "max_tokens": args.anchor_max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "return_token_ids": True,
            "ignore_eos": True,
            "request_id": f"experiment-a-{root.name}",
        }
        token_rows: list[dict[str, Any]] = []

        def on_token(index: int, token_id: int, unix_s: float) -> None:
            token_rows.append(
                {
                    "index": index,
                    "token_id": token_id,
                    "unix_s": unix_s,
                    "monotonic_ns": time.monotonic_ns(),
                }
            )
            emit_event(
                root,
                "client",
                "TOKEN_RECEIVED",
                request_id=payload["request_id"],
                token_index=index,
                token_id=token_id,
                callback_unix_s=unix_s,
            )

        request_started = time.perf_counter()
        request_started_unix_ns = time.time_ns()
        emit_event(root, "client", "REQUEST_SENT", request_id=payload["request_id"])
        port = args.tp1_port if mode == "ALWAYS_TP1" else args.tp4_port
        response = post_streaming_completion(
            f"http://127.0.0.1:{port}", payload, args.run_timeout_s, on_token
        )
        emit_event(
            root,
            "client",
            "REQUEST_COMPLETE",
            request_id=payload["request_id"],
            output_tokens=len(token_rows),
        )
        request_ended = time.perf_counter()
        intervals = [
            (right["monotonic_ns"] - left["monotonic_ns"]) / 1e6
            for left, right in zip(token_rows, token_rows[1:])
        ]
        ttft_ms = (
            (token_rows[0]["monotonic_ns"] / 1e9 - request_started) * 1000
            if token_rows
            else None
        )
        result = {
            "format_version": 1,
            "status": "PASS" if len(token_rows) == args.anchor_max_tokens else "FAIL",
            "mode": mode,
            "request_started_unix_ns": request_started_unix_ns,
            "request_ended_unix_ns": time.time_ns(),
            "output_tokens": len(token_rows),
            "ttft_ms": ttft_ms,
            "e2e_ms": (request_ended - request_started) * 1000,
            "tpot_p50_ms": percentile(intervals, 0.50),
            "tpot_p95_ms": percentile(intervals, 0.95),
            "tpot_p99_ms": percentile(intervals, 0.99),
            "token_rows": token_rows,
            "finish_reason": response.get("finish_reason"),
        }
        common.write_json(root / "result.json", result)
        timeline = merge_parts(root)
        if timeline["status"] != "PASS":
            raise RuntimeError("static client timeline validation failed")
        if result["status"] != "PASS":
            raise RuntimeError(
                f"{mode} emitted {len(token_rows)}/{args.anchor_max_tokens} tokens"
            )
        return result
    finally:
        common.stop_processes(processes)
        common.write_json(
            root / "process_lifetimes.json",
            {
                "format_version": 1,
                "started_unix_ns": started_ns,
                "processes": [
                    {
                        "name": item.name,
                        "pid": item.process.pid,
                        "started_unix_s": item.started_unix_s,
                        "ended_unix_s": item.ended_unix_s,
                        "returncode": item.returncode,
                    }
                    for item in processes
                ],
            },
        )


def main() -> None:
    args = parse_args()
    revision = validate(args)
    contract = {
        "format_version": 1,
        "status": "VALID",
        "revision": revision,
        "modes": (
            [args.mode.upper()] if args.mode != "both" else ["ALWAYS_TP1", "ALWAYS_TP4"]
        ),
        "repetitions": args.repetitions,
        "anchor_prompt_tokens": args.anchor_prompt_tokens,
        "anchor_max_tokens": args.anchor_max_tokens,
    }
    if args.validate_only:
        print(json.dumps(contract, indent=2))
        return
    args.out_root.mkdir(parents=True, exist_ok=False)
    common.write_json(args.out_root / "contract.json", contract)
    runs: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        modes = (
            [args.mode.upper()] if args.mode != "both" else ["ALWAYS_TP1", "ALWAYS_TP4"]
        )
        if repetition % 2 == 0:
            modes.reverse()
        for mode in modes:
            label = f"r{repetition:02d}_{mode.lower()}"
            result = run_one(args, args.out_root / label, mode)
            runs.append(
                {
                    "repetition": repetition,
                    "mode": mode,
                    "root": str((args.out_root / label).resolve()),
                    **{
                        key: value
                        for key, value in result.items()
                        if key != "token_rows"
                    },
                }
            )
            print(f"PASS: {args.out_root / label}", flush=True)
    acceptance = {
        "format_version": 1,
        "status": "PASS",
        "expected_runs": args.repetitions * (1 if args.mode != "both" else 2),
        "recorded_runs": len(runs),
        "runs": runs,
        "errors": [],
    }
    common.write_json(args.out_root / "acceptance.json", acceptance)
    print(f"EXPERIMENT_A_STATIC_SMOKE_COMPLETE: {args.out_root}")


if __name__ == "__main__":
    main()
