#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the complete Phase 9 Section 7.1 TPOT sweep sequentially.

TP1 and TP4 API servers must already be running. The tool deliberately runs
only one benchmark at a time, while an interval telemetry subprocess records
the corresponding server. It then fits the workload-scoped tick TPOT models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDER = ROOT / "tools" / "bridge_tp" / "record_phase9_calibration.py"
FITTER = ROOT / "tools" / "bridge_tp" / "fit_phase9_tick_tpot.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="bridgetp-model")
    parser.add_argument("--tp1-url", default="http://127.0.0.1:8001")
    parser.add_argument("--tp4-url", default="http://127.0.0.1:8200")
    parser.add_argument("--tp1-blocks", type=int, required=True)
    parser.add_argument("--tp4-blocks", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--qps", type=float, nargs="+", default=(1.0, 2.0, 4.0))
    parser.add_argument("--reps", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--num-warmups", type=int, default=10)
    parser.add_argument("--telemetry-interval-s", type=float, default=1.0)
    parser.add_argument("--condition-timeout-s", type=float, default=1800.0)
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.tp1_blocks <= 0 or args.tp4_blocks <= 0 or args.block_size <= 0:
        parser.error("KV geometry must be positive")
    if any(value <= 0 for value in args.qps):
        parser.error("QPS values must be positive")
    if any(value <= 0 for value in args.reps):
        parser.error("repetitions must be positive")
    if args.num_prompts <= 0 or args.num_warmups < 0:
        parser.error("prompt counts are invalid")
    return args


def condition_matrix(args: argparse.Namespace):
    pools = (
        ("tp1", args.tp1_url, args.tp1_blocks),
        ("tp4", args.tp4_url, args.tp4_blocks),
    )
    for side, base_url, blocks in pools:
        for qps in args.qps:
            for rep in args.reps:
                yield side, base_url, blocks, qps, rep


def benchmark_command(
    args: argparse.Namespace,
    base_url: str,
    qps: float,
    rep: int,
    condition_id: str,
    condition_dir: Path,
) -> list[str]:
    return [
        args.vllm_bin,
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        base_url,
        "--endpoint",
        "/v1/completions",
        "--model",
        args.served_model_name,
        "--tokenizer",
        args.model,
        "--dataset-name",
        "random",
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--request-rate",
        str(qps),
        "--num-prompts",
        str(args.num_prompts),
        "--num-warmups",
        str(args.num_warmups),
        "--seed",
        str(rep),
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(condition_dir),
        "--result-filename",
        "benchmark.json",
        "--label",
        condition_id,
    ]


def wait_endpoint(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    url = base_url.rstrip("/") + "/v1/models"
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:
                if response.status == 200:
                    return
        except OSError as error:
            last_error = error
        time.sleep(1.0)
    raise RuntimeError(f"server not ready at {url}: {last_error}")


def stream_command(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("benchmark stdout pipe was not created")
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hashes(condition_dir: Path, paths: list[Path]) -> None:
    lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in paths
        if path.is_file()
    ]
    (condition_dir / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_condition(
    args: argparse.Namespace,
    side: str,
    base_url: str,
    blocks: int,
    qps: float,
    rep: int,
) -> Path:
    condition_id = (
        f"tpot_{side}_qps{qps:g}_r{rep}_{uuid.uuid4()}"
    )
    condition_dir = args.out_root / condition_id
    telemetry = condition_dir / "telemetry.csv"
    stop_file = condition_dir / "telemetry.stop"
    benchmark = condition_dir / "benchmark.json"
    condition_dir.mkdir(parents=True, exist_ok=False)

    recorder_command = [
        sys.executable,
        str(RECORDER),
        "--base-url",
        base_url,
        "--out",
        str(telemetry),
        "--interval-s",
        str(args.telemetry_interval_s),
        "--max-seconds",
        str(args.condition_timeout_s),
        "--block-size",
        str(args.block_size),
        "--total-kv-blocks",
        str(blocks),
        "--condition-id",
        condition_id,
        "--target-rate-gib-s",
        "0",
        "--rep",
        str(rep),
        "--stop-file",
        str(stop_file),
    ]
    bench_command = benchmark_command(
        args,
        base_url,
        qps,
        rep,
        condition_id,
        condition_dir,
    )
    manifest = {
        "format_version": 1,
        "phase": "BridgeTP Phase 9 Section 7.1",
        "condition_id": condition_id,
        "side": side,
        "base_url": base_url,
        "qps": qps,
        "rep": rep,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "block_size": args.block_size,
        "total_kv_blocks": blocks,
        "recorder_command": recorder_command,
        "benchmark_command": bench_command,
    }
    manifest_path = condition_dir / "condition_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n===== {condition_id} =====")
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return telemetry

    wait_endpoint(base_url)
    telemetry_log = (condition_dir / "telemetry.log").open(
        "w", encoding="utf-8"
    )
    recorder = subprocess.Popen(
        recorder_command,
        stdout=telemetry_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        time.sleep(max(2.0, 2 * args.telemetry_interval_s))
        if recorder.poll() is not None:
            raise RuntimeError(
                f"telemetry recorder exited early with {recorder.returncode}"
            )
        return_code = stream_command(
            bench_command,
            condition_dir / "benchmark.log",
        )
        if return_code != 0:
            raise RuntimeError(f"benchmark failed with exit code {return_code}")
    finally:
        stop_file.touch()
        try:
            recorder.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            recorder.terminate()
            recorder.wait(timeout=10.0)
        telemetry_log.close()

    if recorder.returncode != 0:
        raise RuntimeError(
            f"telemetry recorder failed with exit code {recorder.returncode}"
        )
    required = [
        telemetry,
        telemetry.with_suffix(".manifest.json"),
        benchmark,
        manifest_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"condition artifacts are missing: {missing}")
    write_hashes(condition_dir, required)
    return telemetry


def main() -> None:
    args = parse_args()
    if not args.dry_run and shutil.which(args.vllm_bin) is None:
        raise RuntimeError(f"vLLM executable not found: {args.vllm_bin}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    tp1_paths = []
    tp4_paths = []
    for side, base_url, blocks, qps, rep in condition_matrix(args):
        path = run_condition(args, side, base_url, blocks, qps, rep)
        (tp1_paths if side == "tp1" else tp4_paths).append(path)
    if args.dry_run:
        print(f"dry run complete: {len(tp1_paths) + len(tp4_paths)} conditions")
        return

    fit_out = args.out_root / "tick_tpot_candidate.json"
    fit_command = [
        sys.executable,
        str(FITTER),
        "--tp1",
        *(str(path) for path in tp1_paths),
        "--tp4",
        *(str(path) for path in tp4_paths),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--out",
        str(fit_out),
    ]
    return_code = stream_command(
        fit_command,
        args.out_root / "tick_tpot_fit.log",
    )
    if return_code != 0:
        raise RuntimeError(f"TPOT fit failed with exit code {return_code}")
    write_hashes(
        args.out_root,
        [fit_out, args.out_root / "tick_tpot_fit.log"],
    )
    print(f"completed 18 conditions; fitted model: {fit_out}")


if __name__ == "__main__":
    main()
