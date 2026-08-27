#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Automate the Phase 9 Section 7.2 pilot or formal interference grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

from run_phase9_tpot_sweep import (  # noqa: E402
    stream_command,
    wait_endpoint,
    write_hashes,
)

RECORDER = TOOLS_DIR / "record_phase9_calibration.py"
COPY_LOAD = TOOLS_DIR / "run_phase9_copy_load.py"
ANALYZER = TOOLS_DIR / "analyze_phase9_calibration.py"
SUMMARIZER = TOOLS_DIR / "summarize_phase9_calibration.py"
BANDS = {
    "low": (0.15, 0.25),
    "medium": (0.45, 0.55),
    "high": (0.75, 0.85),
}
RATES = (0.0, 0.4, 0.7, 1.2)
REPS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "formal"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="bridgetp-model")
    parser.add_argument("--tp4-url", default="http://127.0.0.1:8200")
    parser.add_argument("--tp4-blocks", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    parser.add_argument("--chunk-mib", type=float, default=16.0)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=2048)
    parser.add_argument("--num-warmups", type=int, default=10)
    parser.add_argument("--copy-delay-s", type=float, default=60.0)
    parser.add_argument("--copy-seconds", type=float, default=300.0)
    parser.add_argument("--drain-margin-s", type=float, default=60.0)
    parser.add_argument("--telemetry-interval-s", type=float, default=1.0)
    parser.add_argument("--condition-timeout-s", type=float, default=3600.0)
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument(
        "--candidate-qps",
        type=float,
        nargs="+",
        default=(0.2, 0.4, 0.53, 0.7, 1.0, 1.5),
    )
    parser.add_argument("--pilot-summary", type=Path)
    args = parser.parse_args()
    if args.tp4_blocks <= 0 or args.block_size <= 0:
        parser.error("KV geometry must be positive")
    if args.source_gpu == args.target_gpu:
        parser.error("source and target GPUs must differ")
    if any(qps <= 0 for qps in args.candidate_qps):
        parser.error("candidate QPS values must be positive")
    if args.mode == "formal" and args.pilot_summary is None:
        parser.error("formal mode requires --pilot-summary")
    return args


def benchmark_command(
    args: argparse.Namespace,
    qps: float,
    rep: int,
    condition_id: str,
    condition_dir: Path,
) -> list[str]:
    duration = args.copy_delay_s + args.copy_seconds + args.drain_margin_s
    prompts = max(1, math.ceil(qps * duration))
    return [
        args.vllm_bin,
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        args.tp4_url,
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
        str(prompts),
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


def recorder_command(
    args: argparse.Namespace,
    condition_id: str,
    condition_dir: Path,
    rate: float,
    rep: int,
    band: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(RECORDER),
        "--base-url",
        args.tp4_url,
        "--out",
        str(condition_dir / "telemetry.csv"),
        "--interval-s",
        str(args.telemetry_interval_s),
        "--max-seconds",
        str(args.condition_timeout_s),
        "--block-size",
        str(args.block_size),
        "--total-kv-blocks",
        str(args.tp4_blocks),
        "--condition-id",
        condition_id,
        "--target-rate-gib-s",
        str(rate),
        "--rep",
        str(rep),
        "--stop-file",
        str(condition_dir / "telemetry.stop"),
    ]
    if band is not None:
        command.extend(("--load-band", band))
    return command


def copy_command(
    args: argparse.Namespace, condition_dir: Path, rate: float
) -> list[str]:
    return [
        sys.executable,
        str(COPY_LOAD),
        "--src",
        str(args.source_gpu),
        "--dst",
        str(args.target_gpu),
        "--chunk-mib",
        str(args.chunk_mib),
        "--target-gib-s",
        str(rate),
        "--seconds",
        str(args.copy_seconds),
        "--out",
        str(condition_dir / "copy_window.json"),
    ]


def window_load_summary(condition_dir: Path) -> dict[str, object]:
    copy = json.loads(
        (condition_dir / "copy_window.json").read_text(encoding="utf-8")
    )
    start = float(copy["start_perf_counter"])
    end = float(copy["end_perf_counter"])
    with (condition_dir / "telemetry.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    values = []
    for row in rows:
        interval_end = float(row["monotonic_s"])
        interval_start = interval_end - float(row["interval_s"])
        if start <= interval_start <= interval_end <= end:
            values.append(float(row["kv_usage_frac"]))
    if not values:
        raise RuntimeError("no telemetry intervals fall inside the copy window")
    mean = sum(values) / len(values)
    fractions = {
        band: sum(low <= value <= high for value in values) / len(values)
        for band, (low, high) in BANDS.items()
    }
    return {
        "samples": len(values),
        "kv_usage_mean": mean,
        "kv_usage_min": min(values),
        "kv_usage_max": max(values),
        "band_fractions": fractions,
    }


def run_condition(
    args: argparse.Namespace,
    prefix: str,
    qps: float,
    rate: float,
    rep: int,
    band: str | None,
) -> tuple[Path, dict[str, object]]:
    condition_id = (
        f"{prefix}_qps{qps:g}_rate{rate:g}_r{rep}_{uuid.uuid4()}"
    )
    condition_dir = args.out_root / condition_id
    condition_dir.mkdir(parents=True, exist_ok=False)
    recorder_cmd = recorder_command(
        args, condition_id, condition_dir, rate, rep, band
    )
    benchmark_cmd = benchmark_command(
        args, qps, rep, condition_id, condition_dir
    )
    copy_cmd = copy_command(args, condition_dir, rate)
    manifest = {
        "format_version": 1,
        "phase": "BridgeTP Phase 9 Section 7.2",
        "condition_id": condition_id,
        "mode": prefix,
        "load_band": band,
        "qps": qps,
        "target_rate_gib_s": rate,
        "rep": rep,
        "recorder_command": recorder_cmd,
        "benchmark_command": benchmark_cmd,
        "copy_command": copy_cmd,
    }
    manifest_path = condition_dir / "condition_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n===== {condition_id} =====")

    telemetry_log = (condition_dir / "telemetry.log").open(
        "w", encoding="utf-8"
    )
    benchmark_log = (condition_dir / "benchmark.log").open(
        "w", encoding="utf-8"
    )
    recorder = subprocess.Popen(
        recorder_cmd,
        stdout=telemetry_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    benchmark = None
    try:
        time.sleep(max(2.0, 2 * args.telemetry_interval_s))
        if recorder.poll() is not None:
            raise RuntimeError("telemetry recorder exited before benchmark")
        benchmark = subprocess.Popen(
            benchmark_cmd,
            stdout=benchmark_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(args.copy_delay_s)
        if benchmark.poll() is not None:
            raise RuntimeError("benchmark ended before the copy window")
        copy_return = stream_command(
            copy_cmd,
            condition_dir / "copy_window.log",
        )
        if copy_return != 0:
            raise RuntimeError(f"copy window failed with exit code {copy_return}")
    finally:
        failed = sys.exc_info()[0] is not None
        (condition_dir / "telemetry.stop").touch()
        try:
            recorder.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            recorder.terminate()
            recorder.wait(timeout=10.0)
        telemetry_log.close()
        if failed:
            if benchmark is not None and benchmark.poll() is None:
                benchmark.terminate()
                try:
                    benchmark.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    benchmark.kill()
                    benchmark.wait(timeout=10.0)
            benchmark_log.close()

    if recorder.returncode != 0:
        raise RuntimeError(f"telemetry recorder failed: {recorder.returncode}")
    if benchmark is None:
        raise RuntimeError("benchmark was not started")
    try:
        benchmark_return = benchmark.wait(timeout=args.condition_timeout_s)
    except subprocess.TimeoutExpired:
        benchmark.terminate()
        benchmark.wait(timeout=30.0)
        raise RuntimeError("benchmark exceeded condition timeout")
    finally:
        benchmark_log.close()
    if benchmark_return != 0:
        raise RuntimeError(f"benchmark failed with exit code {benchmark_return}")

    required = [
        condition_dir / "telemetry.csv",
        condition_dir / "telemetry.manifest.json",
        condition_dir / "benchmark.json",
        condition_dir / "copy_window.json",
        manifest_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"condition artifacts are missing: {missing}")
    load_summary = window_load_summary(condition_dir)
    (condition_dir / "load_summary.json").write_text(
        json.dumps(load_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_hashes(
        condition_dir,
        [*required, condition_dir / "load_summary.json"],
    )
    return condition_dir, load_summary


def choose_band_qps(conditions: list[dict]) -> dict[str, dict | None]:
    selected = {}
    for band, (low, high) in BANDS.items():
        midpoint = (low + high) / 2
        candidates = [
            item
            for item in conditions
            if low <= item["kv_usage_mean"] <= high
            and item["band_fractions"][band] >= 0.80
        ]
        candidates.sort(
            key=lambda item: (abs(item["kv_usage_mean"] - midpoint), item["qps"])
        )
        selected[band] = candidates[0] if candidates else None
    return selected


def run_pilot(args: argparse.Namespace) -> None:
    conditions = []
    for qps in args.candidate_qps:
        condition_dir, summary = run_condition(
            args,
            prefix="pilot",
            qps=qps,
            rate=0.0,
            rep=1,
            band=None,
        )
        conditions.append(
            {
                "condition_dir": str(condition_dir),
                "qps": qps,
                **summary,
            }
        )
    selected = choose_band_qps(conditions)
    payload = {
        "format_version": 1,
        "phase": "BridgeTP Phase 9 Section 7.2 load pilot",
        "status": (
            "READY" if all(value is not None for value in selected.values())
            else "MORE_QPS_CANDIDATES_REQUIRED"
        ),
        "platform_scope": "NVIDIA A100 PCIe only",
        "conditions": conditions,
        "selected": selected,
        "selection_rule": (
            "Require mean inside band and at least 80% of interval samples "
            "inside band; choose the mean nearest the band midpoint."
        ),
        "formal_model_conclusion": None,
    }
    out = args.out_root / "load_pilot_summary.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out}")
    if payload["status"] != "READY":
        raise SystemExit(2)


def analyze_formal_condition(
    args: argparse.Namespace,
    condition_dir: Path,
    band: str,
    rate: float,
) -> Path:
    low, high = BANDS[band]
    out = condition_dir / "condition_result.json"
    command = [
        sys.executable,
        str(ANALYZER),
        "--telemetry",
        str(condition_dir / "telemetry.csv"),
        "--benchmark-json",
        str(condition_dir / "benchmark.json"),
        "--copy-json",
        str(condition_dir / "copy_window.json"),
        "--target-rate-gib-s",
        str(rate),
        "--load-min",
        str(low),
        "--load-max",
        str(high),
        "--min-band-fraction",
        "0.80",
        "--rate-relative-tolerance",
        "0.05",
        "--out",
        str(out),
    ]
    return_code = stream_command(command, condition_dir / "analysis.log")
    if return_code != 0:
        raise RuntimeError(f"condition rejected: {condition_dir}")
    write_hashes(
        condition_dir,
        [
            condition_dir / "telemetry.csv",
            condition_dir / "telemetry.manifest.json",
            condition_dir / "benchmark.json",
            condition_dir / "copy_window.json",
            condition_dir / "load_summary.json",
            out,
        ],
    )
    return out


def run_formal(args: argparse.Namespace) -> None:
    pilot = json.loads(args.pilot_summary.read_text(encoding="utf-8"))
    if pilot.get("status") != "READY":
        raise RuntimeError("pilot summary is not READY")
    qps_by_band = {
        band: float(pilot["selected"][band]["qps"]) for band in BANDS
    }
    results = []
    for band in BANDS:
        for rate in RATES:
            for rep in REPS:
                condition_dir, _ = run_condition(
                    args,
                    prefix=f"formal_{band}",
                    qps=qps_by_band[band],
                    rate=rate,
                    rep=rep,
                    band=band,
                )
                results.append(
                    analyze_formal_condition(
                        args, condition_dir, band, rate
                    )
                )
    summary_out = args.out_root / "bridge_calibration_summary.json"
    command = [
        sys.executable,
        str(SUMMARIZER),
        "--inputs",
        *(str(path) for path in results),
        "--out",
        str(summary_out),
    ]
    return_code = stream_command(
        command,
        args.out_root / "bridge_calibration_summary.log",
    )
    if return_code != 0:
        raise RuntimeError("formal interference summary is incomplete")
    write_hashes(
        args.out_root,
        [
            args.pilot_summary,
            summary_out,
            args.out_root / "bridge_calibration_summary.log",
        ],
    )
    print(f"completed 36 formal conditions: {summary_out}")


def main() -> None:
    args = parse_args()
    if shutil.which(args.vllm_bin) is None:
        raise RuntimeError(f"vLLM executable not found: {args.vllm_bin}")
    wait_endpoint(args.tp4_url)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "pilot":
        run_pilot(args)
    else:
        run_formal(args)


if __name__ == "__main__":
    main()
