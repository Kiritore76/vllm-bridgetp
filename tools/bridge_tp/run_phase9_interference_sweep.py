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
BAND_PROFILES = {
    "standard": {
        "low": (0.10, 0.25),
        "medium": (0.30, 0.45),
        "high": (0.50, 0.65),
    },
    "very_low": {
        "very_low": (0.01, 0.06),
    },
}
BANDS = dict(BAND_PROFILES["standard"])
PROTOCOL_AMENDMENT = (
    "A100 PCIe TP4 I256/O2048 attainable-load amendment made before any "
    "nonzero-copy formal condition: two rate-zero pilots and an isolated "
    "QPS 3.0/3.5/4.0 reachability diagnostic showed running saturated near "
    "256 while waiting increased, so the original 0.15-0.25/0.45-0.55/"
    "0.75-0.85 bands were not jointly sustainable. Offline reclassification "
    "of isolated rate-zero telemetry then froze 0.50-0.65 as the narrowest "
    "high band passing a 120-second stability window followed by a separate "
    "300-second measurement window at the 0.80 fraction threshold; lower "
    "bounds 0.54 and 0.52 failed."
)
RATES = (0.0, 0.4, 0.7, 1.2)
REPS = (1, 2, 3)


def serialized_bands() -> dict[str, list[float]]:
    return {name: [low, high] for name, (low, high) in BANDS.items()}


def validate_pilot_for_formal(pilot: dict[str, object]) -> None:
    if pilot.get("status") != "READY":
        raise RuntimeError("pilot summary is not READY")
    if pilot.get("load_bands") != serialized_bands():
        raise RuntimeError(
            "pilot load bands do not match this runner; rerun the "
            "rate-zero pilot with the current protocol"
        )


def parse_args() -> argparse.Namespace:
    global BANDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "formal"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="bridgetp-model")
    parser.add_argument("--tp4-url", default="http://127.0.0.1:8200")
    parser.add_argument("--tp4-blocks", type=int, required=True)
    parser.add_argument(
        "--band-profile",
        choices=tuple(BAND_PROFILES),
        default="standard",
        help="isolated preregistered load-band profile (default: standard)",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    parser.add_argument("--chunk-mib", type=float, default=16.0)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=2048)
    parser.add_argument("--num-warmups", type=int, default=0)
    parser.add_argument("--copy-delay-s", type=float, default=60.0)
    parser.add_argument("--load-settle-timeout-s", type=float, default=300.0)
    parser.add_argument("--stability-window-s", type=float, default=60.0)
    parser.add_argument("--stability-poll-s", type=float, default=5.0)
    parser.add_argument("--min-band-fraction", type=float, default=0.80)
    parser.add_argument(
        "--measurement-load-policy",
        choices=("strict_band", "observed_safe"),
        default="strict_band",
    )
    parser.add_argument("--measurement-max-kv-p95", type=float, default=0.85)
    parser.add_argument("--min-measurement-samples", type=int, default=290)
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
    parser.add_argument(
        "--formal-bands",
        nargs="+",
        choices=tuple(
            dict.fromkeys(
                band for profile in BAND_PROFILES.values() for band in profile
            )
        ),
        default=None,
    )
    parser.add_argument("--formal-rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--formal-reps", nargs="+", type=int, default=REPS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse ACCEPTED formal condition results already present under "
            "--out-root"
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="maximum new attempts per missing formal condition (default: 2)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "after exhausting retries for one formal condition, continue "
            "with the rest of the grid"
        ),
    )
    args = parser.parse_args()
    BANDS = dict(BAND_PROFILES[args.band_profile])
    if args.tp4_blocks <= 0 or args.block_size <= 0:
        parser.error("KV geometry must be positive")
    if args.source_gpu == args.target_gpu:
        parser.error("source and target GPUs must differ")
    if any(qps <= 0 for qps in args.candidate_qps):
        parser.error("candidate QPS values must be positive")
    if args.num_warmups < 0:
        parser.error("num-warmups cannot be negative")
    if not 0 <= args.copy_delay_s < args.load_settle_timeout_s:
        parser.error(
            "copy-delay-s must be non-negative and below "
            "load-settle-timeout-s"
        )
    if not 0 < args.stability_window_s < args.load_settle_timeout_s:
        parser.error(
            "stability-window-s must be positive and below "
            "load-settle-timeout-s"
        )
    if args.stability_poll_s <= 0:
        parser.error("stability-poll-s must be positive")
    if not 0 < args.min_band_fraction <= 1:
        parser.error("min-band-fraction must be in (0,1]")
    if not 0 < args.measurement_max_kv_p95 <= 1:
        parser.error("measurement-max-kv-p95 must be in (0,1]")
    if args.min_measurement_samples <= 0:
        parser.error("min-measurement-samples must be positive")
    if any(rate not in RATES for rate in args.formal_rates):
        parser.error(f"formal-rates must be selected from {RATES}")
    if any(rep not in REPS for rep in args.formal_reps):
        parser.error(f"formal-reps must be selected from {REPS}")
    if args.formal_bands is None:
        args.formal_bands = tuple(BANDS)
    elif any(band not in BANDS for band in args.formal_bands):
        parser.error(
            f"formal-bands must come from profile {args.band_profile}: "
            f"{tuple(BANDS)}"
        )
    args.formal_bands = tuple(dict.fromkeys(args.formal_bands))
    args.formal_rates = tuple(dict.fromkeys(args.formal_rates))
    args.formal_reps = tuple(dict.fromkeys(args.formal_reps))
    if args.max_attempts <= 0:
        parser.error("max-attempts must be positive")
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
    duration = (
        args.load_settle_timeout_s
        + args.copy_seconds
        + args.drain_margin_s
    )
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


def recent_load_summary(
    telemetry: Path,
    window_s: float,
    interval_s: float,
) -> dict[str, object] | None:
    """Summarize the most recent complete telemetry window."""
    if not telemetry.is_file():
        return None
    try:
        with telemetry.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    parsed = []
    for row in rows:
        try:
            parsed.append(
                (
                    float(row["monotonic_s"]),
                    float(row["kv_usage_frac"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not parsed:
        return None
    end = parsed[-1][0]
    values = [value for timestamp, value in parsed if timestamp >= end - window_s]
    if len(values) < 2:
        return None
    coverage_s = parsed[-1][0] - next(
        timestamp for timestamp, _ in parsed if timestamp >= end - window_s
    )
    # Allow two scrape intervals of scheduling jitter before declaring the
    # rolling window incomplete.
    if coverage_s < max(0.0, window_s - 2.0 * interval_s):
        return None
    mean = sum(values) / len(values)
    fractions = {
        band: sum(low <= value <= high for value in values) / len(values)
        for band, (low, high) in BANDS.items()
    }
    return {
        "samples": len(values),
        "window_s": window_s,
        "coverage_s": coverage_s,
        "kv_usage_mean": mean,
        "kv_usage_min": min(values),
        "kv_usage_max": max(values),
        "band_fractions": fractions,
    }


def matching_stable_band(
    summary: dict[str, object] | None,
    requested_band: str | None,
    min_fraction: float,
) -> str | None:
    """Return the band whose rolling window satisfies the frozen rule."""
    if summary is None:
        return None
    mean = float(summary["kv_usage_mean"])
    fractions = summary["band_fractions"]
    names = (requested_band,) if requested_band is not None else tuple(BANDS)
    for name in names:
        low, high = BANDS[name]
        if low <= mean <= high and float(fractions[name]) >= min_fraction:
            return name
    return None


def wait_for_stable_load(
    args: argparse.Namespace,
    telemetry: Path,
    recorder: subprocess.Popen,
    benchmark: subprocess.Popen,
    requested_band: str | None,
) -> tuple[str | None, dict[str, object] | None, float]:
    """Wait for a measured rolling load window before starting copy."""
    started = time.monotonic()
    not_before = started + args.copy_delay_s
    deadline = started + args.load_settle_timeout_s
    latest = None
    while True:
        now = time.monotonic()
        recorder_return = recorder.poll()
        if recorder_return is not None:
            raise RuntimeError(
                "telemetry recorder exited before load became stable "
                f"with code {recorder_return}"
            )
        if benchmark.poll() is not None:
            raise RuntimeError("benchmark ended before load became stable")
        if now >= not_before:
            latest = recent_load_summary(
                telemetry,
                args.stability_window_s,
                args.telemetry_interval_s,
            )
            matched = matching_stable_band(
                latest,
                requested_band,
                args.min_band_fraction,
            )
            if matched is not None:
                return matched, latest, now - started
        if now >= deadline:
            return None, latest, now - started
        time.sleep(min(args.stability_poll_s, max(0.0, deadline - now)))


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
        "load_stability": {
            "requested_band": band,
            "minimum_delay_s": args.copy_delay_s,
            "timeout_s": args.load_settle_timeout_s,
            "window_s": args.stability_window_s,
            "poll_s": args.stability_poll_s,
            "min_band_fraction": args.min_band_fraction,
            "load_bands": serialized_bands(),
            "protocol_amendment": PROTOCOL_AMENDMENT,
        },
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
    stability = None
    stable_band = None
    settle_elapsed_s = None
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
        stable_band, stability, settle_elapsed_s = wait_for_stable_load(
            args,
            condition_dir / "telemetry.csv",
            recorder,
            benchmark,
            band,
        )
        stability_payload = {
            "format_version": 1,
            "status": "STABLE" if stable_band is not None else "TIMEOUT",
            "requested_band": band,
            "matched_band": stable_band,
            "settle_elapsed_s": settle_elapsed_s,
            "minimum_delay_s": args.copy_delay_s,
            "timeout_s": args.load_settle_timeout_s,
            "window_s": args.stability_window_s,
            "min_band_fraction": args.min_band_fraction,
            "observed": stability,
        }
        (condition_dir / "load_stability.json").write_text(
            json.dumps(stability_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if stable_band is None:
            if band is not None:
                raise RuntimeError(
                    f"load did not stabilize in {band} band before timeout"
                )
            # A pilot miss is evidence about this candidate, not a reason to
            # discard the other preregistered candidates.
            return_code = None
        else:
            return_code = stream_command(
                copy_cmd,
                condition_dir / "copy_window.log",
            )
            if return_code != 0:
                raise RuntimeError(
                    f"copy window failed with exit code {return_code}"
                )
    finally:
        failed = sys.exc_info()[0] is not None
        (condition_dir / "telemetry.stop").touch()
        try:
            recorder.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            recorder.terminate()
            recorder.wait(timeout=10.0)
        telemetry_log.close()
        if failed or stable_band is None:
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
    if stable_band is None:
        latest = stability or {
            "samples": 0,
            "window_s": args.stability_window_s,
            "coverage_s": 0.0,
            "kv_usage_mean": 0.0,
            "kv_usage_min": 0.0,
            "kv_usage_max": 0.0,
            "band_fractions": {name: 0.0 for name in BANDS},
        }
        summary = {
            **latest,
            "stability_status": "TIMEOUT",
            "matched_band": None,
            "settle_elapsed_s": settle_elapsed_s,
        }
        (condition_dir / "load_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_hashes(
            condition_dir,
            [
                condition_dir / "telemetry.csv",
                condition_dir / "telemetry.manifest.json",
                manifest_path,
                condition_dir / "load_stability.json",
                condition_dir / "load_summary.json",
            ],
        )
        return condition_dir, summary
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
        condition_dir / "load_stability.json",
        manifest_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"condition artifacts are missing: {missing}")
    load_summary = window_load_summary(condition_dir)
    load_summary.update(
        {
            "stability_status": "STABLE",
            "matched_band": stable_band,
            "settle_elapsed_s": settle_elapsed_s,
            "pre_copy_stability": stability,
        }
    )
    (condition_dir / "load_summary.json").write_text(
        json.dumps(load_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_hashes(
        condition_dir,
        [*required, condition_dir / "load_summary.json"],
    )
    return condition_dir, load_summary


def choose_band_qps(
    conditions: list[dict],
    min_band_fraction: float = 0.80,
) -> dict[str, dict | None]:
    selected = {}
    for band, (low, high) in BANDS.items():
        midpoint = (low + high) / 2
        candidates = [
            item
            for item in conditions
            if item.get("stability_status", "STABLE") == "STABLE"
            and low <= item["kv_usage_mean"] <= high
            and item["band_fractions"][band] >= min_band_fraction
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
    selected = choose_band_qps(conditions, args.min_band_fraction)
    payload = {
        "format_version": 1,
        "phase": "BridgeTP Phase 9 Section 7.2 load pilot",
        "status": (
            "READY" if all(value is not None for value in selected.values())
            else "MORE_QPS_CANDIDATES_REQUIRED"
        ),
        "platform_scope": "NVIDIA A100 PCIe only",
        "workload_scope": "TP4 I256/O2048",
        "load_bands": serialized_bands(),
        "protocol_amendment": PROTOCOL_AMENDMENT,
        "band_profile": args.band_profile,
        "conditions": conditions,
        "selected": selected,
        "selection_rule": (
            "Start the measurement only after a rolling stable-load window; "
            "then require the full rate-zero measurement mean inside the "
            "band and at least 80% of its samples inside the band; choose "
            "the mean nearest the band midpoint."
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
        str(args.min_band_fraction),
        "--measurement-load-policy",
        args.measurement_load_policy,
        "--measurement-max-kv-p95",
        str(args.measurement_max_kv_p95),
        "--min-measurement-samples",
        str(args.min_measurement_samples),
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


FormalKey = tuple[str, float, int]


def formal_expected_keys(args: argparse.Namespace) -> list[FormalKey]:
    return [
        (band, rate, rep)
        for band in args.formal_bands
        for rate in args.formal_rates
        for rep in args.formal_reps
    ]


def formal_result_key(payload: dict[str, object]) -> FormalKey:
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("formal result lacks inputs")
    return (
        str(payload.get("load_band", "")),
        float(inputs["target_rate_gib_s"]),
        int(payload.get("rep", 0)),
    )


def accepted_formal_results(out_root: Path) -> dict[FormalKey, Path]:
    """Find one reusable ACCEPTED result for each formal grid key."""
    expected = {
        (band, rate, rep)
        for band in BANDS
        for rate in RATES
        for rep in REPS
    }
    accepted: dict[FormalKey, Path] = {}
    for path in sorted(out_root.glob("*/condition_result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = formal_result_key(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"ignoring unreadable formal result {path}: {error}")
            continue
        if payload.get("status") == "ACCEPTED" and key in expected:
            accepted.setdefault(key, path)
    return accepted


def existing_formal_attempt_counts(out_root: Path) -> dict[FormalKey, int]:
    """Count prior formal condition directories for each grid key."""
    expected = {
        (band, rate, rep)
        for band in BANDS
        for rate in RATES
        for rep in REPS
    }
    counts: dict[FormalKey, int] = {}
    for path in sorted(out_root.glob("*/condition_manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = (
                str(payload.get("load_band", "")),
                float(payload["target_rate_gib_s"]),
                int(payload.get("rep", 0)),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"ignoring unreadable formal manifest {path}: {error}")
            continue
        is_formal = str(payload.get("mode", "")).startswith("formal_")
        if is_formal and key in expected:
            counts[key] = counts.get(key, 0) + 1
    return counts


def write_formal_progress(
    out_root: Path,
    results: dict[FormalKey, Path],
    failures: dict[FormalKey, str] | None = None,
    expected: list[FormalKey] | None = None,
) -> Path:
    if expected is None:
        expected = [
            (band, rate, rep)
            for band in BANDS
            for rate in RATES
            for rep in REPS
        ]
    missing = [key for key in expected if key not in results]
    payload = {
        "format_version": 1,
        "phase": "BridgeTP Phase 9 conditional bridge calibration",
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "accepted_conditions": len(results),
        "expected_conditions": len(expected),
        "accepted_results": [
            str(results[key]) for key in expected if key in results
        ],
        "missing_conditions": [
            {"load_band": band, "target_rate_gib_s": rate, "rep": rep}
            for band, rate, rep in missing
        ],
        "failed_conditions": [
            {
                "load_band": band,
                "target_rate_gib_s": rate,
                "rep": rep,
                "last_error": error,
            }
            for (band, rate, rep), error in sorted((failures or {}).items())
        ],
    }
    out = out_root / "formal_progress.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def run_formal(args: argparse.Namespace) -> None:
    pilot = json.loads(args.pilot_summary.read_text(encoding="utf-8"))
    validate_pilot_for_formal(pilot)
    qps_by_band = {
        band: float(pilot["selected"][band]["qps"]) for band in BANDS
    }
    expected = formal_expected_keys(args)
    expected_set = set(expected)
    results = accepted_formal_results(args.out_root) if args.resume else {}
    results = {key: path for key, path in results.items() if key in expected_set}
    prior_attempts = (
        existing_formal_attempt_counts(args.out_root) if args.resume else {}
    )
    if results:
        print(f"resuming with {len(results)} ACCEPTED formal conditions")
    failures: dict[FormalKey, str] = {}
    for band in args.formal_bands:
        for rate in args.formal_rates:
            for rep in args.formal_reps:
                key = (band, rate, rep)
                if key in results:
                    print(f"reusing ACCEPTED condition {key}: {results[key]}")
                    continue
                attempts_used = prior_attempts.get(key, 0)
                if attempts_used >= args.max_attempts:
                    failures[key] = (
                        f"{attempts_used} existing attempts lack an "
                        "ACCEPTED result"
                    )
                for attempt in range(attempts_used + 1, args.max_attempts + 1):
                    try:
                        condition_dir, _ = run_condition(
                            args,
                            prefix=f"formal_{band}",
                            qps=qps_by_band[band],
                            rate=rate,
                            rep=rep,
                            band=band,
                        )
                        results[key] = analyze_formal_condition(
                            args, condition_dir, band, rate
                        )
                        failures.pop(key, None)
                        write_formal_progress(
                            args.out_root, results, failures, expected
                        )
                        break
                    except Exception as error:
                        failures[key] = str(error)
                        print(
                            f"formal condition {key} attempt "
                            f"{attempt}/{args.max_attempts} failed: {error}",
                            file=sys.stderr,
                        )
                        if attempt < args.max_attempts:
                            print(f"retrying formal condition {key}")
                if key not in results and not args.continue_on_error:
                    raise RuntimeError(
                        f"formal condition {key} exhausted retries: "
                        f"{failures[key]}"
                    )
                if key not in results:
                    print(
                        f"continuing after failed formal condition {key}",
                        file=sys.stderr,
                    )
                    write_formal_progress(
                        args.out_root, results, failures, expected
                    )
    progress_out = write_formal_progress(
        args.out_root, results, failures, expected
    )
    if len(results) != len(expected):
        if args.continue_on_error:
            print(
                "completed the formal grid traversal with failed conditions; "
                f"saved incomplete progress to {progress_out}",
                file=sys.stderr,
            )
            return
        raise RuntimeError(
            "formal interference grid remains incomplete after attempting "
            f"all conditions; resume with the same --out-root: {progress_out}"
        )
    full_grid = len(expected) == len(BANDS) * len(RATES) * len(REPS)
    if not full_grid:
        print(
            f"completed {len(expected)} selected formal conditions: "
            f"{progress_out}"
        )
        return
    summary_out = args.out_root / "bridge_calibration_summary.json"
    command = [
        sys.executable,
        str(SUMMARIZER),
        "--inputs",
        *(str(path) for path in results.values()),
        "--out",
        str(summary_out),
        "--expected-bands",
        *BANDS,
        "--expected-rates",
        *(str(rate) for rate in args.formal_rates),
        "--expected-reps",
        *(str(rep) for rep in args.formal_reps),
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
    print(f"completed {len(expected)} formal conditions: {summary_out}")


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
