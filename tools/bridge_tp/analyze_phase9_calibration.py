#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate one Phase 9 bridge-calibration load/rate condition."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--copy-json", type=Path)
    parser.add_argument("--window-start-monotonic-s", type=float)
    parser.add_argument("--window-end-monotonic-s", type=float)
    parser.add_argument("--target-rate-gib-s", type=float, required=True)
    parser.add_argument("--load-min", type=float, required=True)
    parser.add_argument("--load-max", type=float, required=True)
    parser.add_argument("--min-band-fraction", type=float, default=0.80)
    parser.add_argument("--rate-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.load_min < args.load_max <= 1:
        parser.error("load band must satisfy 0 <= min < max <= 1")
    if not 0 <= args.min_band_fraction <= 1:
        parser.error("min-band-fraction must be in [0,1]")
    if args.target_rate_gib_s < 0:
        parser.error("target rate cannot be negative")
    if args.target_rate_gib_s > 0 and args.copy_json is None:
        parser.error("nonzero rate requires --copy-json")
    if args.target_rate_gib_s == 0 and args.copy_json is None:
        if (
            args.window_start_monotonic_s is None
            or args.window_end_monotonic_s is None
        ):
            parser.error("rate=0 requires --copy-json or an explicit window")
    return args


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def copy_window(args: argparse.Namespace) -> tuple[float, float, float | None]:
    if args.copy_json is None:
        if (
            args.window_start_monotonic_s is None
            or args.window_end_monotonic_s is None
        ):
            raise ValueError("explicit analysis window is incomplete")
        return (
            args.window_start_monotonic_s,
            args.window_end_monotonic_s,
            None,
        )
    raw = json.loads(args.copy_json.read_text(encoding="utf-8"))
    start = raw.get("start_perf_counter", raw.get("start_monotonic_s"))
    end = raw.get("end_perf_counter", raw.get("end_monotonic_s"))
    effective = raw.get("effective_gib_s")
    if start is None or end is None or effective is None:
        raise ValueError("copy JSON lacks start/end monotonic time or effective rate")
    return float(start), float(end), float(effective)


def aggregate_histogram(rows: list[dict[str, str]]) -> tuple[int, float, float]:
    buckets: dict[float, float] = {}
    count = 0
    total = 0.0
    for row in rows:
        interval_count = int(float(row["interval_tpot_count"]))
        count += interval_count
        total += float(row["interval_mean_tpot_s"]) * interval_count
        for bound, value in json.loads(row["tpot_histogram_delta_json"]).items():
            numeric = math.inf if bound in ("+Inf", "Inf") else float(bound)
            buckets[numeric] = buckets.get(numeric, 0.0) + float(value)
    if count <= 0:
        return 0, 0.0, 0.0
    target = 0.99 * count
    previous_bound = 0.0
    previous_count = 0.0
    p99 = 0.0
    for bound, cumulative in sorted(buckets.items()):
        if cumulative >= target:
            if math.isinf(bound):
                p99 = previous_bound
            else:
                span = cumulative - previous_count
                fraction = 1.0 if span <= 0 else (target - previous_count) / span
                p99 = previous_bound + fraction * (bound - previous_bound)
            break
        previous_bound = bound
        previous_count = cumulative
    return count, total / count, p99


def window_itls(benchmark: dict, start: float, end: float) -> list[float]:
    starts = benchmark.get("start_times") or []
    ttfts = benchmark.get("ttfts") or []
    all_itls = benchmark.get("itls") or []
    out: list[float] = []
    for request_start, ttft, itls in zip(starts, ttfts, all_itls):
        token_end = float(request_start) + float(ttft)
        for itl in itls or []:
            token_end += float(itl)
            if start <= token_end <= end:
                out.append(float(itl))
    return out


def main() -> None:
    args = parse_args()
    start, end, effective_rate = copy_window(args)
    if end <= start:
        raise ValueError("analysis window must have positive duration")
    with args.telemetry.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    if not all_rows:
        raise ValueError("telemetry CSV is empty")
    condition_id = all_rows[0].get("condition_id", "")
    load_band_name = all_rows[0].get("load_band", "")
    rep = int(all_rows[0].get("rep", "0"))
    rows = [
        row
        for row in all_rows
        if start
        <= float(row["monotonic_s"]) - float(row["interval_s"])
        <= float(row["monotonic_s"])
        <= end
    ]
    if not rows:
        raise ValueError("no telemetry samples overlap the analysis window")

    kv = [float(row["kv_usage_frac"]) for row in rows]
    in_band = [args.load_min <= value <= args.load_max for value in kv]
    band_fraction = sum(in_band) / len(in_band)
    tpot_count, mean_tpot, p99_tpot = aggregate_histogram(rows)
    benchmark = json.loads(args.benchmark_json.read_text(encoding="utf-8"))
    itls = window_itls(benchmark, start, end)

    load_pass = (
        args.load_min <= statistics.mean(kv) <= args.load_max
        and band_fraction >= args.min_band_fraction
    )
    if args.target_rate_gib_s == 0:
        rate_error = effective_rate
        rate_pass = effective_rate in (None, 0.0)
    else:
        if effective_rate is None:
            raise ValueError("nonzero target rate lacks an effective rate")
        rate_error = (
            abs(effective_rate - args.target_rate_gib_s)
            / args.target_rate_gib_s
        )
        rate_pass = rate_error <= args.rate_relative_tolerance
    checks = {
        "telemetry_window_nonempty": bool(rows),
        "load_band_compliant": load_pass,
        "copy_rate_compliant": rate_pass,
        "tpot_observations_present": tpot_count > 0,
        "itl_observations_present": bool(itls),
    }
    payload = {
        "format_version": 1,
        "phase": "BridgeTP Phase 9 conditional bridge calibration",
        "status": "ACCEPTED" if all(checks.values()) else "REJECTED",
        "platform_scope": "NVIDIA A100 PCIe only",
        "condition_id": condition_id,
        "load_band": load_band_name,
        "rep": rep,
        "inputs": {
            "telemetry": str(args.telemetry),
            "benchmark_json": str(args.benchmark_json),
            "copy_json": str(args.copy_json) if args.copy_json else None,
            "window_start_monotonic_s": start,
            "window_end_monotonic_s": end,
            "target_rate_gib_s": args.target_rate_gib_s,
            "load_band": [args.load_min, args.load_max],
        },
        "observed": {
            "telemetry_samples": len(rows),
            "kv_usage_mean": statistics.mean(kv),
            "kv_usage_p50": percentile(kv, 0.50),
            "kv_usage_p95": percentile(kv, 0.95),
            "load_band_fraction": band_fraction,
            "effective_rate_gib_s": effective_rate,
            "rate_relative_error": rate_error,
            "tpot_count": tpot_count,
            "mean_tpot_s": mean_tpot,
            "p99_tpot_s": p99_tpot,
            "itl_count": len(itls),
            "p95_itl_s": percentile(itls, 0.95),
            "p99_itl_s": percentile(itls, 0.99),
        },
        "checks": checks,
        "evidence_boundary": (
            "A condition is accepted only when measured KV occupancy and "
            "effective copy rate satisfy their preregistered ranges. This "
            "calibration does not itself prove D-group migration fidelity."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    if payload["status"] != "ACCEPTED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
