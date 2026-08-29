#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record interval vLLM telemetry for Phase 9 bridge calibration.

The Prometheus TPOT histogram is cumulative. This recorder stores per-scrape
counter deltas so a later copy-window analysis can reconstruct an exact
window histogram instead of treating a server-lifetime P99 as a tick P99.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.bridge_tp.controller.telemetry import (  # noqa: E402
    has_metric,
    histogram_delta_samples,
    interval_pool_from_samples,
    parse_prometheus,
    request_tpot_metric,
)

KV_METRICS = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)
LOAD_BANDS = ("very_low", "low", "medium", "high")
_STOP = False


def _handle_signal(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8200")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--total-kv-blocks", type=int, required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--load-band", choices=LOAD_BANDS)
    parser.add_argument("--target-rate-gib-s", type=float, required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="stop cleanly after this marker appears",
    )
    args = parser.parse_args()
    if args.interval_s <= 0 or args.max_seconds <= 0:
        parser.error("interval and max-seconds must be positive")
    if args.block_size <= 0 or args.total_kv_blocks <= 0:
        parser.error("KV geometry must be positive")
    if args.target_rate_gib_s < 0:
        parser.error("target rate cannot be negative")
    return args


def fetch_samples(url: str, timeout_s: float):
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        text = response.read().decode("utf-8", errors="replace")
    return parse_prometheus(text)


def histogram_delta_json(previous, current, metric: str | None = None) -> str:
    metric = metric or request_tpot_metric(current) or request_tpot_metric(previous)
    if metric is None:
        return "{}"
    buckets = {}
    for sample in histogram_delta_samples(previous, current, metric):
        bound = sample.labels.get("le")
        if bound is not None:
            buckets[bound] = buckets.get(bound, 0.0) + sample.value
    return json.dumps(buckets, sort_keys=True, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    metrics_url = args.base_url.rstrip("/") + "/metrics"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out.with_suffix(".manifest.json")

    previous = fetch_samples(metrics_url, args.timeout_s)
    tpot_metric = request_tpot_metric(previous)
    kv_metric = next((name for name in KV_METRICS if has_metric(previous, name)), None)
    if kv_metric is None:
        raise RuntimeError(
            "neither vllm:kv_cache_usage_perc nor "
            "vllm:gpu_cache_usage_perc is present"
        )
    required = (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    )
    missing = [name for name in required if not has_metric(previous, name)]
    if missing:
        raise RuntimeError(f"required Phase 9 metrics are missing: {missing}")

    fields = [
        "condition_id",
        "load_band",
        "target_rate_gib_s",
        "rep",
        "unix_s",
        "monotonic_s",
        "interval_s",
        "num_running",
        "num_waiting",
        "kv_usage_frac",
        "preemptions_total",
        "interval_tpot_count",
        "interval_mean_tpot_s",
        "interval_p99_tpot_s",
        "tpot_histogram_delta_json",
    ]
    started_unix = time.time()
    started_monotonic = time.perf_counter()
    rows = 0
    args.out.unlink(missing_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        last_monotonic = started_monotonic
        while not _STOP:
            if args.stop_file is not None and args.stop_file.exists():
                break
            now_monotonic = time.perf_counter()
            if now_monotonic - started_monotonic >= args.max_seconds:
                break
            wait_s = max(0.0, args.interval_s - (now_monotonic - last_monotonic))
            if wait_s:
                time.sleep(wait_s)
            current_monotonic = time.perf_counter()
            current_unix = time.time()
            current = fetch_samples(metrics_url, args.timeout_s)
            if tpot_metric is None:
                # Request-level histograms may be registered lazily.  The
                # first scrape containing one is still a valid delta from an
                # empty baseline because no earlier scrape exposed samples.
                tpot_metric = request_tpot_metric(current)
            pool, count = interval_pool_from_samples(
                previous,
                current,
                block_size=args.block_size,
                total_kv_blocks=args.total_kv_blocks,
                now_unix_s=current_unix,
                tpot_metric=tpot_metric,
            )
            writer.writerow(
                {
                    "condition_id": args.condition_id,
                    "load_band": args.load_band or "",
                    "target_rate_gib_s": args.target_rate_gib_s,
                    "rep": args.rep,
                    "unix_s": current_unix,
                    "monotonic_s": current_monotonic,
                    "interval_s": current_monotonic - last_monotonic,
                    "num_running": pool.num_running,
                    "num_waiting": pool.num_waiting,
                    "kv_usage_frac": pool.kv_usage_frac,
                    "preemptions_total": pool.preemptions_total,
                    "interval_tpot_count": count,
                    "interval_mean_tpot_s": pool.mean_tpot_s,
                    "interval_p99_tpot_s": pool.p99_tpot_s,
                    "tpot_histogram_delta_json": histogram_delta_json(
                        previous, current, tpot_metric
                    ),
                }
            )
            handle.flush()
            rows += 1
            previous = current
            last_monotonic = current_monotonic

    if tpot_metric is None:
        raise RuntimeError(
            "request-level TPOT histogram never appeared; expected "
            "vllm:request_time_per_output_token_seconds or the legacy "
            "vllm:time_per_output_token_seconds"
        )

    manifest = {
        "format_version": 1,
        "phase": "BridgeTP Phase 9 conditional bridge calibration",
        "condition_id": args.condition_id,
        "base_url": args.base_url,
        "metrics_url": metrics_url,
        "kv_metric": kv_metric,
        "tpot_metric": tpot_metric,
        "load_band": args.load_band,
        "target_rate_gib_s": args.target_rate_gib_s,
        "rep": args.rep,
        "block_size": args.block_size,
        "total_kv_blocks": args.total_kv_blocks,
        "started_unix_s": started_unix,
        "started_monotonic_s": started_monotonic,
        "ended_unix_s": time.time(),
        "ended_monotonic_s": time.perf_counter(),
        "rows": rows,
        "csv": str(args.out),
        "evidence_boundary": (
            "TPOT values are Prometheus histogram deltas between adjacent "
            "scrapes. Window aggregation and copy/load compliance require "
            "analyze_phase9_calibration.py."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({rows} interval rows)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
