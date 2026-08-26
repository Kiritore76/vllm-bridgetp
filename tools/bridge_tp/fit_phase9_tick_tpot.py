#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fit workload-scoped TPOT models from Phase 9 interval telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp1", type=Path, nargs="+", required=True)
    parser.add_argument("--tp4", type=Path, nargs="+", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, float]]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                count = int(float(raw["interval_tpot_count"]))
                running = int(float(raw["num_running"]))
                mean_tpot = float(raw["interval_mean_tpot_s"])
                if count <= 0 or running <= 0 or mean_tpot <= 0:
                    continue
                rows.append(
                    {
                        "num_running": running,
                        "mean_tpot_s": mean_tpot,
                        "weight": count,
                    }
                )
    return rows


def weighted_fit(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if len(rows) < 3:
        raise ValueError("at least three non-idle telemetry intervals are required")
    total_weight = sum(row["weight"] for row in rows)
    x_mean = sum(
        row["weight"] * row["num_running"] for row in rows
    ) / total_weight
    y_mean = sum(
        row["weight"] * row["mean_tpot_s"] for row in rows
    ) / total_weight
    denominator = sum(
        row["weight"] * (row["num_running"] - x_mean) ** 2 for row in rows
    )
    if denominator <= 0:
        raise ValueError("num_running has no variation")
    slope = sum(
        row["weight"]
        * (row["num_running"] - x_mean)
        * (row["mean_tpot_s"] - y_mean)
        for row in rows
    ) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum(
        row["weight"]
        * (row["mean_tpot_s"] - (intercept + slope * row["num_running"])) ** 2
        for row in rows
    )
    total = sum(
        row["weight"] * (row["mean_tpot_s"] - y_mean) ** 2 for row in rows
    )
    return {
        "base_s": intercept,
        "per_running_s": slope,
        "intervals": len(rows),
        "tpot_observations": int(total_weight),
        "weighted_r_squared": 1.0 - residual / total if total > 0 else 1.0,
        "weighted_rmse_s": math.sqrt(residual / total_weight),
        "num_running_min": min(int(row["num_running"]) for row in rows),
        "num_running_max": max(int(row["num_running"]) for row in rows),
    }


def main() -> None:
    args = parse_args()
    tp1_rows = read_rows(args.tp1)
    tp4_rows = read_rows(args.tp4)
    payload = {
        "format_version": 1,
        "status": "WORKLOAD_SCOPED_TICK_CANDIDATE",
        "platform": "NVIDIA A100 PCIe",
        "scope": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "response": "interval mean TPOT",
            "predictor": "instantaneous vLLM num_requests_running",
        },
        "tpot_tp1": weighted_fit(tp1_rows),
        "tpot_tp4": weighted_fit(tp4_rows),
        "sources": {
            "tp1": [str(path) for path in args.tp1],
            "tp4": [str(path) for path in args.tp4],
        },
        "evidence_boundary": (
            "The fit is workload- and platform-scoped. It replaces the "
            "run-level max-concurrency proxy only after its frozen input "
            "manifests and hashes are retained."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
