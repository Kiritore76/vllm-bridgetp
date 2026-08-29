#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fit workload-scoped TPOT models from Phase 9 interval telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp1", type=Path, nargs="+", required=True)
    parser.add_argument("--tp4", type=Path, nargs="+", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument(
        "--tp4-baseline-inventory",
        type=Path,
        help=(
            "Section 7.2 inventory whose rate=0 cells extend native TP4 "
            "KV-load support; enables the formal load-aware model"
        ),
    )
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


def read_condition_observations(paths: list[Path]) -> list[dict[str, Any]]:
    observations = []
    for telemetry in paths:
        benchmark = telemetry.parent / "benchmark.json"
        if not benchmark.exists():
            raise ValueError(f"benchmark sibling missing for {telemetry}")
        with telemetry.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"empty telemetry file: {telemetry}")
        result = json.loads(benchmark.read_text(encoding="utf-8"))
        observations.append(
            {
                "load": sum(float(row["kv_usage_frac"]) for row in rows) / len(rows),
                "tpot_s": float(result["mean_tpot_ms"]) / 1000.0,
                "source": str(telemetry),
            }
        )
    return observations


def read_tp4_baselines(path: Path) -> list[dict[str, Any]]:
    observations = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["target_rate_gib_s"]) != 0.0:
                continue
            observations.append(
                {
                    "load": float(row["kv_usage_mean"]),
                    "tpot_s": float(row["mean_tpot_s"]),
                    "source": f"{path}:{row['condition_id']}",
                }
            )
    if len(observations) != 9:
        raise ValueError(f"expected 9 TP4 rate-zero baselines, got {len(observations)}")
    return observations


def monotone_load_fit(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if len(observations) < 3:
        raise ValueError("at least three load/TPOT observations are required")
    grouped: list[dict[str, float]] = []
    for row in sorted(observations, key=lambda item: item["load"]):
        if grouped and row["load"] == grouped[-1]["load"]:
            grouped[-1]["sum_tpot"] += row["tpot_s"]
            grouped[-1]["weight"] += 1.0
        else:
            grouped.append(
                {
                    "load": row["load"],
                    "sum_tpot": row["tpot_s"],
                    "weight": 1.0,
                }
            )
    blocks: list[dict[str, Any]] = []
    for index, row in enumerate(grouped):
        blocks.append(
            {
                "indices": [index],
                "sum_tpot": row["sum_tpot"],
                "weight": row["weight"],
            }
        )
        while len(blocks) > 1:
            left, right = blocks[-2], blocks[-1]
            left_mean = left["sum_tpot"] / left["weight"]
            right_mean = right["sum_tpot"] / right["weight"]
            if left_mean <= right_mean:
                break
            blocks[-2:] = [
                {
                    "indices": left["indices"] + right["indices"],
                    "sum_tpot": left["sum_tpot"] + right["sum_tpot"],
                    "weight": left["weight"] + right["weight"],
                }
            ]
    predictions = [0.0] * len(grouped)
    for block in blocks:
        value = block["sum_tpot"] / block["weight"]
        for index in block["indices"]:
            predictions[index] = value
    actual = [row["tpot_s"] for row in observations]
    mean = sum(actual) / len(actual)

    def interpolate(load: float) -> float:
        if load <= grouped[0]["load"]:
            return predictions[0]
        for index in range(1, len(grouped)):
            if load <= grouped[index]["load"]:
                x0, x1 = grouped[index - 1]["load"], grouped[index]["load"]
                y0, y1 = predictions[index - 1], predictions[index]
                fraction = (load - x0) / (x1 - x0)
                return y0 + fraction * (y1 - y0)
        return predictions[-1]

    residual = sum(
        (row["tpot_s"] - interpolate(row["load"])) ** 2 for row in observations
    )
    total = sum((value - mean) ** 2 for value in actual)
    return {
        "base_s": predictions[0],
        "per_running_s": 0.0,
        "model_kind": "load_piecewise_monotone",
        "load_knots": [row["load"] for row in grouped],
        "tpot_knots_s": predictions,
        "min_load_frac": grouped[0]["load"],
        "max_load_frac": grouped[-1]["load"],
        "num_running_min": 0,
        "num_running_max": 1_000_000_000,
        "conditions": len(observations),
        "r_squared": 1.0 - residual / total if total > 0 else 1.0,
        "rmse_s": math.sqrt(residual / len(observations)),
    }


def main() -> None:
    args = parse_args()
    tp1_rows = read_rows(args.tp1)
    tp4_rows = read_rows(args.tp4)
    if args.tp4_baseline_inventory:
        tp1_observations = read_condition_observations(args.tp1)
        tp4_observations = read_condition_observations(args.tp4)
        tp4_observations.extend(read_tp4_baselines(args.tp4_baseline_inventory))
        tp1_model = monotone_load_fit(tp1_observations)
        tp4_model = monotone_load_fit(tp4_observations)
        status = "WORKLOAD_SCOPED_LOAD_TPOT_CANDIDATE"
        predictor = "runtime vLLM kv_usage_frac"
    else:
        tp1_model = weighted_fit(tp1_rows)
        tp4_model = weighted_fit(tp4_rows)
        status = "WORKLOAD_SCOPED_TICK_CANDIDATE"
        predictor = "instantaneous vLLM num_requests_running"
    payload = {
        "format_version": 1,
        "status": status,
        "platform": "NVIDIA A100 PCIe",
        "scope": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "response": "interval mean TPOT",
            "predictor": predictor,
        },
        "tpot_tp1": tp1_model,
        "tpot_tp4": tp4_model,
        "running_linear_diagnostic": {
            "tpot_tp1": weighted_fit(tp1_rows),
            "tpot_tp4": weighted_fit(tp4_rows),
        },
        "sources": {
            "tp1": [str(path) for path in args.tp1],
            "tp4": [str(path) for path in args.tp4],
            "tp4_baseline_inventory": (
                str(args.tp4_baseline_inventory)
                if args.tp4_baseline_inventory
                else None
            ),
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
