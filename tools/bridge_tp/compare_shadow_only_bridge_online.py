#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare successful formal Shadow-only and online-Bridge measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "handoff_stall_ms",
    "bridge_to_commit_ms",
    "anchor_tpot_p50_ms",
    "anchor_tpot_p95_ms",
    "anchor_tpot_p99_ms",
    "shadow_tpot_p99_ms",
    "bridge_tpot_p99_ms",
    "post_commit_tpot_p99_ms",
    "output_throughput_tokens_s",
    "slo_tpot_interval_violation_rate",
    "source_process_wall_time_s",
    "target_process_wall_time_s",
    "stager_process_wall_time_s",
    "controller_process_wall_time_s",
    "remote_attention_p50_ms",
    "remote_attention_p95_ms",
    "remote_attention_p99_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-only-root", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(root: Path) -> list[dict[str, str]]:
    candidates = sorted(root.glob("**/measurements.csv"))
    if len(candidates) != 1:
        raise ValueError(f"expected one measurements.csv below {root}, found {candidates}")
    with candidates[0].open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "PASS"]
    if not rows:
        raise ValueError(f"no successful formal rows in {candidates[0]}")
    return rows


def numbers(rows: list[dict[str, str]], metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if row.get(metric) not in {None, ""}]


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "stdev": None, "ci95": None}
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": stdev,
        "ci95": 1.96 * stdev / math.sqrt(len(values)),
    }


def main() -> None:
    args = parse_args()
    shadow = load_rows(args.shadow_only_root)
    bridge = load_rows(args.bridge_root)
    report: dict[str, Any] = {
        "format_version": 1,
        "shadow_only_runs": len(shadow),
        "bridge_runs": len(bridge),
        "metrics": {},
    }
    flat: list[dict[str, Any]] = []
    for metric in METRICS:
        shadow_summary = summary(numbers(shadow, metric))
        bridge_summary = summary(numbers(bridge, metric))
        delta = (
            float(bridge_summary["mean"]) - float(shadow_summary["mean"])
            if bridge_summary["mean"] is not None
            and shadow_summary["mean"] is not None
            else None
        )
        report["metrics"][metric] = {
            "shadow_only": shadow_summary,
            "bridge": bridge_summary,
            "bridge_minus_shadow_only": delta,
        }
        flat.append(
            {
                "metric": metric,
                "shadow_only_mean": shadow_summary["mean"],
                "bridge_mean": bridge_summary["mean"],
                "bridge_minus_shadow_only": delta,
                "shadow_only_n": shadow_summary["n"],
                "bridge_n": bridge_summary["n"],
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
