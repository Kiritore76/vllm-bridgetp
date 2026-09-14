#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate GPU-direct delta batch smoke runs and select a candidate."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    return float(value) if value not in {"", "None"} else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    for path in sorted(args.root.rglob("measurements.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["measurement_file"] = str(path)
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no measurements.csv below {args.root}")

    grouped: dict[tuple[int, float], list[dict[str, str]]] = {}
    for row in rows:
        batch = int(float(row["gpu_direct_delta_batch_tokens"]))
        flush = float(row["gpu_direct_delta_flush_ms"])
        grouped.setdefault((batch, flush), []).append(row)

    summary: list[dict[str, Any]] = []
    for (batch, flush), values in sorted(grouped.items()):
        passed = [row for row in values if row.get("status") == "PASS"]
        def numbers(key: str) -> list[float]:
            return [value for row in passed if (value := _number(row, key)) is not None]

        summary.append(
            {
                "batch_tokens": batch,
                "flush_ms": flush,
                "runs": len(values),
                "passes": len(passed),
                "handoff_stall_ms_mean": _mean(numbers("handoff_stall_ms")),
                "final_sync_ms_mean": _mean(numbers("final_sync_to_commit_ms")),
                "freeze_stall_ms_mean": _mean(numbers("freeze_boundary_stall_ms")),
                "delta_total_ms_mean": _mean(numbers("gpu_direct_delta_total_ms")),
                "delta_max_batch_ms_mean": _mean(
                    numbers("gpu_direct_delta_max_batch_ms")
                ),
                "anchor_tpot_p99_ms_mean": _mean(numbers("anchor_tpot_p99_ms")),
                "shadow_target_p99_ms_mean": _mean(
                    numbers("shadow_tpot_p99_ms")
                ),
                "tpot_violation_rate_mean": _mean(
                    numbers("slo_tpot_interval_violation_rate")
                ),
            }
        )

    eligible = [
        row for row in summary
        if row["passes"] == row["runs"]
        and row["handoff_stall_ms_mean"] is not None
    ]
    recommended = min(
        eligible,
        key=lambda row: (
            row["handoff_stall_ms_mean"],
            row["anchor_tpot_p99_ms_mean"] or float("inf"),
            row["batch_tokens"],
        ),
        default=None,
    )
    result = {
        "format_version": 1,
        "status": "PASS" if recommended else "INCOMPLETE",
        "selection_rule": (
            "all runs pass; minimize mean handoff stall, then anchor TPOT p99"
        ),
        "recommended": recommended,
        "candidates": summary,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "gpu_direct_delta_batch_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = list(summary[0])
    with (args.root / "gpu_direct_delta_batch_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
