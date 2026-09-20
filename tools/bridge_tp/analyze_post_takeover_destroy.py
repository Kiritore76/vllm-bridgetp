# SPDX-License-Identifier: Apache-2.0
"""Correlate post-takeover communicator destruction with token gaps."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_one(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def interval_summary(values: list[float], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_samples": len(values),
        f"{prefix}_p50_ms": percentile(values, 0.50),
        f"{prefix}_p95_ms": percentile(values, 0.95),
        f"{prefix}_p99_ms": percentile(values, 0.99),
        f"{prefix}_max_ms": max(values) if values else None,
        f"{prefix}_gt50": sum(value > 50 for value in values),
        f"{prefix}_gt100": sum(value > 100 for value in values),
        f"{prefix}_gt200": sum(value > 200 for value in values),
    }


def analyze_run(run: dict[str, Any], out_root: Path) -> dict[str, Any]:
    root = Path(str(run["root"]))
    if not root.is_absolute():
        root = out_root / root
    if not root.exists():
        copied = [path for path in out_root.rglob(root.name) if path.is_dir()]
        if len(copied) == 1:
            root = copied[0]
    proxy_path = find_one(root, "response_proxy_stats.json")
    if proxy_path is None:
        raise FileNotFoundError(f"response_proxy_stats.json missing below {root}")
    proxy = read_json(proxy_path)
    target = [
        row for row in proxy.get("emitted", []) if row.get("origin") == "target"
    ]
    intervals = [
        {
            "start": float(first["unix_s"]),
            "end": float(second["unix_s"]),
            "gap_ms": (float(second["unix_s"]) - float(first["unix_s"]))
            * 1000,
        }
        for first, second in zip(target, target[1:])
    ]
    first_target = float(target[0]["unix_s"]) if target else None
    first_second = (
        [
            row["gap_ms"]
            for row in intervals
            if row["start"] < first_target + 1.0
        ]
        if first_target is not None
        else []
    )

    receipts = [
        read_json(path)
        for path in sorted(
            proxy_path.parent.glob(
                "gpu_communicator_destroy_receipts/tp_rank_*.json"
            )
        )
    ]
    destroy_starts = [
        float(row["destroy_started_unix_s"])
        for row in receipts
        if row.get("destroy_started_unix_s") is not None
    ]
    destroy_ends = [
        float(row["destroy_completed_unix_s"])
        for row in receipts
        if row.get("destroy_completed_unix_s") is not None
    ]
    destroy_start = min(destroy_starts) if destroy_starts else None
    destroy_end = max(destroy_ends) if destroy_ends else None
    during_destroy = (
        [
            row["gap_ms"]
            for row in intervals
            if row["end"] >= destroy_start and row["start"] <= destroy_end
        ]
        if destroy_start is not None and destroy_end is not None
        else []
    )
    after_destroy = (
        [row["gap_ms"] for row in intervals if row["start"] > destroy_end]
        if destroy_end is not None
        else []
    )
    acceptance = run.get("acceptance") or {}
    return {
        "repetition": run.get("repetition"),
        "architecture": run.get("architecture"),
        "post_takeover_comm_destroy": run.get(
            "post_takeover_comm_destroy", False
        ),
        "status": run.get("status"),
        "handoff_stall_ms": acceptance.get("handoff_stall_ms"),
        "destroy_window_ms": (
            (destroy_end - destroy_start) * 1000
            if destroy_start is not None and destroy_end is not None
            else None
        ),
        "destroy_overlap_after_first_target_ms": (
            max(0.0, (destroy_end - first_target) * 1000)
            if destroy_end is not None and first_target is not None
            else None
        ),
        **interval_summary(first_second, "first_second"),
        **interval_summary(during_destroy, "during_destroy"),
        **interval_summary(after_destroy, "after_destroy"),
        "root": str(root),
    }


def mean_or_none(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    batch = read_json(args.out_root / "batch_status.json")
    rows = [analyze_run(run, args.out_root) for run in batch.get("runs", [])]
    if not rows:
        raise RuntimeError("batch_status.json contains no runs")

    csv_path = args.out_root / "post_takeover_destroy_token_gaps.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["architecture"]), []).append(row)
    fields = [
        "handoff_stall_ms",
        "destroy_window_ms",
        "destroy_overlap_after_first_target_ms",
        "first_second_p95_ms",
        "first_second_p99_ms",
        "first_second_max_ms",
        "first_second_gt100",
        "during_destroy_p99_ms",
        "during_destroy_max_ms",
        "during_destroy_gt100",
    ]
    summary = {
        "format_version": 1,
        "status": "COMPLETE",
        "runs": len(rows),
        "groups": {
            name: {
                "runs": len(selected),
                **{
                    f"{field}_mean": mean_or_none(selected, field)
                    for field in fields
                },
            }
            for name, selected in groups.items()
        },
    }
    (args.out_root / "post_takeover_destroy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
