# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run a pure-Shadow strategy simulation calibrated by C1/C2/C3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from tools.bridge_tp.shadow_copy_strategy import (
        InterferenceCell,
        ShadowTransferInputs,
        compare_shadow_transfers,
        interpolate_tpot,
        kv_bytes_per_token,
    )
except ModuleNotFoundError:
    from shadow_copy_strategy import (
        InterferenceCell,
        ShadowTransferInputs,
        compare_shadow_transfers,
        interpolate_tpot,
        kv_bytes_per_token,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--interference-inventory", type=Path, required=True)
    parser.add_argument("--tpot-model", type=Path, required=True)
    parser.add_argument("--survival-table", type=Path, required=True)
    parser.add_argument(
        "--history-tokens",
        type=int,
        nargs="+",
        default=[128, 256, 512, 768, 1024, 1536, 2048],
    )
    parser.add_argument(
        "--remaining-quantiles", type=float, nargs="+", default=[0.5, 0.9]
    )
    parser.add_argument(
        "--source-load-fracs", type=float, nargs="+", default=[0.22, 0.48, 0.62]
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def empirical_quantile(values: list[int], quantile: float) -> int:
    """Return the observed higher quantile without distribution fitting."""
    if not 0 < quantile <= 1 or not values:
        raise ValueError("quantile and observations are invalid")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def load_remaining_cases(
    path: Path,
    history_tokens: list[int],
    quantiles: list[float],
) -> list[tuple[int, float, int, int]]:
    raw = read_json(path)
    by_prefix = dict(zip(raw["bucket_edges"], raw["remaining"], strict=True))
    cases = []
    for prefix in history_tokens:
        if prefix not in by_prefix:
            raise ValueError(f"prefix {prefix} is absent from survival table")
        values = by_prefix[prefix]
        for quantile in quantiles:
            cases.append(
                (
                    prefix,
                    quantile,
                    empirical_quantile(values, quantile),
                    len(values),
                )
            )
    return cases


def load_interference_cells(path: Path) -> list[InterferenceCell]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    baselines = {
        (row["load_band"], int(row["rep"])): row
        for row in rows
        if float(row["target_rate_gib_s"]) == 0
    }
    cells = []
    for row in rows:
        target_rate = float(row["target_rate_gib_s"])
        if target_rate == 0:
            continue
        key = (row["load_band"], int(row["rep"]))
        if key not in baselines:
            raise ValueError(f"missing paired C2 baseline for {key}")
        baseline = baselines[key]
        cells.append(
            InterferenceCell(
                load_band=row["load_band"],
                repetition=int(row["rep"]),
                target_rate_gib_s=target_rate,
                effective_rate_gib_s=float(row["effective_rate_gib_s"]),
                target_load_frac=float(row["kv_usage_mean"]),
                baseline_mean_tpot_s=float(baseline["mean_tpot_s"]),
                copy_mean_tpot_s=float(row["mean_tpot_s"]),
                baseline_p99_tpot_s=float(baseline["p99_tpot_s"]),
                copy_p99_tpot_s=float(row["p99_tpot_s"]),
                baseline_p99_itl_s=float(baseline["p99_itl_s"]),
                copy_p99_itl_s=float(row["p99_itl_s"]),
            )
        )
    if len(cells) != 36:
        raise ValueError(f"expected 36 paired nonzero C2 cells, got {len(cells)}")
    return cells


def flatten(
    comparison_id: int,
    remaining_quantile: float,
    survivor_count: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    inputs = result["inputs"]
    history = result["history_backfill"]
    new_only = result["new_kv_only"]
    return {
        "comparison_id": comparison_id,
        "remaining_quantile": remaining_quantile,
        "survivor_count": survivor_count,
        **inputs,
        "history_takeover_ready": history["takeover_ready"],
        "history_ready_time_s": history["takeover_ready_time_s"],
        "history_copy_active_time_s": history["copy_active_time_s"],
        "new_only_copy_active_time_s": new_only["copy_active_time_s"],
        "history_bytes_sent": history["bytes_sent"],
        "new_only_bytes_sent": new_only["bytes_sent"],
        "history_backlog_end_tokens": history["history_backlog_end_tokens"],
        "new_only_history_backlog_tokens": (
            new_only["history_backlog_end_tokens"]
        ),
        "new_only_copy_duty_cycle": new_only["copy_duty_cycle"],
        "new_only_maximum_kv_lag_s": new_only["maximum_new_kv_lag_s"],
        "history_estimated_target_delay_s": (
            history["estimated_target_delay_s"]
        ),
        "new_only_estimated_target_delay_s": (
            new_only["estimated_target_delay_s"]
        ),
        "history_outcome": history["outcome"],
        "new_only_outcome": new_only["outcome"],
        "lower_bytes": result["lower_bytes"],
        "lower_estimated_interference": result[
            "lower_estimated_interference"
        ],
        "takeover_ready": result["takeover_ready"],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric_summary(
    rows: list[dict[str, Any]], field: str, scale: float = 1.0
) -> dict[str, float]:
    values = sorted(float(row[field]) * scale for row in rows)
    p90_index = max(0, math.ceil(0.9 * len(values)) - 1)
    return {
        "minimum": values[0],
        "median": statistics.median(values),
        "p90": values[p90_index],
        "maximum": values[-1],
    }


def grouped_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    result = {}
    for value in sorted({row[field] for row in rows}):
        selected = [row for row in rows if row[field] == value]
        ready = sum(bool(row["history_takeover_ready"]) for row in selected)
        result[str(value)] = {
            "comparisons": len(selected),
            "history_takeover_ready": ready,
            "history_takeover_ready_fraction": ready / len(selected),
            "history_ready_time_ms": numeric_summary(
                selected, "history_ready_time_s", 1000
            ),
            "new_only_copy_duty_cycle": numeric_summary(
                selected, "new_only_copy_duty_cycle"
            ),
        }
    return result


def prefix_quantile_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    keys = sorted(
        {
            (int(row["history_tokens"]), float(row["remaining_quantile"]))
            for row in rows
        }
    )
    for history, quantile in keys:
        selected = [
            row
            for row in rows
            if int(row["history_tokens"]) == history
            and float(row["remaining_quantile"]) == quantile
        ]
        new_lower = sum(row["lower_bytes"] == "new_kv_only" for row in selected)
        result[f"history_{history}_q{quantile:g}"] = {
            "remaining_tokens": int(selected[0]["remaining_tokens"]),
            "comparisons": len(selected),
            "history_ready_time_ms": numeric_summary(
                selected, "history_ready_time_s", 1000
            ),
            "new_only_lower_bytes_fraction": new_lower / len(selected),
        }
    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    tpot_raw = read_json(args.tpot_model)["tpot_tp1"]
    remaining_cases = load_remaining_cases(
        args.survival_table,
        args.history_tokens,
        args.remaining_quantiles,
    )
    cells = load_interference_cells(args.interference_inventory)
    kv_bytes = kv_bytes_per_token(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        dtype_bytes=args.dtype_bytes,
    )
    rows = []
    full_results = []
    comparison_id = 0
    for source_load in args.source_load_fracs:
        source_tpot = interpolate_tpot(
            source_load,
            tpot_raw["load_knots"],
            tpot_raw["tpot_knots_s"],
        )
        for history, quantile, remaining, survivors in remaining_cases:
            for cell in cells:
                comparison_id += 1
                inputs = ShadowTransferInputs(
                    history_tokens=history,
                    remaining_tokens=remaining,
                    block_size=args.block_size,
                    kv_bytes_per_token=kv_bytes,
                    source_load_frac=source_load,
                    source_tpot_s=source_tpot,
                    interference=cell,
                )
                result = compare_shadow_transfers(inputs)
                full_results.append(result)
                rows.append(
                    flatten(comparison_id, quantile, survivors, result)
                )

    write_csv(rows, args.out_dir / "shadow_transfer_decisions.csv")
    (args.out_dir / "shadow_transfer_comparisons.json").write_text(
        json.dumps(full_results, indent=2) + "\n", encoding="utf-8"
    )
    ready = sum(bool(row["history_takeover_ready"]) for row in rows)
    new_lower_bytes = sum(
        row["lower_bytes"] == "new_kv_only" for row in rows
    )
    new_lower_interference = sum(
        row["lower_estimated_interference"] == "new_kv_only"
        for row in rows
    )
    summary = {
        "format_version": 2,
        "model_scope": "pure Shadow KV transfer; no remote attention",
        "comparisons": len(rows),
        "calibration": {
            "c2_nonzero_cells": len(cells),
            "source_load_cases": args.source_load_fracs,
            "history_prefixes": args.history_tokens,
            "remaining_quantiles": args.remaining_quantiles,
            "kv_bytes_per_token": kv_bytes,
            "input_sha256": {
                "interference_inventory": sha256(
                    args.interference_inventory
                ),
                "tpot_model": sha256(args.tpot_model),
                "survival_table": sha256(args.survival_table),
            },
        },
        "history_backfill": {
            "takeover_ready": ready,
            "takeover_ready_fraction": ready / len(rows),
            "source_finished_first": len(rows) - ready,
            "ready_time_ms": numeric_summary(
                rows, "history_ready_time_s", 1000
            ),
            "bytes_sent_mib": numeric_summary(
                rows, "history_bytes_sent", 1 / 1024**2
            ),
            "estimated_target_delay_ms": numeric_summary(
                rows, "history_estimated_target_delay_s", 1000
            ),
        },
        "new_kv_only": {
            "takeover_ready": 0,
            "history_is_never_transferred": True,
            "lower_bytes": new_lower_bytes,
            "lower_bytes_fraction": new_lower_bytes / len(rows),
            "lower_estimated_interference": new_lower_interference,
            "lower_estimated_interference_fraction": (
                new_lower_interference / len(rows)
            ),
            "bytes_sent_mib": numeric_summary(
                rows, "new_only_bytes_sent", 1 / 1024**2
            ),
            "copy_duty_cycle": numeric_summary(
                rows, "new_only_copy_duty_cycle"
            ),
            "maximum_kv_lag_ms": numeric_summary(
                rows, "new_only_maximum_kv_lag_s", 1000
            ),
            "estimated_target_delay_ms": numeric_summary(
                rows, "new_only_estimated_target_delay_s", 1000
            ),
        },
        "by_copy_rate_gib_s": grouped_summary(rows, "target_rate_gib_s"),
        "by_remaining_quantile": grouped_summary(
            rows, "remaining_quantile"
        ),
        "by_prefix_and_remaining_quantile": prefix_quantile_summary(rows),
        "evidence_boundary": (
            "C1/C2/C3-calibrated deterministic simulation on A100 PCIe. "
            "C2 measures continuous copy interference; new-only burst "
            "impact is an exposure-scaled estimate, not a new GPU result."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
