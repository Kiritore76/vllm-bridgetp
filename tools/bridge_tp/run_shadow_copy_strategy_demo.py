# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare newest-history-first Shadow copy with a new-KV-only bridge."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from tools.bridge_tp.shadow_copy_strategy import (
        ShadowCopyInputs,
        compare_shadow_strategies,
        kv_bytes_per_token,
        remote_attention_bytes_per_token,
    )
except ModuleNotFoundError:
    from shadow_copy_strategy import (
        ShadowCopyInputs,
        compare_shadow_strategies,
        kv_bytes_per_token,
        remote_attention_bytes_per_token,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--history-tokens", type=int, nargs="+", default=[512, 1024])
    parser.add_argument(
        "--remaining-tokens", type=int, nargs="+", default=[128, 256, 512, 1024]
    )
    parser.add_argument(
        "--copy-rate-gib-s", type=float, nargs="+", default=[0.4, 0.7, 1.2]
    )
    parser.add_argument(
        "--remote-penalty-ms",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0, 8.0],
        help="Added TP4 decode time per token for TP1 remote attention.",
    )
    parser.add_argument("--source-tpot-ms", type=float, default=30.0)
    parser.add_argument("--target-tpot-ms", type=float, default=16.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--bridge-start-tokens", type=int, default=1)
    return parser.parse_args()


def flatten(comparison_id: int, result: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = result["inputs"]
    common = {
        "comparison_id": comparison_id,
        **inputs,
        "latency_winner": result["latency_winner"],
        "source_release_winner": result["source_release_winner"],
        "network_bytes_winner": result["network_bytes_winner"],
        "remote_penalty_break_even_ms": result[
            "new_kv_bridge_remote_penalty_break_even_ms"
        ],
    }
    return [
        {**common, **result[strategy]}
        for strategy in ("history_backfill", "new_kv_bridge")
    ]


def flatten_decision(
    comparison_id: int, result: dict[str, Any]
) -> dict[str, Any]:
    """Return one paired row for direct strategy comparison."""
    inputs = result["inputs"]
    history = result["history_backfill"]
    bridge = result["new_kv_bridge"]
    return {
        "comparison_id": comparison_id,
        **inputs,
        "history_completion_ms": history["completion_time_s"] * 1000,
        "bridge_completion_ms": bridge["completion_time_s"] * 1000,
        "history_source_release_ms": history["source_release_time_s"] * 1000,
        "bridge_source_release_ms": bridge["source_release_time_s"] * 1000,
        "history_network_bytes": history["total_network_bytes"],
        "bridge_network_bytes": bridge["total_network_bytes"],
        "history_outcome": history["outcome"],
        "bridge_outcome": bridge["outcome"],
        "latency_winner": result["latency_winner"],
        "source_release_winner": result["source_release_winner"],
        "network_bytes_winner": result["network_bytes_winner"],
        "remote_penalty_break_even_ms": result[
            "new_kv_bridge_remote_penalty_break_even_ms"
        ],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    winner_fields = (
        "latency_winner",
        "source_release_winner",
        "network_bytes_winner",
    )
    winners = {
        field: {
            name: sum(row[field] == name for row in comparisons)
            for name in ("history_backfill", "new_kv_bridge", "tie")
        }
        for field in winner_fields
    }
    break_evens = [
        row["new_kv_bridge_remote_penalty_break_even_ms"]
        for row in comparisons
        if row["new_kv_bridge_remote_penalty_break_even_ms"] is not None
    ]
    return {
        "format_version": 1,
        "comparisons": len(comparisons),
        "rows": len(comparisons) * 2,
        "winners": winners,
        "remote_penalty_break_even_ms": {
            "minimum": min(break_evens) if break_evens else None,
            "maximum": max(break_evens) if break_evens else None,
        },
        "interpretation": {
            "history_backfill": (
                "Can release TP1 after complete history and delta ACKs."
            ),
            "new_kv_bridge": (
                "Starts with little KV transfer but retains TP1 for remote "
                "historical attention until request completion."
            ),
            "warning": (
                "This is a parameterized decision demo, not an end-to-end "
                "remote-attention measurement."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    kv_bytes = kv_bytes_per_token(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        dtype_bytes=args.dtype_bytes,
    )
    remote_bytes = remote_attention_bytes_per_token(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        dtype_bytes=args.dtype_bytes,
    )
    comparisons: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    comparison_id = 0
    for history_tokens in args.history_tokens:
        for remaining_tokens in args.remaining_tokens:
            for copy_rate in args.copy_rate_gib_s:
                for remote_penalty_ms in args.remote_penalty_ms:
                    comparison_id += 1
                    inputs = ShadowCopyInputs(
                        history_tokens=history_tokens,
                        remaining_tokens=remaining_tokens,
                        block_size=args.block_size,
                        kv_bytes_per_token=kv_bytes,
                        copy_rate_bytes_s=copy_rate * 1024**3,
                        source_tpot_s=args.source_tpot_ms / 1000,
                        target_tpot_s=args.target_tpot_ms / 1000,
                        remote_attention_penalty_s=remote_penalty_ms / 1000,
                        remote_attention_bytes_per_token=remote_bytes,
                        bridge_start_tokens=args.bridge_start_tokens,
                    )
                    result = compare_shadow_strategies(inputs)
                    comparisons.append(result)
                    rows.extend(flatten(comparison_id, result))
                    decisions.append(flatten_decision(comparison_id, result))

    write_csv(rows, args.out_dir / "shadow_strategy_rows.csv")
    write_csv(decisions, args.out_dir / "shadow_strategy_decisions.csv")
    (args.out_dir / "shadow_strategy_comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )
    summary = summarize(comparisons)
    summary["geometry"] = {
        "kv_bytes_per_token": kv_bytes,
        "remote_attention_bytes_per_token": remote_bytes,
        "num_layers": args.num_layers,
        "num_kv_heads": args.num_kv_heads,
        "head_size": args.head_size,
        "hidden_size": args.hidden_size,
        "dtype_bytes": args.dtype_bytes,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
