# SPDX-License-Identifier: Apache-2.0
"""Compare exact delta restore with the former blockwise path, before engines."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vllm.bridge_tp.kv_restore import inject_rank_delta  # noqa: E402


def reference_blockwise(destination, delta, blocks, start, end, block_size):
    """Retain the former copy/readback operations as a measured reference."""
    mismatches = None
    for name, source in delta.items():
        cache = destination[name].permute(1, 2, 0, 3, 4)
        for logical in range(start // block_size, (end - 1) // block_size + 1):
            left, right = (
                max(start, logical * block_size),
                min(end, (logical + 1) * block_size),
            )
            target = cache[
                blocks[logical], left % block_size : left % block_size + right - left
            ]
            payload = source[left - start : right - start]
            target.copy_(payload)
            count = torch.count_nonzero(target != payload)
            mismatches = count if mismatches is None else mismatches + count
    if mismatches is None or int(mismatches.item()) != 0:
        raise ValueError("reference exact readback failed")


def benchmark(device: str, *, layers: int, heads: int, head_size: int, iterations: int):
    cache = {
        f"layer.{index}": torch.full(
            (2, 12, 16, heads, head_size), -9.0, device=device, dtype=torch.bfloat16
        )
        for index in range(layers)
    }
    blocks = [7, 2, 10, 1, 8, 5]
    rows = []

    def synchronize():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    for tokens in (10, 17, 29, 32):
        start, end = 13, 13 + tokens
        delta = {
            name: torch.randn(tokens, 2, heads, head_size, device=device).bfloat16()
            for name in cache
        }
        expected = {name: value.clone() for name, value in cache.items()}
        reference_blockwise(expected, delta, blocks, start, end, 16)
        inject_rank_delta(
            cache,
            delta,
            blocks,
            start_token=start,
            end_token=end,
            block_axis=1,
            block_size=16,
        )
        assert all(torch.equal(cache[name], expected[name]) for name in cache)
        timings = {"scatter": [], "blockwise_reference": []}
        for repetition in range(iterations + 2):
            order = ("scatter", "blockwise_reference")
            if repetition % 2:
                order = tuple(reversed(order))
            for method in order:
                synchronize()
                began = time.perf_counter()
                if method == "scatter":
                    inject_rank_delta(
                        cache,
                        delta,
                        blocks,
                        start_token=start,
                        end_token=end,
                        block_axis=1,
                        block_size=16,
                    )
                else:
                    reference_blockwise(cache, delta, blocks, start, end, 16)
                synchronize()
                if repetition >= 2:
                    timings[method].append((time.perf_counter() - began) * 1000)
        rows.append(
            {
                "tokens": tokens,
                "exact_full_cache_match": True,
                "scatter_ms": timings["scatter"],
                "blockwise_reference_ms": timings["blockwise_reference"],
                "median_scatter_ms": statistics.median(timings["scatter"]),
                "median_reference_ms": statistics.median(
                    timings["blockwise_reference"]
                ),
            }
        )
    return {
        "device": device,
        "layers": layers,
        "rank_kv_heads": heads,
        "head_size": head_size,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--devices", nargs="+", default=[f"cuda:{i}" for i in range(1, 5)]
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 3 or args.out.exists():
        raise ValueError("use at least three repetitions and a new output path")
    config = json.loads(args.model_config.read_text())
    if config["num_key_value_heads"] % 4:
        raise ValueError("KV heads cannot be partitioned across TP4")
    reports = [
        benchmark(
            device,
            layers=config["num_hidden_layers"],
            heads=config["num_key_value_heads"] // 4,
            head_size=config["hidden_size"] // config["num_attention_heads"],
            iterations=args.iterations,
        )
        for device in args.devices
    ]
    args.out.write_text(
        json.dumps(
            {
                "format_version": 1,
                "benchmark": "LOCAL_DELTA_APPLY_NOT_END_TO_END_MIGRATION",
                "torch": torch.__version__,
                "model_config": str(args.model_config),
                "devices": reports,
            },
            indent=2,
        )
        + "\n"
    )
    print("delta restore exact/readback microbenchmark:", args.out)


if __name__ == "__main__":
    main()
