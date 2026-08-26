#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a measured, rate-limited P2P copy window for Phase 9 calibration.

Rate zero records an equal-duration baseline window without allocating CUDA
buffers or issuing traffic. Nonzero rates reuse a single chunk buffer and are
synthetic interference calibration, not a claim of real KV migration.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

GIB = 1024**3
MIB = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=int, required=True)
    parser.add_argument("--dst", type=int, required=True)
    parser.add_argument("--chunk-mib", type=float, default=16.0)
    parser.add_argument("--target-gib-s", type=float, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.src == args.dst:
        parser.error("source and destination GPUs must differ")
    if args.chunk_mib <= 0 or args.seconds <= 0:
        parser.error("chunk size and duration must be positive")
    if args.target_gib_s < 0:
        parser.error("target rate cannot be negative")
    return args


def run_copy(args: argparse.Namespace) -> dict:
    if args.target_gib_s == 0:
        start_unix = time.time()
        start = time.perf_counter()
        time.sleep(args.seconds)
        end = time.perf_counter()
        return {
            "src": args.src,
            "dst": args.dst,
            "target_gib_s": 0.0,
            "effective_gib_s": 0.0,
            "chunk_mib": args.chunk_mib,
            "start_unix_s": start_unix,
            "end_unix_s": time.time(),
            "start_perf_counter": start,
            "end_perf_counter": end,
            "completion_s": end - start,
            "total_gib": 0.0,
            "copies": 0,
            "p2p_access": None,
            "mode": "baseline-window-no-copy",
        }

    import torch

    elements = max(1, int(args.chunk_mib * MIB / 2))
    with torch.cuda.device(args.src):
        source = torch.randn(
            elements,
            dtype=torch.float16,
            device=f"cuda:{args.src}",
        )
    with torch.cuda.device(args.dst):
        target = torch.empty(
            elements,
            dtype=torch.float16,
            device=f"cuda:{args.dst}",
        )
        stream = torch.cuda.Stream(device=args.dst)

    start_unix = time.time()
    start = time.perf_counter()
    end_at = start + args.seconds
    copied_bytes = 0
    copies = 0
    chunk_bytes = int(args.chunk_mib * MIB)
    while time.perf_counter() < end_at:
        with torch.cuda.device(args.dst), torch.cuda.stream(stream):
            target.copy_(source, non_blocking=True)
        stream.synchronize()
        copied_bytes += chunk_bytes
        copies += 1
        ideal_s = copied_bytes / (args.target_gib_s * GIB)
        elapsed_s = time.perf_counter() - start
        if ideal_s > elapsed_s:
            time.sleep(ideal_s - elapsed_s)

    end = time.perf_counter()
    wall_s = end - start
    total_gib = copied_bytes / GIB
    return {
        "src": args.src,
        "dst": args.dst,
        "target_gib_s": args.target_gib_s,
        "effective_gib_s": total_gib / wall_s,
        "chunk_mib": args.chunk_mib,
        "start_unix_s": start_unix,
        "end_unix_s": time.time(),
        "start_perf_counter": start,
        "end_perf_counter": end,
        "completion_s": wall_s,
        "total_gib": total_gib,
        "copies": copies,
        "p2p_access": bool(torch.cuda.can_device_access_peer(args.src, args.dst)),
        "mode": "synthetic-sustained-p2p-copy",
        "evidence_boundary": (
            "This is P2-D-style synthetic P2P traffic for interference "
            "calibration; it is not D-group KV migration."
        ),
    }


def main() -> None:
    args = parse_args()
    payload = run_copy(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
