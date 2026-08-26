#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build one target-local A/B/C/D agreement record from token artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.bridge_tp.controller.numerics import agreement_length  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-offset", type=int, default=0)
    parser.add_argument("--right-offset", type=int, default=0)
    parser.add_argument("--boundary-k", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--fixed-prefix-token-ids", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_token_ids(path: Path) -> list[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, dict):
        values = None
        for key in (
            "token_ids",
            "output_token_ids",
            "tokens",
            "assembled_token_ids",
            "fixed_token_ids",
        ):
            candidate = raw.get(key)
            if isinstance(candidate, list):
                values = candidate
                break
    else:
        values = None
    if not isinstance(values, list):
        raise SystemExit(f"{path}: could not find a token-id list")
    return [int(value) for value in values]


def hash_tokens(values: list[int]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    if min(args.left_offset, args.right_offset, args.boundary_k) < 0:
        raise SystemExit("offsets and K must be nonnegative")
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    left_all = load_token_ids(args.left)
    right_all = load_token_ids(args.right)
    prefix = load_token_ids(args.fixed_prefix_token_ids)
    left = left_all[args.left_offset : args.left_offset + args.budget]
    right = right_all[args.right_offset : args.right_offset + args.budget]
    available = min(len(left), len(right), args.budget)
    agreement = agreement_length(left, right, budget=args.budget)
    complete = len(left) >= args.budget and len(right) >= args.budget
    metadata = (
        json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {}
    )
    payload = {
        "format_version": 1,
        "group": args.group,
        "request_id": args.request_id,
        "boundary_k": args.boundary_k,
        "comparison_space": "target_local_after_fixed_prefix_k",
        "fixed_prefix_token_count": len(prefix),
        "fixed_prefix_sha256": hash_tokens(prefix),
        "budget": args.budget,
        "available_tokens": available,
        "complete_budget": complete,
        "agreement_length": agreement,
        "full_agreement": complete and agreement == args.budget,
        "first_divergence_target_local": (agreement if agreement < available else None),
        "first_divergence_global": (
            args.boundary_k + agreement if agreement < available else None
        ),
        "left": {"path": str(args.left), "offset": args.left_offset},
        "right": {"path": str(args.right), "offset": args.right_offset},
        "metadata": metadata,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"group {args.group} request={args.request_id} K={args.boundary_k} "
        f"target-local agreement={agreement}/{args.budget}"
    )
    if not complete:
        print("WARNING: one or both sequences are shorter than the requested budget")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
