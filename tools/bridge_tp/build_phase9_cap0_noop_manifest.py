#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a compactly-configured CAP-0 No-op bring-up manifest.

The generated manifest is consumed only by the independent background workload
process.  It is never passed to the controller.  Long target prompts use token
IDs so their KV demand is deterministic and auditable without tokenizer drift.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-copies", type=int, default=72)
    parser.add_argument("--target-prompt-tokens", type=int, default=7000)
    parser.add_argument("--target-output-tokens", type=int, default=1100)
    parser.add_argument("--target-start-after-s", type=float, default=0.0)
    parser.add_argument("--target-start-interval-s", type=float, default=0.01)
    parser.add_argument("--target-prompt-token-id", type=int, default=100)
    parser.add_argument("--source-copies", type=int, default=4)
    parser.add_argument("--source-output-tokens", type=int, default=7000)
    parser.add_argument("--source-start-after-s", type=float, default=2.0)
    parser.add_argument("--source-start-interval-s", type=float, default=0.2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--model", default="bridgetp-model")
    return parser.parse_args()


def build_manifest(
    *,
    target_copies: int = 72,
    target_prompt_tokens: int = 7000,
    target_output_tokens: int = 1100,
    target_start_after_s: float = 0.0,
    target_start_interval_s: float = 0.01,
    target_prompt_token_id: int = 100,
    source_copies: int = 4,
    source_output_tokens: int = 7000,
    source_start_after_s: float = 2.0,
    source_start_interval_s: float = 0.2,
    max_model_len: int = 8192,
    model: str = "bridgetp-model",
) -> dict[str, Any]:
    if target_copies <= 0 or source_copies <= 0:
        raise ValueError("target and source copies must be positive")
    if target_prompt_tokens <= 0 or target_output_tokens <= 0:
        raise ValueError("target prompt/output tokens must be positive")
    if source_output_tokens <= 0:
        raise ValueError("source output tokens must be positive")
    if target_prompt_tokens + target_output_tokens > max_model_len:
        raise ValueError("target prompt plus output exceeds max model length")
    if source_output_tokens > max_model_len - 128:
        raise ValueError("source output must leave 128 tokens for its prompt")
    for name, value in (
        ("target_start_after_s", target_start_after_s),
        ("target_start_interval_s", target_start_interval_s),
        ("source_start_after_s", source_start_after_s),
        ("source_start_interval_s", source_start_interval_s),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    jobs: list[dict[str, Any]] = []
    for index in range(target_copies):
        jobs.append(
            {
                "job_id": f"target_guard_{index:03d}",
                "pool": "target",
                "start_after_s": (
                    target_start_after_s + index * target_start_interval_s
                ),
                "request": {
                    "model": model,
                    "prompt": [target_prompt_token_id] * target_prompt_tokens,
                    "max_tokens": target_output_tokens,
                    "ignore_eos": True,
                },
            }
        )
    source_prompt = (
        "Write a detailed systems design review for a capacity-limited scheduler."
    )
    for index in range(source_copies):
        jobs.append(
            {
                "job_id": f"source_pressure_{index:03d}",
                "pool": "source",
                "start_after_s": (
                    source_start_after_s + index * source_start_interval_s
                ),
                "request": {
                    "model": model,
                    "prompt": source_prompt,
                    "max_tokens": source_output_tokens,
                    "ignore_eos": True,
                },
            }
        )
    return {
        "format_version": 1,
        "scenario": "CAP-0 No-op bring-up",
        "status": "WORKING_NOT_FROZEN",
        "controller_visibility": "NONE; background workload process only",
        "parameters": {
            "target_copies": target_copies,
            "target_prompt_tokens": target_prompt_tokens,
            "target_output_tokens": target_output_tokens,
            "target_start_after_s": target_start_after_s,
            "target_start_interval_s": target_start_interval_s,
            "target_prompt_token_id": target_prompt_token_id,
            "source_copies": source_copies,
            "source_output_tokens": source_output_tokens,
            "source_start_after_s": source_start_after_s,
            "source_start_interval_s": source_start_interval_s,
            "max_model_len": max_model_len,
        },
        "jobs": jobs,
    }


def write_json(path: Path, value: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        target_copies=args.target_copies,
        target_prompt_tokens=args.target_prompt_tokens,
        target_output_tokens=args.target_output_tokens,
        target_start_after_s=args.target_start_after_s,
        target_start_interval_s=args.target_start_interval_s,
        target_prompt_token_id=args.target_prompt_token_id,
        source_copies=args.source_copies,
        source_output_tokens=args.source_output_tokens,
        source_start_after_s=args.source_start_after_s,
        source_start_interval_s=args.source_start_interval_s,
        max_model_len=args.max_model_len,
        model=args.model,
    )
    write_json(args.out, manifest)
    target_tokens = (
        args.target_copies
        * (args.target_prompt_tokens + args.target_output_tokens)
    )
    print(
        f"wrote CAP-0 No-op working manifest: {args.out.resolve()} "
        f"({len(manifest['jobs'])} jobs, "
        f"target demand={target_tokens} tokens)"
    )


if __name__ == "__main__":
    main()
