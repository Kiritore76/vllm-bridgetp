#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a target-only workload for online Shadow strategy validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-jobs", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument("--prompt-token-id", type=int, default=100)
    parser.add_argument("--start-interval-s", type=float, default=0.02)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--model", default="bridgetp-model")
    return parser.parse_args()


def build_manifest(
    *,
    target_jobs: int = 8,
    prompt_tokens: int = 4096,
    output_tokens: int = 2048,
    prompt_token_id: int = 100,
    start_interval_s: float = 0.02,
    max_model_len: int = 8192,
    model: str = "bridgetp-model",
) -> dict:
    if target_jobs < 2:
        raise ValueError("online validation requires at least two target jobs")
    if prompt_tokens <= 0 or output_tokens <= 0:
        raise ValueError("prompt/output token counts must be positive")
    if prompt_tokens + output_tokens > max_model_len:
        raise ValueError("target context exceeds max model length")
    if start_interval_s < 0:
        raise ValueError("start interval cannot be negative")
    prompt = [prompt_token_id] * prompt_tokens
    return {
        "format_version": 1,
        "scenario": "Online Shadow strategy paired target load",
        "status": "WORKING_NOT_FROZEN",
        "design_note": (
            "Target-only long-running requests begin before the migration anchor; "
            "their token timestamps provide paired pre/Shadow/Bridge/post windows."
        ),
        "parameters": {
            "target_jobs": target_jobs,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "prompt_token_id": prompt_token_id,
            "start_interval_s": start_interval_s,
            "max_model_len": max_model_len,
        },
        "jobs": [
            {
                "job_id": f"target_{index:03d}",
                "pool": "target",
                "start_after_s": index * start_interval_s,
                "request": {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": output_tokens,
                    "ignore_eos": True,
                },
            }
            for index in range(target_jobs)
        ],
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    values = vars(args).copy()
    out = values.pop("out")
    manifest = build_manifest(**values)
    write_json(out, manifest)
    print(
        f"wrote online Shadow workload: {out.resolve()} "
        f"({len(manifest['jobs'])} target jobs)"
    )


if __name__ == "__main__":
    main()
