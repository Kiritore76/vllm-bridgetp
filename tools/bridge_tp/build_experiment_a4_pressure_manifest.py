#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add real TP1 KV pressure requests to a target-only A3 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp.build_phase9_cap0_noop_manifest import (  # noqa: E402
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-target-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-jobs", type=int, default=4)
    parser.add_argument("--source-prompt-tokens", type=int, default=4096)
    parser.add_argument("--source-output-tokens", type=int, default=3500)
    parser.add_argument("--source-start-after-s", type=float, default=2.0)
    parser.add_argument("--source-start-interval-s", type=float, default=0.1)
    parser.add_argument("--source-prompt-token-id", type=int, default=100)
    parser.add_argument("--max-model-len", type=int, default=8192)
    return parser.parse_args()


def build_manifest(
    base: dict[str, Any],
    *,
    source_jobs: int,
    source_prompt_tokens: int,
    source_output_tokens: int,
    source_start_after_s: float,
    source_start_interval_s: float,
    source_prompt_token_id: int,
    max_model_len: int,
) -> dict[str, Any]:
    if not base["jobs"] or any(job["pool"] != "target" for job in base["jobs"]):
        raise ValueError("base manifest must contain target jobs only")
    if source_jobs <= 0:
        raise ValueError("source_jobs must be positive")
    if source_prompt_tokens <= 0 or source_output_tokens <= 0:
        raise ValueError("source prompt and output lengths must be positive")
    if source_prompt_tokens + source_output_tokens > max_model_len:
        raise ValueError("source request exceeds max model length")
    if source_start_after_s < 0 or source_start_interval_s < 0:
        raise ValueError("source start offsets must be nonnegative")
    if not 0 <= source_prompt_token_id <= 2_000_000:
        raise ValueError("source prompt token ID is invalid")
    existing = {str(job["job_id"]) for job in base["jobs"]}
    model = str(base["jobs"][0]["request"]["model"])
    jobs = list(base["jobs"])
    for index in range(source_jobs):
        job_id = f"a4_source_{index:03d}"
        if job_id in existing:
            raise ValueError(f"duplicate job ID: {job_id}")
        jobs.append({
            "job_id": job_id,
            "pool": "source",
            "start_after_s": source_start_after_s
            + index * source_start_interval_s,
            "request": {
                "model": model,
                "prompt": [source_prompt_token_id] * source_prompt_tokens,
                "max_tokens": source_output_tokens,
                "ignore_eos": True,
            },
        })
    return {
        "format_version": 1,
        "scenario": "Experiment A4-P source KV pressure smoke",
        "status": "WORKING_NOT_FROZEN",
        "base_target_manifest": str(base.get("scenario", "A3 target load")),
        "parameters": {
            "source_jobs": source_jobs,
            "source_prompt_tokens": source_prompt_tokens,
            "source_output_tokens": source_output_tokens,
            "source_start_after_s": source_start_after_s,
            "source_start_interval_s": source_start_interval_s,
            "max_model_len": max_model_len,
        },
        "jobs": jobs,
    }


def main() -> None:
    args = parse_args()
    base = json.loads(args.base_target_manifest.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or base.get("format_version") != 1:
        raise ValueError("base manifest requires format_version=1")
    if not isinstance(base.get("jobs"), list):
        raise ValueError("base manifest requires jobs")
    manifest = build_manifest(
        base,
        source_jobs=args.source_jobs,
        source_prompt_tokens=args.source_prompt_tokens,
        source_output_tokens=args.source_output_tokens,
        source_start_after_s=args.source_start_after_s,
        source_start_interval_s=args.source_start_interval_s,
        source_prompt_token_id=args.source_prompt_token_id,
        max_model_len=args.max_model_len,
    )
    write_json(args.out, manifest)
    print(
        f"wrote A4-P working manifest: {args.out.resolve()} "
        f"({len(manifest['jobs'])} jobs)"
    )


if __name__ == "__main__":
    main()
