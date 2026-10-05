#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pair a frozen TraceA arrival slice with held-out OASST1 text prompts.

This is a composite engineering input, not a replay of original TraceA request
contents. TraceA output_len is never read into the request or arrival records.
The output is input preparation only; it does not run a GPU experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("request input must be non-empty JSONL objects")
    return rows


def read_arrival_slice(
    path: Path, *, start_row: int, count: int
) -> list[dict[str, Any]]:
    """Read only arrival and pre-arrival fields from a chronological slice."""
    if start_row < 0 or count <= 0:
        raise ValueError("start_row and count must be non-negative/positive")
    result = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not {"timestamp", "input_len", "type"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError("TraceA needs timestamp, input_len and type")
        for index, row in enumerate(reader):
            if index < start_row:
                continue
            if index >= start_row + count:
                break
            timestamp = float(row["timestamp"])
            input_len = int(row["input_len"])
            if not math.isfinite(timestamp) or timestamp < 0 or input_len <= 0:
                raise ValueError(f"invalid trace row: {index}")
            result.append({
                "row_index": index,
                "timestamp": timestamp,
                "input_len": input_len,
                "type": row["type"],
            })
    if len(result) != count:
        raise ValueError("requested slice extends beyond TraceA")
    if any(b["timestamp"] < a["timestamp"] for a, b in zip(
        result, result[1:]
    )):
        raise ValueError("TraceA arrival times are not monotonic")
    return result


def count_request_tokens(tokenizer: Any, row: dict[str, Any]) -> int:
    """Count token IDs after rendering the same chat template used by vLLM."""
    prompt = row.get("prompt")
    if prompt is None:
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True
        )
    if not isinstance(prompt, str):
        raise ValueError("tokenizer did not render a text prompt")
    ids = tokenizer.encode(prompt)
    if not isinstance(ids, list) or not ids:
        raise ValueError("tokenizer returned no token ID list")
    return len(ids)


def make_composite(
    trace: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    *,
    seed: int,
    time_scale: float,
    source_fraction: float,
    max_tokens: int,
    max_model_len: int,
    token_count: Callable[[dict[str, Any]], int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep trace order; pair content and initial placement independently."""
    if not trace:
        raise ValueError("trace slice is empty")
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("time_scale must be positive and finite")
    if not 0 < source_fraction < 1:
        raise ValueError("source_fraction must be within (0, 1)")
    if max_tokens <= 0 or max_model_len <= max_tokens:
        raise ValueError("invalid model or output token cap")
    candidates = [row for row in requests if row.get("split") == "test"]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = []
    seen_ids: set[str] = set()
    for row in candidates:
        request_id = row.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("OASST1 test request has invalid id")
        if request_id in seen_ids:
            raise ValueError(f"duplicate OASST1 request id: {request_id}")
        seen_ids.add(request_id)
        if "prompt" not in row and "messages" not in row:
            raise ValueError(f"OASST1 request has no content: {request_id}")
        prompt_tokens = token_count(row)
        if prompt_tokens > 0 and prompt_tokens + max_tokens <= max_model_len:
            selected.append((row, prompt_tokens))
        if len(selected) == len(trace):
            break
    if len(selected) != len(trace):
        raise ValueError("not enough held-out requests within model context")
    source_count = max(1, min(len(trace) - 1, round(
        len(trace) * source_fraction
    )))
    placement = ["source"] * source_count + ["target"] * (
        len(trace) - source_count
    )
    rng.shuffle(placement)
    # The first arrival is the single tracked anchor for the initial pilot.
    if placement[0] != "source":
        first_source = placement.index("source")
        placement[first_source], placement[0] = placement[0], placement[first_source]
    zero = trace[0]["timestamp"]
    paired_requests = []
    arrivals = []
    for index, (event, selected_request) in enumerate(zip(trace, selected)):
        row, prompt_tokens = selected_request
        request_id = str(row["id"])
        paired_requests.append({
            "id": request_id,
            **{key: row[key] for key in ("prompt", "messages") if key in row},
            "max_tokens": max_tokens,
            "ignore_eos": False,
        })
        arrivals.append({
            "request_id": request_id,
            "arrival_unix_s": round(
                (event["timestamp"] - zero) * time_scale, 6
            ),
            "pool": placement[index],
            "max_tokens": max_tokens,
            "ignore_eos": False,
            "trace_row_index": event["row_index"],
            "prompt_tokens": prompt_tokens,
        })
    audit = {
        "format_version": 1,
        "status": "COMPOSITE_TRACE_INPUT_ONLY",
        "content_source": "OASST1_held_out_test",
        "arrival_source": "TraceA_timestamp_slice",
        "original_trace_content_replayed": False,
        "original_trace_input_length_matched": False,
        "trace_output_len_used": False,
        "time_axis_scaled": time_scale != 1.0,
        "time_scale": time_scale,
        "seed": seed,
        "trace_first_row": trace[0]["row_index"],
        "trace_last_row": trace[-1]["row_index"],
        "requests": len(arrivals),
        "source_arrivals": source_count,
        "target_arrivals": len(arrivals) - source_count,
        "arrival_span_s": arrivals[-1]["arrival_unix_s"],
        "exact_replay_supported_by_current_runner": False,
    }
    return paired_requests, arrivals, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "requests", "tokenizer", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-trace-sha256", required=True)
    parser.add_argument("--expected-requests-sha256", required=True)
    parser.add_argument("--expected-model-config-sha256", required=True)
    parser.add_argument("--start-row", type=int, default=30140)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--time-scale", type=float, default=50.0)
    parser.add_argument("--source-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise ValueError(f"output directory already exists: {args.out_dir}")
    checks = {
        "trace": (args.trace, args.expected_trace_sha256),
        "requests": (args.requests, args.expected_requests_sha256),
        "model_config": (
            args.tokenizer / "config.json", args.expected_model_config_sha256
        ),
    }
    actual = {name: sha256(path) for name, (path, _) in checks.items()}
    for name, (_, expected) in checks.items():
        if actual[name] != expected.lower():
            raise ValueError(f"{name} SHA mismatch: {actual[name]}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )

    def token_count(row: dict[str, Any]) -> int:
        return count_request_tokens(tokenizer, row)

    requests, arrivals, audit = make_composite(
        read_arrival_slice(
            args.trace, start_row=args.start_row, count=args.count
        ),
        read_jsonl(args.requests),
        seed=args.seed, time_scale=args.time_scale,
        source_fraction=args.source_fraction, max_tokens=args.max_tokens,
        max_model_len=args.max_model_len, token_count=token_count,
    )
    audit["input_sha256"] = actual
    audit["trace_path"] = str(args.trace.resolve())
    audit["requests_path"] = str(args.requests.resolve())
    audit["tokenizer_path"] = str(args.tokenizer.resolve())
    args.out_dir.mkdir(parents=True)
    for name, rows in (("requests.jsonl", requests),
                       ("arrivals.jsonl", arrivals)):
        with (args.out_dir / name).open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit["output_sha256"] = {
        name: sha256(args.out_dir / name)
        for name in ("requests.jsonl", "arrivals.jsonl")
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
