#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze timestamped natural-request windows for benefit experiments.

This prepares input provenance only. The current Phase 9 runner launches one
anchor before its background process and cannot exactly replay arbitrary
timestamps, so output windows are deliberately marked PLAN_ONLY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def freeze_windows(
    requests: list[dict[str, Any]],
    arrivals: list[dict[str, Any]],
    *,
    window_s: float,
    default_max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Keep arrival order and input parameters without using future outputs."""
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be positive and finite")
    by_id: dict[str, dict[str, Any]] = {}
    for row in requests:
        request_id = row.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request id must be a non-empty string")
        if request_id in by_id:
            raise ValueError(f"duplicate request id: {request_id}")
        if "prompt" not in row and "messages" not in row:
            raise ValueError(f"request has no prompt/messages: {request_id}")
        by_id[request_id] = row
    seen_arrivals: set[str] = set()
    ordered = []
    for event in arrivals:
        request_id = event.get("request_id")
        if request_id not in by_id or request_id in seen_arrivals:
            raise ValueError(f"unknown or duplicate arrival request: {request_id}")
        seen_arrivals.add(request_id)
        arrived = event.get("arrival_unix_s")
        if isinstance(arrived, bool) or not isinstance(arrived, (int, float)) or (
            not math.isfinite(arrived)
        ):
            raise ValueError(f"invalid arrival time: {request_id}")
        if event.get("pool") not in {"source", "target"}:
            raise ValueError(f"invalid frozen initial pool: {request_id}")
        source = by_id[request_id]
        max_tokens = event.get("max_tokens", source.get("max_tokens"))
        defaulted = max_tokens is None
        if defaulted:
            max_tokens = default_max_tokens
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or (
            max_tokens <= 0
        ):
            raise ValueError(f"missing or invalid max_tokens: {request_id}")
        ignore_eos = event.get("ignore_eos", source.get("ignore_eos", False))
        if ignore_eos is not False:
            raise ValueError(f"natural benefit data requires EOS: {request_id}")
        ordered.append({
            "request_id": request_id,
            "arrival_unix_s": float(arrived),
            "pool": event["pool"],
            "max_tokens": max_tokens,
            "max_tokens_defaulted": defaulted,
            "sampling": {
                key: event.get(key, source.get(key))
                for key in ("temperature", "top_p", "top_k", "min_p")
                if key in event or key in source
            },
            "request": {
                key: source[key] for key in ("prompt", "messages")
                if key in source
            },
        })
    ordered.sort(key=lambda row: (row["arrival_unix_s"], row["request_id"]))
    zero = ordered[0]["arrival_unix_s"]
    windows: dict[int, list[dict[str, Any]]] = {}
    for row in ordered:
        index = int((row["arrival_unix_s"] - zero) // window_s)
        windows.setdefault(index, []).append(row)
    result = []
    for index, events in sorted(windows.items()):
        window_start = zero + index * window_s
        relative = [{
            **event, "arrival_offset_s": round(
                event["arrival_unix_s"] - window_start, 6
            ),
        } for event in events]
        pools = Counter(event["pool"] for event in events)
        result.append({
            "format_version": 1,
            "status": "TRACE_WINDOW_PLAN_ONLY",
            "window_index": index,
            "window_start_unix_s": window_start,
            "window_end_unix_s": window_start + window_s,
            "window_s": window_s,
            "source_arrivals": pools["source"],
            "target_arrivals": pools["target"],
            "can_supply_single_source_anchor": pools["source"] > 0,
            "arrivals": relative,
        })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--arrivals", type=Path, required=True)
    parser.add_argument("--expected-requests-sha256", required=True)
    parser.add_argument("--expected-arrivals-sha256", required=True)
    parser.add_argument("--window-s", type=float, required=True)
    parser.add_argument("--default-max-tokens", type=int)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise ValueError(f"output directory already exists: {args.out_dir}")
    actual = {
        "requests": sha256(args.requests),
        "arrivals": sha256(args.arrivals),
    }
    expected = {
        "requests": args.expected_requests_sha256,
        "arrivals": args.expected_arrivals_sha256,
    }
    if actual != expected:
        raise ValueError(f"input SHA mismatch: {actual}")
    windows = freeze_windows(
        _read_jsonl(args.requests), _read_jsonl(args.arrivals),
        window_s=args.window_s,
        default_max_tokens=args.default_max_tokens,
    )
    args.out_dir.mkdir(parents=True)
    for window in windows:
        name = f"window-{window['window_index']:04d}.json"
        (args.out_dir / name).write_text(
            json.dumps(window, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "format_version": 1,
        "status": "TRACE_WINDOW_PLAN_ONLY",
        "requests_path": str(args.requests.resolve()),
        "arrivals_path": str(args.arrivals.resolve()),
        "sha256": actual,
        "window_s": args.window_s,
        "windows": len(windows),
        "source_arrivals": sum(window["source_arrivals"] for window in windows),
        "target_arrivals": sum(window["target_arrivals"] for window in windows),
        "default_max_tokens_used": sum(
            event["max_tokens_defaulted"]
            for window in windows for event in window["arrivals"]
        ),
        "exact_replay_supported_by_current_runner": False,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
