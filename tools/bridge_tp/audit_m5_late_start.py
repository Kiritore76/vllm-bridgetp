#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit whether an A100 M5 smoke genuinely tested late Shadow start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def audit(run_dir: Path, minimum_output_tokens: int) -> dict[str, Any]:
    result = run_dir / "online" / "r01_shadow_only"
    audit_rows = read_jsonl(result / "controller" / "phase9_audit.jsonl")
    events = read_jsonl(result / "background" / "background_events.jsonl")
    acceptance = json.loads((result / "provenance" /
                             "shadow_online_acceptance.json").read_text(
                                 encoding="utf-8"))
    manifest = json.loads((result / "background" /
                           "background_manifest.json").read_text(
                               encoding="utf-8"))
    errors: list[str] = []
    source_jobs = [job for job in manifest["jobs"] if job["pool"] == "source"]
    if not source_jobs or any(job.get("start_after_event") !=
                              "ANCHOR_FIRST_OUTPUT" for job in source_jobs):
        errors.append("source jobs were not gated by anchor first output")
    starts = [row for row in audit_rows
              if row.get("kind") == "manager_m1_start_decision"
              and row.get("decision", {}).get("action") == "START_SHADOW"]
    if len(starts) != 1:
        errors.append(f"expected one M1 Shadow start, found {len(starts)}")
    start = starts[0] if starts else None
    output = (start.get("snapshot", {}).get("generated_tokens")
              if start else None)
    if output is None or output < minimum_output_tokens:
        errors.append("M1 started before the delayed output boundary")
    first_output = next((row for row in audit_rows
                         if row.get("kind") == "telemetry"
                         and (row.get("output_tokens") or 0) > 0), None)
    first_output_s = first_output.get("unix_s") if first_output else None
    started_s = start.get("unix_s") if start else None
    source_started = {
        row["job_id"]: row["unix_s"] for row in events
        if row.get("kind") == "job_start" and row.get("pool") == "source"
    }
    source_ended = {
        row["job_id"]: row["unix_s"] for row in events
        if row.get("kind") == "job_end" and row.get("pool") == "source"
    }
    active_at_start = (
        sum(began <= started_s < source_ended.get(job, float("inf"))
            for job, began in source_started.items())
        if started_s is not None else 0
    )
    if not source_started or first_output_s is None or any(
        began < first_output_s for began in source_started.values()
    ):
        errors.append("source jobs did not start after anchor first output")
    if active_at_start < 1:
        errors.append("no source peer was active before M1 started")
    release_s = acceptance.get("source_kv_released_unix_s")
    if release_s is None or started_s is None or release_s <= started_s:
        errors.append("TP1 KV release was not observed after M1 start")
    guard = (start.get("snapshot", {}).get("source_guard_free_kv_tokens")
             if start else None)
    before_release = [row for row in audit_rows
                      if row.get("kind") == "telemetry"
                      and started_s is not None and release_s is not None
                      and started_s <= row.get("unix_s", 0) < release_s]
    signals = [row.get("capacity_signal") or {} for row in before_release]
    free_values = [signal.get("free_kv_tokens") for signal in signals]
    minimum_free = (min(free_values) if free_values
                    and all(value is not None for value in free_values)
                    else None)
    headrooms = [
        signal["free_kv_tokens"] - guard - signal["prefill_pending_kv_tokens"]
        for signal in signals
        if guard is not None and signal.get("free_kv_tokens") is not None
        and signal.get("prefill_pending_kv_tokens") is not None
    ]
    minimum_headroom = (min(headrooms) if len(headrooms) == len(signals)
                        and headrooms else None)
    preemptions = [((row.get("tp1") or {}).get("preemptions_total"))
                   for row in before_release]
    max_preemptions = (max(preemptions) if preemptions
                       and all(value is not None for value in preemptions)
                       else None)
    if guard is None or minimum_headroom is None:
        errors.append("source KV samples during preparation are missing")
    elif minimum_headroom <= 0:
        errors.append("source KV headroom reached the guard before TP1 KV release")
    if max_preemptions is None:
        errors.append("source preemption samples during preparation are missing")
    elif max_preemptions > 0:
        errors.append("source preemption occurred before TP1 KV release")
    if acceptance.get("status") != "PASS":
        errors.append("online migration acceptance did not pass")
    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "minimum_output_tokens": minimum_output_tokens,
        "actual_start_output_tokens": output,
        "anchor_first_output_unix_s": first_output_s,
        "m1_start_unix_s": started_s,
        "active_source_peers_at_start": active_at_start,
        "source_kv_released_unix_s": release_s,
        "start_to_source_kv_release_s": (
            release_s - started_s if release_s is not None
            and started_s is not None else None
        ),
        "source_guard_tokens": guard,
        "minimum_sampled_source_free_tokens_before_release": minimum_free,
        "minimum_reserved_headroom_tokens_before_release": minimum_headroom,
        "maximum_source_preemptions_before_release": max_preemptions,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--minimum-output-tokens", type=int, required=True)
    args = parser.parse_args()
    if args.minimum_output_tokens < 0:
        parser.error("minimum output tokens must be nonnegative")
    result = audit(args.run_dir, args.minimum_output_tokens)
    path = args.run_dir / "late_start_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
