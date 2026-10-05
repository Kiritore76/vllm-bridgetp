#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit pre-action migration states across independent natural-EOS runs.

The input is existing Phase 9 controller audit logs. This script never infers
counterfactual benefit, modifies decisions, or labels guard contact as OOM.
One representative decision per run is used for the 3x3 coverage matrix;
all LOCAL decisions are retained in observations.jsonl for later analysis.
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


def audit_path(run_root: Path) -> Path:
    direct = run_root / "phase9_audit.jsonl"
    nested = run_root / "controller" / "phase9_audit.jsonl"
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    raise ValueError(f"no controller audit in run root: {run_root}")


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _arrival_times(run_root: Path) -> dict[str, list[float] | None] | None:
    """Read actual request starts; absent logs stay unknown, not zero load."""
    background = run_root / "background" / "background_events.jsonl"
    if not background.is_file():
        return None
    result: dict[str, list[float] | None] = {"source": [], "target": []}
    with background.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "job_start" and row.get("pool") in result:
                started = finite_number(row.get("unix_s"))
                if started is not None:
                    assert result[row["pool"]] is not None
                    result[row["pool"]].append(started)
    anchor = run_root / "controller" / "source_response.json"
    started = None
    if anchor.is_file():
        try:
            source_response = json.loads(anchor.read_text(encoding="utf-8"))
            started = finite_number(source_response.get("request_started_unix_s"))
        except (OSError, ValueError, TypeError):
            pass
    if started is None:
        result["source"] = None
    else:
        assert result["source"] is not None
        result["source"].append(started)
    for times in result.values():
        if times is not None:
            times.sort()
    return result


def extract_run(
    run_root: Path, *, arrival_window_s: float = 10.0
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract LOCAL decisions and their preceding telemetry from one run."""
    if not math.isfinite(arrival_window_s) or arrival_window_s <= 0:
        raise ValueError("arrival window must be positive and finite")
    path = audit_path(run_root)
    audit_digest = sha256(path)
    arrivals = _arrival_times(run_root)
    telemetry: dict[int, dict[str, Any]] = {}
    latest_m5: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = []
    invalid_lines = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not isinstance(row, dict):
                invalid_lines += 1
                continue
            kind = row.get("kind")
            if kind == "telemetry":
                tick = _int(row.get("tick"))
                if tick is not None:
                    telemetry[tick] = row
            elif kind == "manager_m5_predictor_shadow":
                latest_m5 = row
            elif kind == "manager_m1_start_decision":
                tick = _int(row.get("tick"))
                snapshot = row.get("snapshot")
                decision = row.get("decision")
                if tick is None or not isinstance(snapshot, dict) or not isinstance(
                    decision, dict
                ):
                    invalid_lines += 1
                    continue
                if snapshot.get("state") != "LOCAL":
                    continue
                sample = telemetry.get(tick, {})
                if sample and sample.get("state") != "LOCAL":
                    raise ValueError(f"state mismatch at {path}:{line_number}")
                if sample and sample.get("output_tokens") != snapshot.get(
                    "generated_tokens"
                ):
                    raise ValueError(f"output mismatch at {path}:{line_number}")
                capacity = sample.get("capacity_signal") or {}
                if not isinstance(capacity, dict):
                    capacity = {}
                m5 = latest_m5 if latest_m5 and latest_m5.get(
                    "output_tokens"
                ) == snapshot.get("generated_tokens") else None
                source_free = _int(snapshot.get("source_free_kv_tokens"))
                guard = _int(snapshot.get("source_guard_free_kv_tokens"))
                pending = _int(snapshot.get("source_prefill_pending_kv_tokens"))
                headroom = (
                    source_free - guard - pending
                    if None not in (source_free, guard, pending) else None
                )
                growth = finite_number(snapshot.get("source_decode_growth_tokens_s"))
                observed_at = finite_number(snapshot.get("unix_s"))
                arrival_rates = {}
                for pool in ("source", "target"):
                    arrival_rates[pool] = (
                        sum(observed_at - arrival_window_s <= started < observed_at
                            for started in arrivals[pool]) / arrival_window_s
                        if arrivals is not None and arrivals[pool] is not None
                        and observed_at is not None
                        else None
                    )
                target_running = _int(snapshot.get("target_running"))
                target_waiting = _int(snapshot.get("target_waiting"))
                observation = {
                    "run_root": str(run_root.resolve()),
                    "audit_sha256": audit_digest,
                    "tick": tick,
                    "unix_s": observed_at,
                    "request_id": snapshot.get("request_id"),
                    "output_tokens": _int(snapshot.get("generated_tokens")),
                    "current_context_tokens": _int(snapshot.get(
                        "current_context_tokens"
                    )),
                    "source_running": _int(snapshot.get("source_running")),
                    "source_waiting": _int(snapshot.get("source_waiting")),
                    "source_headroom_tokens": headroom,
                    "source_decode_growth_tokens_s": growth,
                    "source_arrival_rate_rps": arrival_rates["source"],
                    "target_arrival_rate_rps": arrival_rates["target"],
                    "arrival_window_s": arrival_window_s,
                    "source_time_to_guard_s": finite_number(
                        decision.get("source_time_to_guard_s")
                    ),
                    "source_time_to_guard_trend_s": finite_number(
                        capacity.get("time_to_guard_s")
                    ),
                    "target_running": target_running,
                    "target_waiting": target_waiting,
                    "target_busy_count": (
                        target_running + target_waiting
                        if target_running is not None and target_waiting is not None
                        else None
                    ),
                    "target_kv_usage_frac": finite_number(
                        snapshot.get("target_kv_usage_frac")
                    ),
                    "target_p99_tpot_s": finite_number(
                        snapshot.get("target_p99_tpot_s")
                    ),
                    "target_tpot_samples": _int(snapshot.get(
                        "target_tpot_samples"
                    )),
                    "target_free_kv_tokens": _int(snapshot.get(
                        "target_free_kv_tokens"
                    )),
                    "channel_available": snapshot.get("channel_available"),
                    "initial_rate": row.get("initial_rate_preview"),
                    "m1_action": decision.get("action"),
                    "m1_reason": decision.get("reason"),
                    "m1_missing": decision.get("missing"),
                    "m1_expected_remaining_tokens": finite_number(
                        decision.get("expected_remaining_tokens")
                    ),
                    "m1_probability_remaining": finite_number(
                        decision.get("remaining_probability")
                    ),
                    "m1_estimated_preparation_s": finite_number(
                        decision.get("estimated_preparation_s")
                    ),
                    "m5_status": m5.get("status") if m5 else "NO_MATCHING_EVENT",
                    "m5_prediction_output_tokens": (
                        m5.get("prediction_output_tokens") if m5 else None
                    ),
                    "m5_p_remaining_gt_headroom_runtime_bounds": (
                        m5.get("p_remaining_gt_headroom_runtime_bounds")
                        if m5 else None
                    ),
                    "capacity_pressure": capacity.get("active"),
                    "capacity_transition": capacity.get("transition"),
                    "paired_stay_intervention": False,
                }
                observations.append(observation)
                latest_m5 = None
            elif kind == "paired_stay_intervention" and observations:
                if observations[-1]["tick"] == row.get("tick"):
                    observations[-1]["paired_stay_intervention"] = True
    return observations, {
        "run_root": str(run_root.resolve()),
        "audit_path": str(path.resolve()),
        "audit_sha256": audit_digest,
        "local_decisions": len(observations),
        "arrival_events_available": arrivals is not None,
        "invalid_or_partial_lines": invalid_lines,
    }


def tertile_cuts(values: list[float]) -> list[float] | None:
    """Return exploratory cut points, or unknown when ties collapse bins."""
    if len(values) < 3:
        return None
    ordered = sorted(values)
    first = ordered[len(ordered) // 3]
    second = ordered[(2 * len(ordered)) // 3]
    return [first, second] if first < second else None


def validate_cuts(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        raise ValueError("coverage cuts must be a JSON object")
    if "cuts" in value:
        value = value["cuts"]
    if not isinstance(value, dict):
        raise ValueError("coverage cuts must be a JSON object")
    result = {}
    for key in ("source_arrival_rate_rps", "target_busy_count"):
        limits = value.get(key)
        if not isinstance(limits, list) or len(limits) != 2:
            raise ValueError(f"missing coverage cuts: {key}")
        converted = [finite_number(item) for item in limits]
        if None in converted or converted[0] >= converted[1]:
            raise ValueError(f"invalid coverage cuts: {key}")
        result[key] = converted
    return result


def bin_name(value: float | None, cuts: list[float] | None) -> str:
    if value is None or cuts is None:
        return "UNKNOWN"
    if value < cuts[0]:
        return "LOW"
    if value < cuts[1]:
        return "MEDIUM"
    return "HIGH"


def summarize(
    observations_by_run: list[list[dict[str, Any]]],
    min_output_tokens: int,
    cuts: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Count independent episodes, never treating controller ticks as trials."""
    representatives = []
    for rows in observations_by_run:
        eligible = [
            row for row in rows
            if row["output_tokens"] is not None
            and row["output_tokens"] >= min_output_tokens
        ]
        if eligible:
            representatives.append(eligible[0])
    chosen = cuts or {
        "source_arrival_rate_rps": tertile_cuts([
            row["source_arrival_rate_rps"] for row in representatives
            if row["source_arrival_rate_rps"] is not None
        ]),
        "target_busy_count": tertile_cuts([
            float(row["target_busy_count"]) for row in representatives
            if row["target_busy_count"] is not None
        ]),
    }
    cells: Counter[str] = Counter()
    for row in representatives:
        source = bin_name(
            row["source_arrival_rate_rps"],
            chosen["source_arrival_rate_rps"],
        )
        target = bin_name(
            row["target_busy_count"], chosen["target_busy_count"]
        )
        cells[f"{source}/{target}"] += 1
    return {
        "format_version": 1,
        "status": "OBSERVATIONAL_COVERAGE_ONLY",
        "independent_runs": len(observations_by_run),
        "runs_with_candidate": len(representatives),
        "local_decision_ticks": sum(map(len, observations_by_run)),
        "representative_rule": f"first LOCAL decision at output >= {min_output_tokens}",
        "cuts": chosen,
        "cuts_source": (
            "provided_input" if cuts is not None else "exploratory_same_data"
        ),
        "unresolved_cut_axes": [
            key for key, limits in chosen.items() if limits is None
        ],
        "cells": dict(sorted(cells.items())),
        "unknown_source_metric": sum(
            row["source_arrival_rate_rps"] is None
            for row in representatives
        ),
        "unknown_target_metric": sum(
            row["target_busy_count"] is None for row in representatives
        ),
        "m1_actions": dict(sorted(Counter(
            str(row["m1_action"] or "UNKNOWN") for row in representatives
        ).items())),
        "m5_statuses": dict(sorted(Counter(
            row["m5_status"] for row in representatives
        ).items())),
        "guard_time_is_probability": False,
        "benefit_is_estimated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-output-tokens", type=int, default=32)
    parser.add_argument("--arrival-window-s", type=float, default=10.0)
    parser.add_argument("--cuts-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_output_tokens < 0:
        raise ValueError("minimum output tokens cannot be negative")
    if not math.isfinite(args.arrival_window_s) or args.arrival_window_s <= 0:
        raise ValueError("arrival window must be positive and finite")
    if args.out_dir.exists():
        raise ValueError(f"output directory already exists: {args.out_dir}")
    roots = [root.resolve() for root in args.run_root]
    if len(roots) != len(set(roots)):
        raise ValueError("duplicate run roots")
    rows = [extract_run(root, arrival_window_s=args.arrival_window_s)
            for root in roots]
    cuts = (
        validate_cuts(json.loads(args.cuts_json.read_text(encoding="utf-8")))
        if args.cuts_json else None
    )
    summary = summarize([item[0] for item in rows], args.min_output_tokens, cuts)
    summary["run_audits"] = [item[1] for item in rows]
    summary["arrival_window_s"] = args.arrival_window_s
    summary["cuts_json_path"] = (
        str(args.cuts_json.resolve()) if args.cuts_json else None
    )
    summary["cuts_json_sha256"] = sha256(args.cuts_json) if args.cuts_json else None
    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "observations.jsonl").open("w", encoding="utf-8") as out:
        for observations, _ in rows:
            for row in observations:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "coverage.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
