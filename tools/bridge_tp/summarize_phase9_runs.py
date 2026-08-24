#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate several Phase 9 runs into the numbers the paper reports.

Phase 9 pass condition 6 requires at least three runs on one platform, so this
never reports a single run without saying so. It emits median and full range
rather than a bare mean, because with n=3 a mean hides everything.

    python tools/bridge_tp/summarize_phase9_runs.py \
        --runs runs/e2_seed0 runs/e2_seed1 runs/e2_seed2 \
        --label "E2 single-request commit" \
        --out results/e2_summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.bridge_tp.controller.audit import read_audit  # noqa: E402

MIN_RUNS_FOR_A_CLAIM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def extract(run: Path) -> dict | None:
    audit = run / "phase9_audit.jsonl"
    if not audit.exists():
        return None
    records = list(read_audit(audit))
    end = next((r for r in reversed(records) if r.get("kind") == "run_end"), None)
    if end is None:
        return None

    decisions = [r for r in records if r.get("kind") == "decision"]
    rates = [r for r in records if r.get("kind") == "rate"]
    escalations = [d for d in decisions if d.get("action") == "START_SHADOW"]

    row: dict = {
        "run": str(run),
        "final_state": end.get("final_state"),
        "ticks": end.get("ticks"),
        "decisions": len(decisions),
        "escalations": len(escalations),
        "forced_escalations": sum(1 for d in escalations if d.get("forced")),
    }
    if end.get("t_cutover") and end.get("t_committed"):
        row["cutover_to_commit_s"] = end["t_committed"] - end["t_cutover"]
    if end.get("t_shadow_start") and end.get("t_cutover"):
        row["shadow_duration_s"] = end["t_cutover"] - end["t_shadow_start"]
    if rates:
        row["rate_gib_s_final"] = rates[-1].get("rate_gib_s")
        row["rate_gib_s_max"] = max(r.get("rate_gib_s", 0.0) for r in rates)
        row["rate_backoffs"] = sum(
            1 for r in rates if "backoff" in str(r.get("reason", ""))
        )
        row["rate_deadline_overrides"] = sum(
            1 for r in rates if "deadline override" in str(r.get("reason", ""))
        )

    stats_path = run / "response_proxy_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        row["proxy_mode"] = stats.get("mode")
        row["handoff_stall_s"] = stats.get("handoff_stall_s")
        row["emitted_tokens"] = stats.get("emitted_tokens")
        row["discarded_source_tokens"] = stats.get("discarded_source_tokens")
    return row


def summarize(values: list[float]) -> dict:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return {}
    return {
        "n": len(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    rows = [r for r in (extract(run) for run in args.runs) if r is not None]
    missing = len(args.runs) - len(rows)

    metrics = [
        "handoff_stall_s",
        "cutover_to_commit_s",
        "shadow_duration_s",
        "rate_gib_s_final",
        "rate_gib_s_max",
        "discarded_source_tokens",
    ]
    aggregate = {m: summarize([r.get(m) for r in rows]) for m in metrics}
    aggregate = {k: v for k, v in aggregate.items() if v}

    outcomes: dict[str, int] = {}
    for row in rows:
        outcomes[row["final_state"]] = outcomes.get(row["final_state"], 0) + 1

    warnings = []
    if len(rows) < MIN_RUNS_FOR_A_CLAIM:
        warnings.append(
            f"only {len(rows)} usable run(s); Phase 9 pass condition 6 requires at "
            f"least {MIN_RUNS_FOR_A_CLAIM} on one platform before reporting a number"
        )
    if missing:
        warnings.append(f"{missing} run directory/directories had no usable audit log")
    modes = {r.get("proxy_mode") for r in rows if r.get("proxy_mode")}
    if len(modes) > 1:
        warnings.append(
            f"runs mix response-proxy modes {sorted(modes)}; handoff stall is not "
            "comparable across modes and must be reported separately"
        )

    result = {
        "label": args.label,
        "runs": rows,
        "outcomes": outcomes,
        "aggregate": aggregate,
        "warnings": warnings,
    }

    print(f"== {args.label or 'Phase 9 summary'} ==")
    print(f"runs: {len(rows)} usable  outcomes: {outcomes}")
    for name, stats in aggregate.items():
        print(
            f"  {name:<28} median={stats['median']:.4g}  "
            f"range=[{stats['min']:.4g}, {stats['max']:.4g}]  n={stats['n']}"
        )
    for warning in warnings:
        print(f"  WARNING: {warning}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
