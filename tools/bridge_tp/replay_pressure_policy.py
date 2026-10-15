#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only new-policy replay and stop-aware saved-PMF validation.

Snapshot JSONL format: unix_s, source/target PoolReading objects, candidates
Candidate dictionaries (remaining omitted), and per-request M5 advisories.
Historical missing inventories remain missing; the CLI does not synthesize them.
No CUDA imports, START publication, or network requests are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bridge_tp.pressure_cpu import load  # noqa: E402

b = load("bounded_length")
p = load("pressure_policy")
r = load("pressure_runtime")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_snapshots(path, cfg, output):
    decisions = []
    coordinator = r.PressureCoordinator(cfg, decisions.append)
    last = -float("inf")
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            now = event["unix_s"]
            if now < last:
                raise ValueError("snapshot decision time regressed")
            last = now
            candidates = []
            for raw in event["candidates"]:
                c = p.Candidate(**(raw | {"remaining": None}))
                advisory = event.get("advisories", {}).get(c.request_id)
                if advisory and advisory.get("status") == "AVAILABLE":
                    captured = advisory["captured_unix_ns"] / 1e9
                    published = advisory["published_unix_ns"] / 1e9
                    position = advisory["prediction_output_tokens"]
                    if (
                        0 <= now - captured <= cfg.max_age_s
                        and published <= now
                        and published >= captured
                        and 0 <= c.generated - position <= 40
                        and advisory["request_id"] == c.request_id
                        and advisory["output_tokens"] == c.generated
                    ):
                        c = replace(
                            c,
                            remaining=b.from_advisory(advisory, c.max_output_tokens),
                            prediction_status="AVAILABLE",
                        )
                    else:
                        c = replace(c, prediction_status="STALE_OR_FUTURE")
                candidates.append(c)
            coordinator.tick(
                now,
                p.PoolReading(**event["source"]),
                p.PoolReading(**event["target"]),
                candidates,
            )
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in decisions), encoding="utf-8"
    )
    return {
        "ticks": len(decisions),
        "actions": dict(Counter(row["decision"]["action"] for row in decisions)),
        "execution": "ADVISORY_ONLY_NO_COUNTERFACTUAL_POOL_STATE",
    }


def validate_saved_pmf(events, contracts):
    """Use only actual stop contracts supplied with explicit evidence references."""
    contract_rows = json.loads(contracts.read_text(encoding="utf-8"))
    states = failures = skipped = 0
    max_mass_error = 0.0
    reasons = Counter()
    with events.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        edges = tuple(header["category_upper_edges"])
        expected_sha = (
            "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7"
        )
        if header["checkpoint_sha256"] != expected_sha or len(edges) != 75:
            raise ValueError("saved PMF is not frozen four-layer76 identity")
        for line in stream:
            row = json.loads(line)
            if row["kind"] != "predictor_prediction":
                continue
            contract = contract_rows.get(row["request_id"])
            if not contract or not contract.get("evidence"):
                skipped += 1
                continue
            if row["checkpoint_sha256"] != expected_sha:
                raise ValueError("checkpoint changed within journal")
            if row.get("published_unix_ns", 0) < row["captured_unix_ns"]:
                raise ValueError("missing or invalid prediction publication")
            # Publication time is the replay epoch. No future states or final
            # output truth are used. Lag offsets are contract stress checks,
            # not an assertion that these generated positions occurred online.
            for lag in (0, 1, 20, 40):
                generated = row["generated_tokens"] + lag
                x = b.bounded_length(
                    tuple(row["probabilities"]),
                    edges,
                    generated=generated,
                    predicted_at=row["generated_tokens"],
                    max_output_tokens=contract["max_output_tokens"],
                    ignore_eos=contract["ignore_eos"],
                )
                states += 1
                error = abs(sum(x.pmf) - 1)
                max_mass_error = max(max_mass_error, error)
                if error > 1e-9 or not 0 <= x.mean <= x.cap:
                    failures += 1
                    reasons["mass_or_support"] += 1
                for h in (0, 256, x.cap - 1, x.cap):
                    lo, hi = x.probability_gt_bounds(h)
                    approximate = x.probability_gt(h)
                    if not lo - 1e-10 <= approximate <= hi + 1e-10:
                        failures += 1
                        reasons["risk_bound"] += 1
    if states == 0:
        raise ValueError("no saved PMF had an evidenced runtime stop contract")
    return {
        "prediction_journal_sha256": sha(events),
        "contracts_sha256": sha(contracts),
        "checkpoint_sha256": expected_sha,
        "transforms": states,
        "lag_stress_offsets": [0, 1, 20, 40],
        "unmapped_predictions": skipped,
        "failures": failures,
        "failure_reasons": dict(reasons),
        "max_mass_error": max_mass_error,
        "validation": "saved PMF algebra, not inference or closed-loop validation",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["pressure-balanced-v1"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path)
    parser.add_argument("--predictor-events", type=Path)
    parser.add_argument("--stop-contracts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.snapshots and not args.predictor_events:
        parser.error("supply snapshots and/or saved predictor events")
    if bool(args.predictor_events) != bool(args.stop_contracts):
        parser.error("saved PMF requires evidenced actual stop contracts")
    cfg = p.PressureConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    result = {
        "policy": p.VERSION,
        "config_sha256": cfg.sha256,
        "config_status": "DRAFT_NOT_OPTIMAL_OR_GPU_VALIDATED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshots:
        result["snapshots"] = replay_snapshots(
            args.snapshots, cfg, args.output.with_suffix(".jsonl")
        )
    if args.predictor_events:
        result["saved_pmf"] = validate_saved_pmf(
            args.predictor_events, args.stop_contracts
        )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return int(result.get("saved_pmf", {}).get("failures", 0) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
