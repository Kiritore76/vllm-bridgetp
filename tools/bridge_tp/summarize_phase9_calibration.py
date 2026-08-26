#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize the complete Phase 9 bridge-calibration grid."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

EXPECTED_BANDS = ("low", "medium", "high")
EXPECTED_RATES = (0.0, 0.4, 0.7, 1.2)
EXPECTED_REPS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def key(payload: dict) -> tuple[str, float, int]:
    return (
        str(payload.get("load_band", "")),
        float(payload["inputs"]["target_rate_gib_s"]),
        int(payload.get("rep", 0)),
    )


def main() -> None:
    args = parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.inputs
    ]
    by_key: dict[tuple[str, float, int], list[dict]] = defaultdict(list)
    for payload in payloads:
        by_key[key(payload)].append(payload)

    expected = {
        (band, rate, rep)
        for band in EXPECTED_BANDS
        for rate in EXPECTED_RATES
        for rep in EXPECTED_REPS
    }
    observed = set(by_key)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = sorted(item for item, values in by_key.items() if len(values) != 1)
    rejected = sorted(
        item
        for item, values in by_key.items()
        if any(value.get("status") != "ACCEPTED" for value in values)
    )

    cells = []
    paired_monotonic_failures = []
    for band in EXPECTED_BANDS:
        baselines = {
            rep: by_key[(band, 0.0, rep)][0]
            for rep in EXPECTED_REPS
            if len(by_key.get((band, 0.0, rep), [])) == 1
        }
        previous_delta = None
        previous_rate = None
        for rate in EXPECTED_RATES:
            values = [
                by_key[(band, rate, rep)][0]
                for rep in EXPECTED_REPS
                if len(by_key.get((band, rate, rep), [])) == 1
            ]
            if len(values) != len(EXPECTED_REPS):
                continue
            observed_rows = [value["observed"] for value in values]
            deltas = [
                value["observed"]["p99_tpot_s"]
                - baselines[rep]["observed"]["p99_tpot_s"]
                for rep, value in zip(EXPECTED_REPS, values)
                if rep in baselines
            ]
            mean_delta = statistics.mean(deltas) if deltas else None
            if (
                previous_delta is not None
                and mean_delta is not None
                and mean_delta + 1e-12 < previous_delta
            ):
                paired_monotonic_failures.append(
                    {
                        "load_band": band,
                        "previous_rate_gib_s": previous_rate,
                        "rate_gib_s": rate,
                    }
                )
            if mean_delta is not None:
                previous_delta = mean_delta
                previous_rate = rate
            cells.append(
                {
                    "load_band": band,
                    "target_rate_gib_s": rate,
                    "repetitions": len(values),
                    "kv_usage_mean": statistics.mean(
                        row["kv_usage_mean"] for row in observed_rows
                    ),
                    "effective_rate_gib_s_mean": statistics.mean(
                        row["effective_rate_gib_s"] or 0.0
                        for row in observed_rows
                    ),
                    "p99_tpot_s_mean": statistics.mean(
                        row["p99_tpot_s"] for row in observed_rows
                    ),
                    "paired_delta_p99_tpot_s_mean": mean_delta,
                    "p99_itl_s_mean": statistics.mean(
                        row["p99_itl_s"] for row in observed_rows
                    ),
                }
            )

    complete = not (missing or unexpected or duplicates or rejected)
    result = {
        "format_version": 1,
        "phase": "BridgeTP Phase 9 conditional bridge calibration",
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "platform_scope": "NVIDIA A100 PCIe only",
        "expected_conditions": len(expected),
        "input_files": len(args.inputs),
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "rejected": rejected,
        "paired_rate_monotonicity": {
            "passed": not paired_monotonic_failures,
            "failures": paired_monotonic_failures,
            "note": (
                "Descriptive check only; a single non-monotone cell must be "
                "inspected rather than silently removed."
            ),
        },
        "cells": cells,
        "formal_model_conclusion": None,
        "evidence_boundary": (
            "This summary validates the frozen A100 PCIe calibration grid. "
            "It does not fit or approve the rate-aware controller model."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
