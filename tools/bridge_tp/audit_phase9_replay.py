#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit the formal Phase 9 policy-boundary replay gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--tpot-model", type=Path, required=True)
    parser.add_argument("--interference-model", type=Path, required=True)
    parser.add_argument("--survival-table", type=Path, required=True)
    parser.add_argument("--survival-source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_source_manifest(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        payload = json.loads(text)
        frozen = payload.get("status") == "FROZEN"
        source_hash = payload.get("source_trace_sha256")
        return {
            "source_trace_status": "FOUND" if frozen and source_hash else "MISSING",
            "source_trace_sha256": str(source_hash or "UNAVAILABLE"),
            "valid_requests": str(payload.get("total_requests", 0)),
            "train_requests": str(payload.get("train_requests", -1)),
        }
    result = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    **raw,
                    "produced_tokens": int(raw["produced_tokens"]),
                    "target_load": float(raw["target_load"]),
                    "rate_gib_s": float(raw["rate_gib_s"]),
                    "survival_in_support": raw["survival_in_support"] == "True",
                    "interference_in_support": (
                        raw["interference_in_support"] == "True"
                    ),
                    "tpot_in_support": raw["tpot_in_support"] == "True",
                    "break_even_tokens": float(raw["break_even_tokens"]),
                }
            )
    return rows


def monotonic_load(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    groups = {}
    for row in rows:
        key = (row["produced_tokens"], row["rate_gib_s"])
        groups.setdefault(key, []).append(row)
    comparable = 0
    for key, group in groups.items():
        supported = sorted(
            (row for row in group if row["interference_in_support"]),
            key=lambda row: row["target_load"],
        )
        for left, right in zip(supported, supported[1:]):
            if math.isfinite(left["break_even_tokens"]) and math.isfinite(
                right["break_even_tokens"]
            ):
                comparable += 1
                if right["break_even_tokens"] + 1e-9 < left["break_even_tokens"]:
                    failures.append(
                        {
                            "group": key,
                            "left_load": left["target_load"],
                            "right_load": right["target_load"],
                        }
                    )
    return {
        "status": (
            "NOT_APPLICABLE_NO_FINITE_BOUNDARY"
            if comparable == 0
            else ("PASS" if not failures else "FAIL")
        ),
        "comparable_pairs": comparable,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    rows = read_rows(args.replay)
    source = read_source_manifest(args.survival_source_manifest)
    supported = [
        row
        for row in rows
        if row["survival_in_support"]
        and row["interference_in_support"]
        and row["tpot_in_support"]
    ]
    action_counts = Counter(row["action"] for row in rows)
    supported_action_counts = Counter(row["action"] for row in supported)
    finite = [row for row in supported if math.isfinite(row["break_even_tokens"])]
    source_frozen = source.get("source_trace_status") == "FOUND" and source.get(
        "source_trace_sha256"
    ) not in {None, "", "UNAVAILABLE"}
    valid_requests = int(source.get("valid_requests", "0"))
    expected_train_rows = int(valid_requests * 0.70)
    selected_train_rows = int(source.get("train_requests", "-1"))
    split_matches = selected_train_rows == expected_train_rows
    gate_checks = {
        "expected_90_rows": len(rows) == 90,
        "has_migrate_and_stay_in_supported_region": (
            supported_action_counts["START_SHADOW"] > 0
            and supported_action_counts["STAY"] > 0
        ),
        "has_finite_boundary_in_supported_region": bool(finite),
        "survival_source_frozen": source_frozen,
        "survival_split_matches_floor_70pct": split_matches,
    }
    payload = {
        "format_version": 1,
        "status": "PASS" if all(gate_checks.values()) else "FAILED",
        "gate_checks": gate_checks,
        "rows": len(rows),
        "supported_rows": len(supported),
        "finite_boundary_rows": len(finite),
        "action_counts": dict(action_counts),
        "supported_action_counts": dict(supported_action_counts),
        "survival_split": {
            "valid_requests": valid_requests,
            "expected_floor_70pct_train_rows": expected_train_rows,
            "selected_train_rows": selected_train_rows,
        },
        "reason_counts": dict(Counter(row["reason"] for row in rows)),
        "load_monotonicity": monotonic_load(rows),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in (
                ("replay", args.replay),
                ("tpot_model", args.tpot_model),
                ("interference_model", args.interference_model),
                ("survival_table", args.survival_table),
                ("survival_source_manifest", args.survival_source_manifest),
            )
        },
        "git_revision": git_revision(),
        "evidence_boundary": (
            "FAILED is a stop signal, not permission to tune policy thresholds "
            "against this replay. A formal PASS also requires the original "
            "survival source trace hash and both migrate/stay regions within all "
            "model support ranges."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
