#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fit the Phase 9 rate-aware native-TP4 interference model.

The primary response is request-level mean TPOT.  Each non-zero-rate cell is
paired with the zero-rate observation from the same load band and repetition,
which removes the large baseline difference between low, medium, and high
load.  The fitted response is

    delta_tpot_s = rate_gib_s * (rate_coef + load_coef * kv_usage_mean)

with ``load_coef >= 0``.  The constraint is deliberate: a negative load
interaction would make the controller more aggressive as TP4 gets busier.
If the unconstrained data prefer a negative interaction, the auditable
conservative result is a rate-only model (``load_coef == 0``), not a reversed
controller response.  P99 ITL is reported only as a tail diagnostic and is
never substituted for request-level TPOT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_BANDS = ("low", "medium", "high")
EXPECTED_RATES = (0.0, 0.4, 0.7, 1.2)
EXPECTED_REPS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
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


def read_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    **raw,
                    "target_rate_gib_s": float(raw["target_rate_gib_s"]),
                    "rep": int(raw["rep"]),
                    "telemetry_samples": int(raw["telemetry_samples"]),
                    "kv_usage_mean": float(raw["kv_usage_mean"]),
                    "kv_usage_p95": float(raw["kv_usage_p95"]),
                    "effective_rate_gib_s": float(raw["effective_rate_gib_s"]),
                    "rate_relative_error": float(raw["rate_relative_error"]),
                    "mean_tpot_s": float(raw["mean_tpot_s"]),
                    "p99_tpot_s": float(raw["p99_tpot_s"]),
                    "p99_itl_s": float(raw["p99_itl_s"]),
                }
            )
    return rows


def validate(rows: list[dict[str, Any]], summary_path: Path | None) -> None:
    expected = {
        (band, rate, rep)
        for band in EXPECTED_BANDS
        for rate in EXPECTED_RATES
        for rep in EXPECTED_REPS
    }
    keys = {(row["load_band"], row["target_rate_gib_s"], row["rep"]) for row in rows}
    if len(rows) != 36 or keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(keys - expected)
        raise ValueError(
            f"inventory is not the frozen 36-cell grid; "
            f"rows={len(rows)} missing={missing} unexpected={unexpected}"
        )
    if any(row["measurement_load_policy"] != "observed_safe" for row in rows):
        raise ValueError("all cells must use the frozen observed_safe policy")
    if any(row["telemetry_samples"] < 290 for row in rows):
        raise ValueError("a cell has fewer than 290 measurement samples")
    if any(row["rate_relative_error"] > 0.05 for row in rows):
        raise ValueError("a cell exceeds the preregistered copy-rate tolerance")
    if summary_path is not None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE":
            raise ValueError("calibration summary is not COMPLETE")


def paired_observations(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    by_key = {
        (row["load_band"], row["target_rate_gib_s"], row["rep"]): row for row in rows
    }
    observations = []
    for band in EXPECTED_BANDS:
        for rep in EXPECTED_REPS:
            baseline = by_key[(band, 0.0, rep)]
            for rate in EXPECTED_RATES[1:]:
                row = by_key[(band, rate, rep)]
                observations.append(
                    {
                        "rate": row["effective_rate_gib_s"],
                        "load": row["kv_usage_mean"],
                        "delta_mean_tpot": (
                            row["mean_tpot_s"] - baseline["mean_tpot_s"]
                        ),
                        "delta_p99_itl": row["p99_itl_s"] - baseline["p99_itl_s"],
                    }
                )
    return observations


def solve_two_parameter(
    observations: list[dict[str, float]], response: str
) -> tuple[float, float]:
    xx = xy = yy = xz = yz = 0.0
    for row in observations:
        x = row["rate"]
        y = row["rate"] * row["load"]
        z = row[response]
        xx += x * x
        xy += x * y
        yy += y * y
        xz += x * z
        yz += y * z
    determinant = xx * yy - xy * xy
    if determinant <= 0:
        raise ValueError("singular rate/load design matrix")
    return (xz * yy - yz * xy) / determinant, (yz * xx - xz * xy) / determinant


def fit_metrics(
    observations: list[dict[str, float]],
    response: str,
    rate_coef: float,
    load_coef: float,
) -> dict[str, float]:
    actual = [row[response] for row in observations]
    predicted = [
        row["rate"] * (rate_coef + load_coef * row["load"]) for row in observations
    ]
    residual = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    total_zero = sum(a * a for a in actual)
    return {
        "r_squared_through_zero": 1.0 - residual / total_zero,
        "rmse_s": math.sqrt(residual / len(actual)),
        "max_abs_error_s": max(abs(a - p) for a, p in zip(actual, predicted)),
    }


def fit_primary(observations: list[dict[str, float]]) -> dict[str, Any]:
    unconstrained_rate, unconstrained_load = solve_two_parameter(
        observations, "delta_mean_tpot"
    )
    if unconstrained_load >= 0:
        rate_coef = unconstrained_rate
        load_coef = unconstrained_load
        constraint = "inactive"
    else:
        numerator = sum(row["rate"] * row["delta_mean_tpot"] for row in observations)
        denominator = sum(row["rate"] ** 2 for row in observations)
        rate_coef = numerator / denominator
        load_coef = 0.0
        constraint = "active_at_zero"
    return {
        "response": "paired request-level mean TPOT increment",
        "formula": "delta_tpot_s = rate_gib_s * (rate_coef + load_coef * load)",
        "rate_coef_s2_per_gib": rate_coef,
        "load_coef_s2_per_gib": load_coef,
        "nonnegative_load_constraint": constraint,
        "metrics": fit_metrics(
            observations,
            "delta_mean_tpot",
            rate_coef,
            load_coef,
        ),
        "unconstrained_diagnostic": {
            "rate_coef_s2_per_gib": unconstrained_rate,
            "load_coef_s2_per_gib": unconstrained_load,
            "metrics": fit_metrics(
                observations,
                "delta_mean_tpot",
                unconstrained_rate,
                unconstrained_load,
            ),
        },
    }


def main() -> None:
    args = parse_args()
    rows = read_inventory(args.inventory)
    validate(rows, args.summary)
    observations = paired_observations(rows)
    primary = fit_primary(observations)
    tail_rate, tail_load = solve_two_parameter(observations, "delta_p99_itl")
    payload = {
        "format_version": 1,
        "status": "RATE_AWARE_INTERFERENCE_CANDIDATE",
        "platform_scope": "NVIDIA A100 PCIe only",
        "primary_fit": primary,
        "tail_diagnostic_not_used_as_tpot": {
            "response": "paired P99 inter-token latency increment",
            "rate_coef_s2_per_gib": tail_rate,
            "load_coef_s2_per_gib": tail_load,
            "metrics": fit_metrics(observations, "delta_p99_itl", tail_rate, tail_load),
        },
        "controller_model": {
            "model_kind": "rate_aware_tpot",
            "tpot_rate_coef_s2_per_gib": primary["rate_coef_s2_per_gib"],
            "tpot_rate_load_coef_s2_per_gib": primary["load_coef_s2_per_gib"],
            "min_load_frac": 0.10,
            "max_load_frac": 0.65,
            "min_rate_gib_s": 0.40,
            "max_rate_gib_s": 1.20,
            "calibration_source": (
                "Phase 9 Section 7.2 observed_safe 36-cell grid; paired "
                "request-level mean TPOT; nonnegative load interaction"
            ),
        },
        "support": {
            "load_band": [0.10, 0.65],
            "copy_rate_gib_s": [0.40, 1.20],
            "conditions": len(rows),
            "paired_nonzero_observations": len(observations),
            "telemetry_samples_per_condition_min": min(
                row["telemetry_samples"] for row in rows
            ),
        },
        "integration_note": (
            "Within the measured range the request-level mean-TPOT response is "
            "approximately linear in copy rate. Multiplying that instantaneous "
            "response by the fixed-work drain duration therefore yields an "
            "approximately rate-invariant total interference dose. Equality "
            "across tested rates satisfies the replay requirement that increasing "
            "copy rate must not make migration more aggressive; native-tail SLO "
            "feedback remains the rate controller's separate safety signal."
        ),
        "provenance": {
            "inventory": str(args.inventory.resolve()),
            "inventory_sha256": sha256(args.inventory),
            "summary": str(args.summary.resolve()) if args.summary else None,
            "summary_sha256": sha256(args.summary) if args.summary else None,
            "fit_script_sha256": sha256(Path(__file__).resolve()),
            "git_revision": git_revision(),
        },
        "evidence_boundary": (
            "The primary fit uses request-level mean TPOT and does not treat ITL "
            "as TPOT. The measured unconstrained load interaction is retained as "
            "a diagnostic; the controller fit constrains it to be nonnegative. "
            "The candidate is not sufficient for formal replay until the Section "
            "7.1 TPOT models and C-3 survival table are frozen and hashed."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_entries = [
        (sha256(args.out), args.out.name),
        (sha256(args.inventory), str(args.inventory.resolve())),
        (sha256(Path(__file__).resolve()), str(Path(__file__).resolve())),
    ]
    if args.summary:
        manifest_entries.append((sha256(args.summary), str(args.summary.resolve())))
    hash_manifest = args.out.parent / "SHA256SUMS"
    hash_manifest.write_text(
        "".join(f"{digest}  {name}\n" for digest, name in manifest_entries),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    print(f"wrote {hash_manifest}")


if __name__ == "__main__":
    main()
