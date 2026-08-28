#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reclassify old low runs and collect one accepted Phase 9 36-cell grid."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ANALYZER = TOOLS_DIR / "analyze_phase9_calibration.py"
SUMMARIZER = TOOLS_DIR / "summarize_phase9_calibration.py"
BANDS = {"low": (0.10, 0.25), "medium": (0.30, 0.45), "high": (0.50, 0.65)}
RATES = (0.0, 0.4, 0.7, 1.2)
REPS = (1, 2, 3)
FormalKey = tuple[str, float, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-root", type=Path, required=True)
    parser.add_argument("--medium-high-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--measurement-max-kv-p95", type=float, default=0.85)
    parser.add_argument("--min-measurement-samples", type=int, default=290)
    args = parser.parse_args()
    if not 0 < args.measurement_max_kv_p95 <= 1:
        parser.error("measurement-max-kv-p95 must be in (0,1]")
    if args.min_measurement_samples <= 0:
        parser.error("min-measurement-samples must be positive")
    if args.out_root.resolve() in {
        args.low_root.resolve(),
        args.medium_high_root.resolve(),
    }:
        parser.error("out-root must differ from both input roots")
    return args


def result_key(payload: dict[str, object]) -> FormalKey:
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("condition result lacks inputs")
    return (
        str(payload.get("load_band", "")),
        float(inputs["target_rate_gib_s"]),
        int(payload.get("rep", 0)),
    )


def selection_score(item: tuple[Path, dict[str, object]]) -> tuple[float, float, str]:
    path, payload = item
    observed = payload.get("observed")
    if not isinstance(observed, dict):
        raise ValueError(f"condition result lacks observations: {path}")
    samples = float(observed.get("telemetry_samples", 0))
    rate_error = observed.get("rate_relative_error")
    numeric_error = 0.0 if rate_error is None else float(rate_error)
    return (-samples, numeric_error, str(path))


def run_low_reanalysis(
    source: Path,
    out: Path,
    max_kv_p95: float,
    min_samples: int,
) -> Path | None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("load_band") != "low":
        return None
    inputs = payload["inputs"]
    target_rate = float(inputs["target_rate_gib_s"])
    copy_json = inputs.get("copy_json")
    command = [
        sys.executable,
        str(ANALYZER),
        "--telemetry",
        str(inputs["telemetry"]),
        "--benchmark-json",
        str(inputs["benchmark_json"]),
        "--target-rate-gib-s",
        str(target_rate),
        "--load-min",
        str(BANDS["low"][0]),
        "--load-max",
        str(BANDS["low"][1]),
        "--min-band-fraction",
        "0.80",
        "--measurement-load-policy",
        "observed_safe",
        "--measurement-max-kv-p95",
        str(max_kv_p95),
        "--min-measurement-samples",
        str(min_samples),
        "--rate-relative-tolerance",
        "0.05",
        "--out",
        str(out),
    ]
    if copy_json:
        command.extend(("--copy-json", str(copy_json)))
    else:
        command.extend(
            (
                "--window-start-monotonic-s",
                str(inputs["window_start_monotonic_s"]),
                "--window-end-monotonic-s",
                str(inputs["window_end_monotonic_s"]),
            )
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (out.parent / "analysis.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    return out if out.exists() else None


def accepted_results(
    root: Path, bands: set[str]
) -> list[tuple[Path, dict[str, object]]]:
    found = []
    for path in sorted(root.glob("*/condition_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "ACCEPTED" and payload.get("load_band") in bands:
            found.append((path, payload))
    return found


def write_inventory(
    out: Path,
    selected: dict[FormalKey, tuple[Path, dict[str, object]]],
) -> None:
    fields = [
        "load_band",
        "target_rate_gib_s",
        "rep",
        "condition_id",
        "measurement_load_policy",
        "telemetry_samples",
        "kv_usage_mean",
        "kv_usage_p95",
        "load_band_fraction",
        "effective_rate_gib_s",
        "rate_relative_error",
        "mean_tpot_s",
        "p99_tpot_s",
        "p99_itl_s",
        "source_result",
        "source_condition_dir",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(selected):
            path, payload = selected[key]
            inputs = payload["inputs"]
            observed = payload["observed"]
            writer.writerow(
                {
                    "load_band": key[0],
                    "target_rate_gib_s": key[1],
                    "rep": key[2],
                    "condition_id": payload.get("condition_id"),
                    "measurement_load_policy": inputs.get(
                        "measurement_load_policy", "strict_band"
                    ),
                    "telemetry_samples": observed.get("telemetry_samples"),
                    "kv_usage_mean": observed.get("kv_usage_mean"),
                    "kv_usage_p95": observed.get("kv_usage_p95"),
                    "load_band_fraction": observed.get("load_band_fraction"),
                    "effective_rate_gib_s": observed.get(
                        "effective_rate_gib_s"
                    ),
                    "rate_relative_error": observed.get("rate_relative_error"),
                    "mean_tpot_s": observed.get("mean_tpot_s"),
                    "p99_tpot_s": observed.get("p99_tpot_s"),
                    "p99_itl_s": observed.get("p99_itl_s"),
                    "source_result": path,
                    "source_condition_dir": Path(inputs["telemetry"]).parent,
                }
            )


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    candidates: dict[FormalKey, list[tuple[Path, dict[str, object]]]] = defaultdict(list)
    reclassified_root = args.out_root / "reclassified_low"
    for source in sorted(args.low_root.glob("*/condition_result.json")):
        try:
            source_payload = json.loads(source.read_text(encoding="utf-8"))
            condition_id = str(source_payload.get("condition_id", source.parent.name))
            out = reclassified_root / condition_id / "condition_result.json"
            result = run_low_reanalysis(
                source,
                out,
                args.measurement_max_kv_p95,
                args.min_measurement_samples,
            )
            if result is None:
                continue
            payload = json.loads(result.read_text(encoding="utf-8"))
            if payload.get("status") == "ACCEPTED":
                candidates[result_key(payload)].append((result, payload))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"ignoring unusable low result {source}: {error}", file=sys.stderr)

    for item in accepted_results(args.medium_high_root, {"medium", "high"}):
        candidates[result_key(item[1])].append(item)

    expected = {
        (band, rate, rep)
        for band in BANDS
        for rate in RATES
        for rep in REPS
    }
    selected = {
        key: min(values, key=selection_score)
        for key, values in candidates.items()
        if key in expected
    }
    missing = sorted(expected - set(selected))
    selected_root = args.out_root / "selected_results"
    selected_root.mkdir(exist_ok=True)
    selected_paths = []
    for key in sorted(selected):
        source, _ = selected[key]
        band, rate, rep = key
        target = selected_root / f"{band}_rate{rate:g}_rep{rep}.json"
        shutil.copy2(source, target)
        selected_paths.append(target)

    write_inventory(args.out_root / "condition_inventory.csv", selected)
    raw_dirs = sorted(
        {
            str(Path(payload["inputs"]["telemetry"]).parent)
            for _, payload in selected.values()
        }
    )
    (args.out_root / "selected_condition_dirs.txt").write_text(
        "\n".join(raw_dirs) + ("\n" if raw_dirs else ""), encoding="utf-8"
    )
    manifest = {
        "format_version": 1,
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "expected_conditions": len(expected),
        "accepted_conditions": len(selected),
        "missing_conditions": [
            {"load_band": band, "target_rate_gib_s": rate, "rep": rep}
            for band, rate, rep in missing
        ],
        "candidate_counts": {
            f"{band}|{rate:g}|{rep}": len(candidates[(band, rate, rep)])
            for band, rate, rep in sorted(expected)
        },
        "selection_rule": (
            "ACCEPTED only; prefer the most complete telemetry window, then "
            "the smallest copy-rate error, then lexical source path"
        ),
        "low_root": str(args.low_root),
        "medium_high_root": str(args.medium_high_root),
        "selected_results": [str(path) for path in selected_paths],
    }
    manifest_out = args.out_root / "collection_manifest.json"
    manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(2)

    summary = args.out_root / "bridge_calibration_summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SUMMARIZER),
            "--inputs",
            *(str(path) for path in selected_paths),
            "--out",
            str(summary),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(f"collected complete 36-cell grid in {args.out_root}")


if __name__ == "__main__":
    main()
