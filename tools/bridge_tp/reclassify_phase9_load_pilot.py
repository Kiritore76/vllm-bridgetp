#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reclassify existing rate-zero pilot telemetry under frozen load bands."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from run_phase9_interference_sweep import (  # noqa: E402
    BANDS,
    PROTOCOL_AMENDMENT,
    serialized_bands,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stability-window-s", type=float, default=120.0)
    parser.add_argument("--measurement-window-s", type=float, default=300.0)
    parser.add_argument("--min-band-fraction", type=float, default=0.80)
    args = parser.parse_args()
    if args.stability_window_s <= 0 or args.measurement_window_s <= 0:
        parser.error("window durations must be positive")
    if not 0 < args.min_band_fraction <= 1:
        parser.error("min-band-fraction must be in (0,1]")
    return args


def read_telemetry(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        parsed = []
        for row in rows:
            try:
                parsed.append(
                    (float(row["monotonic_s"]), float(row["kv_usage_frac"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(parsed)


def summarize(values: list[float], duration_s: float) -> dict[str, object]:
    return {
        "samples": len(values),
        "window_s": duration_s,
        "kv_usage_mean": statistics.fmean(values),
        "kv_usage_min": min(values),
        "kv_usage_max": max(values),
        "band_fractions": {
            name: sum(low <= value <= high for value in values) / len(values)
            for name, (low, high) in BANDS.items()
        },
    }


def window_values(
    rows: list[tuple[float, float]], start: float, end: float
) -> tuple[list[float], float]:
    selected = [(time_s, value) for time_s, value in rows if start <= time_s <= end]
    if len(selected) < 2:
        return [], 0.0
    return [value for _, value in selected], selected[-1][0] - selected[0][0]


def compliant(summary: dict[str, object], band: str, fraction: float) -> bool:
    low, high = BANDS[band]
    return (
        low <= float(summary["kv_usage_mean"]) <= high
        and float(summary["band_fractions"][band]) >= fraction
    )


def find_qualifying_window(
    rows: list[tuple[float, float]],
    band: str,
    stability_window_s: float,
    measurement_window_s: float,
    min_band_fraction: float,
) -> dict[str, object] | None:
    if len(rows) < 2:
        return None
    interval_s = statistics.median(
        right[0] - left[0] for left, right in zip(rows, rows[1:])
    )
    tolerance_s = max(2.0, 2.0 * interval_s)
    final_time = rows[-1][0]
    for boundary_time, _ in rows:
        if boundary_time + measurement_window_s > final_time + tolerance_s:
            break
        before, before_coverage = window_values(
            rows, boundary_time - stability_window_s, boundary_time
        )
        after, after_coverage = window_values(
            rows, boundary_time, boundary_time + measurement_window_s
        )
        if (
            before_coverage < stability_window_s - tolerance_s
            or after_coverage < measurement_window_s - tolerance_s
        ):
            continue
        stability = summarize(before, stability_window_s)
        measurement = summarize(after, measurement_window_s)
        if compliant(stability, band, min_band_fraction) and compliant(
            measurement, band, min_band_fraction
        ):
            return {
                "band": band,
                "boundary_monotonic_s": boundary_time,
                "stability": stability,
                "measurement": measurement,
            }
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    evaluated = []
    candidates = {band: [] for band in BANDS}
    hashed_sources = []

    for root in args.pilot_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"pilot root does not exist: {root}")
        for manifest_path in sorted(root.rglob("condition_manifest.json")):
            condition_dir = manifest_path.parent
            telemetry_path = condition_dir / "telemetry.csv"
            if not telemetry_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rate = float(manifest.get("target_rate_gib_s", -1))
            if rate != 0.0:
                continue
            qps = float(manifest["qps"])
            rows = read_telemetry(telemetry_path)
            record = {
                "condition_dir": str(condition_dir),
                "qps": qps,
                "telemetry_rows": len(rows),
                "qualifying_bands": {},
            }
            for band in BANDS:
                result = find_qualifying_window(
                    rows,
                    band,
                    args.stability_window_s,
                    args.measurement_window_s,
                    args.min_band_fraction,
                )
                record["qualifying_bands"][band] = result
                if result is not None:
                    candidate = {
                        "condition_dir": str(condition_dir),
                        "qps": qps,
                        "stability_status": "STABLE",
                        "matched_band": band,
                        "offline_reclassified": True,
                        "boundary_monotonic_s": result["boundary_monotonic_s"],
                        "pre_copy_stability": result["stability"],
                        **result["measurement"],
                    }
                    candidates[band].append(candidate)
            evaluated.append(record)
            hashed_sources.extend((manifest_path, telemetry_path))

    selected = {}
    for band, (low, high) in BANDS.items():
        midpoint = (low + high) / 2.0
        options = candidates[band]
        options.sort(
            key=lambda item: (
                abs(float(item["kv_usage_mean"]) - midpoint),
                float(item["qps"]),
                str(item["condition_dir"]),
            )
        )
        selected[band] = options[0] if options else None

    payload = {
        "format_version": 1,
        "phase": "BridgeTP Phase 9 Section 7.2 load pilot reclassification",
        "status": (
            "READY" if all(value is not None for value in selected.values())
            else "MISSING_ATTAINABLE_BAND_EVIDENCE"
        ),
        "platform_scope": "NVIDIA A100 PCIe only",
        "workload_scope": "TP4 I256/O2048 rate-zero pilot telemetry",
        "load_bands": serialized_bands(),
        "stability_window_s": args.stability_window_s,
        "measurement_window_s": args.measurement_window_s,
        "min_band_fraction": args.min_band_fraction,
        "protocol_amendment": PROTOCOL_AMENDMENT,
        "source_roots": [str(path) for path in args.pilot_roots],
        "evaluated_conditions": evaluated,
        "selected": selected,
        "formal_model_conclusion": None,
        "evidence_boundary": (
            "This is a deterministic reclassification of pre-formal, "
            "rate-zero telemetry. It does not import high-QPS reachability "
            "diagnostics or any nonzero-copy outcome into the formal fit."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hash_path = args.out.with_name(args.out.stem + "_SHA256SUMS")
    paths = sorted(set(hashed_sources), key=str) + [args.out]
    hash_path.write_text(
        "\n".join(f"{sha256(path)}  {path}" for path in paths) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "selected": selected}, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {hash_path}")
    if payload["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
