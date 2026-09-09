#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run fixed migration-rate by TP4-load online Shadow comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402
from tools.bridge_tp.build_shadow_strategy_online_manifest import (  # noqa: E402
    build_manifest,
)

ONLINE_RUNNER = REPO / "tools" / "bridge_tp" / (
    "run_shadow_strategy_online_validation.py"
)
DEFAULT_LOADS = {
    "smoke": (("low", 2), ("high", 16)),
    "formal": (("low", 2), ("medium", 8), ("high", 24)),
}
DEFAULT_RATES = {
    "smoke": (0.4, 1.2),
    "formal": (0.2, 0.4, 0.8, 1.2, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--survival-table", type=Path, required=True)
    parser.add_argument("--guard-file", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-survival-sha256", required=True)
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--expected-guard", type=int, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--tp1-blocks", type=int, required=True)
    parser.add_argument("--tp4-blocks", type=int, required=True)
    parser.add_argument("--phase", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument(
        "--load-profile",
        action="append",
        help="Repeatable LABEL:TARGET_JOBS value; defaults depend on phase.",
    )
    parser.add_argument("--rates-gib-s", type=float, nargs="+")
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument("--trigger-output-tokens", type=int, default=128)
    parser.add_argument("--cutover-output-tokens", type=int, default=160)
    parser.add_argument("--anchor-max-tokens", type=int, default=1024)
    parser.add_argument("--background-lead-s", type=float, default=3.0)
    parser.add_argument("--minimum-window-samples", type=int)
    parser.add_argument("--tp1-gpu", default="0")
    parser.add_argument("--tp4-gpus", default="1,2,3,4")
    parser.add_argument("--tp1-port", type=int, default=8001)
    parser.add_argument("--tp4-port", type=int, default=8200)
    parser.add_argument("--snapshot-port", type=int, default=29800)
    parser.add_argument("--delta-port", type=int, default=29900)
    parser.add_argument("--delivery-port", type=int, default=30000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--server-start-timeout-s", type=float, default=900)
    parser.add_argument("--run-timeout-s", type=float, default=2400)
    parser.add_argument("--stager-timeout-s", type=float, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def parse_load_profiles(
    values: list[str] | None, phase: str
) -> list[tuple[str, int]]:
    if not values:
        return list(DEFAULT_LOADS[phase])
    profiles: list[tuple[str, int]] = []
    for value in values:
        try:
            label, jobs_text = value.split(":", 1)
            jobs = int(jobs_text)
        except ValueError as error:
            raise ValueError(
                f"invalid load profile {value!r}; expected LABEL:TARGET_JOBS"
            ) from error
        if not label or not label.replace("_", "").isalnum():
            raise ValueError(f"invalid load label {label!r}")
        if jobs < 2:
            raise ValueError("each load profile requires at least two jobs")
        profiles.append((label, jobs))
    if len({label for label, _ in profiles}) != len(profiles):
        raise ValueError("load profile labels must be unique")
    return profiles


def rate_label(rate: float) -> str:
    if rate == 0:
        return "unlimited"
    return f"{rate:g}gibs".replace(".", "p")


def resolve_design(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, int]], list[float], int, int]:
    loads = parse_load_profiles(args.load_profile, args.phase)
    rates = list(
        DEFAULT_RATES[args.phase]
        if args.rates_gib_s is None
        else args.rates_gib_s
    )
    if not rates or any(rate < 0 for rate in rates):
        raise ValueError("rates must contain non-negative GiB/s values")
    if len(set(rates)) != len(rates):
        raise ValueError("fixed rates must be unique")
    repetitions = args.repetitions
    if repetitions is None:
        repetitions = 1 if args.phase == "smoke" else 4
    if repetitions <= 0 or (args.phase == "formal" and repetitions < 3):
        raise ValueError("formal needs at least three pairs; smoke needs one")
    minimum_samples = args.minimum_window_samples
    if minimum_samples is None:
        minimum_samples = 16 if args.phase == "smoke" else 128
    if minimum_samples <= 0:
        raise ValueError("minimum window samples must be positive")
    return loads, rates, repetitions, minimum_samples


def validate_shared_inputs(args: argparse.Namespace) -> tuple[str, int]:
    if os.name == "nt":
        raise RuntimeError("online rate/load matrix requires Linux and five GPUs")
    for label, path in {
        "Python executable": args.python_bin,
        "model": args.model_path,
        "survival table": args.survival_table,
        "guard file": args.guard_file,
        "online runner": ONLINE_RUNNER,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    revision = common.git("rev-parse", "HEAD")
    expected = common.git("rev-parse", args.expected_revision)
    if revision != expected:
        raise RuntimeError(f"HEAD {revision} differs from expected {expected}")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--"]
    ).returncode != 0:
        raise RuntimeError("tracked working-tree changes are present")
    if common.sha256(args.survival_table) != args.expected_survival_sha256:
        raise RuntimeError("survival table SHA-256 differs from expected")
    if common.sha256(args.guard_file) != args.expected_guard_sha256:
        raise RuntimeError("guard file SHA-256 differs from expected")
    guard = int(args.guard_file.read_text(encoding="utf-8").strip())
    if guard != args.expected_guard:
        raise RuntimeError(f"frozen guard {guard} differs from expected")
    if args.prompt_tokens + args.output_tokens > args.max_model_len:
        raise ValueError("target context exceeds max model length")
    if not 0 < args.trigger_output_tokens < args.cutover_output_tokens:
        raise ValueError("trigger/cutover boundaries are invalid")
    return revision, guard


def online_command(
    args: argparse.Namespace,
    *,
    revision: str,
    manifest: Path,
    manifest_sha256: str,
    cell_root: Path,
    target_jobs: int,
    rate: float,
    repetitions: int,
    minimum_samples: int,
) -> list[str]:
    return [
        str(args.python_bin),
        str(ONLINE_RUNNER),
        "--phase",
        args.phase,
        "--repetitions",
        str(repetitions),
        "--fixed-rate-gib-s",
        str(rate),
        "--model-path",
        str(args.model_path),
        "--manifest",
        str(manifest),
        "--survival-table",
        str(args.survival_table),
        "--guard-file",
        str(args.guard_file),
        "--out-root",
        str(cell_root),
        "--expected-revision",
        revision,
        "--expected-manifest-sha256",
        manifest_sha256,
        "--expected-survival-sha256",
        args.expected_survival_sha256,
        "--expected-guard-sha256",
        args.expected_guard_sha256,
        "--expected-guard",
        str(args.expected_guard),
        "--python-bin",
        str(args.python_bin),
        "--tp1-blocks",
        str(args.tp1_blocks),
        "--tp4-blocks",
        str(args.tp4_blocks),
        "--trigger-output-tokens",
        str(args.trigger_output_tokens),
        "--cutover-output-tokens",
        str(args.cutover_output_tokens),
        "--anchor-max-tokens",
        str(args.anchor_max_tokens),
        "--minimum-ready-target-jobs",
        str(target_jobs),
        "--background-lead-s",
        str(args.background_lead_s),
        "--minimum-window-samples",
        str(minimum_samples),
        "--tp1-gpu",
        args.tp1_gpu,
        "--tp4-gpus",
        args.tp4_gpus,
        "--tp1-port",
        str(args.tp1_port),
        "--tp4-port",
        str(args.tp4_port),
        "--snapshot-port",
        str(args.snapshot_port),
        "--delta-port",
        str(args.delta_port),
        "--delivery-port",
        str(args.delivery_port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--server-start-timeout-s",
        str(args.server_start_timeout_s),
        "--run-timeout-s",
        str(args.run_timeout_s),
        "--stager-timeout-s",
        str(args.stager_timeout_s),
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_matrix(out_root: Path, cells: list[dict[str, Any]]) -> None:
    measurements: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for cell in cells:
        prefix = {
            "load_label": cell["load_label"],
            "target_jobs": cell["target_jobs"],
            "fixed_rate_gib_s": cell["fixed_rate_gib_s"],
        }
        cell_root = Path(cell["root"])
        for row in read_csv(cell_root / "measurements.csv"):
            measurements.append({**prefix, **row})
        for row in read_csv(cell_root / "paired_comparisons.csv"):
            paired.append({**prefix, **row})
    write_csv(out_root / "matrix_measurements.csv", measurements)
    write_csv(out_root / "matrix_paired_comparisons.csv", paired)


def main() -> None:
    args = parse_args()
    loads, rates, repetitions, minimum_samples = resolve_design(args)
    revision, guard = validate_shared_inputs(args)
    plan = {
        "format_version": 1,
        "phase": args.phase,
        "revision": revision,
        "guard_free_kv_tokens": guard,
        "loads": [
            {"label": label, "target_jobs": jobs} for label, jobs in loads
        ],
        "fixed_rates_gib_s": rates,
        "repetitions_per_cell": repetitions,
        "expected_cells": len(loads) * len(rates),
        "expected_runs": len(loads) * len(rates) * repetitions * 2,
        "minimum_window_samples": minimum_samples,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "evidence_boundary": "online Phase 8 takeover without remote attention",
    }
    if args.validate_only:
        print(json.dumps({"status": "VALID", "plan": plan}, indent=2))
        return

    out_root = args.out_root.resolve()
    if out_root.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse output root {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = out_root / "input_manifests"
    manifest_dir.mkdir(exist_ok=True)
    common.write_json(out_root / "contract.json", plan)
    batch: dict[str, Any] = {
        "format_version": 1,
        "status": "RUNNING",
        "started_unix_s": time.time(),
        "plan": plan,
        "cells": [],
    }

    for load_label, target_jobs in loads:
        manifest = manifest_dir / f"{load_label}.json"
        if not manifest.exists():
            payload = build_manifest(
                target_jobs=target_jobs,
                prompt_tokens=args.prompt_tokens,
                output_tokens=args.output_tokens,
                max_model_len=args.max_model_len,
            )
            payload["status"] = "MATRIX_INPUT_FROZEN_BY_HASH"
            common.write_json(manifest, payload)
        manifest_sha256 = common.sha256(manifest)
        for rate in rates:
            label = f"{load_label}__rate_{rate_label(rate)}"
            cell_root = out_root / label
            acceptance_path = cell_root / "acceptance.json"
            if args.resume and acceptance_path.is_file():
                acceptance = common.read_json(acceptance_path)
                if acceptance.get("status") != "PASS":
                    raise RuntimeError(f"cannot resume failed cell {label}")
                print(f"[{label}] already PASS; skipping", flush=True)
            else:
                if cell_root.exists():
                    raise FileExistsError(f"cell output already exists: {cell_root}")
                print(
                    f"[{label}] starting {repetitions} paired repetitions",
                    flush=True,
                )
                command = online_command(
                    args,
                    revision=revision,
                    manifest=manifest,
                    manifest_sha256=manifest_sha256,
                    cell_root=cell_root,
                    target_jobs=target_jobs,
                    rate=rate,
                    repetitions=repetitions,
                    minimum_samples=minimum_samples,
                )
                completed = subprocess.run(command, cwd=REPO)
                if completed.returncode != 0:
                    batch["status"] = "FAILED"
                    batch["failed_cell"] = label
                    batch["returncode"] = completed.returncode
                    common.write_json(out_root / "batch_status.json", batch)
                    raise RuntimeError(
                        f"matrix cell {label} failed with {completed.returncode}"
                    )
                acceptance = common.read_json(acceptance_path)
            cell = {
                "load_label": load_label,
                "target_jobs": target_jobs,
                "fixed_rate_gib_s": rate,
                "manifest": str(manifest),
                "manifest_sha256": manifest_sha256,
                "root": str(cell_root),
                "status": acceptance.get("status"),
                "recorded_runs": acceptance.get("recorded_runs"),
            }
            batch["cells"].append(cell)
            common.write_json(out_root / "batch_status.json", batch)

    errors = [
        f"{cell['load_label']} rate={cell['fixed_rate_gib_s']} failed"
        for cell in batch["cells"]
        if cell.get("status") != "PASS"
    ]
    collect_matrix(out_root, batch["cells"])
    final = {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "expected_cells": plan["expected_cells"],
        "recorded_cells": len(batch["cells"]),
        "expected_runs": plan["expected_runs"],
        "recorded_runs": sum(
            int(cell.get("recorded_runs") or 0) for cell in batch["cells"]
        ),
        "errors": errors,
    }
    common.write_json(out_root / "acceptance.json", final)
    if errors:
        raise RuntimeError("; ".join(errors))
    batch["status"] = "COMPLETE"
    batch["ended_unix_s"] = time.time()
    common.write_json(out_root / "batch_status.json", batch)
    print(f"SHADOW_RATE_LOAD_{args.phase.upper()}_COMPLETE: {out_root}")


if __name__ == "__main__":
    main()
