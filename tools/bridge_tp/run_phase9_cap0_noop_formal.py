#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run three or more reportable CAP-0 No-op repetitions."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402
from tools.bridge_tp import run_phase9_cap0_noop as noop  # noqa: E402
from tools.bridge_tp.freeze_phase9_cap0_noop import (  # noqa: E402
    require_passing_acceptance,
)
from tools.bridge_tp.run_phase9_capacity_background import (  # noqa: E402
    load_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-provenance", type=Path, required=True)
    parser.add_argument("--bringup-root", type=Path, required=True)
    parser.add_argument("--survival-table", type=Path, required=True)
    parser.add_argument("--guard-file", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-survival-sha256", required=True)
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--expected-guard", type=int, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--tp1-blocks", type=int, required=True)
    parser.add_argument("--tp4-blocks", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--anchor-max-tokens", type=int, default=8000)
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
    parser.add_argument("--run-timeout-s", type=float, default=1800)
    parser.add_argument("--stager-timeout-s", type=float, default=1800)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_bringup_contract(args: argparse.Namespace) -> dict[str, Any]:
    root = args.bringup_root.resolve()
    status_path = root / "status.json"
    acceptance_path = root / "provenance" / "noop_acceptance.json"
    inputs_path = root / "provenance" / "inputs.json"
    expanded_path = root / "background" / "background_manifest.json"
    for path in (
        args.manifest_provenance,
        status_path,
        acceptance_path,
        inputs_path,
        expanded_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = load_manifest(args.manifest)
    if manifest.get("scenario") != "CAP-0 No-op formal":
        raise RuntimeError("formal manifest has the wrong scenario")
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("formal manifest is not FROZEN")
    frozen_from = manifest.get("frozen_from")
    if not isinstance(frozen_from, dict):
        raise RuntimeError("formal manifest has no frozen_from evidence")

    status = common.read_json(status_path)
    acceptance = common.read_json(acceptance_path)
    inputs = common.read_json(inputs_path)
    provenance = common.read_json(args.manifest_provenance)
    require_passing_acceptance(acceptance)
    if status.get("status") != "BRINGUP_COMPLETE":
        raise RuntimeError("bring-up status is not BRINGUP_COMPLETE")
    if frozen_from.get("bringup_run_id") != status.get("run_id"):
        raise RuntimeError("frozen manifest names a different bring-up")
    acceptance_sha = common.sha256(acceptance_path)
    if frozen_from.get("bringup_acceptance_sha256") != acceptance_sha:
        raise RuntimeError("frozen manifest acceptance hash differs")
    origin_sha = common.sha256(expanded_path)
    if frozen_from.get("origin_manifest_sha256") != origin_sha:
        raise RuntimeError("frozen manifest origin hash differs")
    if inputs.get("manifest_sha256") != origin_sha:
        raise RuntimeError("bring-up inputs manifest hash differs")
    if inputs.get("survival_table_sha256") != args.expected_survival_sha256:
        raise RuntimeError("bring-up and formal survival inputs differ")
    if inputs.get("guard_file_sha256") != args.expected_guard_sha256:
        raise RuntimeError("bring-up and formal guard files differ")
    if int(inputs.get("guard_free_kv_tokens", -1)) != args.expected_guard:
        raise RuntimeError("bring-up and formal guard values differ")
    if manifest.get("jobs") != load_manifest(expanded_path).get("jobs"):
        raise RuntimeError("frozen jobs differ from executed bring-up jobs")
    if provenance.get("frozen_manifest_sha256") != common.sha256(args.manifest):
        raise RuntimeError("manifest provenance records a different frozen hash")
    if provenance.get("origin_manifest_sha256") != origin_sha:
        raise RuntimeError("manifest provenance records a different origin")
    return {
        "bringup_run_id": status.get("run_id"),
        "bringup_revision": status.get("revision"),
        "bringup_status_sha256": common.sha256(status_path),
        "bringup_acceptance_sha256": acceptance_sha,
        "bringup_inputs_sha256": common.sha256(inputs_path),
        "origin_manifest_sha256": origin_sha,
        "manifest_provenance_sha256": common.sha256(args.manifest_provenance),
    }


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[str, int, dict[str, Any], dict[str, Any]]:
    if args.repetitions < 3:
        raise ValueError("formal No-op requires at least three repetitions")
    revision, guard, pressure = noop.validate_inputs(args)
    bringup = validate_bringup_contract(args)
    return revision, guard, pressure, bringup


def formal_contract(
    args: argparse.Namespace,
    revision: str,
    guard: int,
    pressure: dict[str, Any],
    bringup: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "scenario": "CAP-0 No-op formal",
        "revision": revision,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": common.sha256(args.manifest),
        "survival_table": str(args.survival_table.resolve()),
        "survival_table_sha256": common.sha256(args.survival_table),
        "guard_file": str(args.guard_file.resolve()),
        "guard_file_sha256": common.sha256(args.guard_file),
        "guard_free_kv_tokens": guard,
        "model_path": str(args.model_path.resolve()),
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tp1_blocks": args.tp1_blocks,
        "tp4_blocks": args.tp4_blocks,
        "anchor_max_tokens": args.anchor_max_tokens,
        "pressure_preflight": pressure,
        "bringup_evidence": bringup,
    }


def main() -> None:
    args = parse_args()
    revision, guard, pressure, bringup = validate_inputs(args)
    contract = formal_contract(args, revision, guard, pressure, bringup)
    if args.validate_only:
        print(json.dumps({"status": "VALID", "contract": contract}, indent=2))
        return

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=False)
    status_path = out_root / "batch_status.json"
    batch: dict[str, Any] = {
        "format_version": 1,
        "status": "RUNNING",
        "scenario": "noop",
        "revision": revision,
        "repetitions_expected": args.repetitions,
        "started_unix_s": time.time(),
        "contract": contract,
        "runs": [],
    }
    common.write_json(status_path, batch)
    try:
        for repetition in range(1, args.repetitions + 1):
            label = f"r{repetition:02d}"
            rep_args = copy.copy(args)
            rep_args.out_root = out_root / label
            run_id = f"{out_root.name}-{label}"
            print(
                f"[{repetition}/{args.repetitions}] starting formal No-op",
                flush=True,
            )
            result = noop.run(
                rep_args,
                revision,
                guard,
                pressure,
                phase="formal",
                repetition=repetition,
                run_id=run_id,
            )
            summary = {
                "repetition": repetition,
                "run_id": run_id,
                "status": result["status"],
                "root": str(rep_args.out_root.resolve()),
                "acceptance": result["acceptance"],
            }
            batch["runs"].append(summary)
            common.write_json(status_path, batch)

        errors = [
            f"r{item['repetition']:02d} did not pass"
            for item in batch["runs"]
            if item.get("status") != "PASS"
            or item.get("acceptance", {}).get("status") != "PASS"
        ]
        formal_acceptance = {
            "format_version": 1,
            "status": "PASS" if not errors else "FAIL",
            "repetitions_expected": args.repetitions,
            "repetitions_passed": len(batch["runs"]) - len(errors),
            "manifest_sha256": contract["manifest_sha256"],
            "guard_free_kv_tokens": guard,
            "runs": batch["runs"],
            "errors": errors,
        }
        common.write_json(out_root / "formal_acceptance.json", formal_acceptance)
        if errors:
            raise RuntimeError("; ".join(errors))
        batch["status"] = "FORMAL_COMPLETE"
        batch["ended_unix_s"] = time.time()
        batch["formal_acceptance"] = formal_acceptance
        common.write_json(status_path, batch)
        print(f"NOOP_FORMAL_COMPLETE: {out_root}", flush=True)
    except BaseException as error:
        batch["status"] = "FAILED"
        batch["ended_unix_s"] = time.time()
        batch["error"] = f"{type(error).__name__}: {error}"
        common.write_json(status_path, batch)
        raise


if __name__ == "__main__":
    main()
