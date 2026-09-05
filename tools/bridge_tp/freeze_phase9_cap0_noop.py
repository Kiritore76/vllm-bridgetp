#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze the exact workload proven by a passing CAP-0 No-op bring-up."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402
from tools.bridge_tp.run_phase9_cap0_noop import (  # noqa: E402
    validate_noop_pressure,
)
from tools.bridge_tp.run_phase9_capacity_background import (  # noqa: E402
    load_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bringup-root", type=Path, required=True)
    parser.add_argument("--working-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-bringup-revision", required=True)
    parser.add_argument("--expected-working-sha256", required=True)
    parser.add_argument("--expected-survival-sha256", required=True)
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--expected-guard", type=int, required=True)
    parser.add_argument("--tp1-blocks", type=int, required=True)
    parser.add_argument("--tp4-blocks", type=int, required=True)
    parser.add_argument("--anchor-max-tokens", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    return parser.parse_args()


def require_passing_acceptance(acceptance: dict[str, Any]) -> None:
    errors: list[str] = []
    if acceptance.get("status") != "PASS" or acceptance.get("errors"):
        errors.append("bring-up acceptance is not a clean PASS")
    if int(acceptance.get("active_capacity_decisions", 0)) <= 0:
        errors.append("bring-up has no active capacity decisions")
    if int(acceptance.get("blocked_stay_decisions", 0)) <= 0:
        errors.append("bring-up has no target-guarded STAY decisions")
    if acceptance.get("blocked_stay_decisions") != acceptance.get(
        "active_capacity_decisions"
    ):
        errors.append("not every active bring-up decision was target-guarded")
    if int(acceptance.get("start_shadow_decisions", -1)) != 0:
        errors.append("bring-up attempted START_SHADOW")
    if int(acceptance.get("migration_transitions", -1)) != 0:
        errors.append("bring-up contains a migration transition")
    if acceptance.get("final_state") != "COMPLETED_ON_TP1":
        errors.append("bring-up did not complete on TP1")
    if int(acceptance.get("anchor_emitted_tokens", 0)) != 8000:
        errors.append("bring-up anchor did not emit 8000 source tokens")
    if acceptance.get("forbidden_artifacts"):
        errors.append("bring-up contains forbidden migration artifacts")
    if errors:
        raise RuntimeError("; ".join(errors))


def build_frozen_manifest(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = args.bringup_root.resolve()
    paths = {
        "status": root / "status.json",
        "acceptance": root / "provenance" / "noop_acceptance.json",
        "inputs": root / "provenance" / "inputs.json",
        "expanded_manifest": root / "background" / "background_manifest.json",
    }
    for path in (args.working_manifest, *paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    status = common.read_json(paths["status"])
    acceptance = common.read_json(paths["acceptance"])
    inputs = common.read_json(paths["inputs"])
    if status.get("status") != "BRINGUP_COMPLETE":
        raise RuntimeError("bring-up status is not BRINGUP_COMPLETE")
    require_passing_acceptance(acceptance)

    working_sha = common.sha256(args.working_manifest)
    expanded_sha = common.sha256(paths["expanded_manifest"])
    if working_sha != args.expected_working_sha256:
        raise RuntimeError("working manifest SHA-256 differs from expected")
    if expanded_sha != working_sha:
        raise RuntimeError("executed manifest differs from the working manifest")
    if inputs.get("manifest_sha256") != working_sha:
        raise RuntimeError("bring-up inputs record a different manifest")
    if inputs.get("revision") != args.expected_bringup_revision:
        raise RuntimeError("bring-up revision differs from expected")
    if status.get("revision") != args.expected_bringup_revision:
        raise RuntimeError("bring-up status revision differs from expected")
    if inputs.get("survival_table_sha256") != args.expected_survival_sha256:
        raise RuntimeError("bring-up survival input differs from expected")
    if inputs.get("guard_file_sha256") != args.expected_guard_sha256:
        raise RuntimeError("bring-up guard input differs from expected")
    if int(inputs.get("guard_free_kv_tokens", -1)) != args.expected_guard:
        raise RuntimeError("bring-up guard value differs from expected")

    working = load_manifest(args.working_manifest)
    expanded = load_manifest(paths["expanded_manifest"])
    if working != expanded:
        raise RuntimeError("executed and working manifest JSON differ")
    pressure = validate_noop_pressure(
        working,
        anchor_max_tokens=args.anchor_max_tokens,
        tp1_total_tokens=args.tp1_blocks * 16,
        tp4_total_tokens=args.tp4_blocks * 16,
        max_model_len=args.max_model_len,
    )

    frozen = copy.deepcopy(working)
    frozen["scenario"] = "CAP-0 No-op formal"
    frozen["status"] = "FROZEN"
    frozen["frozen_from"] = {
        "bringup_run_id": status.get("run_id"),
        "bringup_revision": args.expected_bringup_revision,
        "bringup_acceptance_sha256": common.sha256(paths["acceptance"]),
        "origin_manifest_sha256": working_sha,
    }
    evidence = {
        "format_version": 1,
        "status": "FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_code_revision": common.git("rev-parse", "HEAD"),
        "bringup_root": str(root),
        "bringup_status_sha256": common.sha256(paths["status"]),
        "bringup_acceptance_sha256": common.sha256(paths["acceptance"]),
        "bringup_inputs_sha256": common.sha256(paths["inputs"]),
        "origin_manifest_sha256": working_sha,
        "pressure_preflight": pressure,
    }
    return frozen, evidence


def write_new(path: Path, value: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if os.name == "nt":
        raise RuntimeError("CAP-0 No-op freezing requires Linux")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--"]
    ).returncode != 0:
        raise RuntimeError("tracked working-tree changes are present")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--cached", "--quiet", "HEAD", "--"]
    ).returncode != 0:
        raise RuntimeError("staged changes are present")

    provenance = args.out.with_suffix(".provenance.json")
    checksum = args.out.with_suffix(".sha256")
    for path in (args.out, provenance, checksum):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path.resolve()}")

    frozen, evidence = build_frozen_manifest(args)
    serialized = json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    write_new(args.out, serialized)
    frozen_sha = common.sha256(args.out)
    evidence["frozen_manifest"] = str(args.out.resolve())
    evidence["frozen_manifest_sha256"] = frozen_sha
    write_new(provenance, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    write_new(checksum, f"{frozen_sha}  {args.out.name}\n")
    print(f"FROZEN_NOOP_MANIFEST={args.out.resolve()}")
    print(f"FROZEN_NOOP_SHA256={frozen_sha}")
    print(f"FROZEN_NOOP_PROVENANCE={provenance.resolve()}")


if __name__ == "__main__":
    main()
