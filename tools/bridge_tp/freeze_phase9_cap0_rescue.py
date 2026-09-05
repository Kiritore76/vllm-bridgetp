#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze the exact workload proven by a passing CAP-0 Rescue bring-up."""

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
from tools.bridge_tp.run_phase9_cap0_rescue import (  # noqa: E402
    validate_rescue_pressure,
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


def require_passing_acceptance(
    acceptance: dict[str, Any],
    expected_anchor_tokens: int = 8000,
) -> None:
    errors: list[str] = []
    if acceptance.get("status") != "PASS" or acceptance.get("errors"):
        errors.append("bring-up acceptance is not a clean PASS")
    blocked = int(acceptance.get("blocked_stay_decisions_before_rescue", 0))
    starts = int(acceptance.get("start_shadow_decisions", -1))
    decisions = int(acceptance.get("capacity_decisions", 0))
    if blocked <= 0:
        errors.append("bring-up has no target-guarded STAY decisions")
    if starts != 1:
        errors.append("bring-up does not contain exactly one START_SHADOW")
    if decisions != blocked + starts:
        errors.append("bring-up contains unexplained capacity decisions")
    if acceptance.get("trigger_path") != "CAPACITY_PILOT":
        errors.append("bring-up did not use the CAPACITY_PILOT trigger")
    if acceptance.get("transition_states") != [
        "SHADOW",
        "HANDOFF",
        "TAKEOVER",
    ]:
        errors.append("bring-up migration path is incomplete or illegal")
    if acceptance.get("final_state") != "TAKEOVER":
        errors.append("bring-up did not finish in TAKEOVER")
    if acceptance.get("takeover_state") != "COMMITTED":
        errors.append("bring-up takeover was not committed")
    if acceptance.get("source_abort_dispatched") is not True:
        errors.append("bring-up did not dispatch source abort")
    if acceptance.get("receiver_ranks") != [0, 1, 2, 3]:
        errors.append("bring-up receiver ranks are incomplete")
    if acceptance.get("exact_readback") != [True, True, True, True]:
        errors.append("bring-up exact readback is incomplete")
    source_tokens = int(acceptance.get("source_origin_tokens", 0))
    target_tokens = int(acceptance.get("target_origin_tokens", 0))
    emitted = int(acceptance.get("anchor_emitted_tokens", 0))
    if source_tokens <= 0 or target_tokens <= 0:
        errors.append("bring-up output does not contain both owners")
    if emitted != expected_anchor_tokens:
        errors.append("bring-up emitted an unexpected anchor length")
    if source_tokens + target_tokens != emitted:
        errors.append("bring-up owner token counts do not sum to output length")
    if errors:
        raise RuntimeError("; ".join(errors))


def build_frozen_manifest(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = args.bringup_root.resolve()
    paths = {
        "status": root / "status.json",
        "acceptance": root / "provenance" / "rescue_acceptance.json",
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
    require_passing_acceptance(acceptance, args.anchor_max_tokens)

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
    pressure = validate_rescue_pressure(
        working,
        anchor_max_tokens=args.anchor_max_tokens,
        tp1_total_tokens=args.tp1_blocks * 16,
        tp4_total_tokens=args.tp4_blocks * 16,
        max_model_len=args.max_model_len,
    )

    frozen = copy.deepcopy(working)
    frozen["scenario"] = "CAP-0 Rescue formal"
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
        raise RuntimeError("CAP-0 Rescue freezing requires Linux")
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
    print(f"FROZEN_RESCUE_MANIFEST={args.out.resolve()}")
    print(f"FROZEN_RESCUE_SHA256={frozen_sha}")
    print(f"FROZEN_RESCUE_PROVENANCE={provenance.resolve()}")


if __name__ == "__main__":
    main()
