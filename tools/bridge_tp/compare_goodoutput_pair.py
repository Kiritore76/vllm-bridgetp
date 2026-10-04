#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check one same-input STAY/MIGRATE pilot and report TPOT-first GoodOutput."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bridge_tp.audit_goodoutput import audit_payload  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_arm(root: Path, paired_stay: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(root / "contract.json")
    acceptance = read_json(root / "acceptance.json")
    if contract.get("paired_stay") is not paired_stay:
        raise ValueError(f"wrong paired_stay arm: {root}")
    if acceptance.get("status") != "PASS" or len(acceptance.get("runs", [])) != 1:
        raise ValueError(f"online acceptance is not a single PASS: {root}")
    accepted = acceptance["runs"][0]["acceptance"]
    run_name = Path(acceptance["runs"][0]["root"]).name
    if run_name != "r01_shadow_only":
        raise ValueError(f"unexpected online run directory: {run_name}")
    run_root = root / run_name
    if accepted.get("status") != "PASS":
        raise ValueError(f"arm acceptance failed: {root}")
    if paired_stay and accepted.get("final_state") != "COMPLETED_ON_TP1":
        raise ValueError("STAY arm did not finish on TP1")
    if paired_stay and accepted.get("natural_start_decisions", 0) <= 0:
        raise ValueError("STAY arm never observed an M1 START opportunity")
    if not paired_stay and accepted.get("target_origin_tokens", 0) <= 0:
        raise ValueError("MIGRATE arm produced no TP4-owned output")
    values = {"online/contract.json": contract}
    for name in (
        "background/background_summary.json",
        "controller/response_proxy_stats.json",
        "controller/source_response.json",
    ):
        values[name] = read_json(run_root / name)
    target = run_root / "controller/target_response.json"
    if target.is_file():
        values["controller/target_response.json"] = read_json(target)
    report = audit_payload(values)
    if not report["computable"]:
        raise ValueError(f"GoodOutput audit failed: {report['errors']}")
    return contract, report


def compare(stay_root: Path, migrate_root: Path) -> dict[str, Any]:
    stay_contract, stay = load_arm(stay_root, True)
    migrate_contract, migrate = load_arm(migrate_root, False)
    for key in (
        "revision", "manifest_sha256", "survival_table_sha256",
        "guard_file_sha256", "predictor_checkpoint_sha256",
        "manager_m5_predictor_shadow", "anchor_max_tokens",
        "anchor_prompt_tokens", "manager_m1_auto_start", "manager_m2_rate",
        "manager_m3_commit", "manager_m4_cancel", "m2_profiles_gib_s",
        "ready_sync_mode", "ready_notification_mode", "source_pressure",
        "gpu_direct_history", "persistent_channel",
    ):
        if stay_contract.get(key) != migrate_contract.get(key):
            raise ValueError(f"paired contract differs in {key}")
    for field in ("v5_thresholds",):
        if stay[field] != migrate[field]:
            raise ValueError(f"paired audit differs in {field}")
    stay_pool = stay["metrics"]["by_pool"]["system"]
    migrate_pool = migrate["metrics"]["by_pool"]["system"]
    if stay_pool["requests"] != migrate_pool["requests"]:
        raise ValueError("paired runs have different request counts")
    if stay_pool["output_tokens"] != migrate_pool["output_tokens"]:
        raise ValueError("paired runs have different output token counts")
    return {
        "format_version": 1,
        "status": "PILOT_COMPARABLE_NOT_STATISTICAL",
        "revision": stay_contract["revision"],
        "manifest_sha256": stay_contract["manifest_sha256"],
        "stay_root": str(stay_root.resolve()),
        "migrate_root": str(migrate_root.resolve()),
        "stay": stay_pool,
        "migrate": migrate_pool,
        "delta_goodoutput_v4_tokens_s": (
            migrate_pool["goodoutput_v4_tokens_s"]
            - stay_pool["goodoutput_v4_tokens_s"]
        ),
        "delta_goodoutput_v5_tokens_s": (
            migrate_pool["goodoutput_v5_tokens_s"]
            - stay_pool["goodoutput_v5_tokens_s"]
        ),
        "note": (
            "one pilot pair has no uncertainty estimate; repeat interleaved "
            "pairs before changing an online migration policy"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stay-root", type=Path, required=True)
    parser.add_argument("--migrate-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.stay_root, args.migrate_root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    if args.out_json.exists():
        raise FileExistsError(args.out_json)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "delta_goodoutput_v5_tokens_s": result["delta_goodoutput_v5_tokens_s"],
        "delta_goodoutput_v4_tokens_s": result["delta_goodoutput_v4_tokens_s"],
        "out_json": str(args.out_json.resolve()),
    }))


if __name__ == "__main__":
    main()
