#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed inspector for BridgeTP Phase 8 evidence directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object: {path}")
    return value


def _inspect_bridge(run_dir: Path) -> dict[str, Any]:
    initial = _load(run_dir / "session_manifest.json")
    cutover = _load(run_dir / "cutover_manifest.json")
    staging = _load(run_dir / "staging_manifest.json")
    result = _load(run_dir / "phase8_result.json")
    takeover = _load(run_dir / "takeover_state.json")
    receiver_dirs = [
        path for path in (run_dir / "receiver_receipts").iterdir() if path.is_dir()
    ]
    if len(receiver_dirs) != 1:
        raise ValueError("Phase 8 bridge requires one receiver directory")
    deliveries = [
        _load(run_dir / "stage_delivery_receipts" / f"tp_rank_{rank}.json")
        for rank in range(4)
    ]
    receivers = [
        _load(receiver_dirs[0] / f"tp_rank_{rank}.json") for rank in range(4)
    ]
    expected_delta_start = int(initial["num_computed_tokens"])
    expected_delta_end = int(cutover["num_computed_tokens"])
    coverage_exact = all(
        record.get("delta_coverage")
        and int(record["delta_coverage"][0][0]) == expected_delta_start
        and int(record["delta_coverage"][-1][1]) == expected_delta_end
        and all(
            int(left[1]) == int(right[0])
            for left, right in zip(
                record["delta_coverage"], record["delta_coverage"][1:]
            )
        )
        for record in staging["ranks"]
    )
    delivery_exact = all(
        sender.get("status") == "READY"
        and receiver.get("status") == "OWNERSHIP_COMMITTED"
        and receiver.get("exact_readback")
        and sender.get("payload_sha256") == receiver.get("payload_sha256")
        and int(sender["payload_bytes"]) == int(receiver["payload_bytes"])
        for sender, receiver in zip(deliveries, receivers)
    )
    passed = all(
        (
            initial.get("phase") == "BridgeTP D3 Phase 8",
            cutover.get("phase") == "BridgeTP D3 Phase 8",
            staging.get("phase") == "BridgeTP D3 Phase 8",
            int(staging["new_kv_delta_tokens"]) > 0,
            coverage_exact,
            delivery_exact,
            takeover.get("state") == "COMMITTED",
            takeover.get("source_abort_dispatched") is True,
            result.get("old_kv_new_kv_overlap_proven") is True,
            result.get("exact_end_to_end_token_continuity") is True,
            result.get("status") == "PASS",
        )
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "phase": "BridgeTP D3 Phase 8",
        "mode": "dualwrite_commit",
        "migration_id": staging["migration_id"],
        "old_kv_num_computed_tokens": staging["old_kv_num_computed_tokens"],
        "new_kv_delta_tokens": staging["new_kv_delta_tokens"],
        "cutover_num_computed_tokens": staging["num_computed_tokens"],
        "all_rank_delta_coverage_exact": coverage_exact,
        "all_rank_delivery_exact": delivery_exact,
        "old_kv_new_kv_overlap_proven": result[
            "old_kv_new_kv_overlap_proven"
        ],
        "takeover_state": takeover["state"],
        "end_to_end_continuity": result[
            "exact_end_to_end_token_continuity"
        ],
        "early_finish_cleanup_proven": False,
    }


def _inspect_cleanup(run_dir: Path) -> dict[str, Any]:
    result = _load(run_dir / "phase8_cleanup_result.json")
    source = _load(run_dir / "source_cleanup_receipt.json")
    stager = _load(run_dir / "stager_cleanup_receipt.json")
    passed = all(
        (
            result.get("status") == "PASS",
            result.get("takeover_state") == "CANCELLED",
            result.get("source_abort_dispatched") is True,
            result.get("source_finish_reason") == "abort",
            source.get("status") == "CLEANED",
            stager.get("status") == "CLEANED",
            int(source.get("delta_tokens_drained", 0)) > 0,
            int(stager.get("released_rank_buffers", 0)) == 4,
            not (run_dir / "staging_manifest.json").exists(),
            not (run_dir / "takeover_response.json").exists(),
        )
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "phase": "BridgeTP D3 Phase 8",
        "mode": "pre_cutover_controller_cancellation",
        "takeover_state": result.get("takeover_state"),
        "source_abort_dispatched": result.get("source_abort_dispatched"),
        "delta_tokens_drained": source.get("delta_tokens_drained"),
        "released_rank_buffers": stager.get("released_rank_buffers"),
        "target_request_created": False,
        "takeover_committed": False,
        "pre_cutover_cancellation_cleanup_proven": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if (args.run_dir / "phase8_result.json").exists():
        result = _inspect_bridge(args.run_dir)
    elif (args.run_dir / "phase8_cleanup_result.json").exists():
        result = _inspect_cleanup(args.run_dir)
    else:
        raise FileNotFoundError("No Phase 8 result file found")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
