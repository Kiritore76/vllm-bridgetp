# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inspect one explicit BridgeTP Phase 7 commit or rollback run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = _load(root / "session_manifest.json")
    state = _load(root / "takeover_state.json")
    result = _load(root / "takeover_result.json")
    migration_id = str(manifest["migration_id"])
    if manifest.get("phase") != "BridgeTP D3 Phase 7":
        raise ValueError("Run does not contain a Phase 7 manifest")
    for document in (state, result):
        if document.get("migration_id") != migration_id:
            raise ValueError("Phase 7 documents have different migration IDs")
    if result.get("status") != "PASS":
        raise ValueError("Phase 7 controller result is not PASS")

    senders = [
        _load(root / "sender_receipts" / f"tp_rank_{rank}.json")
        for rank in range(4)
    ]
    receiver_root = root / "receiver_receipts"
    target_dirs = sorted(path for path in receiver_root.iterdir() if path.is_dir())
    if len(target_dirs) != 1:
        raise ValueError("Expected exactly one target receipt directory")
    receivers = [
        _load(target_dirs[0] / f"tp_rank_{rank}.json") for rank in range(4)
    ]
    for rank, (sender, receiver) in enumerate(zip(senders, receivers)):
        for receipt in (sender, receiver):
            if receipt.get("migration_id") != migration_id:
                raise ValueError(f"Rank {rank} receipt migration ID differs")
        if sender.get("status") != "READY":
            raise ValueError(f"Rank {rank} sender did not reach READY")
        if not receiver.get("exact_readback"):
            raise ValueError(f"Rank {rank} exact readback failed")
        if sender.get("payload_sha256") != receiver.get("payload_sha256"):
            raise ValueError(f"Rank {rank} sender/receiver SHA256 differs")

    mode = str(result["mode"])
    expected_state = "COMMITTED" if mode == "commit" else "ROLLED_BACK"
    expected_receiver = (
        "OWNERSHIP_COMMITTED" if mode == "commit" else "ROLLED_BACK"
    )
    if state.get("state") != expected_state:
        raise ValueError(f"Takeover state is not {expected_state}")
    if any(receipt.get("status") != expected_receiver for receipt in receivers):
        raise ValueError(f"Receiver ranks did not reach {expected_receiver}")

    if mode == "commit":
        if not state.get("source_abort_dispatched"):
            raise ValueError("Commit contains no source-abort evidence")
        if result.get("source_finish_reason") != "abort":
            raise ValueError("Source request did not finish with abort")
        if not result.get("source_prefix_matches_snapshot"):
            raise ValueError("Source response prefix differs from snapshot")
        if not result.get("exact_end_to_end_token_continuity"):
            raise ValueError("Committed output differs from clean TP1 control")
    elif mode == "rollback":
        if state.get("source_abort_dispatched"):
            raise ValueError("Rollback incorrectly aborted the source")
        if result.get("source_finish_reason") == "abort":
            raise ValueError("Rollback source request was aborted")
        if not result.get("source_prefix_equals_control"):
            raise ValueError("Rollback source output differs from control")
    else:
        raise ValueError(f"Unknown Phase 7 mode: {mode}")

    summary = {
        "status": "PASS",
        "phase": "BridgeTP D3 Phase 7",
        "mode": mode,
        "migration_id": migration_id,
        "source_request_id": manifest["source_request_id"],
        "target_request_id": receivers[0]["target_request_id"],
        "takeover_state": state["state"],
        "source_abort_dispatched": state["source_abort_dispatched"],
        "all_ranks_exact_readback": True,
        "receiver_states": [receipt["status"] for receipt in receivers],
        "end_to_end_continuity": result.get(
            "exact_end_to_end_token_continuity",
            result.get("source_prefix_equals_control"),
        ),
        "rollback_proven": result["rollback_proven"],
        "crash_consensus_proven": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
