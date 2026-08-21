# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inspect one explicit BridgeTP Phase 6 run directory."""

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
    continuity = _load(root / "continuity_result.json")
    migration_id = str(manifest["migration_id"])
    if continuity.get("migration_id") != migration_id:
        raise ValueError("Manifest and continuity result migration IDs differ")

    sender_receipts = [
        _load(root / "sender_receipts" / f"tp_rank_{rank}.json")
        for rank in range(4)
    ]
    receiver_root = root / "receiver_receipts"
    target_dirs = sorted(path for path in receiver_root.iterdir() if path.is_dir())
    if len(target_dirs) != 1:
        raise ValueError(
            f"Expected exactly one target request receipt directory, got {target_dirs}"
        )
    receiver_receipts = [
        _load(target_dirs[0] / f"tp_rank_{rank}.json") for rank in range(4)
    ]
    for rank in range(4):
        sender = sender_receipts[rank]
        receiver = receiver_receipts[rank]
        for receipt in (sender, receiver):
            if receipt.get("migration_id") != migration_id:
                raise ValueError(f"Rank {rank} receipt migration ID differs")
        if sender.get("status") != "READY":
            raise ValueError(f"Rank {rank} sender is not READY")
        if receiver.get("status") != "READY":
            raise ValueError(f"Rank {rank} receiver is not READY")
        if not receiver.get("exact_readback"):
            raise ValueError(f"Rank {rank} exact GPU readback failed")
        if sender.get("payload_sha256") != receiver.get("payload_sha256"):
            raise ValueError(f"Rank {rank} sender/receiver SHA256 differs")
        if int(sender["payload_bytes"]) != int(receiver["payload_bytes"]):
            raise ValueError(f"Rank {rank} sender/receiver byte count differs")

    summary = {
        "status": "PASS",
        "phase": "BridgeTP D3 Phase 6",
        "scope": "live TP1-to-TP4 stream and shadow continuation",
        "migration_id": migration_id,
        "source_request_id": manifest["source_request_id"],
        "target_request_id": receiver_receipts[0]["target_request_id"],
        "num_computed_tokens": manifest["num_computed_tokens"],
        "pending_tokens_to_compute": manifest["pending_known_tokens"],
        "tp_size": 4,
        "all_ranks_exact_readback": True,
        "raw_tensor_bytes_total": sum(
            int(receipt["raw_tensor_bytes"]) for receipt in receiver_receipts
        ),
        "wire_payload_bytes_total": sum(
            int(receipt["payload_bytes"]) for receipt in receiver_receipts
        ),
        "receiver_total_ms_by_rank": [
            receipt["total_ms"] for receipt in receiver_receipts
        ],
        "sender_total_seconds_by_rank": [
            receipt["total_seconds"] for receipt in sender_receipts
        ],
        "exact_token_continuity": continuity["exact_token_continuity"],
        "continuation_tokens": continuity["continuation_tokens"],
        "ownership_takeover_proven": False,
    }
    if not summary["exact_token_continuity"]:
        raise ValueError("TP1 and TP4 continuation tokens differ")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
