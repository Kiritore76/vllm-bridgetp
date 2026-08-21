# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path
from typing import Any

from vllm.bridge_tp.kv_restore import load_restore_artifact


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate four BridgeTP Phase 5 restore receipts."
    )
    parser.add_argument("reshard_dir", type=Path)
    parser.add_argument("receipt_request_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_restore_artifact(args.reshard_dir)
    receipts = [
        _load_json(args.receipt_request_dir / f"tp_rank_{rank}.json")
        for rank in range(4)
    ]
    target_request_ids = {receipt["target_request_id"] for receipt in receipts}
    if len(target_request_ids) != 1:
        raise ValueError("TP4 receipts have different target request IDs")

    block_tables = []
    for rank, receipt in enumerate(receipts):
        if receipt["phase"] != "BridgeTP D3 Phase 5":
            raise ValueError(f"Rank {rank} receipt phase differs")
        if receipt["source_request_id"] != artifact.source_request_id:
            raise ValueError(f"Rank {rank} source request ID differs")
        if receipt["tp_rank"] != rank:
            raise ValueError(f"Rank {rank} receipt rank differs")
        if receipt["num_computed_tokens"] != artifact.num_computed_tokens:
            raise ValueError(f"Rank {rank} token boundary differs")
        if len(receipt["target_block_ids"]) != artifact.num_blocks:
            raise ValueError(f"Rank {rank} block count differs")
        if not receipt["exact_readback"]:
            raise ValueError(f"Rank {rank} did not pass exact readback")
        block_tables.append(receipt["target_block_ids"])
    if any(block_table != block_tables[0] for block_table in block_tables[1:]):
        raise ValueError("TP4 workers received different logical block tables")

    summary = {
        "status": "PASS",
        "source_request_id": artifact.source_request_id,
        "target_request_id": next(iter(target_request_ids)),
        "num_computed_tokens": artifact.num_computed_tokens,
        "pending_tokens_to_compute": 1,
        "tp_size": 4,
        "target_block_table": block_tables[0],
        "all_ranks_exact_readback": True,
        "restore_ms_by_rank": [receipt["restore_ms"] for receipt in receipts],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
