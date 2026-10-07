# SPDX-License-Identifier: Apache-2.0
"""Compact evidence for a full-rank, exact historical KV readback."""

from __future__ import annotations

from typing import Any


def history_block_coverage(
    *,
    block_size: int,
    end_token: int,
    target_block_ids: list[int],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Describe verified blocks without writing one file per logical block."""
    count = len(target_block_ids)
    if (
        type(block_size) is not int
        or block_size <= 0
        or type(end_token) is not int
        or end_token <= 0
        or count != (end_token + block_size - 1) // block_size
        or any(type(block) is not int or block < 0 for block in target_block_ids)
        or len(set(target_block_ids)) != count
        or validation.get("exact_readback") is not True
        or validation.get("num_target_blocks") != count
        or type(validation.get("num_layers")) is not int
        or validation["num_layers"] <= 0
    ):
        raise ValueError("historical KV coverage does not match exact readback")
    return {
        "format_version": 1,
        "verification_scope": "FULL_RANK_EXACT_READBACK",
        "logical_block_start": 0,
        "logical_block_end_exclusive": count,
        "block_size": block_size,
        "end_token": end_token,
        "target_block_ids": list(target_block_ids),
        "num_layers": validation["num_layers"],
        "exact_readback": True,
    }


def valid_history_block_coverage(
    receipt: dict[str, Any],
    *,
    migration_id: str,
    rank: int,
    end_token: int,
    block_size: int,
    expected_blocks: int,
    target_block_ids: list[int],
    target_request_id: str,
) -> bool:
    """Require exact coverage and the final receiver's allocation identity."""
    coverage = receipt.get("history_block_coverage")
    if not isinstance(coverage, dict) or not isinstance(target_block_ids, list):
        return False
    if (
        receipt.get("status") != "INITIAL_HISTORY_GPU_RESIDENT"
        or receipt.get("migration_id") != migration_id
        or type(receipt.get("tp_rank")) is not int
        or receipt["tp_rank"] != rank
        or receipt.get("target_request_id") != target_request_id
        or receipt.get("exact_readback") is not True
        or receipt.get("end_token") != end_token
        or len(target_block_ids) < expected_blocks
    ):
        return False
    try:
        expected = history_block_coverage(
            block_size=block_size,
            end_token=end_token,
            target_block_ids=target_block_ids[:expected_blocks],
            validation={
                "exact_readback": True,
                "num_target_blocks": expected_blocks,
                "num_layers": coverage.get("num_layers"),
            },
        )
    except (TypeError, ValueError):
        return False
    # Reject bools masquerading as integer fields (True == 1 in Python).
    for key in (
        "format_version",
        "logical_block_start",
        "logical_block_end_exclusive",
        "block_size",
        "end_token",
    ):
        if type(coverage.get(key)) is not int:
            return False
    blocks = coverage.get("target_block_ids")
    if not isinstance(blocks, list) or any(type(v) is not int for v in blocks):
        return False
    return coverage == expected
