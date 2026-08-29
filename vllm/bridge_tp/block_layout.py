# SPDX-License-Identifier: Apache-2.0
"""Pure block-layout validation helpers for BridgeTP restore."""

from __future__ import annotations

from collections.abc import Sequence


def snapshot_target_block_ids(
    block_ids: Sequence[Sequence[int]],
    *,
    request_num_tokens: int,
    block_size: int,
    snapshot_blocks: int,
    error_message: str,
) -> list[int]:
    """Validate a target allocation and select streamed-prefix blocks.

    A request carries one pending token beyond the streamed computed prefix.
    If the prefix ends on a block boundary, the pending token legitimately
    owns an additional scheduler block which must not receive snapshot KV.

    Args:
        block_ids: Scheduler allocation, grouped by KV-cache group.
        request_num_tokens: Computed and pending tokens owned by the request.
        block_size: Number of tokens represented by one KV block.
        snapshot_blocks: Prefix blocks present in the streamed snapshot.
        error_message: Fail-closed message for an inconsistent allocation.

    Returns:
        Logical prefix block IDs which should receive streamed KV.

    Raises:
        ValueError: If the allocation does not exactly cover the request or
            cannot contain the streamed snapshot.
    """
    if len(block_ids) != 1 or block_size <= 0 or request_num_tokens <= 0:
        raise ValueError(error_message)
    allocated = list(block_ids[0])
    expected_allocated = (request_num_tokens + block_size - 1) // block_size
    if len(allocated) != expected_allocated:
        raise ValueError(error_message)
    if snapshot_blocks not in (expected_allocated, expected_allocated - 1):
        raise ValueError(error_message)
    return allocated[:snapshot_blocks]
