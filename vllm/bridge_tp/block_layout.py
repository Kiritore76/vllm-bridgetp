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
    If the prefix ends on a block boundary, the scheduler may allocate the
    pending token's block now or at the first target-side step. That block
    must not receive snapshot KV.

    Args:
        block_ids: Scheduler allocation, grouped by KV-cache group.
        request_num_tokens: Computed and pending tokens owned by the request.
        block_size: Number of tokens represented by one KV block.
        snapshot_blocks: Prefix blocks present in the streamed snapshot.
        error_message: Fail-closed message for an inconsistent allocation.

    Returns:
        Logical prefix block IDs which should receive streamed KV.

    Raises:
        ValueError: If the allocation cannot contain the streamed snapshot or
            has blocks beyond the one legitimate pending-token tail block.
    """
    if len(block_ids) != 1 or block_size <= 0 or request_num_tokens <= 0:
        raise ValueError(error_message)
    allocated = list(block_ids[0])
    expected_allocated = (request_num_tokens + block_size - 1) // block_size
    # The scheduler may allocate only the externally computed prefix when the
    # one pending token starts a new block. It allocates that tail block on
    # the first target-side step, after update_state_after_alloc().
    pending_tail_unallocated = (
        request_num_tokens % block_size == 1
        and snapshot_blocks == expected_allocated - 1
        and len(allocated) == snapshot_blocks
    )
    if len(allocated) != expected_allocated and not pending_tail_unallocated:
        raise ValueError(error_message)
    if snapshot_blocks not in (expected_allocated, expected_allocated - 1):
        raise ValueError(error_message)
    return allocated[:snapshot_blocks]
