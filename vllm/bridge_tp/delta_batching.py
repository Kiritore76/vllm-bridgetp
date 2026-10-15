# SPDX-License-Identifier: Apache-2.0
"""CPU-only queue coalescing with a bounded wait near the final KV boundary."""

from __future__ import annotations

import queue
import time
from typing import Any


def coalesce_delta(
    first: dict[str, Any],
    work_queue: queue.Queue,
    *,
    max_remaining_tokens: int = 4,
    max_batch_tokens: int = 16,
    max_wait_s: float = 0.2,
    clock=time.monotonic,
) -> tuple[dict[str, Any], int]:
    """Merge contiguous work, briefly waiting only for a small final tail.

    The caller owns the first queue task until the physical send/ACK completes.
    Additional dequeued tasks are acknowledged here, including on exceptions;
    the outstanding first task still prevents join() from returning early.
    Only already-computed slots are merged and the latest CUDA event is kept.
    Nothing modifies a batch that has started transmission.
    """
    merged = dict(first)
    consumed = 1
    wait_started = None
    outcome = "NOT_NEEDED"
    while True:
        try:
            candidate = work_queue.get_nowait()
        except queue.Empty:
            end = merged.get("final_end_token")
            remaining = end - merged["end_token"] if end is not None else None
            size = merged["end_token"] - merged["start_token"]
            if remaining == 0:
                outcome = "FINAL_BOUNDARY"
                break
            if not (
                remaining is not None
                and 0 < remaining <= max_remaining_tokens
                and 0
                < size
                <= (
                    max_batch_tokens
                    if wait_started is None
                    else max_batch_tokens + max_remaining_tokens
                )
                and max_wait_s > 0
            ):
                outcome = "OUTSIDE_TAIL_LIMITS"
                break
            if wait_started is None:
                wait_started = clock()
            timeout = max_wait_s - (clock() - wait_started)
            if timeout <= 0:
                outcome = "TIMEOUT"
                break
            try:
                candidate = work_queue.get(timeout=timeout)
            except queue.Empty:
                outcome = "TIMEOUT"
                break
        try:
            if candidate is None:
                # Preserve the stop marker for the physical sender loop.
                work_queue.put(None)
                outcome = "STOP_REQUESTED"
                break
            if int(candidate["start_token"]) != int(merged["end_token"]):
                raise RuntimeError("GPU-direct delta submissions are not contiguous")
            merged = {
                **candidate,
                "start_token": first["start_token"],
            }
            consumed += 1
        finally:
            work_queue.task_done()
    if wait_started is not None:
        merged["tail_coalescing"] = dict(
            waited_ms=(clock() - wait_started) * 1000,
            outcome=outcome,
            final_end_token=merged.get("final_end_token"),
            max_remaining_tokens=max_remaining_tokens,
            max_batch_tokens=max_batch_tokens,
            max_wait_ms=max_wait_s * 1000,
        )
    return merged, consumed
