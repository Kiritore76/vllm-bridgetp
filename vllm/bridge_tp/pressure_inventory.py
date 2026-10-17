# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU block-ledger observations; no allocation or migration mutation."""

from __future__ import annotations

import atexit
import os
import time
from pathlib import Path


def inventory_snapshot(scheduler, pool_id: str, now: float) -> dict:
    """Copy live scheduler/request/block facts before any policy decision.

    Physical free includes evictable cached blocks, as in vLLM's allocator.
    Shared and null blocks are never counted as uniquely releasable. The first
    implementation supports a single full-attention KV group, such as Qwen2.
    """
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    running = {request.request_id for request in scheduler.running}
    rows = []
    for request_id, request in list(scheduler.requests.items()):
        groups = manager.get_blocks(request_id).blocks
        supported = len(groups) == 1
        blocks = {
            block.block_id: block
            for group in groups
            for block in group
            if not block.is_null
        }
        params = request.sampling_params
        rows.append(
            {
                "request_id": request_id,
                "owner": pool_id,
                "running": request_id in running,
                "generated": request.num_output_tokens,
                "prompt_tokens": request.num_prompt_tokens,
                "computed_tokens": request.num_computed_tokens,
                "max_output_tokens": request.max_tokens,
                "ignore_eos": bool(params.ignore_eos) if params is not None else None,
                "owned_blocks": len(blocks) if supported else None,
                "releasable_blocks": sum(
                    block.ref_cnt == 1 for block in blocks.values()
                )
                if supported
                else None,
                "blocks": [
                    {"id": block.block_id, "refs": block.ref_cnt}
                    for block in blocks.values()
                ],
                "block_group_supported": supported,
            }
        )
    return {
        "schema": "PRESSURE_INVENTORY_V1",
        "pool_id": pool_id,
        "sampled_at": now,
        "free_blocks": pool.get_num_free_blocks(),
        "total_blocks": pool.num_gpu_blocks,
        # BlockPool.get_usage() excludes its permanently reserved null block.
        "usage_denominator_blocks": pool.num_gpu_blocks - 1,
        "block_size": scheduler.block_size,
        "context_limit": scheduler.max_model_len,
        "running": len(scheduler.running),
        "waiting": len(scheduler.waiting) + len(scheduler.skipped_waiting),
        "pending_prompt_tokens": sum(
            max(0, row["prompt_tokens"] - row["computed_tokens"]) for row in rows
        ),
        "requests": rows,
        "prefill_generation": getattr(
            getattr(scheduler, "_bridgetp_pressure_inventory", None),
            "prefill_generation",
            None,
        ),
        "last_scheduled_batch": getattr(
            scheduler._bridgetp_pressure_inventory, "batch", None
        )
        if hasattr(scheduler, "_bridgetp_pressure_inventory")
        else None,
    }


class InventoryPublisher:
    def __init__(self, path: Path, pool_id: str, interval_s: float):
        from vllm.bridge_tp.async_journal import AsyncJournal

        if pool_id not in {"tp1", "tp4"} or interval_s <= 0:
            raise ValueError("invalid pressure inventory configuration")
        self.pool_id, self.interval_s = pool_id, interval_s
        self.journal = AsyncJournal(path, capacity=64, batch_size=4)
        self.last_sample = -float("inf")
        self.generation = 0
        self.batch = None
        self.last_batch_sample = -float("inf")
        self.last_membership = None
        self.prefill_generation = 0
        atexit.register(self.journal.close)

    @classmethod
    def from_env(cls):
        path = os.environ.get("BRIDGETP_PRESSURE_INVENTORY_PATH")
        if not path:
            return None
        return cls(
            Path(path),
            os.environ.get("BRIDGETP_PRESSURE_POOL", ""),
            float(os.environ.get("BRIDGETP_PRESSURE_SAMPLE_S", "0.2")),
        )

    def sample(self, scheduler) -> None:
        now = time.time()
        membership = (
            frozenset(scheduler.requests),
            frozenset(r.request_id for r in scheduler.running),
            len(scheduler.waiting),
            len(scheduler.skipped_waiting),
        )
        if (
            now - self.last_sample < self.interval_s
            and membership == self.last_membership
        ):
            return
        self.last_membership = membership
        self.last_sample = now
        self.generation += 1
        try:
            row = inventory_snapshot(scheduler, self.pool_id, now)
        except Exception as error:
            row = {
                "schema": "PRESSURE_INVENTORY_V1",
                "pool_id": self.pool_id,
                "sampled_at": now,
                "inventory_error": str(error),
            }
        row.update(
            generation=self.generation,
            dropped_before=self.journal.dropped,
            writer_error=str(self.journal.error) if self.journal.error else None,
        )
        self.journal.submit(row)

    def record_batch(self, requests, output):
        now = time.time()
        # Scalar only, before the metadata sampling gate: even a short prefill
        # between publications must invalidate the controller's quiet window.
        if any(
            count > 0
            and requests[key].num_computed_tokens < requests[key].num_prompt_tokens
            for key, count in output.num_scheduled_tokens.items()
        ):
            self.prefill_generation += 1
        if now - self.last_batch_sample < self.interval_s:
            return
        from vllm.bridge_tp.performance_trace import batch_composition

        try:
            self.batch = {
                **batch_composition(requests, output.num_scheduled_tokens),
                "sampled_at": now,
            }
        except Exception as error:
            self.batch = {"sampled_at": now, "error": str(error)}
        self.last_batch_sample = now
