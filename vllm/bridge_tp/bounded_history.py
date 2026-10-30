# SPDX-License-Identifier: Apache-2.0
"""One-layer source packing into fixed-size per-rank wire chunks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackingPlan:
    rank_elements: int
    chunk_elements: int
    chunk_buffer_bytes: int
    largest_layer_bytes: int
    block_index_bytes: int
    source_peak_bound_bytes: int


def packing_plan(caches, block_ids, block_axis, head_axis, target_tp, budget_bytes):
    if (
        not caches
        or not block_ids
        or type(budget_bytes) is not int
        or budget_bytes <= 0
        or target_tp <= 0
    ):
        raise ValueError("positive history shape and packing budget required")
    element_size = caches[0].element_size()
    counts = []
    for cache in caches:
        if not (
            -cache.ndim <= block_axis < cache.ndim
            and -cache.ndim <= head_axis < cache.ndim
        ):
            raise ValueError("history axes exceed tensor rank")
        b, h = block_axis % cache.ndim, head_axis % cache.ndim
        if (
            b == h
            or cache.dtype != caches[0].dtype
            or cache.device != caches[0].device
            or cache.shape[h] % target_tp
            or min(block_ids) < 0
            or max(block_ids) >= cache.shape[b]
        ):
            raise ValueError("history block/head geometry differs")
        counts.append(cache.numel() // cache.shape[b] * len(block_ids))
    rank_elements = sum(counts) // target_tp
    chunk = min(rank_elements, budget_bytes // (element_size * target_tp))
    if chunk <= 0:
        raise ValueError("packing budget cannot hold one element per rank")
    buffers = chunk * element_size * target_tp
    largest = max(counts) * element_size
    indices = len(block_ids) * 8
    return PackingPlan(
        rank_elements, chunk, buffers, largest, indices, buffers + 2 * largest + indices
    )


class HistoryPacker:
    """Keep at most a selected layer and its contiguous rank shards live.

    Callers synchronize the producer stream before sending a filled chunk and
    finish every send before overwriting its buffers.
    """

    def __init__(
        self, caches, block_ids, block_axis, head_axis, target_tp, memory_observer=None
    ):
        import torch

        self.caches = caches
        self.index = torch.tensor(block_ids, dtype=torch.long, device=caches[0].device)
        self.block_axis, self.head_axis, self.target_tp = (
            block_axis,
            head_axis,
            target_tp,
        )
        self.layer = 0
        self.offset = 0
        self.shards = None
        self.memory_observer = memory_observer

    def fill(self, buffers, length):
        if len(buffers) != self.target_tp or length <= 0:
            raise ValueError("invalid history chunk buffers")
        written = 0
        while written < length:
            if self.shards is None:
                if self.layer >= len(self.caches):
                    raise ValueError("history chunk exceeds payload")
                selected = self.caches[self.layer].index_select(
                    self.block_axis, self.index
                )
                if self.memory_observer is not None:
                    self.memory_observer()
                self.shards = [
                    s.contiguous().view(-1)
                    for s in selected.chunk(self.target_tp, dim=self.head_axis)
                ]
                if self.memory_observer is not None:
                    self.memory_observer()
                del selected
                self.offset = 0
            count = min(length - written, self.shards[0].numel() - self.offset)
            for rank in range(self.target_tp):
                buffers[rank].narrow(0, written, count).copy_(
                    self.shards[rank].narrow(0, self.offset, count)
                )
            written += count
            self.offset += count
            if self.offset == self.shards[0].numel():
                self.shards = None
                self.layer += 1

    def close(self):
        self.shards = None
        self.index = None
