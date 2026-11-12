# SPDX-License-Identifier: Apache-2.0
"""Write streamed migration history straight into the target's KV blocks.

GPU 11-12 (guard40 standby): the receiver staged a whole history outside
the KV pool (~73.7 KB/token/rank, never released), which, next to a sleeping
engine, left too little memory and the woken TP2 deadlocked. Here the history
is received in the sender's chunks into a small ring and every chunk is
copied into the request's already allocated KV blocks.

Wire layout per target rank (``bounded_history.HistoryPacker``): layers in
manifest order; each layer is its rank shard ``[blocks, *block_shape]``
(``block_axis == 0``), blocks in the request's logical order, flattened in C
order. A chunk is any contiguous element range of that stream; a block is
contiguous, so a chunk maps to whole-block runs plus at most a partial block
at either end.

Only the TP worker thread may mutate KV (``378ff1062``): it calls
``write_chunk`` from its pre-forward connector hook, on its model stream.
The receive thread only fills ring slots.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class Segment:
    """One contiguous destination run of a chunk.

    ``element_count == 0``: ``block_count`` whole blocks from ``block_start``.
    Otherwise: elements ``[element_start, element_start + element_count)`` of
    the single block ``block_start``.
    """

    layer: int
    block_start: int
    block_count: int
    source_offset: int
    element_start: int = 0
    element_count: int = 0

    @property
    def whole(self) -> bool:
        return self.element_count == 0


class HistoryLayout:
    """Element offsets of a packed per-rank history (block axis 0)."""

    def __init__(self, rank_shapes: Sequence[Sequence[int]]):
        if not rank_shapes:
            raise ValueError("history has no layers")
        self.block_shapes: list[tuple[int, ...]] = []
        self.blocks: list[int] = []
        self.block_elements: list[int] = []
        self.layer_offsets: list[int] = []
        offset = 0
        for shape in rank_shapes:
            shape = tuple(int(v) for v in shape)
            if len(shape) < 2 or any(v <= 0 for v in shape):
                raise ValueError("history layer shape must be [blocks, ...]")
            self.layer_offsets.append(offset)
            self.blocks.append(shape[0])
            self.block_shapes.append(shape[1:])
            self.block_elements.append(math.prod(shape[1:]))
            offset += math.prod(shape)
        if len(set(self.blocks)) != 1:
            raise ValueError("every history layer must cover the same blocks")
        self.num_blocks = self.blocks[0]
        self.total_elements = offset

    def segments(self, offset: int, length: int) -> list[Segment]:
        """Destination runs of chunk ``[offset, offset + length)``."""
        if offset < 0 or length <= 0 or offset + length > self.total_elements:
            raise ValueError("history chunk exceeds the payload")
        result: list[Segment] = []
        position, end = offset, offset + length
        layer = 0
        while position < end:
            while (
                layer + 1 < len(self.layer_offsets)
                and self.layer_offsets[layer + 1] <= position
            ):
                layer += 1
            per_block = self.block_elements[layer]
            local = position - self.layer_offsets[layer]
            layer_end = self.layer_offsets[layer] + self.blocks[layer] * per_block
            stop = min(end, layer_end)
            block, inner = divmod(local, per_block)
            source = position - offset
            if inner:
                count = min(per_block - inner, stop - position)
                result.append(Segment(layer, block, 1, source, inner, count))
                position += count
                continue
            whole = (stop - position) // per_block
            if whole:
                result.append(Segment(layer, block, whole, source))
                position += whole * per_block
                continue
            count = stop - position  # leading part of one block
            result.append(Segment(layer, block, 1, source, 0, count))
            position += count
        return result


def write_chunk(
    destinations: Sequence[Any],
    block_index: Any,
    layout: HistoryLayout,
    chunk: Any,
    offset: int,
    mismatches: Any,
) -> int:
    """Copy one received chunk into KV and add any readback mismatch.

    ``destinations[layer]`` is that layer's paged KV ``[total_blocks, ...]``;
    ``block_index`` is the request's physical block ids (a long tensor on the
    KV device, in logical order); ``chunk`` is the 1-D received elements;
    ``mismatches`` is a 0-d integer tensor on the KV device that accumulates
    the number of differing elements (read once, after the last chunk).
    Returns the number of elements written.
    """
    if chunk.dim() != 1:
        raise ValueError("history chunk must be one-dimensional")
    if block_index.numel() != layout.num_blocks:
        raise ValueError("target block count differs from the history")
    written = 0
    for segment in layout.segments(offset, chunk.numel()):
        destination = destinations[segment.layer]
        block_shape = layout.block_shapes[segment.layer]
        if tuple(destination.shape[1:]) != block_shape:
            raise ValueError("destination KV block shape differs from history")
        if destination.dtype != chunk.dtype:
            raise ValueError("destination KV dtype differs from history")
        per_block = layout.block_elements[segment.layer]
        if segment.whole:
            count = segment.block_count * per_block
            source = chunk.narrow(0, segment.source_offset, count).view(
                segment.block_count, *block_shape
            )
            index = block_index.narrow(0, segment.block_start, segment.block_count)
            destination.index_copy_(0, index, source)
            restored = destination.index_select(0, index)
        else:
            count = segment.element_count
            source = chunk.narrow(0, segment.source_offset, count)
            block = destination[int(block_index[segment.block_start])].view(-1)
            target = block.narrow(0, segment.element_start, count)
            target.copy_(source)
            restored = target
        mismatches.add_((restored != source).sum())
        written += count
    return written


class ChunkRing:
    """A few fixed receive slots shared by the receive and worker threads.

    Slot life: FREE -> (receive thread) FILLING -> READY -> (worker) FREE.
    ``ready_event`` (recorded by the receiver after its receive was enqueued)
    and ``free_event`` (recorded by the worker after its copy was enqueued)
    order the two streams; this class only hands slots over.
    """

    def __init__(self, slots: int, chunk_elements: int, allocate: Callable[[int], Any]):
        if slots <= 0 or chunk_elements <= 0:
            raise ValueError("ring needs slots and a positive chunk size")
        self.buffers = [allocate(chunk_elements) for _ in range(slots)]
        self.chunk_elements = chunk_elements
        self._free: deque[int] = deque(range(slots))
        self._ready: deque[tuple[int, int, int, Any]] = deque()
        self._free_events: list[Any] = [None] * slots
        self._cond = threading.Condition()
        self._error: BaseException | None = None
        self._closed = False

    def acquire(self, timeout_s: float | None = None) -> tuple[int, Any, Any]:
        """Receive thread: (slot, buffer, event to wait before overwriting)."""
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._free or self._error is not None or self._closed,
                timeout=timeout_s,
            ):
                raise TimeoutError("no free history receive slot")
            self._raise_if_failed()
            slot = self._free.popleft()
            event, self._free_events[slot] = self._free_events[slot], None
            return slot, self.buffers[slot], event

    def publish(self, slot: int, offset: int, length: int, ready_event: Any) -> None:
        """Receive thread: slot holds elements ``[offset, offset + length)``."""
        if not 0 < length <= self.chunk_elements:
            raise ValueError("received chunk does not fit its slot")
        with self._cond:
            self._raise_if_failed()
            self._ready.append((slot, offset, length, ready_event))
            self._cond.notify_all()

    def take_ready(
        self, is_complete: Callable[[Any], bool] | None = None
    ) -> list[tuple[int, Any, int, int, Any]]:
        """Worker thread, never blocking: slots whose receive has finished.

        A slot is published once its receive is enqueued, not once its data
        has arrived. The worker must not make its model stream wait for the
        network (every request on the instance would stall), so it passes
        ``is_complete`` (``event.query``). Receives finish in order on one
        stream, so taking stops at the first unfinished slot.
        """
        with self._cond:
            self._raise_if_failed()
            items = []
            while self._ready:
                slot, offset, length, event = self._ready[0]
                if is_complete is not None and event is not None and not is_complete(
                    event
                ):
                    break
                self._ready.popleft()
                items.append((slot, self.buffers[slot], offset, length, event))
            return items

    def release(self, slot: int, free_event: Any) -> None:
        """Worker thread: the slot may be overwritten after ``free_event``."""
        with self._cond:
            self._free_events[slot] = free_event
            self._free.append(slot)
            self._cond.notify_all()

    def fail(self, error: BaseException) -> None:
        with self._cond:
            if self._error is None:
                self._error = error
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self.buffers = []
            self._cond.notify_all()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("direct history transfer failed") from self._error
        if self._closed:
            raise RuntimeError("direct history ring is closed")


class DiscardRing(ChunkRing):
    """A ring whose chunks nobody loads (migration cancelled before admission).

    ``publish`` waits for the receive to finish and frees the slot at once,
    so the receive side drains the sender with a few chunks of memory.
    """

    def publish(self, slot: int, offset: int, length: int, ready_event: Any) -> None:
        if not 0 < length <= self.chunk_elements:
            raise ValueError("received chunk does not fit its slot")
        if ready_event is not None:
            ready_event.synchronize()
        self.discarded = getattr(self, "discarded", 0) + length
        self.release(slot, None)
