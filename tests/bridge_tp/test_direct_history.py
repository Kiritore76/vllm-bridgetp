# SPDX-License-Identifier: Apache-2.0
"""Direct-to-KV history chunks: exact equivalence with whole-payload inject."""

import threading
import unittest

import torch

from vllm.bridge_tp.bounded_history import HistoryPacker
from vllm.bridge_tp.direct_history import ChunkRing, HistoryLayout, write_chunk
from vllm.bridge_tp.kv_restore import inject_rank_shard

BLOCK = (2, 4, 2, 8)  # (k/v, block size, heads per rank, head dim), small


def history(blocks=9, layers=3, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    shards = [
        torch.randn((blocks, *BLOCK), generator=generator).to(dtype)
        for _ in range(layers)
    ]
    packed = torch.cat([s.reshape(-1) for s in shards])
    return shards, packed


def kv(total_blocks=40, layers=3, dtype=torch.float32):
    return [torch.zeros((total_blocks, *BLOCK), dtype=dtype) for _ in range(layers)]


def stream(packed, chunk):
    for offset in range(0, packed.numel(), chunk):
        yield offset, packed[offset : offset + chunk]


class LayoutTests(unittest.TestCase):
    def test_segments_cover_every_element_once(self):
        shards, packed = history(blocks=7, layers=3)
        layout = HistoryLayout([list(s.shape) for s in shards])
        self.assertEqual(layout.total_elements, packed.numel())
        per_block = layout.block_elements[0]
        for chunk in (1, 5, per_block - 1, per_block, per_block + 3, 3 * per_block,
                      7 * per_block, packed.numel()):
            covered = torch.zeros(packed.numel(), dtype=torch.int32)
            for offset, piece in stream(packed, chunk):
                for seg in layout.segments(offset, piece.numel()):
                    start = offset + seg.source_offset
                    count = (
                        seg.block_count * per_block if seg.whole else seg.element_count
                    )
                    covered[start : start + count] += 1
                    flat = (
                        layout.layer_offsets[seg.layer]
                        + seg.block_start * per_block
                        + seg.element_start
                    )
                    self.assertEqual(flat, start)  # source and destination agree
            self.assertTrue(bool((covered == 1).all()), chunk)

    def test_aligned_chunks_are_whole_block_runs_split_only_at_layers(self):
        shards, packed = history(blocks=10, layers=3)
        layout = HistoryLayout([list(s.shape) for s in shards])
        per_block = layout.block_elements[0]
        segments = layout.segments(8 * per_block, 4 * per_block)  # crosses a layer
        self.assertEqual(
            [(s.layer, s.block_start, s.block_count, s.whole) for s in segments],
            [(0, 8, 2, True), (1, 0, 2, True)],
        )

    def test_invalid_layouts_and_ranges(self):
        with self.assertRaises(ValueError):
            HistoryLayout([])
        with self.assertRaises(ValueError):
            HistoryLayout([[4, 2], [5, 2]])  # layers must cover the same blocks
        layout = HistoryLayout([[4, 2]])
        with self.assertRaises(ValueError):
            layout.segments(6, 3)


class WriteTests(unittest.TestCase):
    def check_equivalent(self, chunk, blocks=9, dtype=torch.float32):
        shards, packed = history(blocks=blocks, dtype=dtype)
        layout = HistoryLayout([list(s.shape) for s in shards])
        target = torch.randperm(40)[:blocks].tolist()
        expected, actual = kv(dtype=dtype), kv(dtype=dtype)
        inject_rank_shard(
            {str(i): d for i, d in enumerate(expected)},
            {str(i): s for i, s in enumerate(shards)},
            target,
            block_axis=0,
        )
        index = torch.tensor(target, dtype=torch.long)
        mismatches = torch.zeros((), dtype=torch.int64)
        written = sum(
            write_chunk(actual, index, layout, piece, offset, mismatches)
            for offset, piece in stream(packed, chunk)
        )
        self.assertEqual(written, packed.numel())
        self.assertEqual(int(mismatches), 0)
        for e, a in zip(expected, actual):
            self.assertTrue(torch.equal(e, a))

    def test_equals_whole_payload_inject_for_aligned_and_odd_chunks(self):
        per_block = HistoryLayout([[1, *BLOCK]]).block_elements[0]
        for chunk in (per_block, 4 * per_block, 256 * per_block, 7, per_block + 1,
                      3 * per_block - 2, 10**9):
            with self.subTest(chunk=chunk):
                self.check_equivalent(chunk)

    def test_bfloat16_and_single_block_history(self):
        self.check_equivalent(5, dtype=torch.bfloat16)
        self.check_equivalent(3, blocks=1)

    def test_production_geometry_chunks_are_whole_blocks(self):
        # Qwen3-8B TP2 rank: [blocks, 2, 16, 4, 128] bf16; 16 MiB budget over 2 ranks.
        per_block = 2 * 16 * 4 * 128
        chunk = 16 * 1024 * 1024 // (2 * 2)
        self.assertEqual(chunk % per_block, 0)
        layout = HistoryLayout([[1280, 2, 16, 4, 128]] * 36)
        segments = [
            s
            for offset in range(0, layout.total_elements, chunk)
            for s in layout.segments(offset, min(chunk, layout.total_elements - offset))
        ]
        self.assertTrue(all(s.whole for s in segments))
        self.assertLessEqual(max(s.block_count for s in segments), 256)

    def test_readback_difference_is_counted_not_hidden(self):
        shards, packed = history(blocks=3, layers=1)
        layout = HistoryLayout([list(s.shape) for s in shards])
        packed = packed.clone()
        packed[5] = float("nan")  # never equal to itself after the copy
        mismatches = torch.zeros((), dtype=torch.int64)
        write_chunk(kv(layers=1), torch.tensor([3, 1, 2]), layout, packed, 0, mismatches)
        self.assertEqual(int(mismatches), 1)

    def test_shape_dtype_and_block_count_are_checked(self):
        shards, packed = history(blocks=3, layers=1)
        layout = HistoryLayout([list(s.shape) for s in shards])
        zero = torch.zeros((), dtype=torch.int64)
        with self.assertRaises(ValueError):
            write_chunk(kv(layers=1), torch.tensor([1, 2]), layout, packed, 0, zero)
        with self.assertRaises(ValueError):
            write_chunk(
                kv(layers=1, dtype=torch.float16),
                torch.tensor([1, 2, 3]),
                layout,
                packed,
                0,
                zero,
            )
        wrong = [torch.zeros((40, 2, 4, 2, 4))]
        with self.assertRaises(ValueError):
            write_chunk(wrong, torch.tensor([1, 2, 3]), layout, packed, 0, zero)

    def test_matches_the_senders_bounded_packer_stream(self):
        # Pack exactly as the TP1 sender does, then write rank 1's chunks.
        caches = [torch.randn((12, 2, 4, 4, 8)) for _ in range(3)]  # 4 heads, TP2
        block_ids = [7, 2, 9, 0, 11]
        packer = HistoryPacker(caches, block_ids, 0, 3, 2)
        rank_shapes = [[5, 2, 4, 2, 8]] * 3
        layout = HistoryLayout(rank_shapes)
        chunk = 37  # deliberately unaligned
        target = [30, 3, 17, 25, 8]
        index = torch.tensor(target)
        actual = [torch.zeros((40, 2, 4, 2, 8)) for _ in range(3)]
        mismatches = torch.zeros((), dtype=torch.int64)
        offset = 0
        while offset < layout.total_elements:
            length = min(chunk, layout.total_elements - offset)
            buffers = [torch.empty(length), torch.empty(length)]
            packer.fill(buffers, length)
            write_chunk(actual, index, layout, buffers[1], offset, mismatches)
            offset += length
        self.assertEqual(int(mismatches), 0)
        for layer, cache in enumerate(caches):
            expected = cache[block_ids].chunk(2, dim=3)[1]
            self.assertTrue(torch.equal(actual[layer][target], expected))


class RingTests(unittest.TestCase):
    def test_slots_cycle_in_order_with_backpressure_and_events(self):
        ring = ChunkRing(2, 4, lambda n: torch.zeros(n))
        a = ring.acquire(0.1)
        b = ring.acquire(0.1)
        with self.assertRaises(TimeoutError):
            ring.acquire(0.01)  # both slots in flight
        ring.publish(a[0], 0, 4, "ready-a")
        ring.publish(b[0], 4, 3, "ready-b")
        ready = ring.take_ready()
        self.assertEqual([(s, o, n, e) for s, _, o, n, e in ready],
                         [(a[0], 0, 4, "ready-a"), (b[0], 4, 3, "ready-b")])
        self.assertEqual(ring.take_ready(), [])
        ring.release(a[0], "free-a")
        slot, _, event = ring.acquire(0.1)
        self.assertEqual((slot, event), (a[0], "free-a"))

    def test_receiver_waits_for_the_worker_to_free_a_slot(self):
        ring = ChunkRing(1, 4, lambda n: torch.zeros(n))
        slot, _, _ = ring.acquire(0.1)
        got = []
        thread = threading.Thread(target=lambda: got.append(ring.acquire(2.0)))
        thread.start()
        ring.publish(slot, 0, 4, None)
        ring.take_ready()
        ring.release(slot, "free")
        thread.join(3)
        self.assertEqual(got[0][2], "free")

    def test_failure_and_close_wake_both_sides(self):
        ring = ChunkRing(1, 4, lambda n: torch.zeros(n))
        ring.acquire(0.1)
        ring.fail(ValueError("nccl"))
        with self.assertRaises(RuntimeError):
            ring.acquire(0.1)
        with self.assertRaises(RuntimeError):
            ring.take_ready()
        other = ChunkRing(1, 4, lambda n: torch.zeros(n))
        other.close()
        with self.assertRaises(RuntimeError):
            other.acquire(0.1)
        with self.assertRaises(ValueError):
            ChunkRing(1, 4, lambda n: torch.zeros(n)).publish(0, 0, 5, None)


if __name__ == "__main__":
    unittest.main()
