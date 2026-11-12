# SPDX-License-Identifier: Apache-2.0
"""Receiver side of direct-to-KV history: chunks into a ring, same protocol."""

import threading
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

from vllm.bridge_tp.direct_history import ChunkRing, HistoryLayout, write_chunk
from vllm.bridge_tp.gpu_direct_history import (
    DirectReceiveResult,
    GpuDirectHistoryReceiver,
    ReceiveResult,
)
from vllm.bridge_tp.stream_protocol import (
    PayloadType,
    PersistentChannelLifecycle,
    SessionEnvelope,
)

SHAPE = [6, 2, 4, 2, 8]  # [blocks, k/v, block size, heads per rank, head dim]
LAYERS = 3


def receiver():
    value = object.__new__(GpuDirectHistoryReceiver)
    value.device = torch.device("cpu")
    value.connection = Mock()
    value.nccl = Mock()
    value.comm = Mock()
    value.stream = Mock()
    value.persistent_channel = True
    value.lifecycle = PersistentChannelLifecycle(
        topology_key="tp1-to-tp2-rank1",
        expected_ranks=frozenset({1}),
        channel_generation=7,
    )
    value.lifecycle.mark_open()
    value.allow_idle_wait = False
    value._active_request_id = None
    value._last_delta_envelope = None
    value._packed_buffer = None
    value.buffer_capacity_elements = 0
    value.buffer_high_water_bytes = 0
    value.last_session_payload_released_bytes = 0
    value.close = Mock()
    return value


def messages(chunk_elements):
    def envelope(sequence, payload_type):
        return SessionEnvelope(
            channel_generation=7,
            migration_id="m-1",
            request_id="source-1",
            sequence_number=sequence,
            payload_type=payload_type,
            token_start=0,
            token_end=0,
            rank=1,
        ).to_wire()

    history = {"op": "HISTORY", **envelope(1, PayloadType.HISTORY)}
    if chunk_elements is not None:
        history["history_chunk_elements"] = chunk_elements
    return [{"op": "START_SESSION", **envelope(0, PayloadType.START_SESSION)}, history]


def records():
    return [
        dict(layer_name=f"layer.{i}", rank_shape=SHAPE, dtype="torch.float32")
        for i in range(LAYERS)
    ]


class FakeWire:
    """The sender's packed rank stream, delivered by successive receives."""

    def __init__(self, packed):
        self.packed, self.offset, self.sizes = packed, 0, []

    def __call__(self, nccl, comm, tensor, stream):
        n = tensor.numel()
        tensor.copy_(self.packed[self.offset : self.offset + n])
        self.offset += n
        self.sizes.append(n)
        return ("received", self.offset)


class Worker(threading.Thread):
    """Plays the TP worker's pre-forward hook: write ready chunks, free slots."""

    def __init__(self, ring, layout, destinations, index, total):
        super().__init__(daemon=True)
        self.ring, self.layout = ring, layout
        self.destinations, self.index, self.total = destinations, index, total
        self.mismatches = torch.zeros((), dtype=torch.int64)
        self.written, self.error = 0, None
        self.stop = threading.Event()

    def run(self):
        try:
            while self.written < self.total and not self.stop.is_set():
                for slot, buffer, offset, length, event in self.ring.take_ready():
                    self.written += write_chunk(
                        self.destinations,
                        self.index,
                        self.layout,
                        buffer.narrow(0, 0, length),
                        offset,
                        self.mismatches,
                    )
                    self.ring.release(slot, ("free", slot, offset))
                self.stop.wait(0.001)
        except BaseException as error:  # surfaced by the test
            self.error = error


class DirectReceiveTests(unittest.TestCase):
    def run_receive(self, chunk, slots=2):
        layout = HistoryLayout([SHAPE] * LAYERS)
        packed = torch.randn(layout.total_elements)
        value = receiver()
        wire = FakeWire(packed)
        target = [11, 3, 20, 7, 0, 15]
        destinations = [torch.zeros((24, *SHAPE[1:])) for _ in range(LAYERS)]
        made = {}

        def make_ring(chunk_elements, dtype):
            made["ring"] = ChunkRing(
                slots, chunk_elements, lambda n: torch.empty(n, dtype=dtype)
            )
            made["worker"] = Worker(
                made["ring"],
                layout,
                destinations,
                torch.tensor(target),
                layout.total_elements,
            )
            made["worker"].start()
            return made["ring"]

        with (
            patch(
                "vllm.bridge_tp.gpu_direct_history.recv_json",
                side_effect=messages(chunk),
            ),
            patch("vllm.bridge_tp.gpu_direct_history.send_json") as send,
            patch("vllm.bridge_tp.gpu_direct_history._recv_tensor", wire),
            patch(
                "vllm.bridge_tp.gpu_direct_history.torch.cuda.device",
                return_value=nullcontext(),
            ),
        ):
            result = value.receive_into_ring(
                migration_id="m-1",
                request_id="source-1",
                rank=1,
                layer_records=records(),
                make_ring=make_ring,
                slot_timeout_s=5,
            )
        worker = made.get("worker")
        if worker is not None:
            worker.join(5)
            self.assertIsNone(worker.error)
        return result, value, send, wire, packed, layout, destinations, target, made

    def test_chunks_land_in_kv_with_the_same_protocol(self):
        result, value, send, wire, packed, layout, destinations, target, made = (
            self.run_receive(chunk=500, slots=2)
        )
        self.assertIsInstance(result, DirectReceiveResult)
        statuses = [call.args[1]["status"] for call in send.call_args_list]
        self.assertEqual(
            statuses, ["SESSION_READY", "READY_TO_RECV", "RECEIVED", "COMPLETE"]
        )
        ready = send.call_args_list[1].args[1]
        self.assertEqual(ready["numel"], layout.total_elements)
        # The receiver followed the sender's chunking exactly.
        self.assertEqual(sum(wire.sizes), layout.total_elements)
        self.assertTrue(all(n == 500 for n in wire.sizes[:-1]))
        self.assertEqual(result.chunks, len(wire.sizes))
        self.assertEqual(result.raw_tensor_bytes, layout.total_elements * 4)
        # Nothing history-sized: the ring holds two chunk-sized slots.
        self.assertEqual([b.numel() for b in made["ring"].buffers], [500, 500])
        self.assertEqual(int(made["worker"].mismatches), 0)
        offset = 0
        for layer in range(LAYERS):
            expected = packed[offset : offset + layout.block_elements[0] * SHAPE[0]]
            offset += expected.numel()
            got = destinations[layer][target].reshape(-1)
            self.assertTrue(torch.equal(got, expected))
        value.close.assert_not_called()

    def test_reused_slot_waits_for_the_workers_free_event(self):
        _, value, *_ = self.run_receive(chunk=300, slots=1)
        waited = [call.args[0] for call in value.stream.wait_event.call_args_list]
        self.assertTrue(waited)
        self.assertTrue(all(event[0] == "free" for event in waited))

    def test_unchunked_sender_falls_back_to_the_whole_payload(self):
        result, value, send, wire, packed, layout, *_ = self.run_receive(chunk=None)
        self.assertIsInstance(result, ReceiveResult)
        self.assertEqual(wire.sizes, [layout.total_elements])
        self.assertEqual(sorted(result.layers), [f"layer.{i}" for i in range(LAYERS)])
        statuses = [call.args[1]["status"] for call in send.call_args_list]
        self.assertEqual(
            statuses, ["SESSION_READY", "READY_TO_RECV", "RECEIVED", "COMPLETE"]
        )

    def test_receive_failure_fails_the_ring_and_closes(self):
        value = receiver()
        rings = []

        def make_ring(chunk_elements, dtype):
            rings.append(ChunkRing(2, chunk_elements, lambda n: torch.empty(n)))
            return rings[-1]

        def broken(*args):
            raise RuntimeError("nccl recv failed")

        with (
            patch(
                "vllm.bridge_tp.gpu_direct_history.recv_json",
                side_effect=messages(100),
            ),
            patch("vllm.bridge_tp.gpu_direct_history.send_json"),
            patch("vllm.bridge_tp.gpu_direct_history._recv_tensor", broken),
        ):
            with self.assertRaises(RuntimeError):
                value.receive_into_ring(
                    migration_id="m-1",
                    request_id="source-1",
                    rank=1,
                    layer_records=records(),
                    make_ring=make_ring,
                )
        value.close.assert_called_once()
        with self.assertRaises(RuntimeError):
            rings[0].take_ready()

    def test_protocol_violations_are_rejected(self):
        for bad in ("not-int", 0, 10**9):
            value = receiver()
            with (
                patch(
                    "vllm.bridge_tp.gpu_direct_history.recv_json",
                    side_effect=messages(bad),
                ),
                patch("vllm.bridge_tp.gpu_direct_history.send_json"),
                self.assertRaises(Exception),
            ):
                value.receive_into_ring(
                    migration_id="m-1",
                    request_id="source-1",
                    rank=1,
                    layer_records=records(),
                    make_ring=lambda n, d: ChunkRing(1, n, lambda m: torch.empty(m)),
                )
            value.close.assert_called_once()
        value = receiver()
        value.persistent_channel = False
        with (
            patch("vllm.bridge_tp.gpu_direct_history.send_json"),
            self.assertRaises(Exception),
        ):
            value.receive_into_ring(
                migration_id="m-1",
                request_id="source-1",
                rank=1,
                layer_records=records(),
                make_ring=lambda n, d: None,
            )


if __name__ == "__main__":
    unittest.main()
