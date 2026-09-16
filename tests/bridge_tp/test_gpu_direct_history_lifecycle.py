# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import queue
import threading
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class TestGpuDirectHistoryLifecycle(unittest.TestCase):
    def test_delta_backlog_coalesces_to_latest_contiguous_watermark(self) -> None:
        from vllm.bridge_tp.kv_stream import _GpuDirectHistoryPublisher

        publisher = object.__new__(_GpuDirectHistoryPublisher)
        publisher.delta_queue = queue.Queue()
        first = {
            "block_ids": [10],
            "start_token": 64,
            "end_token": 80,
            "ready_event": "event-80",
        }
        publisher.delta_queue.put(
            {
                "block_ids": [10, 11],
                "start_token": 80,
                "end_token": 96,
                "ready_event": "event-96",
            }
        )
        publisher.delta_queue.put(
            {
                "block_ids": [10, 11],
                "start_token": 96,
                "end_token": 112,
                "ready_event": "event-112",
            }
        )

        merged, consumed = publisher._take_coalesced_delta(first)

        self.assertEqual(consumed, 3)
        self.assertEqual(merged["start_token"], 64)
        self.assertEqual(merged["end_token"], 112)
        self.assertEqual(merged["block_ids"], [10, 11])
        self.assertEqual(merged["ready_event"], "event-112")
        self.assertTrue(publisher.delta_queue.empty())

    def test_delta_backlog_rejects_a_watermark_gap(self) -> None:
        from vllm.bridge_tp.kv_stream import _GpuDirectHistoryPublisher

        publisher = object.__new__(_GpuDirectHistoryPublisher)
        publisher.delta_queue = queue.Queue()
        publisher.delta_queue.put(
            {
                "block_ids": [10],
                "start_token": 96,
                "end_token": 112,
                "ready_event": "event-112",
            }
        )
        first = {
            "block_ids": [10],
            "start_token": 64,
            "end_token": 80,
            "ready_event": "event-80",
        }

        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            publisher._take_coalesced_delta(first)

    def test_terminal_close_defers_communicator_destroy(self) -> None:
        from vllm.bridge_tp.gpu_direct_history import GpuDirectHistoryReceiver

        receiver = object.__new__(GpuDirectHistoryReceiver)
        receiver.connection = object()
        receiver.defer_communicator_destroy = True
        receiver.terminal_close_received_unix_s = None
        receiver._close_control = Mock()
        receiver.close = Mock()

        with (
            patch(
                "vllm.bridge_tp.gpu_direct_history.recv_json",
                return_value={"op": "CLOSE"},
            ),
            patch("vllm.bridge_tp.gpu_direct_history.send_json") as send,
        ):
            result = receiver.receive_delta(
                migration_id="migration",
                rank=0,
            )

        self.assertIsNone(result)
        send.assert_called_once_with(receiver.connection, {"status": "CLOSED"})
        receiver._close_control.assert_called_once_with()
        receiver.close.assert_not_called()
        self.assertIsNotNone(receiver.terminal_close_received_unix_s)

    def test_destroy_async_does_not_block_caller(self) -> None:
        from vllm.bridge_tp.gpu_direct_history import GpuDirectHistoryReceiver

        started = threading.Event()
        release = threading.Event()

        class FakeNccl:
            def ncclCommDestroy(self, comm: object) -> None:
                self.comm = comm
                started.set()
                release.wait(timeout=5)

        receiver = object.__new__(GpuDirectHistoryReceiver)
        receiver.nccl = FakeNccl()
        receiver.comm = object()
        receiver.stream = object()
        receiver.terminal_close_received_unix_s = 1.0
        receiver.control_closed_unix_s = 2.0
        receiver.destroy_started_unix_s = None
        receiver.destroy_completed_unix_s = None
        receiver.destroy_error = None
        receiver._destroy_thread = None
        updates: list[dict] = []

        receiver.destroy_async(updates.append)

        self.assertTrue(started.wait(timeout=1))
        self.assertIsNone(receiver.comm)
        self.assertIsNone(receiver.nccl)
        self.assertEqual(updates[0]["status"], "DESTROYING")
        release.set()
        assert receiver._destroy_thread is not None
        receiver._destroy_thread.join(timeout=1)
        self.assertFalse(receiver._destroy_thread.is_alive())
        self.assertEqual(updates[-1]["status"], "DESTROYED")
        self.assertIsNotNone(updates[-1]["destroy_ms"])


if __name__ == "__main__":
    unittest.main()
