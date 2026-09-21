# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
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

    def test_sender_control_close_does_not_destroy_communicators(self) -> None:
        from vllm.bridge_tp.gpu_direct_history import GpuDirectHistorySender

        sender = object.__new__(GpuDirectHistorySender)
        sender.connections = [Mock(), Mock()]
        sender.comms = [object(), object()]
        sender.nccl = Mock()
        with (
            patch("vllm.bridge_tp.gpu_direct_history.send_json"),
            patch(
                "vllm.bridge_tp.gpu_direct_history.recv_json",
                return_value={"status": "CLOSED"},
            ),
        ):
            sender.close_control()

        self.assertEqual(sender.connections, [])
        self.assertEqual(len(sender.comms), 2)
        sender.nccl.ncclCommDestroy.assert_not_called()
        sender.nccl.ncclCommAbort.assert_not_called()

    def test_persistent_sender_ends_session_without_closing_channel(self) -> None:
        from vllm.bridge_tp.gpu_direct_history import GpuDirectHistorySender
        from vllm.bridge_tp.stream_protocol import (
            ChannelState,
            PayloadType,
            PersistentChannelLifecycle,
            SessionEnvelope,
        )

        sender = object.__new__(GpuDirectHistorySender)
        sender.persistent_channel = True
        sender.channel_generation = 9
        sender.lifecycle = PersistentChannelLifecycle(
            topology_key="tp1-to-tp4",
            expected_ranks=frozenset({0, 1}),
            channel_generation=9,
        )
        sender.lifecycle.mark_open()
        sender.lifecycle.start_session(
            migration_id="migration-1",
            request_id="request-1",
        )
        sender._active_migration_id = "migration-1"
        sender._active_request_id = "request-1"
        sender._next_sequence_by_rank = {0: 0, 1: 0}
        sender.connections = [Mock(), Mock()]

        acknowledgements = [
            {
                "status": "SESSION_ENDED",
                **SessionEnvelope(
                    channel_generation=9,
                    migration_id="migration-1",
                    request_id="request-1",
                    sequence_number=0,
                    payload_type=PayloadType.ACK,
                    token_start=0,
                    token_end=0,
                    rank=rank,
                ).to_wire(),
            }
            for rank in range(2)
        ]
        with (
            patch("vllm.bridge_tp.gpu_direct_history.send_json") as send,
            patch(
                "vllm.bridge_tp.gpu_direct_history.recv_json",
                side_effect=acknowledgements,
            ),
        ):
            sender.end_session()

        self.assertEqual(send.call_count, 2)
        self.assertEqual(sender.lifecycle.state, ChannelState.IDLE)
        self.assertEqual(sender.lifecycle.destroy_count, 0)
        self.assertEqual(sender.lifecycle.session_count, 1)
        self.assertEqual(len(sender.connections), 2)
        for connection in sender.connections:
            connection.close.assert_not_called()

    def test_persistent_receiver_ends_session_without_closing_channel(self) -> None:
        from vllm.bridge_tp.gpu_direct_history import GpuDirectHistoryReceiver
        from vllm.bridge_tp.stream_protocol import (
            ChannelState,
            PayloadType,
            PersistentChannelLifecycle,
            SessionEnvelope,
        )

        receiver = object.__new__(GpuDirectHistoryReceiver)
        receiver.connection = Mock()
        receiver.persistent_channel = True
        receiver.lifecycle = PersistentChannelLifecycle(
            topology_key="tp1-to-tp4-rank0",
            expected_ranks=frozenset({0}),
            channel_generation=3,
        )
        receiver.lifecycle.mark_open()
        session = receiver.lifecycle.start_session(
            migration_id="migration-1",
            request_id="request-1",
        )
        for sequence, payload_type in enumerate(
            (PayloadType.START_SESSION, PayloadType.HISTORY)
        ):
            session.accept_inbound(
                SessionEnvelope(
                    channel_generation=3,
                    migration_id="migration-1",
                    request_id="request-1",
                    sequence_number=sequence,
                    payload_type=payload_type,
                    token_start=0,
                    token_end=0,
                    rank=0,
                )
            )
        receiver._active_request_id = "request-1"
        receiver._last_delta_envelope = None
        terminal = {
            "op": "END_SESSION",
            **SessionEnvelope(
                channel_generation=3,
                migration_id="migration-1",
                request_id="request-1",
                sequence_number=2,
                payload_type=PayloadType.END_SESSION,
                token_start=0,
                token_end=0,
                rank=0,
            ).to_wire(),
        }
        with (
            patch(
                "vllm.bridge_tp.gpu_direct_history.recv_json",
                return_value=terminal,
            ),
            patch("vllm.bridge_tp.gpu_direct_history.send_json") as send,
        ):
            result = receiver.receive_delta(
                migration_id="migration-1",
                rank=0,
            )

        self.assertIsNone(result)
        self.assertEqual(receiver.lifecycle.state, ChannelState.IDLE)
        self.assertEqual(receiver.lifecycle.destroy_count, 0)
        receiver.connection.close.assert_not_called()
        self.assertEqual(send.call_count, 1)

    def test_source_pool_aborts_senders_only_at_process_shutdown(self) -> None:
        from vllm.bridge_tp import kv_stream

        sender = Mock()
        with kv_stream._retained_gpu_senders_lock:
            kv_stream._retained_gpu_senders.clear()
        self.assertEqual(kv_stream._retain_gpu_sender(sender), 1)
        sender.abort.assert_not_called()

        kv_stream._abort_retained_gpu_senders()

        sender.abort.assert_called_once_with()
        self.assertFalse(kv_stream._retained_gpu_senders)

    def test_connector_pool_retains_receiver_until_shutdown(self) -> None:
        from vllm.bridge_tp.streaming_connector import BridgeTPStreamingConnector

        connector = object.__new__(BridgeTPStreamingConnector)
        connector._retained_gpu_receivers = {}
        connector._retained_gpu_receivers_lock = threading.Lock()
        receiver = Mock()
        with tempfile.TemporaryDirectory() as directory:
            connector.manifest_path = Path(directory) / "manifest.json"
            connector._retain_gpu_receiver(
                receiver,
                request_id="request",
                migration_id="migration",
                tp_rank=0,
            )

            receipt_path = (
                Path(directory)
                / "gpu_communicator_destroy_receipts"
                / "tp_rank_0.json"
            )
            self.assertTrue(receipt_path.is_file())
            self.assertEqual(len(connector._retained_gpu_receivers), 1)
            receiver.abort.assert_not_called()

            connector.shutdown()

            receiver.abort.assert_called_once_with()
            self.assertFalse(connector._retained_gpu_receivers)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["status"], "ABORTED_AT_CONNECTOR_SHUTDOWN"
            )

    def test_post_takeover_destroy_records_commit_ordering(self) -> None:
        from vllm.bridge_tp.streaming_connector import (
            BridgeTPStreamingConnector,
        )

        connector = object.__new__(BridgeTPStreamingConnector)
        receiver = Mock()

        def destroy_async(callback: object) -> None:
            callback(
                {
                    "status": "DESTROYED",
                    "destroy_started_unix_s": 12.5,
                    "destroy_completed_unix_s": 13.0,
                    "destroy_ms": 500.0,
                    "error": None,
                }
            )

        receiver.destroy_async.side_effect = destroy_async
        with tempfile.TemporaryDirectory() as directory:
            connector.manifest_path = Path(directory) / "manifest.json"
            connector._destroy_gpu_receiver_after_takeover(
                receiver,
                request_id="request",
                migration_id="migration",
                tp_rank=2,
                target_ready_unix_s=10.0,
                commit_observed_unix_s=12.0,
            )

            receipt = json.loads(
                (
                    Path(directory)
                    / "gpu_communicator_destroy_receipts"
                    / "tp_rank_2.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "DESTROYED")
            self.assertEqual(
                receipt["lifecycle"], "POST_TAKEOVER_ASYNC_DESTROY"
            )
            self.assertEqual(
                receipt["destroy_started_after_target_ready_ms"], 2500.0
            )
            self.assertEqual(
                receipt["destroy_started_after_commit_ms"], 500.0
            )


if __name__ == "__main__":
    unittest.main()
