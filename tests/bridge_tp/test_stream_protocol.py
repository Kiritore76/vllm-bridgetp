# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import socket
import threading
import unittest

import torch

from vllm.bridge_tp import stream_protocol
from vllm.bridge_tp.stream_protocol import (
    ChannelState,
    MigrationSession,
    PayloadType,
    PersistentChannelLifecycle,
    ProtocolViolation,
    SessionEnvelope,
    SessionState,
    deserialize_rank_payload,
    recv_payload_frames,
    send_payload_frames,
    serialize_rank_payload,
    sha256_bytes,
)


class TestStreamProtocol(unittest.TestCase):
    def test_tensor_payload_roundtrip(self) -> None:
        source = {
            "format_version": 1,
            "layers": {"layer.0": torch.arange(24).reshape(2, 3, 4)},
        }
        restored = deserialize_rank_payload(serialize_rank_payload(source))
        self.assertEqual(restored["format_version"], 1)
        self.assertTrue(
            torch.equal(restored["layers"]["layer.0"], source["layers"]["layer.0"])
        )

    def test_framed_socket_roundtrip(self) -> None:
        sender, receiver = socket.socketpair()
        payload = bytes(range(251)) * 100
        thread = threading.Thread(
            target=send_payload_frames,
            args=(sender, payload),
            kwargs={"chunk_bytes": 1024},
        )
        thread.start()
        received, metrics = recv_payload_frames(
            receiver,
            payload_bytes=len(payload),
            num_frames=(len(payload) + 1023) // 1024,
            payload_sha256=sha256_bytes(payload),
            max_frame_bytes=1024,
        )
        thread.join(timeout=5)
        sender.close()
        receiver.close()
        self.assertEqual(received, payload)
        self.assertEqual(metrics["payload_sha256"], sha256_bytes(payload))

    def test_corrupted_frame_is_rejected(self) -> None:
        sender, receiver = socket.socketpair()
        original = b"correct payload"
        corrupted = b"corrupt payload"
        sender.sendall(
            stream_protocol._FRAME_HEADER.pack(
                0, len(corrupted), hashlib.sha256(original).digest()
            )
        )
        sender.sendall(corrupted)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            recv_payload_frames(
                receiver,
                payload_bytes=len(corrupted),
                num_frames=1,
                payload_sha256=sha256_bytes(corrupted),
                max_frame_bytes=1024,
            )
        sender.close()
        receiver.close()


class TestPersistentChannelProtocol(unittest.TestCase):
    def envelope(
        self,
        *,
        sequence: int = 0,
        generation: int = 7,
        migration_id: str = "migration-1",
        request_id: str = "request-1",
        payload_type: PayloadType = PayloadType.HISTORY,
        token_start: int = 0,
        token_end: int = 16,
        rank: int = 0,
    ) -> SessionEnvelope:
        return SessionEnvelope(
            channel_generation=generation,
            migration_id=migration_id,
            request_id=request_id,
            sequence_number=sequence,
            payload_type=payload_type,
            token_start=token_start,
            token_end=token_end,
            rank=rank,
        )

    def test_envelope_wire_roundtrip(self) -> None:
        envelope = self.envelope()
        self.assertEqual(SessionEnvelope.from_wire(envelope.to_wire()), envelope)

    def test_envelope_requires_complete_identity(self) -> None:
        value = self.envelope().to_wire()
        del value["request_id"]
        with self.assertRaisesRegex(ProtocolViolation, "missing request_id"):
            SessionEnvelope.from_wire(value)

    def test_session_rejects_stale_and_out_of_order_messages(self) -> None:
        session = MigrationSession(
            channel_generation=7,
            migration_id="migration-1",
            request_id="request-1",
            expected_ranks=frozenset({0, 1, 2, 3}),
        )
        session.start()
        with self.assertRaisesRegex(ProtocolViolation, "stale channel"):
            session.accept_inbound(self.envelope(generation=6))
        with self.assertRaisesRegex(ProtocolViolation, "out-of-order"):
            session.accept_inbound(self.envelope(sequence=1))
        session.accept_inbound(self.envelope())
        with self.assertRaisesRegex(ProtocolViolation, "duplicate or replayed"):
            session.accept_inbound(self.envelope())

    def test_session_rejects_cross_request_and_overlapping_ranges(self) -> None:
        session = MigrationSession(
            channel_generation=7,
            migration_id="migration-1",
            request_id="request-1",
            expected_ranks=frozenset({0}),
        )
        session.start()
        with self.assertRaisesRegex(ProtocolViolation, "request_id differs"):
            session.accept_inbound(self.envelope(request_id="request-old"))
        session.accept_inbound(self.envelope())
        with self.assertRaisesRegex(ProtocolViolation, "overlapping"):
            session.accept_inbound(
                self.envelope(sequence=1, token_start=8, token_end=24)
            )

    def test_sequences_and_watermarks_are_independent_per_rank(self) -> None:
        session = MigrationSession(
            channel_generation=7,
            migration_id="migration-1",
            request_id="request-1",
            expected_ranks=frozenset({0, 1, 2, 3}),
        )
        session.start()
        for rank in range(4):
            session.accept_inbound(self.envelope(rank=rank))
        self.assertEqual(
            session.next_inbound_sequences,
            {0: 1, 1: 1, 2: 1, 3: 1},
        )
        self.assertEqual(
            session.history_watermarks,
            {0: 16, 1: 16, 2: 16, 3: 16},
        )

    def test_session_reset_clears_request_local_state(self) -> None:
        session = MigrationSession(
            channel_generation=7,
            migration_id="migration-1",
            request_id="request-1",
            expected_ranks=frozenset({0, 1}),
        )
        session.start()
        session.accept_inbound(self.envelope())
        session.expect_ack(rank=0, sequence_number=11)
        session.acknowledge(rank=0, sequence_number=11)
        session.mark_rank_ready(0)
        session.mark_rank_ready(1)
        session.commit()
        session.reset()
        self.assertEqual(session.state, SessionState.RESET)
        self.assertEqual(session.next_inbound_sequences, {})
        self.assertEqual(session.history_watermarks, {})
        self.assertEqual(session.delta_watermarks, {})
        self.assertEqual(session.ready_ranks, set())
        self.assertEqual(session.pending_ack_sequences, set())

    def test_channel_reuses_one_generation_for_sequential_sessions(self) -> None:
        channel = PersistentChannelLifecycle(
            topology_key="tp1-to-tp4",
            expected_ranks=frozenset({0, 1, 2, 3}),
            channel_generation=4,
        )
        channel.mark_open()
        for index in range(2):
            session = channel.start_session(
                migration_id=f"migration-{index}",
                request_id=f"request-{index}",
            )
            for rank in range(4):
                session.mark_rank_ready(rank)
            session.commit()
            channel.finish_session()
        self.assertEqual(channel.state, ChannelState.IDLE)
        self.assertEqual(channel.channel_generation, 4)
        self.assertEqual(channel.create_count, 1)
        self.assertEqual(channel.destroy_count, 0)
        self.assertEqual(channel.session_count, 2)

    def test_channel_rejects_two_active_sessions(self) -> None:
        channel = PersistentChannelLifecycle(
            topology_key="tp1-to-tp4",
            expected_ranks=frozenset({0}),
        )
        channel.mark_open()
        channel.start_session(migration_id="migration-1", request_id="request-1")
        with self.assertRaisesRegex(ProtocolViolation, "not idle"):
            channel.start_session(
                migration_id="migration-2", request_id="request-2"
            )

    def test_rebuild_changes_generation_and_shutdown_is_idempotent(self) -> None:
        channel = PersistentChannelLifecycle(
            topology_key="topology-a",
            expected_ranks=frozenset({0}),
        )
        channel.mark_open()
        channel.begin_rebuild(topology_key="topology-b")
        self.assertEqual(channel.channel_generation, 1)
        self.assertEqual(channel.rebuild_count, 1)
        self.assertEqual(channel.destroy_count, 1)
        channel.mark_open()
        channel.shutdown()
        channel.shutdown()
        self.assertEqual(channel.state, ChannelState.DESTROYED)
        self.assertEqual(channel.create_count, 2)
        self.assertEqual(channel.destroy_count, 2)


if __name__ == "__main__":
    unittest.main()
