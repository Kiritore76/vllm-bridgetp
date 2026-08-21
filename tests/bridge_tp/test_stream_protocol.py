# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import socket
import threading
import unittest

import torch

from vllm.bridge_tp import stream_protocol
from vllm.bridge_tp.stream_protocol import (
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


if __name__ == "__main__":
    unittest.main()
