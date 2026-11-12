# SPDX-License-Identifier: Apache-2.0
"""Direct-to-KV: prebind defers the history; cancel before admission drains."""

import ast
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from vllm.bridge_tp.direct_history import DiscardRing, HistoryLayout
from vllm.bridge_tp.streaming_connector import (
    BridgeTPStreamingConnector,
    _ShadowCancelled,
)

SOURCE = Path(__file__).resolve().parents[2] / "vllm/bridge_tp/streaming_connector.py"
SHAPE = [4, 2, 4, 2, 8]


def connector(folder):
    value = object.__new__(BridgeTPStreamingConnector)
    value.manifest_path = Path(folder) / "session_manifest.json"
    value.history_direct_to_kv = True
    value.history_direct_ring_slots = 2
    value.socket_timeout_s = 5
    value._direct_history_lock = threading.Lock()
    value._deferred_histories = {}
    value._prebound_gpu_receiver_lock = threading.Lock()
    value._registered_kv_caches = {"layer.0": torch.zeros((10, *SHAPE[1:]))}
    return value


def manifest():
    return {"layers": [{"layer_name": "layer.0", "rank_shape": SHAPE}]}


class FakeReceiver:
    """Plays the sender's history arriving into whatever ring it is given."""

    def __init__(self, chunk):
        self.chunk = chunk
        self.calls = []

    def receive_into_ring(self, **kwargs):
        self.calls.append(kwargs)
        total = HistoryLayout([SHAPE]).total_elements
        ring = kwargs["make_ring"](self.chunk, torch.float32)
        for offset in range(0, total, self.chunk):
            slot, buffer, _ = ring.acquire(1)  # never blocks: slots come back
            length = min(self.chunk, total - offset)
            buffer.narrow(0, 0, length).fill_(1.0)
            ring.publish(slot, offset, length, None)
        self.ring = ring
        return "received"


class ClaimTests(unittest.TestCase):
    def test_live_load_claims_a_deferred_history_once(self):
        with tempfile.TemporaryDirectory() as folder:
            value = connector(folder)
            value._deferred_histories["m-1"] = "DEFERRED"
            value._claim_deferred_history("m-1")
            self.assertNotIn("m-1", value._deferred_histories)
            value._claim_deferred_history("m-2")  # never deferred: no-op
            value._deferred_histories["m-3"] = "DISCARDING"
            with self.assertRaises(_ShadowCancelled):
                value._claim_deferred_history("m-3")

    def test_connectors_without_direct_state_ignore_claims(self):
        value = object.__new__(BridgeTPStreamingConnector)
        value._claim_deferred_history("m-1")


class CancelledBeforeAdmissionTests(unittest.TestCase):
    def cleanup(self, value, receiver, migration="m-1"):
        with value._prebound_gpu_receiver_lock:
            value._prebound_gpu_receiver = receiver
        value._finish_cancelled_receiver = Mock()
        return value._cleanup_cancelled_deferred_history(
            manifest=manifest(),
            migration_id=migration,
            source_request_id="source-1",
            tp_rank=1,
        )

    def test_cancelled_deferred_history_is_received_discarded_and_drained(self):
        with tempfile.TemporaryDirectory() as folder:
            value = connector(folder)
            value._deferred_histories["m-1"] = "DEFERRED"
            (Path(folder) / "takeover_state.json").write_text(
                json.dumps({"state": "CANCELLED"})
            )
            receiver = FakeReceiver(chunk=50)
            self.assertTrue(self.cleanup(value, receiver))
            self.assertIsInstance(receiver.ring, DiscardRing)
            self.assertEqual(receiver.ring.discarded, HistoryLayout([SHAPE]).total_elements)
            # Only a couple of chunk-sized slots, never a history-sized buffer.
            self.assertEqual([b.numel() for b in receiver.ring.buffers], [50, 50])
            self.assertEqual(receiver.calls[0]["request_id"], "source-1")
            value._finish_cancelled_receiver.assert_called_once()
            self.assertEqual(
                value._finish_cancelled_receiver.call_args.kwargs["request_id"],
                "NOT_ADMITTED",
            )
            receipt = json.loads(
                (Path(folder) / "gpu_cleanup_receipts/not_admitted/tp_rank_1.json")
                .read_text()
            )
            self.assertEqual(receipt["status"], "CANCELLED_DEFERRED_HISTORY_DISCARDED")
            self.assertEqual(value._deferred_histories, {})

    def test_nothing_happens_unless_deferred_and_cancelled(self):
        with tempfile.TemporaryDirectory() as folder:
            value = connector(folder)
            receiver = FakeReceiver(chunk=50)
            self.assertFalse(self.cleanup(value, receiver))  # not deferred
            value._deferred_histories["m-1"] = "DEFERRED"
            self.assertFalse(self.cleanup(value, receiver))  # no state file yet
            (Path(folder) / "takeover_state.json").write_text(
                json.dumps({"state": "PREPARING"})
            )
            self.assertFalse(self.cleanup(value, receiver))  # still live
            self.assertEqual(value._deferred_histories, {"m-1": "DEFERRED"})
            self.assertEqual(receiver.calls, [])

    def test_a_live_claim_wins_over_the_discard(self):
        with tempfile.TemporaryDirectory() as folder:
            value = connector(folder)
            value._deferred_histories["m-1"] = "DEFERRED"
            (Path(folder) / "takeover_state.json").write_text(
                json.dumps({"state": "CANCELLED"})
            )
            value._claim_deferred_history("m-1")  # admitted first
            receiver = FakeReceiver(chunk=50)
            self.assertFalse(self.cleanup(value, receiver))
            self.assertEqual(receiver.calls, [])

    def test_discard_failure_still_clears_the_state(self):
        with tempfile.TemporaryDirectory() as folder:
            value = connector(folder)
            value._deferred_histories["m-1"] = "DEFERRED"
            (Path(folder) / "takeover_state.json").write_text(
                json.dumps({"state": "CANCELLED"})
            )
            with self.assertRaises(RuntimeError):
                self.cleanup(value, None)  # no receiver published
            self.assertEqual(value._deferred_histories, {})


class DiscardRingTests(unittest.TestCase):
    def test_publish_frees_the_slot_at_once(self):
        ring = DiscardRing(1, 4, lambda n: torch.zeros(n))
        for offset in (0, 4, 8):
            slot, _, _ = ring.acquire(0.1)
            ring.publish(slot, offset, 4, None)
        self.assertEqual(ring.discarded, 12)
        self.assertEqual(ring.take_ready(), [])


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(self.text)
        cls = next(
            n for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "BridgeTPStreamingConnector"
        )
        self.methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

    def test_prebind_receives_history_only_outside_direct_mode(self):
        prebind = self.methods["_start_gpu_direct_receiver_prebind"]
        receives = []
        for node in ast.walk(prebind):
            if not isinstance(node, ast.If):
                continue
            test = ast.get_source_segment(self.text, node.test)
            if "history_direct_to_kv" not in test or "persistent_channel" not in test:
                continue
            in_body = [
                c for n in node.body for c in ast.walk(n)
                if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "receive"
            ]
            in_else = [
                c for n in node.orelse for c in ast.walk(n)
                if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "receive"
            ]
            receives.append((len(in_body), len(in_else)))
        self.assertEqual(receives, [(0, 1)])

    def test_live_claims_before_receiving_and_records_the_path(self):
        load = ast.get_source_segment(self.text, self.methods["_live_gpu_load"])
        self.assertLess(
            load.index("_claim_deferred_history"), load.index("receive_into_ring")
        )
        self.assertIn('"history_path"', load)
        self.assertIn('"DIRECT_TO_KV"', load)
        hook = ast.get_source_segment(self.text, self.methods["_service_direct_history"])
        self.assertIn("record_stream", hook)


if __name__ == "__main__":
    unittest.main()
