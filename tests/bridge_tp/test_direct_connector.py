# SPDX-License-Identifier: Apache-2.0
"""Connector side of direct-to-KV history: the worker hook writes, load waits."""

import ast
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import torch

from vllm.bridge_tp.direct_history import HistoryLayout
from vllm.bridge_tp.streaming_connector import (
    BridgeTPStreamingConnector,
    BridgeTPStreamRequest,
    _ShadowCancelled,
)

SHAPE = [5, 2, 4, 2, 8]
NAMES = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]


def connector(slots=2, control=None):
    value = object.__new__(BridgeTPStreamingConnector)
    value.history_direct_ring_slots = slots
    value._registered_kv_caches = {
        name: torch.zeros((30, *SHAPE[1:])) for name in NAMES
    }
    value._direct_history_jobs = {}
    value._direct_history_lock = threading.Lock()
    value.takeover_control_path = control
    return value


def manifest(shape=SHAPE):
    return {
        "block_axis": 0,
        "layers": [
            {"layer_name": n, "rank_shape": shape, "block_axis": 0} for n in NAMES
        ],
    }


def request(blocks=(9, 2, 17, 4, 23)):
    return BridgeTPStreamRequest(
        migration_id="m-1",
        source_request_id="s-1",
        target_request_id="t-1",
        target_block_ids=list(blocks) + [29],  # one scheduler-owned tail block
        num_computed_tokens=5 * 4,
    )


def feed(ring, packed, chunk, delay_s=0.0):
    """Plays receive_into_ring: fill free slots in order, publish each."""
    def run():
        try:
            for offset in range(0, packed.numel(), chunk):
                piece = packed[offset : offset + chunk]
                slot, buffer, _ = ring.acquire(5)
                buffer.narrow(0, 0, piece.numel()).copy_(piece)
                ring.publish(slot, offset, piece.numel(), None)
                time.sleep(delay_s)
        except RuntimeError:
            pass  # the ring failed: the writer reported it

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def drive(value, until, limit_s=5.0):
    """Plays the scheduler loop: one pre-forward hook call per step."""
    stop = time.monotonic() + limit_s
    while not until() and time.monotonic() < stop:
        value._service_direct_history()
        time.sleep(0.001)


class DirectConnectorTests(unittest.TestCase):
    def test_hook_writes_every_chunk_and_finish_checks_once(self):
        value = connector(slots=2)
        layout = HistoryLayout([SHAPE] * len(NAMES))
        packed = torch.randn(layout.total_elements)
        req = request()
        ring = value._begin_direct_history(
            req, manifest(), 5, 70, torch.float32, torch.device("cpu")
        )
        self.assertEqual([b.numel() for b in ring.buffers], [70, 70])
        sender = feed(ring, packed, 70)
        job = value._direct_history_jobs["t-1"]
        drive(value, job.done.is_set)
        sender.join(5)
        result = value._finish_direct_history("t-1", time.monotonic() + 5)
        self.assertTrue(result["exact_readback"])
        self.assertTrue(result["direct_to_kv"])
        self.assertEqual(result["num_target_blocks"], 5)
        self.assertEqual(result["chunks"], -(-layout.total_elements // 70))
        self.assertGreaterEqual(result["hook_calls"], 1)
        # Exactly the history in the request's first blocks; tail block untouched.
        per_layer = layout.block_elements[0] * SHAPE[0]
        for i, name in enumerate(NAMES):
            kv = value._registered_kv_caches[name]
            expected = packed[i * per_layer : (i + 1) * per_layer]
            self.assertTrue(torch.equal(kv[[9, 2, 17, 4, 23]].reshape(-1), expected))
            self.assertEqual(int(kv[29].count_nonzero()), 0)
        self.assertEqual(value._direct_history_jobs, {})
        self.assertEqual(ring.buffers, [])  # closed

    def test_idle_steps_alone_finish_the_copy(self):
        # No forward work: only kv_connector_no_forward calls the hook.
        value = connector(slots=1)
        layout = HistoryLayout([SHAPE] * len(NAMES))
        packed = torch.randn(layout.total_elements)
        ring = value._begin_direct_history(
            request(), manifest(), 5, 40, torch.float32, torch.device("cpu")
        )
        feed(ring, packed, 40, delay_s=0.001)
        job = value._direct_history_jobs["t-1"]
        drive(value, job.done.is_set)
        self.assertTrue(job.done.is_set())
        self.assertGreater(job.hook_calls, 1)  # one slot: many steps
        self.assertTrue(value._finish_direct_history("t-1", time.monotonic() + 5)[
            "exact_readback"
        ])

    def test_readback_difference_fails_the_transfer(self):
        value = connector()
        layout = HistoryLayout([SHAPE] * len(NAMES))
        packed = torch.randn(layout.total_elements)
        packed[123] = float("nan")
        ring = value._begin_direct_history(
            request(), manifest(), 5, 64, torch.float32, torch.device("cpu")
        )
        feed(ring, packed, 64)
        drive(value, value._direct_history_jobs["t-1"].done.is_set)
        with self.assertRaises(ValueError):
            value._finish_direct_history("t-1", time.monotonic() + 5)
        self.assertEqual(value._direct_history_jobs, {})

    def test_copy_error_fails_the_ring_and_the_load(self):
        value = connector()
        # The model's KV has another block shape than the streamed history.
        value._registered_kv_caches[NAMES[1]] = torch.zeros((30, 2, 4, 2, 4))
        layout = HistoryLayout([SHAPE] * len(NAMES))
        ring = value._begin_direct_history(
            request(), manifest(), 5, layout.total_elements, torch.float32,
            torch.device("cpu"),
        )
        feed(ring, torch.randn(layout.total_elements), layout.total_elements)
        drive(value, value._direct_history_jobs["t-1"].done.is_set)
        with self.assertRaises(RuntimeError):
            value._finish_direct_history("t-1", time.monotonic() + 5)
        with self.assertRaises(RuntimeError):
            ring.acquire(0.1)  # the receive side sees the failure too

    def test_cancelled_shadow_and_timeout_release_the_job(self):
        with tempfile.TemporaryDirectory() as folder:
            control = Path(folder) / "takeover_control.json"
            control.write_text(json.dumps({"state": "CANCELLED"}))
            value = connector(control=control)
            value._begin_direct_history(
                request(), manifest(), 5, 64, torch.float32, torch.device("cpu")
            )
            with self.assertRaises(_ShadowCancelled):
                value._finish_direct_history("t-1", time.monotonic() + 5)
            self.assertEqual(value._direct_history_jobs, {})
        value = connector()
        value._begin_direct_history(
            request(), manifest(), 5, 64, torch.float32, torch.device("cpu")
        )
        with self.assertRaises(TimeoutError):
            value._finish_direct_history("t-1", time.monotonic() + 0.05)
        self.assertEqual(value._direct_history_jobs, {})

    def test_allocation_and_layout_must_match(self):
        value = connector()
        with self.assertRaises(ValueError):  # 4 blocks allocated for 5 streamed
            value._begin_direct_history(
                request(blocks=(1, 2, 3, 4)), manifest(), 4, 64, torch.float32,
                torch.device("cpu"),
            )
        bad = manifest()
        bad["layers"][0]["block_axis"] = 1
        with self.assertRaises(ValueError):
            value._begin_direct_history(
                request(), bad, 5, 64, torch.float32, torch.device("cpu")
            )
        missing = manifest()
        missing["layers"][1]["layer_name"] = "model.layers.9.self_attn.attn"
        with self.assertRaises(ValueError):
            value._begin_direct_history(
                request(), missing, 5, 64, torch.float32, torch.device("cpu")
            )
        value._begin_direct_history(
            request(), manifest(), 5, 64, torch.float32, torch.device("cpu")
        )
        with self.assertRaises(RuntimeError):
            value._begin_direct_history(
                request(), manifest(), 5, 64, torch.float32, torch.device("cpu")
            )

    def test_wiring_hook_before_forward_and_receive_selection(self):
        source = Path(__file__).resolve().parents[2] / (
            "vllm/bridge_tp/streaming_connector.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(
            n for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "BridgeTPStreamingConnector"
        )
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

        def calls(name):
            return [
                c.func.attr
                for c in ast.walk(methods[name])
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            ]

        start = calls("start_load_kv")
        self.assertLess(
            start.index("_service_gpu_restore_quiescence"),
            start.index("_service_direct_history"),
        )
        load = calls("_live_gpu_load")
        self.assertLess(load.index("receive_into_ring"), load.index("receive"))
        self.assertIn("_finish_direct_history", load)
        # DirectReceiveResult has no receive_done_event: every read of it in
        # the load thread outside the whole-payload inject must be guarded.
        text = ast.get_source_segment(source.read_text(encoding="utf-8"),
                                      methods["_live_gpu_load"])
        inject = text.index("history_layers is None")
        unguarded = [
            line for line in text[:inject].splitlines()
            if "direct.receive_done_event" in line and "getattr" not in line
            and ".query()" not in line
        ]
        self.assertEqual(unguarded, [])


if __name__ == "__main__":
    unittest.main()
