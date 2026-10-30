# SPDX-License-Identifier: Apache-2.0
"""CPU byte/order checks and actual sender chunk-loop regression coverage."""

import ast
import os
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from vllm.bridge_tp.bounded_history import HistoryPacker, packing_plan
from vllm.bridge_tp.stream_protocol import PayloadType, PersistentChannelLifecycle


def caches():
    return [
        torch.arange(2 * 9 * 4 * 8 * 3).reshape(2, 9, 4, 8, 3).float() + i * 10000
        for i in range(3)
    ]


def reference(values, ids, b, h, tp):
    selected = [x.index_select(b, torch.tensor(ids)) for x in values]
    return [
        torch.cat([x.chunk(tp, dim=h)[r].contiguous().view(-1) for x in selected])
        for r in range(tp)
    ]


class BoundedHistoryTests(unittest.TestCase):
    def test_real_publication_guard_defers_before_starting_a_wire_session(self):
        path = Path(__file__).resolve().parents[2] / "vllm/bridge_tp/kv_stream.py"
        tree = ast.parse(path.read_text())
        publish = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_publish_request"
        )
        guard = next(
            n
            for n in publish.body
            if isinstance(n, ast.If) and "pack_budget" in ast.unparse(n.test)
        )
        fn = ast.parse("def guarded():\n    pass").body[0]
        fn.body = [guard, ast.Return(value=ast.Constant(True))]
        with tempfile.TemporaryDirectory() as temp:
            receipt = []
            ns = dict(
                torch=torch,
                os=os,
                time=time,
                kv_caches=caches(),
                block_ids=[0, 1],
                block_axis=1,
                pack_budget=128,
                raw_source_bytes=123,
                request_id="r",
                config=NS(
                    gpu_direct_history=True,
                    head_axis=3,
                    target_tp_size=2,
                    run_dir=Path(temp),
                    migration_id="m",
                ),
                _atomic_json_dump=lambda row, path: receipt.append(row),
            )
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                    str(path),
                    "exec",
                ),
                ns,
            )
            with patch.object(torch.cuda, "mem_get_info", return_value=(1, 2**30)):
                self.assertIsNone(ns["guarded"]())
            self.assertEqual(receipt[-1]["status"], "DEFERRED_GPU_MEMORY")
            with patch.object(torch.cuda, "mem_get_info", return_value=(2**30, 2**31)):
                self.assertTrue(ns["guarded"]())
            self.assertEqual(receipt[-1]["status"], "READY_FOR_BOUNDED_HISTORY")

    def test_chunk_bytes_equal_original_for_tp2_tp4_and_cross_layer_chunks(self):
        for tp in (2, 4):
            for values, b, h in (
                (caches(), 1, 3),
                ([x.permute(1, 0, 3, 2, 4) for x in caches()], 0, 2),
            ):
                ids = [6, 1, 8, 0]
                expected = reference(values, ids, b, h, tp)
                for budget in (tp * 4 * 7, tp * 4 * 193, 100000):
                    plan = packing_plan(values, ids, b, h, tp, budget)
                    buffers = [torch.empty(plan.chunk_elements) for _ in range(tp)]
                    packer = HistoryPacker(values, ids, b, h, tp)
                    actual = [[] for _ in range(tp)]
                    for off in range(0, plan.rank_elements, plan.chunk_elements):
                        n = min(plan.chunk_elements, plan.rank_elements - off)
                        packer.fill(buffers, n)
                        for r in range(tp):
                            actual[r].append(buffers[r][:n].clone())
                    for r in range(tp):
                        self.assertTrue(torch.equal(torch.cat(actual[r]), expected[r]))
                    self.assertIsNone(packer.shards)
                    packer.close()
                    self.assertIsNone(packer.index)
                    self.assertLessEqual(plan.chunk_buffer_bytes, budget)

    def test_peak_depends_on_one_layer_not_number_of_layers(self):
        first = packing_plan(caches(), [0, 1, 2], 1, 3, 2, 128)
        many = packing_plan(caches() * 20, [0, 1, 2], 1, 3, 2, 128)
        self.assertEqual(first.source_peak_bound_bytes, many.source_peak_bound_bytes)
        self.assertEqual(many.rank_elements, 20 * first.rank_elements)
        with self.assertRaises(ValueError):
            packing_plan(caches(), [9], 1, 3, 2, 128)
        with self.assertRaises(ValueError):
            packing_plan(caches(), [0], 1, 3, 2, 1)

    def run_sender(self, budget, paced=False):
        # Compile the real method with fake CUDA/NCCL boundaries; tensor packing
        # and bytes are real CPU torch operations. No engine or GPU is loaded.
        path = (
            Path(__file__).resolve().parents[2] / "vllm/bridge_tp/gpu_direct_history.py"
        )
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "GpuDirectHistorySender"
        )
        methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in {"send", "_next_envelope"}
        ]
        ns = dict(
            torch=torch,
            time=time,
            PayloadType=PayloadType,
            ProtocolViolation=RuntimeError,
            _cuda_device=lambda d: d,
            _HISTORY_PACING_CHUNK_BYTES=1024,
            _session_message=lambda op, e, **kw: dict(op=op, **e.to_wire(), **kw),
            _validate_ack=lambda *a, **kw: None,
        )
        from vllm.bridge_tp.stream_protocol import SessionEnvelope

        ns["SessionEnvelope"] = SessionEnvelope
        exec(
            compile(
                ast.Module(
                    body=[
                        ast.parse("from __future__ import annotations").body[0],
                        *methods,
                    ],
                    type_ignores=[],
                ),
                str(path),
                "exec",
            ),
            ns,
        )
        values = caches()
        ids = [6, 1, 8, 0]
        tp = 2
        expected = reference(values, ids, 1, 3, tp)
        lifecycle = PersistentChannelLifecycle(
            topology_key="a|b", expected_ranks=frozenset({0, 1}), channel_generation=1
        )
        lifecycle.mark_open()
        owner = NS(
            device=torch.device("cpu"),
            nccl=Mock(),
            persistent_channel=True,
            channel_generation=1,
            topology_key="a|b",
            target_tp_size=2,
            connections=[Mock(), Mock()],
            comms=[Mock(), Mock()],
            streams=[Mock(), Mock()],
            producer_stream=Mock(),
            lifecycle=lifecycle,
            _packed_buffers=[],
            buffer_capacity_elements=0,
            buffer_high_water_bytes=0,
            preconnect_completed_unix_s=1,
            warmup_completed_unix_s=2,
        )
        owner._next_envelope = lambda **kw: ns["_next_envelope"](owner, **kw)
        ready = (
            [dict(status="SESSION_READY")] * 2
            + [dict(status="READY_TO_RECV", numel=len(expected[0]))] * 2
            + [dict(status="RECEIVED")] * 2
            + [dict(status="COMPLETE")] * 2
        )
        sent = [[] for _ in range(tp)]
        messages = []

        def transfer(nccl, comms, tensors, streams):
            if budget:
                self.assertLessEqual(
                    sum(t.numel() * t.element_size() for t in tensors), budget
                )
            for r, t in enumerate(tensors):
                sent[r].append(t.clone())

        ns.update(
            recv_json=Mock(side_effect=ready),
            send_json=lambda c, m: messages.append(m),
            _send_group=transfer,
        )
        with (
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
        ):
            receipt = ns["send"](
                owner,
                migration_id="m",
                request_id="r",
                channel_generation=1,
                persistent_channel=True,
                history_end_token=16,
                kv_caches=values,
                layer_names=["a", "b", "c"],
                block_ids=ids,
                block_axis=1,
                head_axis=3,
                target_tp_size=2,
                target_addresses=["a", "b"],
                keep_open=True,
                history_pacing=paced,
                history_rate_provider=lambda: 1e12,
                source_pack_budget_bytes=budget,
            )
        for r in range(tp):
            self.assertTrue(torch.equal(torch.cat(sent[r]), expected[r]))
        if budget:
            self.assertEqual(owner._packed_buffers, [])
            self.assertGreater(receipt[0]["source_history_packing_chunks"], 1)
            self.assertTrue(receipt[0]["source_history_payload_released"])
        else:
            self.assertEqual(len(owner._packed_buffers), 2)
            self.assertFalse(receipt[0]["source_bounded_history"])
        for m in messages:
            if m["op"] == "HISTORY":
                chunk = m.get("history_chunk_elements", len(expected[0]))
                expected_sizes = [
                    min(chunk, len(expected[0]) - i)
                    for i in range(0, len(expected[0]), chunk)
                ]
                self.assertEqual([t.numel() for t in sent[0]], expected_sizes)
        self.assertEqual(receipt[0]["raw_tensor_bytes"], expected[0].numel() * 4)
        self.assertEqual(receipt[0]["history_pacing_enabled"], paced)
        self.assertEqual(receipt[0]["history_pacing_chunk_count"] > 0, paced)

    def test_real_sender_full_speed_uses_chunk_protocol_and_no_full_source_buffer(self):
        self.run_sender(128)

    def test_legacy_full_pack_and_pacing_bytes_remain_compatible(self):
        self.run_sender(0)
        self.run_sender(0, True)
