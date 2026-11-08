# SPDX-License-Identifier: Apache-2.0
"""First receiver model/topology barrier and formal warmup gates; CPU only."""

import ast
import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from tests.bridge_tp import test_six_conversion as lifecycle
from tools.bridge_tp import six_conversion as six
from tools.bridge_tp import six_edge_batch as batch

REPO = Path(__file__).resolve().parents[2]


class OrderingTests(unittest.TestCase):
    def test_actual_cold_receiver_starts_and_warms_before_publishing_channels(self):
        lifecycle.LifecycleTests().run_edge("L0-L1AC", require_cold_order=True)

    def test_single_edge_cli_plan_records_only_requested_trial(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with patch(
                "sys.argv",
                [
                    "six_edge_batch",
                    "plan",
                    "--directory",
                    folder,
                    "--edges",
                    "L0-L1AC",
                    "--inputs",
                    str(
                        REPO
                        / "experiments/platforms/qwen3_pressure_observation/inputs.json"
                    ),
                ],
            ):
                status = batch.main()
            value = json.loads((Path(folder) / "batch_plan.json").read_text())
            self.assertEqual(status, 0)
            self.assertEqual(value["edges"], ["L0-L1AC"])
            self.assertTrue(value["execution"].startswith("1 independent"))

    def test_single_edge_summary_never_implies_other_edges_passed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            d = root / "L0-L1AC"
            d.mkdir()
            (d / "case_acceptance.json").write_text(
                json.dumps(dict(edge="L0-L1AC", passed=True, cleanup_confirmed=True))
            )
            summary = batch.aggregate(root, ["L0-L1AC"])
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["expected_edges"], ["L0-L1AC"])
            self.assertFalse(batch.aggregate(root)["passed"])
            for bad in ([], ["L0-L1AC", "L0-L1AC"], ["wrong"]):
                with self.assertRaises(ValueError):
                    batch.aggregate(root, bad)


class BarrierTests(unittest.TestCase):
    def backend(self, root):
        backend = object.__new__(six.SixBackend)
        target = dict(
            instance_id="B-g8-tp2",
            mechanism_root=str(root / "receiver"),
            channel_plan=str(root / "plan.json"),
            defer_channels_until_model_ready=True,
            connector_sources=[
                dict(source="A-g1-tp1", generation=1001, base_port=18030)
            ],
            mechanism_dir=str(root / "receiver/active"),
        )
        backend.targets = {"B-g8-tp2": target}
        backend.rows = {"A-g1-tp1": dict(targets_file=str(root / "source.json"))}
        backend.registry = NS(instances={"A-g1-tp1": NS(lifecycle="ACTIVE")})
        backend.ops = NS(event=Mock())
        backend.channel_graph = NS(acknowledged=Mock(return_value=True))
        backend.joined_at = 0
        backend.clock = lambda: 0
        (root / "plan.json").write_text("[]")
        (root / "source.json").write_text("[]")
        return backend

    def test_no_topology_publication_before_model_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            b = self.backend(root)
            with patch.object(
                six.SequenceBackend, "warm_and_verify", return_value=False
            ):
                self.assertFalse(b.warm_and_verify({}))
            self.assertEqual(json.loads((root / "plan.json").read_text()), [])
            self.assertEqual(json.loads((root / "source.json").read_text()), [])
            b.ops.event.assert_not_called()

    def test_model_ready_still_waits_for_formal_warmup_and_timeout_has_sender_stage(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            b = self.backend(root)
            with patch.object(
                six.SequenceBackend, "warm_and_verify", return_value=True
            ):
                self.assertFalse(b.warm_and_verify({}))
                self.assertEqual(len(json.loads((root / "plan.json").read_text())), 1)
                self.assertEqual(len(json.loads((root / "source.json").read_text())), 1)
                b.ops.event.assert_called_once()
                p = (
                    root
                    / "receiver/sessions/s/gpu_channel_connection_receipts"
                    / "source-A-g1-tp1-B-g8-tp2.json"
                )
                p.parent.mkdir(parents=True)
                p.write_text(
                    json.dumps(dict(stage="sender.comm_init", status="STARTED"))
                )
                b.clock = lambda: 781
                with self.assertRaisesRegex(TimeoutError, "sender.comm_init"):
                    b.warm_and_verify({})
            b.channel_graph.acknowledged.assert_not_called()

    def test_controller_budget_covers_trigger_deadline_ack_without_changing_deadline(
        self,
    ):
        request = dict(sequence=[dict(trigger_s=60, deadline_s=900)])
        self.assertEqual(six.controller_timeout_s(request, 900), 1200)
        self.assertEqual(six.controller_timeout_s(request, 1500), 1500)
        self.assertEqual(request["sequence"][0]["deadline_s"], 900)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.scene = Path(self.tmp.name)
        self.root = self.scene / "controller/migration/B-g8-tp2"
        self.root.mkdir(parents=True)
        self.channels = [
            dict(
                source=f"S-g{i + 1}-tp1", generation=1001 + i, base_port=18030 + i * 10
            )
            for i in range(4)
        ]
        (self.root / "gpu_channel_plan.json").write_text(json.dumps(self.channels))
        self.events = [
            dict(kind="CHANNEL_TOPOLOGY_DEFERRED", instance="B-g8-tp2", unix_s=1),
            dict(kind="VERIFIED", instance="B-g8-tp2", unix_s=2),
            dict(kind="CHANNEL_TOPOLOGY_PUBLISHED", instance="B-g8-tp2", unix_s=3),
            dict(kind="ROUTE_PUBLISHED", unix_s=4),
        ]
        self.receipts = self.root / "sessions/cpu/gpu_channel_preconnect_receipts"
        self.receipts.mkdir(parents=True)
        for c in self.channels:
            for name in [f"source-{c['source']}-B-g8-tp2.json"] + [
                f"tp_rank_{r}-{c['source']}.json" for r in range(2)
            ]:
                (self.receipts / name).write_text(
                    json.dumps(
                        dict(
                            status="WARMUP_COMPLETE",
                            channel_generation=c["generation"],
                            completed_unix_s=3.5,
                        )
                    )
                )
        self.write_events()

    def write_events(self):
        (self.scene / "conversion_events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in self.events) + "\n"
        )

    def test_all_twelve_formal_receipts_and_order_required(self):
        self.assertEqual(
            batch.first_target_order(self.scene)["formal_warmup_receipts"], 12
        )

    def test_topology_before_model_verified_rejected(self):
        self.events[2]["unix_s"] = 1.5
        self.write_events()
        with self.assertRaises(ValueError):
            batch.first_target_order(self.scene)

    def test_bind_or_missing_receipt_never_counts_as_warmup(self):
        p = next(self.receipts.glob("*.json"))
        p.unlink()
        with self.assertRaises(ValueError):
            batch.first_target_order(self.scene)
        p.write_text(
            json.dumps(
                dict(status="COMPLETE", channel_generation=1001, completed_unix_s=3.5)
            )
        )
        with self.assertRaises(ValueError):
            batch.first_target_order(self.scene)

    def test_wrong_generation_and_after_route_and_before_model_rejected(self):
        p = next(self.receipts.glob("*.json"))
        original = json.loads(p.read_text())
        for mutation in (
            dict(channel_generation=999),
            dict(completed_unix_s=4.5),
            dict(completed_unix_s=1.5),
        ):
            p.write_text(json.dumps(original | mutation))
            with self.assertRaises(ValueError):
                batch.first_target_order(self.scene)


class SenderDiagnosticTests(unittest.TestCase):
    def run_sender(self, fail):
        source = REPO / "vllm/bridge_tp/kv_stream.py"
        fn = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_preconnect_one"
        )
        namespace = dict(
            os=os,
            time=time,
            local_role=lambda: NS(own_file=lambda s: s),
            _sender_addresses=lambda c: ["127.0.0.1:18030", "127.0.0.1:18031"],
            _release_persistent_gpu_sender=Mock(),
            _discard_persistent_gpu_sender=Mock(),
        )

        def dump(value, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))

        namespace["_atomic_json_dump"] = dump
        sender = NS(
            nccl=None, warmup_completed_unix_s=None, preconnect_completed_unix_s=2
        )

        def preconnect(**kwargs):
            sender.channel_progress("sender.comm_init", 1)
            if fail:
                raise RuntimeError("injected comm failure")
            sender.nccl = object()

        def warmup():
            sender.channel_progress("sender.warmup.nccl_send")
            sender.warmup_completed_unix_s = 3
            return 1024

        sender.preconnect = preconnect
        sender.warmup_preconnected_channel = warmup
        namespace["_acquire_persistent_gpu_sender"] = lambda c, d: ("key", sender)
        exec(
            compile(
                ast.Module(
                    body=[ast.parse("from __future__ import annotations").body[0], fn],
                    type_ignores=[],
                ),
                str(source),
                "exec",
            ),
            namespace,
        )
        with tempfile.TemporaryDirectory() as folder:
            cfg = NS(
                run_dir=Path(folder),
                target_instance="B-g8-tp2",
                channel_generation=1001,
                gpu_direct_base_port=18030,
                target_tp_size=2,
            )
            with patch.dict(os.environ, {"BRIDGETP_PRESSURE_POOL": "A-g1-tp1"}):
                if fail:
                    with self.assertRaises(RuntimeError):
                        namespace["_preconnect_one"](
                            cfg, None, "source-A-g1-tp1-B-g8-tp2"
                        )
                else:
                    namespace["_preconnect_one"](cfg, None, "source-A-g1-tp1-B-g8-tp2")
            diagnostic = json.loads(
                next(
                    (Path(folder) / "gpu_channel_connection_receipts").glob("*.json")
                ).read_text()
            )
            formal = list(
                (Path(folder) / "gpu_channel_preconnect_receipts").glob("*.json")
            )
        self.assertIsNone(sender.channel_progress)
        return diagnostic, len(formal), namespace

    def test_failure_keeps_exact_stage_identity_without_formal_warmup(self):
        d, n, ns = self.run_sender(True)
        self.assertEqual(
            (d["stage"], d["target_tp_rank"], d["status"]),
            ("sender.comm_init", 1, "FAILED"),
        )
        self.assertEqual((d["channel_generation"], d["base_port"]), (1001, 18030))
        self.assertIn("injected comm failure", d["error"])
        self.assertEqual(n, 0)
        ns["_discard_persistent_gpu_sender"].assert_called_once()
        ns["_release_persistent_gpu_sender"].assert_not_called()

    def test_success_keeps_formal_receipt_separate_from_diagnostic(self):
        d, n, ns = self.run_sender(False)
        self.assertEqual((d["stage"], d["status"]), ("sender.complete", "COMPLETE"))
        self.assertEqual(n, 1)
        ns["_release_persistent_gpu_sender"].assert_called_once()
        ns["_discard_persistent_gpu_sender"].assert_not_called()

    def test_transport_actual_preconnect_and_warmup_emit_stages(self):
        from tests.bridge_tp import test_gpu_direct_history_lifecycle as transport_tests
        from vllm.bridge_tp import gpu_direct_history as transport

        with patch.object(transport, "_channel_progress") as progress:
            transport_tests.TestGpuDirectHistoryLifecycle().test_preconnect_opens_channel_without_starting_session()
            self.assertIn(
                "sender.comm_init", [c.args[1] for c in progress.call_args_list]
            )
        with patch.object(transport, "_channel_progress") as progress:
            transport_tests.TestGpuDirectHistoryLifecycle().test_preconnected_sender_warms_each_rank_without_session()
            stages = [c.args[1] for c in progress.call_args_list]
            self.assertLess(
                stages.index("sender.warmup.fill_sync"),
                stages.index("sender.warmup.header"),
            )
            self.assertLess(
                stages.index("sender.warmup.nccl_send"),
                stages.index("sender.warmup.complete_ack"),
            )


if __name__ == "__main__":
    unittest.main()
