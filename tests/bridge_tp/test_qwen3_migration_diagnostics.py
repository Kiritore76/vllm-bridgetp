# SPDX-License-Identifier: Apache-2.0
"""Bounded inventory reads, stager identity, monitoring and hard SLO gates."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.bridge_tp import monitor_pressure_host as monitor
from tools.bridge_tp import qwen3_migration_diagnostics as d
from tools.bridge_tp import run_qwen3_online_loaded_validation as runner
from tools.bridge_tp.qwen3_online_validation import latest_inventory


class DiagnosticTests(unittest.TestCase):
    def test_tail_handles_partial_and_large_last_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory"
            path.write_text('{"value":1}\n{"value":')
            self.assertEqual(latest_inventory(path), dict(value=1))
            row = dict(value="x" * 100000)
            path.write_text(json.dumps(row) + "\n{")
            self.assertEqual(latest_inventory(path), row)

    def test_read_volume_does_not_grow_with_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory"
            path.write_bytes(b'{"value":1}\n' * 100000)
            reads = []

            class Reader:
                def __init__(self):
                    self.f = path.open("rb")

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.f.close()

                def seek(self, *args):
                    return self.f.seek(*args)

                def tell(self):
                    return self.f.tell()

                def read(self, n):
                    reads.append(n)
                    return self.f.read(n)

            fake = SimpleNamespace(exists=lambda: True, open=lambda mode: Reader())
            self.assertEqual(latest_inventory(fake), dict(value=1))
            self.assertLessEqual(sum(reads), 65536)

    def test_ready_gate_matches_owned_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            process = SimpleNamespace(process=Mock(pid=42), started_unix_s=10)
            process.process.poll.return_value = None
            (path / "stager_process_ready.json").write_text(
                json.dumps(dict(status="STAGER_PROCESS_READY", pid=42, unix_s=12))
            )
            self.assertEqual(d.wait_stager_ready(path, process)["spawn_to_ready_s"], 2)
            process.process.pid = 43
            with self.assertRaises(ValueError):
                d.wait_stager_ready(path, process)

    def test_exited_stager_is_not_ready(self):
        process = SimpleNamespace(process=Mock(pid=42))
        process.process.poll.return_value = 1
        with self.assertRaises(RuntimeError):
            d.wait_stager_ready(Path("unused"), process)

    def test_diagnostic_child_does_not_repeat_natural_probe(self):
        args = SimpleNamespace(
            model=Path("model"),
            inputs=Path("inputs"),
            output=Path("out"),
            checkpoint=Path("checkpoint"),
            expected_revision="rev",
            diagnostic_probe=True,
        )
        argv = runner.child_arguments(args, runner.stages()[0])
        self.assertIn("--diagnostic-probe", argv)
        self.assertNotIn("--natural-probe", argv)

    def test_quality_gate_detects_slow_stream_even_without_long_gap(self):
        events = [dict(index=i, unix_s=i * 0.12) for i in range(10)]
        self.assertFalse(d.stream_quality(events)["passed"])
        events = [dict(index=i, unix_s=i * 0.02) for i in range(10)]
        self.assertTrue(d.stream_quality(events)["passed"])

    def test_handoff_above_one_second_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = path / "mechanism/controller"
            c.mkdir(parents=True)
            (c / "unified_tokens.json").write_text(
                json.dumps([dict(index=i, unix_s=i * 0.02) for i in range(10)])
            )
            report = d.performance_report(path, dict(visible_handoff_gap_s=1.001))
            self.assertFalse(report["passed"])
            self.assertFalse(report["handoff_pass"])

    def test_optional_gpu_monitor_keeps_process_and_clock_queries(self):
        responses = [
            SimpleNamespace(stdout="0, GPU-id\n", stderr="", returncode=0),
            SimpleNamespace(
                stdout="42, GPU-id, worker, 1000\n", stderr="", returncode=0
            ),
            SimpleNamespace(
                stdout="0, 2000, 10000, P0, 425\n", stderr="", returncode=0
            ),
        ]
        with patch.object(monitor.subprocess, "run", side_effect=responses):
            sample = monitor.gpu_sample(True)
        self.assertEqual(sample["compute_processes"][0][0], "42")
        self.assertIsNone(sample["clock_query_error"])


class PerformanceOrchestrationTests(unittest.TestCase):
    def exercise(self, diagnostic):
        import time

        from tools.bridge_tp.qwen3_online_validation import CHECKPOINT_SHA

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                model=Path("model"),
                inputs=Path("inputs"),
                output=root,
                checkpoint=Path("checkpoint"),
                expected_revision="rev",
                diagnostic_probe=diagnostic,
            )
            for stage in runner.stages(diagnostic):
                c = root / stage["name"] / "mechanism/controller"
                c.mkdir(parents=True)
                header = dict(
                    kind="predictor_header",
                    format_version=1,
                    checkpoint_sha256=CHECKPOINT_SHA,
                    feature_layer="decoder:31",
                    interval=20,
                    category_upper_edges=list(range(61)) + [8192],
                    overflow_category=False,
                    support_scope="natural_eos_within_finite_support",
                )
                event = dict(
                    kind="predictor_prediction",
                    request_id="r",
                    generated_tokens=0,
                    checkpoint_sha256=CHECKPOINT_SHA,
                    captured_unix_ns=time.time_ns(),
                    probabilities=[1.0] + [0.0] * 61,
                )
                (c / "predictor_events.jsonl").write_text(
                    json.dumps(header) + "\n" + json.dumps(event) + "\n"
                )
                (c / "online_predictor_receipt.json").write_text(
                    json.dumps(dict(status="COMPLETED", fresh_advisories=1))
                )
            ledger = []
            with (
                patch.object(runner, "gpu_idle", return_value=True),
                patch.object(runner, "migration_main", return_value=0) as child,
                patch.object(
                    runner,
                    "verify_migration",
                    side_effect=lambda path: dict(
                        predictor_loaded=True,
                        target_background_requests=2 if path.name.endswith("2") else 4,
                    ),
                ),
                patch.object(d, "performance_report", return_value=dict(passed=False)),
                patch.object(
                    monitor, "validate_journal", return_value=dict(status="COMPLETED")
                ),
            ):
                if diagnostic:
                    runner.execute(args, ledger)
                    self.assertEqual(child.call_count, 2)
                    self.assertTrue(
                        all(r["status"] == "PERFORMANCE_FAILURE" for r in ledger)
                    )
                else:
                    with self.assertRaises(runner.PerformanceFailure):
                        runner.execute(args, ledger)
                    self.assertEqual(child.call_count, 1)
                    self.assertEqual(ledger[0]["status"], "PERFORMANCE_FAILURE")

    def test_diagnostic_continues_only_after_safe_mechanism_on_slo_failure(self):
        self.exercise(True)

    def test_normal_validation_stops_on_slo_failure(self):
        self.exercise(False)
