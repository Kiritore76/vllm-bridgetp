# SPDX-License-Identifier: Apache-2.0
"""Retain slow I/O evidence without modifying durable audit or policy behavior."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests.bridge_tp.test_performance_trace import journal
from tools.bridge_tp import launch_pressure_balanced_pilot as launcher
from tools.bridge_tp.monitor_pressure_host import process_samples
from tools.bridge_tp.pressure_cpu import load

d = load("pause_diagnostics")
audit = load("audit")
io = load("online_io")


class PauseDiagnosticsTests(unittest.TestCase):
    def test_disabled_preserves_return_and_exception_without_clock_calls(self):
        with patch.object(d.time, "monotonic", side_effect=AssertionError):
            self.assertEqual(d.call("disabled", lambda: 7), 7)
            with self.assertRaisesRegex(ValueError, "original"):
                d.call(
                    "disabled", lambda: (_ for _ in ()).throw(ValueError("original"))
                )

    def test_real_journal_retains_slow_events_and_aggregated_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.jsonl"
            observer = d.TimingObserver(journal.AsyncJournal(path))
            with d.installed(observer):
                observer.record("audit.fsync", 100, 2.5, 0.001)
                observer.record("audit.fsync", 103, 0.002, 0.001)
                observer.record("controller.tick_interval", 103, 0.25, 0)
                observer.record("controller.tick_interval", 104, 3, 0)
            observer.close()
            receipt = d.verify(path)
            self.assertEqual(receipt["slow_operations"], 2)
            self.assertEqual(receipt["maximum_operation_s"], 3)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            summary = next(r for r in rows if r["kind"] == "timing_summary")
            self.assertEqual(summary["operations"]["audit.fsync"]["count"], 2)
            rows[-1]["dropped"] = 1
            path.write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaises(RuntimeError):
                d.verify(path)

    def test_audit_still_fsyncs_each_record_and_marks_storage_phase(self):
        observer = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            with d.installed(observer), patch.object(audit.os, "fsync") as fsync:
                writer = audit.AuditLog(path)
                writer.write(dict(kind="unchanged", value=7))
                writer.close()
                fsync.assert_called_once()
            self.assertEqual(json.loads(path.read_text())["value"], 7)
            names = [c.args[0] for c in observer.record.call_args_list]
            self.assertIn("audit.fsync", names)
            self.assertIn("audit.lock_and_sync", names)

    def test_atomic_write_preserves_payload_and_reports_failures(self):
        observer = Mock()
        with (
            tempfile.TemporaryDirectory(
                dir=Path(__file__).resolve().parents[2]
            ) as directory,
            d.installed(observer),
        ):
            path = Path(directory) / "value.json"
            io.atomic_json_dump(dict(value=8), path)
            self.assertEqual(json.loads(path.read_text()), dict(value=8))
            with (
                patch.object(io.os, "replace", side_effect=OSError("disk")),
                self.assertRaisesRegex(OSError, "disk"),
            ):
                io.atomic_json_dump(dict(value=9), path)
            self.assertEqual(json.loads(path.read_text()), dict(value=8))
        self.assertIn(
            "atomic_json.replace", [c.args[0] for c in observer.record.call_args_list]
        )

    def test_process_sampling_follows_only_owned_descendants_and_tolerates_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "owned.json"
            registry.write_text(json.dumps([dict(pid=10, name="source")]))
            for pid in (10, 11, 99):
                sub = root / str(pid)
                sub.mkdir()
                (sub / "stat").write_text(str(pid) + " worker")
            children = root / "10/task/10"
            children.mkdir(parents=True)
            (children / "children").write_text("11 12")
            rows = process_samples(registry, root)
            self.assertEqual({r["pid"] for r in rows}, {10, 11})
            self.assertEqual(process_samples(None, root), [])

    def test_diagnostic_scene_owns_monitor_and_sets_only_controller_environment(self):
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[2]
        ) as directory:
            root = Path(directory)
            arrivals = root / "arrivals.json"
            arrivals.write_text(json.dumps([dict(request={})]))
            plan = dict(
                run_dir=root / "scene/controller",
                config_path=root / "scene/config.json",
                config=dict(source_url="source", target_url="target"),
                source_argv=["source"],
                target_argv=["target"],
                source_env={},
                target_env={},
                stager_argv=["stager"],
                controller_argv=["python", "controller"],
            )
            common = Mock()

            def write(path, value):
                path.write_text(json.dumps(value))

            common.write_json.side_effect = write
            created = []

            def start(name, argv, env, log):
                child = SimpleNamespace(
                    name=name, process=Mock(pid=100 + len(created), returncode=0)
                )
                child.process.poll.return_value = 0
                created.append((child, argv, env))
                return child

            common.start_process.side_effect = start
            common.process_group_alive.return_value = False
            fake_runtime = SimpleNamespace(RuntimeControl=Mock())
            fake_warmup = SimpleNamespace(warmup_prompt_shapes=Mock())
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "tools.bridge_tp.run_phase9_cap0_noop": fake_warmup,
                        "tools.bridge_tp.run_phase9_controller": fake_runtime,
                    },
                ),
                patch.object(launcher.controller, "parse_args"),
                patch.object(launcher.controller, "preflight"),
                patch.object(launcher, "scene_result", return_value="done"),
                patch(
                    "tools.bridge_tp.monitor_pressure_host.validate_journal",
                    return_value={},
                ),
                patch.object(d, "verify", return_value={}),
            ):
                result = launcher.run_scene(
                    dict(startup_timeout_s=10, run_timeout_s=10),
                    dict(name="test", arrivals=str(arrivals)),
                    plan,
                    common,
                    None,
                    diagnose_pauses=True,
                )
            self.assertEqual(result, "done")
            self.assertEqual(created[0][0].name, "host monitor")
            self.assertIn("--process-file", created[0][1])
            self.assertIn("BRIDGETP_PRESSURE_DIAGNOSTICS", created[-1][2])
            self.assertNotIn("BRIDGETP_PRESSURE_DIAGNOSTICS", created[1][2])
            self.assertTrue((root / "scene/host_monitor.stop").exists())
            common.stop_processes.assert_called_once()
            registry = json.loads((root / "scene/owned_processes.json").read_text())
            self.assertEqual(len(registry), 4)
            self.assertTrue(all(r["name"] != "host monitor" for r in registry))


if __name__ == "__main__":
    unittest.main()
