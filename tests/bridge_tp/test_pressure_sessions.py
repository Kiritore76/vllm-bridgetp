# SPDX-License-Identifier: Apache-2.0
"""Repeated migration safety without CUDA or a live server."""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from tests.bridge_tp.test_pressure_session import control
from tools.bridge_tp.pressure_progress import completed_streams
from tools.bridge_tp.pressure_sessions import (
    PressureSessions,
    follow_stager,
    publish_active,
)

ROOT = Path(__file__).resolve().parents[2]


def actual_function(path, name, context):
    tree = ast.parse((ROOT / path).read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), context)
    return context[name]


class RepeatedSessions(unittest.TestCase):
    def test_direct_script_bootstrap_without_pythonpath_or_repo_cwd(self):
        path = ROOT / "tools/bridge_tp/pressure_sessions.py"
        code = (
            "import ast; from pathlib import Path; "
            f"p=Path({str(path)!r}); "
            "t=ast.parse(p.read_text(encoding='utf-8')); "
            "assert isinstance(t.body[-1],ast.If); "
            "exec(compile(ast.Module(body=t.body[:-1],type_ignores=[]),str(p),'exec'),"
            "{'__name__':'__main__','__file__':str(p)})"
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        self.root = Path(self.temp.name)
        (self.root / "sessions" / "migration-0001").mkdir(parents=True)
        self.entry = NS(state="COMMITTED", service_observed=True)
        self.co = NS(
            ledger=NS(entries={"run-0001": self.entry}), policy=NS(note_started=Mock())
        )
        self.audit = NS(write=Mock())

        def factory(**kwargs):
            return NS(
                **kwargs,
                record=None,
                stream=None,
                target_future=None,
                settled=False,
                cleanup_requires_shutdown=False,
                old=NS(RuntimeControl=control.RuntimeControl),
                adapter=NS(close=Mock()),
            )

        with patch("tools.bridge_tp.pressure_session.RollingPressureSession", factory):
            self.sequence = PressureSessions(
                run_dir=self.root,
                migration_id="run",
                maximum_migrations=0,
                coordinator=self.co,
                audit=self.audit,
            )
        s = self.sequence.sessions[0]
        s.record = NS(request_id="source-a")
        source = Future()
        source.set_result({})
        s.stream = NS(future=source)
        s.target_future = Future()  # Deliberately still decoding on TP4.
        s.settled = True
        self.write(
            "gpu_direct_sender.json",
            dict(
                communicator_lifecycle="PERSISTENT_CHANNEL_IDLE",
                migration_id="run-0001",
                source_request_id="source-a",
                channel_create_count=1,
                channel_destroy_count=0,
                channel_session_count=1,
            ),
        )
        for rank in range(4):
            self.write(
                f"persistent_channel_receipts/tp_rank_{rank}.json",
                dict(
                    status="PERSISTENT_CHANNEL_IDLE",
                    migration_id="run-0001",
                    session_request_id="source-a",
                    tp_rank=rank,
                    channel_create_count=1,
                    channel_destroy_count=0,
                    channel_session_count=1,
                ),
            )
        self.write("rolling_target_finalized.json", dict(migration_id="run-0001"))
        self.write("pressure_stager_receipt.json", dict(returncode=0))

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        path = self.sequence.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_next_session_while_previous_target_decodes_preserves_completion(self):
        old = self.sequence.sessions[0]
        with patch("tools.bridge_tp.pressure_sessions.publish_active") as publish:
            self.assertTrue(self.sequence.rotate())
        self.assertEqual(self.sequence.migration_id, "run-0002")
        self.assertTrue(self.sequence.ready_for_start)
        publish.assert_called_once_with(self.root, self.sequence.run_dir)
        self.assertTrue((old.run_dir / "gpu_direct_sender.json").exists())
        self.assertEqual(completed_streams({"a": old.stream}, self.sequence), 0)
        old.target_future.set_result({})
        self.assertEqual(completed_streams({"a": old.stream}, self.sequence), 1)
        self.co.policy.note_started.assert_called_once()
        self.sequence.close()
        self.assertTrue(all(s.adapter.close.called for s in self.sequence.sessions))

    def test_missing_rank_or_native_confirmation_or_shutdown_blocks_reuse(self):
        self.entry.service_observed = False
        self.assertFalse(self.sequence.rotate())
        self.entry.service_observed = True
        rank = self.sequence.run_dir / "persistent_channel_receipts/tp_rank_3.json"
        rank.unlink()
        self.assertFalse(self.sequence.rotate())
        self.sequence.sessions[0].cleanup_requires_shutdown = True
        self.assertFalse(self.sequence.rotate())

    def test_wrong_identity_failed_stager_or_busy_channel_is_fatal(self):
        for name, row in (
            (
                "persistent_channel_receipts/tp_rank_3.json",
                dict(status="PERSISTENT_CHANNEL_IDLE", migration_id="wrong"),
            ),
            ("pressure_stager_receipt.json", dict(returncode=1)),
            ("gpu_direct_sender.json", dict(communicator_lifecycle="ACTIVE")),
        ):
            path = self.sequence.run_dir / name
            original = path.read_text()
            self.write(name, row)
            with self.assertRaisesRegex(ValueError, "receipts disagree"):
                self.sequence.rotate()
            path.write_text(original)

    def test_bounded_mode_honors_limit_and_unsettled_session_never_rotates(self):
        self.sequence.maximum = 1
        self.assertFalse(self.sequence.rotate())
        self.sequence.maximum = 0
        self.sequence.sessions[0].settled = False
        self.assertFalse(self.sequence.rotate())

    def test_active_link_cannot_escape_owned_tree(self):
        with self.assertRaisesRegex(ValueError, "owned sessions"):
            publish_active(self.root, self.root.parent)

    def test_repeated_acceptance_requires_actual_multiple_commits(self):
        from tools.bridge_tp.pressure_health_screen import assess_repeated

        def audit(count):
            return [
                row
                for index in range(count)
                for row in (
                    dict(kind="pressure_actual_start", migration_id=f"m{index}"),
                    dict(
                        kind="pressure_source_release_confirmed",
                        migration_id=f"m{index}",
                    ),
                    dict(
                        execution="RESERVED",
                        decision=dict(
                            capacity_path=False,
                            request_id=f"request{index}",
                            candidates=[],
                        ),
                    ),
                )
            ]

        self.assertEqual(
            assess_repeated(audit(1), [dict(migrated=True)], "spare", "migrate")[
                "coverage"
            ],
            "COVERAGE_GAP",
        )
        self.assertEqual(
            assess_repeated(audit(2), [dict(migrated=True)] * 2, "spare", "migrate")[
                "coverage"
            ],
            "REPEATED_SCENARIO_OBSERVED",
        )
        with self.assertRaisesRegex(RuntimeError, "busy target"):
            assess_repeated(audit(1), [dict(migrated=True)], "busy", "migrate")

    def test_stager_reuses_runner_with_distinct_immutable_paths(self):
        first = self.sequence.run_dir
        second = first.with_name("migration-0002")
        second.mkdir()
        active = self.root / "migration_active"
        run = Mock()
        with (
            patch.object(Path, "resolve", side_effect=[first, second]),
            patch(
                "tools.bridge_tp.pressure_sessions.time.sleep",
                side_effect=[None, InterruptedError("test stop")],
            ),
            self.assertRaises(InterruptedError),
        ):
            follow_stager(["--run-dir", str(active)], run=run)
        self.assertEqual(run.call_args_list[0].args[0][-1], str(first))
        self.assertEqual(run.call_args_list[1].args[0][-1], str(second))
        for directory in (first, second):
            receipt = json.loads(
                (directory / "pressure_stager_receipt.json").read_text()
            )
            self.assertEqual(receipt["returncode"], 0)

    def test_stager_error_is_preserved_and_stops_reuse(self):
        directory = self.sequence.run_dir
        with (
            patch.object(Path, "resolve", return_value=directory),
            self.assertRaisesRegex(RuntimeError, "failed transfer"),
        ):
            follow_stager(
                ["--run-dir", str(self.root / "migration_active")],
                run=Mock(side_effect=RuntimeError("failed transfer")),
            )
        row = json.loads((directory / "pressure_stager_receipt.json").read_text())
        self.assertEqual(row["returncode"], 1)
        self.assertIn("failed transfer", row["error"])

    def test_target_api_has_persistent_identity_without_source_publisher_env(self):
        directory = self.sequence.run_dir
        self.write("session_manifest.json", dict(migration_id="run-0001"))
        configured = actual_function(
            "vllm/bridge_tp/takeover_api.py",
            "_configured_session",
            dict(
                os=os,
                Path=Path,
                HTTPException=RuntimeError,
                HTTPStatus=NS(SERVICE_UNAVAILABLE=503),
                _load_json=lambda p: json.loads(p.read_text()),
            ),
        )
        with patch.dict(
            os.environ,
            dict(
                BRIDGETP_TAKEOVER_RUN_DIR=str(directory),
                BRIDGETP_TAKEOVER_MIGRATION_ID="obsolete",
                BRIDGETP_TAKEOVER_PERSISTENT_CHANNEL="1",
                BRIDGETP_PERSISTENT_CHANNEL="",
            ),
        ):
            self.assertEqual(configured(), (directory.resolve(), "run-0001"))

    def test_actual_launch_plan_separates_observation_and_session_directories(self):
        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        bundle = ROOT / "experiments/phase9/pressure_balanced_v2"
        spec = launch.read(bundle / "pilot_spec.json")
        spec.update(
            maximum_migrations=0,
            policy_config=str(bundle / "health_service_policy.json"),
            health_reference=str(bundle / "health_reference_v2.json"),
        )
        jobs = [
            dict(
                id="a",
                at_s=0,
                route="tp1",
                request=dict(prompt=[1] * 2048, max_tokens=4096),
            )
        ]
        arrivals = self.root / "arrivals.json"
        arrivals.write_text(json.dumps(jobs))
        connector = actual_function(
            "tools/bridge_tp/run_phase9_cap0_calibration.py",
            "target_connector",
            dict(json=json, Path=Path),
        )
        common = NS(
            source_environment=lambda args, name, directory: {
                "BRIDGETP_STREAM_RUN_DIR": str(directory)
            },
            server_command=lambda *args: ["python", "-m", "server"],
            target_connector=connector,
        )
        settings = NS(
            model_path=Path("qwen"),
            tp1_gpu="0",
            tp1_port=8000,
            tp4_port=8001,
            tp4_gpus="1,2,3,4",
            ready_notification_port=30501,
            channel_generation=1,
            gpu_direct_base_port=30400,
            delta_port=30300,
            delivery_port=30200,
            max_model_len=8192,
            tp4_max_model_len=16384,
        )
        plan = launch.build_scene(
            spec,
            dict(name="matrix", arrivals=str(arrivals)),
            self.root,
            common,
            settings,
        )
        active = plan["run_dir"] / "migration_active"
        self.assertEqual(plan["source_env"]["BRIDGETP_STREAM_RUN_DIR"], str(active))
        self.assertEqual(plan["target_env"]["BRIDGETP_TAKEOVER_RUN_DIR"], str(active))
        self.assertNotIn("BRIDGETP_PERSISTENT_CHANNEL", plan["target_env"])
        self.assertIn("pressure_sessions.py", plan["stager_argv"][1])
        extra = json.loads(plan["target_argv"][-1])["kv_connector_extra_config"]
        self.assertFalse(extra["bridgetp_release_persistent_payload_buffer"])
        self.assertEqual(
            extra["bridgetp_stream_manifest"], str(active / "session_manifest.json")
        )
        plan["config_path"].parent.mkdir(parents=True)
        plan["config_path"].write_text(json.dumps(plan["config"]))
        with self.assertRaises(FileNotFoundError):
            launch.controller.preflight(
                launch.controller.parse_args(plan["controller_argv"][2:])
            )
        launch.write_instances(plan)
        args = launch.controller.parse_args(plan["controller_argv"][2:])
        receipt = launch.controller.preflight(args)[-1]
        self.assertEqual(receipt["maximum_migrations"], 0)


if __name__ == "__main__":
    unittest.main()
