# SPDX-License-Identifier: Apache-2.0
"""Idle-only retirement, proof guards and restart configuration isolation."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

from tools.bridge_tp import run_qwen3_online_loaded_validation as runner
from tools.bridge_tp.qwen3_channel_ablation import restart_plan
from vllm.bridge_tp import channel_diagnostics as diag


def lifecycle(state="IDLE"):
    return NS(
        state=NS(value=state), active_session=None if state == "IDLE" else object()
    )


class ChannelTests(unittest.TestCase):
    def test_active_sender_is_refused_without_closing_sockets(self):
        sender = NS(lifecycle=lifecycle("ACTIVE"), connections=[Mock()], comms=[1])
        with self.assertRaises(RuntimeError):
            diag.retire_senders({"key": sender}, set(), threading.Lock())
        sender.connections[0].close.assert_not_called()

    def test_idle_sender_drops_only_explicit_pool(self):
        connection = Mock()
        sender = NS(
            lifecycle=lifecycle(),
            connections=[connection],
            comms=[1, 2],
            streams=[1, 2],
            nccl=object(),
        )

        def abort():
            sender.comms = []
            sender.streams = []
            sender.nccl = None

        sender.abort = Mock(side_effect=abort)
        pool = {"key": sender}
        proof = diag.retire_senders(pool, set(), threading.Lock())
        self.assertEqual(proof["communicators"], 2)
        self.assertEqual(pool, {})
        connection.shutdown.assert_called_once()
        sender.abort.assert_called_once()

    def test_active_target_or_restore_thread_refused(self):
        receiver = NS(lifecycle=lifecycle("ACTIVE"), comm=object())
        c = NS(
            persistent_channel=True,
            _retained_gpu_receivers={},
            _load_threads={},
            _persistent_gpu_receiver=receiver,
        )
        with self.assertRaises(RuntimeError):
            diag.retire_receiver(c)
        c._load_threads = {"r": NS(is_alive=lambda: True)}
        with self.assertRaises(RuntimeError):
            diag.retire_receiver(c)

    def test_idle_target_keeps_connector_and_model_bookkeeping(self):
        receiver = NS(
            lifecycle=lifecycle(), comm=object(), stream=object(), nccl=object()
        )

        def abort():
            receiver.comm = None
            receiver.stream = None
            receiver.nccl = None

        receiver.abort = Mock(side_effect=abort)
        bookkeeping = {"unchanged": True}
        c = NS(
            persistent_channel=True,
            _retained_gpu_receivers={},
            _load_threads={},
            _persistent_gpu_receiver=receiver,
            _prebound_gpu_receiver=receiver,
            _prebind_receiver_stop=threading.Event(),
            _prebind_receiver_thread=None,
            _prebound_gpu_history=None,
            _registered_kv_caches=bookkeeping,
        )
        proof = diag.retire_receiver(c)
        self.assertEqual(proof["status"], "RETIRED")
        self.assertIs(c._registered_kv_caches, bookkeeping)
        self.assertIsNone(c._persistent_gpu_receiver)
        self.assertTrue(c._prebind_receiver_stop.is_set())

    def test_arm_rejects_busy_native_pool_and_wrong_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "channel_retirement_arm.json").write_text(
                json.dumps(
                    dict(
                        migration_id="mechanism",
                        source_api_pid=123,
                        matching_channel_idle_verified=True,
                        native_generations=[1, 1],
                    )
                )
            )
            (d / "takeover_state.json").write_text(
                json.dumps(
                    dict(
                        state="COMMITTED",
                        source_request_id="r",
                        source_abort_dispatched=True,
                    )
                )
            )
            (d / "source_kv_release_receipt.json").write_text(
                json.dumps(dict(status="SOURCE_KV_RELEASED", request_id="r"))
            )
            for role in ["source", "target"]:
                (d / (role + "_inventory.jsonl")).write_text(
                    json.dumps(dict(requests=[], running=0, waiting=0, generation=1))
                    + "\n"
                )
            diag.validate_arm(d, "source", 123)
            with self.assertRaises(ValueError):
                diag.validate_arm(d, "source", 456)
            (d / "target_inventory.jsonl").write_text(
                json.dumps(dict(requests=["busy"], running=1, waiting=0, generation=2))
                + "\n"
            )
            with self.assertRaises(ValueError):
                diag.validate_arm(d, "source", 123)

    def test_restart_keeps_connector_and_service_args_but_uses_fresh_session(self):
        config = dict(
            kv_connector="BridgeTPStreamingConnector", kv_connector_extra_config={}
        )
        plan = dict(
            target_argv=[
                "python",
                "--model",
                "model",
                "--kv-transfer-config",
                json.dumps(config),
                "--max-model-len",
                "16384",
                "--enforce-eager",
            ],
            target_env=dict(
                CUDA_VISIBLE_DEVICES="0,1", BRIDGETP_CHANNEL_DIAGNOSTIC_TOKEN="secret"
            ),
        )
        before = json.dumps(plan)
        argv, env = restart_plan(plan, Path("fresh"))
        self.assertEqual(json.dumps(plan), before)
        self.assertIn("--enforce-eager", argv)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertNotIn("BRIDGETP_CHANNEL_DIAGNOSTIC_TOKEN", env)
        rebuilt = json.loads(argv[argv.index("--kv-transfer-config") + 1])
        self.assertEqual(rebuilt["kv_connector"], config["kv_connector"])
        self.assertFalse(
            rebuilt["kv_connector_extra_config"]["bridgetp_preconnect_gpu_direct"]
        )

    def test_ablation_is_one_bounded_stage(self):
        stages = runner.stages(True, True)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]["background"], 4)
        self.assertFalse(stages[0]["probe"])
        args = NS(
            model=Path("m"),
            inputs=Path("i"),
            output=Path("o"),
            checkpoint=Path("c"),
            expected_revision="r",
            diagnostic_probe=True,
            channel_ablation=True,
        )
        self.assertIn("--channel-ablation", runner.child_arguments(args, stages[0]))


class AblationOrchestrationTests(unittest.TestCase):
    def exercise(self, old_group_alive=False):
        from unittest.mock import patch

        from tools.bridge_tp import (
            pressure_session,
            run_atomic_takeover,
            run_pressure_calibration,
        )
        from tools.bridge_tp import qwen3_channel_ablation as ablation

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "controller"
            directory.mkdir()
            (directory / "takeover_state.json").write_text(
                json.dumps(dict(source_request_id="r"))
            )
            old = NS(process=NS(pid=100), name="target")
            source = NS(process=NS(pid=200), name="source")
            new = NS(process=NS(pid=456), name="target restarted")
            processes = [old, source]
            plan = dict(
                run_dir=directory,
                target_argv=[
                    "p",
                    "--kv-transfer-config",
                    json.dumps(
                        dict(
                            kv_connector="BridgeTPStreamingConnector",
                            kv_connector_extra_config={},
                        )
                    ),
                ],
                source_env=dict(BRIDGETP_CHANNEL_DIAGNOSTIC_TOKEN="s"),
                target_env=dict(BRIDGETP_CHANNEL_DIAGNOSTIC_TOKEN="t"),
            )
            common = Mock()
            common.process_group_alive.return_value = old_group_alive
            common.start_process.return_value = new

            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            common.write_json.side_effect = write
            with (
                patch.object(
                    run_pressure_calibration,
                    "wait_empty",
                    return_value=[dict(generation=1)] * 2,
                ),
                patch.object(pressure_session, "channel_idle", return_value=True),
                patch.object(
                    run_atomic_takeover,
                    "_post_json",
                    return_value=dict(status="RETIRED"),
                ),
                patch.object(ablation, "control_batch") as control,
                patch.object(
                    ablation,
                    "wait_target_empty",
                    side_effect=[
                        dict(free_blocks=11712, generation=1),
                        dict(free_blocks=11712, generation=2),
                    ],
                ),
            ):
                args = NS(out_root=Path(tmp))
                config = NS(source_url="s", target_url="t")
                if old_group_alive:
                    with self.assertRaises(RuntimeError):
                        ablation.run_ablation(
                            args, plan, processes, {}, Mock(), Mock(), common, config
                        )
                    common.start_process.assert_not_called()
                else:
                    # Warmup request still uses the real strict request builder.
                    inputs = dict(
                        requests={
                            "2048": dict(
                                model="m", prompt=[1] * 2048, temperature=0.0, top_p=1.0
                            )
                        }
                    )
                    ablation.run_ablation(
                        args, plan, processes, inputs, Mock(), Mock(), common, config
                    )
                    self.assertEqual(
                        [call.args[-1] for call in control.call_args_list],
                        ["retired", "restarted"],
                    )
                    proof = json.loads(
                        (directory / "channel_ablation_receipt.json").read_text()
                    )
                    self.assertEqual(proof["retained_and_retired_target_api_pid"], 100)
                    self.assertEqual(proof["restarted_target_api_pid"], 456)

    def test_retirement_then_restart_control_order(self):
        self.exercise()

    def test_live_old_target_blocks_restart(self):
        self.exercise(True)


class RetirementEndpointTests(unittest.TestCase):
    def endpoint(self):
        import ast
        import asyncio
        import hmac
        import os
        import typing

        source = Path(__file__).resolve().parents[2] / "vllm/bridge_tp/takeover_api.py"
        node = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "retire_diagnostic_channel"
        )
        node.decorator_list = []

        class HTTPError(Exception):
            def __init__(self, status_code, detail):
                self.status_code = status_code

        namespace = dict(
            Request=object,
            Any=typing.Any,
            os=os,
            hmac=hmac,
            HTTPException=HTTPError,
            Path=Path,
            _takeover_lock=asyncio.Lock(),
            session_target_tp=lambda p: 2,
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                str(source),
                "exec",
            ),
            namespace,
        )
        return namespace["retire_diagnostic_channel"], HTTPError

    def test_disabled_and_bad_token_never_execute_worker_rpc(self):
        import os
        from unittest.mock import AsyncMock, patch

        handler, error = self.endpoint()
        rpc = AsyncMock()
        request = NS(
            json=AsyncMock(return_value=dict(token="wrong")),
            app=NS(state=NS(engine_client=NS(collective_rpc=rpc))),
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(error) as caught:
                handler(request).send(None)
            self.assertEqual(caught.exception.status_code, 404)
        with patch.dict(
            os.environ,
            dict(
                BRIDGETP_CHANNEL_DIAGNOSTIC_TOKEN="right",
                BRIDGETP_CHANNEL_DIAGNOSTIC_ROLE="target",
                BRIDGETP_CHANNEL_DIAGNOSTIC_DIR="unused",
            ),
        ):
            with self.assertRaises(error) as caught:
                handler(request).send(None)
            self.assertEqual(caught.exception.status_code, 403)
        rpc.assert_not_called()

    def test_busy_arm_blocks_worker_rpc(self):
        import os
        from unittest.mock import AsyncMock, patch

        handler, error = self.endpoint()
        rpc = AsyncMock()
        request = NS(
            json=AsyncMock(return_value=dict(token="right")),
            app=NS(state=NS(engine_client=NS(collective_rpc=rpc))),
        )
        with (
            patch.dict(
                os.environ,
                dict(
                    BRIDGETP_CHANNEL_DIAGNOSTIC_TOKEN="right",
                    BRIDGETP_CHANNEL_DIAGNOSTIC_ROLE="target",
                    BRIDGETP_CHANNEL_DIAGNOSTIC_DIR="unused",
                ),
            ),
            patch.object(diag, "validate_arm", side_effect=ValueError("busy")),
        ):
            with self.assertRaises(error) as caught:
                handler(request).send(None)
            self.assertEqual(caught.exception.status_code, 409)
        rpc.assert_not_called()
