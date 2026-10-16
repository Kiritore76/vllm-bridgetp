# SPDX-License-Identifier: Apache-2.0
"""CPU call-order checks using actual old ready/commit helper source, no CUDA."""

import ast
import importlib.util
import sys
import tempfile
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS

from tools.bridge_tp.pressure_cpu import load
from tools.bridge_tp.pressure_session import RollingPressureSession

ROOT = Path(__file__).resolve().parents[2]
events, machine, m2, io = (
    load(n) for n in ("events", "state_machine", "manager_m2", "online_io")
)
spec = importlib.util.spec_from_file_location(
    "pressure_session_runtime_control", ROOT / "vllm/bridge_tp/runtime_control.py"
)
control = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = control
spec.loader.exec_module(control)


class Adapter:
    def __init__(self, source_url, run_dir, **kwargs):
        self.run_dir, self.ranks, self.commits = run_dir, set(), 0

    def arm_shadow(self, *args, **kwargs):
        self.arm = args

    def set_rate(self, *args, **kwargs):
        pass

    def wait_for_target_ready(self, timeout):
        return len(self.ranks) == 4, self.ranks, "test CPU receipts"

    def commit(self):
        self.commits += 1
        return {"state": "COMMITTED"}

    def ready_notification_evidence(self):
        return {}

    def read_takeover_state(self):
        return self.state


def real_helper(name, context):
    tree = ast.parse((ROOT / "tools/bridge_tp/run_phase9_controller.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "<actual ready helper>",
            "exec",
        ),
        context,
    )
    return context[name]


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.path = Path(self.temp.name)
        self.audit = NS(write=lambda value: None)
        self.recorder = io.ProxyRecorder("external", io.ProxyMode.HOLD_BACK)
        for index in range(40):
            self.recorder.on_source_token(index, index, 1 + index * 0.01)
        self.source = NS(
            payload=dict(max_tokens=1000), future=Future(), recorder=self.recorder
        )
        self.target_future = Future()
        self.settlements = []
        ctx = dict(
            time=time,
            ActionError=RuntimeError,
            ActionAdapter=object,
            AuditLog=object,
            MigrationStateMachine=machine.MigrationStateMachine,
            MigrationRecord=machine.MigrationRecord,
            ProxyRecorder=io.ProxyRecorder,
            MigrationState=events.MigrationState,
            IllegalTransition=machine.IllegalTransition,
        )
        helper = real_helper("step_shadow_only_takeover", ctx)

        def reserve(**kwargs):
            kwargs["record"].candidate_cutover_output_tokens = 500
            return ""

        backend = NS(
            ActionAdapter=Adapter,
            MigrationStateMachine=machine.MigrationStateMachine,
            M2RateController=m2.M2RateController,
            M2RateConfig=m2.M2RateConfig,
            RuntimeControl=control.RuntimeControl,
            MigrationState=events.MigrationState,
            TriggerPath=events.TriggerPath,
            atomic_json_dump=io.atomic_json_dump,
            load_json=io.load_json,
            step_rolling_shadow=reserve,
            _start_target_if_ready=lambda **kwargs: self.target_future,
            step_shadow_only_takeover=helper,
        )
        self.driver = RollingPressureSession(
            run_dir=self.path,
            migration_id="s1",
            config=NS(
                source_url="source",
                target_url="target",
                rate=NS(b_min_bytes_s=1, b_start_bytes_s=2, b_max_bytes_s=3),
            ),
            coordinator=NS(
                finish=lambda *args, **kwargs: self.settlements.append((args, kwargs))
            ),
            ingress=NS(for_server_id=lambda request_id: self.source),
            executor=NS(),
            audit=self.audit,
            backend=backend,
        )
        self.candidate = NS(request_id="request", generated=20, max_output_tokens=1000)
        self.snapshot = load("manager_m0").RuntimeSnapshot(
            time.time(), "s1", "request", "LOCAL", 20
        )
        self.view = NS(request_id="request")
        self.driver.start(self.candidate, self.snapshot)
        io.atomic_json_dump(
            {"cutover_num_output_tokens": 40}, self.path / "cutover_manifest.json"
        )
        io.atomic_json_dump(
            {"cutover_output_tokens": 40}, self.path / "rolling_freeze_selection.json"
        )

    def tearDown(self):
        if self.driver.cleanup_executor is not None:
            self.driver.cleanup_executor.shutdown(wait=True)
        self.temp.cleanup()

    def test_three_ranks_do_not_commit_then_four_ranks_commit_and_set_seam(self):
        self.driver.adapter.ranks = {0, 1, 2}
        self.driver.tick(self.view, self.snapshot)
        self.assertEqual(self.driver.adapter.commits, 0)
        self.assertEqual(self.recorder.proxy.cutover_index, 40)
        self.driver.adapter.ranks = {0, 1, 2, 3}
        self.driver.tick(self.view, self.snapshot)
        self.assertEqual(self.driver.adapter.commits, 1)
        self.assertTrue(self.recorder.proxy.committed)
        self.assertFalse(self.settlements)  # commit alone does not establish release

    def test_release_requires_matching_owner_and_request(self):
        self.driver.adapter.ranks = {0, 1, 2, 3}
        self.driver.tick(self.view, self.snapshot)
        self.driver.adapter.state = dict(
            state="COMMITTED", migration_id="s1", source_request_id="request"
        )
        receipt = dict(
            status="SOURCE_KV_RELEASED",
            request_id="other",
            released_unix_ns=time.time_ns(),
        )
        io.atomic_json_dump(receipt, self.path / "source_kv_release_receipt.json")
        with self.assertRaises(ValueError):
            self.driver.tick(self.view, self.snapshot)
        receipt["request_id"] = "request"
        io.atomic_json_dump(receipt, self.path / "source_kv_release_receipt.json")
        self.driver.tick(self.view, self.snapshot)
        self.assertTrue(self.driver.settled)
        self.assertEqual(len(self.settlements), 1)
        self.driver.tick(self.view, self.snapshot)
        self.assertEqual(len(self.settlements), 1)

    def test_actual_source_stop_cap_cannot_expand(self):
        self.driver.record = None
        self.candidate.max_output_tokens = 1001
        with self.assertRaisesRegex(ValueError, "stop contract"):
            self.driver.start(self.candidate, self.snapshot)

    def test_eos_before_binding_retains_promise_until_owner_shutdown(self):
        self.driver.adapter.refresh_binding = lambda: None
        self.driver.adapter.wait_for_preparing_binding = lambda **kwargs: None
        self.driver.adapter.disarm = lambda **kwargs: None
        self.source.future.set_result({"finish_reason": "stop"})

        self.driver.tick(self.view, self.snapshot)
        self.driver.cancel_future.result(timeout=2)
        self.assertTrue(self.driver.cleanup_requires_shutdown)
        self.assertFalse(self.driver.settled)
        self.assertFalse(self.settlements)


if __name__ == "__main__":
    unittest.main()
