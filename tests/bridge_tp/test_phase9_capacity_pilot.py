# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bridge_tp.run_phase9_controller import step_local
from vllm.bridge_tp.controller.action_adapter import ActionAdapter
from vllm.bridge_tp.controller.anchor_selector import select_source_request_id
from vllm.bridge_tp.controller.capacity_signal import (
    CapacityHeadroomTracker,
    CapacityPilotConfig,
    CapacitySignal,
)
from vllm.bridge_tp.controller.config import ControllerConfig
from vllm.bridge_tp.controller.events import (
    Action,
    MigrationState,
    SourceRequestView,
    TriggerPath,
)
from vllm.bridge_tp.controller.response_proxy import ProxyMode
from vllm.bridge_tp.controller.online_io import ProxyRecorder
from vllm.bridge_tp.controller.state_machine import MigrationStateMachine

ROOT = Path(__file__).resolve().parents[2]
BACKGROUND_SCRIPT = ROOT / "tools" / "bridge_tp" / "run_phase9_capacity_background.py"
BACKGROUND_TEMPLATE = (
    ROOT / "experiments" / "phase9" / "configs" / "cap0_background.template.json"
)
spec = importlib.util.spec_from_file_location("phase9_capacity_background", BACKGROUND_SCRIPT)
background = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(background)


class TestAnchorSelection(unittest.TestCase):
    def test_default_retains_single_request_contract(self) -> None:
        self.assertEqual(select_source_request_id(["only"], ""), "only")
        self.assertIsNone(
            select_source_request_id(["anchor-0", "load-0"], "")
        )

    def test_explicit_prefix_selects_anchor_from_multi_request_batch(self) -> None:
        selected = select_source_request_id(
            ["background-0", "cap0-anchor-17", "background-1"],
            "cap0-anchor",
        )
        self.assertEqual(selected, "cap0-anchor-17")

    def test_ambiguous_prefix_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            select_source_request_id(
                ["cap0-anchor-17", "cap0-anchor-18"],
                "cap0-anchor",
            )


class TestCapacityHeadroomTracker(unittest.TestCase):
    def config(self, **overrides) -> CapacityPilotConfig:
        values = {
            "enabled": True,
            "guard_free_kv_tokens": 100,
            "trigger_time_to_guard_s": 5.0,
            "clear_time_to_guard_s": 10.0,
            "ewma_alpha": 1.0,
            "minimum_samples": 2,
            "minimum_decline_tokens_s": 1.0,
            "maximum_observation_gap_s": 2.0,
        }
        values.update(overrides)
        return CapacityPilotConfig(**values)

    def test_enters_and_clears_with_hysteresis(self) -> None:
        tracker = CapacityHeadroomTracker(self.config())
        first = tracker.update(200, 1.0)
        entered = tracker.update(150, 2.0)
        cleared = tracker.update(160, 3.0)
        self.assertFalse(first.active)
        self.assertEqual(entered.transition, "ENTER")
        self.assertTrue(entered.active)
        self.assertAlmostEqual(entered.time_to_guard_s, 1.0)
        self.assertEqual(cleared.transition, "CLEAR")
        self.assertFalse(cleared.active)
        self.assertIsNone(cleared.to_json()["time_to_guard_s"])

    def test_stale_sample_does_not_manufacture_urgency(self) -> None:
        tracker = CapacityHeadroomTracker(self.config())
        tracker.update(200, 1.0)
        signal = tracker.update(101, 10.0)
        self.assertFalse(signal.active)
        self.assertEqual(signal.transition, "NORMAL")

    def test_enabled_config_requires_measured_guard(self) -> None:
        with self.assertRaisesRegex(ValueError, "guard_free_kv_tokens"):
            CapacityHeadroomTracker(CapacityPilotConfig(enabled=True))


class TestNonDestructiveCancellation(unittest.TestCase):
    def test_adapter_can_drain_staging_without_aborting_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "session_manifest.json").write_text(
                json.dumps(
                    {
                        "migration_id": "m",
                        "session_token": "s",
                        "source_request_id": "anchor-0",
                    }
                ),
                encoding="utf-8",
            )
            adapter = ActionAdapter("http://source", run_dir)
            with patch(
                "vllm.bridge_tp.controller.action_adapter._post",
                return_value={"state": "CANCELLED"},
            ) as post:
                adapter.cancel("pressure cleared", abort_source=False)
            payload = post.call_args.args[1]
            self.assertFalse(payload["abort_source"])
            self.assertEqual(payload["reason"], "pressure cleared")


class TestBackgroundManifest(unittest.TestCase):
    def test_checked_in_template_is_valid(self) -> None:
        manifest = background.load_manifest(BACKGROUND_TEMPLATE)
        self.assertGreaterEqual(len(manifest["jobs"]), 2)
        self.assertEqual(
            {job["pool"] for job in manifest["jobs"]},
            {"source", "target"},
        )

    def test_duplicate_job_ids_fail_closed(self) -> None:
        value = json.loads(BACKGROUND_TEMPLATE.read_text(encoding="utf-8"))
        value["jobs"][1]["job_id"] = value["jobs"][0]["job_id"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                background.load_manifest(path)


class TestCapacityPilotActuation(unittest.TestCase):
    def test_capacity_signal_can_arm_exclusive_shadow(self) -> None:
        class Decision:
            action = Action.STAY
            reason = "performance path says stay"
            trigger_path = None

            @staticmethod
            def to_json() -> dict:
                return {"action": Action.STAY.value, "reason": "stay"}

        class Policy:
            @staticmethod
            def evaluate(*_args, **_kwargs) -> Decision:
                return Decision()

        class Audit:
            def __init__(self) -> None:
                self.records: list[dict] = []

            def write(self, value: dict) -> None:
                self.records.append(value)

        class Adapter:
            def __init__(self) -> None:
                self.armed = None

            def arm_shadow(self, trigger, rate, cutover_output_tokens, note):
                self.armed = (trigger, rate, cutover_output_tokens, note)

        class Rate:
            rate_bytes_s = 1024.0
            rate_gib_s = 0.5

        capacity_config = CapacityPilotConfig(
            enabled=True,
            guard_free_kv_tokens=100,
        )
        config = ControllerConfig(
            handoff_output_tokens=32,
            capacity_pilot=capacity_config,
        )
        signal = CapacitySignal(
            sampled_unix_s=10.0,
            free_kv_tokens=120,
            guard_free_kv_tokens=100,
            decline_rate_tokens_s=10.0,
            time_to_guard_s=2.0,
            active=True,
            transition="ENTER",
            samples=4,
            reason="test",
        )
        request = SourceRequestView(
            request_id="anchor-0",
            prompt_tokens=8,
            output_tokens=40,
            computed_tokens=47,
            pending_tokens=1,
            arrival_unix_s=0.0,
            last_token_unix_s=10.0,
        )
        audit = Audit()
        adapter = Adapter()
        machine = MigrationStateMachine(audit_sink=audit.write)
        record = machine.create("migration", request.request_id)
        step_local(
            Policy(),
            machine,
            adapter,
            audit,
            record,
            request,
            SimpleNamespace(),
            SimpleNamespace(kv_usage_frac=0.2, num_waiting=0),
            0.0,
            Rate(),
            10.0,
            False,
            config,
            ProxyRecorder("anchor", ProxyMode.HOLD_BACK),
            256,
            capacity_signal=signal,
        )
        self.assertEqual(record.state, MigrationState.SHADOW)
        self.assertEqual(record.trigger_path, TriggerPath.CAPACITY_PILOT)
        self.assertEqual(adapter.armed[:3], (41, 0.5, 73))
        self.assertTrue(
            any(item["kind"] == "capacity_pilot_decision" for item in audit.records)
        )


if __name__ == "__main__":
    unittest.main()
