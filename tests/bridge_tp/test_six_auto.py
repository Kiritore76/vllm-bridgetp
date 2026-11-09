# SPDX-License-Identifier: Apache-2.0
"""Automatic sequence terminal evidence and the registered real preflight."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.bridge_tp import test_six_rotation as previous
from tools.bridge_tp import qwen3_phase_shift as phase
from tools.bridge_tp import six_conversion as six
from tools.bridge_tp import six_rotation as rotation


def scene_files(scene, steps):
    row = dict(
        job_id="one",
        server_id="server-one",
        initial_pool="A-g1-tp1",
        source_done=True,
        source_error=None,
        request=dict(prompt=[], max_tokens=1),
        response=dict(emitted=[dict(unix_s=0.3)]),
        dispatch_timing=dict(planned_unix_s=0.1, http_thread_started_unix_s=0.1),
    )
    sequence = dict(state="COMPLETED", error=None, steps=steps)
    for name, value in [
        ("conversion_result.json", sequence),
        ("conversion_sequence_result.json", sequence),
        ("closed_loop_acceptance.json", dict(sessions=[])),
        ("instances.json", {}),
        ("controller/pressure_request_results.json", [row]),
    ]:
        rotation.write(scene / name, value)
    (scene / "conversion_events.jsonl").write_text("")
    (scene / "controller/pressure_audit.jsonl").write_text("")


class AutoSequenceTests(unittest.TestCase):
    def test_no_conversion_is_valid_but_not_migration_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            scene = Path(folder)
            scene_files(scene, [])
            value = six.assess(scene, 1, require_conversion=False)
            self.assertTrue(value["passed"])
            self.assertFalse(value["actual_migration_coverage"])
            self.assertFalse(six.assess(scene, 1, require_conversion=True)["passed"])

    def test_safe_abort_requires_ack_and_valid_six_card_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            scene = Path(folder)
            step = dict(
                plan_id="safe",
                plan=dict(state="ABORTED"),
                layout=six.PATH["L1AC"],
                triggered_unix_s=0,
                completed_unix_s=0.2,
                layout_revision=1,
                controller_acknowledged=True,
            )
            scene_files(scene, [step])
            value = six.assess(scene, 1, require_conversion=False)
            self.assertTrue(value["passed"])
            self.assertFalse(value["all_steps_completed"])
            self.assertFalse(six.assess(scene, 1, require_conversion=True)["passed"])
            for change in (
                dict(controller_acknowledged=False),
                dict(layout={"A": 2}),
                dict(plan=dict(state="RECOVERY_REQUIRED")),
            ):
                scene_files(scene, [dict(step, **change)])
                self.assertFalse(
                    six.assess(scene, 1, require_conversion=False)["passed"]
                )


class AutoPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = phase.config(
            rotation.read(previous.ASSETS / "inputs.json"), rotation.SIMPLE_SCENARIO
        )

    def test_real_runner_auto_profile_uses_same_input_and_eight_attempts(self):
        self.automatic = True
        self.layouts = ("L1AC",)
        previous.RotationTests.test_real_runner_builds_and_preflights_static_arms_before_gpu(
            self
        )

    def test_real_runner_unlimited_assisted_comparison_preflight(self):
        self.automatic = self.assist = self.dynamic_all = self.unlimited = True
        self.layouts = ("L1AC",)
        self.attempts = 0
        previous.RotationTests.test_real_runner_builds_and_preflights_static_arms_before_gpu(
            self
        )


class AutoEvidenceTests(unittest.TestCase):
    def test_current_three_receivers_and_budget_limit_are_explicit(self):
        registered = dict(
            input_sha256="fixture", requests=1, output_tokens=1, horizon_s=3420
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "trial"
            scene = root / "scene"
            expected = {
                name: dict(instance_id=name, tp_degree=2, lifecycle="ACTIVE")
                for name in ["A-new", "B-kept", "C-new"]
            }
            payloads = {
                "result.json": dict(
                    status="COMPLETED_CONVERSION_SEQUENCE",
                    runs=[dict(status="COMPLETED", input_sha256="fixture")],
                ),
                "comparison_recipe.json": {},
                "scene/owned_shutdown_receipt.json": dict(all_owned_groups_ended=True),
                "scene/rotation_contract.json": dict(
                    input_sha256="fixture",
                    automatic_conversion=True,
                    instances=[dict(instance_id="old")],
                ),
                "scene/comparison_metrics.json": dict(
                    summary=dict(
                        requests=1,
                        output_tokens=1,
                        window_s=3420,
                        fixed_arrivals_valid=True,
                    )
                ),
                "scene/conversion_sequence_acceptance.json": dict(
                    passed=True, actual_migration_coverage=False
                ),
                "scene/native_empty_receipt.json": {
                    k: dict(running=0, waiting=0, requests=[]) for k in expected
                },
                "scene/controller/layout_control.json": dict(
                    instances=list(expected.values())
                ),
                "scene/controller/layout_status.json": dict(instances=expected),
                "scene/controller/pressure_result_export_receipt.json": dict(
                    status="WRITTEN", requests=1
                ),
                "scene/conversion_sequence_result.json": dict(steps=[]),
            }
            for name, value in payloads.items():
                rotation.write(root / name, value)
            for name in expected:
                rotation.write(
                    scene / (name + "_trace_finalization.json"),
                    dict(status="FINALIZED"),
                )
                rotation.write(
                    scene / (name + "_trace_acceptance.json"), dict(samples=1, ranks=2)
                )
            (scene / "conversion_events.jsonl").write_text(
                json.dumps(dict(kind="DECIDER_LIMIT", limit=8))
            )
            (scene / "controller/pressure_timing.jsonl").write_text(
                "\n".join(
                    map(
                        json.dumps,
                        [
                            dict(
                                kind="timing_header", schema="PRESSURE_PAUSE_TIMING_V1"
                            ),
                            dict(kind="timing_summary"),
                            dict(
                                kind="evidence_footer",
                                dropped=0,
                                writer_error=None,
                                saved_records=2,
                            ),
                        ],
                    )
                )
            )
            with (
                patch.object(rotation, "registration", return_value=registered),
                patch.object(
                    rotation,
                    "validate_recipe",
                    return_value=dict(input_sha256="fixture"),
                ),
            ):
                value = rotation.assess(root, automatic=True)
                self.assertTrue(value["passed"], value)
                self.assertTrue(value["automatic_coverage"]["budget_limit_reached"])
                self.assertFalse(value["automatic_coverage"]["conversion_observed"])
                self.assertFalse(value["performance_validated"])
                value = rotation.assess(root, rotation.RETURN_SCENARIO, automatic=True)
                self.assertFalse(value["passed"])
                self.assertTrue(value["mechanism_complete"])
                self.assertFalse(value["automatic_coverage"]["a_round_trip_completed"])
                rotation.write(
                    scene / "conversion_sequence_result.json",
                    dict(
                        steps=[
                            dict(
                                plan=dict(
                                    state="COMPLETED",
                                    bundle_id="A",
                                    target_bundle_mode=mode,
                                )
                            )
                            for mode in ("TP2", "TP1")
                        ]
                    ),
                )
                self.assertTrue(
                    rotation.assess(
                        root,
                        rotation.RETURN_SCENARIO,
                        automatic=True,
                    )["passed"]
                )
                (scene / "A-new_trace_finalization.json").unlink()
                self.assertFalse(rotation.assess(root, automatic=True)["passed"])


if __name__ == "__main__":
    unittest.main()
