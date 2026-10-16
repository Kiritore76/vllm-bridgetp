# SPDX-License-Identifier: Apache-2.0
"""Frozen matched service-screen inputs and fail-closed busy-target acceptance."""

import copy
import unittest

from tests.bridge_tp import test_pressure_kv_pilot as previous
from tools.bridge_tp import pressure_health_screen as screen
from tools.bridge_tp import run_pressure_comparison as compare


class ScreenInputs(previous.KvInputTests):
    def setUp(self):
        super().setUp()
        self.config = compare.pilot.read(previous.BUNDLE / "health_screen.json")

    def test_frozen_inputs_preserve_measured_shapes_and_sampling(self):
        screen.validate(self.config, self.inputs)
        for recipe in self.config["recipes"]:
            jobs = compare.arrivals(self.inputs, recipe)
            self.assertEqual(len(jobs), 34 if recipe["kind"] == "spare" else 57)
            self.assertTrue(all(j["request"]["ignore_eos"] for j in jobs))
            self.assertEqual(max(j["at_s"] for j in jobs), 78)

    def test_only_sustain_can_be_changed(self):
        for key, value in (("sustain_s", 0), ("health_target_load", 1)):
            config = copy.deepcopy(self.config)
            config["policy_overrides"][key] = value
            with self.assertRaises(ValueError):
                screen.validate(config, self.inputs)

    def test_runtime_capacity_must_allow_one_offload_to_relieve_pressure(self):
        receipts = []
        from types import SimpleNamespace
        from unittest.mock import patch

        for recipe in self.config["recipes"]:
            with patch.object(
                compare.calibration,
                "wait_empty",
                return_value=[
                    dict(block_size=16, free_blocks=1962),
                    dict(block_size=16, free_blocks=35738),
                ],
            ):
                compare.check_health_shape(
                    dict(run_dir=previous.BUNDLE / "fake/controller"),
                    compare.arrivals(self.inputs, recipe),
                    SimpleNamespace(
                        write_json=lambda path, value: receipts.append(value)
                    ),
                )
        self.assertTrue(all(r["valid"] for r in receipts))
        self.assertEqual(receipts[0]["source_required_blocks"], 1405)


class ScreenAcceptance(unittest.TestCase):
    def test_busy_target_start_is_failure_and_unknown_is_gap(self):
        self.assertEqual(
            screen.assess([], [], "busy", "migrate")["coverage"], "COVERAGE_GAP"
        )
        with self.assertRaises(RuntimeError):
            screen.assess([dict(kind="pressure_actual_start")], [], "busy", "migrate")

    def test_spare_requires_service_not_KV_path(self):
        audit = [
            dict(kind="pressure_actual_start"),
            dict(kind="pressure_source_release_confirmed"),
            dict(
                execution="RESERVED",
                decision=dict(service_path="legacy", capacity_path=True),
            ),
        ]
        metrics = [dict(migrated=True)]
        self.assertEqual(
            screen.assess(audit, metrics, "spare", "migrate")["coverage"],
            "COVERAGE_GAP",
        )
        audit[-1]["decision"].update(service_path="health", capacity_path=False)
        audit[-1]["decision"].update(
            request_id="a",
            candidates=[
                dict(
                    request=dict(request_id="a", max_output_tokens=4096, generated=1000)
                )
            ],
        )
        self.assertEqual(
            screen.assess(audit, metrics, "spare", "migrate")["coverage"],
            "SERVICE_MOVE_OBSERVED",
        )


if __name__ == "__main__":
    unittest.main()
