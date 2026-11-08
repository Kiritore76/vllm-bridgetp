# SPDX-License-Identifier: Apache-2.0
"""Simplified main comparison, using the real runner preflight on CPU."""

import tempfile
import unittest
from pathlib import Path

from tests.bridge_tp import test_six_rotation as previous
from tools.bridge_tp import qwen3_phase_shift as phase
from tools.bridge_tp import six_rotation as rotation


class SimpleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = phase.config(
            rotation.read(previous.ASSETS / "inputs.json"), rotation.SIMPLE_SCENARIO
        )

    def test_registered_jobs_and_two_workload_changes(self):
        recipe = rotation.validate_recipe(self.config)
        self.assertEqual(len(recipe["jobs"]), 30731)
        self.assertEqual(recipe["planned_output_tokens"], 4748032)
        self.assertEqual(self.config["horizon_s"], 3420)
        self.assertEqual(
            recipe["phases"],
            [
                [0, 60, 4, 0, "warm"],
                [60, 660, 18, 0.01, "s1_short"],
                [660, 2460, 2, 0.06, "l_long"],
                [2460, 3360, 18, 0, "s2_recovery"],
                [3360, 3420, 1, 0, "tail"],
            ],
        )
        self.assertEqual(
            sum(j["workload_role"] != "short_probe" for j in recipe["jobs"]), 101
        )

    def test_actual_runner_static_L1AC_preflight(self):
        self.layouts = ("L1AC",)
        previous.RotationTests.test_real_runner_builds_and_preflights_static_arms_before_gpu(
            self
        )

    def test_each_registered_scenario_keeps_its_budget_and_rejects_cross_input(self):
        self.assertEqual(
            rotation.registration(rotation.SCENARIO)["max_conversions"], 12
        )
        registered = rotation.registration(rotation.SIMPLE_SCENARIO)
        self.assertEqual(registered["max_conversions"], 8)
        with tempfile.TemporaryDirectory() as folder:
            current = dict(
                controller_argv=[
                    "controller",
                    "--max-seconds",
                    "5400",
                    "--hold-unstarted",
                    "8",
                ],
                scheduler="unified",
                expected_context_limits=(32768, 32768),
                tp2_max_num_seqs=16,
                conversion=dict(max_conversions=8),
                instances=dict(instances=[]),
                run_dir=Path(folder) / "controller",
            )
            recipe = self.config["recipes"][0]
            rotation.configure_plan(current, recipe=recipe)
            with self.assertRaisesRegex(ValueError, "budget/options"):
                rotation.configure_plan(
                    dict(current, conversion=dict(max_conversions=12)), recipe=recipe
                )
            with self.assertRaisesRegex(ValueError, "service contract"):
                rotation.configure_plan(
                    current, recipe=dict(recipe, input_sha256=rotation.INPUT_SHA256)
                )


if __name__ == "__main__":
    unittest.main()
