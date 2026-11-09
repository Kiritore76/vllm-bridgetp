# SPDX-License-Identifier: Apache-2.0
"""Registered lower long pressure and automatic return comparison (CPU)."""

import unittest

from tests.bridge_tp import test_six_rotation as previous
from tools.bridge_tp import qwen3_phase_shift as phase
from tools.bridge_tp import six_rotation as rotation


class ReturnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = phase.config(
            rotation.read(previous.ASSETS / "inputs.json"), rotation.RETURN_SCENARIO
        )

    def test_frozen_input_and_return_window(self):
        recipe = rotation.validate_recipe(self.config)
        self.assertEqual(len(recipe["jobs"]), 11049)
        self.assertEqual(recipe["planned_output_tokens"], 1704576)
        self.assertEqual(self.config["horizon_s"], 1860)
        self.assertEqual(
            sum(j["workload_role"] == "long_input_output" for j in recipe["jobs"]), 36
        )
        self.assertEqual(
            recipe["phases"],
            [
                [0, 60, 4, 0, "warm"],
                [60, 240, 12, 0, "s1_short"],
                [240, 1080, 2, 0.04, "l_long"],
                [1080, 1320, 2, 0, "l_retreat"],
                [1320, 1560, 17, 0, "s2_return"],
                [1560, 1800, 10, 0, "s3_recovery"],
                [1800, 1860, 1, 0, "tail"],
            ],
        )

    def test_real_runner_auto_all_assist_unlimited_preflight(self):
        self.automatic = self.assist = self.dynamic_all = self.unlimited = True
        self.layouts = ("L1AC",)
        self.attempts = 0
        previous.RotationTests.test_real_runner_builds_and_preflights_static_arms_before_gpu(
            self
        )

    def test_round_trip_needs_completed_merge_then_split(self):
        def step(mode, state="COMPLETED", bundle="A"):
            return dict(
                plan=dict(
                    state=state,
                    bundle_id=bundle,
                    target_bundle_mode=mode,
                )
            )

        for steps in (
            [],
            [step("TP2")],
            [step("TP1"), step("TP2")],
            [step("TP2", "ABORTED"), step("TP1")],
            [step("TP2"), step("TP1", "ABORTED")],
            [step("TP2", bundle="C"), step("TP1")],
        ):
            with self.subTest(steps=steps):
                self.assertFalse(rotation.return_coverage(steps))
        self.assertTrue(rotation.return_coverage([step("TP2"), step("TP1")]))

    def test_registration_does_not_replace_old_inputs(self):
        self.assertEqual(
            rotation.registration(rotation.SIMPLE_SCENARIO)["requests"], 30731
        )
        recipe = self.config["recipes"][0]
        wrong = dict(recipe, input_sha256=rotation.INPUT_SHA256)
        with self.assertRaisesRegex(ValueError, "input/phase contract"):
            rotation.validate_recipe(dict(self.config, recipes=[wrong]))


if __name__ == "__main__":
    unittest.main()
