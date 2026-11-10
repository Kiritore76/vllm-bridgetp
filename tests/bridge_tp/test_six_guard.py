# SPDX-License-Identifier: Apache-2.0
"""Frozen sustained/pulse workload and honest live-observer coverage."""

import copy
import unittest

from tests.bridge_tp import test_six_rotation as previous
from tools.bridge_tp import qwen3_phase_shift as phase
from tools.bridge_tp import six_rotation as rotation


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = phase.config(
            rotation.read(previous.ASSETS / "inputs.json"), rotation.GUARD_SCENARIO
        )

    def test_frozen_input_and_equal_peak_distinct_duration(self):
        recipe = rotation.validate_recipe(self.config)
        self.assertEqual(len(recipe["jobs"]), 16385)
        self.assertEqual(recipe["planned_output_tokens"], 2589184)
        self.assertEqual(
            sum(j["workload_role"] == "long_input_output" for j in recipe["jobs"]),
            61,
        )
        pulse, sustained = recipe["phases"][4], recipe["phases"][6]
        self.assertEqual(pulse, [1500, 1510, 17.5, 0, "short_pulse"])
        self.assertEqual(sustained, [1620, 2220, 17.5, 0, "s2_return"])
        self.assertEqual(recipe["phases"][-1][1], 2400)
        self.assertEqual(recipe["arrival_seed"], 6)
        plan = rotation.plan(
            rotation.read(previous.ASSETS / "inputs.json"), rotation.GUARD_SCENARIO
        )
        self.assertEqual(plan["schema"], "QWEN3_SIX_GUARD_V1")
        self.assertEqual(
            rotation.registration(rotation.RETURN_SCENARIO)["requests"], 11049
        )

    def test_real_runner_auto_all_assist_unlimited_preflight(self):
        self.automatic = self.assist = self.dynamic_all = self.unlimited = True
        self.layouts = ("L1AC",)
        self.attempts = 0
        previous.RotationTests.test_real_runner_builds_and_preflights_static_arms_before_gpu(
            self
        )

    def records(self):
        return [
            dict(
                evaluated_unix_s=10000 + t,
                signals=dict(layout="L2C"),
                policy="SIX_GAP_RULES_V1",
                decision="KEEP",
                proposal=None,
                prepared=None,
                tilt=[],
                candidates={"A->TP1": dict(budget=dict(supported=True))},
            )
            for t in range(1500, 1620)
        ]

    def test_only_the_intended_sustained_windows_cover_the_round_trip(self):
        steps = [
            dict(
                plan=dict(state="COMPLETED", bundle_id="A", target_bundle_mode=mode),
                triggered_unix_s=10000 + start,
                completed_unix_s=10000 + end,
            )
            for mode, start, end in [("TP2", 399, 600), ("TP1", 1672, 1720)]
        ]
        self.assertTrue(rotation.guard_round_trip(steps, 10000))
        for start in (1505, 1600, 2220):
            changed = copy.deepcopy(steps)
            changed[1]["triggered_unix_s"] = 10000 + start
            self.assertFalse(rotation.guard_round_trip(changed, 10000))
        steps[0]["completed_unix_s"] = 11501
        self.assertFalse(rotation.guard_round_trip(steps, 10000))

    def test_live_supported_keep_covers_pulse_and_recovery(self):
        self.assertTrue(rotation.pulse_coverage(self.records(), 10000)["passed"])

    def test_missing_observation_or_wrong_layout_cannot_pass_as_resistance(self):
        records = self.records()
        for changed in ([], records[:10], records[::10]):
            self.assertFalse(rotation.pulse_coverage(changed, 10000)["passed"])
        for changes in (
            dict(signals=dict(layout="L1AC")),
            dict(proposal=dict(bundle="A", target_mode="TP1")),
            dict(candidates={"A->TP1": dict(budget=dict(supported=False))}),
            dict(tilt=["A"]),
            dict(decision="COMMITTED"),
        ):
            current = copy.deepcopy(records)
            current[45].update(changes)
            self.assertFalse(rotation.pulse_coverage(current, 10000)["passed"])


if __name__ == "__main__":
    unittest.main()
