# SPDX-License-Identifier: Apache-2.0
import json
import unittest

from tests.bridge_tp import test_six_rotation as previous
from tools.bridge_tp import qwen3_phase_shift as phase
from tools.bridge_tp import six_auto_smoke as smoke


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = phase.config(
            json.loads((previous.ASSETS / "inputs.json").read_text()), smoke.SCENARIO
        )

    def test_short_registered_input(self):
        self.assertEqual(self.config["horizon_s"], 600)
        self.assertEqual(self.config["recipes"][0]["input_sha256"], smoke.INPUT_SHA256)
        self.assertEqual(len(self.config["recipes"][0]["jobs"]), 619)
        self.assertTrue(all(j["at_s"] < 600 for j in self.config["recipes"][0]["jobs"]))

    def test_zero_and_safe_abort_do_not_prove_conversion(self):
        self.assertFalse(smoke.completion_coverage([], []))
        step = dict(
            plan=dict(state="ABORTED"),
            controller_acknowledged=True,
            layout=dict(A=1, B=2, C=1),
        )
        events = [dict(kind="DECIDER_PROPOSAL")]
        self.assertFalse(smoke.completion_coverage([step], events))
        step.update(plan=dict(state="COMPLETED"), layout=dict(A=2, B=2, C=1))
        self.assertFalse(smoke.completion_coverage([step], []))
        self.assertTrue(smoke.completion_coverage([step], events))

    def test_real_runner_assisted_auto_preflight(self):
        self.layouts = ("L1AC",)
        self.automatic = self.assist = True
        self.attempts = 1
        self.timeout = "1200"
        previous.RotationTests.test_real_runner_builds_and_preflights_static_arms_before_gpu(
            self
        )
