# SPDX-License-Identifier: Apache-2.0
"""Probability and urgency remain separate experimental state inputs."""
import unittest

from tools.bridge_tp.experiment_probability_gate import (
    ProbabilityGate,
    urgency_eligibility,
)


class TestUrgencyStratum(unittest.TestCase):
    def test_probability_crossing_waits_for_urgency_in_both_assignments(self):
        for action in ("START", "STAY"):
            gate = ProbabilityGate(0.8, "seed", assigned_action=action)
            risk = {"status": "VALID", "p_guard_est_bounds": [0.92, 0.96],
                    "physical_feasible": True, "U": 0.6}
            first = gate.observe(risk, 1, urgency_eligibility(risk, 1.0))
            self.assertEqual(first["reason"], "LOAD_STRATUM_NOT_READY")
            self.assertIsNone(gate.candidate)
            urgent = {**risk, "U": 1.2}
            selected = gate.observe(urgent, 2, urgency_eligibility(urgent, 1.0))
            self.assertTrue(selected["first_feasible_candidate"])
            self.assertEqual(gate.candidate["snapshot"]["U"], 1.2)
            self.assertEqual(selected["requested_action"],
                             "START_SHADOW" if action == "START" else "STAY")

    def test_guard_reached_is_urgent_but_physical_gate_still_applies(self):
        risk = {"status": "GUARD_REACHED", "U": None,
                "source_guard_policy": "WARNING_NOT_START_DEADLINE",
                "p_guard_est_bounds": [1.0, 1.0],
                "physical_feasible": False,
                "source_physical_capacity_exhausted": True}
        self.assertEqual(urgency_eligibility(risk, 1.0), ())
        gate = ProbabilityGate(0.8, "seed", assigned_action="START")
        result = gate.observe(risk, 1, urgency_eligibility(risk, 1.0))
        self.assertIsNone(gate.candidate)
        self.assertTrue(result["safety_protection_required"])
        self.assertEqual(result["requested_action"], "STAY")

    def test_invalid_u_never_fakes_an_urgent_candidate(self):
        for value in (None, float("nan"), float("inf"), -1):
            self.assertTrue(urgency_eligibility({"U": value}, 1.0))
        self.assertEqual(urgency_eligibility({"U": None}, 0.0), ())
        for minimum in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                urgency_eligibility({}, minimum)
