"""Focused checks for the experimental M1 timing gate."""

import unittest

from tools.bridge_tp.experiment_m1_wait import M1PredictorRefreshGate


def event(position: int) -> dict:
    return {"status": "AVAILABLE", "prediction_output_tokens": position}


def decide(gate: M1PredictorRefreshGate, action: str,
           position: int | None, guard: float = 100.0) -> tuple:
    return gate.decide(
        m1_action=action, m5_row=event(position) if position is not None
        else None, source_time_to_guard_s=guard,
        estimated_preparation_s=5.0, source_release_tail_s=5.0,
    )


class TestM1PredictorRefreshGate(unittest.TestCase):
    def test_now_requires_available_m5_and_m1_start(self) -> None:
        gate = M1PredictorRefreshGate("NOW")
        self.assertEqual(decide(gate, "STAY", 120), (False, None))
        self.assertEqual(decide(gate, "START_SHADOW", None),
                         (False, "NOW_M5_UNAVAILABLE"))
        self.assertEqual(decide(gate, "START_SHADOW", 120),
                         (True, "NOW_M5_AVAILABLE"))

    def test_wait_requires_new_prediction_then_rechecks_m1(self) -> None:
        gate = M1PredictorRefreshGate("WAIT")
        self.assertEqual(decide(gate, "START_SHADOW", 120),
                         (False, "WAIT_ARMED"))
        self.assertEqual(decide(gate, "START_SHADOW", 120),
                         (False, "WAIT_FOR_FRESH_M5"))
        self.assertEqual(decide(gate, "STAY", 140),
                         (False, "WAIT_RECHECK_FRESH_M5"))
        self.assertEqual(decide(gate, "START_SHADOW", 140), (True, None))

    def test_capacity_deadline_releases_only_safe_m1_start(self) -> None:
        gate = M1PredictorRefreshGate("WAIT")
        decide(gate, "START_SHADOW", 120)
        self.assertEqual(decide(gate, "STAY", 120, guard=8),
                         (False, None))
        self.assertEqual(decide(gate, "START_SHADOW", 120, guard=8),
                         (True, "CAPACITY_SAFETY_RELEASE"))

    def test_urgent_first_candidate_does_not_wait(self) -> None:
        gate = M1PredictorRefreshGate("WAIT")
        self.assertEqual(decide(gate, "START_SHADOW", 120, guard=8),
                         (True, "CAPACITY_SAFETY_RELEASE"))

    def test_natural_eos_before_first_candidate_does_not_arm(self) -> None:
        gate = M1PredictorRefreshGate("WAIT")
        self.assertEqual(decide(gate, "STAY", 20), (False, None))
        self.assertIsNone(gate.baseline_position)


if __name__ == "__main__":
    unittest.main()
