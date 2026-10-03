# SPDX-License-Identifier: Apache-2.0
"""Decision precedence for the advisory TP1-residency policy."""

import importlib.util
import sys
import unittest
from pathlib import Path


POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm/bridge_tp/controller/residency_policy.py"
)
SPEC = importlib.util.spec_from_file_location("residency_policy_test", POLICY_PATH)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)


def evidence(**changes):
    data = dict(
        prefill_complete=True, state="LOCAL", source_fresh=True,
        source_free_kv_tokens=20000, source_guard_kv_tokens=8448,
        unallocated_prefill_tokens=0, safe_pool_growth_tokens_s=100,
        ready_best_s=5.0, target_safe=True, channel_available=True,
        long_probability_lower=0.2, gain_lower_s=1.0,
    )
    data.update(changes)
    return POLICY.ResidencyEvidence(**data)


class TestResidencyPolicy(unittest.TestCase):
    def test_safe_short_request_stays_on_tp1(self):
        result = POLICY.decide_residency(evidence())
        self.assertEqual(result.action, "STAY_TP1")
        self.assertEqual(result.headroom_tokens, 11552)

    def test_capacity_deadline_overrides_low_length_probability(self):
        result = POLICY.decide_residency(evidence(
            source_free_kv_tokens=9000, long_probability_lower=0.0,
        ))
        self.assertEqual(result.action, "START_SHADOW_CAPACITY")
        self.assertLess(result.time_to_guard_s, 7)

    def test_unavailable_target_requires_protection_if_due(self):
        result = POLICY.decide_residency(evidence(
            source_free_kv_tokens=9000, target_safe=False,
        ))
        self.assertEqual(result.action, "CAPACITY_PROTECT")

    def test_already_late_requires_protection_alongside_shadow(self):
        result = POLICY.decide_residency(evidence(
            source_free_kv_tokens=8600,
        ))
        self.assertEqual(result.action, "CAPACITY_PROTECT_AND_START_SHADOW")

    def test_long_start_requires_both_probability_and_gain(self):
        config = POLICY.ResidencyConfig(long_probability_min=0.8)
        self.assertEqual(POLICY.decide_residency(evidence(
            long_probability_lower=0.9, gain_lower_s=1.0,
        ), config).action, "START_SHADOW_LONG")
        self.assertEqual(POLICY.decide_residency(evidence(
            long_probability_lower=0.9, gain_lower_s=-0.1,
        ), config).action, "STAY_TP1")

    def test_unknown_growth_does_not_become_infinite_slack(self):
        result = POLICY.decide_residency(evidence(
            safe_pool_growth_tokens_s=None,
        ))
        self.assertEqual(result.action, "CAPACITY_UNKNOWN")
        self.assertIsNone(result.time_to_guard_s)

    def test_no_fixed_generated_token_gate(self):
        result = POLICY.decide_residency(evidence(
            long_probability_lower=0.9,
        ))
        self.assertEqual(result.action, "START_SHADOW_LONG")


if __name__ == "__main__":
    unittest.main()
