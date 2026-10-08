# SPDX-License-Identifier: Apache-2.0
"""Causal-boundary checks for passive migration risk observations."""

import unittest

from tools.bridge_tp.audit_migration_risk import audit_episode, summarize
from tools.bridge_tp.risk_observation import build_risk_observation


class TestRiskObservation(unittest.TestCase):
    def test_predecision_features_keep_length_probability_separate(self):
        row = build_risk_observation(
            tick=4,
            snapshot={"unix_s": 10.0, "source_free_kv_tokens": 120,
                      "source_guard_free_kv_tokens": 20,
                      "source_prefill_pending_kv_tokens": 10,
                      "source_decode_growth_tokens_s": 30.0},
            m5_row={"status": "AVAILABLE",
                    "p_remaining_gt_headroom_runtime_bounds": [0.7, 0.9]},
            natural_m1={"action": "START_SHADOW"},
            applied_m1={"action": "STAY"},
            initial_rate={"profile": "LOW", "rate_bytes_s": 1.0},
            assigned_action="WAIT",
        )
        self.assertEqual(row["source_safe_headroom_tokens"], 100)
        self.assertEqual(row["point_time_to_guard_s"], 100 / 30)
        self.assertEqual(row["natural_m1_action"], "START_SHADOW")
        self.assertEqual(row["applied_m1_action"], "STAY")
        self.assertFalse(row["m5_probability_is_pool_oom_risk"])
        self.assertEqual(row["pool_guard_probability_status"], "UNCALIBRATED")

    def test_observed_guard_and_counterfactual_censoring(self):
        observation = {"kind": "manager_risk_observation_shadow",
                       "tick": 1, "unix_s": 10.0, "m5_status": "AVAILABLE"}
        telemetry = [
            {"kind": "telemetry", "unix_s": 11.0,
             "capacity_signal": {"free_kv_tokens": 21,
                                 "guard_free_kv_tokens": 20,
                                 "prefill_pending_kv_tokens": 1000}},
            {"kind": "telemetry", "unix_s": 12.0,
             "capacity_signal": {"free_kv_tokens": 19,
                                 "guard_free_kv_tokens": 20,
                                 "prefill_pending_kv_tokens": 0}},
        ]
        receipt = {"status": "SOURCE_KV_RELEASED",
                   "released_unix_ns": 13_000_000_000}
        items = audit_episode("run/now/r01", [observation, *telemetry],
                              None, receipt)
        label = items[0]["label"]
        self.assertEqual(label["observed_guard_status"],
                         "OBSERVED_SAMPLED_GUARD_HIT")
        self.assertEqual(label["observed_time_to_guard_s"], 2.0)
        self.assertEqual(label["counterfactual_stay_guard_status"], "UNKNOWN")
        self.assertEqual(summarize(items)["observed_guard_hit_episodes"], 1)

    def test_unfinished_path_is_censored(self):
        rows = [{"kind": "manager_risk_observation_shadow", "tick": 1,
                 "unix_s": 10.0, "m5_status": "UNAVAILABLE"}]
        items = audit_episode("run/stay/r01", rows,
                              {"finish_reason": "length",
                               "completed_unix_s": 12.0}, None)
        self.assertEqual(items[0]["label"]["observed_guard_status"],
                         "CENSORED")
        self.assertIsNone(items[0]["label"]["observed_time_to_release_s"])


if __name__ == "__main__":
    unittest.main()
