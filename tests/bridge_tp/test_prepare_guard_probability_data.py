# SPDX-License-Identifier: Apache-2.0
"""Check observed-arm selection and independent-event accounting."""

import unittest

from tools.bridge_tp.prepare_guard_probability_data import coverage, eligible_rows


def observation(episode: str, arm: str, state: str, tick: int = 1) -> dict:
    return {
        "episode": episode,
        "arm": arm,
        "observation": {
            "tick": tick, "unix_s": float(tick),
            "point_time_to_guard_s": 4.0, "m5_status": "AVAILABLE",
        },
        "label": {
            "counterfactual_stay_guard_status": (
                "OBSERVED_ARM" if arm == "stay" else "UNKNOWN"
            ),
            "guard_horizon_s": {"5": state, "10": state, "30": state},
            "source_release_evidence": "NATURAL_EOS_RESPONSE_PROXY",
        },
    }


class TestGuardProbabilityData(unittest.TestCase):
    def test_intervention_and_failed_runner_cannot_train_stay_risk(self):
        items = [
            observation("run/case/stay/r01_shadow_only", "stay", "OBSERVED_HIT"),
            observation("run/case/now/r01_shadow_only", "now", "OBSERVED_HIT"),
            observation("run/bad/stay/r01_shadow_only", "stay", "OBSERVED_HIT"),
        ]
        statuses = {
            "run/case/stay/r01_shadow_only": "PASS",
            "run/case/now/r01_shadow_only": "PASS",
            "run/bad/stay/r01_shadow_only": "FAILED",
        }
        rows, excluded = eligible_rows(items, statuses, "a.tar.gz", "digest")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["guard_hit_within_horizon"]["5"], 1)
        self.assertEqual(excluded["excluded_episodes"], {
            "INTERVENED_ARM": 1, "RUNNER_FAILED": 1,
        })

    def test_repeated_ticks_do_not_become_independent_positive_events(self):
        episode = "run/case/stay/r01_shadow_only"
        items = [observation(episode, "stay", "OBSERVED_HIT", tick)
                 for tick in range(1, 101)]
        rows, _ = eligible_rows(items, {episode: "PASS"}, "a.tar.gz", "digest")
        summary = coverage(rows)
        self.assertEqual(summary["horizons_s"]["5"]["positive_ticks"], 100)
        self.assertEqual(summary["horizons_s"]["5"]["episodes_with_positive"], 1)
        self.assertEqual(summary["horizons_s"]["5"]["fit_readiness"],
                         "INSUFFICIENT_INDEPENDENT_EVENTS")

    def test_censored_tick_has_no_negative_label(self):
        episode = "run/case/stay/r01_shadow_only"
        rows, _ = eligible_rows(
            [observation(episode, "stay", "CENSORED")],
            {episode: "PASS"}, "a.tar.gz", "digest",
        )
        self.assertIsNone(rows[0]["guard_hit_within_horizon"]["30"])
        self.assertEqual(coverage(rows)["horizons_s"]["30"]["negative_ticks"], 0)

    def test_augmented_prompt_hits_are_not_natural_prompt_hits(self):
        natural = "run/natural/stay/r01_shadow_only"
        augmented = "run/augmented/stay/r01_shadow_only"
        natural_rows, _ = eligible_rows(
            [observation(natural, "stay", "NO_HIT_IN_WINDOW_SAMPLES")],
            {natural: "PASS"}, "natural.tar.gz", "natural-sha",
            "NATURAL_PROMPTS",
        )
        augmented_rows, _ = eligible_rows(
            [observation(augmented, "stay", "OBSERVED_HIT")],
            {augmented: "PASS"}, "augmented.tar.gz", "augmented-sha",
            "PROMPT_AUGMENTED_DIAGNOSTIC",
        )
        classes = coverage(natural_rows + augmented_rows)["workload_classes"]
        self.assertEqual(classes["NATURAL_PROMPTS"]["guard_hit_episodes"]["5"], 0)
        self.assertEqual(
            classes["PROMPT_AUGMENTED_DIAGNOSTIC"]["guard_hit_episodes"]["5"],
            1,
        )

    def test_same_seed_and_case_remain_one_workload_block(self):
        episode = "run/case/stay/r01_shadow_only"
        first, _ = eligible_rows(
            [observation(episode, "stay", "OBSERVED_HIT")],
            {episode: "PASS"}, "first.tar.gz", "first-sha",
            "NATURAL_PROMPTS", 7,
        )
        second, _ = eligible_rows(
            [observation(episode, "stay", "OBSERVED_HIT")],
            {episode: "PASS"}, "second.tar.gz", "second-sha",
            "NATURAL_PROMPTS", 7,
        )
        summary = coverage(first + second)
        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["workload_block_count"], 1)
        self.assertEqual(
            summary["horizons_s"]["5"]["workload_blocks_with_positive"], 1
        )


if __name__ == "__main__":
    unittest.main()
