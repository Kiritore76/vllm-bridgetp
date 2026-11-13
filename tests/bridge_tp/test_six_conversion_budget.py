# SPDX-License-Identifier: Apache-2.0
"""Causal demand, conversion payback and all six decisive entry directions."""

import copy
import unittest

from tests.bridge_tp.test_six_card_decider import native_status
from tools.bridge_tp import conversion_decider as d
from tools.bridge_tp import six_conversion_budget as b


def status(layout="L1AC", short=2, long_rate=0, now=1000, blocks=0):
    value, modes, members = native_status(layout, short=short, blocks=blocks)
    value["unix_s"] = now
    value["service_shape"] = dict(
        platform="qwen3",
        tp1_context=32768,
        tp2_context=32768,
        tp1_slots=16,
        tp2_slots=16,
    )
    value["workload"]["conversion_demand"] = dict(
        schema="SIX_CAUSAL_DEMAND_V1",
        sampled_at=now,
        coverage_s=240,
        window_s=240,
        fixed_output=True,
        short_count=round(short * 30),
        short_window_s=30,
        long_count=round(long_rate * 240),
        long_recent_count=round(long_rate * 60),
        prototypes=dict(short=[512, 128], long=[16384, 8192]),
    )
    for bundle, ids in members.items():
        for name in ids:
            value["instances"][name]["inventory"].update(
                total_blocks=b.BLOCKS[modes[bundle]],
                block_size=16,
            )
    return value, modes, members


def estimate(value, modes, members, edge):
    decider = d.ConversionDecider(profile="six_card")
    s = decider.signals(value, modes, members, value["unix_s"])
    return b.estimate(value, modes, members, edge, value["unix_s"], s)


class CausalityTests(unittest.TestCase):
    def test_held_counts_are_published_by_the_owner_not_by_reader_iteration(self):
        from collections import deque
        from types import SimpleNamespace

        dispatch = d.load("pressure_dispatch")
        hold = SimpleNamespace(
            short=deque([(self.job(0), 0), (self.job(0, 16384, 8192), 0)]), long=deque()
        )
        owner = dispatch.ScheduledDispatch(
            [], None, [], 0, 10, hold=hold, observe_held=True
        )
        owner._publish_held_demand()
        self.assertEqual(owner.held_demand, (1, 1))
        snapshot = owner.held_demand
        hold.short.clear()
        self.assertEqual(snapshot, (1, 1))
        owner._publish_held_demand()
        self.assertEqual(owner.held_demand, (0, 0))

    def job(self, at, prompt=512, output=128, fixed=True):
        return dict(
            at_s=at,
            request=dict(
                prompt=[1] * prompt,
                max_tokens=output,
                ignore_eos=fixed,
            ),
        )

    def test_future_jobs_cannot_change_current_model_support_or_counts(self):
        jobs = [self.job(1), self.job(100, 16384, 8192, False)]
        observer = b.Arrivals()
        first = observer.snapshot(jobs, 30, 1000)
        self.assertTrue(first["fixed_output"])
        self.assertIsNone(first["prototypes"]["long"])
        self.assertEqual(first["long_count"], 0)
        self.assertEqual(observer.cursor, 1)
        later = observer.snapshot(jobs, 100, 1070)
        self.assertFalse(later["fixed_output"])
        self.assertEqual(later["long_count"], 1)

    def test_window_counts_release_even_if_not_dispatched_and_prune(self):
        observer = b.Arrivals()
        jobs = [self.job(1), self.job(20, 16384, 8192)]
        row = observer.snapshot(jobs, 30, 1000)
        self.assertEqual((row["short_count"], row["long_count"]), (1, 1))
        row = observer.snapshot(jobs, 300, 1270)
        self.assertEqual((row["short_count"], row["long_count"]), (0, 0))


class BudgetTests(unittest.TestCase):
    def test_light_long_demand_is_not_enough_to_pay_for_a_merge(self):
        value, modes, members = status(long_rate=0.04)
        result = estimate(value, modes, members, ("A", "TP2"))
        self.assertTrue(result["supported"])
        self.assertFalse(result["worthwhile"], result)

    def test_sustained_heavy_demand_can_pay_for_more_tp2(self):
        value, modes, members = status(long_rate=0.2)
        result = estimate(value, modes, members, ("A", "TP2"))
        self.assertTrue(result["worthwhile"], result)
        self.assertGreater(result["gain_tokens"], result["required_gain_tokens"])

    def test_short_and_long_are_both_considered_for_return(self):
        light = status("L2C", short=17)
        self.assertTrue(estimate(*light, ("A", "TP1"))["worthwhile"])
        heavy = status("L2C", short=17, long_rate=0.2)
        self.assertFalse(estimate(*heavy, ("A", "TP1"))["worthwhile"])

    def test_unfinished_source_work_is_charged_not_ignored(self):
        value, modes, members = status("L2C", short=17)
        value["instances"][members["A"][0]]["requests"] = [
            dict(
                request_id="long",
                prompt_tokens=16384,
                generated=0,
                max_output_tokens=8192,
            )
        ]
        result = estimate(value, modes, members, ("A", "TP1"))
        self.assertGreater(result["ready_s"], b.HORIZON_S)
        self.assertFalse(result["worthwhile"])

    def test_completion_releases_demand_and_current_kv_is_not_permanent_load(self):
        value, modes, members = status("L1AC", blocks=16800)
        # These native fixture requests have already emitted their fixed output.
        for row in value["instances"].values():
            for q in row["requests"]:
                q["generated"] = q["max_output_tokens"]
        result = estimate(value, modes, members, ("A", "TP2"))
        self.assertFalse(result["worthwhile"])

    def test_cost_and_gain_have_same_unit_and_margin(self):
        result = estimate(*status(long_rate=0.2), ("A", "TP2"))
        self.assertEqual(result["unit"], "token_service_opportunity")
        self.assertAlmostEqual(
            result["required_gain_tokens"], b.COST_MARGIN * result["cost_tokens"]
        )
        self.assertAlmostEqual(
            result["net_tokens"], result["gain_tokens"] - result["required_gain_tokens"]
        )

    def test_missing_stale_variable_and_foreign_shapes_cannot_allow_conversion(self):
        value, modes, members = status()
        changes = [{}, dict(unix_s=900), dict(service_shape={})]
        unknown = copy.deepcopy(value)
        unknown["workload"]["conversion_demand"]["fixed_output"] = False
        changes.append(unknown)
        for changed in changes:
            with self.subTest(changed=list(changed)):
                current = (
                    changed
                    if not changed or "workload" in changed
                    else dict(value, **changed)
                )
                result = (
                    estimate(current, modes, members, ("A", "TP2"))
                    if current
                    else b.estimate(current, modes, members, ("A", "TP2"), 1000, {})
                )
                self.assertFalse(result["supported"])
                self.assertFalse(result["worthwhile"])


    def test_sleep_mode_kv_shape_is_the_profile_shape(self):
        """11-13 guard40 standby: sleep mode left TP1 2343 / TP2 11747 blocks;
        an exact match made every budget unsupported for the whole run."""
        value, modes, members = status(long_rate=0.2)
        reference = estimate(value, modes, members, ("A", "TP2"))
        sleep = copy.deepcopy(value)
        for bundle, ids in members.items():
            for name in ids:
                sleep["instances"][name]["inventory"]["total_blocks"] = (
                    2343 if modes[bundle] == 1 else 11747
                )
        result = estimate(sleep, modes, members, ("A", "TP2"))
        self.assertTrue(result["supported"], result)
        self.assertEqual(result["native_blocks"], {"tp1": 2343, "tp2": 11747})
        # Packing keeps the profile blocks: the same estimate as the profile.
        for key in ("worthwhile", "gain_tokens", "cost_tokens", "stay_long_slots"):
            self.assertEqual(result[key], reference[key], key)
        for tp1, tp2, size in (
            (2324, 11712, 16),  # fewer blocks than measured
            (2325, int(11712 * 1.03), 16),  # beyond the tolerance
            (2343, 11747, 32),  # another block size
        ):
            with self.subTest(tp1=tp1, tp2=tp2, size=size):
                other = copy.deepcopy(value)
                for bundle, ids in members.items():
                    for name in ids:
                        other["instances"][name]["inventory"].update(
                            total_blocks=tp1 if modes[bundle] == 1 else tp2,
                            block_size=size,
                        )
                rejected = estimate(other, modes, members, ("A", "TP2"))
                self.assertFalse(rejected["supported"])
                self.assertEqual(
                    rejected["reason"], "native shape differs from service-budget profile"
                )


class DecisionTests(unittest.TestCase):
    def run_case(self, layout, short, long_rate, edge, blocks=0, seconds=120):
        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=1)
        )
        proposals = []
        for second in range(seconds + 1):
            value, modes, members = status(
                layout, short, long_rate, 1000 + second, blocks
            )
            record = decider.observe(value, modes, members, 1000 + second)
            self.assertFalse(record["tilt"])
            self.assertIsNone(record["prepared"])
            if record["proposal"]:
                proposals.append((second, record["proposal"]))
        self.assertEqual(len(proposals), 1, record)
        self.assertEqual(
            (proposals[0][1]["bundle"], proposals[0][1]["target_mode"]), edge
        )
        return proposals[0][0]

    def test_all_six_directions_share_the_payback_policy(self):
        for layout, short, long_rate, edge, blocks, when in [
            ("L0", 2, 0.2, ("B", "TP2"), 0, 30),
            ("L1AC", 20, 0, ("B", "TP1"), 0, 120),
            ("L1AC", 2, 0.2, ("A", "TP2"), 0, 30),
            ("L2C", 17, 0, ("A", "TP1"), 0, 30),
            ("L2C", 2, 0.2, ("C", "TP2"), 18432, 120),
            ("L3", 17, 0, ("C", "TP1"), 0, 30),
        ]:
            with self.subTest(layout=layout, edge=edge):
                self.assertEqual(
                    self.run_case(layout, short, long_rate, edge, blocks), when
                )

    def test_light_workload_remains_keep_without_half_preparation(self):
        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=1)
        )
        for second in range(180):
            value, modes, members = status(long_rate=0.04, now=1000 + second)
            record = decider.observe(value, modes, members, 1000 + second)
            self.assertEqual(record["decision"], "KEEP")
            self.assertIsNone(record["proposal"])
            self.assertIsNone(record["prepared"])
            self.assertFalse(record["tilt"])

    def test_jitter_does_not_restart_confirmation_and_submission_latches(self):
        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=1)
        )
        proposals = []
        for second in range(50):
            short = 17.1 if second % 2 else 16.9
            value, modes, members = status("L2C", short=short, now=1000 + second)
            record = decider.observe(value, modes, members, 1000 + second)
            if record["proposal"]:
                proposals.append(second)
        self.assertEqual(proposals, [30])
        value, modes, members = status("L2C", short=2, now=1050)
        self.assertEqual(
            decider.observe(value, modes, members, 1050)["decision"], "COMMITTED"
        )

    def test_prior_need_can_submit_immediately_when_source_cost_falls(self):
        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=1)
        )
        for second in range(31):
            value, modes, members = status("L2C", short=18, now=1000 + second)
            value["instances"][members["A"][0]]["requests"] = [
                dict(
                    request_id="tail",
                    prompt_tokens=16384,
                    generated=0,
                    max_output_tokens=8192,
                )
            ]
            self.assertIsNone(
                decider.observe(value, modes, members, 1000 + second)["proposal"]
            )
        value, modes, members = status("L2C", short=18, now=1031)
        self.assertIsNotNone(decider.observe(value, modes, members, 1031)["proposal"])

    def test_continuation_request_preserves_known_fixed_final_context(self):
        value, modes, members = status("L1AC", long_rate=0.2)
        value["instances"][members["B"][0]]["requests"] = [
            dict(
                request_id="continued",
                prompt_tokens=17408,
                generated=10,
                max_output_tokens=7168,
            )
        ]
        self.assertTrue(estimate(value, modes, members, ("A", "TP2"))["supported"])

    def test_window_tolerates_brief_economic_reversals_but_not_retreat(self):
        window = b.Evidence()
        for t in range(31):
            result = window.observe("return", t, True, t not in (10, 11, 12), True, 30)
        self.assertTrue(result["ready"])
        for t in range(31, 62):
            result = window.observe("return", t, True, False, True, 30)
        self.assertFalse(result["ready"])

    def test_duplicate_samples_clock_reversal_and_gaps_do_not_confirm(self):
        window = b.Evidence()
        window.observe("edge", 1, True, True, True, 30)
        for _ in range(50):
            self.assertFalse(window.observe("edge", 1, True, True, True, 30)["ready"])
        self.assertFalse(window.observe("edge", 40, True, True, True, 30)["ready"])
        self.assertFalse(window.observe("edge", 0, True, True, True, 30)["ready"])


if __name__ == "__main__":
    unittest.main()
