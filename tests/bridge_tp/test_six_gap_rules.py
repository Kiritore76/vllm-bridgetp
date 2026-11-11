# SPDX-License-Identifier: Apache-2.0
"""Six-card gap rules: propose on an observed, sustained deficit (CPU)."""

import unittest
from collections import deque
from types import SimpleNamespace

from tests.bridge_tp.test_six_conversion_budget import status
from tools.bridge_tp import conversion_decider as d
from tools.bridge_tp import run_pressure_balanced_pilot as pilot


def sample(layout, now, short=2, blocks=0, **held):
    """One status; ``held`` overrides the published held-request demand."""
    value, modes, members = status(layout, short, 0, now, blocks)
    value["workload_origin_unix_s"] = 900.0
    value["workload"]["conversion_demand"].update(
        dict(
            held_long_count=0,
            held_short_count=0,
            held_long_wait_s=0.0,
            held_short_wait_s=0.0,
            held_long_prompt_tokens=[],
        )
        | held
    )
    return value, modes, members


def run(layout, seconds, at, decider=None):
    """Proposals [(second, bundle, mode)]; ``at(second)`` gives sample kwargs."""
    decider = decider or d.ConversionDecider(profile="six_card", execute=True)
    proposals, record = [], None
    for second in range(seconds + 1):
        value, modes, members = sample(layout, 1000 + second, **at(second))
        record = decider.observe(value, modes, members, 1000 + second)
        assert record["policy"] == "SIX_GAP_RULES_V1"
        assert not record["tilt"] and record["prepared"] is None
        if record["proposal"]:
            p = record["proposal"]
            proposals.append((second, p["bundle"], p["target_mode"]))
    return proposals, record, decider


# 11 in-flight long requests (16000 blocks) plus held 16K prompts (1024 each).
def long_demand(held_prompts, wait=1.0):
    return dict(
        blocks=16000,
        held_long_count=held_prompts,
        held_long_wait_s=wait if held_prompts else 0.0,
        held_long_prompt_tokens=[16384] * held_prompts,
    )


class MergeTests(unittest.TestCase):
    def test_demand_near_all_of_l1ac_merges_after_the_hold(self):
        # 16000 + 4 * 1024 = 20096 >= 20000.
        proposals, record, _ = run("L1AC", 60, lambda s: long_demand(4))
        self.assertEqual(proposals, [(30, "A", "TP2")])
        self.assertEqual(record["decision"], "COMMITTED")

    def test_demand_l1ac_still_holds_does_not_merge(self):
        # 19072 blocks: above the old 12600 line, where static L1AC passed.
        proposals, record, _ = run("L1AC", 200, lambda s: long_demand(3))
        self.assertEqual(proposals, [])
        self.assertEqual(record["decision"], "KEEP")
        failed = record["candidates"]["A->TP2"]["failed"]
        self.assertEqual(failed, ["long demand within what L1AC holds"])

    def test_a_long_request_held_long_enough_merges_at_once(self):
        def at(second):
            return dict(
                blocks=8000,
                held_long_count=1,
                held_long_wait_s=float(second),
                held_long_prompt_tokens=[16384],
            )

        proposals, _, _ = run("L1AC", 60, at)
        self.assertEqual(proposals, [(45, "A", "TP2")])

    def test_high_short_rate_or_no_new_long_request_blocks_the_merge(self):
        proposals, _, _ = run("L1AC", 90, lambda s: dict(long_demand(4), short=15.5))
        self.assertEqual(proposals, [])
        # A burst that stopped arriving: nothing new or held after the window.
        decider = d.ConversionDecider(profile="six_card", execute=True)
        proposals, record, _ = run("L1AC", 200, lambda s: dict(blocks=16000), decider)
        self.assertEqual(proposals, [])
        self.assertIn(
            "no new long request: a burst is left to migration",
            record["candidates"]["A->TP2"]["failed"],
        )

    def test_a_dip_restarts_the_hold(self):
        def at(second):
            return long_demand(3 if second == 20 else 4)

        proposals, _, _ = run("L1AC", 80, at)
        self.assertEqual(proposals, [(51, "A", "TP2")])


class SplitTests(unittest.TestCase):
    def test_short_rate_at_what_l2c_carries_splits_after_the_hold(self):
        # User 11-11: the 30s rate window is the perception; 5s confirms it.
        proposals, _, _ = run("L2C", 60, lambda s: dict(short=16.5))
        self.assertEqual(proposals, [(5, "A", "TP1")])

    def test_one_noisy_dip_restarts_the_split_confirmation(self):
        proposals, _, _ = run(
            "L2C", 60, lambda s: dict(short=15.9 if s == 3 else 16.5)
        )
        self.assertEqual(proposals, [(9, "A", "TP1")])

    def test_split_backs_confirm_in_5s_and_l0_entry_in_30s(self):
        rules = d.ConversionDecider(profile="six_card").rules
        self.assertEqual(rules["split_confirm_s"], 5.0)
        self.assertEqual(rules["l0_entry_confirm_s"], 30.0)
        self.assertNotIn("gap_split_hold_s", rules)
        for layout, at, edge, hold in (
            ("L2C", lambda s: dict(short=16.5), "A", 5.0),
            ("L3", lambda s: dict(short=17), "C", 5.0),
            (
                "L1AC",
                lambda s: dict(short=20, held_short_count=16, held_short_wait_s=2),
                "B",
                30.0,
            ),
        ):
            with self.subTest(layout=layout):
                proposals, record, _ = run(layout, 40, at)
                self.assertEqual(proposals, [(int(hold), edge, "TP1")])
                self.assertEqual(record["candidates"][f"{edge}->TP1"]["hold_s"], hold)

    def test_below_the_line_or_a_ten_second_spike_does_not_split(self):
        proposals, _, _ = run("L2C", 200, lambda s: dict(short=15.9))
        self.assertEqual(proposals, [])
        # ``short`` is the 30s window reading: a 10s 17.5/s pulse over 2/s
        # reads at most (10 * 17.5 + 20 * 2) / 30 = 7.2/s (guard40 1500-1510s).
        def window(second):
            seconds = range(second - 29, second + 1)
            return dict(short=sum(17.5 if 50 <= t < 60 else 2.0 for t in seconds) / 30)

        proposals, _, _ = run("L2C", 200, window)
        self.assertEqual(proposals, [])
        self.assertLess(max(window(s)["short"] for s in range(200)), 7.2)
        # The same window still splits a sustained 17.5/s: reading >= 16 after
        # 28s of it (27.1s exactly), then 5s confirmation.
        sustained = [
            sum(17.5 if t >= 50 else 2.0 for t in range(s - 29, s + 1)) / 30
            for s in range(120)
        ]
        proposals, _, _ = run("L2C", 119, lambda s: dict(short=sustained[s]))
        self.assertEqual(proposals, [(82, "A", "TP1")])

    def test_shorts_held_behind_long_requests_are_not_a_short_deficit(self):
        """GPU 11-10 long phase: 63 shorts held 32s at a 2.3/s short rate."""

        def at(second):
            return dict(
                short=17.5,
                held_short_count=63,
                held_short_wait_s=32.0,
                held_long_count=4,
                held_long_wait_s=30.0,
                held_long_prompt_tokens=[16384] * 4,
            )

        proposals, record, _ = run("L2C", 120, at)
        self.assertEqual([p for p in proposals if p[2] == "TP1"], [])
        self.assertIn(
            "a long request is held: not a short deficit",
            record["candidates"]["A->TP1"]["failed"],
        )

    def test_a_long_request_on_the_group_delays_the_split(self):
        decider = d.ConversionDecider(profile="six_card", execute=True)
        proposals = []
        for second in range(120):
            value, modes, members = sample("L2C", 1000 + second, short=16.5)
            if second < 80:
                value["instances"][members["A"][0]]["requests"].append(
                    dict(
                        request_id="long-on-A",
                        prompt_tokens=16384,
                        generated=1000,
                        max_output_tokens=8192,
                    )
                )
            record = decider.observe(value, modes, members, 1000 + second)
            if record["proposal"]:
                proposals.append(second)
        self.assertEqual(proposals, [80])


class CommitAndSignalTests(unittest.TestCase):
    def test_a_submitted_move_is_not_reconsidered(self):
        proposals, record, decider = run(
            "L2C", 90, lambda s: dict(short=16.5 if s <= 30 else 2)
        )
        self.assertEqual(proposals, [(5, "A", "TP1")])
        self.assertEqual(record["decision"], "COMMITTED")
        self.assertEqual(decider.committed_edge, ("A", "TP1"))

    def test_missing_signals_after_the_workload_started_fail_loudly(self):
        decider = d.ConversionDecider(profile="six_card", execute=True)
        for drop in ("service_shape", "conversion_demand"):
            value, modes, members = sample("L1AC", 1000)
            if drop == "service_shape":
                value["service_shape"] = None
            else:
                del value["workload"]["conversion_demand"]
            with self.assertRaisesRegex(RuntimeError, "signals missing"):
                decider.observe(value, modes, members, 1000)
        # Before the workload starts the controller has no demand yet.
        value, modes, members = sample("L1AC", 1000)
        value["workload_origin_unix_s"] = None
        del value["workload"]["conversion_demand"]
        self.assertEqual(
            decider.observe(value, modes, members, 1000)["decision"], "KEEP"
        )

    def test_a_withdrawal_or_abort_cools_down_before_the_next_proposal(self):
        """GPU 11-10: an aborted merge was proposed again 0.7s later."""
        proposals, _, decider = run("L2C", 40, lambda s: dict(short=16.5))
        self.assertEqual(proposals, [(5, "A", "TP1")])
        decider.cancelled(1040.0)
        self.assertIsNone(decider.committed_edge)
        again = []
        for second in range(41, 260):
            value, modes, members = sample("L2C", 1000 + second, short=16.5)
            record = decider.observe(value, modes, members, 1000 + second)
            if record["proposal"]:
                again.append(second)
        self.assertEqual(again, [160])  # 120s cooldown after the withdrawal
        # Four-card behaviour is unchanged: no cooldown after a cancel.
        four = d.ConversionDecider()
        four.cancelled(5.0)
        self.assertIsNone(four.converted_at)

    def test_pending_check_reports_a_vanished_deficit(self):
        decider = d.ConversionDecider(profile="six_card", execute=True)
        edge = ("A", "TP2")
        value, modes, members = sample("L1AC", 1000, **long_demand(4))
        self.assertEqual(
            decider.gap_pending_failed(value, modes, members, 1000, "L1AC", edge), []
        )
        value, modes, members = sample("L1AC", 1200)
        self.assertIn(
            "long demand within what L1AC holds",
            decider.gap_pending_failed(value, modes, members, 1200, "L1AC", edge),
        )

    def test_extreme_exits_are_not_delayed_by_the_stricter_entries(self):
        proposals, _, _ = run("L1AC", 150, lambda s: dict(short=20))
        self.assertEqual(proposals, [])  # rate alone no longer enters L0
        proposals, _, _ = run("L3", 60, lambda s: dict(short=17))
        self.assertEqual(proposals, [(5, "C", "TP1")])
        proposals, _, _ = run("L0", 60, lambda s: dict(short=2))
        self.assertEqual(proposals, [(30, "B", "TP2")])

    def test_defaults_and_the_retired_estimate_is_only_recorded(self):
        rules = d.ConversionDecider(profile="six_card").rules
        self.assertEqual(
            [
                rules[k]
                for k in (
                    "decision_budget",
                    "gap_merge_long_blocks",
                    "gap_merge_hold_s",
                    "gap_merge_held_long_s",
                    "gap_split_short",
                    "split_confirm_s",
                    "extreme_exit_hold_s",
                    "l3_hold_s",
                )
            ],
            # Merges keep their holds; only splits back confirm in 5s.
            [2, 20000, 30.0, 45.0, 16.0, 5.0, 30.0, 120.0],
        )
        _, record, _ = run("L1AC", 5, lambda s: long_demand(4))
        self.assertIn("worthwhile", record["candidates"]["A->TP2"]["budget"])
        # Four-card rules are untouched.
        self.assertNotIn("gap_split_short", d.ConversionDecider().rules)


class ExtremeEntryTests(unittest.TestCase):
    def test_l3_requires_multiple_long_requests_with_substantial_wait(self):
        for count, wait in ((0, 0), (1, 300), (2, 44.9)):
            with self.subTest(count=count, wait=wait):
                proposals, record, _ = run(
                    "L2C",
                    150,
                    lambda s, count=count, wait=wait: dict(
                        short=2, **long_demand(count, wait)
                    ),
                )
                self.assertEqual(proposals, [])
                self.assertIn(
                    "no substantial held long backlog for L3",
                    record["candidates"]["C->TP2"]["failed"],
                )

    def test_l3_still_enters_for_a_sustained_real_long_backlog(self):
        proposals, _, _ = run(
            "L2C", 120, lambda s: dict(short=2, **long_demand(2, wait=45))
        )
        self.assertEqual(proposals, [(120, "C", "TP2")])

    def test_l0_requires_a_short_backlog_not_just_a_high_short_rate(self):
        for count, wait in ((0, 0), (15, 10), (16, 1.99)):
            with self.subTest(count=count, wait=wait):
                proposals, _, _ = run(
                    "L1AC",
                    150,
                    lambda s, count=count, wait=wait: dict(
                        short=20, held_short_count=count, held_short_wait_s=wait
                    ),
                )
                self.assertEqual(proposals, [])
        proposals, _, _ = run(
            "L1AC",
            120,
            lambda s: dict(short=20, held_short_count=16, held_short_wait_s=2),
        )
        self.assertEqual(proposals, [(30, "B", "TP1")])

    def test_l0_requires_even_the_last_long_request_to_finish(self):
        value, modes, members = sample(
            "L1AC",
            1000,
            short=20,
            blocks=1000,
            held_short_count=32,
            held_short_wait_s=10,
        )
        decider = d.ConversionDecider(profile="six_card")
        reasons = decider.gap_pending_failed(
            value, modes, members, 1000, "L1AC", ("B", "TP1")
        )
        self.assertIn("L0 requires all long requests to finish", reasons)

    def test_a_brief_short_backlog_does_not_enter_l0(self):
        # A 25s backlog under a sustained window reading: shorter than the
        # 30s L0 confirmation, then gone, so no L0 (user 11-11).
        proposals, _, _ = run(
            "L1AC",
            150,
            lambda s: dict(
                short=20,
                held_short_count=32 if s < 25 else 0,
                held_short_wait_s=10 if s < 25 else 0,
            ),
        )
        self.assertEqual(proposals, [])

    def test_extreme_confirmation_restarts_after_the_backlog_clears(self):
        proposals, _, _ = run(
            "L2C",
            181,
            lambda s: dict(short=2, **long_demand(1 if s == 60 else 2, wait=60)),
        )
        self.assertEqual(proposals, [(181, "C", "TP2")])

    def test_observation_gap_cannot_count_toward_the_extreme_hold(self):
        decider = d.ConversionDecider(profile="six_card", execute=True)
        proposals = []
        for t in [*range(61), *range(100, 221)]:
            value, modes, members = sample("L2C", 1000 + t, **long_demand(2, wait=60))
            if decider.observe(value, modes, members, 1000 + t)["proposal"]:
                proposals.append(t)
        self.assertEqual(proposals, [220])

    def test_stale_or_unknown_backlog_cannot_enter_an_extreme(self):
        for change in (
            dict(sampled_at=900),
            dict(sampled_at=2000),
            dict(held_long_wait_s=float("nan")),
        ):
            decider = d.ConversionDecider(profile="six_card", execute=True)
            for t in range(121):
                value, modes, members = sample(
                    "L2C", 1000 + t, **long_demand(2, wait=60)
                )
                value["workload"]["conversion_demand"].update(change)
                self.assertIsNone(
                    decider.observe(value, modes, members, 1000 + t)["proposal"]
                )


class WiringTests(unittest.TestCase):
    def test_six_cards_are_recognised_from_the_validated_plan(self):
        """GPU 11-10: the controller read args.cards, which it never has."""
        self.assertTrue(pilot.six_card_plan(dict(runtime_pool_ids=dict(tp1="a"))))
        for manifest in (None, {}, dict(runtime_pool_ids=None), dict(instances=[])):
            self.assertFalse(pilot.six_card_plan(manifest))

    def test_the_owner_publishes_oldest_held_arrivals_and_long_prompts(self):
        dispatch = d.load("pressure_dispatch")

        def job(prompt, output):
            return dict(request=dict(prompt=[1] * prompt, max_tokens=output))

        hold = SimpleNamespace(
            short=deque(
                [(job(512, 128), 7.0), (job(16384, 8192), 5.0), (job(512, 128), 3.0)]
            ),
            long=deque(),
        )
        owner = dispatch.ScheduledDispatch(
            [], None, [], 0, 10, hold=hold, observe_held=True
        )
        owner._publish_held_demand()
        self.assertEqual(owner.held_demand, (2, 1))
        self.assertEqual(owner.held_detail, (3.0, 5.0, (16384,)))
        hold.short.clear()
        owner.held_demand_at = -float("inf")
        owner._publish_held_demand()
        self.assertEqual(owner.held_detail, (None, None, ()))


if __name__ == "__main__":
    unittest.main()
