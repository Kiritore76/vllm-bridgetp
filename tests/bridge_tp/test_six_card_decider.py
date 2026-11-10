# SPDX-License-Identifier: Apache-2.0
"""Six-card policy boundaries; no inference engines or GPU are started."""

import math
import unittest

from tools.bridge_tp import conversion_decider as d

MODES = {
    "L0": {"A": 1, "B": 1, "C": 1},
    "L1AC": {"A": 1, "B": 2, "C": 1},
    "L2C": {"A": 2, "B": 2, "C": 1},
    "L3": {"A": 2, "B": 2, "C": 2},
}
EXPECTED = {
    "L0": [["B", "TP2"]],
    "L1AC": [["A", "TP2"], ["B", "TP1"]],
    "L2C": [["C", "TP2"], ["A", "TP1"]],
    "L3": [["C", "TP1"]],
}


def signal(layout, short, blocks, new=1, trend=None, busy=None, capacity=0.2):
    return dict(
        layout=layout,
        short_rate=short,
        long_blocks=blocks,
        long_arrivals=new,
        trend_blocks=blocks if trend is None else trend,
        bundle_long_blocks={b: int(b == busy) for b in "ABC"},
        tp1_capacity_max=capacity,
        tpk_capacity_max=capacity,
        tpk_waiting=False,
    )


def native_status(layout, short=0, blocks=0, tag="0", busy=None, capacity=0.2):
    modes = MODES[layout]
    members, instances = {}, {}
    for b, tp in modes.items():
        members[b] = [f"{b}-tp{tp}-{n}" for n in range(2 if tp == 1 else 1)]
        for pool in members[b]:
            instances[pool] = dict(
                lifecycle="ACTIVE",
                pressure=dict(capacity=capacity, zone="SPARE"),
                inventory=dict(waiting=0),
                requests=[],
            )
    # Keep each synthetic request within 24K and native physical pool capacity.
    slots = [
        p
        for b, tp in modes.items()
        for p in members[b]
        for _ in range(7 if tp == 2 else 1)
    ]
    if busy:
        first = members[busy][0]
        slots.remove(first)
        slots.insert(0, first)
    count = math.ceil(blocks / 1536)
    assert count <= len(slots)
    if count:
        width, rest = divmod(blocks, count)
        for n in range(count):
            size = width + int(n < rest)
            prompt = min(size, 1024)
            instances[slots[n]]["requests"].append(
                dict(
                    request_id=f"r{n}-{tag if n == 0 else 'stable'}",
                    prompt_tokens=prompt * 16,
                    generated=(size - prompt) * 16,
                    max_output_tokens=8192,
                )
            )
    status = dict(
        instances=instances,
        workload=dict(
            short_window_s=30,
            short_window_arrivals=round(short * 30),
            short_window_long=0,
            held_long_prompt_tokens=[],
        ),
    )
    return status, modes, members


class ProfilesTests(unittest.TestCase):
    def test_selected_path_has_exactly_six_neighbouring_directions(self):
        self.assertEqual(d.profile_edges("six_card"), EXPECTED)
        self.assertEqual(sum(map(len, EXPECTED.values())), 6)
        for before, edges in EXPECTED.items():
            for b, mode in edges:
                after = dict(MODES[before], **{b: int(mode[-1])})
                self.assertIn(after, MODES.values())
                self.assertEqual(sum(MODES[before][k] != after[k] for k in "ABC"), 1)

    def test_registry_layout_labels_cover_all_three_bundle_assignments(self):
        for n in range(8):
            modes = {b: 2 if n & (1 << i) else 1 for i, b in enumerate("ABC")}
            count = sum(tp == 2 for tp in modes.values())
            expected = f"L{count}" + (
                "".join(b for b in "ABC" if modes[b] == 1) if count in (1, 2) else ""
            )
            self.assertEqual(d.layout_name(modes), expected)

    def test_four_card_profile_and_preference_are_unchanged(self):
        old = d.ConversionDecider(d.EDGES)
        self.assertEqual(old.rules, d.RULES)
        self.assertEqual(d.profile_edges("four_card"), d.EDGES)
        self.assertEqual(old.rules["l0_min_short"], 11)
        self.assertEqual(old.rules["l2_min_long_blocks"], 9817)
        self.assertNotIn("l3_max_short", old.rules)

    def test_profiles_do_not_share_mutable_rules(self):
        one = d.ConversionDecider(profile="six_card")
        one.rules["l2_max_short"] = 99
        self.assertEqual(
            d.ConversionDecider(profile="six_card").rules["l2_max_short"], 15
        )
        self.assertEqual(d.RULES["l2_max_short"], 6)

    def test_invalid_profile_path_or_six_card_rules_are_rejected(self):
        with self.assertRaises(ValueError):
            d.ConversionDecider(profile="guess")
        for ids, order in [
            (("A",), ("A",)),
            (("A", "A"), ("A", "A")),
            (("A", "B", "C"), ("A", "B")),
        ]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                d.adjacent_edges(ids, order)
        for rules in [
            dict(current_length=False),
            dict(evacuate_busy=True),
            dict(nothing=1),
            dict(l3_hold_s=-1),
            dict(l3_max_short=math.nan),
        ]:
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                d.ConversionDecider(profile="six_card", rules=rules)
        with self.assertRaises(ValueError):
            d.ConversionDecider({"L0": [["C", "TP2"]]}, profile="six_card")
        with self.assertRaises(ValueError):
            d.ConversionDecider({"L4": []}, profile="six_card")

    def test_launcher_selects_the_six_card_policy_explicitly(self):
        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        request = launch.decider_request(
            dict(decider="EXECUTE", deadline_s=600, base_port=8400, profile="six_card")
        )
        self.assertEqual(request["edges"], d.profile_edges("six_card"))
        self.assertEqual(request["profile"], "six_card")


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.d = d.ConversionDecider(profile="six_card")

    def check(self, layout, edge, short, blocks, **kw):
        return self.d.check(edge, signal(layout, short, blocks, **kw), MODES[layout])

    def test_l0_entry_needs_all_three_conditions(self):
        edge = ("B", "TP1")
        self.assertEqual(self.check("L1AC", edge, 19, 1536, new=0)[0], [])
        for short, blocks, new in [(18.99, 1536, 0), (19, 1537, 0), (19, 1536, 1)]:
            self.assertTrue(self.check("L1AC", edge, short, blocks, new=new)[0])

    def test_l0_exit_uses_any_of_short_blocks_or_new_long(self):
        for short, blocks, new in [(17, 0, 0), (20, 3072, 0), (20, 0, 1)]:
            self.assertFalse(self.check("L0", ("B", "TP2"), short, blocks, new=new)[0])
        self.assertTrue(self.check("L0", ("B", "TP2"), 17.01, 3071, new=0)[0])

    def test_middle_merge_uses_trend_and_needs_new_long(self):
        edge = ("A", "TP2")
        self.assertFalse(self.check("L1AC", edge, 15, 10000, trend=12600)[0])
        for short, trend, new in [(15.01, 12600, 1), (15, 12599, 1), (15, 12600, 0)]:
            self.assertTrue(
                self.check("L1AC", edge, short, 10000, trend=trend, new=new)[0]
            )

    def test_middle_split_short_boundary_and_queue_urgency(self):
        self.assertFalse(self.check("L2C", ("A", "TP1"), 17, 18000)[0])
        self.assertTrue(self.check("L2C", ("A", "TP1"), 16.99, 18000)[0])
        s = signal("L2C", 17, 18000)
        s["tpk_waiting"] = True
        self.assertTrue(self.d.check(("A", "TP1"), s, MODES["L2C"])[2])

    def test_l3_entry_uses_current_demand_not_a_large_trend(self):
        edge = ("C", "TP2")
        self.assertFalse(self.check("L2C", edge, 6, 16800)[0])
        for short, blocks, new in [(6.01, 16800, 1), (6, 16799, 1), (6, 16800, 0)]:
            self.assertTrue(
                self.check("L2C", edge, short, blocks, new=new, trend=99999)[0]
            )

    def test_l3_exit_can_follow_falling_long_demand_at_low_short_rate(self):
        edge = ("C", "TP1")
        self.assertFalse(self.check("L3", edge, 8, 20000, new=0)[0])
        self.assertFalse(self.check("L3", edge, 1, 14500, new=0)[0])
        self.assertTrue(self.check("L3", edge, 7.99, 14501, new=0)[0])

    def test_extreme_entries_never_take_the_urgent_shortcut(self):
        for layout, edge, short, blocks, new in [
            ("L1AC", ("B", "TP1"), 19, 1536, 0),
            ("L2C", ("C", "TP2"), 6, 16800, 1),
        ]:
            s = signal(layout, short, blocks, new=new, capacity=99)
            failed, _, urgent = self.d.check(edge, s, MODES[layout])
            self.assertFalse(failed)
            self.assertFalse(urgent)
            self.assertEqual(self.d._hold_seconds(edge, s, urgent=True), 120)

    def test_middle_merge_keeps_its_kv_urgent_hold(self):
        s = signal("L1AC", 15, 12600, capacity=1)
        failed, _, urgent = self.d.check(("A", "TP2"), s, MODES["L1AC"])
        self.assertFalse(failed)
        self.assertTrue(urgent)
        self.assertEqual(self.d._hold_seconds(("A", "TP2"), s, urgent), 30)

    def test_splitting_a_busy_tp2_waits_without_same_tp_evacuation(self):
        for layout, edge, short, blocks in [
            ("L1AC", ("B", "TP1"), 19, 1536),
            ("L2C", ("A", "TP1"), 17, 18000),
            ("L3", ("C", "TP1"), 8, 14500),
        ]:
            failed, waiting, _ = self.check(
                layout, edge, short, blocks, new=0, busy=edge[0]
            )
            self.assertFalse(failed)
            self.assertTrue(waiting)
        self.assertFalse(self.d.rules["evacuate_busy"])

    def test_held_long_prevents_entering_l0_even_when_blocks_are_small(self):
        value, modes, members = native_status("L1AC", short=19)
        value["workload"]["held_long_prompt_tokens"] = [8192]
        s = self.d.signals(value, modes, members, 1000)
        self.assertEqual((s["long_blocks"], s["long_arrivals"]), (512, 1))
        self.assertTrue(self.d.check(("B", "TP1"), s, modes)[0])


class LegacyTimingTests(unittest.TestCase):
    """The explicit decision_budget=0 compatibility path remains reproducible."""

    def test_l0_needs_the_full_120_seconds(self):
        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=0)
        )
        value, modes, members = native_status(
            "L1AC", short=19, blocks=1536, capacity=99
        )
        decider.signals(value, modes, members, 0)
        proposals = []
        for t in range(1000, 1121):
            row = decider.observe(value, modes, members, t)
            if row["proposal"]:
                proposals.append((t, row["proposal"]))
        self.assertEqual([t for t, _ in proposals], [1120])
        self.assertEqual(proposals[0][1]["target_mode"], "TP1")

    def test_l3_needs_120_seconds_of_sustained_conditions(self):
        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=0)
        )
        proposals = []
        for second in range(121):
            value, modes, members = native_status(
                "L2C",
                short=6,
                blocks=16800,
                tag=str(second // 20),
                capacity=99,
            )
            row = decider.observe(value, modes, members, 1000 + second)
            if row["proposal"]:
                proposals.append(1000 + second)
        self.assertEqual(proposals, [1120])

    def test_a_stopped_burst_cannot_enter_l3_after_120_seconds(self):
        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=0)
        )
        value, modes, members = native_status("L2C", short=6, blocks=16800)
        for t in range(1000, 1201):
            self.assertIsNone(decider.observe(value, modes, members, t)["proposal"])

    def test_l0_returns_in_30_seconds_when_short_rate_falls(self):
        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=0)
        )
        value, modes, members = native_status("L0", short=17)
        rows = [decider.observe(value, modes, members, t) for t in range(1000, 1031)]
        self.assertFalse(any(r["proposal"] for r in rows[:-1]))
        self.assertEqual(rows[-1]["proposal"]["bundle"], "B")

    def test_long_demand_drop_can_exit_l3_and_cooldown_still_applies(self):
        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=0)
        )
        decider.converted(1000)
        value, modes, members = native_status("L3", short=1)
        rows = [decider.observe(value, modes, members, t) for t in range(1000, 1121)]
        self.assertFalse(any(r["proposal"] for r in rows[:-1]))
        self.assertEqual(rows[-1]["proposal"]["bundle"], "C")

    def test_busy_tp2_tilts_but_waits_until_naturally_empty(self):
        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=0)
        )
        value, modes, members = native_status("L3", short=8, blocks=1024, busy="C")
        for t in range(1000, 1050):
            for instance in value["instances"].values():
                instance["unified_fullness"] = dict(
                    kind="route",
                    complete=True,
                    observed_at=t,
                    sampled_at=t,
                    published_at=t,
                    itl_sampled_at=t,
                    slots=0.1,
                    kv=0.1,
                    speed=0.1,
                    value=0.1,
                )
            row = decider.observe(value, modes, members, t)
            self.assertIsNone(row["proposal"])
        self.assertEqual(row["tilt"], ["C"])
        empty, _, _ = native_status("L3", short=8)
        self.assertEqual(
            decider.observe(empty, modes, members, 1050)["proposal"]["bundle"], "C"
        )

    def test_incomplete_registry_clears_hold_instead_of_becoming_l0(self):
        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=0)
        )
        value, modes, members = native_status("L0", short=17)
        decider.observe(value, modes, members, 1000)
        partial = {b: tp for b, tp in modes.items() if b != "B"}
        row = decider.observe(value, partial, {b: members[b] for b in partial}, 1020)
        self.assertIsNone(row["proposal"])
        self.assertIsNone(row["signals"]["layout"])
        self.assertFalse(decider.since)
        self.assertIsNone(decider.observe(value, modes, members, 1030)["proposal"])
        self.assertIsNotNone(decider.observe(value, modes, members, 1060)["proposal"])


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.d = d.ConversionDecider(profile="six_card")

    def test_l0_entry_cancels_on_new_long_or_more_current_demand(self):
        args = (("B", "TP1"), "L1AC", 19)
        self.assertIsNone(self.d.cancel_reason(*args, 1536, long_arrivals=0))
        self.assertTrue(self.d.cancel_reason(*args, 1536, long_arrivals=1))
        self.assertTrue(self.d.cancel_reason(*args, 1537, long_arrivals=0))

    def test_l3_entry_cancels_even_if_only_the_trend_stays_high(self):
        args = (("C", "TP2"), "L2C", 6)
        self.assertIsNone(self.d.cancel_reason(*args, 16800, long_arrivals=1))
        self.assertTrue(
            self.d.cancel_reason(*args, 16799, long_arrivals=1, trend_blocks=99999)
        )
        self.assertTrue(self.d.cancel_reason(*args, 16800, long_arrivals=0))

    def test_low_long_exit_is_cancelled_only_when_both_exit_reasons_disappear(self):
        args = (("C", "TP1"), "L3")
        self.assertIsNone(self.d.cancel_reason(*args, 1, 14500))
        self.assertIsNone(self.d.cancel_reason(*args, 8, 20000))
        self.assertTrue(self.d.cancel_reason(*args, 7.99, 14501))

    def test_middle_merge_preserves_existing_short_only_cancel(self):
        self.assertIsNone(
            self.d.cancel_reason(("A", "TP2"), "L1AC", 15, 0, long_arrivals=0)
        )
        self.assertTrue(self.d.cancel_reason(("A", "TP2"), "L1AC", 15.01, 12600))

    def test_extreme_entry_requires_the_new_long_signal_for_cancel_checks(self):
        self.assertEqual(
            self.d.cancel_reason(("B", "TP1"), "L1AC", 19, 0),
            "new-long signal unavailable",
        )


if __name__ == "__main__":
    unittest.main()
