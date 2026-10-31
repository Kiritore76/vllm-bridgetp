# SPDX-License-Identifier: Apache-2.0
"""S4: SHADOW planner inside a live DRAIN conversion (CPU state tests)."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from tools.bridge_tp import conversion_planner as cp
from tools.bridge_tp import run_pressure_balanced_pilot as pilot
from tools.bridge_tp.pressure_cpu import load

bounded = load("bounded_length")

MANIFEST = [
    dict(
        instance_id="A-g1-tp1", tp_degree=1, context_limit=32768, physical_gpu_ids=["2"]
    ),
    dict(
        instance_id="A-g3-tp1", tp_degree=1, context_limit=32768, physical_gpu_ids=["3"]
    ),
    dict(
        instance_id="B-g2-tp2",
        tp_degree=2,
        context_limit=32768,
        physical_gpu_ids=["0", "1"],
    ),
]


def row(request_id, remaining, tokens_s=20.0, action=True, transfer=10**8):
    return dict(
        request_id=request_id,
        generated=100,
        max_output_tokens=100 + remaining,
        remaining_bins=[(remaining, 1.0)],
        remaining_basis="HARD_CAP",
        tokens_s=tokens_s,
        transfer_bytes=transfer,
        action_available=action,
    )


def status(requests, *, target="B-g2-tp2", target_state="ACTIVE", sessions=0, age=0):
    instances = {
        "A-g1-tp1": dict(lifecycle="QUIESCING", sessions=sessions, requests=requests),
        "A-g3-tp1": dict(lifecycle="QUIESCING", sessions=0, requests=[]),
        "B-g2-tp2": dict(lifecycle=target_state, sessions=0, requests=[]),
    }
    return dict(
        unix_s=time.time() - age,
        instances=instances,
        release_budget_s=6.62,
        migration_target=target,
    )


def plan(mode="TP2", deadline_s=400):
    return dict(
        plan_id="manual-A",
        bundle_id="A",
        target_bundle_mode=mode,
        deadline=time.time() + deadline_s,
        refs=[
            dict(instance_id="A-g1-tp1", instance_generation=1),
            dict(instance_id="A-g3-tp1", instance_generation=3),
        ],
    )


class RemainingTests(unittest.TestCase):
    def test_hard_cap_without_a_prediction(self):
        bins, basis = pilot.remaining_bins(
            dict(generated=100, max_output_tokens=4096), None
        )
        self.assertEqual((bins, basis), ([(3996, 1.0)], "HARD_CAP"))

    def test_deterministic_fixed_output(self):
        value = bounded.BoundedLength(3996, (0.0,) * 3996 + (1.0,), (), True)
        bins, basis = pilot.remaining_bins(
            dict(generated=100, max_output_tokens=4096), NS(remaining=value)
        )
        self.assertEqual((bins, basis), ([(3996, 1.0)], "DETERMINISTIC"))

    def test_predicted_bins_never_understate_and_cover_all_mass(self):
        pmf = [0.0] * 1001
        for i in range(100, 600):
            pmf[i] = 1 / 500
        value = bounded.BoundedLength(1000, tuple(pmf), ())
        bins, basis = pilot.remaining_bins(
            dict(generated=0, max_output_tokens=1000), NS(remaining=value)
        )
        self.assertEqual(basis, "PREDICTED")
        self.assertAlmostEqual(sum(m for _, m in bins), 1.0, places=5)
        cumulative = 0.0
        for upper, mass in bins:
            cumulative += mass
            # Mass up to each edge never exceeds the true probability there.
            self.assertLessEqual(cumulative, sum(pmf[: upper + 1]) + 1e-6)
        self.assertLessEqual(bins[-1][0], 1000)

    def test_unknown_speed_keeps_time_unknown(self):
        q = cp.remaining_request(
            row("r", 200, tokens_s=None),
            migratable_target=True,
            release_budget_s=6.6,
            speed_margin=0.8,
        )
        self.assertIsNone(q.natural_s)
        self.assertIsNone(q.duration_pmf)

    def test_speed_margin_makes_time_conservative(self):
        q = cp.remaining_request(
            row("r", 200, tokens_s=20.0),
            migratable_target=True,
            release_budget_s=6.6,
            speed_margin=0.8,
        )
        self.assertAlmostEqual(q.natural_s, 200 / 16.0)
        self.assertTrue(q.migratable)
        self.assertEqual(q.transfer_s, 6.6)

    def test_migratable_needs_target_action_and_bytes(self):
        for kwargs, target in (
            (dict(action=False), True),
            (dict(transfer=None), True),
            ({}, False),
        ):
            q = cp.remaining_request(
                row("r", 200, **kwargs),
                migratable_target=target,
                release_budget_s=6.6,
                speed_margin=0.8,
            )
            self.assertFalse(q.migratable)
            self.assertIsNone(q.transfer_s)


class ProfileTests(unittest.TestCase):
    def test_frozen_profile_matches_only_its_platform_shape(self):
        profile = cp.load_profile()
        self.assertIsNone(cp.profile_mismatch(profile, MANIFEST[:2], "TP2"))
        self.assertIsNone(cp.profile_mismatch(profile, MANIFEST[2:], "TP1"))
        self.assertIn(
            "context",
            cp.profile_mismatch(
                profile, [dict(MANIFEST[0], context_limit=8192), MANIFEST[1]], "TP2"
            ),
        )
        self.assertIn("bundle", cp.profile_mismatch(profile, MANIFEST[:1], "TP2"))
        self.assertIn("TP4", cp.profile_mismatch(profile, MANIFEST[:2], "TP4"))


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.shadow = cp.ShadowPlanner(cp.load_profile(), MANIFEST)

    def run_shadow(self, requests, current=None, **kwargs):
        value = status(requests, **kwargs)
        return self.shadow(current or plan(), value, time.time())

    def test_long_request_is_worth_migrating_out(self):
        record = self.run_shadow([row("long", 4000), row("short", 50)])
        mechanism = record["mechanism"]
        self.assertEqual(mechanism["support_status"], "SUPPORTED")
        self.assertEqual(mechanism["decision"], "START_NEXT_MIGRATION")
        self.assertEqual(mechanism["ordered_migration_set"], ["long"])
        self.assertGreater(mechanism["predicted_clear_saving_s"], 200)
        self.assertEqual(record["drain_only"]["support_status"], "SUPPORTED")
        self.assertAlmostEqual(record["drain_only"]["estimated_clear_s"], 4000 / 16)

    def test_short_requests_are_left_to_finish(self):
        record = self.run_shadow([row("a", 40), row("b", 60)])
        self.assertEqual(record["mechanism"]["ordered_migration_set"], [])
        self.assertEqual(record["mechanism"]["decision"], "HOLD")

    def test_empty_bundle_begins_rebuild(self):
        record = self.run_shadow([])
        self.assertEqual(record["mechanism"]["decision"], "BEGIN_REBUILD")
        self.assertEqual(record["drain_only"]["estimated_clear_s"], 0)

    def test_net_utility_stays_unknown_without_matched_profiles(self):
        record = self.run_shadow([row("long", 4000)])
        self.assertEqual(record["net_utility"]["support_status"], "UNKNOWN")
        self.assertEqual(record["net_utility"]["decision"], "HOLD")

    def test_unknown_speed_is_unknown_not_guessed(self):
        record = self.run_shadow([row("r", 4000, tokens_s=None)])
        self.assertEqual(record["drain_only"]["support_status"], "UNKNOWN")
        self.assertEqual(record["mechanism"]["support_status"], "UNKNOWN")

    def test_stale_status_holds(self):
        record = self.run_shadow([row("long", 4000)], age=5)
        self.assertEqual(record["mechanism"]["decision"], "HOLD")
        self.assertIn("protection", record["mechanism"]["reason"])

    def test_active_session_holds_the_channel(self):
        record = self.run_shadow([row("long", 4000)], sessions=1)
        self.assertEqual(record["mechanism"]["decision"], "HOLD")

    def test_no_or_leaving_target_means_drain_only(self):
        for kwargs in (dict(target=None), dict(target_state="QUIESCING")):
            record = self.run_shadow([row("long", 4000)], **kwargs)
            self.assertIsNone(record["migration_target"])
            self.assertFalse(any(q["migratable"] for q in record["requests"]))
            self.assertEqual(record["mechanism"]["ordered_migration_set"], [])

    def test_split_is_reported_unsupported_with_a_drain_estimate(self):
        shadow = cp.ShadowPlanner(cp.load_profile(), MANIFEST)
        value = status([])
        value["instances"]["B-g2-tp2"]["requests"] = [row("x", 160)]
        current = dict(
            plan("TP1"),
            plan_id="manual-B",
            bundle_id="B",
            refs=[dict(instance_id="B-g2-tp2", instance_generation=2)],
        )
        record = shadow(current, value, value["unix_s"])
        self.assertEqual(record["mechanism"]["decision"], "UNSUPPORTED")
        self.assertAlmostEqual(record["drain_only"]["estimated_clear_s"], 10.0)
        stages = cp.load_profile()["stage_upper_s"]
        self.assertAlmostEqual(
            record["drain_only"]["estimated_ready_s"],
            10.0
            + stages["rebuild"]["TP1"]
            + stages["stop_to_released"]
            + stages["channel_retire"]
            + stages["controller_apply"],
        )

    def test_profile_mismatch_leaves_stage_time_unknown(self):
        manifest = [dict(r, context_limit=8192) for r in MANIFEST]
        shadow = cp.ShadowPlanner(cp.load_profile(), manifest)
        value = status([row("long", 4000)])
        record = shadow(plan(), value, value["unix_s"])
        self.assertIsNotNone(record["profile_mismatch"])
        self.assertEqual(record["mechanism"]["support_status"], "UNKNOWN")
        self.assertEqual(record["drain_only"]["support_status"], "UNKNOWN")

    def test_passed_deadline_aborts(self):
        record = self.run_shadow([row("long", 4000)], current=plan(deadline_s=-1))
        self.assertEqual(record["mechanism"]["decision"], "PLAN_ABORT")


class BackendShadowTests(unittest.TestCase):
    def test_throttled_and_a_planner_fault_never_stops_the_drain(self):
        from tests.bridge_tp.test_pressure_conversion import Harness

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = Harness(root, "A", "TP2")
            backend = harness.backend
            calls = []

            def failing(plan_, status_, now):
                calls.append(now)
                raise ValueError("planner fault")

            backend.shadow_planner = failing
            now = [time.time()]
            backend.clock = lambda: now[0]
            value = status([])
            for _ in range(3):
                backend.shadow(plan(), value)
            now[0] += 1.5
            backend.shadow(plan(), value)
            self.assertEqual(len(calls), 2)
            rows = [
                json.loads(line)
                for line in (root / "conversion_planner_shadow.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertIn("planner fault", rows[0]["error"])


if __name__ == "__main__":
    unittest.main()
