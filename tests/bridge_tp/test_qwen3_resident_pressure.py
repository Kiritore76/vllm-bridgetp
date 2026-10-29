# SPDX-License-Identifier: Apache-2.0
"""Resident/new-arrival separation, contemporaneous guard parity and KV promises."""

import unittest

from tests.bridge_tp.test_pressure_policy import config, pool
from tools.bridge_tp import qwen3_migration_matrix as matrix
from tools.bridge_tp import qwen3_resident_pressure as resident
from tools.bridge_tp import qwen3_stay_migrate as matched
from tools.bridge_tp.pressure_cpu import load

p = load("pressure_policy")


class ResidentPressureTests(unittest.TestCase):
    def setUp(self):
        self.cfg = resident.config(
            dict(requests={"2048": dict(prompt=list(range(2048)))})
        )

    def test_eight_plans_same_jobs_across_guards(self):
        self.assertEqual(len(self.cfg["recipes"]) * len(self.cfg["arms"]), 8)
        self.assertEqual(self.cfg["policy_overrides"], {})
        for family in ("resident_spare", "resident_busy"):
            recipes = [
                r for r in self.cfg["recipes"] if r["comparison_family"] == family
            ]
            self.assertEqual({r["guard_blocks"] for r in recipes}, {125, 384})
            self.assertEqual(recipes[0]["jobs"], recipes[1]["jobs"])
            self.assertEqual(recipes[0]["input_sha256"], recipes[1]["input_sha256"])
            for recipe in recipes:
                self.assertEqual(
                    recipe["policy_overrides"],
                    dict(capacity_guard_blocks=recipe["guard_blocks"]),
                )

    def test_initial_residents_fixed_later_arrivals_routable(self):
        for recipe in self.cfg["recipes"]:
            stress = [
                j for j in recipe["jobs"] if j["comparison_role"] == "source-stress"
            ]
            initial = [j for j in stress if j["at_s"] < 2]
            later = [j for j in stress if j["at_s"] >= 100]
            self.assertEqual(len(initial), 6)
            self.assertEqual(len(later), 12)
            self.assertTrue(all(j["route"] == "tp1" for j in initial))
            self.assertTrue(all("route" not in j for j in later))
            self.assertEqual(recipe["input_sha256"], matched.digest(recipe["jobs"]))

    def test_target_background_is_bounded_fixed_work_not_fake_overlap(self):
        total = 0
        requests = 0
        for recipe in self.cfg["recipes"]:
            background = [
                j for j in recipe["jobs"] if j["comparison_role"] == "target-background"
            ]
            busy = recipe["comparison_family"] == "resident_busy"
            self.assertEqual(len(background), 8 if busy else 3)
            self.assertEqual(recipe["require_target_background"], busy)
            self.assertEqual(recipe["minimum_matching_migrations"], 1)
            if busy:
                self.assertTrue(all(j["at_s"] < 2 for j in background))
                self.assertTrue(
                    all(j["request"]["max_tokens"] == 6144 for j in background)
                )
            for job in recipe["jobs"]:
                request = job["request"]
                self.assertTrue(request["ignore_eos"])
                self.assertLessEqual(
                    len(request["prompt"]) + request["max_tokens"], 8192
                )
                total += request["max_tokens"]
                requests += 1
        self.assertEqual(total * 2, 990720)
        self.assertEqual(requests * 2, 308)

    def test_new_arrival_migration_cannot_count_as_resident_coverage(self):
        from tests.bridge_tp.test_qwen3_migration_matrix import fixture

        recipe, raw, audit, score = fixture()
        recipe["matching_job_ids"] = ["another-resident"]
        self.assertFalse(
            matrix.assess(recipe, "migrate", raw, audit, score)[
                "mechanism_coverage_pass"
            ]
        )
        recipe["matching_job_ids"] = ["s"]
        self.assertTrue(
            matrix.assess(recipe, "migrate", raw, audit, score)[
                "mechanism_coverage_pass"
            ]
        )

    def test_migration_promises_never_credit_source_free_or_double_target_kv(self):
        ledger = p.ReservationLedger()
        source = pool(free_blocks=600, unallocated_prompt_blocks=96)
        target = pool("tp4", free_blocks=2000, unallocated_prompt_blocks=0)
        ledger.reserve(
            p.Reservation("m", "old", "tp4", 300, 0.1, 0, "migration"), target
        )
        ledger.track_ingress("new", "new", "tp1", 1536)
        self.assertEqual(ledger.routing_reading(source).free_blocks, 600)
        source_fit = p.prompt_fit(source, 1536, ledger, config(), 10)
        self.assertEqual(source_fit["available_blocks"], 408)
        self.assertEqual(
            p.prompt_fit(target, 1536, ledger, config(), 10)["available_blocks"], 1700
        )
        ledger.observe("m", allocated_blocks=200, service_observed=False)
        allocated = pool("tp4", free_blocks=1800, unallocated_prompt_blocks=0)
        self.assertEqual(
            p.prompt_fit(allocated, 1536, ledger, config(), 10)["available_blocks"],
            1700,
        )
        self.assertFalse(ledger.channel_available("tp4"))
