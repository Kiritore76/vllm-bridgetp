# SPDX-License-Identifier: Apache-2.0
"""Guard screening must preserve workload parity, default policy and routing."""

import unittest
from types import SimpleNamespace

from tools.bridge_tp import qwen3_kv_guard_trial as trial
from tools.bridge_tp import qwen3_migration_matrix as matrix
from tools.bridge_tp import qwen3_stay_migrate as matched


class GuardTrialTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(requests={"2048": dict(prompt=list(range(2048)))})
        self.cfg = trial.config(self.inputs)

    def test_six_runs_only_experimental_guard_changes(self):
        original = matrix.config(self.inputs)
        self.assertEqual(original["policy_overrides"], {})
        self.assertEqual(self.cfg["policy_overrides"], {"capacity_guard_blocks": 384})
        self.assertEqual(len(self.cfg["recipes"]) * len(self.cfg["arms"]), 6)
        self.assertEqual(self.cfg["arms"], ["stay", "migrate"])
        self.assertEqual(self.cfg["maximum_migrations"], 0)
        self.assertEqual(self.cfg, trial.config(self.inputs))

    def test_burst_is_original_seed_one_workload(self):
        original = matrix.config(self.inputs)["recipes"][0]
        burst = self.cfg["recipes"][1]
        self.assertEqual(burst["jobs"], original["jobs"])
        self.assertEqual(burst["input_sha256"], original["input_sha256"])

    def test_growth_excludes_later_source_waves_not_probes(self):
        growth = self.cfg["recipes"][0]
        self.assertEqual(len(growth["jobs"]), 22)
        source = [j for j in growth["jobs"] if j["comparison_role"] == "source-stress"]
        self.assertEqual(len(source), 6)
        self.assertTrue(all(j["at_s"] < 2 for j in source))
        self.assertEqual(
            sum(j["comparison_role"] == "short-probe" for j in growth["jobs"]), 15
        )

    def test_routed_pair_changes_only_source_initial_route(self):
        burst, routed = self.cfg["recipes"][1:]
        for fixed, dynamic in zip(burst["jobs"], routed["jobs"]):
            if fixed["comparison_role"] == "source-stress":
                self.assertNotIn("route", dynamic)
                self.assertEqual(
                    dynamic, {k: v for k, v in fixed.items() if k != "route"}
                )
            else:
                self.assertEqual(fixed, dynamic)
        self.assertEqual(routed["minimum_matching_migrations"], 0)
        self.assertEqual(burst["minimum_matching_migrations"], 2)

    def test_fixed_contract_contexts_hashes_and_full_work(self):
        total = 0
        for recipe in self.cfg["recipes"]:
            self.assertEqual(recipe["input_sha256"], matched.digest(recipe["jobs"]))
            for job in recipe["jobs"]:
                request = job["request"]
                self.assertTrue(request["ignore_eos"])
                self.assertLessEqual(
                    len(request["prompt"]) + request["max_tokens"], 8192
                )
                total += request["max_tokens"]
        self.assertEqual(total * 2, 493184)

    def test_controller_preflight_guard_scope_and_physical_geometry(self):
        from tools.bridge_tp import run_pressure_balanced_pilot as pilot

        cfg = SimpleNamespace(capacity_guard_blocks=384, health_service=True)
        controller = SimpleNamespace(
            target_tp_size=2,
            block_size=16,
            tp1_total_kv_blocks=2326,
            tp4_total_kv_blocks=11713,
            policy=SimpleNamespace(kv_bytes_per_token=147456),
        )
        pilot.validate_guard_geometry(cfg, controller, "qwen3")
        for platform in ("a100", "unknown"):
            with self.assertRaises(ValueError):
                pilot.validate_guard_geometry(cfg, controller, platform)
        controller.target_tp_size = 4
        with self.assertRaises(ValueError):
            pilot.validate_guard_geometry(cfg, controller, "qwen3")
        controller.target_tp_size = 2
        cfg.capacity_guard_blocks = 512
        with self.assertRaises(ValueError):
            pilot.validate_guard_geometry(cfg, controller, "qwen3")
        cfg.capacity_guard_blocks = 384
        controller.tp1_total_kv_blocks = 384
        with self.assertRaises(ValueError):
            pilot.validate_guard_geometry(cfg, controller, "qwen3")
        controller.tp1_total_kv_blocks = 2326
        controller.block_size = 32
        with self.assertRaises(ValueError):
            pilot.validate_guard_geometry(cfg, controller, "qwen3")
        controller.block_size = 16
        cfg.capacity_guard_blocks = 125
        pilot.validate_guard_geometry(cfg, controller, "a100")
