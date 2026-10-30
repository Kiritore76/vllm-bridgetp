# SPDX-License-Identifier: Apache-2.0
"""Unselected four-mode workloads, frozen inputs and actual CLI plan."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

from tools.bridge_tp import qwen3_pressure_four_mode as pressure
from tools.bridge_tp import qwen3_routine_four_mode as routine
from tools.bridge_tp import qwen3_stay_migrate as matched
from tools.bridge_tp import run_qwen3_stay_migrate as runner


class RoutineFourModeTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(
            requests={str(n): dict(prompt=list(range(n))) for n in (2048, 6144)}
        )

    def test_twelve_runs_fixed_work_no_forced_pool_or_policy_tuning(self):
        cfg = routine.config(self.inputs)
        self.assertEqual(cfg, routine.config(self.inputs))
        self.assertEqual(cfg["arms"], ["weighted", "least", "stay", "migrate"])
        self.assertEqual(len(cfg["recipes"]) * len(cfg["arms"]), 12)
        reference = pressure.updated_config(self.inputs)
        for key in (
            "policy_overrides",
            "urgent_target_admission",
            "routing_weights",
            "horizon_s",
            "maximum_gap_s",
            "maximum_migrations",
        ):
            self.assertEqual(cfg[key], reference[key])
        self.assertEqual([len(r["jobs"]) for r in cfg["recipes"]], [75, 75, 84])
        self.assertEqual(
            [r["planned_output_tokens"] for r in cfg["recipes"]],
            [124800, 186240, 92160],
        )
        for recipe in cfg["recipes"]:
            self.assertEqual(recipe["input_sha256"], matched.digest(recipe["jobs"]))
            self.assertEqual(
                len({j["id"] for j in recipe["jobs"]}), len(recipe["jobs"])
            )
            self.assertEqual(
                [j["at_s"] for j in recipe["jobs"]],
                sorted(j["at_s"] for j in recipe["jobs"]),
            )
            for job in recipe["jobs"]:
                self.assertIsNone(job["route"])
                req = job["request"]
                self.assertLessEqual(len(req["prompt"]) + req["max_tokens"], 8192)
                self.assertLessEqual(len(req["prompt"]), 2048)
                self.assertTrue(req["ignore_eos"])
                self.assertLess(job["at_s"], 300)

    def test_transient_burst_returns_to_background_without_pressure_search(self):
        jobs = routine.config(self.inputs)["recipes"][2]["jobs"]
        long = [j for j in jobs if j["workload_role"] == "long_decode"]
        self.assertEqual(len(long), 12)
        self.assertEqual([j["at_s"] for j in long], [100 + i * 1.5 for i in range(12)])
        self.assertEqual(sum(j["workload_role"] == "background" for j in jobs), 60)
        self.assertTrue(any(j["at_s"] > 280 for j in jobs))
        self.assertTrue(all(j["request"]["max_tokens"] == 4096 for j in long))

    def test_cli_plans_twelve_without_old_results_or_starting_servers(self):
        out = io.StringIO()
        with (
            contextlib.redirect_stdout(out),
            patch.object(runner, "serving_preflight") as serving,
        ):
            code = runner.main(
                [
                    "--routine-four-mode",
                    "--current-machine",
                    "--repetitions",
                    "1",
                    "--model",
                    "unused",
                    "--output",
                    "unused",
                    "--expected-revision",
                    "CPU",
                    "--validate-only",
                ]
            )
        self.assertEqual(code, 0)
        serving.assert_not_called()
        plan = json.loads(out.getvalue())
        self.assertEqual(plan["runs"], 12)
        self.assertEqual(len(plan["recipes"]), 3)
        self.assertEqual(
            plan["environment_mode"], "CURRENT_MACHINE_NO_PRIOR_COMPARISON"
        )
        self.assertTrue(plan["policy_overrides"]["kv_routing_priority"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(
                [
                    "--routine-four-mode",
                    "--updated-pressure-four-mode",
                    "--current-machine",
                    "--model",
                    "unused",
                    "--output",
                    "unused",
                    "--expected-revision",
                    "CPU",
                    "--validate-only",
                ]
            )
