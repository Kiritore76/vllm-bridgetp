# SPDX-License-Identifier: Apache-2.0
"""Autonomous routing, workload parity and separate pressure-family scoring."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.bridge_tp import qwen3_four_mode as full
from tools.bridge_tp import qwen3_kv_priority_verify as priority
from tools.bridge_tp import qwen3_pressure_four_mode as pressure
from tools.bridge_tp import qwen3_stay_migrate as matched
from tools.bridge_tp import run_pressure_comparison as comparison
from tools.bridge_tp import run_qwen3_stay_migrate as runner


class PressureFourModeTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(
            requests={str(n): dict(prompt=list(range(n))) for n in (2048, 6144)}
        )
        self.cfg = pressure.config(self.inputs)

    def test_updated_eight_runs_preserve_exact_old_arrivals(self):
        cfg = pressure.updated_config(self.inputs)
        self.assertEqual(cfg["recipes"], self.cfg["recipes"][1:])
        self.assertEqual(len(cfg["recipes"]) * len(cfg["arms"]), 8)
        self.assertTrue(cfg["policy_overrides"]["kv_routing_priority"])
        self.assertTrue(cfg["policy_overrides"]["kv_emergency_priority"])
        self.assertEqual(cfg["policy_overrides"]["capacity_guard_blocks"], 125)

    def test_updated_cli_has_eight_current_machine_runs_without_prior(self):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch.object(runner, "serving_preflight") as serving,
        ):
            code = runner.main(
                [
                    "--updated-pressure-four-mode",
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
        serving.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(plan["runs"], 8)
        self.assertEqual(plan["arms"], full.ARMS)
        self.assertEqual(
            plan["environment_mode"], "CURRENT_MACHINE_NO_PRIOR_COMPARISON"
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(
                [
                    "--updated-pressure-four-mode",
                    "--current-machine",
                    "--kv-priority-verify",
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

    def test_updated_real_plan_builder_keeps_serving_equal_and_migration_opt_in(self):
        def scene(spec, row, root, common, settings):
            return dict(
                run_dir=root / row["name"] / "controller",
                controller_argv=["python", "controller"],
                source_argv=["unchanged-source"],
                target_argv=["serve", "--kv-transfer-config", "{}"],
                source_env={},
                target_env={},
            )

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(comparison.pilot, "read", return_value={}),
            patch.object(comparison.pilot, "build_scene", side_effect=scene),
        ):
            plans = comparison.build_plans(
                dict(policy_config="unused"),
                pressure.updated_config(self.inputs),
                self.inputs,
                Path(temp),
                Mock(),
                None,
            )
        self.assertEqual(len(plans), 8)
        for _, arm, _, plan in plans:
            with (
                patch.object(runner, "adjust_migration_plan"),
                patch.object(runner, "configure_predictor"),
            ):
                runner.adapt_plan(plan, arm, Path("checkpoint"), Path("model"))
            priority.enable_urgent_target(plan)
            self.assertEqual(
                "--no-migration" in plan["controller_argv"], arm != "migrate"
            )
            self.assertEqual(
                "--qwen3-execute-migrations" in plan["controller_argv"],
                arm == "migrate",
            )
        self.assertTrue(
            all(
                p[3]["source_argv"] == plans[0][3]["source_argv"]
                and p[3]["target_argv"] == plans[0][3]["target_argv"]
                for p in plans
            )
        )

    def test_twelve_runs_no_forced_pool_and_common_context(self):
        self.assertEqual(self.cfg["arms"], full.ARMS)
        self.assertEqual(len(self.cfg["recipes"]) * len(full.ARMS), 12)
        self.assertEqual(self.cfg["policy_overrides"], dict(capacity_guard_blocks=125))
        for recipe in self.cfg["recipes"]:
            self.assertEqual(recipe["input_sha256"], matched.digest(recipe["jobs"]))
            self.assertTrue(all(j["route"] is None for j in recipe["jobs"]))
            for job in recipe["jobs"]:
                request = job["request"]
                self.assertTrue(request["ignore_eos"])
                self.assertLessEqual(
                    len(request["prompt"]) + request["max_tokens"], 8192
                )
                self.assertLess(job["at_s"], 300)

    def test_frozen_counts_costs_seed_and_mode_order_rotation(self):
        self.assertEqual(self.cfg, pressure.config(self.inputs))
        self.assertEqual([len(r["jobs"]) for r in self.cfg["recipes"]], [75, 168, 375])
        self.assertEqual(
            [r["planned_output_tokens"] for r in self.cfg["recipes"]],
            [309120, 224256, 201600],
        )
        self.assertEqual(
            [r["arm_order_offset"] for r in self.cfg["recipes"]], [0, 1, 2]
        )

    def test_long_tasks_arrive_naturally_before_short_burst(self):
        recipe = self.cfg["recipes"][1]
        phases = {
            name: [j for j in recipe["jobs"] if j["workload_phase"] == name]
            for name in ("long_accumulation", "short_burst", "recovery")
        }
        self.assertEqual([len(v) for v in phases.values()], [36, 120, 12])
        self.assertLess(max(j["at_s"] for j in phases["long_accumulation"]), 100)
        self.assertGreaterEqual(min(j["at_s"] for j in phases["short_burst"]), 100)
        self.assertTrue(all(j["route"] is None for j in recipe["jobs"]))

    def test_role_metrics_and_families_do_not_pool_different_pressures(self):
        data = [
            dict(
                job_id="probe",
                initial_pool="tp1",
                source_error=None,
                request=dict(prompt=[1], max_tokens=2),
                source_transport=dict(request_started_unix_s=0.0),
                response=dict(
                    committed=False,
                    emitted=[
                        dict(index=0, token_id=1, unix_s=0.1, origin="source"),
                        dict(index=1, token_id=2, unix_s=0.12, origin="source"),
                    ],
                ),
            )
        ]
        score = full.full_score(data, self.cfg, 0.0, [])
        recipe = dict(
            jobs=[dict(id="probe", workload_role="short_probe", workload_phase="burst")]
        )
        decorated = pressure.decorate(score, recipe)
        self.assertEqual(decorated["workload_role_groups"]["short_probe"]["passed"], 1)
        a = dict(scenario="a", arm="stay", score=decorated)
        b = dict(copy.deepcopy(a), scenario="b")
        result = pressure.aggregate([a, b])
        self.assertEqual(result["a"]["stay"]["denominator_s"], 300)
        self.assertEqual(result["b"]["stay"]["all_requests"]["requests"], 1)
