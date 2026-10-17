# SPDX-License-Identifier: Apache-2.0
"""Validate matched inputs, disabled actions and late-arrival classification."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.bridge_tp.test_pressure_policy import candidate, config, pool, r
from tools.bridge_tp import run_pressure_comparison as compare

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "experiments/phase9/pressure_balanced_v2"


class ComparisonTests(unittest.TestCase):
    def test_migrate_retest_uses_exact_r1_jobs_and_has_no_paired_benefit(self):
        full = compare.pool_comparison_config(BUNDLE)
        selected = compare.pool_comparison_config(BUNDLE, migrate_only=True)
        self.assertEqual(selected["recipes"], full["recipes"][:1])
        self.assertEqual(selected["arms"], ["migrate"])
        written = []
        common = SimpleNamespace(write_json=lambda path, value: written.append(value))
        for valid in (True, False):
            rows = [
                dict(arm="migrate", pair="pool-mixed-r1", fixed_arrivals_valid=valid)
            ]
            rc = compare.pool_acceptance(Path("unused"), rows, selected, common)
            self.assertEqual(rc, 0 if valid else 1)
            result = written[-1]["repetitions"][0]
            self.assertFalse(result["comparison_available"])
            self.assertNotIn("migration_delta_G", result)

    def test_r1_retest_preserves_all_four_arms_and_exact_first_seed_inputs(self):
        full = compare.pool_comparison_config(BUNDLE)
        selected = compare.pool_comparison_config(BUNDLE, first_repetition_only=True)
        expected = copy.deepcopy(full)
        expected["recipes"] = full["recipes"][:1]
        self.assertEqual(selected, expected)
        self.assertEqual(selected["arms"], ["weighted", "least", "stay", "migrate"])
        self.assertEqual(len(selected["recipes"][0]["jobs"]), 75)

    def setUp(self):
        self.config = compare.pilot.read(BUNDLE / "comparison.json")
        self.inputs = compare.pilot.read(BUNDLE / "calibration.json")

    def test_frozen_inputs_preserve_measured_shapes_and_sampling(self):
        compare.validate(self.config, self.inputs)
        for recipe in self.config["recipes"]:
            jobs = compare.arrivals(self.inputs, recipe)
            self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
            for job in jobs:
                req = job["request"]
                self.assertTrue(req["ignore_eos"])
                self.assertEqual(req["seed"], 20261015)
                if "background" not in job["id"]:
                    self.assertEqual(job["route"], "tp1")
                    self.assertEqual(req["max_tokens"], 16)
                    self.assertEqual(
                        len(req["prompt"]), 2048 if recipe["kind"] == "pulse" else 6144
                    )

    def test_only_sustain_can_be_changed(self):
        self.config["policy_overrides"]["spare_ratio"] = 0.6
        with self.assertRaises(ValueError):
            compare.validate(self.config, self.inputs)

    def test_all_frozen_jobs_pass_actual_controller_preflight(self):
        worker = compare.pilot.controller
        sampling = worker.load("sampling_contract")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = (
                compare.pilot.read(BUNDLE / "policy.json")
                | self.config["policy_overrides"]
            )
            (root / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
            for recipe in self.config["recipes"]:
                jobs = compare.arrivals(self.inputs, recipe)
                for job in jobs:
                    sampling.freeze_strict_greedy_sampling(job["request"])
                (root / "arrivals.json").write_text(json.dumps(jobs), encoding="utf-8")
                args = worker.parse_args(
                    [
                        "--policy",
                        "pressure-balanced-v1",
                        "--policy-config",
                        str(root / "policy.json"),
                        "--controller-config",
                        str(BUNDLE / "controller_template.json"),
                        "--service-evidence",
                        str(BUNDLE / "service_evidence.json"),
                        "--arrivals",
                        str(root / "arrivals.json"),
                        "--run-dir",
                        str(root),
                        "--source-inventory",
                        str(root / "source.jsonl"),
                        "--target-inventory",
                        str(root / "target.jsonl"),
                        "--predictor-events",
                        str(root / "predictor.jsonl"),
                        "--predictor-sha256",
                        "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7",
                        "--migration-id",
                        recipe["name"],
                        "--no-migration",
                    ]
                )
                if policy.get("health_service"):
                    args.health_reference = BUNDLE / "health_reference_v2.json"
                prepared = worker.preflight(args)
                self.assertEqual(prepared[-1]["jobs"], len(jobs))
                self.assertFalse(prepared[-1]["migration_execution_enabled"])

    def test_pair_has_identical_arrival_file_and_counterbalanced_order(self):
        def build(spec, scene, root, *_):
            return dict(
                run_dir=root / scene["name"] / "controller",
                controller_argv=[
                    "python",
                    "controller",
                    "--arrivals",
                    scene["arrivals"],
                ],
                source_env={},
                target_env={},
            )

        def write(path, data):
            path.write_text(json.dumps(data), encoding="utf-8")

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(compare.pilot, "build_scene", side_effect=build),
        ):
            plans = compare.build_plans(
                {"policy_config": str(BUNDLE / "policy.json")},
                self.config,
                self.inputs,
                Path(temp),
                SimpleNamespace(write_json=write),
                None,
            )
            self.assertEqual(len(plans), 8)
            for i in range(0, 8, 2):
                recipe, _, first_scene, _ = plans[i]
                self.assertEqual(first_scene["arrivals"], plans[i + 1][2]["arrivals"])
                self.assertEqual(
                    [v[1] for v in plans[i : i + 2]],
                    ["stay", "migrate"]
                    if recipe["repetition"] == 1
                    else ["migrate", "stay"],
                )
            for _, arm, _, plan in plans:
                self.assertEqual(
                    "--no-migration" in plan["controller_argv"], arm == "stay"
                )
            policy = compare.pilot.read(Path(temp) / "comparison_policy.json")
            original = compare.pilot.read(BUNDLE / "policy.json")
            self.assertEqual(policy, original | {"sustain_s": 1.5})

    def test_advisory_candidate_never_reserves_or_starts(self):
        coordinator = r.PressureCoordinator(config(sustain_s=0), lambda _: None)
        decision = coordinator.tick(
            10, pool(decode_ratio=2), pool("tp4"), [candidate()], actions=None
        )
        self.assertEqual(decision.action, "START_SHADOW")
        self.assertEqual(coordinator.ledger.snapshot(), [])
        self.assertEqual(coordinator._sessions, {})

    def fixture(self):
        jobs = [dict(id="background", route="tp1", at_s=1, request=dict(max_tokens=3))]
        rows = [
            dict(
                job_id="background",
                initial_pool="tp1",
                source_error=None,
                source_transport=dict(request_started_unix_s=101.1),
                response=dict(
                    committed=True,
                    handoff_stall_s=0.2,
                    emitted=[
                        dict(index=i, unix_s=t)
                        for i, t in enumerate((101.2, 101.4, 101.45))
                    ],
                ),
            )
        ]
        return rows, jobs

    def test_handoff_counts_in_visible_slo(self):
        rows, jobs = self.fixture()
        metric = compare.result_metrics(rows, jobs, 100, 0.5)[0]
        self.assertEqual(metric["slow_intervals"], 1)
        self.assertEqual(metric["slow_fraction"], 0.5)
        self.assertTrue(metric["arrival_valid"])
        self.assertAlmostEqual(metric["ttft_s"], 0.1)

    def test_late_arrival_is_not_accepted_as_matched(self):
        rows, jobs = self.fixture()
        rows[0]["source_transport"]["request_started_unix_s"] = 101.8
        metric = compare.result_metrics(rows, jobs, 100, 0.5)[0]
        self.assertFalse(metric["arrival_valid"])

    def test_one_second_dispatch_budget_is_inclusive_and_preserves_lag(self):
        budget = compare.pool_comparison_config(BUNDLE)["max_dispatch_lag_s"]
        self.assertEqual(budget, 1.0)
        for lag, valid in ((0.557625, True), (1.0, True), (1.001, False)):
            with self.subTest(lag=lag):
                rows, jobs = self.fixture()
                rows[0]["source_transport"]["request_started_unix_s"] = 101 + lag
                metric = compare.result_metrics(rows, jobs, 100, budget)[0]
                self.assertEqual(metric["arrival_valid"], valid)
                self.assertAlmostEqual(metric["dispatch_lag_s"], lag)

    def test_short_output_or_noncontiguous_proxy_is_failure(self):
        rows, jobs = self.fixture()
        for bad in (rows[:-1], copy.deepcopy(rows), copy.deepcopy(rows)):
            if bad:
                bad[0]["response"]["emitted"][-1]["index"] = 9
            with self.assertRaises(RuntimeError):
                compare.result_metrics(bad, jobs, 100, 0.5)
        rows[0]["response"]["emitted"].pop()
        with self.assertRaises(RuntimeError):
            compare.result_metrics(rows, jobs, 100, 0.5)


class PoolComparisonTests(unittest.TestCase):
    def test_pair_summary_keeps_output_differences_and_late_arrivals(self):
        cfg = compare.pool_comparison_config(BUNDLE, pair_only=True)
        results = []
        for recipe in cfg["recipes"]:
            for arm in cfg["arms"]:
                results.append(
                    dict(
                        pair=recipe["name"],
                        arm=arm,
                        G300=10 if arm == "stay" else 11,
                        fixed_arrivals_valid=recipe["repetition"] != 2,
                        token_hashes={"r": arm},
                        full_output_tokens=100,
                    )
                )
        captured = []
        common = SimpleNamespace(write_json=lambda path, data: captured.append(data))
        self.assertEqual(compare.pool_acceptance(Path("."), results, cfg, common), 1)
        self.assertEqual(len(captured[0]["repetitions"]), 3)
        self.assertFalse(
            captured[0]["repetitions"][0]["identical_output_for_migration_pair"]
        )

    def test_reused_natural_workload_and_measured_weights_are_frozen(self):
        cfg = compare.pool_comparison_config(BUNDLE)
        self.assertEqual(cfg["arms"], ["weighted", "least", "stay", "migrate"])
        self.assertEqual(cfg, compare.pool_comparison_config(BUNDLE))
        self.assertAlmostEqual(cfg["routing_weights"][0], 231.0625)
        self.assertAlmostEqual(cfg["routing_weights"][1], 597.4625)
        for recipe in cfg["recipes"]:
            jobs = compare.arrivals({}, recipe)
            self.assertEqual(len(jobs), 75)
            self.assertEqual(jobs[-1]["at_s"], 296)
            self.assertTrue(all(j["route"] is None for j in jobs))
            self.assertTrue(all(not j["request"]["ignore_eos"] for j in jobs))

    def test_all_strategies_have_same_jobs_and_only_full_arm_migrates(self):
        def build(spec, scene, root, *_):
            return dict(
                run_dir=root / scene["name"] / "controller",
                controller_argv=["python", "controller"],
                source_env={},
                target_env={},
            )

        def write(path, data):
            path.write_text(json.dumps(data), encoding="utf-8")

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(compare.pilot, "build_scene", side_effect=build),
        ):
            plans = compare.build_plans(
                dict(policy_config=str(BUNDLE / "policy.json")),
                compare.pool_comparison_config(BUNDLE),
                {},
                Path(temp),
                SimpleNamespace(write_json=write),
                None,
            )
            self.assertEqual(len(plans), 12)
            self.assertEqual(len({scene["arrivals"] for _, _, scene, _ in plans}), 3)
            for _, arm, _, plan in plans:
                argv = plan["controller_argv"]
                self.assertEqual("--no-migration" in argv, arm != "migrate")
                self.assertEqual(
                    argv[argv.index("--routing-strategy") + 1],
                    {"weighted": "weighted-round-robin", "least": "least-inflight"}.get(
                        arm, "pressure"
                    ),
                )

    def test_weighted_baseline_does_not_use_health_and_least_counts_outstanding(self):
        co = r.PressureCoordinator(
            config(),
            lambda _: None,
            routing_strategy="weighted-round-robin",
            routing_weights=(1, 2),
        )
        counts = dict(tp1=0, tp4=0)
        for i in range(30):
            result = co.admission(
                str(i), 1, 10, (pool(), pool("tp4")), now=10, key=str(i)
            )
            counts[result.pool_id] += 1
        self.assertEqual(counts, dict(tp1=10, tp4=20))
        # Historical arrivals do not affect least-inflight after completion.
        for entry in co.ledger.entries.values():
            entry.state = "COMPLETED"
        co.routing_strategy = "least-inflight"
        result = co.admission("next", 1, 10, (pool(), pool("tp4")), now=10, key="next")
        self.assertEqual(result.pool_id, "tp4")

    def test_capacity_fallback_is_shared_with_baselines(self):
        for strategy in ("weighted-round-robin", "least-inflight"):
            co = r.PressureCoordinator(
                config(), lambda _: None, routing_strategy=strategy
            )
            result = co.admission(
                "r",
                2048,
                10,
                (pool(free_blocks=0), pool("tp4", unallocated_prompt_blocks=0)),
                now=10,
                key="k",
            )
            self.assertEqual(result.pool_id, "tp4")
            self.assertEqual(result.reservation_basis, "none")

    def test_natural_EOS_is_valid_but_truncation_abort_and_bad_indices_are_not(self):
        rows, jobs = ComparisonTests().fixture()
        row, job = rows[0], jobs[0]
        row["source_done"] = True
        row["source_transport"]["finish_reason"] = "stop"
        row["response"]["committed"] = False
        row["response"]["emitted"].pop()
        job["route"] = None
        row["request"] = job["request"]
        self.assertEqual(
            compare.result_metrics(rows, jobs, 100, 0.5, True)[0]["output_tokens"], 2
        )
        cfg = dict(
            horizon_s=300,
            metric=compare.sustained.METRIC,
            itl_s=0.1,
            slow_fraction=0.01,
            maximum_gap_s=1,
            natural_eos=True,
            origin_unix_s=100,
        )
        self.assertEqual(compare.sustained.score(rows, cfg)["raw_window_tokens"], 2)
        for reason in (None, "abort", "length", "error"):
            row["source_transport"]["finish_reason"] = reason
            with self.assertRaises(RuntimeError):
                compare.result_metrics(rows, jobs, 100, 0.5, True)
        row["source_transport"]["finish_reason"] = "stop"
        row["response"]["emitted"][1]["index"] = 5
        with self.assertRaises(RuntimeError):
            compare.result_metrics(rows, jobs, 100, 0.5, True)

    def test_actual_controller_preflight_for_all_four_arms(self):
        cfg = compare.pool_comparison_config(BUNDLE)
        worker = compare.pilot.controller
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, value in (
                ("policy", cfg["policy_overrides"]),
                ("arrivals", cfg["recipes"][0]["jobs"]),
            ):
                (root / (name + ".json")).write_text(json.dumps(value))
            for arm in cfg["arms"]:
                argv = [
                    "--policy",
                    "pressure-balanced-v1",
                    "--policy-config",
                    str(root / "policy.json"),
                    "--controller-config",
                    str(BUNDLE / "controller_template.json"),
                    "--service-evidence",
                    str(BUNDLE / "service_evidence.json"),
                    "--health-reference",
                    str(BUNDLE / "health_reference_v2.json"),
                    "--arrivals",
                    str(root / "arrivals.json"),
                    "--run-dir",
                    str(root),
                    "--source-inventory",
                    str(root / "source.jsonl"),
                    "--target-inventory",
                    str(root / "target.jsonl"),
                    "--predictor-events",
                    str(root / "predictor.jsonl"),
                    "--predictor-sha256",
                    "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7",
                    "--migration-id",
                    arm,
                    "--maximum-migrations",
                    "0",
                    "--routing-strategy",
                    {"weighted": "weighted-round-robin", "least": "least-inflight"}.get(
                        arm, "pressure"
                    ),
                    "--routing-weights",
                    *map(str, cfg["routing_weights"]),
                ]
                if arm != "migrate":
                    argv.append("--no-migration")
                args = worker.parse_args(argv)
                self.assertEqual(worker.preflight(args)[-1]["jobs"], 75)
                if arm in {"weighted", "least"}:
                    args.no_migration = False
                    with self.assertRaises(ValueError):
                        worker.preflight(args)


if __name__ == "__main__":
    unittest.main()
