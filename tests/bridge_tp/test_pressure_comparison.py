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
            policy = compare.pilot.read(BUNDLE / "policy.json") | {"sustain_s": 1.5}
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


if __name__ == "__main__":
    unittest.main()
