# SPDX-License-Identifier: Apache-2.0
"""Frozen measured weights, four actual strategies and full-window accounting."""

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

from tools.bridge_tp import qwen3_four_mode as four
from tools.bridge_tp import qwen3_pressure_observation as obs
from tools.bridge_tp import run_pressure_comparison as comparison
from tools.bridge_tp import run_qwen3_stay_migrate as runner


def config():
    templates = [
        dict(
            scenario="natural_long_pressure",
            jobs=[dict(request=dict(prompt=[1] * 34, max_tokens=1600))],
        ),
        dict(
            scenario="short_burst_recovery",
            jobs=[dict(request=dict(prompt=[1] * 512, max_tokens=128))],
        ),
    ]
    inputs = dict(requests={str(n): dict(prompt=[1] * n) for n in (2048, 6144)})
    with patch.object(four.matched, "recipes", return_value=templates):
        return four.config(
            Path("prior"), inputs, obs.read(obs.ASSETS / "health_reference.json")
        )


class FourModeTests(unittest.TestCase):
    def prior(self, root):
        cfg = config()

        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")

        write(root / "comparison_recipe.json", cfg)
        write(root / "comparison_policy.json", obs.read(obs.ASSETS / "policy.json"))
        write(
            root / "identity.json",
            dict(checkpoint_sha256=runner.CHECKPOINT_SHA, revision="original"),
        )
        runs = []
        for rec in cfg["recipes"]:
            write(root / (rec["name"] + "_arrivals.json"), rec["jobs"])
            for arm in four.ARMS:
                if rec["repetition"] == 3 and arm in ("weighted", "least"):
                    continue
                name = rec["name"] + "-" + arm
                runs.append(
                    dict(
                        name=name,
                        arm=arm,
                        pair=rec["name"],
                        status="COMPLETED",
                        summary=dict(window_soft_goodoutput_s=1.0),
                    )
                )
                write(
                    root / name / "owned_shutdown_receipt.json",
                    dict(all_owned_groups_ended=True),
                )
                write(
                    root / name / "sustained_goodoutput.json",
                    dict(goodoutput_tokens_s=1.0),
                )
        failed = "four-mixed-r3-weighted"
        runs.append(dict(name=failed, status="FAILED"))
        write(
            root / failed / "owned_shutdown_receipt.json",
            dict(all_owned_groups_ended=True),
        )
        write(
            root / "result.json",
            dict(status="TECHNICAL_FAILURE", total_profile_attempts=60, runs=runs),
        )
        return cfg

    def test_tracked_assets_match_original_checkpoint_and_input_identity(self):
        from tools.bridge_tp.qwen3_calibration_contract import INPUT_SHA

        self.assertEqual(
            hashlib.sha256(runner.CHECKPOINT_PATH.read_bytes()).hexdigest(),
            runner.CHECKPOINT_SHA,
        )
        self.assertEqual(
            hashlib.sha256((obs.ASSETS / "inputs.json").read_bytes()).hexdigest(),
            INPUT_SHA,
        )

    def test_supplement_selects_two_missing_cells_and_keeps_original_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prior(root)
            before = (root / "result.json").read_bytes()
            cfg, inherited, scores, missing, attempts = four.supplement(root)
            self.assertEqual(len(inherited), 10)
            self.assertEqual(len(scores), 10)
            self.assertEqual(missing, {"four-mixed-r3-weighted", "four-mixed-r3-least"})
            self.assertEqual(attempts, 60)
            self.assertEqual((root / "result.json").read_bytes(), before)

    def test_supplement_rejects_modified_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.prior(root)
            cfg["recipes"][0]["jobs"][0]["request"]["max_tokens"] += 1
            (root / "comparison_recipe.json").write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                four.supplement(root)

    def test_supplement_rejects_unconfirmed_old_process_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prior(root)
            (root / "four-mixed-r3-weighted/owned_shutdown_receipt.json").write_text(
                json.dumps(dict(all_owned_groups_ended=False))
            )
            with self.assertRaises(ValueError):
                four.supplement(root)

    def test_supplement_rejects_changed_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prior(root)
            (root / "comparison_policy.json").write_text("{}")
            with self.assertRaises(ValueError):
                four.supplement(root)

    def test_measured_weights_use_common_repeated_peaks(self):
        weights, cells = four.measured_weights(
            obs.read(obs.ASSETS / "health_reference.json")
        )
        self.assertEqual(weights, [176.8, 740.6375])
        self.assertEqual(len(cells), 4)
        self.assertTrue(all(c["runs"] >= 2 for c in cells))

    def test_wrong_geometry_or_single_run_weights_rejected(self):
        ref = obs.read(obs.ASSETS / "health_reference.json")
        ref["target_tp_size"] = 4
        with self.assertRaises(ValueError):
            four.measured_weights(ref)
        ref["target_tp_size"] = 2
        for cell in ref["cells"]:
            cell["runs"] = 1
        with self.assertRaises(ValueError):
            four.measured_weights(ref)

    def test_three_seed_sustained_fixed_work_and_no_forced_routes(self):
        cfg = config()
        self.assertEqual(cfg, config())
        self.assertEqual(cfg["arms"], four.ARMS)
        self.assertEqual(len(cfg["recipes"]), 3)
        self.assertEqual(len({r["input_sha256"] for r in cfg["recipes"]}), 3)
        for recipe in cfg["recipes"]:
            jobs = recipe["jobs"]
            self.assertEqual(len(jobs), 75)
            self.assertEqual([j["at_s"] for j in jobs], list(range(0, 300, 4)))
            self.assertEqual(
                Counter(j["template"] for j in jobs), dict.fromkeys(range(5), 15)
            )
            self.assertTrue(
                all(j["route"] is None and j["request"]["ignore_eos"] for j in jobs)
            )
            self.assertTrue(
                all(
                    len(j["request"]["prompt"]) + j["request"]["max_tokens"] <= 8192
                    for j in jobs
                )
            )

    def test_original_build_plans_rotates_order_and_selects_real_strategies(self):
        cfg = config()
        fake = dict(
            run_dir=Path("r/controller"),
            controller_argv=["python", "entry"],
            source_env={},
            target_env={},
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                comparison.pilot,
                "build_scene",
                side_effect=lambda *a: copy.deepcopy(fake),
            ),
        ):
            plans = comparison.build_plans(
                dict(policy_config=str(obs.ASSETS / "policy.json")),
                cfg,
                {},
                Path(tmp),
                Mock(),
                Mock(),
            )
        self.assertEqual(len(plans), 12)
        self.assertEqual([a for _, a, _, _ in plans[:4]], four.ARMS)
        self.assertEqual(
            [a for _, a, _, _ in plans[4:8]], four.ARMS[1:] + four.ARMS[:1]
        )
        for _, arm, _, plan in plans:
            argv = plan["controller_argv"]
            self.assertEqual("--no-migration" in argv, arm != "migrate")
            expected = {
                "weighted": "weighted-round-robin",
                "least": "least-inflight",
            }.get(arm, "pressure")
            self.assertEqual(argv[argv.index("--routing-strategy") + 1], expected)

    def test_weighted_and_least_keep_migration_disabled(self):
        for arm in ("weighted", "least"):
            plan = dict(controller_argv=["python", "entry", "--no-migration"])
            with (
                patch.object(runner, "adjust_migration_plan"),
                patch.object(runner, "configure_predictor"),
            ):
                runner.adapt_plan(plan, arm, Path("checkpoint"), Path("model"))
            self.assertNotIn("--qwen3-execute-migrations", plan["controller_argv"])
            plan["controller_argv"].remove("--no-migration")
            with (
                patch.object(runner, "adjust_migration_plan"),
                patch.object(runner, "configure_predictor"),
                self.assertRaises(ValueError),
            ):
                runner.adapt_plan(plan, arm, Path("checkpoint"), Path("model"))

    def test_full_score_counts_target_origin_and_handoff_and_full_drain(self):
        payload = dict(prompt=[1], max_tokens=3)
        rows = [
            dict(
                job_id="r",
                request=payload,
                initial_pool="tp1",
                source_error=None,
                response=dict(
                    committed=True,
                    handoff_stall_s=2.0,
                    emitted=[
                        dict(index=i, token_id=i, unix_s=t, origin=p)
                        for i, (t, p) in enumerate(
                            ((100.1, "source"), (102.1, "target"), (401.0, "target"))
                        )
                    ],
                ),
                source_transport=dict(request_started_unix_s=100.0),
            )
        ]
        score = four.full_score(rows, config(), 100.0, [])
        self.assertEqual(score["raw_window_tokens"], 2)
        self.assertEqual(score["request_rows"][0]["drain_tokens"], 1)
        self.assertEqual(score["actual_pool_window_tokens"], dict(tp1=1, tp4=1))
        self.assertEqual(score["slo_attainment"]["all_requests"]["passed"], 0)
        self.assertEqual(score["slo_request_goodput"], 0)
        self.assertEqual(score["tail_interval_counts"]["1000"], 2)

    def test_aggregate_pools_raw_requests_not_run_percentiles(self):
        def score(times):
            payload = dict(prompt=[1], max_tokens=2)
            rows = [
                dict(
                    job_id=str(i),
                    request=payload,
                    initial_pool="tp1",
                    source_error=None,
                    response=dict(
                        committed=False,
                        emitted=[
                            dict(
                                index=0, token_id=1, unix_s=100.0 + t, origin="source"
                            ),
                            dict(
                                index=1,
                                token_id=2,
                                unix_s=100.0 + t + 0.01,
                                origin="source",
                            ),
                        ],
                    ),
                    source_transport=dict(request_started_unix_s=100.0),
                )
                for i, t in enumerate(times)
            ]
            return four.full_score(rows, config(), 100.0, [])

        a, b = score([1] * 19 + [100]), score([2] * 20)
        merged = four.aggregate([dict(arm="stay", score=a), dict(arm="stay", score=b)])[
            "stay"
        ]
        self.assertEqual(merged["denominator_s"], 600)
        self.assertEqual(merged["all_requests"]["ttft_s"]["p95"], 2)
        self.assertEqual(merged["all_requests"]["requests"], 40)
        self.assertEqual(merged["raw_rate"], 80 / 600)

    def test_four_mode_coverage_does_not_require_forced_migration(self):
        summary = dict(
            requests=1,
            output_tokens=2,
            fixed_arrivals_valid=True,
            migration_starts=0,
            committed_requests=0,
            source_releases=0,
            slo_pass=1,
            load_coverage_pass=True,
        )
        for key in (
            "window_soft_goodoutput_s",
            "window_hard_token_goodput_s",
            "e2e_mean_s",
            "e2e_p95_s",
            "short_ttft_p95_s",
            "short_e2e_p95_s",
            "max_itl_s",
            "workload_makespan_s",
        ):
            summary[key] = 1.0
        ledger = [
            dict(
                pair="r1", arm=a, status="COMPLETED", input_sha256="x", summary=summary
            )
            for a in four.ARMS
        ]
        rows = four.comparisons(ledger)
        self.assertTrue(rows[0]["matched_output_work"])
        self.assertTrue(rows[0]["all_load_covered"])
        self.assertFalse(rows[0]["comparisons"]["stay"]["actual_migration_observed"])
        ledger[-1]["summary"] = dict(summary, load_coverage_pass=False)
        self.assertFalse(four.comparisons(ledger)[0]["all_load_covered"])
