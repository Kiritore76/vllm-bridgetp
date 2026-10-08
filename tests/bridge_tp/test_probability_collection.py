"""CPU contract tests for F1/F2, independent of CUDA and predictor training."""

import itertools
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bridge_tp.experiment_probability_gate import ProbabilityGate
from tools.bridge_tp.horizon_goodoutput import score_horizon
from tools.bridge_tp.risk_urgency import build_snapshot, remaining_ge_bounds
from tools.bridge_tp.run_randomized_goodoutput_pilot_a100 import (
    build_setup,
    collect_arm,
    configure_action,
    execute_pilot,
    probability_artifact_errors,
    probability_results,
)
from tools.bridge_tp.run_randomized_goodoutput_pilot_a100 import (
    parse_args as parse_pilot_args,
)


def prediction(position=0, generated=0, cap=100):
    return {
        "status": "AVAILABLE",
        "probabilities": [0.1, 0.3, 0.6],
        "category_upper_edges": [0, 4],
        "output_tokens": generated,
        "prediction_output_tokens": position,
        "max_remaining_output_tokens": cap,
        "ignore_eos": False,
    }


def snapshot(growth=10, pending=0):
    return {
        "state": "LOCAL",
        "unix_s": 10,
        "source_sampled_unix_s": 10,
        "target_sampled_unix_s": 10,
        "source_free_kv_tokens": 1100,
        "source_guard_free_kv_tokens": 100,
        "source_prefill_pending_kv_tokens": pending,
        "source_decode_growth_tokens_s": growth,
        "generated_tokens": 0,
        "current_context_tokens": 20,
        "target_free_kv_tokens": 10000,
        "target_prefill_pending_kv_tokens": 0,
        "target_kv_usage_frac": 0.1,
        "channel_available": True,
    }


def risk(growth=10, speed=10, pending=0, pred=None, free=1100,
         prefill_growth=None, unallocated=0):
    state = snapshot(growth, pending)
    state["source_free_kv_tokens"] = free
    state["source_prefill_growth_tokens_s"] = prefill_growth
    state["source_prefill_unallocated_kv_tokens"] = unallocated
    return build_snapshot(
        snapshot=state,
        prediction=pred or prediction(cap=2000),
        candidate_rate=speed,
        initial_rate={"rate_bytes_s": 1000000, "profile": "LOW"},
        kv_bytes_per_token=100,
        release_tail_s=5,
        block_size=1,
    ).to_json()


class TestProbabilityCollection(unittest.TestCase):
    def test_soft_slo_requires_full_request_window_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.pilot_fixture(Path(directory), "test", 0)
            args.slo_profile = "slow1pct_soft"
            with patch(
                "tools.bridge_tp.run_randomized_goodoutput_pilot_a100.verify",
                side_effect=RuntimeError("hardware preflight"),
            ) as verify_mock:
                with self.assertRaisesRegex(ValueError, "soft SLO profile"):
                    execute_pilot(args)
                verify_mock.assert_not_called()
                args.window_token_goodoutput = True
                args.arrival_window_s = args.evaluation_horizon_s
                with self.assertRaisesRegex(RuntimeError, "hardware preflight"):
                    execute_pilot(args)

    def test_nonzero_only_threshold_reaches_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.pilot_fixture(Path(directory), "test", 0)
            args.probability_thresholds = [0.8]
            args.constructed_workload = True
            args.expected_input_sha256 = "a" * 64
            args.constructed_source_count = 3
            args.constructed_target_count = 0
            args.window_token_goodoutput = True
            args.evaluation_horizon_s = 300
            args.arrival_window_s = 300
            args.arrival_wave_period_s = 24
            with patch(
                "tools.bridge_tp.run_randomized_goodoutput_pilot_a100.verify",
                side_effect=RuntimeError("reached hardware preflight"),
            ) as verify_mock:
                with self.assertRaisesRegex(RuntimeError, "hardware preflight"):
                    execute_pilot(args)
                verify_mock.assert_called_once_with(args)

    def test_invalid_thresholds_rejected_before_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.pilot_fixture(Path(directory), "test", 0)
            with patch(
                "tools.bridge_tp.run_randomized_goodoutput_pilot_a100.verify"
            ) as verify_mock:
                for thresholds in ([], [0.8, 0.8], [-0.1], [1.1], [math.nan],
                                   [math.inf], [0.8, 0.80000001]):
                    with self.subTest(thresholds=thresholds):
                        args.probability_thresholds = thresholds
                        with self.assertRaisesRegex(ValueError, "unique finite"):
                            execute_pilot(args)
                verify_mock.assert_not_called()

    def pilot_fixture(self, root, name, target_count):
        files = {}
        for key in (
            "model",
            "input",
            "base",
            "survival",
            "guard",
            "checkpoint",
            "reference",
        ):
            files[key] = root / key
            files[key].write_text("8448" if key == "guard" else "{}")
        argv = ["pilot", "--expected-revision", "test-head", "--out-dir", str(root)]
        for key, path in files.items():
            argv += [f"--{key}", str(path)]
        argv += [
            "--probability-pilot",
            "--max-model-len",
            "16384",
            "--tp4-max-model-len",
            "32768",
            "--risk-observation-shadow",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_pilot_args()
        anchor = root / "anchor.json"
        anchor.write_text(
            json.dumps(
                {
                    "prompt": [11, 12],
                    "max_tokens": 4096,
                    "ignore_eos": False,
                }
            )
        )
        jobs = [
            {
                "job_id": f"{pool}-{i}",
                "pool": pool,
                "start_after_s": 0,
                "request": {"model": "test", "prompt": [11], "max_tokens": 128},
            }
            for pool, count in (("source", 1), ("target", target_count))
            for i in range(count)
        ]
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"format_version": 1, "jobs": jobs}))
        setup = {
            "anchors": {
                name: {
                    "path": str(anchor),
                    "sha256": "anchor-sha",
                    "prompt_tokens": 2,
                    "max_tokens": 4096,
                    "total_max_tokens": 4096,
                }
            },
            "manifests": {name: {"path": str(manifest), "sha256": "manifest-sha"}},
        }
        return args, setup

    def test_actual_probability_commands_validate_all_pilot_blocks(self):
        from tools.bridge_tp import run_shadow_strategy_online_validation as online

        for name, targets in (
            ("p00_source1_target2", 0),
            ("p02_source3_target8", 8),
            ("p05_source5_target24", 24),
        ):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, setup = self.pilot_fixture(root, name, targets)
                args.probability_min_urgency = 1.0

                def validate_command(command, log, args=args, targets=targets):
                    with patch.object(sys, "argv", command[1:]):
                        parsed = online.parse_args()
                    self.assertEqual(parsed.probability_min_urgency, 1.0)
                    hashes = {
                        parsed.manifest: parsed.expected_manifest_sha256,
                        parsed.guard_file: parsed.expected_guard_sha256,
                        parsed.survival_table: parsed.expected_survival_sha256,
                        parsed.predictor_checkpoint: parsed.predictor_checkpoint_sha256,
                        parsed.anchor_request_file: (
                            parsed.expected_anchor_request_sha256
                        ),
                    }
                    with (
                        patch.object(online, "os", SimpleNamespace(name="posix")),
                        patch.object(online.common, "model_kv_geometry"),
                        patch.object(online.common, "git", return_value="test-head"),
                        patch.object(online.common, "sha256", side_effect=hashes.get),
                        patch.object(online.common, "CONFIG_TEMPLATE", args.base),
                        patch.object(online.common, "SOURCE_REQUEST", args.base),
                        patch.object(
                            online.subprocess,
                            "run",
                            return_value=SimpleNamespace(returncode=0),
                        ),
                    ):
                        _, _, pressure = online.validate_inputs(parsed)
                        self.assertEqual(pressure["target_jobs"], targets)
                        if targets == 0:
                            parsed.probability_threshold = None
                            parsed.probability_min_urgency = 0.0
                            with self.assertRaisesRegex(
                                ValueError, "target background"
                            ):
                                online.validate_inputs(parsed)
                            parsed.probability_threshold = 0
                            parsed.probability_min_urgency = 1.0
                            parsed.minimum_ready_target_jobs = 1
                            with self.assertRaisesRegex(ValueError, "readiness gate"):
                                online.validate_inputs(parsed)
                    return 1  # No GPU launch; exercise the missing-evidence path.

                for theta, assignment in itertools.product(
                    (0, 0.001, 0.01, 0.05), ("START", "STAY")
                ):
                    action = f"prob_{theta:g}_{assignment}"
                    with (
                        self.subTest(name=name, action=action),
                        patch(
                            "tools.bridge_tp.run_randomized_goodoutput_pilot_a100.execute",
                            side_effect=validate_command,
                        ) as execute,
                    ):
                        result = collect_arm(args, root, setup, name, action)
                        self.assertEqual(execute.call_count, 1)
                        self.assertTrue(result["fatal_error"])
                        self.assertFalse(result["fixed_horizon_eligible"])
                        self.assertIsNone(result["fixed_horizon_goodoutput_tokens_s"])
                        self.assertEqual(
                            result["observed_action"]["probability_episode_outcome"],
                            "TECHNICAL_FAILURE",
                        )
                        saved = root / name / f"{action}.result.json"
                        self.assertEqual(json.loads(saved.read_text()), result)
                        report = probability_results(
                            {
                                "seed": args.seed,
                                "probability_thresholds": [theta],
                                "cases": {name: {action: result}},
                            }
                        )
                        self.assertFalse(report["samples"][0]["effect_sample_eligible"])

    def test_probability_evidence_checks_parent_contract_and_corrupt_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "r01_shadow_only"
            run.mkdir()
            (run.parent / "contract.json").write_text("{}")
            for relative in (
                "background/background_manifest.json",
                "background/background_summary.json",
                "controller/response_proxy_stats.json",
                "controller/source_response.json",
                "controller/phase9_audit.jsonl",
            ):
                path = run / relative
                path.parent.mkdir(exist_ok=True)
                path.write_text("{}")
            self.assertEqual(probability_artifact_errors(run), [])
            (run / "controller/phase9_audit.jsonl").write_text('{"kind":')
            errors = probability_artifact_errors(run)
            self.assertEqual(len(errors), 1)
            self.assertIn(
                "invalid run artifact: controller/phase9_audit.jsonl", errors[0]
            )

    def test_partial_service_error_and_input_error_have_separate_receipts(self):
        import urllib.error
        from io import BytesIO

        from vllm.bridge_tp.controller.online_io import post_streaming_completion

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.json"
            truncated = BytesIO(b'data: {"choices":[{"token_ids":[42]}]}\n')
            with (
                patch("urllib.request.urlopen", return_value=truncated),
                self.assertRaisesRegex(RuntimeError, "ended before"),
            ):
                post_streaming_completion(
                    "http://example", {}, 1, lambda *args: None, failure_path=path
                )
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt["token_ids"], [42])
            self.assertEqual(receipt["failure_kind"], "SERVICE_REQUEST_FAILURE")
            self.assertGreaterEqual(
                receipt["completed_unix_s"], receipt["request_started_unix_s"]
            )
            error = urllib.error.HTTPError(
                "http://example", 400, "bad input", {}, BytesIO(b"invalid budget")
            )
            with (
                patch("urllib.request.urlopen", side_effect=error),
                self.assertRaisesRegex(RuntimeError, "HTTP 400"),
            ):
                post_streaming_completion(
                    "http://example", {}, 1, lambda *args: None, failure_path=path
                )
            self.assertEqual(
                json.loads(path.read_text())["failure_kind"], "TECHNICAL_INPUT_FAILURE"
            )

    def test_one_effect_sample_per_episode_and_safety_not_randomized(self):
        from tools.bridge_tp.run_randomized_goodoutput_pilot_a100 import (
            probability_results,
        )

        candidate = {"snapshot": {"generated_tokens": 9, "H_tokens": 1000}}
        observed = {
            "probability_candidate": candidate,
            "actual_probability_executions": [{"actual_action": "START_SHADOW"}],
        }
        start = {"fixed_horizon_goodoutput_tokens_s": 10, "observed_action": observed}
        stay = {
            "fixed_horizon_goodoutput_tokens_s": 8,
            "observed_action": {"probability_candidate": candidate},
        }
        summary = {
            "seed": 1,
            "probability_thresholds": [0.1],
            "cases": {"load": {"prob_0.1_START": start, "prob_0.1_STAY": stay}},
        }
        result = probability_results(summary)
        self.assertEqual(len(result["samples"]), 1)
        self.assertEqual(result["samples"][0]["descriptive_delta_tokens_s"], 2)
        observed["safety_override_ticks"] = [12]
        self.assertFalse(
            probability_results(summary)["samples"][0]["effect_sample_eligible"]
        )

    def test_bucket_bounds_contain_all_discrete_support_allocations(self):
        # Enumerate actual distributions: the middle bucket [1,4] and
        # tail [5,infinity] each put mass at possible points. No reference
        # calculation copies the production interval algorithm.
        for lag, n in itertools.product(range(5), range(1, 9)):
            p = prediction(generated=lag)
            bounds = remaining_ge_bounds(p, n)
            for middle, tail in itertools.product(range(1, 5), [5, 6, 8, 1000]):
                points = [(0, 0.1), (middle, 0.3), (tail, 0.6)]
                survived = sum(m for r, m in points if r >= lag)
                exact = (
                    sum(m for r, m in points if r >= lag and r - lag >= n) / survived
                )
                self.assertLessEqual(bounds[0], exact + 1e-12)
                self.assertGreaterEqual(bounds[1], exact - 1e-12)

    def test_tail_integer_boundary_stop_cap_and_survival(self):
        p = prediction(generated=4)
        self.assertAlmostEqual(remaining_ge_bounds(p, 1)[0], 2 / 3)
        self.assertEqual(remaining_ge_bounds(p, 1)[1], 1)
        self.assertEqual(remaining_ge_bounds(p, 100.1), [0, 0])
        p["ignore_eos"] = True
        self.assertIsNone(remaining_ge_bounds(p, 1))
        p["ignore_eos"] = False
        p["status"] = "STALE"
        self.assertIsNone(remaining_ge_bounds(p, 1))

    def test_single_request_reduction_and_pool_conversion(self):
        single = risk()
        self.assertEqual(single["H_tokens"], 1000)
        self.assertEqual(single["r_guard_tokens"], 1000)
        self.assertEqual(single["p_guard_est_bounds"], single["p_capacity_bounds"])
        parallel = risk(growth=100)
        self.assertEqual(parallel["r_guard_tokens"], 100)
        self.assertEqual(parallel["T_guard_s"], 10)
        self.assertGreater(parallel["U"], single["U"])
        self.assertEqual(risk(pending=128)["H_tokens"], 1000)

    def test_pending_prefill_does_not_consume_capacity(self):
        baseline = risk()
        pending = risk(pending=100000)
        for key in ("H_tokens", "source_physical_headroom_tokens",
                    "T_guard_s", "U", "p_guard_est_bounds", "physical_feasible"):
            self.assertEqual(pending[key], baseline[key], key)
        self.assertEqual(pending["source"]["source_prefill_pending_kv_tokens"],
                         100000)
        state = snapshot()
        state["target_prefill_pending_kv_tokens"] = 100000
        result = build_snapshot(
            snapshot=state, prediction=prediction(cap=2000), candidate_rate=10,
            initial_rate={"rate_bytes_s": 1000000, "profile": "LOW"},
            kv_bytes_per_token=100, release_tail_s=5, block_size=1,
        ).to_json()
        self.assertTrue(result["physical_feasible"])

    def test_prefill_and_decode_rates_both_advance_guard(self):
        mixed = risk(growth=10, prefill_growth=90, unallocated=2000)
        prefill_only = risk(growth=0, prefill_growth=100, unallocated=2000)
        for row in (mixed, prefill_only):
            self.assertEqual(row["H_tokens"], 1000)
            self.assertEqual(row["pool_growth_tokens_s"], 100)
            self.assertEqual(row["T_guard_s"], 10)
            self.assertTrue(row["physical_feasible"])
        self.assertEqual(mixed["U"], prefill_only["U"])
        self.assertEqual(mixed["p_guard_est_bounds"],
                         prefill_only["p_guard_est_bounds"])

    def test_prefill_guard_time_without_candidate_decode_rate(self):
        row = risk(growth=0, prefill_growth=100, speed=0, unallocated=2000)
        self.assertEqual(row["T_guard_s"], 10)
        self.assertIsNone(row["p_guard_est_bounds"])
        self.assertEqual(row["status"], "NO_CANDIDATE_RATE")

    def test_finite_prefill_cannot_fill_guard_without_decode(self):
        row = risk(growth=0, prefill_growth=100, unallocated=500)
        self.assertEqual(row["status"], "NO_PROJECTED_GUARD_REACH")
        self.assertEqual(row["U"], 0)
        self.assertEqual(row["p_guard_est_bounds"], [0, 0])
        self.assertIsNone(row["T_guard_s"])
        json.dumps(row, allow_nan=False)

    def test_release_calibration_includes_freeze_wait_budget(self):
        row = build_snapshot(
            snapshot=snapshot(), prediction=prediction(cap=2000),
            candidate_rate=10, initial_rate={"rate_bytes_s": 1000000},
            kv_bytes_per_token=100, release_tail_s=5, block_size=1,
            timing_calibration={"release_budget_s": 11.665,
                                "calibration_id": "test-budget",
                                "status": "ENGINEERING_BUDGET"},
        ).to_json()
        self.assertEqual(row["T_release_s"], 11.665)
        self.assertAlmostEqual(row["U"], (11.665 + 2) / 100)
        self.assertEqual(row["release_calibration_id"], "test-budget")

    def test_no_growth_protection_stale_and_delta_catchup(self):
        for growth in (-1, None, math.nan):
            row = risk(growth=growth)
            self.assertIsNone(row["T_guard_s"])
            self.assertIsNone(row["p_guard_est_bounds"])
            self.assertFalse(row["physical_feasible"])
            json.dumps(row, allow_nan=False)
        self.assertEqual(risk(free=100)["status"], "GUARD_REACHED")
        self.assertIn("delta_cannot_catch_up", risk(speed=10000)["physical_rejections"])
        row = snapshot()
        row["target_sampled_unix_s"] = 1
        result = build_snapshot(
            snapshot=row,
            prediction=prediction(),
            candidate_rate=10,
            initial_rate={"rate_bytes_s": 1e6},
            kv_bytes_per_token=100,
            release_tail_s=5,
        ).to_json()
        self.assertFalse(result["physical_feasible"])

    def test_missing_source_guard_deadline_still_allows_assigned_start(self):
        row = risk(growth=500, speed=1)
        self.assertGreater(row["U"], 1)
        self.assertLess(row["S_bounds_s"][0], 0)
        self.assertTrue(row["guard_deadline_warning"])
        self.assertIn("source_release_may_miss_guard", row["guard_warnings"])
        self.assertTrue(row["physical_feasible"])
        gate = ProbabilityGate(0.5, "late", assigned_action="START")
        decision = gate.observe(row, 1)
        self.assertTrue(decision["first_feasible_candidate"])
        self.assertEqual(decision["requested_action"], "START_SHADOW")
        self.assertFalse(decision["safety_protection_required"])
        self.assertTrue(decision["guard_deadline_warning"])

    def test_guard_reached_is_not_physical_capacity_exhaustion(self):
        row = risk(free=100)
        self.assertEqual(row["H_tokens"], 0)
        self.assertEqual(row["source_physical_headroom_tokens"], 100)
        self.assertEqual(row["p_guard_est_bounds"], [1, 1])
        self.assertIsNone(row["U"])
        self.assertTrue(row["physical_feasible"])
        json.dumps(row, allow_nan=False)
        gate = ProbabilityGate(0.8, "guard", assigned_action="START")
        self.assertEqual(gate.observe(row, 1)["requested_action"], "START_SHADOW")
        exhausted = risk(free=0)
        self.assertFalse(exhausted["physical_feasible"])
        self.assertIn(
            "source_physical_capacity_exhausted", exhausted["physical_rejections"]
        )
        decision = ProbabilityGate(0, "full", assigned_action="START").observe(
            exhausted, 1
        )
        self.assertEqual(decision["requested_action"], "STAY")
        self.assertTrue(decision["safety_protection_required"])

    def test_new_guard_warning_preserves_effect_but_legacy_replay_is_unchanged(self):
        row = risk(growth=500, speed=1)
        candidate = ProbabilityGate(0, "warning", assigned_action="START").observe(
            row, 1
        )
        legacy = dict(row, physical_feasible=False)
        legacy.pop("source_guard_policy")
        legacy["physical_rejections"] = ["source_release_may_miss_guard"]
        old_gate = ProbabilityGate(0, "legacy", assigned_action="START").observe(
            legacy, 1
        )
        self.assertTrue(old_gate["safety_protection_required"])
        self.assertEqual(old_gate["requested_action"], "STAY")
        observed = {
            "probability_candidate": candidate,
            "guard_deadline_warning_ticks": [1],
            "actual_probability_executions": [{"actual_action": "START_SHADOW"}],
        }
        summary = {
            "seed": 1,
            "probability_thresholds": [0],
            "cases": {
                "load": {
                    "prob_0_START": {
                        "fixed_horizon_goodoutput_tokens_s": 10,
                        "observed_action": observed,
                    },
                    "prob_0_STAY": {
                        "fixed_horizon_goodoutput_tokens_s": 8,
                        "observed_action": observed,
                    },
                }
            },
        }
        self.assertTrue(
            probability_results(summary)["samples"][0]["effect_sample_eligible"]
        )
        observed["safety_protection_required_ticks"] = [2]
        sample = probability_results(summary)["samples"][0]
        self.assertTrue(sample["effect_sample_eligible"])
        self.assertEqual(sample["descriptive_delta_tokens_s"], 2)
        self.assertEqual(sample["postdecision_capacity_exhaustion_ticks"],
                         {"START": [2], "STAY": [2]})
        summary["cases"]["load"]["prob_0_START"][
            "fixed_horizon_goodoutput_tokens_s"
        ] = 0
        harmed = probability_results(summary)["samples"][0]
        self.assertTrue(harmed["effect_sample_eligible"])
        self.assertEqual(harmed["descriptive_delta_tokens_s"], -8)
        summary["cases"]["load"]["prob_0_START"][
            "fixed_horizon_goodoutput_tokens_s"
        ] = 10
        observed["safety_protection_required_ticks"] = [1]
        self.assertFalse(
            probability_results(summary)["samples"][0]["effect_sample_eligible"]
        )
        observed["safety_protection_required_ticks"] = [2]
        observed["safety_override_ticks"] = [2]
        self.assertFalse(
            probability_results(summary)["samples"][0]["effect_sample_eligible"]
        )
        del observed["safety_override_ticks"]
        candidate["snapshot"]["physical_feasible"] = False
        self.assertFalse(
            probability_results(summary)["samples"][0]["effect_sample_eligible"]
        )

    def test_legacy_postdecision_protection_remains_excluded(self):
        candidate = {"snapshot": {"generated_tokens": 9, "H_tokens": 1000}}
        observed = {
            "probability_candidate": candidate,
            "actual_probability_executions": [{"actual_action": "START_SHADOW"}],
            "safety_protection_required_ticks": [12],
        }
        summary = {
            "seed": 1, "probability_thresholds": [0.1],
            "cases": {"load": {
                "prob_0.1_START": {
                    "fixed_horizon_goodoutput_tokens_s": 10,
                    "observed_action": observed,
                },
                "prob_0.1_STAY": {
                    "fixed_horizon_goodoutput_tokens_s": 8,
                    "observed_action": {"probability_candidate": candidate},
                },
            }},
        }
        sample = probability_results(summary)["samples"][0]
        self.assertFalse(sample["effect_sample_eligible"])
        self.assertEqual(sample["policy_delta_tokens_s"], 2)
        self.assertEqual(sample["effect_sample_policy"],
                         "LEGACY_ANY_PROTECTION_EXCLUSION_V1")

    def test_first_crossing_common_feasibility_stay_and_simultaneous(self):
        gate = ProbabilityGate(
            0.5, "seed", assigned_action="STAY", thresholds=(0, 0.2, 0.5)
        )
        s = {
            "status": "VALID",
            "p_guard_est_bounds": [0.6, 0.8],
            "physical_feasible": False,
        }
        first = gate.observe(s, 1)
        self.assertTrue(first["simultaneous_crossing"])
        self.assertFalse(first["first_feasible_candidate"])
        s["physical_feasible"] = True
        second = gate.observe(s, 2)
        self.assertTrue(second["first_feasible_candidate"])
        self.assertEqual(second["requested_action"], "STAY")
        self.assertFalse(gate.observe(s, 3)["first_feasible_candidate"])
        self.assertEqual(gate.candidate["tick"], 2)

    def test_horizon_zeros_late_and_service_failed_without_excluding_window(self):
        rows = [
            dict(request_id=x, pool="source", status=status, slo_success=True)
            for x, status in [
                ("on_time", "COMPLETED"),
                ("late", "COMPLETED"),
                ("failed", "FAILED"),
            ]
        ]
        rows.append(
            dict(
                request_id="anchor", pool="anchor", status="COMPLETED", slo_success=True
            )
        )
        background = {
            "results": [
                dict(
                    job_id=x,
                    request_started_unix_s=0,
                    request_ended_unix_s=end,
                    token_times_unix_s=[1, 2],
                )
                for x, end in [("on_time", 2), ("late", 11), ("failed", 3)]
            ]
        }
        source = {
            "request_started_unix_s": 0,
            "completed_unix_s": 5,
            "finish_reason": "stop",
        }
        proxy = {"emitted": [{"unix_s": 1}], "external_request_id": "anchor"}
        result = score_horizon(
            {"computable": True, "request_rows": rows},
            background,
            source,
            {},
            proxy,
            10,
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["good_output_tokens"], 3)
        self.assertEqual(result["unfinished_or_failed_at_H"], 2)
        # Drain allows late COMPLETED requests to contribute only their
        # pre-cutoff tokens. Failed service requests remain zero.
        background["results"][1]["token_times_unix_s"] = [1, 2, 10, 11]
        streaming = score_horizon(
            {"computable": True, "request_rows": rows},
            background, source, {}, proxy, 10, settle_after_h=True,
        )
        self.assertEqual(streaming["good_output_tokens"], 5)
        self.assertEqual(streaming["drain_completed_requests"], 1)
        self.assertFalse(streaming["drain_tokens_counted"])
        self.assertEqual(streaming["window_coverage"]["inflight_fraction"], 1)
        rows[1]["slo_success"] = False
        self.assertEqual(score_horizon(
            {"computable": True, "request_rows": rows},
            background, source, {}, proxy, 10, settle_after_h=True,
        )["good_output_tokens"], 3)
        del background["results"][0]["token_times_unix_s"]
        self.assertFalse(
            score_horizon(
                {"computable": True, "request_rows": rows},
                background,
                source,
                {},
                proxy,
                10,
            )["eligible"]
        )

    def test_probability_command_has_no_fixed_token_trigger_gate(self):
        command = [
            "--manager-m2-force-initial-high",
            "--manager-m2-expected-profile",
            "HIGH",
            "--manager-m2-min-history-byte-frac",
            ".9",
            "--m1-min-output-tokens",
            "128",
        ]
        configure_action(command, "prob_0.01_STAY")
        self.assertNotIn("--experiment-m1-action", command)
        self.assertEqual(command[command.index("--m1-min-output-tokens") + 1], "0")
        self.assertIn("--probability-threshold", command)

    def test_collection_identity_distinguishes_runs_keeps_shared_seed_group(self):
        summary = {"seed": 1, "probability_thresholds": [0.8],
                   "cases": {"load": {}}, "collection_id": "recipe-long"}
        first = probability_results(summary)["samples"][0]
        summary["collection_id"] = "recipe-short"
        second = probability_results(summary)["samples"][0]
        self.assertNotEqual(first["sample_id"], second["sample_id"])
        self.assertEqual(first["episode_group_id"], second["episode_group_id"])

    def test_rotated_unique_prompts_actual_context_and_idle_target(self):
        tokenizer = SimpleNamespace(encode=lambda s: [1] * len(s))
        fake = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.jsonl"
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": str(i),
                            "source_tree_id": str(i),
                            "split": "test",
                            "workload_group": "long_form" if i % 4 == 0 else "natural",
                            "prompt": "hi" + str(i),
                        }
                    )
                    + "\n"
                    for i in range(350)
                )
            )
            args = SimpleNamespace(
                input=path,
                model=root,
                probability_pilot=True,
                cases=["p00_source1_target2"],
                max_model_len=16384,
                tp4_max_model_len=32768,
                anchor_context_limit=True,
                background_context_limit=True,
                arrival_window_s=300,
                arrival_wave_period_s=24,
            )
            with patch.dict(sys.modules, {"transformers": fake}):
                setup = build_setup(args, root)
            manifest = json.loads(
                Path(setup["manifests"][args.cases[0]]["path"]).read_text()
            )
            jobs = manifest["jobs"]
            self.assertEqual(len(jobs), len({j["input_id"] for j in jobs}))
            self.assertEqual(manifest["target_count"], 0)
            self.assertGreater(max(j["start_after_s"] for j in jobs), 275)
            self.assertLess(max(j["start_after_s"] for j in jobs), 300)
            self.assertTrue(all(j["pool"] == "source" for j in jobs))
            self.assertTrue(all(j["request"]["max_tokens"] > 8192 for j in jobs))


if __name__ == "__main__":
    unittest.main()
