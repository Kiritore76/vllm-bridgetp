# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the Phase 9 conditional-calibration tools."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load_tool(name: str):
    path = ROOT / "tools" / "bridge_tp" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = load_tool("analyze_phase9_calibration")
record = load_tool("record_phase9_calibration")
summarize = load_tool("summarize_phase9_calibration")
fit_tpot = load_tool("fit_phase9_tick_tpot")
run_tpot = load_tool("run_phase9_tpot_sweep")
run_interference = load_tool("run_phase9_interference_sweep")
reclassify_pilot = load_tool("reclassify_phase9_load_pilot")
collect_interference = load_tool("collect_phase9_interference_grid")


class TestCalibrationAnalysis(unittest.TestCase):
    def test_very_low_profile_builds_twelve_formal_cells(self):
        original_bands = dict(run_interference.BANDS)
        try:
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_phase9_interference_sweep.py",
                    "pilot",
                    "--out-root",
                    "out",
                    "--model",
                    "model",
                    "--tp4-blocks",
                    "35739",
                    "--band-profile",
                    "very_low",
                ],
            ):
                args = run_interference.parse_args()
            self.assertEqual(args.formal_bands, ("very_low",))
            args.formal_rates = (0.0, 0.4, 0.7, 1.2)
            args.formal_reps = (1, 2, 3)
            self.assertEqual(len(run_interference.formal_expected_keys(args)), 12)
        finally:
            run_interference.BANDS = original_bands

    def test_summarizer_accepts_complete_very_low_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for rate in (0.0, 0.4, 0.7, 1.2):
                for rep in (1, 2, 3):
                    path = root / f"rate{rate}_rep{rep}.json"
                    path.write_text(
                        json.dumps(
                            {
                                "status": "ACCEPTED",
                                "load_band": "very_low",
                                "rep": rep,
                                "inputs": {"target_rate_gib_s": rate},
                                "observed": {
                                    "kv_usage_mean": 0.03,
                                    "effective_rate_gib_s": rate,
                                    "p99_tpot_s": 0.02 + rate * 0.01,
                                    "p99_itl_s": 0.03 + rate * 0.01,
                                },
                            }
                        )
                    )
                    inputs.append(str(path))
            out = root / "summary.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "summarize_phase9_calibration.py",
                    "--inputs",
                    *inputs,
                    "--out",
                    str(out),
                    "--expected-bands",
                    "very_low",
                    "--expected-rates",
                    "0",
                    "0.4",
                    "0.7",
                    "1.2",
                    "--expected-reps",
                    "1",
                    "2",
                    "3",
                ],
            ):
                summarize.main()
            self.assertEqual(json.loads(out.read_text())["status"], "COMPLETE")

    def test_collection_prefers_complete_low_error_result(self):
        base = {
            "observed": {
                "telemetry_samples": 299,
                "rate_relative_error": 0.01,
            }
        }
        more_complete = (
            Path("more_complete.json"),
            {
                "observed": {
                    "telemetry_samples": 300,
                    "rate_relative_error": 0.02,
                }
            },
        )
        lower_error = (Path("lower_error.json"), base)
        self.assertEqual(
            min(
                [lower_error, more_complete],
                key=collect_interference.selection_score,
            ),
            more_complete,
        )

    def test_observed_safe_policy_accepts_shifted_but_safe_load(self):
        args = type(
            "Args",
            (),
            {
                "measurement_load_policy": "observed_safe",
                "measurement_max_kv_p95": 0.85,
                "load_min": 0.50,
                "load_max": 0.65,
                "min_band_fraction": 0.80,
            },
        )()
        name, passed, p95 = analyze.measurement_load_check(
            [0.40, 0.42, 0.44, 0.46], args
        )
        self.assertEqual(name, "measurement_load_safe")
        self.assertTrue(passed)
        self.assertLess(p95, 0.85)

    def test_observed_safe_policy_rejects_unsafe_p95(self):
        args = type(
            "Args",
            (),
            {
                "measurement_load_policy": "observed_safe",
                "measurement_max_kv_p95": 0.85,
                "load_min": 0.50,
                "load_max": 0.65,
                "min_band_fraction": 0.80,
            },
        )()
        _, passed, _ = analyze.measurement_load_check(
            [0.80, 0.86, 0.90], args
        )
        self.assertFalse(passed)

    def test_attainable_load_bands_are_frozen(self):
        self.assertEqual(
            run_interference.serialized_bands(),
            {
                "low": [0.10, 0.25],
                "medium": [0.30, 0.45],
                "high": [0.50, 0.65],
            },
        )

    def test_formal_rejects_legacy_or_missing_band_definition(self):
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            run_interference.validate_pilot_for_formal({"status": "READY"})

    def test_formal_accepts_current_ready_pilot(self):
        run_interference.validate_pilot_for_formal(
            {
                "status": "READY",
                "load_bands": run_interference.serialized_bands(),
            }
        )

    def test_resume_reuses_only_accepted_formal_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accepted_dir = root / "accepted"
            rejected_dir = root / "rejected"
            accepted_dir.mkdir()
            rejected_dir.mkdir()
            base = {
                "load_band": "low",
                "rep": 1,
                "inputs": {"target_rate_gib_s": 0.4},
            }
            accepted_path = accepted_dir / "condition_result.json"
            accepted_path.write_text(
                json.dumps({**base, "status": "ACCEPTED"}),
                encoding="utf-8",
            )
            (rejected_dir / "condition_result.json").write_text(
                json.dumps({**base, "status": "REJECTED", "rep": 2}),
                encoding="utf-8",
            )
            self.assertEqual(
                run_interference.accepted_formal_results(root),
                {("low", 0.4, 1): accepted_path},
            )

    def test_formal_progress_lists_missing_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "condition_result.json"
            result.touch()
            out = run_interference.write_formal_progress(
                root,
                {("low", 0.0, 1): result},
                {("low", 0.4, 2): "condition rejected"},
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "INCOMPLETE")
            self.assertEqual(payload["accepted_conditions"], 1)
            self.assertEqual(len(payload["missing_conditions"]), 35)
            self.assertEqual(
                payload["failed_conditions"],
                [
                    {
                        "load_band": "low",
                        "target_rate_gib_s": 0.4,
                        "rep": 2,
                        "last_error": "condition rejected",
                    }
                ],
            )

    def test_formal_subset_keys_are_explicit(self):
        args = type(
            "Args",
            (),
            {
                "formal_bands": ("medium", "high"),
                "formal_rates": (0.7,),
                "formal_reps": (1,),
            },
        )()
        self.assertEqual(
            run_interference.formal_expected_keys(args),
            [("medium", 0.7, 1), ("high", 0.7, 1)],
        )

    def test_formal_counts_prior_failure_then_retries_once_and_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior_failure = root / "formal_low_prior_failure"
            prior_failure.mkdir()
            (prior_failure / "condition_manifest.json").write_text(
                json.dumps(
                    {
                        "mode": "formal_low",
                        "load_band": "low",
                        "target_rate_gib_s": 0.0,
                        "rep": 1,
                    }
                ),
                encoding="utf-8",
            )
            pilot_summary = root / "load_pilot_summary.json"
            pilot_summary.write_text(
                json.dumps(
                    {
                        "status": "READY",
                        "load_bands": run_interference.serialized_bands(),
                        "selected": {
                            band: {"qps": 0.7}
                            for band in run_interference.BANDS
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "pilot_summary": pilot_summary,
                    "out_root": root,
                    "resume": True,
                    "max_attempts": 2,
                    "continue_on_error": True,
                    "formal_bands": tuple(run_interference.BANDS),
                    "formal_rates": run_interference.RATES,
                    "formal_reps": run_interference.REPS,
                },
            )()
            failed_key = ("low", 0.0, 1)
            attempts = 0

            def fake_run_condition(_args, *, band, rate, rep, **_kwargs):
                nonlocal attempts
                key = (band, rate, rep)
                if key == failed_key:
                    attempts += 1
                    raise RuntimeError("unstable load")
                return root / f"{band}_{rate}_{rep}", {}

            with (
                mock.patch.object(
                    run_interference,
                    "run_condition",
                    side_effect=fake_run_condition,
                ) as run_mock,
                mock.patch.object(
                    run_interference,
                    "analyze_formal_condition",
                    side_effect=lambda _args, path, _band, _rate: path
                    / "condition_result.json",
                ) as analyze_mock,
            ):
                run_interference.run_formal(args)

            self.assertEqual(attempts, 1)
            self.assertEqual(run_mock.call_count, 36)
            self.assertEqual(analyze_mock.call_count, 35)
            progress = json.loads(
                (root / "formal_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["accepted_conditions"], 35)
            self.assertEqual(
                progress["failed_conditions"][0]["last_error"],
                "unstable load",
            )

    def test_load_pilot_selection_is_band_scoped(self):
        conditions = [
            {
                "qps": 0.2,
                "kv_usage_mean": 0.20,
                "band_fractions": {"low": 0.9, "medium": 0.0, "high": 0.0},
            },
            {
                "qps": 0.4,
                "kv_usage_mean": 0.375,
                "band_fractions": {"low": 0.0, "medium": 0.9, "high": 0.0},
            },
            {
                "qps": 0.8,
                "kv_usage_mean": 0.60,
                "band_fractions": {"low": 0.0, "medium": 0.0, "high": 0.9},
            },
        ]
        selected = run_interference.choose_band_qps(conditions)
        self.assertEqual(selected["low"]["qps"], 0.2)
        self.assertEqual(selected["medium"]["qps"], 0.4)
        self.assertEqual(selected["high"]["qps"], 0.8)

    def test_load_pilot_ignores_stability_timeout(self):
        conditions = [
            {
                "qps": 0.7,
                "kv_usage_mean": 0.20,
                "band_fractions": {"low": 1.0, "medium": 0.0, "high": 0.0},
                "stability_status": "TIMEOUT",
            },
            {
                "qps": 0.75,
                "kv_usage_mean": 0.21,
                "band_fractions": {"low": 0.9, "medium": 0.0, "high": 0.0},
                "stability_status": "STABLE",
            },
        ]
        selected = run_interference.choose_band_qps(conditions)
        self.assertEqual(selected["low"]["qps"], 0.75)

    def test_load_pilot_honors_configured_band_fraction(self):
        conditions = [
            {
                "qps": 0.7,
                "kv_usage_mean": 0.20,
                "band_fractions": {"low": 0.85, "medium": 0.0, "high": 0.0},
                "stability_status": "STABLE",
            }
        ]
        self.assertIsNotNone(
            run_interference.choose_band_qps(conditions, 0.80)["low"]
        )
        self.assertIsNone(
            run_interference.choose_band_qps(conditions, 0.90)["low"]
        )

    def test_matching_stable_band_requires_mean_and_fraction(self):
        summary = {
            "kv_usage_mean": 0.375,
            "band_fractions": {"low": 0.0, "medium": 0.85, "high": 0.0},
        }
        self.assertEqual(
            run_interference.matching_stable_band(summary, None, 0.80),
            "medium",
        )
        self.assertIsNone(
            run_interference.matching_stable_band(summary, "low", 0.80)
        )

    def test_recent_load_summary_rejects_incomplete_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.csv"
            path.write_text(
                "monotonic_s,kv_usage_frac\n100,0.18\n110,0.20\n",
                encoding="utf-8",
            )
            self.assertIsNone(
                run_interference.recent_load_summary(path, 60.0, 1.0)
            )

    def test_recent_load_summary_accepts_complete_stable_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.csv"
            rows = ["monotonic_s,kv_usage_frac"]
            rows.extend(
                f"{second},{0.18 + (second % 2) * 0.01}"
                for second in range(61)
            )
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            summary = run_interference.recent_load_summary(path, 60.0, 1.0)
            self.assertIsNotNone(summary)
            self.assertEqual(
                run_interference.matching_stable_band(summary, "low", 0.80),
                "low",
            )

    def test_reclassification_requires_stable_then_measurement_window(self):
        rows = [(float(second), 0.37) for second in range(421)]
        result = reclassify_pilot.find_qualifying_window(
            rows,
            "medium",
            stability_window_s=120.0,
            measurement_window_s=300.0,
            min_band_fraction=0.80,
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["measurement"]["kv_usage_mean"], 0.37)

    def test_reclassification_rejects_a_ramp_through_band(self):
        rows = [
            (float(second), min(0.9, second / 500.0))
            for second in range(601)
        ]
        self.assertIsNone(
            reclassify_pilot.find_qualifying_window(
                rows,
                "medium",
                stability_window_s=120.0,
                measurement_window_s=300.0,
                min_band_fraction=0.80,
            )
        )

    def test_tpot_sweep_matrix_has_eighteen_conditions(self):
        args = type(
            "Args",
            (),
            {
                "tp1_url": "http://127.0.0.1:8001",
                "tp4_url": "http://127.0.0.1:8200",
                "tp1_blocks": 1968,
                "tp4_blocks": 35739,
                "qps": (1.0, 2.0, 4.0),
                "reps": (1, 2, 3),
            },
        )()
        matrix = list(run_tpot.condition_matrix(args))
        self.assertEqual(len(matrix), 18)
        self.assertEqual(matrix[0], ("tp1", args.tp1_url, 1968, 1.0, 1))
        self.assertEqual(matrix[-1], ("tp4", args.tp4_url, 35739, 4.0, 3))

    def test_tpot_resume_reuses_only_complete_hashed_condition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition = root / "tpot_tp1_qps0.1_r2_test"
            condition.mkdir()
            manifest = condition / "condition_manifest.json"
            benchmark = condition / "benchmark.json"
            telemetry = condition / "telemetry.csv"
            manifest.write_text(
                json.dumps(
                    {
                        "side": "tp1",
                        "qps": 0.1,
                        "rep": 2,
                        "input_len": 256,
                        "output_len": 2048,
                        "num_prompts": 40,
                    }
                )
            )
            benchmark.write_text(json.dumps({"completed": 40, "failed": 0}))
            telemetry.write_text("num_running\n1\n")
            run_tpot.write_hashes(condition, [manifest, benchmark, telemetry])

            accepted = run_tpot.accepted_conditions(
                root,
                input_len=256,
                output_len=2048,
                num_prompts=40,
            )
            self.assertEqual(
                accepted[("tp1", 0.1, 2)],
                telemetry,
            )

            telemetry.write_text("num_running\n2\n")
            self.assertEqual(
                run_tpot.accepted_conditions(
                    root,
                    input_len=256,
                    output_len=2048,
                    num_prompts=40,
                ),
                {},
            )

    def test_weighted_tpot_fit_recovers_linear_model(self):
        rows = [
            {
                "num_running": running,
                "mean_tpot_s": 0.01 + 0.002 * running,
                "weight": running + 1,
            }
            for running in range(1, 6)
        ]
        result = fit_tpot.weighted_fit(rows)
        self.assertAlmostEqual(result["base_s"], 0.01, places=12)
        self.assertAlmostEqual(result["per_running_s"], 0.002, places=12)
        self.assertAlmostEqual(result["weighted_r_squared"], 1.0, places=12)

    def test_monotone_load_fit_pools_reversed_observations(self):
        result = fit_tpot.monotone_load_fit(
            [
                {"load": 0.1, "tpot_s": 0.02, "source": "a"},
                {"load": 0.2, "tpot_s": 0.04, "source": "b"},
                {"load": 0.3, "tpot_s": 0.03, "source": "c"},
            ]
        )
        self.assertEqual(result["model_kind"], "load_piecewise_monotone")
        self.assertEqual(result["tpot_knots_s"], [0.02, 0.035, 0.035])

    def test_aggregate_interval_histograms(self):
        rows = [
            {
                "interval_tpot_count": "10",
                "interval_mean_tpot_s": "0.02",
                "tpot_histogram_delta_json": json.dumps(
                    {"0.01": 2, "0.02": 8, "0.04": 10, "+Inf": 10}
                ),
            },
            {
                "interval_tpot_count": "10",
                "interval_mean_tpot_s": "0.04",
                "tpot_histogram_delta_json": json.dumps(
                    {"0.01": 0, "0.02": 2, "0.04": 10, "+Inf": 10}
                ),
            },
        ]
        count, mean, p99 = analyze.aggregate_histogram(rows)
        self.assertEqual(count, 20)
        self.assertAlmostEqual(mean, 0.03, places=9)
        self.assertGreater(p99, 0.02)
        self.assertLessEqual(p99, 0.04)

    def test_itls_are_selected_by_token_end_time(self):
        benchmark = {
            "start_times": [100.0],
            "ttfts": [1.0],
            "itls": [[1.0, 2.0, 3.0]],
        }
        self.assertEqual(
            analyze.window_itls(benchmark, start=102.0, end=104.0),
            [1.0, 2.0],
        )

    def test_histogram_delta_is_serialized_by_bound(self):
        previous = record.parse_prometheus(
            'vllm:time_per_output_token_seconds_bucket{le="0.02"} 4\n'
            'vllm:time_per_output_token_seconds_bucket{le="+Inf"} 5\n'
        )
        current = record.parse_prometheus(
            'vllm:time_per_output_token_seconds_bucket{le="0.02"} 7\n'
            'vllm:time_per_output_token_seconds_bucket{le="+Inf"} 9\n'
        )
        self.assertEqual(
            json.loads(record.histogram_delta_json(previous, current)),
            {"+Inf": 4.0, "0.02": 3.0},
        )

    def test_current_request_tpot_histogram_delta_is_serialized(self):
        previous = record.parse_prometheus(
            'vllm:request_time_per_output_token_seconds_bucket{le="0.04"} 2\n'
            'vllm:request_time_per_output_token_seconds_bucket{le="+Inf"} 3\n'
        )
        current = record.parse_prometheus(
            'vllm:request_time_per_output_token_seconds_bucket{le="0.04"} 5\n'
            'vllm:request_time_per_output_token_seconds_bucket{le="+Inf"} 7\n'
        )
        self.assertEqual(
            json.loads(record.histogram_delta_json(previous, current)),
            {"+Inf": 4.0, "0.04": 3.0},
        )

    def test_complete_grid_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for band_index, band in enumerate(summarize.EXPECTED_BANDS):
                for rate in summarize.EXPECTED_RATES:
                    for rep in summarize.EXPECTED_REPS:
                        path = root / f"{band}_{rate}_{rep}.json"
                        path.write_text(
                            json.dumps(
                                {
                                    "status": "ACCEPTED",
                                    "load_band": band,
                                    "rep": rep,
                                    "inputs": {"target_rate_gib_s": rate},
                                    "observed": {
                                        "kv_usage_mean": 0.2 + 0.3 * band_index,
                                        "effective_rate_gib_s": rate,
                                        "p99_tpot_s": 0.02 + 0.01 * rate,
                                        "p99_itl_s": 0.03 + 0.01 * rate,
                                    },
                                }
                            ),
                            encoding="utf-8",
                        )
                        inputs.append(path)
            out = root / "summary.json"
            argv = [
                "summarize_phase9_calibration.py",
                "--inputs",
                *(str(path) for path in inputs),
                "--out",
                str(out),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("builtins.print"),
            ):
                summarize.main()
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["expected_conditions"], 36)
            self.assertEqual(len(result["cells"]), 12)


if __name__ == "__main__":
    unittest.main()
