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


class TestCalibrationAnalysis(unittest.TestCase):
    def test_load_pilot_selection_is_band_scoped(self):
        conditions = [
            {
                "qps": 0.2,
                "kv_usage_mean": 0.20,
                "band_fractions": {"low": 0.9, "medium": 0.0, "high": 0.0},
            },
            {
                "qps": 0.4,
                "kv_usage_mean": 0.50,
                "band_fractions": {"low": 0.0, "medium": 0.9, "high": 0.0},
            },
            {
                "qps": 0.8,
                "kv_usage_mean": 0.80,
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
            "kv_usage_mean": 0.50,
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
