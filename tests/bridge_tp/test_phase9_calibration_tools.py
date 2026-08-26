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


class TestCalibrationAnalysis(unittest.TestCase):
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
