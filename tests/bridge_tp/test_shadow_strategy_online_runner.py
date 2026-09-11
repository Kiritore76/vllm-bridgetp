# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.bridge_tp.build_shadow_strategy_online_manifest import build_manifest
from tools.bridge_tp.run_phase9_capacity_background import percentile
from tools.bridge_tp.run_shadow_rate_load_matrix import (
    parse_load_profiles,
    rate_label,
    resolve_design,
)
from tools.bridge_tp.run_shadow_strategy_online_validation import (
    build_controller_config_overrides,
    summarize_slo,
    write_measurements,
)
from vllm.bridge_tp.online_shadow_strategy_protocol import (
    summarize_background_windows,
    validate_strategy_timing,
)


class TestOnlineShadowManifest(unittest.TestCase):
    def test_builds_target_only_exact_token_jobs(self) -> None:
        manifest = build_manifest(
            target_jobs=2,
            prompt_tokens=4,
            output_tokens=3,
            max_model_len=8,
        )
        self.assertEqual(len(manifest["jobs"]), 2)
        self.assertTrue(all(job["pool"] == "target" for job in manifest["jobs"]))
        self.assertEqual(manifest["jobs"][0]["request"]["prompt"], [100] * 4)

    def test_rejects_context_overflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_manifest(prompt_tokens=6, output_tokens=3, max_model_len=8)


class TestOnlineStrategyTiming(unittest.TestCase):
    def test_controller_window_tracks_cli_boundaries(self) -> None:
        overrides = build_controller_config_overrides(
            trigger_output_tokens=64,
            cutover_output_tokens=160,
            fixed_rate_gib_s=0.4,
        )
        self.assertEqual(overrides["handoff_output_tokens"], 96)
        expected_rate = 0.4 * 1024**3
        self.assertEqual(overrides["rate"]["b_min_bytes_s"], expected_rate)
        self.assertEqual(overrides["rate"]["b_max_bytes_s"], expected_rate)

    def test_s_new_requires_history_at_bridge(self) -> None:
        self.assertFalse(
            validate_strategy_timing(
                "S_NEW",
                shadow_start_unix_s=10.0,
                bridge_start_unix_s=12.0,
                history_start_unix_s=12.0,
            )
        )
        self.assertTrue(
            validate_strategy_timing(
                "S_NEW",
                shadow_start_unix_s=10.0,
                bridge_start_unix_s=12.0,
                history_start_unix_s=10.5,
            )
        )

    def test_s_new_old_requires_history_at_shadow(self) -> None:
        self.assertFalse(
            validate_strategy_timing(
                "S_NEW_OLD",
                shadow_start_unix_s=10.0,
                bridge_start_unix_s=12.0,
                history_start_unix_s=10.0,
            )
        )


class TestOnlineWindows(unittest.TestCase):
    def test_partitions_target_tpot(self) -> None:
        results = [
            {
                "job_id": "target_000",
                "pool": "target",
                "status": "COMPLETED",
                "token_times_unix_s": [9.0, 9.5, 10.5, 11.0, 11.5,
                                        12.5, 12.8, 13.5, 14.0],
            }
        ]
        windows = summarize_background_windows(
            results,
            shadow_start_unix_s=10.0,
            bridge_start_unix_s=12.0,
            committed_unix_s=13.0,
        )
        self.assertEqual(windows["PRE_SHADOW"]["samples"], 1)
        self.assertEqual(windows["SHADOW"]["samples"], 2)
        self.assertEqual(windows["BRIDGE"]["samples"], 1)
        self.assertEqual(windows["POST_COMMIT"]["samples"], 1)
        self.assertEqual(percentile([1.0, 3.0], 0.5), 2.0)

    def test_writes_shadow_only_architecture_pair(self) -> None:
        windows = {
            name: {
                "jobs": 1,
                "samples": 1,
                "tpot_p50_ms": 1.0,
                "tpot_p95_ms": 1.0,
                "tpot_p99_ms": 1.0,
            }
            for name in ("PRE_SHADOW", "SHADOW", "BRIDGE", "POST_COMMIT")
        }
        runs = []
        for architecture, strategy, stall in (
            ("BRIDGE", "S_NEW", 20.0),
            ("SHADOW_ONLY", "S_NEW_OLD", 10.0),
        ):
            architecture_windows = dict(windows)
            if architecture == "SHADOW_ONLY":
                architecture_windows["FINAL_SYNC"] = dict(windows["BRIDGE"])
            runs.append(
                {
                    "repetition": 1,
                    "architecture": architecture,
                    "strategy": strategy,
                    "acceptance": {
                        "status": "PASS",
                        "shadow_duration_ms": 100.0,
                        "bridge_to_commit_ms": stall,
                        "handoff_stall_ms": stall,
                        "source_origin_tokens": 10,
                        "target_origin_tokens": 20,
                        "target_tpot_windows": architecture_windows,
                    },
                }
            )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_measurements(root, runs)
            paired = (root / "paired_comparisons.csv").read_text(
                encoding="utf-8"
            )
            measurements = (root / "measurements.csv").read_text(
                encoding="utf-8"
            )
        self.assertIn("final_sync_ms_saved_by_shadow_only", paired)
        self.assertIn("10.0", paired)
        self.assertIn("final_sync_tpot_p99_ms", measurements)

    def test_slo_summary_counts_token_and_request_violations(self) -> None:
        summary = summarize_slo(
            [
                {
                    "status": "COMPLETED",
                    "token_times_unix_s": [1.0, 1.01, 1.08],
                    "tpot_p99_ms": 70.0,
                    "ttft_ms": 1200.0,
                    "e2e_ms": 2000.0,
                }
            ],
            tpot_ms=50.0,
            ttft_ms=1000.0,
            e2e_ms=3000.0,
        )
        self.assertEqual(summary["tpot_interval_violations"], 1)
        self.assertEqual(summary["request_p99_tpot_violations"], 1)
        self.assertEqual(summary["ttft_violations"], 1)
        self.assertEqual(summary["e2e_violations"], 0)


class TestShadowRateLoadMatrix(unittest.TestCase):
    def test_default_formal_design_covers_three_loads_and_five_rates(self) -> None:
        args = Namespace(
            phase="formal",
            load_profile=None,
            rates_gib_s=None,
            repetitions=None,
            minimum_window_samples=None,
        )
        loads, rates, repetitions, minimum_samples = resolve_design(args)
        self.assertEqual(loads, [("low", 2), ("medium", 8), ("high", 24)])
        self.assertEqual(rates, [0.2, 0.4, 0.8, 1.2, 0.0])
        self.assertEqual(repetitions, 4)
        self.assertEqual(minimum_samples, 128)

    def test_load_profiles_and_rate_labels_are_unambiguous(self) -> None:
        self.assertEqual(
            parse_load_profiles(["quiet:3", "busy:20"], "smoke"),
            [("quiet", 3), ("busy", 20)],
        )
        self.assertEqual(rate_label(0.4), "0p4gibs")
        self.assertEqual(rate_label(0.0), "unlimited")

    def test_rejects_duplicate_load_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_load_profiles(["busy:4", "busy:8"], "formal")


if __name__ == "__main__":
    unittest.main()
