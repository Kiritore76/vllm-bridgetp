# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace

from tools.bridge_tp.build_shadow_strategy_online_manifest import build_manifest
from tools.bridge_tp.run_phase9_capacity_background import percentile
from tools.bridge_tp.run_shadow_rate_load_matrix import (
    parse_load_profiles,
    rate_label,
    resolve_design,
)
from tools.bridge_tp.run_shadow_strategy_online_validation import summarize_slo
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

    def test_slo_summary_counts_violations(self) -> None:
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
