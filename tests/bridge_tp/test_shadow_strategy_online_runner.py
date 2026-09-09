# SPDX-License-Identifier: Apache-2.0

import unittest

from tools.bridge_tp.build_shadow_strategy_online_manifest import build_manifest
from tools.bridge_tp.run_phase9_capacity_background import percentile
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


if __name__ == "__main__":
    unittest.main()
