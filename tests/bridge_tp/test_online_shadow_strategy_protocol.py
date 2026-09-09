# SPDX-License-Identifier: Apache-2.0

import unittest

from vllm.bridge_tp.online_shadow_strategy_protocol import (
    summarize_background_windows,
    validate_strategy_timing,
)


class TestOnlineShadowStrategyProtocol(unittest.TestCase):
    def test_window_summary_uses_token_intervals(self) -> None:
        summary = summarize_background_windows(
            [
                {
                    "job_id": "target-0",
                    "pool": "target",
                    "status": "COMPLETED",
                    "token_times_unix_s": [0.5, 0.9, 1.1, 1.5, 1.9, 2.1,
                                            2.5, 2.9, 3.1, 3.5],
                }
            ],
            shadow_start_unix_s=1.0,
            bridge_start_unix_s=2.0,
            committed_unix_s=3.0,
        )
        self.assertEqual(summary["PRE_SHADOW"]["samples"], 1)
        self.assertEqual(summary["SHADOW"]["samples"], 2)
        self.assertEqual(summary["BRIDGE"]["samples"], 2)
        self.assertEqual(summary["POST_COMMIT"]["samples"], 1)

    def test_strategy_timing_is_fail_closed(self) -> None:
        self.assertFalse(
            validate_strategy_timing(
                "S_NEW", shadow_start_unix_s=1, bridge_start_unix_s=2,
                history_start_unix_s=2,
            )
        )
        self.assertTrue(
            validate_strategy_timing(
                "S_NEW", shadow_start_unix_s=1, bridge_start_unix_s=2,
                history_start_unix_s=1,
            )
        )
        self.assertFalse(
            validate_strategy_timing(
                "S_NEW_OLD", shadow_start_unix_s=1, bridge_start_unix_s=2,
                history_start_unix_s=1.01,
            )
        )


if __name__ == "__main__":
    unittest.main()
