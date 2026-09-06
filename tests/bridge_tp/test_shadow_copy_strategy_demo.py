# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from tools.bridge_tp.run_shadow_copy_strategy_demo import empirical_quantile
from tools.bridge_tp.shadow_copy_strategy import (
    InterferenceCell,
    ShadowTransferInputs,
    compare_shadow_transfers,
    interpolate_tpot,
    kv_bytes_per_token,
    simulate_history_backfill,
    simulate_new_kv_only,
)


class TestShadowCopyStrategyDemo(unittest.TestCase):
    def inputs(self, **overrides) -> ShadowTransferInputs:
        cell = InterferenceCell(
            load_band="low",
            repetition=1,
            target_rate_gib_s=0.4,
            effective_rate_gib_s=0.4,
            target_load_frac=0.2,
            baseline_mean_tpot_s=0.05,
            copy_mean_tpot_s=0.06,
            baseline_p99_tpot_s=0.08,
            copy_p99_tpot_s=0.09,
            baseline_p99_itl_s=0.1,
            copy_p99_itl_s=0.13,
        )
        values = {
            "history_tokens": 64,
            "remaining_tokens": 32,
            "block_size": 16,
            "kv_bytes_per_token": 1024,
            "source_load_frac": 0.4,
            "source_tpot_s": 0.03,
            "interference": cell,
        }
        values.update(overrides)
        return ShadowTransferInputs(**values)

    def test_qwen_geometry(self) -> None:
        self.assertEqual(
            kv_bytes_per_token(
                num_layers=48,
                num_kv_heads=8,
                head_size=128,
                dtype_bytes=2,
            ),
            196608,
        )

    def test_interpolate_tpot(self) -> None:
        self.assertAlmostEqual(
            interpolate_tpot(0.25, [0.2, 0.3], [0.02, 0.04]),
            0.03,
        )
        with self.assertRaises(ValueError):
            interpolate_tpot(0.1, [0.2, 0.3], [0.02, 0.04])

    def test_empirical_higher_quantile(self) -> None:
        self.assertEqual(empirical_quantile([1, 2, 3, 4], 0.5), 2)
        self.assertEqual(empirical_quantile([1, 2, 3, 4], 0.9), 4)

    def test_history_backfill_reaches_takeover_ready(self) -> None:
        result = simulate_history_backfill(self.inputs())
        self.assertTrue(result.takeover_ready)
        self.assertEqual(result.history_backlog_end_bytes, 0)
        self.assertEqual(result.outcome, "TAKEOVER_READY")

    def test_history_backfill_can_miss_request_end(self) -> None:
        slow_cell = InterferenceCell(
            **{
                **self.inputs().interference.__dict__,
                "effective_rate_gib_s": 1e-6,
            }
        )
        result = simulate_history_backfill(
            self.inputs(
                history_tokens=1024,
                remaining_tokens=2,
                interference=slow_cell,
            )
        )
        self.assertFalse(result.takeover_ready)
        self.assertGreater(result.history_backlog_end_tokens, 0)

    def test_new_only_keeps_all_history_backlog(self) -> None:
        inputs = self.inputs()
        result = simulate_new_kv_only(inputs)
        self.assertFalse(result.takeover_ready)
        self.assertEqual(result.history_bytes_sent, 0)
        self.assertEqual(result.history_backlog_end_tokens, 64)
        self.assertLess(result.copy_duty_cycle, 1)
        self.assertAlmostEqual(
            result.maximum_new_kv_lag_s,
            1024 / (0.4 * 1024**3),
        )

    def test_target_delay_uses_paired_c2_penalty(self) -> None:
        result = simulate_new_kv_only(self.inputs())
        expected_tokens = (
            result.copy_active_time_s
            / self.inputs().interference.copy_mean_tpot_s
        )
        self.assertAlmostEqual(
            result.estimated_target_delay_s,
            expected_tokens * 0.01,
        )

    def test_comparison_separates_overhead_and_readiness(self) -> None:
        result = compare_shadow_transfers(self.inputs())
        self.assertEqual(result["lower_bytes"], "new_kv_only")
        self.assertEqual(
            result["lower_estimated_interference"], "new_kv_only"
        )
        self.assertEqual(result["takeover_ready"], "history_backfill")


if __name__ == "__main__":
    unittest.main()
