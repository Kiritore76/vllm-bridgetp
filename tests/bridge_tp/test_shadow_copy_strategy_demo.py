# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from tools.bridge_tp.shadow_copy_strategy import (
    ShadowCopyInputs,
    compare_shadow_strategies,
    kv_bytes_per_token,
    newest_first_history_blocks,
    remote_attention_bytes_per_token,
    simulate_history_backfill,
    simulate_new_kv_bridge,
)
from tools.bridge_tp.run_shadow_copy_strategy_demo import flatten_decision


class TestShadowCopyStrategyDemo(unittest.TestCase):
    def inputs(self, **overrides) -> ShadowCopyInputs:
        values = {
            "history_tokens": 64,
            "remaining_tokens": 32,
            "block_size": 16,
            "kv_bytes_per_token": 1024,
            "copy_rate_bytes_s": 1024 * 1024,
            "source_tpot_s": 0.03,
            "target_tpot_s": 0.01,
            "remote_attention_penalty_s": 0.002,
            "remote_attention_bytes_per_token": 2048,
            "bridge_start_tokens": 1,
        }
        values.update(overrides)
        return ShadowCopyInputs(**values)

    def test_history_blocks_start_at_shadow_boundary(self) -> None:
        self.assertEqual(
            newest_first_history_blocks(40, 16),
            ((24, 40), (8, 24), (0, 8)),
        )

    def test_qwen_geometry_counts_aggregate_bytes(self) -> None:
        self.assertEqual(
            kv_bytes_per_token(
                num_layers=48,
                num_kv_heads=8,
                head_size=128,
                dtype_bytes=2,
            ),
            196608,
        )
        self.assertEqual(
            remote_attention_bytes_per_token(
                num_layers=48,
                hidden_size=5120,
                dtype_bytes=2,
            ),
            983040,
        )

    def test_history_backfill_can_reach_standalone_takeover(self) -> None:
        result = simulate_history_backfill(self.inputs())
        self.assertEqual(result.outcome, "TAKEOVER")
        self.assertTrue(result.standalone_takeover)
        self.assertFalse(result.source_needed_after_switch)
        self.assertEqual(result.history_bytes_sent, 64 * 1024)
        self.assertEqual(
            result.history_block_order,
            ((48, 64), (32, 48), (16, 32), (0, 16)),
        )

    def test_history_backfill_cancels_when_source_finishes_first(self) -> None:
        result = simulate_history_backfill(
            self.inputs(
                history_tokens=1024,
                remaining_tokens=2,
                copy_rate_bytes_s=1024,
            )
        )
        self.assertEqual(result.outcome, "SOURCE_FINISHED_BEFORE_TAKEOVER")
        self.assertFalse(result.standalone_takeover)
        self.assertEqual(result.completion_time_s, 0.06)

    def test_new_kv_bridge_keeps_source_until_request_end(self) -> None:
        result = simulate_new_kv_bridge(self.inputs())
        self.assertEqual(result.outcome, "BRIDGE_TO_REQUEST_END")
        self.assertFalse(result.standalone_takeover)
        self.assertTrue(result.source_needed_after_switch)
        self.assertEqual(result.source_release_time_s, result.completion_time_s)
        self.assertEqual(result.history_bytes_sent, 0)
        self.assertGreater(result.remote_attention_bytes_sent, 0)

    def test_new_kv_bridge_cancels_if_source_finishes_before_start(self) -> None:
        result = simulate_new_kv_bridge(
            self.inputs(
                remaining_tokens=1,
                copy_rate_bytes_s=1,
            )
        )
        self.assertEqual(result.outcome, "SOURCE_FINISHED_BEFORE_BRIDGE")
        self.assertEqual(result.completion_time_s, 0.03)
        self.assertEqual(result.target_tokens_after_switch, 0)
        self.assertEqual(result.remote_attention_bytes_sent, 0)

    def test_comparison_reports_separate_objectives(self) -> None:
        result = compare_shadow_strategies(self.inputs())
        self.assertIn(
            result["latency_winner"],
            ("history_backfill", "new_kv_bridge", "tie"),
        )
        self.assertEqual(result["source_release_winner"], "history_backfill")
        self.assertIsNotNone(result["new_kv_bridge_remote_penalty_break_even_ms"])

    def test_decision_row_pairs_both_strategies(self) -> None:
        comparison = compare_shadow_strategies(self.inputs())
        row = flatten_decision(7, comparison)
        self.assertEqual(row["comparison_id"], 7)
        self.assertIn("history_completion_ms", row)
        self.assertIn("bridge_completion_ms", row)
        self.assertEqual(row["source_release_winner"], "history_backfill")


if __name__ == "__main__":
    unittest.main()
