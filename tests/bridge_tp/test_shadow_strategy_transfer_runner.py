# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import unittest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - documentation hosts
    torch = None

if torch is not None:
    from tools.bridge_tp.run_shadow_strategy_transfer_validation import (
        aggregate_bytes_for_tokens,
        make_cases,
        scheduled_load_repeats,
        validate_rows,
        validate_step_rows,
    )


@unittest.skipIf(torch is None, "torch is required")
class TestShadowStrategyTransferRunner(unittest.TestCase):
    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            history_tokens=[1024],
            shadow_steps=[8, 32],
            target_load_repeats=[0, 4, 8],
            outcomes=["CANCEL", "COMMIT"],
            strategy_order=["S_NEW", "S_NEW_OLD"],
            block_size=16,
            num_layers=48,
            num_kv_heads=8,
            target_tp_size=4,
            head_dim=128,
            dtype="bfloat16",
        )

    def test_qwen_full_geometry_transfer_bytes(self) -> None:
        args = self.args()
        self.assertEqual(aggregate_bytes_for_tokens(args, 1), 196608)
        self.assertEqual(aggregate_bytes_for_tokens(args, 16), 3145728)

    def test_pilot_matrix_has_twelve_paired_cells(self) -> None:
        self.assertEqual(len(make_cases(self.args())), 12)

    def test_strategy_order_must_be_a_permutation(self) -> None:
        args = self.args()
        args.strategy_order = ["S_NEW", "S_NEW"]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            make_cases(args)

    def test_dynamic_load_profiles_are_phase_causal(self) -> None:
        self.assertEqual(scheduled_load_repeats("CONSTANT", 8, "SHADOW", 3), 8)
        self.assertEqual(
            scheduled_load_repeats("STEP_UP_BRIDGE", 8, "SHADOW", 3), 0
        )
        self.assertEqual(
            scheduled_load_repeats("STEP_UP_BRIDGE", 8, "BRIDGE", 3), 8
        )
        self.assertEqual(
            scheduled_load_repeats("STEP_DOWN_BRIDGE", 8, "SHADOW", 3), 8
        )
        self.assertEqual(scheduled_load_repeats("PULSE", 8, "BRIDGE", 4), 8)
        self.assertEqual(scheduled_load_repeats("PULSE", 8, "BRIDGE", 5), 0)
        self.assertEqual(scheduled_load_repeats("OSCILLATE", 8, "SHADOW", 2), 8)
        self.assertEqual(scheduled_load_repeats("OSCILLATE", 8, "SHADOW", 3), 0)

    def test_acceptance_fails_closed(self) -> None:
        row = {
            "case_index": 1,
            "strategy": "S_NEW",
            "outcome": "COMMIT",
            "status": "FAIL",
            "all_payloads_verified": False,
            "actual_transfer_bytes": 1,
            "expected_transfer_bytes": 2,
            "source_released_tokens_in_shadow": 1,
            "shadow_history_tokens": 0,
            "bridge_steps": 1,
            "takeover_ready": False,
        }
        acceptance = validate_rows([row], expected_rows=2)
        self.assertEqual(acceptance["status"], "FAIL")
        self.assertGreaterEqual(len(acceptance["errors"]), 7)

    def test_raw_steps_must_reproduce_summary(self) -> None:
        summary = {
            "case_index": 1,
            "shadow_steps": 1,
            "bridge_steps": 1,
            "actual_transfer_bytes": 30,
            "shadow_target_interference_harm_ms": 0.1,
            "bridge_target_interference_harm_ms": 0.0,
        }
        valid_steps = [
            {
                "case_index": 1,
                "phase": "SHADOW",
                "actual_bytes": 10,
                "all_verified": True,
                "scheduled_load_repeats": 4,
                "target_load_ms": 1.1,
                "target_control_before_ms": 1.0,
                "target_control_after_ms": 1.0,
                "target_control_ms": 1.0,
                "target_interference_delta_ms": 0.1,
                "target_interference_harm_ms": 0.1,
            },
            {
                "case_index": 1,
                "phase": "BRIDGE",
                "actual_bytes": 20,
                "all_verified": True,
                "scheduled_load_repeats": 0,
                "target_load_ms": 0.0,
                "target_control_before_ms": 0.0,
                "target_control_after_ms": 0.0,
                "target_control_ms": 0.0,
                "target_interference_delta_ms": 0.0,
                "target_interference_harm_ms": 0.0,
            },
        ]
        self.assertEqual(validate_step_rows([summary], valid_steps), [])
        valid_steps.pop()
        self.assertGreaterEqual(len(validate_step_rows([summary], valid_steps)), 2)


if __name__ == "__main__":
    unittest.main()
