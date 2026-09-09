# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from vllm.bridge_tp.bridge_joint_interference_protocol import (
    MODES,
    build_cases,
    factorial_summary,
    validate_results,
)


class TestBridgeJointInterferenceProtocol(unittest.TestCase):
    def test_matrix_product(self) -> None:
        cases = build_cases(
            context_tokens=[1024, 4096],
            remote_fractions=[0.25, 0.75],
            target_load_repeats=[4, 8],
            copy_bytes_per_rank_step=[835584],
        )
        self.assertEqual(len(cases), 8)

    def test_factorial_interaction(self) -> None:
        rows = []
        for mode, slowdown in zip(MODES, [0.10, 0.05, 0.18], strict=True):
            rows.append(
                {
                    "cell_index": 1,
                    "mode": mode,
                    "context_tokens": 1024,
                    "remote_fraction": 0.5,
                    "target_load_repeats": 4,
                    "copy_bytes_per_rank_step": 835584,
                    "target_signed_slowdown_frac": slowdown,
                    "bridge_path_p50_ms": 1.0,
                }
            )
        summary = factorial_summary(rows)[0]
        self.assertAlmostEqual(summary["joint_interaction_slowdown_frac"], 0.03)

    def test_validation_fails_closed(self) -> None:
        row = {
            "cell_index": 1,
            "mode": "ATTENTION_ONLY",
            "status": "FAIL",
            "copy_verified": False,
            "attention_finite": False,
            "attention_accurate": False,
            "target_signed_slowdown_frac": float("nan"),
            "target_harm_ms": 0.0,
            "bridge_path_p50_ms": 0.0,
        }
        result = validate_results([row], [], expected_cells=1, measured_steps=2)
        self.assertEqual(result["status"], "FAIL")
        self.assertGreaterEqual(len(result["errors"]), 6)


if __name__ == "__main__":
    unittest.main()
