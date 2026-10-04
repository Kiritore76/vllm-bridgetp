# SPDX-License-Identifier: Apache-2.0
"""Validate the calibration sample plan and frozen threshold candidate."""

from __future__ import annotations

import unittest

from tools.bridge_tp.run_slo_v6_ttft_calibration_a100 import (
    LENGTHS, prompt_for_length, schedule, summarize,
)


class TestSloV6TTFTCalibration(unittest.TestCase):
    def test_plan_has_interleaved_reproducible_samples(self) -> None:
        first = schedule(30, 2, 20261004)
        self.assertEqual(first, schedule(30, 2, 20261004))
        self.assertEqual(len(first), 2)
        for round_plan in first:
            self.assertEqual(len(round_plan), 105)
            self.assertEqual({length: round_plan.count(length) for length in LENGTHS},
                             {length: 15 for length in LENGTHS})
        self.assertNotEqual(first[0], first[1])

    def test_prompt_preserves_frozen_prefix_and_exact_length(self) -> None:
        base = [100, 101, 102]
        self.assertEqual(prompt_for_length(base, 2), [100, 101])
        self.assertEqual(prompt_for_length(base, 8), [100, 101, 102, 100, 101, 102, 100, 101])

    def test_summary_does_not_freeze_and_enforces_monotone_baseline(self) -> None:
        rows = [
            {"prompt_tokens": length, "round": round_index,
             "ttft_ms": float(1000 - length / 100)}
            for length in LENGTHS for round_index in range(2) for _ in range(15)
        ]
        summary = summarize(rows, 2, 30)
        self.assertEqual(summary["status"], "CANDIDATE_UNREVIEWED_NOT_FROZEN")
        self.assertEqual(len(summary["anchors"]), len(LENGTHS))
        self.assertEqual(
            summary["anchors"][-1]["monotone_reference_p95_ms"],
            summary["anchors"][0]["ttft_p95_ms"],
        )
        self.assertEqual(
            summary["anchors"][0]["candidate_ttft_slo_ms"],
            summary["anchors"][0]["ttft_p95_ms"] + 1000,
        )


if __name__ == "__main__":
    unittest.main()
