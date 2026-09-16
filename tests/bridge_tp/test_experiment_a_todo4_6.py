# SPDX-License-Identifier: Apache-2.0

import unittest

from tools.bridge_tp.run_experiment_a_todo4_6 import (
    MODES,
    aggregate,
    design,
    paired,
)


class TestExperimentATodoDesign(unittest.TestCase):
    def test_todo4_fixes_three_round_baseline(self) -> None:
        cells = design("todo4")
        self.assertEqual(len(cells), 12)
        for repetition in range(1, 4):
            modes = {cell.mode for cell in cells if cell.repetition == repetition}
            self.assertEqual(modes, set(MODES))
        stop = next(cell for cell in cells if cell.mode == "STOP_AND_COPY")
        shadow = next(cell for cell in cells if cell.mode == "SHADOW_ONLY")
        self.assertEqual(stop.commit_tokens, 64)
        self.assertEqual(shadow.commit_tokens, 256)

    def test_todo5_covers_lengths_modes_and_repetitions(self) -> None:
        cells = design("todo5")
        self.assertEqual(len(cells), 60)
        keys = {
            (cell.output_tokens, cell.mode, cell.repetition) for cell in cells
        }
        expected = {
            (output, mode, repetition)
            for output in (256, 1024, 4096)
            for mode in MODES
            for repetition in range(1, 6)
        }
        self.assertEqual(keys, expected)
        short_shadow = next(
            cell
            for cell in cells
            if cell.output_tokens == 256 and cell.mode == "SHADOW_ONLY"
        )
        self.assertEqual(short_shadow.commit_tokens, 128)

    def test_todo6_is_same_commit_boundary(self) -> None:
        cells = design("todo6")
        self.assertEqual(len(cells), 20)
        self.assertTrue(all(cell.commit_tokens == 256 for cell in cells))
        self.assertEqual({cell.output_tokens for cell in cells}, {1024, 4096})
        self.assertEqual(
            {cell.mode for cell in cells}, {"STOP_AND_COPY", "SHADOW_ONLY"}
        )


class TestExperimentATodoSummaries(unittest.TestCase):
    def test_paired_net_gain_and_same_commit_metrics(self) -> None:
        base = {
            "repetition": 1,
            "output_tokens": 1024,
            "status": "PASS",
            "itl_p95_ms": 1.0,
            "itl_p99_ms": 2.0,
        }
        a1_rows = [
            base | {"mode": "ALWAYS_TP1", "e2e_ms": 100.0,
                    "handoff_stall_ms": None},
            base | {"mode": "ALWAYS_TP4", "e2e_ms": 60.0,
                    "handoff_stall_ms": None},
            base | {"mode": "STOP_AND_COPY", "e2e_ms": 90.0,
                    "handoff_stall_ms": 20.0},
            base | {"mode": "SHADOW_ONLY", "e2e_ms": 80.0,
                    "handoff_stall_ms": 10.0},
        ]
        gains = paired(a1_rows, "todo5")
        shadow = next(row for row in gains if row["mode"] == "SHADOW_ONLY")
        self.assertEqual(shadow["net_gain_vs_always_tp1_ms"], 20.0)
        summary = aggregate(a1_rows)
        self.assertEqual(len(summary), 4)

        mechanism = paired(a1_rows[2:], "todo6")
        self.assertEqual(mechanism[0]["shadow_e2e_advantage_ms"], 10.0)
        self.assertEqual(mechanism[0]["handoff_ms_hidden_by_shadow"], 10.0)


if __name__ == "__main__":
    unittest.main()
