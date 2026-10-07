# SPDX-License-Identifier: Apache-2.0
"""Urgent prearming requires an exact scheduled and frozen boundary."""
import unittest

from tools.bridge_tp.run_shadow_strategy_online_validation import (
    valid_urgent_prearmed_selection,
)


class TestUrgentPrearmedSelection(unittest.TestCase):
    def test_exact_prearmed_boundary_is_a_separate_selection_path(self):
        self.assertTrue(valid_urgent_prearmed_selection(
            [{"cutover_output_tokens": 528, "unix_s": 10.0,
              "source_time_to_guard_s": 4.5}],
            [{"cutover_output_tokens": 528, "unix_s": 10.1}], 528, 12.0,
        ))

    def test_missing_duplicate_wrong_or_late_evidence_is_rejected(self):
        arm = {"cutover_output_tokens": 528, "unix_s": 10.0,
               "source_time_to_guard_s": 4.5}
        candidate = {"cutover_output_tokens": 528, "unix_s": 10.1}
        for arms, candidates, boundary in (
            ([], [candidate], 528), ([arm, arm], [candidate], 528),
            ([arm], [], 528), ([arm], [candidate, candidate], 528),
            ([arm], [candidate], 529),
            ([{**arm, "source_time_to_guard_s": float("nan")}],
             [candidate], 528),
            ([arm], [{**candidate, "unix_s": 13.0}], 528),
        ):
            self.assertFalse(valid_urgent_prearmed_selection(
                arms, candidates, boundary, 12.0,
            ))
