# SPDX-License-Identifier: Apache-2.0
"""Check the event boundaries used by the offline TP1 capacity ledger."""

from __future__ import annotations

import unittest

from tools.bridge_tp.audit_tp1_capacity_ledger import background_state


class TestBackgroundCapacityLedger(unittest.TestCase):
    def test_prefill_and_release_windows_are_not_claimed_exact(self) -> None:
        job = {
            "job_id": "source_000", "prompt_tokens": 32,
            "request_started_unix_s": 1.0, "first_token_unix_s": 2.0,
            "last_token_unix_s": 3.0, "request_ended_unix_s": 3.1,
            "token_times_unix_s": [2.0, 3.0],
        }
        blocks, _, uncertain = background_state([job], 1.5, 16, 0.5)
        self.assertEqual(blocks, 0)
        self.assertIn("source_000:prefill_allocation_time_unknown", uncertain)

        blocks, active, uncertain = background_state([job], 2.5, 16, 0.5)
        self.assertEqual(blocks, 3)
        self.assertEqual(active[0]["generated_tokens"], 1)
        self.assertEqual(uncertain, [])

        blocks, _, uncertain = background_state([job], 3.2, 16, 0.5)
        self.assertEqual(blocks, 0)
        self.assertIn("source_000:release_time_unknown", uncertain)

        blocks, _, uncertain = background_state([job], 3.7, 16, 0.5)
        self.assertEqual((blocks, uncertain), (0, []))


if __name__ == "__main__":
    unittest.main()
