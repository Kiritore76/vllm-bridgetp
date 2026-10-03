# SPDX-License-Identifier: Apache-2.0
"""A paired benefit number is valid only for matching experimental inputs."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.compare_goodoutput_pair import compare


class TestCompareGoodOutputPair(unittest.TestCase):
    def test_matching_pair_reports_direction_without_significance_claim(self) -> None:
        contract = {
            "revision": "a", "manifest_sha256": "b",
            "survival_table_sha256": "c", "guard_file_sha256": "d",
            "predictor_checkpoint_sha256": "e",
            "manager_m5_predictor_shadow": True,
            "anchor_max_tokens": 1024,
        }
        def report(rate: float) -> dict:
            return {
                "computable": True, "v4_thresholds": {"mean_tpot_ms": 50},
                "metrics": {"by_pool": {"system": {
                    "requests": 13, "output_tokens": 13312,
                    "goodoutput_v4_tokens_s": rate,
                }}},
            }
        with patch(
            "tools.bridge_tp.compare_goodoutput_pair.load_arm",
            side_effect=[(contract, report(100)), (contract, report(120))],
        ):
            result = compare(Path("stay"), Path("migrate"))
        self.assertEqual(result["delta_goodoutput_v4_tokens_s"], 20)
        self.assertEqual(result["status"], "PILOT_COMPARABLE_NOT_STATISTICAL")

    def test_rejects_different_manifest(self) -> None:
        base = {
            "revision": "a", "manifest_sha256": "b",
            "survival_table_sha256": "c", "guard_file_sha256": "d",
            "predictor_checkpoint_sha256": "e",
            "manager_m5_predictor_shadow": True,
            "anchor_max_tokens": 1024,
        }
        other = {**base, "manifest_sha256": "different"}
        with patch(
            "tools.bridge_tp.compare_goodoutput_pair.load_arm",
            side_effect=[(base, {}), (other, {})],
        ):
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                compare(Path("stay"), Path("migrate"))


if __name__ == "__main__":
    unittest.main()
