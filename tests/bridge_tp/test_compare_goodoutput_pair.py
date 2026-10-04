# SPDX-License-Identifier: Apache-2.0
"""A paired benefit number is valid only for matching experimental inputs."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.compare_goodoutput_pair import compare, load_arm
from tests.bridge_tp.test_audit_goodoutput import payload


class TestCompareGoodOutputPair(unittest.TestCase):
    def test_load_arm_uses_actual_online_runner_layout(self) -> None:
        value = payload()
        value["online/contract.json"]["paired_stay"] = True
        proxy = value["controller/response_proxy_stats.json"]
        proxy.update({"committed": False, "target_origin_tokens": 0,
                      "handoff_stall_s": None})
        value.pop("controller/target_response.json")
        source = value["controller/source_response.json"]
        source.update({"finish_reason": "length", "completed_unix_s": 0.2})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "r01_shadow_only"
            run.mkdir()
            (root / "contract.json").write_text(
                json.dumps(value.pop("online/contract.json")), encoding="utf-8")
            (root / "acceptance.json").write_text(json.dumps({
                "status": "PASS", "runs": [{
                    "root": "/server/results/stay/r01_shadow_only",
                    "acceptance": {"status": "PASS",
                                   "final_state": "COMPLETED_ON_TP1",
                                   "natural_start_decisions": 1},
                }],
            }), encoding="utf-8")
            for name, contents in value.items():
                path = run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(contents), encoding="utf-8")
            contract, report = load_arm(root, True)
        self.assertTrue(contract["paired_stay"])
        self.assertTrue(report["computable"])
        self.assertEqual(report["metrics"]["by_pool"]["anchor"]["requests"], 1)

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
                "computable": True, "v5_thresholds": {
                    "mean_tpot_ms": 50, "max_bad_interval_rate": 0.01,
                    "max_visible_interval_ms": 1000,
                },
                "metrics": {"by_pool": {"system": {
                    "requests": 13, "output_tokens": 13312,
                    "goodoutput_v4_tokens_s": rate,
                    "goodoutput_v5_tokens_s": rate,
                }}},
            }
        with patch(
            "tools.bridge_tp.compare_goodoutput_pair.load_arm",
            side_effect=[(contract, report(100)), (contract, report(120))],
        ):
            result = compare(Path("stay"), Path("migrate"))
        self.assertEqual(result["delta_goodoutput_v4_tokens_s"], 20)
        self.assertEqual(result["delta_goodoutput_v5_tokens_s"], 20)
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
