# SPDX-License-Identifier: Apache-2.0
"""Verify length interpolation and visible-stream SLO decisions."""

from __future__ import annotations

import copy
import unittest

from tests.bridge_tp.test_audit_goodoutput import payload
from tools.bridge_tp.audit_slo_v6 import audit_v6_payload, ttft_limit_ms


REFERENCE = {
    "slo_version": "v6",
    "anchors": [
        {"prompt_tokens": 128, "reference_p95_ms": 100.0},
        {"prompt_tokens": 512, "reference_p95_ms": 300.0},
    ],
    "queue_allowance_ms": 1000.0,
    "primary_tpot_mean_ms": 100.0,
    "primary_slow_interval_threshold_ms": 100.0,
    "primary_max_slow_interval_rate": 0.01,
    "max_visible_interval_ms": 1000.0,
    "max_handoff_ms": 1000.0,
}


def test_payload() -> dict:
    value = payload()
    value["online/contract.json"]["anchor_prompt_tokens"] = 128
    value["background/background_manifest.json"] = {
        "jobs": [{"job_id": "peer", "pool": "source", "request": {
            "prompt": [100] * 320, "ignore_eos": False,
        }}],
    }
    return value


class TestSloV6Audit(unittest.TestCase):
    def test_interpolates_per_request_length(self) -> None:
        self.assertEqual(ttft_limit_ms(320, REFERENCE), 1200.0)
        self.assertEqual(ttft_limit_ms(1, REFERENCE), 1100.0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            ttft_limit_ms(513, REFERENCE)

    def test_scores_anchor_once_and_reports_goodoutput(self) -> None:
        report = audit_v6_payload(test_payload(), REFERENCE)
        self.assertTrue(report["computable"], report["errors"])
        self.assertEqual(report["metrics"]["requests"], 2)
        self.assertEqual(report["metrics"]["slo_success_requests"], 2)
        self.assertEqual(report["metrics"]["good_output_tokens"], 6)
        self.assertEqual(report["reference_applicability"],
                         "UNVERIFIED_GPU_ROSTER")
        self.assertEqual(report["request_rows"][0]["prompt_tokens"], 320)
        self.assertEqual(report["request_rows"][0]["ttft_limit_ms"], 1200)

    def test_long_visible_pause_fails_despite_low_average(self) -> None:
        value = test_payload()
        peer = value["background/background_summary.json"]["results"][0]
        times = [0.1 + index * 0.001 for index in range(100)]
        times.append(times[-1] + 1.001)
        peer.update({"output_tokens": len(times), "token_times_unix_s": times,
                     "request_ended_unix_s": 1.3})
        report = audit_v6_payload(value, REFERENCE)
        self.assertTrue(report["computable"], report["errors"])
        row = report["request_rows"][0]
        self.assertLess(row["mean_tpot_ms"], 100)
        self.assertFalse(row["slo_success"])
        self.assertGreater(row["max_visible_interval_ms"], 1000)
        self.assertIn("MAX_VISIBLE_INTERVAL", row["failure_reasons"])

    def test_missing_prompt_ids_fails_closed(self) -> None:
        value = copy.deepcopy(test_payload())
        value["background/background_manifest.json"]["jobs"][0][
            "request"
        ]["prompt"] = "text without frozen token ids"
        report = audit_v6_payload(value, REFERENCE)
        self.assertFalse(report["computable"])
        self.assertIn("original prompt token ids", " ".join(report["errors"]))

    def test_too_many_slow_intervals_fail(self) -> None:
        value = test_payload()
        peer = value["background/background_summary.json"]["results"][0]
        times = [0.1 + index * 0.001 for index in range(99)]
        times.append(times[-1] + 0.101)
        times.append(times[-1] + 0.201)
        peer.update({"output_tokens": len(times), "token_times_unix_s": times,
                     "request_ended_unix_s": 0.5})
        report = audit_v6_payload(value, REFERENCE)
        self.assertTrue(report["computable"], report["errors"])
        row = report["request_rows"][0]
        self.assertLess(row["mean_tpot_ms"], 100)
        self.assertGreater(row["slow_interval_rate"], 0.01)
        self.assertFalse(row["slo_success"])
        self.assertIn("SLOW_INTERVAL_RATE", row["failure_reasons"])

    def test_reference_provenance_is_reported_separately(self) -> None:
        config = {**REFERENCE, "gpu_uuids": ["gpu-a"],
                  "model_config_sha256": "model", "base_manifest_sha256": "input"}
        matching = {"gpu_uuids": ["gpu-a"],
                    "model_config_sha256": "model"}
        verified = audit_v6_payload(test_payload(), config, matching)
        self.assertEqual(verified["reference_applicability"],
                         "VERIFIED_GPU_AND_MODEL_CONFIG")
        mismatched = audit_v6_payload(
            test_payload(), config, {**matching, "gpu_uuids": ["gpu-b"]}
        )
        self.assertTrue(mismatched["computable"])
        self.assertEqual(mismatched["reference_applicability"],
                         "REFERENCE_INPUT_MISMATCH")


if __name__ == "__main__":
    unittest.main()
