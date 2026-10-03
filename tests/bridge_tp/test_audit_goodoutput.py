# SPDX-License-Identifier: Apache-2.0
"""Checks the metric boundary, especially proxy stitching and missing data."""

from __future__ import annotations

import copy
import unittest

from tools.bridge_tp.audit_goodoutput import audit_payload


def payload() -> dict:
    return {
        "online/contract.json": {"slo_thresholds": {
            "tpot_ms": 50, "ttft_ms": 1000,
            "e2e_ms": 60000, "handoff_ms": 1000,
        }},
        "background/background_summary.json": {
            "jobs": 1, "completed": 1,
            "results": [{
                "job_id": "peer", "pool": "source", "status": "COMPLETED",
                "output_tokens": 3, "request_started_unix_s": 0.0,
                "request_ended_unix_s": 0.25,
                "token_times_unix_s": [0.1, 0.12, 0.2],
            }],
        },
        "controller/response_proxy_stats.json": {
            "external_request_id": "anchor", "emitted_tokens": 3,
            "handoff_stall_s": 0.03,
            "emitted": [
                {"index": index, "unix_s": time}
                for index, time in enumerate((0.1, 0.13, 0.16))
            ],
        },
        "controller/source_response.json": {
            "request_started_unix_s": 0.0, "finish_reason": "abort",
            "completed_unix_s": 0.14,
        },
        "controller/target_response.json": {
            "finish_reason": "length", "completed_unix_s": 0.2,
        },
    }


class TestGoodOutputAudit(unittest.TestCase):
    def test_anchor_is_counted_once_and_one_bad_gap_loses_one_token(self) -> None:
        report = audit_payload(payload())
        self.assertTrue(report["computable"], report["errors"])
        system = report["metrics"]["by_pool"]["system"]
        self.assertEqual(system["requests"], 2)
        self.assertEqual(system["output_tokens"], 6)
        self.assertEqual(system["good_tokens"], 5)
        self.assertEqual(system["strict_request_good_tokens"], 3)
        self.assertEqual(system["request_slo_p99_good_tokens"], 3)
        self.assertEqual(system["bad_intervals"], 1)
        self.assertAlmostEqual(system["goodoutput_tokens_s"], 20.0)
        self.assertIsNone(report["benefit_vs_stay"])

    def test_ttft_failure_removes_request_tokens(self) -> None:
        value = payload()
        peer = value["background/background_summary.json"]["results"][0]
        peer["request_started_unix_s"] = -1.0
        report = audit_payload(value)
        self.assertTrue(report["computable"], report["errors"])
        system = report["metrics"]["by_pool"]["system"]
        self.assertEqual(system["good_tokens"], 3)
        self.assertEqual(system["strict_request_good_tokens"], 3)

    def test_missing_or_corrupt_timestamps_are_not_assumed_good(self) -> None:
        for change in ("missing", "nonmonotonic"):
            with self.subTest(change=change):
                value = copy.deepcopy(payload())
                peer = value["background/background_summary.json"]["results"][0]
                peer["token_times_unix_s"] = (
                    None if change == "missing" else [0.1, 0.2, 0.12]
                )
                report = audit_payload(value)
                self.assertFalse(report["computable"])
                self.assertIsNone(report["metrics"])
                self.assertTrue(report["errors"])

    def test_observed_failed_request_counts_zero_instead_of_disappearing(self) -> None:
        value = payload()
        peer = value["background/background_summary.json"]["results"][0]
        peer["status"] = "FAILED"
        peer.pop("token_times_unix_s")
        value["background/background_summary.json"]["completed"] = 0
        report = audit_payload(value)
        self.assertTrue(report["computable"], report["errors"])
        system = report["metrics"]["by_pool"]["system"]
        self.assertEqual(system["requests"], 2)
        self.assertEqual(system["good_tokens"], 3)

    def test_one_tail_gap_can_fail_legacy_strict_but_pass_p99(self) -> None:
        value = payload()
        peer = value["background/background_summary.json"]["results"][0]
        times = [0.1 + index * 0.001 for index in range(100)]
        times.append(times[-1] + 0.1)
        peer["output_tokens"] = len(times)
        peer["token_times_unix_s"] = times
        peer["request_ended_unix_s"] = 0.35
        report = audit_payload(value)
        self.assertTrue(report["computable"], report["errors"])
        source = report["metrics"]["by_pool"]["source"]
        self.assertEqual(source["bad_intervals"], 1)
        self.assertEqual(source["good_tokens"], 100)
        self.assertEqual(source["strict_request_good_tokens"], 0)
        self.assertEqual(source["request_slo_p99_good_tokens"], 101)

    def test_candidate_allows_one_slow_gap_if_mean_tpot_is_below_limit(self) -> None:
        value = payload()
        peer = value["background/background_summary.json"]["results"][0]
        times = [0.1 + index * 0.001 for index in range(100)]
        times.append(times[-1] + 0.1)
        peer.update({
            "output_tokens": len(times),
            "token_times_unix_s": times,
            "request_ended_unix_s": 0.35,
            "request_started_unix_s": -1.5,
        })
        report = audit_payload(value)
        self.assertTrue(report["computable"], report["errors"])
        source = report["metrics"]["by_pool"]["source"]
        self.assertEqual(source["good_tokens"], 0)
        self.assertEqual(source["goodoutput_v3_success_requests"], 1)
        self.assertEqual(source["goodoutput_v3_tokens"], 101)
        self.assertLess(report["request_rows"][0]["mean_itl_ms"], 50)

    def test_candidate_fails_high_mean_tpot_or_ttft(self) -> None:
        value = payload()
        peer = value["background/background_summary.json"]["results"][0]
        peer["token_times_unix_s"] = [0.1, 0.2, 0.3]
        peer["request_ended_unix_s"] = 0.35
        report = audit_payload(value)
        self.assertFalse(report["request_rows"][0]["goodoutput_v3_success"])
        peer["token_times_unix_s"] = [3.1, 3.12, 3.14]
        peer["request_ended_unix_s"] = 3.2
        report = audit_payload(value)
        self.assertFalse(report["request_rows"][0]["goodoutput_v3_success"])


if __name__ == "__main__":
    unittest.main()
