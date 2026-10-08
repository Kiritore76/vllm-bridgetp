# SPDX-License-Identifier: Apache-2.0
"""Check soft SLO severity, fractional window output and evidence boundaries."""

import hashlib
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.bridge_tp.test_audit_slo_v6 import REFERENCE, test_payload
from tools.bridge_tp.audit_slo_v6 import audit_v6_payload
from tools.bridge_tp.horizon_goodoutput import score_horizon
from tools.bridge_tp.output_quality import SOFT_POLICY, request_quality
from tools.bridge_tp.run_goodoutput_matrix_a100 import reference_contract

CONFIG = {
    **REFERENCE,
    "max_visible_interval_policy": "DIAGNOSTIC_ONLY",
    "output_quality_policy": SOFT_POLICY,
    "output_quality_beta": 1.0,
}


def row(**changes):
    return dict(
        status="COMPLETED",
        pool="anchor",
        output_tokens=201,
        ttft_ms=1000,
        ttft_limit_ms=1000,
        mean_tpot_ms=100,
        slow_interval_rate=0.01,
        handoff_ms=1000,
        **changes,
    )


class TestOutputQuality(unittest.TestCase):
    def sample(self, **changes):
        result = row()
        result.update(changes)
        return request_quality(result, CONFIG)

    def test_boundary_and_progressive_penalty(self):
        self.assertEqual(self.sample()["quality_weight"], 1)
        self.assertAlmostEqual(self.sample(ttft_ms=1010)["quality_weight"], 1 / 1.01)
        self.assertEqual(self.sample(ttft_ms=2000)["quality_weight"], 0.5)
        self.assertEqual(self.sample(ttft_ms=5000)["quality_weight"], 0.2)

    def test_slow_fraction_one_percent_is_the_boundary(self):
        self.assertEqual(self.sample(slow_interval_rate=0.01)["quality_weight"], 1)
        self.assertAlmostEqual(
            self.sample(slow_interval_rate=0.011)["quality_weight"], 1 / 1.1
        )
        self.assertEqual(self.sample(slow_interval_rate=0.02)["quality_weight"], 0.5)

    def test_correlated_latency_penalties_are_not_multiplied(self):
        self.assertEqual(
            self.sample(mean_tpot_ms=200, slow_interval_rate=0.02)["quality_weight"],
            0.5,
        )
        self.assertEqual(self.sample(handoff_ms=4000)["quality_weight"], 0.25)

    def test_service_failure_zero_and_single_token_has_no_tpot(self):
        self.assertEqual(
            self.sample(status="FAILED", ttft_ms=None)["quality_weight"], 0
        )
        self.assertEqual(
            self.sample(output_tokens=1, mean_tpot_ms=None, slow_interval_rate=0)[
                "quality_weight"
            ],
            1,
        )

    def test_missing_evidence_is_not_a_service_zero(self):
        for changes in [
            {"ttft_ms": None},
            {"mean_tpot_ms": None},
            {"handoff_ms": math.nan},
            {"slow_interval_rate": math.inf},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.sample(**changes)

    def test_invalid_quality_config_is_rejected(self):
        for change in [
            {"output_quality_policy": "typo"},
            {"output_quality_beta": 0},
            {"output_quality_beta": True},
            {"primary_max_slow_interval_rate": 0},
            {"max_visible_interval_policy": "HARD_GATE"},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit_v6_payload(test_payload(), {**CONFIG, **change})

    def test_window_fractional_score_preserves_hard_slo(self):
        value = test_payload()
        peer = value["background/background_summary.json"]["results"][0]
        peer.update(token_times_unix_s=[1.201, 1.221, 1.241], request_ended_unix_s=1.3)
        report = audit_v6_payload(value, CONFIG)
        self.assertTrue(report["computable"], report["errors"])
        self.assertEqual(report["metrics"]["slo_success_requests"], 1)
        self.assertEqual(report["metrics"]["good_output_tokens"], 3)
        self.assertAlmostEqual(
            report["metrics"]["quality_adjusted_output_tokens"], 3 + 3 * 1200 / 1201
        )
        horizon = score_horizon(
            report,
            value["background/background_summary.json"],
            value["controller/source_response.json"],
            value["controller/target_response.json"],
            value["controller/response_proxy_stats.json"],
            1.23,
            settle_after_h=True,
        )
        self.assertTrue(horizon["eligible"], horizon.get("errors"))
        self.assertAlmostEqual(horizon["good_output_tokens"], 3 + 2 * 1200 / 1201)
        self.assertEqual(horizon["hard_good_output_tokens"], 3)
        self.assertEqual(horizon["drain_completed_requests"], 1)
        self.assertEqual(horizon["output_metric"], "QUALITY_ADJUSTED_OUTPUT")
        del report["request_rows"][0]["quality_weight"]
        self.assertFalse(
            score_horizon(
                report,
                value["background/background_summary.json"],
                value["controller/source_response.json"],
                value["controller/target_response.json"],
                value["controller/response_proxy_stats.json"],
                1.23,
                settle_after_h=True,
            )["eligible"]
        )

    def test_one_percent_reference_matches_linux_sha(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "experiments/phase9/slo"
            / "slo_v6_slow1pct_soft_a100_tp4_reference_20261008.json"
        )
        data = path.read_bytes().replace(b"\r\n", b"\n")
        args = SimpleNamespace(slo_profile="slow1pct_soft", probability_pilot=True)
        self.assertEqual(hashlib.sha256(data).hexdigest(), reference_contract(args))


if __name__ == "__main__":
    unittest.main()
