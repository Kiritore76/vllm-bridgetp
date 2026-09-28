# SPDX-License-Identifier: Apache-2.0
"""M2 profile selection and its safety gates."""

from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from unittest.mock import patch

from tools.bridge_tp.run_phase9_controller import parse_args
from tools.bridge_tp.run_shadow_strategy_online_validation import (
    build_controller_config_overrides,
)
from vllm.bridge_tp.controller.manager_m0 import RuntimeSnapshot
from vllm.bridge_tp.controller.manager_m2 import M2RateConfig, M2RateController


def sample(**updates: object) -> RuntimeSnapshot:
    baseline = RuntimeSnapshot(
        unix_s=100.0,
        migration_id="migration-1",
        request_id="request-1",
        state="SHADOW",
        generated_tokens=100,
        source_sampled_unix_s=99.9,
        target_sampled_unix_s=99.9,
        source_free_kv_tokens=30000,
        source_guard_free_kv_tokens=8448,
        source_pool_growth_tokens_s=100,
        target_running=0,
        target_waiting=0,
        target_kv_usage_frac=0.1,
        history_total_bytes=1024,
        history_resident_bytes=0,
        delta_lag_tokens=0,
    )
    return replace(baseline, **updates)


class TestM2RateController(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = M2RateController(
            M2RateConfig(
                low_bytes_s=0.25 * 1024**3,
                medium_bytes_s=0.5 * 1024**3,
                high_bytes_s=1.0 * 1024**3,
                cooldown_s=2.0,
                stable_ticks=2,
            )
        )

    def test_busy_target_downshifts_only_after_stable_samples(self) -> None:
        first = self.controller.decide(sample(target_waiting=3))
        self.assertEqual(first.action, "HOLD")
        second = self.controller.decide(sample(unix_s=100.2, target_waiting=3))
        self.assertEqual((second.action, second.profile), ("SET_RATE", "LOW"))

    def test_guard_risk_upshifts_immediately_even_during_cooldown(self) -> None:
        self.controller.decide(sample(target_waiting=3))
        self.controller.decide(sample(unix_s=100.2, target_waiting=3))
        urgent = self.controller.decide(
            sample(unix_s=100.3, source_free_kv_tokens=9000, target_waiting=3)
        )
        self.assertEqual((urgent.action, urgent.profile), ("SET_RATE", "HIGH"))
        self.assertEqual(urgent.rate_bytes_s, 1024**3)
        self.assertEqual(
            self.controller.decide(sample(unix_s=100.4)).profile, "HIGH"
        )

    def test_delta_backlog_upshifts(self) -> None:
        decision = self.controller.decide(sample(delta_lag_tokens=64))
        self.assertEqual((decision.action, decision.profile), ("SET_RATE", "HIGH"))

    def test_four_active_target_requests_select_low_rate(self) -> None:
        self.controller.decide(sample(target_running=4))
        decision = self.controller.decide(
            sample(unix_s=100.2, target_running=4)
        )
        self.assertEqual((decision.action, decision.profile), ("SET_RATE", "LOW"))

    def test_missing_or_stale_evidence_holds_rate(self) -> None:
        for state in (
            sample(target_waiting=None),
            sample(target_sampled_unix_s=90.0),
            sample(source_pool_growth_tokens_s=None),
        ):
            with self.subTest(state=state):
                decision = self.controller.decide(state)
                self.assertEqual((decision.action, decision.profile),
                                 ("HOLD", "MEDIUM"))

    def test_no_rate_action_outside_shadow(self) -> None:
        decision = self.controller.decide(sample(state="LOCAL", target_waiting=3))
        self.assertEqual(decision.action, "HOLD")

    def test_explicit_rate_order_required(self) -> None:
        with self.assertRaises(ValueError):
            M2RateController(M2RateConfig(1.0, 1.0, 2.0))

    def test_online_controller_requires_m0_and_m1_for_m2(self) -> None:
        base = [
            "run_phase9_controller.py", "--config", "config.json",
            "--run-dir", "run", "--source-request", "request.json",
            "--manager-m2-rate", "--manager-m1-auto-start",
            "--diagnostic-earliest-ready-cutover", "--handoff-mode",
            "shadow-only", "--gpu-resident-shadow",
        ]
        with patch("sys.argv", base):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args()
        with patch("sys.argv", base + ["--manager-m0-shadow"]):
            self.assertTrue(parse_args().manager_m2_rate)

    def test_runner_config_uses_three_explicit_rates(self) -> None:
        overrides = build_controller_config_overrides(
            trigger_output_tokens=64,
            cutover_output_tokens=128,
            fixed_rate_gib_s=None,
            m2_profiles_gib_s=(0.25, 0.5, 1.0),
        )
        self.assertEqual(
            overrides["rate"],
            {
                "b_min_bytes_s": 0.25 * 1024**3,
                "b_start_bytes_s": 0.5 * 1024**3,
                "b_max_bytes_s": 1.0 * 1024**3,
                "b_hard_max_bytes_s": 1.0 * 1024**3,
            },
        )


if __name__ == "__main__":
    unittest.main()
