# SPDX-License-Identifier: Apache-2.0
"""M1 Start uses only fresh online evidence and keeps safety guards explicit."""

from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from unittest.mock import patch

from tools.bridge_tp.run_phase9_controller import parse_args
from vllm.bridge_tp.controller.manager_m0 import RuntimeSnapshot
from vllm.bridge_tp.controller.manager_m1 import M1StartConfig, M1StartController
from vllm.bridge_tp.controller.manager_m2 import M2RateConfig, M2RateController
from vllm.bridge_tp.controller.predictor import SurvivalTable


def snapshot() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        unix_s=100.0,
        migration_id="migration-1",
        request_id="request-1",
        state="LOCAL",
        generated_tokens=40,
        current_context_tokens=2088,
        source_sampled_unix_s=99.9,
        target_sampled_unix_s=99.9,
        source_free_kv_tokens=30000,
        source_guard_free_kv_tokens=8448,
        source_pool_growth_tokens_s=30.0,
        source_decode_growth_tokens_s=30.0,
        source_prefill_pending_kv_tokens=0,
        target_free_kv_tokens=50000,
        target_kv_usage_frac=0.1,
        target_waiting=0,
        channel_available=True,
    )


class TestM1Start(unittest.TestCase):
    def test_controller_accepts_only_dynamic_gpu_resident_m1(self) -> None:
        base = [
            "run_phase9_controller.py", "--config", "config.json",
            "--run-dir", "run", "--source-request", "request.json",
            "--manager-m1-auto-start", "--diagnostic-earliest-ready-cutover",
            "--m1-source-release-tail-s", "5.0",
            "--handoff-mode", "shadow-only", "--gpu-resident-shadow",
        ]
        with patch("sys.argv", base):
            self.assertTrue(parse_args().manager_m1_auto_start)
        without_floor = base[:]
        index = without_floor.index("--m1-source-release-tail-s")
        del without_floor[index:index + 2]
        with patch("sys.argv", without_floor):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args()
        with patch("sys.argv", base + ["--diagnostic-trigger-output-tokens", "64"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def setUp(self) -> None:
        self.controller = M1StartController(
            M1StartConfig(source_release_tail_s=5.0)
        )
        self.table = SurvivalTable.from_output_lengths([1024] * 30)

    def decide(self, state: RuntimeSnapshot, table=None):
        return self.controller.decide(
            state,
            self.table if table is None else table,
            max_output_tokens=1024,
            rate_bytes_s=0.5 * 1024**3,
            kv_bytes_per_token=196608,
        )

    def test_supported_long_request_starts(self) -> None:
        decision = self.decide(snapshot())
        self.assertEqual(decision.action, "START_SHADOW")
        self.assertEqual(decision.survivors, 30)
        self.assertGreater(
            decision.source_time_to_guard_s, decision.estimated_preparation_s
        )

    def test_preparation_uses_previewed_m2_initial_low_rate(self) -> None:
        state = replace(snapshot(), target_running=4)
        m2 = M2RateController(M2RateConfig(
            low_bytes_s=0.5 * 1024**3,
            medium_bytes_s=2.4 * 1024**3,
            high_bytes_s=8.0 * 1024**3,
        ))
        initial = replace(
            state,
            history_total_bytes=state.current_context_tokens * 196608,
            history_resident_bytes=0,
        )
        preview = m2.preview_initial(initial)
        self.assertEqual(preview.profile, "LOW")
        decision = self.controller.decide(
            state, self.table, max_output_tokens=1024,
            rate_bytes_s=preview.rate_bytes_s,
            kv_bytes_per_token=196608,
        )
        self.assertEqual(decision.action, "START_SHADOW")
        self.assertAlmostEqual(
            decision.estimated_preparation_s,
            state.current_context_tokens * 196608 / (0.5 * 1024**3) + 5.0,
        )

    def test_no_start_without_fresh_capacity_or_channel(self) -> None:
        cases = (
            replace(snapshot(), target_sampled_unix_s=95.0),
            replace(snapshot(), target_free_kv_tokens=None),
            replace(snapshot(), target_free_kv_tokens=2000),
            replace(snapshot(), target_kv_usage_frac=0.9),
            replace(snapshot(), target_waiting=5),
            replace(snapshot(), channel_available=False),
            replace(snapshot(), source_free_kv_tokens=8448),
            replace(snapshot(), source_decode_growth_tokens_s=None),
            replace(snapshot(), source_prefill_pending_kv_tokens=None),
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(self.decide(case).action, "STAY")

    def test_prefill_ewma_does_not_override_decode_growth(self) -> None:
        state = replace(
            snapshot(), source_free_kv_tokens=10288,
            source_pool_growth_tokens_s=700.6,
            source_decode_growth_tokens_s=283.8,
        )
        decision = self.decide(state)
        self.assertEqual(decision.action, "START_SHADOW")
        self.assertEqual(decision.source_safe_headroom_tokens, 1840)
        self.assertAlmostEqual(decision.source_time_to_guard_s, 1840 / 283.8)
        self.assertAlmostEqual(
            decision.estimated_preparation_s,
            state.current_context_tokens * 196608 / (0.5 * 1024**3) + 5.0,
        )
        self.assertEqual(
            decision.source_capacity_model,
            "allocated_kv_plus_decode_growth",
        )
        reserved = self.decide(replace(
            state, source_prefill_pending_kv_tokens=600,
        ))
        self.assertEqual(reserved.action, decision.action)
        self.assertEqual(reserved.source_safe_headroom_tokens, 1840)
        self.assertEqual(
            reserved.source_time_to_guard_s, decision.source_time_to_guard_s,
        )

    def test_prefill_only_has_finite_guard_horizon(self) -> None:
        state = replace(snapshot(), source_free_kv_tokens=10288,
                        source_decode_growth_tokens_s=0.0,
                        source_prefill_growth_tokens_s=100.0)
        decision = self.decide(state)
        self.assertAlmostEqual(decision.source_time_to_guard_s, 18.4)
        self.assertEqual(decision.source_capacity_model,
                         "allocated_kv_plus_scheduled_growth")

    def test_missing_release_calibration_fails_closed(self) -> None:
        controller = M1StartController(M1StartConfig())
        decision = controller.decide(
            snapshot(), self.table, max_output_tokens=1024,
            rate_bytes_s=0.5 * 1024**3, kv_bytes_per_token=196608,
        )
        self.assertEqual(decision.action, "STAY")
        self.assertIn("source_release_tail_s", decision.missing)

    def test_no_start_without_supported_remaining_work(self) -> None:
        self.assertEqual(
            self.decide(replace(snapshot(), generated_tokens=10)).action, "STAY"
        )
        self.assertEqual(
            self.decide(
                snapshot(), SurvivalTable.from_output_lengths([1024] * 5)
            ).action,
            "STAY",
        )
        self.assertEqual(
            self.decide(
                snapshot(), SurvivalTable.from_output_lengths([120] * 30)
            ).action,
            "STAY",
        )
        self.assertEqual(
            self.decide(replace(snapshot(), generated_tokens=1024)).action,
            "STAY",
        )

    def test_invalid_rate_and_geometry_do_not_start(self) -> None:
        for rate, geometry in ((0.0, 196608), (0.5 * 1024**3, 0)):
            with self.subTest(rate=rate, geometry=geometry):
                decision = self.controller.decide(
                    snapshot(), self.table,
                    max_output_tokens=1024,
                    rate_bytes_s=rate,
                    kv_bytes_per_token=geometry,
                )
                self.assertEqual(decision.action, "STAY")


if __name__ == "__main__":
    unittest.main()
