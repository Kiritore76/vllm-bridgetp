# SPDX-License-Identifier: Apache-2.0
"""M2 profile selection and its safety gates."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.build_experiment_a4_pressure_manifest import (
    build_manifest as build_pressure_manifest,
)
from tools.bridge_tp.run_phase9_capacity_background import (
    anchor_output_threshold,
    load_manifest,
    wait_for_controller_event,
    wait_for_m2_initial_rate,
)
from tools.bridge_tp.run_phase9_controller import parse_args
from tools.bridge_tp.run_shadow_strategy_online_validation import (
    build_controller_config_overrides,
    m2_expected_profile_used,
)
from vllm.bridge_tp.controller.manager_m0 import (
    RuntimeSnapshot,
    snapshot_from_telemetry,
)
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

    def test_expected_medium_accepts_effective_initial_hold(self) -> None:
        profiles = (0.5, 2.4, 8.0)
        initial = [{"decision": {
            "action": "HOLD", "profile": "MEDIUM",
            "rate_bytes_s": 2.4 * 1024**3,
        }}]
        active = [{
            "manager_m2_decision": {"action": "HOLD", "profile": "MEDIUM"},
            "rate_gib_s": 2.4,
        }]
        self.assertTrue(m2_expected_profile_used(
            initial, active, "MEDIUM", profiles,
        ))
        self.assertFalse(m2_expected_profile_used(
            initial, [{**active[0], "rate_gib_s": 0.5}],
            "MEDIUM", profiles,
        ))

    def test_source_peer_event_start_waits_for_initial_rate(self) -> None:
        base = {"scenario": "target load", "jobs": [{
            "job_id": "target_000", "pool": "target",
            "start_after_s": 0.0,
            "request": {"model": "model", "prompt": [100], "max_tokens": 1},
        }]}
        manifest = build_pressure_manifest(
            base, source_jobs=2, source_prompt_tokens=1,
            source_output_tokens=1, source_start_after_s=0.0,
            source_start_interval_s=0.05, source_prompt_token_id=100,
            max_model_len=8192, source_start_after_m2_initial=True,
        )
        peers = [job for job in manifest["jobs"] if job["pool"] == "source"]
        self.assertEqual([job["start_after_event"] for job in peers],
                         ["M2_INITIAL_RATE", "M2_INITIAL_RATE"])
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.jsonl"

            def publish() -> None:
                time.sleep(0.05)
                audit.write_text(json.dumps({
                    "kind": "manager_m2_initial_rate",
                    "decision": {"profile": "LOW"},
                }) + "\n", encoding="utf-8")

            writer = threading.Thread(target=publish)
            writer.start()
            started = time.monotonic()
            observed = wait_for_m2_initial_rate(audit, 1.0)
            writer.join()
            self.assertGreaterEqual(observed - started, 0.04)

    def test_source_arrival_waits_for_anchor_output_not_m2_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.jsonl"

            def publish() -> None:
                audit.write_text(json.dumps({
                    "kind": "telemetry", "output_tokens": 0,
                }) + "\n", encoding="utf-8")
                time.sleep(0.05)
                with audit.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "kind": "telemetry", "output_tokens": 1,
                    }) + "\n")

            writer = threading.Thread(target=publish)
            writer.start()
            started = time.monotonic()
            observed = wait_for_controller_event(
                audit, "ANCHOR_FIRST_OUTPUT", 1.0,
            )
            writer.join()
            self.assertGreaterEqual(observed - started, 0.04)

    def test_late_target_arrival_waits_for_anchor_output_800(self) -> None:
        self.assertEqual(anchor_output_threshold("ANCHOR_OUTPUT_800"), 800)
        self.assertIsNone(anchor_output_threshold("ANCHOR_OUTPUT_0"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest = {"format_version": 1, "jobs": [{
                "job_id": "target_002", "pool": "target",
                "start_after_event": "ANCHOR_OUTPUT_800",
                "start_after_s": 0.0,
                "request": {"model": "model", "prompt": [100],
                            "max_tokens": 1024},
            }]}
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(
                load_manifest(manifest_path)["jobs"][0]["pool"], "target")
            manifest["jobs"][0]["pool"] = "source"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "requires target"):
                load_manifest(manifest_path)

            audit = root / "audit.jsonl"

            def publish() -> None:
                audit.write_text(json.dumps({
                    "kind": "telemetry", "output_tokens": 799,
                }) + "\n", encoding="utf-8")
                time.sleep(0.05)
                with audit.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "kind": "telemetry", "output_tokens": 800,
                    }) + "\n")

            writer = threading.Thread(target=publish)
            writer.start()
            started = time.monotonic()
            observed = wait_for_controller_event(
                audit, "ANCHOR_OUTPUT_800", 1.0)
            writer.join()
            self.assertGreaterEqual(observed - started, 0.04)

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

    def test_prefill_ewma_spike_does_not_force_initial_high(self) -> None:
        decision = self.controller.decide(
            sample(
                state="LOCAL",
                target_running=4,
                source_pool_growth_tokens_s=1448.0,
                source_pool_sustained_growth_tokens_s=32.0,
                capacity_pressure=True,
                history_total_bytes=1024**3,
            ),
            before_start=True,
        )
        self.assertEqual((decision.action, decision.profile), ("SET_RATE", "LOW"))

    def test_snapshot_retains_sustained_source_growth(self) -> None:
        snapshot = snapshot_from_telemetry({
            "unix_s": 100.0,
            "state": "LOCAL",
            "tp1": {"sampled_unix_s": 100.0},
            "tp4": {"sampled_unix_s": 100.0},
            "capacity_signal": {
                "transition": "HOLD",
                "decline_rate_tokens_s": 1448.0,
                "sustained_decline_rate_tokens_s": 67.0,
                "prefill_pending_kv_tokens": 1024,
                "decode_growth_tokens_s": 32.0,
            },
        })
        self.assertEqual(snapshot.source_pool_growth_tokens_s, 1448.0)
        self.assertEqual(snapshot.source_pool_sustained_growth_tokens_s, 67.0)
        self.assertEqual(snapshot.source_prefill_pending_kv_tokens, 1024)
        self.assertEqual(snapshot.source_decode_growth_tokens_s, 32.0)

    def test_prefill_and_decode_use_separate_capacity_terms(self) -> None:
        burst = self.controller.decide(
            sample(
                state="LOCAL", target_running=4,
                source_pool_growth_tokens_s=8000.0,
                source_pool_sustained_growth_tokens_s=8000.0,
                source_prefill_pending_kv_tokens=0,
                source_decode_growth_tokens_s=32.0,
            ),
            before_start=True,
        )
        self.assertEqual(burst.profile, "LOW")
        self.assertEqual(
            burst.source_capacity_model,
            "prefill_reservation_plus_decode_growth",
        )
        reserved = self.controller.decide(
            sample(
                unix_s=100.2, target_running=4,
                source_prefill_pending_kv_tokens=20000,
                source_decode_growth_tokens_s=100.0,
            )
        )
        self.assertEqual((reserved.action, reserved.profile),
                         ("SET_RATE", "HIGH"))

    def test_sustained_pressure_overrides_busy_target(self) -> None:
        decision = self.controller.decide(
            sample(
                state="LOCAL",
                target_running=4,
                source_pool_growth_tokens_s=32.0,
                source_pool_sustained_growth_tokens_s=2400.0,
            ),
            before_start=True,
        )
        self.assertEqual((decision.action, decision.profile), ("SET_RATE", "HIGH"))

    def test_delta_backlog_upshifts(self) -> None:
        decision = self.controller.decide(sample(delta_lag_tokens=64))
        self.assertEqual((decision.action, decision.profile), ("SET_RATE", "HIGH"))

    def test_four_active_target_requests_select_low_rate(self) -> None:
        self.controller.decide(sample(target_running=4))
        decision = self.controller.decide(
            sample(unix_s=100.2, target_running=4)
        )
        self.assertEqual((decision.action, decision.profile), ("SET_RATE", "LOW"))

    def test_initial_rate_is_selected_before_shadow_copy(self) -> None:
        initial = self.controller.decide(
            sample(state="LOCAL", target_running=4), before_start=True
        )
        self.assertEqual((initial.action, initial.profile), ("SET_RATE", "LOW"))
        ongoing = self.controller.decide(
            sample(unix_s=100.2, target_running=4)
        )
        self.assertEqual((ongoing.action, ongoing.profile), ("HOLD", "LOW"))

    def test_initial_preview_does_not_change_rate_before_m1_admission(self) -> None:
        state = sample(state="LOCAL", target_running=4)
        preview = self.controller.preview_initial(state)
        self.assertEqual((preview.action, preview.profile), ("SET_RATE", "LOW"))
        self.assertEqual(self.controller.profile, "MEDIUM")
        self.assertIsNone(self.controller._last_change_s)
        self.assertEqual(preview, self.controller.decide(state, before_start=True))

    def test_initial_low_is_rejected_when_preparation_misses_guard(self) -> None:
        decision = self.controller.decide(
            sample(
                state="LOCAL", target_running=4,
                source_free_kv_tokens=14448,
                source_pool_growth_tokens_s=100,
                history_total_bytes=20 * 1024**3,
            ),
            before_start=True,
        )
        self.assertEqual((decision.action, decision.profile),
                         ("SET_RATE", "HIGH"))

    def test_diagnostic_high_arms_before_first_history_chunk(self) -> None:
        controller = M2RateController(
            self.controller.config, force_initial_high=True
        )
        decision = controller.decide(
            sample(state="LOCAL", target_running=4), before_start=True
        )
        self.assertEqual((decision.action, decision.profile),
                         ("SET_RATE", "HIGH"))
        self.assertEqual(decision.reason, "diagnostic HIGH transfer smoke")

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
            "--m1-source-release-tail-s", "5.0",
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
