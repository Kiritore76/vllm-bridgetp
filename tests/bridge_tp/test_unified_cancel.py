# SPDX-License-Identifier: Apache-2.0
"""Unified source-demand cancellation contracts; CPU observations/receivers."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

from tests.bridge_tp.test_pressure_cancel import reading_at
from tests.bridge_tp.test_pressure_health_service import HealthPolicyTests
from tools.bridge_tp.pressure_cpu import load
from tools.bridge_tp.pressure_session import RollingPressureSession

p, runtime, u, cancel, events = (
    load(name)
    for name in (
        "pressure_policy",
        "pressure_runtime",
        "unified_scheduler",
        "pressure_cancel",
        "events",
    )
)
FIXTURE = Path(__file__).parent / "fixtures/unified_cancel_20261108.json"
SOURCE = "A-g5-tp1"


def set_itl(co, source, now, interval=0.02):
    # Feed a known steady observation through the actual smoother API.
    co.itl = u.SmoothedItl()
    co.itl.update(source, now - 0.2, 0, 0, 2)
    co.itl.update(source, now, 20 * interval, 20, 2)


class UnifiedCancellationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = replace(HealthPolicyTests().cfg(), shadow_cancel=True)
        self.logs = []
        self.co = runtime.PressureCoordinator(
            self.cfg,
            self.logs.append,
            scheduler="unified",
            slots={SOURCE: 16, "A-g6-tp1": 16, "B-g2-tp2": 16},
        )
        self.driver = self.session(self.co, SOURCE)

    def session(self, co, source):
        driver = object.__new__(RollingPressureSession)
        driver.coordinator = co
        driver.config = NS(source_pool_id=source)
        driver.source_instance = source
        driver.migration_id = "cpu-session"
        driver.record = NS(request_id="cpu-request", state=events.MigrationState.SHADOW)
        driver.old = NS(MigrationState=events.MigrationState)
        driver.run_dir = self.root
        driver.settled = False
        driver.cancel_future = None
        driver.cancel_policy = cancel.ShadowCancellation()
        driver._cancel_need_status = None
        driver.audit = NS(write=self.logs.append)
        driver.cancel = Mock()
        return driver

    def source(self, now, **changes):
        values = dict(
            total_blocks=2325, free_blocks=2000, engine_requests=2, health_eta=1
        )
        values.update(changes)
        return reading_at(SOURCE, now, **values)

    def tick(self, now, *, interval=0.02, driver=None, **changes):
        driver = driver or self.driver
        source = self.source(now, **changes)
        set_itl(driver.coordinator, source.pool_id, now, interval)
        driver.reconsider(
            now,
            source,
            reading_at("B-g2-tp2", now),
            None,
            driver.coordinator.policy.cfg,
        )
        return driver.coordinator.cancellation_source_need(
            source, now, driver.source_instance
        )

    def run_recovered(self, start=10, **changes):
        for i in range(16):
            self.tick(start + i * 0.2, **changes)

    def test_real_session_blocks_recovery_while_KV_trigger_remains(self):
        for i in range(21):
            snapshot = self.tick(10 + i * 0.2, free_blocks=220)
            self.assertEqual((snapshot["state"], snapshot["need"]), ("NEEDED", "KV"))
        self.driver.cancel.assert_not_called()
        self.assertIsNone(self.driver.cancel_policy.since)
        transitions = [
            e for e in self.logs if e.get("kind") == "pressure_cancel_source_need"
        ]
        self.assertEqual(len(transitions), 1)  # not one large record per tick

    def test_pressure_still_present_blocks_recovery_even_without_KV_risk(self):
        for i in range(21):
            snapshot = self.tick(10 + i * 0.2, interval=0.04)
            self.assertEqual(snapshot["need"], "PRESSURE")
        self.driver.cancel.assert_not_called()

    def test_slot_pressure_and_predicted_exhaustion_are_source_demand(self):
        for changes, expected in [
            (dict(engine_requests=15), "PRESSURE"),
            (dict(free_blocks=200, total_blocks=1000, growth_blocks_s=10), "KV"),
        ]:
            with self.subTest(changes=changes):
                self.assertEqual(self.tick(10, **changes)["need"], expected)
        self.driver.cancel.assert_not_called()

    def test_active_channel_or_cooldown_does_not_mean_source_recovered(self):
        self.co.migration_state.cooldown_until[SOURCE] = 1000
        self.co.ledger.channel_available = Mock(return_value=False)
        self.run_recovered(free_blocks=220)
        self.driver.cancel.assert_not_called()
        self.co.ledger.channel_available.assert_not_called()

    def test_genuine_recovery_still_cancels_after_existing_three_seconds(self):
        for i in range(15):
            self.tick(10 + i * 0.2)
        self.driver.cancel.assert_not_called()
        self.tick(13)
        self.driver.cancel.assert_called_once_with(
            "source service and projected KV recovered"
        )
        decision = next(
            e for e in self.logs if e.get("kind") == "pressure_cancel_decision"
        )
        self.assertEqual(decision["source_need"]["state"], "CLEAR")
        self.assertEqual(decision["scheduler"], "unified")

    def test_pressure_turning_into_KV_risk_does_not_cancel(self):
        for i in range(20):
            self.tick(
                10 + i * 0.2,
                interval=0.04 if i < 10 else 0.02,
                free_blocks=2000 if i < 10 else 220,
            )
        self.driver.cancel.assert_not_called()
        self.assertEqual(
            [
                e["source_need"]["need"]
                for e in self.logs
                if e.get("kind") == "pressure_cancel_source_need"
            ],
            ["PRESSURE", "KV"],
        )

    def test_missing_or_stale_observation_resets_recovery_confirmation(self):
        for bad in (
            dict(engine_requests=None),
            dict(growth_blocks_s=None),
            dict(sampled_at=0),
        ):
            with self.subTest(bad=bad):
                self.driver = self.session(self.co, SOURCE)
                for i in range(15):
                    self.tick(10 + i * 0.2)
                self.tick(13, **bad)
                self.assertIsNone(self.driver.cancel_policy.since)
                for i in range(15):
                    self.tick(13.2 + i * 0.2)
                self.driver.cancel.assert_not_called()
                self.tick(16.2)
                self.driver.cancel.assert_called_once()

    def test_brief_return_to_pressure_resets_recovery_confirmation(self):
        for i in range(15):
            self.tick(10 + i * 0.2)
        self.tick(13, free_blocks=220)
        for i in range(15):
            self.tick(13.2 + i * 0.2)
        self.driver.cancel.assert_not_called()
        self.tick(16.2)
        self.driver.cancel.assert_called_once()

    def test_wrong_incarnation_and_stale_migrate_EMA_are_unknown(self):
        source = self.source(10)
        set_itl(self.co, SOURCE, 10)
        wrong = self.co.cancellation_source_need(source, 10, "A-g6-tp1")
        self.assertEqual(wrong["state"], "UNKNOWN")
        newer = self.source(13)
        stale = self.co.cancellation_source_need(newer, 13, SOURCE)
        self.assertEqual(stale["state"], "UNKNOWN")
        self.assertFalse(u.source_recovery_ready(wrong, source, 10, self.cfg.max_age_s))

    def test_missing_migrate_speed_resets_recovery_and_cannot_mean_zero(self):
        source = self.source(10)
        self.co.itl = u.SmoothedItl()
        self.co.itl.update(SOURCE, 10, 0, 0, 2)
        snapshot = self.co.cancellation_source_need(source, 10, SOURCE)
        self.assertEqual(snapshot["state"], "UNKNOWN")
        self.assertFalse(snapshot["complete"])
        self.assertIsNone(snapshot["speed"])

    def test_need_helper_does_not_apply_start_eligibility_or_advance_timers(self):
        full = dict(kv=0.95, value=0.95)
        state = u.MigrationState(cooldown_until={SOURCE: 1000})
        self.assertEqual(u.source_migration_need(full, False), "KV")
        self.assertEqual(
            u.migration_trigger(full, False, None, state, SOURCE, 10)[0], "KV"
        )
        full = dict(kv=0.3, value=0.95)
        before = copy.deepcopy(state)
        self.assertEqual(u.source_migration_need(full, False), "PRESSURE")
        self.assertEqual(state, before)
        self.assertIsNone(u.migration_trigger(full, False, None, state, SOURCE, 10)[0])

    def test_frozen_or_committed_source_cannot_policy_cancel(self):
        for frozen in (True, False):
            with self.subTest(frozen=frozen):
                self.driver = self.session(self.co, SOURCE)
                if frozen:
                    (self.root / "request_freeze_control.json").write_text("{}")
                else:
                    (self.root / "request_freeze_control.json").unlink()
                    self.driver.record.state = events.MigrationState.TAKEOVER
                self.run_recovered()
                self.driver.cancel.assert_not_called()

    def test_adjusted_source_is_not_ledger_adjusted_a_second_time(self):
        self.co.ledger.routing_reading = Mock(
            side_effect=AssertionError("double counted")
        )
        snapshot = self.tick(10, unallocated_prompt_blocks=100)
        self.assertAlmostEqual(snapshot["kv"], 425 / 2325)
        self.co.ledger.routing_reading.assert_not_called()

    def test_legacy_pressure_call_path_preserves_the_captured_cancellation(self):
        captured = json.loads(FIXTURE.read_text())
        cfg = p.PressureConfig(**captured["policy"])
        co = runtime.PressureCoordinator(cfg, self.logs.append)
        co.cancellation_source_need = Mock(side_effect=AssertionError("legacy path"))
        driver = self.session(co, captured["cases"][0]["source"])
        for tick in captured["cases"][0]["observations"]:
            source = p.PoolReading(**tick["source"])
            driver.reconsider(tick["now"], source, source, None, cfg)
        driver.cancel.assert_called_once_with(
            "source service and projected KV recovered"
        )
        co.cancellation_source_need.assert_not_called()

    def test_four_captured_cancellations_are_blocked_through_actual_session(self):
        captured = json.loads(FIXTURE.read_text())
        for case in captured["cases"]:
            with self.subTest(case=case["session_suffix"]):
                co = runtime.PressureCoordinator(
                    p.PressureConfig(**captured["policy"]),
                    self.logs.append,
                    scheduler="unified",
                    slots={case["source"]: 16},
                )
                driver = self.session(co, case["source"])
                for tick in case["observations"]:
                    source = p.PoolReading(**tick["source"])
                    set_itl(co, source.pool_id, tick["now"], tick["migrate_itl_s"])
                    driver.reconsider(tick["now"], source, source, None, co.policy.cfg)
                driver.cancel.assert_not_called()

    def cancellation_audit(self):
        self.run_recovered()
        decision = next(
            e for e in self.logs if e.get("kind") == "pressure_cancel_decision"
        )
        return [
            dict(decision, unix_s=13),
            dict(
                kind="pressure_cancel_requested",
                migration_id="cpu-session",
                request_id="cpu-request",
                reason="source service and projected KV recovered",
                unix_s=13.01,
            ),
        ]

    def test_gpu_audit_accepts_genuine_cancellation_and_zero_cancellation(self):
        self.assertTrue(cancel.assess_unified_cancellations([], 2)["passed"])
        result = cancel.assess_unified_cancellations(self.cancellation_audit(), 2)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["recovery_requests"], 1)

    def test_gpu_audit_rejects_missing_stale_wrong_identity_or_still_needed(self):
        audit = self.cancellation_audit()
        changes = [
            lambda a: a[0].pop("source_need"),
            lambda a: a[0].update(unix_s=16),
            lambda a: a[0]["source_need"].update(expected_pool_id="A-g7-tp1"),
            lambda a: a[0]["source_need"].update(kv=0.95, value=0.95),
            lambda a: a[0].update(scheduler="pressure"),
            lambda a: a.pop(),
        ]
        for change in changes:
            with self.subTest(change=change):
                value = copy.deepcopy(audit)
                change(value)
                self.assertFalse(
                    cancel.assess_unified_cancellations(value, 2)["passed"]
                )
