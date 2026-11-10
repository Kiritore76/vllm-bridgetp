# SPDX-License-Identifier: Apache-2.0
"""Unified conversion gate and driver plumbing; CPU only, execution stays closed."""

import copy
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from tests.bridge_tp.test_conversion_safety import PLAN, Base, status
from tests.bridge_tp.test_six_card_decider import native_status
from tests.bridge_tp.test_unified_scheduler import coordinator, inst
from tools.bridge_tp import conversion_decider as d
from tools.bridge_tp import conversion_sequence as seq
from tools.bridge_tp import pressure_conversion as pc
from tools.bridge_tp.pressure_cpu import load

u = load("unified_scheduler")


def snapshot(now, value=0.4):
    return dict(
        kind="route",
        complete=True,
        observed_at=now,
        sampled_at=now,
        published_at=now,
        itl_sampled_at=now,
        slots=value,
        kv=0.2,
        speed=0.1,
        value=value,
    )


class GateTests(Base):
    def setUp(self):
        super().setUp()
        self.backend.gate_mode = "UNIFIED"

    def publish(self, full=0.4, **kwargs):
        value = status(**kwargs)
        value["instances"]["B-g2-tp2"]["unified_fullness"] = snapshot(self.now[0], full)
        self.write(value)
        return value

    def test_route_gate_does_not_require_old_spare_classification(self):
        self.publish(target_zone="CRITICAL", target_capacity=0.95)
        self.assertTrue(self.backend.gate(PLAN))
        self.assertFalse(self.decisions("GATE")[-1]["relaxed"])

    def test_exact_limit_waiting_and_tilt_never_waive_gate(self):
        for full in (0.9, 1.0):
            self.backend.service_gate_passed = True
            self.now[0] += 400
            self.publish(full)
            self.assertFalse(self.backend.gate(PLAN))
        self.publish(0.8999)
        self.assertTrue(self.backend.gate(PLAN))

    def test_physical_hard_safety_still_blocks(self):
        self.publish(target_capacity=1.0)
        self.assertFalse(self.backend.gate(PLAN))
        self.assertIn("hard", self.decisions("GATE")[-1]["reason"])

    def test_missing_and_unexpected_active_instances_block(self):
        value = self.publish()
        del value["instances"]["B-g2-tp2"]
        self.write(value)
        self.assertFalse(self.backend.gate(PLAN))
        value = self.publish()
        value["instances"]["ghost"] = copy.deepcopy(value["instances"]["B-g2-tp2"])
        self.write(value)
        self.assertFalse(self.backend.gate(PLAN))

    def test_one_busy_retained_instance_is_enough_to_block(self):
        original = self.backend.registry.instances["B-g2-tp2"]
        extra_id = "C-g4-tp2"
        extra = replace(original, ref=replace(original.ref, instance_id=extra_id))
        self.backend.registry.instances[extra_id] = extra
        value = self.publish()
        value["instances"][extra_id] = copy.deepcopy(value["instances"]["B-g2-tp2"])
        value["instances"][extra_id]["unified_fullness"] = snapshot(self.now[0], 0.91)
        self.write(value)
        self.assertFalse(self.backend.gate(PLAN))

    def test_fresh_status_does_not_refresh_old_measurement(self):
        value = self.publish()
        value["instances"]["B-g2-tp2"]["unified_fullness"] = snapshot(self.now[0] - 3)
        self.write(value)
        self.assertFalse(self.backend.gate(PLAN))


class SnapshotTests(unittest.TestCase):
    def test_unobserved_ingress_counts_once_without_hypothetical_prompt(self):
        co, _ = coordinator(slots=dict(b=16))
        reading = inst("b", total=2000, free=2000)
        co.itl.update("b", 9, 0, 0, 0)
        co.itl.update("b", 10, 0, 0, 0)
        result = co.admission_multi("q", 1024, 128, [reading], dict(b=2), 10, "k")
        self.assertEqual(result.pool_id, "b")
        value = co.route_fullness_snapshot([reading], 10)["b"]
        self.assertEqual(value["slots"], 1 / 16)
        self.assertEqual(value["kv"], 64 / 2000)
        self.assertTrue(u.route_gate_ready(value, 10))

    def test_missing_components_are_not_mistaken_for_idle_zero(self):
        for changes in (dict(), dict(engine_requests=None), dict(total_blocks=None)):
            co, _ = coordinator(slots=dict(b=16))
            value = co.route_fullness_snapshot([inst("b", **changes)], 10)["b"]
            self.assertFalse(u.route_gate_ready(value, 10))

    def test_route_smoothing_is_used_instead_of_migration_smoothing(self):
        co, _ = coordinator(slots=dict(b=16))
        co.itl.update("b", 8, 0, 0, 1)
        co.itl.update("b", 9, 0.01, 1, 1)
        co.itl.update("b", 10, 0.04, 1, 1)
        value = co.route_fullness_snapshot([inst("b")], 10)["b"]
        self.assertEqual(
            value["speed"], co.itl.value("b", "route") / co.itl_reference_s
        )
        self.assertNotEqual(
            value["speed"], co.itl.value("b", "migrate") / co.itl_reference_s
        )

    def test_each_timestamp_must_be_fresh_and_not_in_future(self):
        for key in ("observed_at", "sampled_at", "published_at", "itl_sampled_at"):
            for invalid in (None, 7.9, 10.1, math.nan):
                value = snapshot(10)
                value[key] = invalid
                self.assertFalse(u.route_gate_ready(value, 10), (key, invalid))

    def test_malformed_components_and_wrong_smoothing_fail_closed(self):
        for key in ("slots", "kv", "speed", "value"):
            for invalid in (None, True, -1, math.nan, math.inf):
                value = snapshot(10)
                value[key] = invalid
                self.assertFalse(u.route_gate_ready(value, 10), (key, invalid))
        self.assertFalse(u.route_gate_ready(dict(snapshot(10), kind="migrate"), 10))
        self.assertFalse(u.route_gate_ready(dict(snapshot(10), value=0.1), 10))


class DriverTests(Base):
    def test_cancel_passes_new_long_signal_only_for_six_card(self):
        for profile in ("six_card", "four_card"):
            decider = NS(
                profile=profile,
                signals=Mock(
                    return_value=dict(
                        short_rate=20,
                        long_blocks=100,
                        long_arrivals=1,
                        trend_blocks=200,
                    )
                ),
                cancel_reason=Mock(return_value=None),
            )
            driver = NS(
                backend=self.backend,
                decider=decider,
                executor=NS(plan=dict(state="CREATED")),
                cancel_checked=-math.inf,
                decision_interval_s=1,
                step_edge=("L1AC", ("B", "TP1")),
                cancel_since=1,
            )
            with patch.object(
                self.backend, "status", return_value=dict(unix_s=self.now[0])
            ):
                seq.DecidedSequenceDriver.plan_tick(driver)
            extra = (
                dict(long_arrivals=1, trend_blocks=200) if profile == "six_card" else {}
            )
            decider.cancel_reason.assert_called_once_with(
                ("B", "TP1"),
                "L1AC",
                20,
                100,
                **extra,
            )
            self.assertIsNone(driver.cancel_since)

    def test_profile_reaches_real_decider(self):
        driver = pc.ConversionDriver(
            self.backend,
            dict(
                decider="RECORD_ONLY",
                edges=d.profile_edges("six_card"),
                profile="six_card",
            ),
            self.root / "journal.jsonl",
        )
        self.assertEqual(driver.decider.profile, "six_card")

    def test_all_six_edges_keep_unified_gate_even_with_tilt(self):
        for edges in d.profile_edges("six_card").values():
            for bundle, mode in edges:
                driver = NS(
                    backend=self.backend,
                    decider=NS(profile="six_card"),
                    tilt_before_step=[bundle],
                )
                seq.DecidedSequenceDriver.after_begin(
                    driver, dict(bundle=bundle, target_mode=mode)
                )
                self.assertEqual(self.backend.gate_mode, "UNIFIED")
                self.publish_busy_after_tilt()
                self.assertFalse(self.backend.gate(PLAN))

    def publish_busy_after_tilt(self):
        value = status()
        value["instances"]["B-g2-tp2"]["unified_fullness"] = snapshot(self.now[0], 0.95)
        self.write(value)


class PreparationTests(unittest.TestCase):
    def test_forecast_prepares_before_confirmation_without_proposing(self):
        from tests.bridge_tp.test_six_card_decider import signal

        decider = d.ConversionDecider(
            profile="six_card", execute=True, rules=dict(decision_budget=0)
        )
        s = signal("L1AC", short=2, blocks=9000, trend=11000)
        s.update(kv_room=dict.fromkeys("ABC", True), quiet=dict.fromkeys("ABC", False))
        decider.signals = lambda *args: s
        modes = dict(A=1, B=2, C=1)
        record = decider.observe({}, modes, {}, 1000)
        self.assertEqual(record["tilt"], ["A"])
        self.assertEqual(record["preparation"]["phase"], "PREPARING")
        self.assertIsNone(record["proposal"])
        s["trend_blocks"] = 5000
        record = decider.observe({}, modes, {}, 1001)
        self.assertEqual(record["tilt"], [])
        self.assertIsNone(record["prepared"])

    def test_six_preparation_reads_route_evidence_instead_of_spare(self):
        value, modes, members = native_status("L1AC", short=14, blocks=13000)
        decider = d.ConversionDecider(profile="six_card")
        signals = decider.signals(value, modes, members, 10)
        self.assertFalse(any(signals["quiet"].values()))
        for row in value["instances"].values():
            row["pressure"]["zone"] = "CRITICAL"
            row["unified_fullness"] = snapshot(10)
        signals = decider.signals(value, modes, members, 10)
        self.assertTrue(all(signals["quiet"].values()))

    def test_preparation_can_start_while_busy_but_needs_capacity(self):
        decider = d.ConversionDecider(profile="six_card")
        signals = dict(
            kv_room=dict(A=True, B=True, C=True), quiet=dict(A=False, B=True, C=True)
        )
        decider.update_prepared(
            ("B->TP1", ("B", "TP1"), True), signals, dict(A=1, B=2, C=1), 10
        )
        self.assertIsNotNone(decider.prepared)
        decider.prepared = None
        signals["kv_room"]["A"] = False
        decider.update_prepared(
            ("B->TP1", ("B", "TP1"), True), signals, dict(A=1, B=2, C=1), 11
        )
        self.assertIsNone(decider.prepared)


class SixValveTests(Base):
    def setUp(self):
        super().setUp()
        self.backend.gate_mode = "UNIFIED"

    def test_sustained_hard_pressure_never_cancels_a_six_card_conversion(self):
        for _ in range(4):
            self.write(status(target_capacity=1.1))
            self.assertFalse(self.backend.kv_emergency(PLAN))
            self.now[0] += 2
        # The warning is recorded once; neither a continuation nor an abort
        # decision follows for the six-card gate.
        self.assertEqual(len(self.decisions("VALVE_WARNING")), 1)
        self.assertFalse(self.decisions("VALVE_ABORT"))
        self.assertFalse(self.decisions("VALVE_CONTINUE"))


class CarryGateTests(Base):
    """Gap policy merge: the retained side must carry the short requests."""

    def setUp(self):
        super().setUp()
        self.backend.gate_mode = "UNIFIED"
        self.backend.gate_carry = True

    def publish(self, short, capacity=1.6, full=1.4):
        # GPU 11-10 at the proposal: retained KV past the hard line, slow ITL.
        value = status(target_capacity=capacity)
        retained = value["instances"]["B-g2-tp2"]
        retained["unified_fullness"] = snapshot(self.now[0], full)
        retained["tp_degree"] = 2
        value["workload"] = dict(
            short_window_s=30,
            short_window_arrivals=round(short * 30),
            short_window_long=0,
            conversion_demand=dict(
                short_window_s=30,
                short_count=round(short * 30),
                sampled_at=self.now[0],
            ),
        )
        self.write(value)

    def test_merge_starts_under_long_pressure_when_shorts_are_carried(self):
        self.publish(short=1.9)
        self.assertTrue(self.backend.gate(PLAN))
        record = self.decisions("GATE")[-1]
        self.assertEqual(
            record["carry"], dict(short_rate=1.9, retained_short_capacity=4.0)
        )

    def test_merge_waits_while_the_retained_side_cannot_carry_the_shorts(self):
        self.publish(short=4.5, capacity=0.1, full=0.2)
        self.assertFalse(self.backend.gate(PLAN))
        self.assertIn("cannot carry", self.decisions("GATE")[-1]["reason"])

    def test_causal_arrivals_are_used_even_when_admissions_are_held(self):
        self.publish(short=5.0)
        value = self.backend.status()
        value["workload"]["short_window_arrivals"] = 10  # few actually dispatched
        self.write(value)
        self.assertFalse(self.backend.gate(PLAN))
        self.assertEqual(self.decisions("GATE")[-1]["carry"]["short_rate"], 5.0)

    def test_invalid_or_stale_arrivals_and_unknown_tp_cannot_open(self):
        for change in (
            dict(short_count=float("nan")),
            dict(short_count=-1),
            dict(short_window_s=0),
            dict(short_window_s=float("inf")),
            dict(sampled_at=self.now[0] - 3),
            dict(sampled_at=self.now[0] + 1),
        ):
            self.publish(short=1.9)
            value = self.backend.status()
            value["workload"]["conversion_demand"].update(change)
            self.write(value)
            self.assertFalse(self.backend.gate(PLAN))
        self.publish(short=1.9)
        value = self.backend.status()
        del value["instances"]["B-g2-tp2"]["tp_degree"]
        self.write(value)
        self.assertFalse(self.backend.gate(PLAN))

    def test_fresh_status_cannot_hide_old_or_incomplete_retained_evidence(self):
        for change in (
            dict(sampled_at=self.now[0] - 3),
            dict(published_at=self.now[0] + 1),
            dict(complete=False),
            dict(value=None),
        ):
            self.publish(short=1.9)
            value = self.backend.status()
            value["instances"]["B-g2-tp2"]["unified_fullness"].update(change)
            self.write(value)
            self.assertFalse(self.backend.gate(PLAN))

    def test_missing_arrival_window_or_identity_still_blocks(self):
        value = status(target_capacity=0.1)
        value["instances"]["B-g2-tp2"]["unified_fullness"] = snapshot(self.now[0])
        self.write(value)
        self.assertFalse(self.backend.gate(PLAN))
        self.assertIn("cannot carry", self.decisions("GATE")[-1]["reason"])
        value = status()
        del value["instances"]["B-g2-tp2"]
        self.write(value)
        self.assertFalse(self.backend.gate(PLAN))
        self.assertIn("mismatch", self.decisions("GATE")[-1]["reason"])

    def test_without_the_gap_policy_the_hard_item_still_blocks(self):
        self.backend.gate_carry = False
        self.publish(short=1.9)
        self.assertFalse(self.backend.gate(PLAN))
        self.assertIn("hard", self.decisions("GATE")[-1]["reason"])

    def test_split_is_unchanged_by_the_carry_check(self):
        self.publish(short=17.5, capacity=0.1, full=1.4)
        split = dict(PLAN, target_bundle_mode="TP1")
        self.assertTrue(self.backend.gate(split))
        self.assertIsNone(self.decisions("GATE")[-1]["carry"])


class PrestartTests(Base):
    def test_waiting_prestart_is_idempotent_and_withdrawal_stops_it(self):
        backend = self.backend
        backend.prestart_enabled = backend.prestart_at_proposal = True
        backend.registry = NS(get=lambda ref: NS(lifecycle="ACTIVE"))
        backend.channel_graph = backend.channel_plan = None
        backend.gate = lambda plan: False
        backend.replacement_specs = lambda plan: [
            ("replacement", 10, dict(instance_id="replacement"), {})
        ]
        started, stopped = [], []
        backend.ops.start = lambda extra: started.append(extra)
        backend.ops.stop = lambda ids: stopped.extend(ids)
        backend.write_layout = Mock()
        for _ in range(3):
            self.assertFalse(backend.precheck(PLAN))
        self.assertEqual(len(started), 1)
        self.assertFalse(pc.Path(started[0]["start_gate"]).exists())
        backend.write_layout.assert_not_called()  # sources still accept requests
        backend.restore_admission(PLAN)
        self.assertEqual(stopped, ["replacement"])
        self.assertIsNone(backend.prestarted)

    def test_gap_policy_prestarts_at_the_proposal_not_at_the_closure(self):
        started = []
        backend = self.backend
        backend.gate_mode = "UNIFIED"
        backend.prestart = lambda plan: started.append(plan["plan_id"])
        backend.registry = NS(get=lambda ref: NS(lifecycle="ACTIVE"))
        backend.channel_graph = backend.channel_plan = None
        backend.gate = lambda plan: False  # the start check is still waiting
        self.assertFalse(backend.precheck(PLAN))
        self.assertEqual(started, [])  # legacy: only at the closure
        backend.prestart_at_proposal = True
        self.assertFalse(backend.precheck(PLAN))
        self.assertEqual(started, ["manual-A"])


class SplitGateTests(Base):
    def setUp(self):
        super().setUp()
        self.backend.gate_mode = "UNIFIED"
        self.split_plan = dict(PLAN, target_bundle_mode="TP1")

    def publish(self, full=0.95, capacity=0.1):
        value = status(target_capacity=capacity)
        value["instances"]["B-g2-tp2"]["unified_fullness"] = snapshot(self.now[0], full)
        self.write(value)

    def test_split_skips_the_route_fullness_item(self):
        self.publish(full=0.95)
        self.assertTrue(self.backend.gate(self.split_plan))
        self.assertTrue(self.decisions("GATE")[-1]["split"])

    def test_split_still_requires_hard_capacity(self):
        self.publish(full=0.5, capacity=1.0)
        self.assertFalse(self.backend.gate(self.split_plan))
        self.assertIn("hard", self.decisions("GATE")[-1]["reason"])

    def test_split_still_requires_matching_identities(self):
        value = status()
        value["instances"]["B-g2-tp2"]["unified_fullness"] = snapshot(self.now[0], 0.5)
        del value["instances"]["B-g2-tp2"]
        self.write(value)
        self.assertFalse(self.backend.gate(self.split_plan))
        self.assertIn("mismatch", self.decisions("GATE")[-1]["reason"])
