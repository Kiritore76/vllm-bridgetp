# SPDX-License-Identifier: Apache-2.0
"""Unified scheduler (user 11-06) on CPU: smoothing, fullness, routing, hold,
migration triggers, and the pressure scheduler left unchanged by default."""

import dataclasses
import math
import types
import unittest

from tests.bridge_tp.test_pressure_policy import b, candidate, config, p, pool, r
from tools.bridge_tp.pressure_cpu import load

u = load("unified_scheduler")
hold_module = load("pressure_hold")
observation = load("pressure_observation")

TP = dict(a1=1, a2=1, b=2, c=2)
# The routing configuration of the six-card runs (L2C-s16, 11-06).
PRODUCTION = dict(
    max_age_s=2.0,
    min_decode_samples=20,
    spare_ratio=0.8,
    enter_ratio=1.2,
    leave_ratio=1.0,
    imbalance_enter=0.4,
    imbalance_leave=0.2,
    sustain_s=3,
    max_tick_gap_s=0.5,
    cooldown_s=2.0,
    prompt_boundary=2048,
    very_long_prompt_tokens=4096,
    prefill_pending_budget=2048,
    prefill_backlog_budget=8192,
    prefill_waiting_budget=4,
    capacity_guard_blocks=125,
    capacity_lookahead_s=8.0,
    capacity_only=False,
    kv_emergency_priority=True,
    kv_routing_priority=True,
    queue_routing=True,
    kv_short_exempt_blocks=64,
    overload_migrates=True,
    overload_sustain_s=10.0,
    overload_min_waiting=4,
    overload_kv_fraction=0.7,
    migration_current_length=True,
    migration_target_reserve_blocks=1171,
    migration_target_running_max=15,
    health_service=True,
    pressure_routing=True,
    workload_pressure=True,
    routing_relief=True,
    health_min_efficiency=0.9,
    health_target_load=0.8,
    survival_horizon=256,
    min_survival_lower=0.5,
    length_resolution=512,
    shadow_cancel=True,
    cancel_sustain_s=3.0,
    cancel_recover_eta=1.0,
)


def inst(name, total=11712, free=10000, requests=0, unallocated=0, **kwargs):
    return pool(
        name,
        **dict(
            dict(
                total_blocks=total,
                free_blocks=free,
                engine_requests=requests,
                unallocated_prompt_blocks=unallocated,
                context_limit=32768,
            ),
            **kwargs,
        ),
    )


def coordinator(slots=None, cfg=None):
    records = []
    co = r.PressureCoordinator(
        cfg or p.PressureConfig(**PRODUCTION),
        records.append,
        scheduler="unified",
        slots=slots or dict(a1=16, a2=16, b=16, c=16),
    )
    return co, records


def busy(s, pool_id, times, interval, streams=10):
    """Updates at ``times`` with every stream emitting at ``interval``."""
    for t0, t1 in zip(times, times[1:]):
        dt = t1 - t0
        s.update(pool_id, t1, dt * streams, dt / interval * streams, streams)


class SmoothingTests(unittest.TestCase):
    def test_the_first_update_starts_the_clock_then_rates_set_the_value(self):
        s = u.SmoothedItl(route_tau_s=5.0, migrate_tau_s=10.0)
        s.update("b", 0.0, 0.0, 0, 0)
        self.assertIsNone(s.value("b"))
        s.update("b", 0.2, 0.0, 0, 0)
        self.assertEqual(s.value("b"), 0.0)
        s = u.SmoothedItl()
        s.update("b", 0.0, 9.0, 300, 10)
        self.assertIsNone(s.value("b"))
        busy(s, "b", [0.0, 0.2], 0.025)
        self.assertAlmostEqual(s.value("b"), 0.025)
        self.assertAlmostEqual(s.value("b", "migrate"), 0.025)

    def test_exponential_weighting_by_elapsed_time(self):
        s = u.SmoothedItl(route_tau_s=5.0, migrate_tau_s=10.0)
        s.update("b", 0.0, 0.0, 0, 10)
        busy(s, "b", [0.0, 0.2], 0.025)
        busy(s, "b", [0.2, 5.2], 0.045)
        a = 1 - math.exp(-1)
        # The ratio of smoothed rates: time per second over tokens per second.
        seconds = 10
        tokens = a * 10 / 0.045 + (1 - a) * 10 / 0.025
        self.assertAlmostEqual(s.value("b"), seconds / tokens)

    def test_a_prefill_stall_counts_as_the_mean_does(self):
        s = u.SmoothedItl()
        s.update("b", 0.0, 0.0, 0, 10)
        busy(s, "b", [i * 0.2 for i in range(51)], 0.025)
        # 0.4s with no token at all: the value rises while the stall lasts.
        s.update("b", 10.2, 2.0, 0, 10)
        during = s.value("b")
        self.assertGreater(during, 0.025)
        s.update("b", 10.4, 2.0, 10, 10)
        self.assertGreater(s.value("b"), during)
        # A median would have stayed at 25ms; the mean of the last 10s is higher.
        self.assertGreater(s.value("b"), 0.026)

    def test_idle_decays_and_prefill_only_keeps_the_value(self):
        s = u.SmoothedItl()
        s.update("b", 0.0, 0.0, 0, 10)
        busy(s, "b", [0.0, 0.2], 0.030)
        s.update("b", 30.0, 0.0, 0, 3)
        self.assertAlmostEqual(s.value("b"), 0.030)
        s.update("b", 60.0, 0.0, 0, 0)
        self.assertLess(s.value("b"), 0.030 * math.exp(-6) * 1.01)
        self.assertIsNone(s.value("unknown"))
        with self.assertRaises(ValueError):
            s.update("b", 61.0, -1.0, 0, 0)


class DecodeProgressTests(unittest.TestCase):
    def test_closed_and_open_gaps_split_at_the_update_time(self):
        w = observation.NativeServiceWindow()
        w.add_pool("b")
        w.arrive("x", "b", 0.0)
        w.token("x", 1.0)  # first token: no interval yet
        w.token("x", 1.1)
        w.token("x", 1.3)
        # (1.0, 1.4]: closed 0.1 + 0.2, open 0.1 since the last token.
        seconds, tokens, active = w.decode_progress("b", 1.0, 1.4)
        self.assertEqual((tokens, active), (2, 1))
        self.assertAlmostEqual(seconds, 0.4)
        # Next window: the open gap closes at 1.6; only its new part counts.
        w.token("x", 1.6)
        seconds, tokens, _ = w.decode_progress("b", 1.4, 1.6)
        self.assertEqual(tokens, 1)
        self.assertAlmostEqual(seconds, 0.2)
        w.arrive("y", "b", 1.5)  # waiting for its first token: no decode time
        seconds, tokens, active = w.decode_progress("b", 1.6, 1.8)
        self.assertEqual((tokens, active), (0, 2))
        self.assertAlmostEqual(seconds, 0.2)


class FullnessTests(unittest.TestCase):
    def test_the_largest_share_of_its_own_limit(self):
        value = u.fullness(inst("b", free=2000, requests=4), 16, 0.020)
        self.assertAlmostEqual(value["kv"], (11712 - 2000) / 11712)
        self.assertAlmostEqual(value["slots"], 4 / 16)
        self.assertAlmostEqual(value["speed"], 0.020 / u.ITL_REFERENCE_S)
        self.assertEqual(value["value"], value["kv"])

    def test_tp1_and_tp2_are_compared_as_shares(self):
        tp1 = u.fullness(inst("a1", total=2325, free=2325), 16, None, extra_blocks=1024)
        tp2 = u.fullness(inst("b", free=9000), 16, None, extra_blocks=1024)
        self.assertGreater(tp1["kv"], 0.44)
        self.assertLess(tp2["kv"], tp1["kv"])

    def test_missing_evidence_counts_as_zero_and_bad_slots_fail(self):
        value = u.fullness(inst("b", requests=None, total=None), 16, None)
        self.assertEqual((value["kv"], value["speed"], value["value"]), (None, None, 0))
        with self.assertRaises(ValueError):
            u.fullness(inst("b"), 0, None)


class AdmissionTests(unittest.TestCase):
    def test_growth_guard_prefers_alternative_before_two_short_prompts_fill_C4(self):
        co, records = coordinator()
        # Observed C4 room98 then66, C5 room488; fullness slightly favors C4.
        co.unified_fullness = lambda reading, *args, **kwargs: dict(
            value=0.918 if reading.pool_id == "b" else 0.942
        )
        readings = [
            inst("b", total=2325, free=223, growth_blocks_s=4.6),
            inst("c", total=2325, free=613, growth_blocks_s=4.6),
        ]
        self.assertEqual(
            co.admission_multi("one", 512, 128, readings, TP, 10, "one").pool_id, "b"
        )
        self.assertEqual(
            co.admission_multi("two", 512, 128, readings, TP, 10, "two").pool_id, "c"
        )
        rows = records[-1]["unified_rows"]
        self.assertTrue(rows[0]["fits"])
        self.assertFalse(rows[0]["growth_guard_fits"])
        self.assertTrue(rows[1]["growth_guard_fits"])

    def test_growth_preference_is_not_an_additional_all_pool_gate(self):
        co, _ = coordinator()
        result = co.admission_multi(
            "one", 512, 128, [inst("b", free=180, growth_blocks_s=4.6)], TP, 10, "one"
        )
        self.assertEqual(result.pool_id, "b")

    def test_assist_reservation_blocks_no_fit_fallback_until_room_recovers(self):
        co, records = coordinator()
        target = inst("b", free=2455)
        co.ledger.reserve(
            p.Reservation("assist", "moving", "b", 1536, 0, 0, "migration"), target
        )
        with self.assertRaises(p.AdmissionDeferred) as caught:
            co.admission_multi("long", 16384, 8192, [target], TP, 10, "ingress")
        self.assertEqual(caught.exception.diagnostics["b"]["room_blocks"], 794)
        self.assertNotIn("ingress", co.ledger.entries)
        self.assertNotIn("long", co._owners)
        self.assertEqual(records, [])
        # A commit and partial native allocation do not discard remaining growth.
        co.ledger.settle("assist", "COMMITTED", evidence=True)
        co.ledger.observe("assist", allocated_blocks=1024, service_observed=True)
        self.assertEqual(co.ledger.pending("b")[0], 512)
        reduced = dataclasses.replace(target, free_blocks=1431)
        with self.assertRaises(p.AdmissionDeferred):
            co.admission_multi("long", 16384, 8192, [reduced], TP, 10, "ingress")
        # Different available target remains eligible; no all-pool blocking gate.
        result = co.admission_multi(
            "long", 16384, 8192, [reduced, inst("c")], TP, 10, "ingress"
        )
        self.assertEqual(result.pool_id, "c")

    def test_short_fallback_is_unchanged_during_assist(self):
        co, _ = coordinator()
        result = co.admission_multi(
            "short", 512, 128, [inst("b", free=125)], TP, 10, "ingress"
        )
        self.assertEqual(result.pool_id, "b")

    def test_stale_only_long_fallback_keeps_the_existing_bounded_wait_contract(self):
        co, _ = coordinator()
        result = co.admission_multi(
            "stale", 16384, 8192, [inst("b", free=125)], TP, 100, "ingress"
        )
        self.assertEqual(result.pool_id, "b")

    def test_a_long_prompt_goes_where_kv_is_free_not_where_requests_are_few(self):
        # Long-capacity L3:32 at 704s: C at 79% got the long request.
        co, records = coordinator(slots=dict(b=32, c=32))
        readings = [
            inst("c", free=2476, requests=16),
            inst("b", free=7575, requests=17),
        ]
        result = co.admission_multi("q", 16384, 8192, readings, TP, 10.0, "k")
        self.assertEqual(result.pool_id, "b")
        self.assertEqual(records[-1]["scheduler"], "unified")
        self.assertIn("unified_rows", records[-1])

    def test_slots_and_speed_also_count(self):
        co, _ = coordinator()
        for pool_id, interval in (("b", 0.050), ("c", 0.020)):
            co.itl.update(pool_id, 8.8, 0.0, 0, 5)
            busy(co.itl, pool_id, [8.8, 9.0], interval, streams=5)
        readings = [inst("b"), inst("c")]
        self.assertEqual(
            co.admission_multi("q", 512, 128, readings, TP, 10.0, "k").pool_id, "c"
        )
        co, _ = coordinator()
        readings = [inst("b", requests=15), inst("c", requests=2)]
        self.assertEqual(
            co.admission_multi("q", 512, 128, readings, TP, 10.0, "k").pool_id, "c"
        )

    def test_routed_but_unreported_requests_count_at_once(self):
        co, _ = coordinator()
        readings = [inst("b", free=11000), inst("c", free=11000)]
        first = co.admission_multi("q1", 16384, 8192, readings, TP, 10.0, "k1")
        second = co.admission_multi("q2", 16384, 8192, readings, TP, 10.0, "k2")
        self.assertNotEqual(first.pool_id, second.pool_id)

    def test_a_prompt_that_does_not_fit_is_not_chosen_while_another_fits(self):
        co, _ = coordinator()
        readings = [inst("b", free=1100), inst("c", free=5000, requests=15)]
        result = co.admission_multi("q", 16384, 8192, readings, TP, 10.0, "k")
        self.assertEqual(result.pool_id, "c")

    def test_context_and_fixed_routes_still_bind(self):
        co, _ = coordinator()
        short = [inst("b", context_limit=8192), inst("c")]
        result = co.admission_multi("q", 16384, 8192, short, TP, 10.0, "k")
        self.assertEqual(result.pool_id, "c")
        result = co.admission_multi(
            "q2",
            512,
            128,
            [inst("b"), inst("c")],
            TP,
            10.0,
            "k2",
            allowed_pool_id="b",
        )
        self.assertEqual(result.pool_id, "b")
        with self.assertRaises(p.RoutingError):
            co.admission_multi(
                "q3", 30000, 8192, [inst("b"), inst("c")], TP, 10.0, "k3"
            )

    def test_the_pressure_scheduler_is_the_default_and_unchanged(self):
        co = r.PressureCoordinator(config(), lambda record: None)
        self.assertEqual(co.scheduler, "pressure")
        with self.assertRaises(ValueError):
            r.PressureCoordinator(config(), lambda record: None, scheduler="unified")
        with self.assertRaises(ValueError):
            r.PressureCoordinator(config(), lambda record: None, scheduler="other")


class HoldTests(unittest.TestCase):
    def queue(self):
        return hold_module.HoldQueue(
            8, 300, 8192, 2.0, lambda pool_id: True, unified_guard_blocks=125
        )

    def job(self, prompt):
        return dict(request=dict(prompt=[1] * prompt))

    def test_a_prompt_that_fits_nowhere_does_not_stop_later_ones(self):
        q = self.queue()
        q.add(self.job(16384), 0.0)
        q.add(self.job(512), 0.1)
        pools = [inst("b", free=900)]
        job, allowed = q.next(pools, 1.0, 10.0)
        self.assertEqual(len(job["request"]["prompt"]), 512)
        self.assertEqual([x.pool_id for x in allowed], ["b"])
        self.assertIsNone(q.next(pools, 1.0, 10.0))
        self.assertEqual(q.waiting(), 1)
        job, _ = q.next([inst("b", free=5000)], 2.0, 10.0)
        self.assertEqual(len(job["request"]["prompt"]), 16384)

    def test_the_engine_queue_limit_and_sent_prompts_are_respected(self):
        q = self.queue()
        stream = types.SimpleNamespace(
            payload=dict(prompt=[1] * 16384),
            recorder=types.SimpleNamespace(tokens_seen=0),
            future=types.SimpleNamespace(done=lambda: False),
        )
        q.unstarted["b"] = [stream]
        q.add(self.job(16384), 0.0)
        # 2000 free - 1024 sent - 125 guard leaves no room for another 1024.
        self.assertIsNone(q.next([inst("b", free=2000)], 1.0, 10.0))
        q.unstarted["b"] = [stream] * 8
        self.assertIsNone(q.next([inst("b", free=11000)], 1.0, 10.0))

    def test_a_prompt_that_can_never_fit_fails_at_once(self):
        q = self.queue()
        q.add(self.job(40000), 0.0)
        with self.assertRaisesRegex(ValueError, "can never fit"):
            q.next([inst("a1", total=2325, free=2325)], 1.0, 10.0)

    def test_without_fresh_readings_the_wait_is_bounded(self):
        q = self.queue()
        q.add(self.job(512), 0.0)
        stale = [inst("b", sampled_at=1.0, published_at=1.0)]
        self.assertIsNone(q.next(stale, 100.0, 10.0))
        job, allowed = q.next(stale, 301.0, 10.0)
        self.assertEqual([x.pool_id for x in allowed], ["b"])
        self.assertGreaterEqual(job["held_s"], 300)

    def test_no_long_queue_and_no_tp1_rule(self):
        q = self.queue()
        q.add(self.job(16384), 0.0)
        self.assertEqual((len(q.short), len(q.long), q.held_long), (1, 0, ()))
        with self.assertRaises(ValueError):
            hold_module.HoldQueue(
                8,
                300,
                8192,
                2.0,
                lambda pool_id: True,
                tp1_long_kv=0.7,
                unified_guard_blocks=125,
            )


class Started:
    def __init__(self):
        self.calls = []

    def bind(self, request_id, session_id):
        return r.BoundAction(
            request_id, session_id, lambda: self.calls.append(request_id)
        )


def source_candidates(*sizes, owner="a1", generated=100, cap=8192):
    """Fixed-output requests: the predictor's remaining length is exact."""
    return [
        candidate(
            f"q{i}",
            owner=owner,
            prompt_tokens=size,
            generated=generated,
            max_output_tokens=cap,
            releasable_blocks=(size + generated) // 16,
            owned_blocks=(size + generated) // 16,
            remaining=b.bounded_length(
                (0.0, 0.0, 1.0, 0.0),
                (0, 1, 2),
                generated=generated,
                predicted_at=generated,
                max_output_tokens=cap,
                ignore_eos=True,
            ),
            source_tokens_s=30.0,
        )
        for i, size in enumerate(sizes)
    ]


class MigrationTests(unittest.TestCase):
    def tick(self, co, now, source, target, rows, started=None):
        # Fresh readings and inventory at this tick.
        source, target = (
            dataclasses.replace(x, sampled_at=now, published_at=now)
            for x in (source, target)
        )
        rows = [
            dataclasses.replace(c, sampled_at=now, published_at=now)
            if c.sampled_at == 10.0
            else c
            for c in rows
        ]
        actions = (
            {c.request_id: started.bind(c.request_id, f"s{now}") for c in rows}
            if started
            else None
        )
        return co.tick_multi(
            now, {"a1": source, "b": target}, TP, {("a1", "b"): rows}, actions
        )

    def test_kv_trigger_moves_the_largest_block_seconds_at_once(self):
        co, records = coordinator()
        started = Started()
        source = inst("a1", total=2325, free=150, requests=3)
        rows = source_candidates(512, 16384, 4096)
        result = self.tick(co, 10.0, source, inst("b", free=9000), rows, started)
        # Same remaining output: the most releasable blocks win.
        self.assertEqual(started.calls, ["q1"])
        self.assertEqual(result[2]["trigger"], "KV")
        decision = next(x for x in records if x["kind"] == "pressure_decision_unified")
        self.assertEqual(decision["execution"], "RESERVED")

    def test_pressure_trigger_needs_sustain_gap_and_hysteresis(self):
        co, _ = coordinator()
        started = Started()
        # 15 of 16 slots: pressure, not KV.
        source = inst("a1", total=2325, free=1800, requests=15)
        rows = source_candidates(2048)
        self.tick(co, 10.0, source, inst("b", free=9000), rows, started)
        self.assertEqual(started.calls, [])
        self.tick(co, 21.0, source, inst("b", free=9000), rows, started)
        self.assertEqual(started.calls, ["q0"])
        # Cooling: the same instances do not move again on pressure.
        co.ledger.entries.clear()
        self.tick(co, 22.0, source, inst("b", free=9000), rows, started)
        self.assertEqual(started.calls, ["q0"])

    def test_no_pressure_move_into_a_target_that_would_be_nearly_full(self):
        co, records = coordinator()
        started = Started()
        source = inst("a1", total=2325, free=1800, requests=16)
        # 11 of 16 now (gap 0.31), 12 of 16 after the move: above 0.7.
        target = inst("b", requests=11)
        rows = source_candidates(2048)
        self.tick(co, 10.0, source, target, rows, started)
        self.tick(co, 21.0, source, target, rows, started)
        self.assertEqual(started.calls, [])
        reasons = [
            rejected["reasons"]
            for x in records
            if x["kind"] == "pressure_decision_unified"
            for e in x["evaluations"]
            for rejected in e["rejected"]
        ]
        self.assertIn(["target would not stay below hysteresis"], reasons)

    def test_physical_safety_still_refuses(self):
        co, _ = coordinator()
        started = Started()
        source = inst("a1", total=2325, free=100, requests=3)
        rows = source_candidates(16384)
        self.tick(co, 10.0, source, inst("b", free=1000), rows, started)
        self.assertEqual(started.calls, [])
        stale = [candidate("q", owner="a1", sampled_at=1.0, published_at=1.0)]
        self.tick(co, 10.0, source, inst("b"), stale, started)
        self.assertEqual(started.calls, [])

    def test_a_kv_move_does_not_take_room_promised_to_pending_prompts(self):
        # GPU 11-06: B had 948 free after pending prompts and took a 1037-block
        # move, then preempted. Physically 1500 free fits 1030 blocks, but 1024
        # of them belong to a prompt awaiting prefill.
        co, records = coordinator()
        started = Started()
        source = inst("a1", total=2325, free=150, requests=3)
        rows = source_candidates(16384, 4096)
        target = inst("b", free=1500, unallocated=1024)
        self.tick(co, 10.0, source, target, rows, started)
        self.assertEqual(started.calls, ["q1"])
        reasons = [
            rejected
            for x in records
            if x["kind"] == "pressure_decision_unified"
            for e in x["evaluations"]
            for rejected in e["rejected"]
        ]
        self.assertIn(
            dict(
                request_id="q0",
                reasons=["target room promised to pending prompts"],
                target="b",
            ),
            reasons,
        )
        # Nothing fits: no move at all, and the source is left to its engine.
        co, _ = coordinator()
        started = Started()
        target = inst("b", free=1200, unallocated=1024)
        self.tick(co, 10.0, source, target, rows, started)
        self.assertEqual(started.calls, [])

    def test_short_or_nearly_finished_requests_are_never_moved(self):
        # Unified L2C:16 GPU 11-06: only short requests left on a TP1 at KV
        # trigger; one had finished, START failed and stopped the controller.
        co, records = coordinator()
        started = Started()
        source = inst("a1", total=2325, free=150, requests=20)
        rows = source_candidates(512, 512, generated=111, cap=128)
        rows += source_candidates(16384, generated=8100)
        rows = [dataclasses.replace(c, request_id=f"x{i}") for i, c in enumerate(rows)]
        self.tick(co, 10.0, source, inst("b", free=9000), rows, started)
        self.assertEqual(started.calls, [])
        reasons = {
            reason
            for e in records[-1]["evaluations"]
            for rejected in e["rejected"]
            for reason in rejected["reasons"]
        }
        self.assertIn("insufficient sustained work probability", reasons)

    def test_a_refused_start_releases_the_session_and_cools_the_request(self):
        co, records = coordinator()
        source = inst("a1", total=2325, free=150, requests=3)
        rows = source_candidates(16384)

        def refuse():
            error = ValueError("source ended or has no safe output budget")
            error.migration_not_started = True
            raise error

        actions = {"q0": r.BoundAction("q0", "s1", refuse)}
        now = 10.0
        fresh = [dataclasses.replace(c, sampled_at=now, published_at=now) for c in rows]
        readings = {
            "a1": dataclasses.replace(source, sampled_at=now, published_at=now),
            "b": dataclasses.replace(inst("b"), sampled_at=now, published_at=now),
        }
        self.assertIsNone(
            co.tick_multi(now, readings, TP, {("a1", "b"): fresh}, actions)
        )
        self.assertEqual(records[-1]["kind"], "pressure_start_refused")
        self.assertNotIn("s1", co._sessions)
        self.assertNotIn("s1", co.ledger.entries)
        # The same session id can be used again after the cooldown.
        started = Started()
        actions = {"q0": started.bind("q0", "s1")}
        later = 13.0
        fresh = [
            dataclasses.replace(c, sampled_at=later, published_at=later) for c in rows
        ]
        readings = {
            k: dataclasses.replace(v, sampled_at=later, published_at=later)
            for k, v in readings.items()
        }
        co.tick_multi(later, readings, TP, {("a1", "b"): fresh}, actions)
        self.assertEqual(started.calls, ["q0"])

    def test_any_other_start_error_still_stops(self):
        co, _ = coordinator()
        source = inst("a1", total=2325, free=150, requests=3)
        rows = source_candidates(16384)

        def broken():
            raise RuntimeError("published, then failed")

        with self.assertRaises(RuntimeError):
            self.tick(co, 10.0, source, inst("b"), rows)
            co.tick_multi(
                10.0,
                {"a1": dataclasses.replace(source, sampled_at=10.0), "b": inst("b")},
                TP,
                {("a1", "b"): rows},
                {"q0": r.BoundAction("q0", "s9", broken)},
            )

    def test_advisory_without_actions_and_one_session_at_a_time(self):
        co, records = coordinator()
        source = inst("a1", total=2325, free=150, requests=3)
        rows = source_candidates(16384)
        self.assertIsNone(self.tick(co, 10.0, source, inst("b"), rows))
        self.assertEqual(records[-1]["execution"], "ADVISORY")
        self.assertEqual(records[-1]["chosen"][2], "q0")
        started = Started()
        self.tick(co, 11.0, source, inst("b"), rows, started)
        self.assertEqual(started.calls, ["q0"])
        self.tick(co, 12.0, source, inst("b"), source_candidates(8192), started)
        self.assertEqual(started.calls, ["q0"])


if __name__ == "__main__":
    unittest.main()
