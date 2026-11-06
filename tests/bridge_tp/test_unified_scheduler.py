# SPDX-License-Identifier: Apache-2.0
"""Unified scheduler (user 11-06) on CPU: smoothing, fullness, routing, hold,
migration triggers, and the pressure scheduler left unchanged by default."""

import dataclasses
import math
import types
import unittest

from tests.bridge_tp.test_pressure_policy import candidate, config, p, pool, r
from tools.bridge_tp.pressure_cpu import load

u = load("unified_scheduler")
hold_module = load("pressure_hold")

TP = dict(a1=1, a2=1, b=2, c=2)


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
        cfg or config(capacity_lookahead_s=8.0),
        records.append,
        scheduler="unified",
        slots=slots or dict(a1=16, a2=16, b=16, c=16),
    )
    return co, records


class SmoothingTests(unittest.TestCase):
    def test_first_value_then_exponential_weighting_by_elapsed_time(self):
        s = u.SmoothedItl(route_tau_s=5.0, migrate_tau_s=10.0)
        s.update("b", 0.0, [0.020, 0.030, 0.025], active=3)
        self.assertAlmostEqual(s.value("b"), 0.025)
        s.update("b", 5.0, [0.045], active=3)
        a = 1 - math.exp(-1)
        self.assertAlmostEqual(s.value("b"), a * 0.045 + (1 - a) * 0.025)
        a = 1 - math.exp(-0.5)
        self.assertAlmostEqual(
            s.value("b", "migrate"), a * 0.045 + (1 - a) * 0.025
        )

    def test_one_prefill_stall_moves_the_route_value_little(self):
        s = u.SmoothedItl()
        s.update("b", 0.0, [0.025] * 10, active=10)
        # A long prefill makes a few intervals slow; the median ignores them.
        s.update("b", 0.2, [0.025] * 8 + [0.4, 0.4], active=10)
        self.assertAlmostEqual(s.value("b"), 0.025)
        # Even a whole slow update moves it by only a few percent.
        s.update("b", 0.4, [0.4] * 10, active=10)
        self.assertLess(s.value("b"), 0.045)

    def test_idle_decays_and_a_stall_keeps_its_value(self):
        s = u.SmoothedItl()
        s.update("b", 0.0, [0.030], active=1)
        s.update("b", 30.0, [], active=1)
        self.assertAlmostEqual(s.value("b"), 0.030)
        s.update("b", 60.0, [], active=0)
        self.assertLess(s.value("b"), 0.030 * math.exp(-5) * 1.01)
        self.assertIsNone(s.value("unknown"))


class FullnessTests(unittest.TestCase):
    def test_the_largest_share_of_its_own_limit(self):
        value = u.fullness(inst("b", free=2000, requests=4), 16, 0.020)
        self.assertAlmostEqual(value["kv"], (11712 - 2000) / 11712)
        self.assertAlmostEqual(value["slots"], 4 / 16)
        self.assertAlmostEqual(value["speed"], 0.020 / 0.035)
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
        co.itl.update("b", 9.0, [0.050] * 5, active=5)
        co.itl.update("c", 9.0, [0.020] * 5, active=5)
        readings = [inst("b"), inst("c")]
        self.assertEqual(co.admission_multi("q", 512, 128, readings, TP, 10.0, "k")
                         .pool_id, "c")
        co, _ = coordinator()
        readings = [inst("b", requests=15), inst("c", requests=2)]
        self.assertEqual(co.admission_multi("q", 512, 128, readings, TP, 10.0, "k")
                         .pool_id, "c")

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
            "q2", 512, 128, [inst("b"), inst("c")], TP, 10.0, "k2",
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

    def test_no_long_queue_and_no_tp1_rule(self):
        q = self.queue()
        q.add(self.job(16384), 0.0)
        self.assertEqual((len(q.short), len(q.long), q.held_long), (1, 0, ()))
        with self.assertRaises(ValueError):
            hold_module.HoldQueue(
                8, 300, 8192, 2.0, lambda pool_id: True, tp1_long_kv=0.7,
                unified_guard_blocks=125,
            )


class Started:
    def __init__(self):
        self.calls = []

    def bind(self, request_id, session_id):
        return r.BoundAction(
            request_id, session_id, lambda: self.calls.append(request_id)
        )


def source_candidates(*sizes, owner="a1"):
    return [
        candidate(
            f"q{i}",
            owner=owner,
            prompt_tokens=size,
            generated=100,
            max_output_tokens=8192,
            releasable_blocks=(size + 100) // 16,
            owned_blocks=(size + 100) // 16,
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

    def test_kv_trigger_moves_the_longest_context_at_once(self):
        co, records = coordinator()
        started = Started()
        source = inst("a1", total=2325, free=150, requests=3)
        rows = source_candidates(512, 16384, 4096)
        result = self.tick(co, 10.0, source, inst("b", free=9000), rows, started)
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
