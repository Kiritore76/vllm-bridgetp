# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; simulated inventory/actions are not GPU acceptance."""

import itertools
import math
import random
import unittest

from tools.bridge_tp.pressure_cpu import load

b = load("bounded_length")
p = load("pressure_policy")
r = load("pressure_runtime")


def config(**kwargs):
    return p.PressureConfig(
        **dict(
            prompt_boundary=2048,
            prefill_pending_budget=2048,
            prefill_waiting_budget=4,
            capacity_guard_blocks=125,
            **kwargs,
        )
    )


def pool(name="tp1", **kwargs):
    defaults = dict(
        pool_id=name,
        sampled_at=10.0,
        published_at=10.0,
        free_blocks=2000,
        block_size=16,
        context_limit=8192,
        decode_ratio=0.4,
        decode_samples=30,
        decode_sampled_at=10.0,
        idle=False,
        pending_prompt_tokens=0,
        waiting=0,
        growth_blocks_s=1.0,
        prefill_batch_tokens=0,
    )
    return p.PoolReading(**(defaults | kwargs))


def length(mean=4096, generated=20, cap=6000):
    # Degenerate original finite bin at mean+generated-predicted_at.
    return b.bounded_length(
        (0.0, 0.0, 1.0, 0.0),
        (0, mean - 1, mean),
        generated=generated,
        predicted_at=generated,
        max_output_tokens=cap,
    )


def candidate(name="a", **kwargs):
    defaults = dict(
        request_id=name,
        owner="tp1",
        sampled_at=10.0,
        published_at=10.0,
        generated=20,
        prompt_tokens=1024,
        max_output_tokens=6000,
        releasable_blocks=80,
        owned_blocks=80,
        transfer_bytes=1000,
        remaining=length(),
        prediction_status="AVAILABLE",
        source_decode_relief=0.4,
        target_decode_cost=0.2,
        estimated_release_s=1.3,
    )
    return p.Candidate(**(defaults | kwargs))


class BoundedTests(unittest.TestCase):
    def test_cap_crossing_and_tail_mass(self):
        x = b.bounded_length(
            (0.1, 0.5, 0.4), (0, 8), generated=3, predicted_at=0, max_output_tokens=7
        )
        # Surviving uniform mass: 3..8, of which 7..8 clips to C=4.
        self.assertAlmostEqual(x.pmf[4], (0.4 + 2 * 0.5 / 8) / (0.4 + 6 * 0.5 / 8))
        self.assertAlmostEqual(sum(x.pmf), 1)
        self.assertEqual(len(x.pmf), 5)
        self.assertEqual(x.probability_gt_bounds(4), (0, 0))
        self.assertGreater(x.probability_ge(4), 0)

    def test_zero_cap_and_ignore_eos(self):
        for generated in (7, 8, 9000):
            x = b.bounded_length(
                (0.1, 0.5, 0.4),
                (0, 8192),
                generated=generated,
                predicted_at=generated,
                max_output_tokens=7,
            )
            self.assertEqual(x.pmf, (1.0,))
        x = b.bounded_length(
            (0.1, 0.5, 0.4),
            (0, 8192),
            generated=3,
            predicted_at=0,
            max_output_tokens=7,
            ignore_eos=True,
        )
        self.assertEqual(x.mean, 4)
        self.assertEqual(x.probability_gt_bounds(3), (1, 1))
        self.assertEqual(x.probability_ge_bounds(4), (1, 1))

    def test_total_system_cap_not_remaining8192(self):
        x = b.bounded_length(
            (0, 0, 1),
            (0, 8192),
            generated=2000,
            predicted_at=1990,
            max_output_tokens=16000,
        )
        self.assertEqual(x.mean, 6192)

    def test_cap_boundaries_lag_and_probability_bounds(self):
        for cap, lag in itertools.product((0, 1, 4, 8, 9, 16), (0, 1, 4, 9)):
            x = b.bounded_length(
                (0.1, 0.3, 0.3, 0.3),
                (0, 8, 16),
                generated=lag,
                predicted_at=0,
                max_output_tokens=cap,
            )
            self.assertAlmostEqual(sum(x.pmf), 1)
            self.assertTrue(0 <= x.mean <= x.cap)
            for q in (0.5, 0.75, 1.0):
                self.assertLessEqual(x.quantile(q), x.cap)
            for horizon in range(-1, cap + 2):
                lo, hi = x.probability_gt_bounds(horizon)
                self.assertLessEqual(lo - 1e-12, x.probability_gt(horizon))
                self.assertLessEqual(x.probability_gt(horizon), hi + 1e-12)

    def test_bounds_contain_arbitrary_bin_placements(self):
        rng = random.Random(71)
        for _ in range(100):
            raw = [rng.random() for _ in range(4)]
            probs = tuple(v / sum(raw) for v in raw)
            lag, cap = rng.randrange(8), 8
            x = b.bounded_length(
                probs,
                (0, 8, 16),
                generated=lag,
                predicted_at=0,
                max_output_tokens=lag + cap,
            )
            # Place each WHOLE bin's mass at an arbitrary integer (no uniform).
            support = (0, rng.randint(1, 8), rng.randint(9, 16), 17)
            survive = sum(m for m, value in zip(probs, support) if value >= lag)
            for h in range(cap + 1):
                actual = (
                    sum(
                        m
                        for m, value in zip(probs, support)
                        if value >= lag and min(value - lag, cap) > h
                    )
                    / survive
                )
                lo, hi = x.probability_gt_bounds(h)
                self.assertLessEqual(lo - 1e-12, actual)
                self.assertLessEqual(actual, hi + 1e-12)

    def test_invalid_and_unavailable(self):
        for row in ({"status": "STALE"}, {"status": "UNAVAILABLE"}):
            with self.assertRaises(ValueError):
                b.from_advisory(row, 100)
        for probs in ((math.nan, 0, 1), (-0.1, 0.5, 0.6), (0, 0, 0)):
            with self.assertRaises(ValueError):
                b.bounded_length(
                    probs, (0, 8), generated=0, predicted_at=0, max_output_tokens=4
                )
        with self.assertRaisesRegex(ValueError, "survival"):
            b.bounded_length(
                (1, 0, 0), (0, 8), generated=1, predicted_at=0, max_output_tokens=4
            )
        with self.assertRaisesRegex(ValueError, "unresolved"):
            b.bounded_length(
                (0, 0, 1), (0, 8), generated=0, predicted_at=0, max_output_tokens=100
            )


class PressureTests(unittest.TestCase):
    def decide(self, source=None, target=None, candidates=None, cfg=None):
        policy = p.PressurePolicy(cfg or config(sustain_s=0), p.ReservationLedger())
        return policy.decide(
            10,
            source or pool(decode_ratio=1.5),
            target or pool("tp4"),
            candidates or [candidate()],
        )

    def test_light_pools_stay_and_source_pressure_selects(self):
        self.assertEqual(self.decide(source=pool()).action, "STAY")
        self.assertEqual(self.decide().action, "START_SHADOW")

    def test_longer_over_smaller_faster_kv(self):
        short = candidate(
            "short",
            remaining=length(512),
            releasable_blocks=20,
            owned_blocks=20,
            transfer_bytes=1,
        )
        result = self.decide(candidates=[short, candidate("long")])
        self.assertEqual(result.request_id, "long")

    def test_near_length_ties_choose_service_cost_then_bytes_then_id(self):
        a = candidate("a", remaining=length(4100), releasable_blocks=70)
        z = candidate("z", remaining=length(4096), transfer_bytes=2000)
        self.assertEqual(self.decide(candidates=[a, z]).request_id, "a")
        self.assertEqual(
            self.decide(candidates=[candidate("z"), candidate("a")]).request_id, "a"
        )

    def test_service_tier_prefers_source_relief_not_largest_KV(self):
        light = candidate(
            "light",
            releasable_blocks=40,
            owned_blocks=40,
            source_decode_relief=0.6,
            target_decode_cost=0.1,
            transfer_bytes=1000,
        )
        heavy = candidate(
            "heavy", source_decode_relief=0.2, target_decode_cost=0.1, transfer_bytes=1
        )
        result = self.decide(candidates=[heavy, light])
        self.assertEqual(result.request_id, "light")
        selected = next(r for r in result.candidates if r["selected"])
        self.assertIn("service:", selected["selection_policy"])

    def test_service_tier_prefers_lower_target_cost_before_release_bytes(self):
        costly = candidate("costly", target_decode_cost=0.4, transfer_bytes=1)
        cheap = candidate(
            "cheap",
            target_decode_cost=0.1,
            transfer_bytes=1000,
            releasable_blocks=40,
            owned_blocks=40,
        )
        self.assertEqual(self.decide(candidates=[costly, cheap]).request_id, "cheap")

    def test_KV_prefers_timely_sufficient_release_not_longest_remaining(self):
        fast = candidate(
            "fast",
            remaining=length(1024),
            releasable_blocks=40,
            owned_blocks=40,
            estimated_release_s=1,
            transfer_bytes=2000,
        )
        long = candidate(
            "long", remaining=length(5000), estimated_release_s=4, transfer_bytes=1000
        )
        source = pool(free_blocks=100)
        self.assertEqual(
            self.decide(source=source, candidates=[long, fast]).request_id, "fast"
        )
        # The same two requests under service pressure keep the longest tier.
        self.assertEqual(self.decide(candidates=[long, fast]).request_id, "long")

    def test_KV_same_release_budget_prefers_less_transport_not_larger_KV(self):
        small = candidate(
            "small", releasable_blocks=40, owned_blocks=40, transfer_bytes=1000
        )
        large = candidate("large", releasable_blocks=80, transfer_bytes=2000)
        result = self.decide(source=pool(free_blocks=100), candidates=[large, small])
        self.assertEqual(result.request_id, "small")
        selected = next(r for r in result.candidates if r["selected"])
        self.assertIn("KV:", selected["selection_policy"])

    def test_KV_short_or_insufficient_or_too_late_candidates_still_rejected(self):
        requests = [
            candidate("short", remaining=length(20), estimated_release_s=0.1),
            candidate("small", releasable_blocks=20, estimated_release_s=0.1),
            candidate("late", estimated_release_s=101),
            candidate("safe"),
        ]
        result = self.decide(source=pool(free_blocks=100), candidates=requests)
        self.assertEqual(result.request_id, "safe")
        for row in result.candidates:
            if row["request"]["request_id"] != "safe":
                self.assertTrue(row["rejections"])
                self.assertFalse(row["selection_eligible"])

    def test_KV_missing_prediction_is_last_resort_and_short_stop_is_rejected(self):
        unknown = candidate(
            "unknown",
            remaining=None,
            prediction_status="UNAVAILABLE",
            estimated_release_s=0.1,
        )
        result = self.decide(
            source=pool(free_blocks=100), candidates=[unknown, candidate("known")]
        )
        self.assertEqual(result.request_id, "known")
        short = candidate(
            "short",
            remaining=None,
            prediction_status="UNAVAILABLE",
            max_output_tokens=200,
        )
        result = self.decide(source=pool(free_blocks=100), candidates=[short])
        self.assertEqual(result.action, "STAY")
        self.assertIn(
            "stop budget cannot cover migration horizon",
            result.candidates[0]["rejections"],
        )

    def test_service_requires_known_release_budget(self):
        for budget in (None, math.nan, math.inf, -1):
            result = self.decide(candidates=[candidate(estimated_release_s=budget)])
            self.assertEqual(result.action, "STAY")

    def test_invalid_transport_sizes_never_enter_either_ranking(self):
        for source in (pool(decode_ratio=1.5), pool(free_blocks=100)):
            for size in (None, math.nan, math.inf, -1):
                result = self.decide(
                    source=source, candidates=[candidate(transfer_bytes=size)]
                )
                self.assertEqual(result.action, "STAY")

    def test_path_selection_is_stable_under_input_order(self):
        requests = [candidate("a"), candidate("b")]
        for source in (pool(decode_ratio=1.5), pool(free_blocks=100)):
            for order in (requests, list(reversed(requests))):
                self.assertEqual(
                    self.decide(source=source, candidates=order).request_id, "a"
                )

    def test_target_overload_missing_and_prefill_are_not_idle(self):
        for target in (
            pool("tp4", decode_ratio=1.5),
            pool("tp4", decode_ratio=None, decode_samples=0),
            pool("tp4", pending_prompt_tokens=14338),
        ):
            self.assertEqual(self.decide(target=target).action, "STAY")

    def test_stale_future_missing_inventory_rejected(self):
        for c in (
            candidate(sampled_at=7),
            candidate(published_at=11),
            candidate(releasable_blocks=None),
            candidate(owner="tp4"),
            candidate(migrating=True),
            candidate(finished=True),
            candidate(prediction_status="STALE"),
        ):
            self.assertEqual(self.decide(candidates=[c]).action, "STAY")

    def test_capacity_bypasses_guard_urgency_and_uses_release_requirement(self):
        source = pool(free_blocks=100, urgency=2, guard_time_s=0.1)
        result = self.decide(source=source, cfg=config(sustain_s=100, cooldown_s=100))
        self.assertEqual(result.action, "START_SHADOW")
        self.assertTrue(result.capacity_path)
        self.assertEqual(result.required_release_blocks, 25)
        too_small = candidate(releasable_blocks=20)
        self.assertEqual(
            self.decide(source=source, candidates=[too_small]).action, "STAY"
        )

    def test_capacity_missing_prediction_fallback_is_explicit(self):
        source = pool(free_blocks=100)
        c = candidate(remaining=None, prediction_status="UNAVAILABLE")
        result = self.decide(source=source, candidates=[c])
        self.assertEqual(result.action, "START_SHADOW")
        self.assertTrue(result.candidates[0]["prediction_fallback"])
        self.assertIsNone(result.candidates[0]["remaining"])

    def test_emergency_prefill_is_stricter_but_not_blanket_ban(self):
        source = pool(free_blocks=100)
        self.assertEqual(
            self.decide(
                source=source, target=pool("tp4", pending_prompt_tokens=1024)
            ).action,
            "START_SHADOW",
        )
        self.assertEqual(
            self.decide(
                source=source, target=pool("tp4", pending_prompt_tokens=14338)
            ).action,
            "STAY",
        )

    def test_capacity_never_waits_for_physical_exhaustion(self):
        self.assertEqual(
            self.decide(
                source=pool(free_blocks=1),
                candidates=[candidate(estimated_release_s=2)],
            ).action,
            "STAY",
        )

    def test_post_move_reverse_imbalance_and_overload(self):
        for c in (
            candidate(target_decode_cost=1.0),
            candidate(source_decode_relief=1.5, target_decode_cost=0.3),
        ):
            self.assertEqual(self.decide(candidates=[c]).action, "STAY")

    def test_persistence_hysteresis_gap_and_cooldown(self):
        policy = p.PressurePolicy(config(), p.ReservationLedger())

        def tick(t, ratio=1.5):
            return policy.decide(
                t,
                pool(
                    sampled_at=t,
                    published_at=t,
                    decode_sampled_at=t,
                    decode_ratio=ratio,
                ),
                pool("tp4", sampled_at=t, published_at=t, decode_sampled_at=t),
                [candidate(sampled_at=t, published_at=t)],
            )

        self.assertEqual(tick(10).action, "STAY")
        self.assertEqual(tick(10.5, 1.1).action, "STAY")
        self.assertEqual(tick(11, 1.1).action, "START_SHADOW")
        policy.note_started(11)
        for t in (11.5, 12, 12.5):
            self.assertEqual(tick(t).action, "STAY")
        self.assertEqual(tick(13).action, "START_SHADOW")
        self.assertEqual(tick(14).action, "STAY")  # gap breaks continuous evidence
        with self.assertRaises(ValueError):
            tick(13)


class LedgerRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.co = r.PressureCoordinator(config(sustain_s=0), self.rows.append)

    def test_admission_cap_is_frozen_and_context_feasible(self):
        value = self.co.admission(
            "new",
            2048,
            8192,
            (pool(), pool("tp4", context_limit=16384)),
            now=10,
            key="admit",
        )
        self.assertEqual(value.pool_id, "tp4")
        self.assertEqual(value.max_output_tokens, 8192)
        value2 = self.co.admission(
            "small",
            512,
            8192,
            (pool(), pool("tp4", decode_ratio=1.5, context_limit=16384)),
            now=10,
            key="admit2",
        )
        self.assertEqual(value2.pool_id, "tp4")
        self.assertEqual(value2.max_output_tokens, 8192)

    def test_migration_uses_actual_free_blocks_without_ingress_promises(self):
        target = pool("tp4", free_blocks=450)
        self.co.admission(
            "new",
            512,
            6000,
            (pool(decode_ratio=1.5, free_blocks=20), target),
            now=10,
            key="a",
            allowed_pool_id="tp4",
        )
        self.assertEqual(self.co.ledger.pending("tp4"), (0, 0, 512))
        result = self.co.tick(10, pool(decode_ratio=1.5), target, [candidate()])
        self.assertEqual(result.action, "START_SHADOW")
        target = pool("tp4", free_blocks=418)  # ingress actually allocated 32 blocks
        result = self.co.tick(10, pool(decode_ratio=1.5), target, [candidate()])
        self.assertEqual(result.action, "STAY")
        self.assertIn("physical commitments", result.candidates[0]["rejections"][0])

    def start(self, callback=lambda: None):
        return self.co.tick(
            10,
            pool(decode_ratio=1.5),
            pool("tp4"),
            [candidate("a"), candidate("b")],
            {"a": r.BoundAction("a", "s1", callback)},
        )

    def test_new_target_arrival_invalidates_previous_idle_for_migration(self):
        target = pool("tp4", idle=True, decode_ratio=None, decode_samples=0)
        self.co.admission(
            "queued",
            512,
            1024,
            (pool(), target),
            now=10,
            key="queued",
            allowed_pool_id="tp4",
        )
        result = self.co.tick(10, pool(decode_ratio=1.5), target, [candidate()])
        self.assertEqual(result.action, "STAY")
        self.assertEqual(result.target.zone, "UNKNOWN")
        self.assertEqual(self.co.ledger.pending("tp4")[:2], (0, 0))

    def test_same_request_action_one_active_and_idempotent_commit(self):
        calls = []
        result = self.start(lambda: calls.append("a"))
        self.assertEqual(result.request_id, "a")
        self.assertEqual(calls, ["a"])
        self.assertEqual(
            self.co.tick(
                10.1, pool(decode_ratio=1.5), pool("tp4"), [candidate("b")], {}
            ).action,
            "STAY",
        )
        with self.assertRaises(ValueError):
            self.co.finish(
                "s1",
                "b",
                "COMMITTED",
                ownership_confirmed=True,
                source_release_confirmed=True,
            )
        with self.assertRaises(ValueError):
            self.co.finish("s1", "a", "COMMITTED", ownership_confirmed=True)
        for _ in range(2):
            self.co.finish(
                "s1",
                "a",
                "COMMITTED",
                ownership_confirmed=True,
                source_release_confirmed=True,
            )
        self.assertGreater(self.co.ledger.pending("tp4")[0], 0)
        blocks = self.co.ledger.entries["s1"].blocks
        self.co.ledger.observe("s1", allocated_blocks=blocks, service_observed=True)
        self.assertEqual(self.co.ledger.pending("tp4"), (0, 0, 0))
        with self.assertRaisesRegex(ValueError, "ownership"):
            self.co.tick(10.2, pool(), pool("tp4"), [candidate("a")])

    def test_start_exception_keeps_capacity_until_confirmed_cleanup(self):
        def failure():
            raise OSError("publication outcome unknown")

        with self.assertRaises(OSError):
            self.start(failure)
        self.assertFalse(self.co.ledger.channel_available("tp4"))
        with self.assertRaises(ValueError):
            self.co.finish("s1", "a", "FAILED")
        for _ in range(2):
            self.co.finish("s1", "a", "FAILED", cleanup_confirmed=True)
        self.assertEqual(self.co.ledger.pending("tp4"), (0, 0, 0))

    def test_partial_allocation_does_not_double_count(self):
        self.start()
        blocks = self.co.ledger.entries["s1"].blocks
        self.co.ledger.observe("s1", allocated_blocks=20, service_observed=False)
        self.assertEqual(self.co.ledger.pending("tp4")[0], blocks - 20)
        with self.assertRaises(ValueError):
            self.co.ledger.observe("s1", allocated_blocks=10, service_observed=False)

    def test_binding_mismatch_prevents_start(self):
        with self.assertRaises(ValueError):
            self.co.tick(
                10,
                pool(decode_ratio=1.5),
                pool("tp4"),
                [candidate()],
                {"a": r.BoundAction("b", "session", lambda: None)},
            )
        self.assertFalse(self.co.ledger.entries)

    def test_all_candidates_receive_pmf_and_publication_gate(self):
        class Reader:
            def __init__(self):
                self.calls = []

            def advisory(self, request_id, output_tokens, *args, **kwargs):
                self.calls.append(request_id)
                return dict(
                    status="AVAILABLE",
                    probabilities=[0, 1, 0],
                    category_upper_edges=[0, 8192],
                    output_tokens=output_tokens,
                    prediction_output_tokens=output_tokens,
                    captured_unix_ns=9_000_000_000,
                    published_unix_ns=(11 if request_id == "b" else 10) * 1_000_000_000,
                )

        reader = Reader()
        rows = self.co.with_predictions(
            [candidate("a"), candidate("b")], reader, 10, 2000, {"a": False, "b": False}
        )
        self.assertEqual(reader.calls, ["a", "b"])
        self.assertIsNotNone(rows[0].remaining)
        self.assertIsNone(rows[1].remaining)
        self.assertEqual(rows[1].prediction_status, "PUBLICATION_NOT_PROVEN")


if __name__ == "__main__":
    unittest.main()
