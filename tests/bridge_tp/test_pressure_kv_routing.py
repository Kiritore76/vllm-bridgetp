# SPDX-License-Identifier: Apache-2.0
"""KV protection overrides service pressure only during warning/recovery."""

import dataclasses
import unittest

from tests.bridge_tp.test_pressure_workload import cfg
from tests.bridge_tp.test_pressure_workload import reading as base_reading
from tools.bridge_tp.pressure_cpu import load

p = load("pressure_policy")
k = load("pressure_kv_routing")
r = load("pressure_runtime")


def reading(name="tp1", now=10, load=0.2, **changes):
    return dataclasses.replace(base_reading(name, now=now, load=load), **changes)


def config(enabled=True):
    return dataclasses.replace(
        cfg(), capacity_lookahead_s=8.007738590240479, kv_routing_priority=enabled
    )


def route(source, target, prompt=512, *, enabled=True, state=None, **kwargs):
    return p.admit(
        "new",
        prompt,
        512,
        (source, target),
        config(enabled),
        p.ReservationLedger(),
        10,
        "new-key",
        kv_routing_state=state,
        **kwargs,
    )


class KVRoutingTests(unittest.TestCase):
    def test_normal_kv_retains_exact_original_service_choice(self):
        for a, b in [(0.2, 5), (3, 2), (2, 5), (0.5, 0.2)]:
            source = reading(load=a, free_blocks=3000)
            target = reading("tp4", load=b, free_blocks=10000)
            old = route(source, target, enabled=False)
            new = route(source, target)
            for name in (
                "pool_id",
                "routing_reason",
                "routing_pressure",
                "routing_capacity",
            ):
                self.assertEqual(getattr(old, name), getattr(new, name))
            self.assertEqual(new.routing_kv["mode"], "SERVICE_PRESSURE")
            self.assertFalse(new.routing_kv["redirect"])

    def test_recorded_32_block_request_prefers_kv_headroom_despite_busy_target(self):
        source = reading(
            load=2,
            free_blocks=179,
            unallocated_prompt_blocks=128,
            growth_blocks_s=7.254206456561281,
        )
        target = reading(
            "tp4",
            load=5.17063888763126,
            free_blocks=6529,
            unallocated_prompt_blocks=6209,
            growth_blocks_s=20.606157426250263,
        )
        self.assertEqual(route(source, target, enabled=False).pool_id, "tp1")
        new = route(source, target)
        self.assertEqual(new.pool_id, "tp4")
        self.assertEqual(new.routing_capacity["tp1"]["available_blocks"], 51)
        self.assertEqual(new.routing_capacity["tp4"]["available_blocks"], 320)
        self.assertGreater(new.routing_kv["target"]["pressure"], 1)
        self.assertTrue(new.routing_kv["changed_route"])

    def test_both_no_fit_can_choose_safer_native_queue_without_faking_kv(self):
        source = reading(
            load=2,
            free_blocks=288,
            unallocated_prompt_blocks=256,
            growth_blocks_s=13.68,
        )
        target = reading(
            "tp4",
            load=5.42,
            free_blocks=6475,
            unallocated_prompt_blocks=6160,
            growth_blocks_s=14.67,
        )
        new = route(source, target, prompt=6144)
        self.assertEqual(new.pool_id, "tp4")
        self.assertTrue(new.routing_kv["native_queue_only"])
        self.assertTrue(
            all(v["status"] == "NO_FIT" for v in new.routing_capacity.values())
        )
        self.assertEqual(new.reservation_basis, "none")

    def test_worse_target_kv_unknown_or_context_infeasible_is_not_forced(self):
        source = reading(load=2, free_blocks=179, unallocated_prompt_blocks=128)
        for changes in [
            dict(free_blocks=100, unallocated_prompt_blocks=500),
            dict(sampled_at=7, published_at=7),
            dict(context_limit=256),
        ]:
            target = reading("tp4", load=5, **changes)
            new = route(source, target)
            self.assertEqual(new.pool_id, route(source, target, enabled=False).pool_id)
            if new.routing_kv:
                self.assertFalse(new.routing_kv["redirect"])

    def test_fixed_and_nonpressure_routes_are_unchanged(self):
        source = reading(load=2, free_blocks=179, unallocated_prompt_blocks=128)
        target = reading("tp4", load=5, free_blocks=10000)
        fixed = route(source, target, allowed_pool_id="tp1")
        self.assertEqual(fixed.pool_id, "tp1")
        self.assertIsNone(fixed.routing_kv)
        for strategy in ("weighted-round-robin", "least-inflight"):
            old = route(source, target, enabled=False, routing_strategy=strategy)
            new = route(source, target, routing_strategy=strategy)
            self.assertEqual(old.pool_id, new.pool_id)
            self.assertIsNone(new.routing_kv)

    def test_risk_counts_native_unpublished_and_migration_demand_once(self):
        source = reading(
            free_blocks=1000, unallocated_prompt_blocks=80, growth_blocks_s=0
        )
        ledger = p.ReservationLedger()
        ledger.track_ingress("i", "unpublished", "tp1", 160)
        ledger.reserve(
            p.Reservation("m", "moving", "tp1", 50, 0, 0, "migration"), source
        )
        value = k.risk(source, ledger, config(), 10, 512)
        self.assertEqual(value["unpublished_prompt_blocks"], 10)
        self.assertEqual(value["migration_pending_blocks"], 50)
        self.assertAlmostEqual(value["pressure"], (80 + 10 + 50 + 32 + 125) / 1000)
        self.assertEqual(source.unallocated_prompt_blocks, 80)

    def test_recovery_requires_fresh_stable_margin_not_one_free_snapshot(self):
        state = k.KVRouteProtection()
        c = config()
        state.observe(dict(pressure=2, sampled_at=10), c, 10)
        state.observe(dict(pressure=0.9, sampled_at=11), c, 11)
        self.assertTrue(state.active)
        state.observe(dict(pressure=0.7, sampled_at=12), c, 12)
        state.observe(dict(pressure=0.7, sampled_at=12), c, 13)
        state.observe(dict(pressure=0.7, sampled_at=14), c, 14)
        self.assertTrue(state.active)
        state.observe(dict(pressure=0.7, sampled_at=15), c, 15)
        self.assertFalse(state.active)
        state.observe(dict(pressure=2, sampled_at=16), c, 16)
        state.observe(None, c, 17)
        state.observe(dict(pressure=0.7, sampled_at=20), c, 20)
        self.assertTrue(state.active)
        self.assertEqual(state.recovery_since, 20)

    def test_actual_coordinator_latches_and_returns_to_service_routing(self):
        records = []
        co = r.PressureCoordinator(config(), records.append)
        source = reading(free_blocks=179, unallocated_prompt_blocks=128, load=2)
        target = reading("tp4", free_blocks=10000, load=5)
        first = co.admission(
            "first", 512, 512, (source, target), now=10, key="first-key"
        )
        self.assertEqual(first.pool_id, "tp4")
        self.assertTrue(records[-1]["kv_routing_state_after"]["active"])
        for now in (11, 12, 13, 14):
            source = reading(now=now, load=0.2, free_blocks=2000, growth_blocks_s=0)
            target = reading(
                "tp4", now=now, load=5, free_blocks=10000, growth_blocks_s=0
            )
            value = co.admission(
                str(now), 512, 512, (source, target), now=now, key=str(now)
            )
            self.assertEqual(value.pool_id, "tp1" if now == 14 else "tp4")
        self.assertFalse(co.policy.kv_route.active)

    def test_new_opt_in_flag_is_validated_and_changes_identity_only_when_enabled(self):
        self.assertNotEqual(config().sha256, config(False).sha256)
        with self.assertRaises(ValueError):
            dataclasses.replace(config(), kv_routing_priority=1)
        with self.assertRaises(ValueError):
            dataclasses.replace(config(), capacity_lookahead_s=0)
