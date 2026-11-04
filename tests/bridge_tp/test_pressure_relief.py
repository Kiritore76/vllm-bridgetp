# SPDX-License-Identifier: Apache-2.0
"""Router intervention acknowledgement, native drain barrier and hysteresis."""

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp.test_pressure_policy import candidate, p, r
from tests.bridge_tp.test_pressure_workload import cfg, degraded, reading
from tools.bridge_tp.pressure_cpu import load


def source(now, **changes):
    return replace(degraded(now), **(dict(prefill_generation=0) | changes))


def coordinator():
    records = []
    co = r.PressureCoordinator(cfg(routing_relief=True), records.append)
    return co, records


def tick(co, now, **changes):
    return co.tick(
        now,
        source(now, **changes),
        reading("tp4", now=now),
        [candidate(sampled_at=now, published_at=now)],
    )


class ReliefTests(unittest.TestCase):
    def test_no_arrivals_still_confirms_post_ack_quiet_window(self):
        co, records = coordinator()
        for now in range(10, 15):
            result = tick(co, now)
            self.assertEqual(result.action, "START_SHADOW" if now == 14 else "STAY")
        self.assertEqual(co.policy.relief.epoch, 1)
        self.assertEqual(co.policy.relief.redirected_requests, 0)
        self.assertIsNotNone(records[-1]["routing_relief"]["acknowledged_at"])

    def test_unacknowledged_policy_never_claims_router_intervention(self):
        policy = p.PressurePolicy(cfg(routing_relief=True), p.ReservationLedger())
        for now in range(10, 18):
            self.assertEqual(
                policy.decide(
                    now,
                    source(now),
                    reading("tp4", now=now),
                    [candidate(sampled_at=now, published_at=now)],
                ).action,
                "STAY",
            )
        self.assertEqual(policy.relief.phase, "WAIT_ROUTER")

    def overload_policy(self):
        return p.PressurePolicy(
            cfg(
                routing_relief=True,
                kv_routing_priority=True,
                kv_short_exempt_blocks=64,
                overload_migrates=True,
                overload_sustain_s=2,
            ),
            p.ReservationLedger(),
        )

    def test_sustained_overload_moves_a_long_request_past_a_full_queue(self):
        policy = self.overload_policy()
        results = [
            policy.decide(
                now,
                source(now, waiting=50, pending_prompt_tokens=25600, decode_ratio=2),
                reading("tp4", now=now, waiting=500, pending_prompt_tokens=256000),
                [candidate(sampled_at=now, published_at=now)],
            )
            for now in range(10, 14)
        ]
        self.assertEqual([r.action for r in results[:2]], ["STAY", "STAY"])
        self.assertEqual(results[-1].action, "START_SHADOW")
        self.assertEqual(results[-1].reason, "overload relief")

    def test_a_target_only_has_to_hold_the_current_length_when_asked(self):
        def decide(**changes):
            policy = p.PressurePolicy(
                cfg(
                    routing_relief=True,
                    kv_routing_priority=True,
                    kv_short_exempt_blocks=64,
                    overload_migrates=True,
                    overload_sustain_s=2,
                    **changes,
                ),
                p.ReservationLedger(),
            )
            for now in range(10, 14):
                result = policy.decide(
                    now,
                    source(
                        now, waiting=50, pending_prompt_tokens=25600, decode_ratio=2
                    ),
                    reading("tp4", now=now, free_blocks=100),
                    [candidate(sampled_at=now, published_at=now)],
                )
            return result

        # 1024 prompt + 20 generated is 66 blocks; with 6000 of output, 440.
        self.assertIn(
            "target physical commitments exceed free blocks",
            decide().candidates[0]["rejections"],
        )
        result = decide(
            migration_current_length=True, migration_target_reserve_blocks=0
        )
        self.assertEqual(result.action, "START_SHADOW")
        self.assertEqual(result.candidates[0]["target_required_blocks"], 66)

    def overload_on_target(self, **target):
        policy = p.PressurePolicy(
            cfg(
                routing_relief=True,
                kv_routing_priority=True,
                kv_short_exempt_blocks=64,
                overload_migrates=True,
                overload_sustain_s=2,
                migration_current_length=True,
            ),
            p.ReservationLedger(),
        )
        for now in range(10, 14):
            result = policy.decide(
                now,
                source(now, waiting=50, pending_prompt_tokens=25600, decode_ratio=2),
                replace(reading("tp4", now=now), **target),
                [candidate(sampled_at=now, published_at=now)],
            )
        return result

    def pressed(self, **changes):
        config = cfg(
            routing_relief=True,
            kv_routing_priority=True,
            kv_short_exempt_blocks=64,
            overload_migrates=True,
            overload_min_waiting=4,
            overload_kv_fraction=0.7,
        )
        return p.overload_pressed(replace(source(10), **changes), config)

    def test_load_alone_is_not_overload(self):
        self.assertFalse(self.pressed(waiting=3, free_blocks=1000, total_blocks=2325))

    def test_a_queue_is_overload(self):
        self.assertTrue(self.pressed(waiting=4, free_blocks=2000, total_blocks=2325))

    def test_filling_up_is_overload(self):
        self.assertTrue(self.pressed(waiting=0, free_blocks=697, total_blocks=2325))
        self.assertFalse(self.pressed(waiting=0, free_blocks=698, total_blocks=2325))

    def test_without_the_total_only_the_queue_counts(self):
        self.assertFalse(self.pressed(waiting=0, free_blocks=1, total_blocks=None))

    def test_overload_keeps_a_kv_reserve_on_the_target(self):
        # 66 blocks now: 1171 + 66 needed, unallocated prompts counted.
        self.assertEqual(
            self.overload_on_target(free_blocks=1237).action, "START_SHADOW"
        )
        result = self.overload_on_target(free_blocks=1236)
        self.assertIn(
            "target KV reserve too small", result.candidates[0]["rejections"]
        )
        result = self.overload_on_target(
            free_blocks=1237, unallocated_prompt_blocks=1
        )
        self.assertIn(
            "target KV reserve too small", result.candidates[0]["rejections"]
        )

    def test_overload_does_not_fill_running_slots_with_a_queue(self):
        target = reading("tp4", now=13)
        workload = dict(target.workload or {}, requests=17)
        result = self.overload_on_target(
            free_blocks=9000, waiting=2, workload=workload
        )
        self.assertIn(
            "target running slots full with a queue",
            result.candidates[0]["rejections"],
        )

    def test_a_source_out_of_kv_only_needs_physical_room(self):
        policy = p.PressurePolicy(
            cfg(
                routing_relief=True,
                kv_routing_priority=True,
                kv_short_exempt_blocks=64,
                overload_migrates=True,
                migration_current_length=True,
            ),
            p.ReservationLedger(),
        )
        result = policy.decide(
            10,
            source(10, free_blocks=50),
            reading("tp4", now=10, free_blocks=100),
            [candidate(sampled_at=10, published_at=10)],
        )
        self.assertNotIn(
            "target KV reserve too small",
            result.candidates[0]["rejections"],
        )
        self.assertEqual(result.candidates[0]["target_required_blocks"], 66)

    def test_current_length_off_leaves_the_policy_identity_alone(self):
        self.assertEqual(cfg().sha256, cfg(migration_current_length=False).sha256)
        self.assertNotEqual(cfg().sha256, cfg(migration_current_length=True).sha256)

    def test_overload_keeps_short_requests_in_place(self):
        policy = self.overload_policy()
        short = dict(prompt_tokens=512, max_output_tokens=128)
        for now in range(10, 15):
            result = policy.decide(
                now,
                source(now, waiting=50, decode_ratio=2),
                reading("tp4", now=now),
                [candidate(sampled_at=now, published_at=now, **short)],
            )
        self.assertEqual(result.action, "STAY")
        self.assertIn(
            "short request stays under overload relief",
            result.candidates[0]["rejections"],
        )

    def test_overload_off_leaves_the_policy_identity_alone(self):
        self.assertEqual(
            cfg(overload_sustain_s=99).sha256, cfg(overload_sustain_s=10).sha256
        )
        with self.assertRaises(ValueError):
            cfg(overload_migrates=True)

    def test_fresh_timestamp_and_missing_prefill_counter_fail_closed(self):
        for changes in (dict(prefill_generation=None), dict(sampled_at=10)):
            co, _ = coordinator()
            for now in range(10, 15):
                self.assertEqual(tick(co, now, **changes).action, "STAY")

    def test_unpublished_source_request_blocks_confirmation_until_accounted(self):
        co, _ = coordinator()
        co.ledger.track_ingress("pending", "old", "tp1", 2048)
        for now in range(10, 15):
            self.assertEqual(tick(co, now).action, "STAY")
        co.ledger.retire_admission("pending", "COMPLETED", evidence=True)
        for now in range(15, 19):
            result = tick(co, now)
        self.assertEqual(result.action, "START_SHADOW")

    def test_waiting_prefill_and_backlog_cannot_count_as_failed_intervention(self):
        for changes in (
            dict(waiting=1),
            dict(pending_prompt_tokens=128),
            dict(prefill_batch_tokens=1),
        ):
            co, _ = coordinator()
            for now in range(10, 15):
                self.assertEqual(tick(co, now, **changes).action, "STAY")

    def test_short_prefill_between_observations_restarts_window(self):
        co, _ = coordinator()
        for now in (10, 11, 12, 13):
            self.assertEqual(tick(co, now).action, "STAY")
        for now in (14, 15, 16, 17):
            result = tick(co, now, prefill_generation=1)
            self.assertEqual(result.action, "START_SHADOW" if now == 17 else "STAY")

    def test_relief_routes_even_short_prompts_to_safe_target_and_is_replayable(self):
        co, rows = coordinator()
        result = co.admission(
            "r", 64, 1024, (source(10), reading("tp4")), now=10, key="r"
        )
        self.assertEqual(result.pool_id, "tp4")
        self.assertTrue(result.routing_reason.startswith("router relief epoch"))
        row = rows[-1]
        replay = p.admit(
            "r",
            64,
            1024,
            (source(10), reading("tp4")),
            co.policy.cfg,
            p.ReservationLedger(),
            10,
            "r",
            routing_relief=row["routing_relief"],
        )
        self.assertEqual(replay, result)
        self.assertEqual(co.policy.relief.redirected_prompt_tokens, 64)

    def test_target_not_safe_and_fixed_routes_keep_admitting_and_reset_window(self):
        for target, fixed in (
            (reading("tp4", load=2), None),
            (reading("tp4"), "tp1"),
            (reading("tp4", context_limit=1024), None),
            (reading("tp4", free_blocks=1), None),
        ):
            co, _ = coordinator()
            for now in (10, 11, 12):
                tick(co, now)
            result = co.admission(
                "r",
                2048,
                1024,
                (
                    source(12.5),
                    replace(
                        target,
                        sampled_at=12.5,
                        published_at=12.5,
                        decode_sampled_at=12.5,
                        workload=target.workload | dict(sampled_at=12.5),
                    ),
                ),
                now=12.5,
                key="r",
                allowed_pool_id=fixed,
            )
            self.assertEqual(result.pool_id, "tp1")
            self.assertEqual(co.policy.relief.last_source_admission_at, 12.5)
            self.assertIsNone(co.policy._service_confirmation)
            self.assertEqual(tick(co, 13).action, "STAY")

    def test_relief_stays_after_migration_until_both_service_and_KV_recover(self):
        co, _ = coordinator()
        tick(co, 10)
        co.policy.note_started(11)
        self.assertTrue(co.policy.relief.active)
        for now in (12, 13, 14):
            co.tick(
                now,
                replace(reading(now=now), prefill_generation=0),
                reading("tp4", now=now),
                [],
            )
            self.assertTrue(co.policy.relief.active)
        co.tick(
            15,
            replace(reading(now=15), prefill_generation=0),
            reading("tp4", now=15),
            [],
        )
        self.assertFalse(co.policy.relief.active)
        co, _ = coordinator()
        tick(co, 10)
        for now in range(11, 18):
            co.tick(
                now,
                replace(reading(now=now, free_blocks=100), prefill_generation=0),
                reading("tp4", now=now),
                [],
            )
        self.assertTrue(co.policy.relief.active)

    def test_low_workload_quality_alone_no_longer_triggers_under_relief_gate(self):
        co, _ = coordinator()
        for now in range(10, 20):
            value = replace(source(now), workload=source(now).workload | dict(load=0.7))
            self.assertEqual(
                co.tick(
                    now,
                    value,
                    reading("tp4", now=now),
                    [candidate(sampled_at=now, published_at=now)],
                ).action,
                "STAY",
            )

    def test_recovery_does_not_accumulate_across_missing_observations(self):
        co, _ = coordinator()
        tick(co, 10)
        for now in (11, 12, 20):
            co.tick(
                now,
                replace(reading(now=now), prefill_generation=0),
                reading("tp4", now=now),
                [],
            )
            self.assertTrue(co.policy.relief.active)
        for now in (21, 22, 23, 24):
            co.tick(
                now,
                replace(reading(now=now), prefill_generation=0),
                reading("tp4", now=now),
                [],
            )
        self.assertFalse(co.policy.relief.active)

    def test_KV_emergency_bypasses_missing_ack_counter_and_prefill_drain(self):
        co, _ = coordinator()
        value = source(
            10,
            free_blocks=100,
            growth_blocks_s=10,
            prefill_generation=None,
            pending_prompt_tokens=128,
        )
        result = co.tick(10, value, reading("tp4"), [candidate()])
        self.assertTrue(result.capacity_path)
        self.assertEqual(result.action, "START_SHADOW")

    def test_every_scheduled_prefill_updates_scalar_before_sampling_throttle(self):
        from tests.bridge_tp.test_pressure_observation import inv

        publisher = object.__new__(inv.InventoryPublisher)
        publisher.prefill_generation = 0
        publisher.last_batch_sample, publisher.interval_s = 10, 1
        requests = dict(r=NS(num_computed_tokens=0, num_prompt_tokens=2048))
        output = NS(num_scheduled_tokens=dict(r=1))
        with patch.object(inv.time, "time", return_value=10.01):
            publisher.record_batch(requests, output)
        self.assertEqual(publisher.prefill_generation, 1)
        raw = dict(schema="PRESSURE_INVENTORY_V1", prefill_generation=1, requests=[])
        compact = load("pressure_inventory_process").compact_inventory(raw)
        self.assertEqual(compact["prefill_generation"], 1)
        self.assertEqual(json.loads(json.dumps(compact))["prefill_generation"], 1)
