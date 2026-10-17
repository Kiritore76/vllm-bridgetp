# SPDX-License-Identifier: Apache-2.0
"""Passive dynamic pressure, post-arrival routing and bounded observation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tests.bridge_tp.test_pressure_policy import candidate, config, p, pool, r
from tools.bridge_tp.pressure_cpu import load

w = load("pressure_workload")
o = load("pressure_observation")
BUNDLE = Path(__file__).resolve().parents[2] / "experiments/phase9/pressure_balanced_v2"


def profiles():
    return w.build_profiles(
        json.loads((BUNDLE / "health_reference_v2.json").read_text()),
        json.loads((BUNDLE / "service_evidence.json").read_text()),
    )


def cfg(**kwargs):
    return config(
        health_service=True,
        workload_pressure=True,
        sustain_s=3,
        capacity_lookahead_s=16.86410117149353,
        **kwargs,
    )


def reading(name="tp1", now=10, load=0.2, **kwargs):
    profile = profiles()[name]
    return pool(
        name,
        sampled_at=now,
        published_at=now,
        decode_sampled_at=now,
        health_status="UNKNOWN",
        unallocated_prompt_blocks=0,
        workload=dict(
            status="OBSERVED",
            reason="fresh passive native workload",
            sampled_at=now,
            load=load,
            prefill_s=0,
            requests=1,
            profile=profile,
        ),
        **kwargs,
    )


def degraded(now, load=1.25, count=None, slow=None):
    return reading(
        now=now,
        load=load,
        decode_ratio=2,
        service_evidence=dict(
            interval_count=100 + int((now - 10) * 100) if count is None else count,
            slow_interval_count=10 + int((now - 10) * 2) if slow is None else slow,
            window_s=3,
            slow_budget=0.01,
        ),
    )


class WorkloadTests(unittest.TestCase):
    def test_reference_diagnostic_has_reason_and_is_cached_only_when_opted_in(self):
        h = load("pressure_health")
        tokens = Mock()
        tokens.score.return_value = dict(goodoutput_s=40)
        reference = json.loads((BUNDLE / "health_reference_v2.json").read_text())
        tracker = h.HealthTracker(reference, 0, tokens, cache_closed_windows=True)
        self.assertIn("not ready", tracker.snapshot("tp1", 1)["health_reason"])
        for index in range(21):
            now = index * 0.5
            tracker.observe(
                dict(
                    pool_id="tp1",
                    generation=index,
                    sampled_at=now,
                    last_scheduled_batch=dict(
                        sampled_at=now,
                        decode_requests=1,
                        decode_context_sum=3000,
                        prefill_tokens=0,
                        recompute_tokens=0,
                    ),
                ),
                now,
            )
        tracker.snapshot("tp1", 10)
        tracker.snapshot("tp1", 10.5)
        self.assertEqual(tokens.score.call_count, 1)

    def test_projected_pressure_includes_pending_migration_demand(self):
        target = reading("tp4")
        self.assertGreater(
            w.projected(target, 2048, cfg(), 10, 0.7),
            w.projected(target, 2048, cfg(), 10),
        )

    def test_profile_uses_existing_repeated_points_and_excludes_tiny_tail_scaling(self):
        values = profiles()
        self.assertEqual(values["tp1"]["capacities"], [[1, 8], [2, 6], [3, 3]])
        self.assertEqual(values["tp4"]["capacities"], [[1, 16], [2, 24], [3, 16]])
        self.assertLess(values["tp1"]["prefill_s_per_token"] * 2048, 1)
        self.assertLess(values["tp4"]["prefill_s_per_token"] * 2048, 2)

    def test_slot_cost_interpolates_only_in_context_support(self):
        profile = profiles()["tp1"]
        self.assertAlmostEqual(w.slot_cost(profile, 3072)[0], 1 / 8)
        self.assertAlmostEqual(w.slot_cost(profile, 4096)[0], (1 / 8 + 1 / 6) / 2)
        for context in (1000, 20000):
            cost, basis = w.slot_cost(profile, context)
            self.assertAlmostEqual(cost, 1 / 3)
            self.assertIn("fallback", basis)

    def test_changing_native_shapes_remain_observed_and_pending_prefill_counted(self):
        for nd in (1, 5, 7, 3):
            raw = dict(
                sampled_at=10,
                requests=[
                    dict(prompt_tokens=3072, computed_tokens=1024) for _ in range(nd)
                ],
            )
            value = w.observe(profiles()["tp1"], raw, 10)
            self.assertEqual(value["status"], "OBSERVED")
            self.assertAlmostEqual(value["load"], nd / 8)
            self.assertGreater(value["prefill_s"], 0)
            self.assertEqual(w.observe(profiles()["tp1"], raw, 13)["status"], "UNKNOWN")

    def test_service_migration_does_not_require_eta_or_exact_postmove_health_cell(self):
        policy = p.PressurePolicy(cfg(), p.ReservationLedger())
        for now in (10, 11, 12, 13):
            result = policy.decide(
                now,
                degraded(now),
                reading("tp4", now=now),
                [candidate(sampled_at=now, published_at=now)],
            )
        self.assertEqual(result.action, "START_SHADOW")
        self.assertFalse(result.capacity_path)
        self.assertFalse(
            result.candidates[0]["request"]["target_health_after_supported"]
        )

    def test_healthy_or_busy_or_stale_workloads_do_not_trigger_service_move(self):
        for source, target in (
            (reading(), reading("tp4")),
            (reading(load=1.25), reading("tp4", load=0.95)),
            (reading(now=7, load=1.25), reading("tp4")),
        ):
            result = p.PressurePolicy(cfg(), p.ReservationLedger()).decide(
                10, source, target, [candidate()]
            )
            self.assertEqual(result.action, "STAY")

    def test_postmove_target_workload_budget_remains_required(self):
        policy = p.PressurePolicy(cfg(), p.ReservationLedger())
        for now in (10, 11, 12, 13):
            result = policy.decide(
                now,
                degraded(now),
                reading("tp4", now=now),
                [candidate(sampled_at=now, published_at=now, target_decode_cost=0.9)],
            )
        self.assertEqual(result.action, "STAY")

    def test_high_workload_without_service_degradation_does_not_migrate(self):
        policy = p.PressurePolicy(cfg(), p.ReservationLedger())
        for now in range(10, 18):
            result = policy.decide(
                now,
                reading(now=now, load=1.5),
                reading("tp4", now=now),
                [candidate(sampled_at=now, published_at=now)],
            )
            self.assertEqual(result.action, "STAY")

    def test_old_bad_samples_and_active_gap_cannot_confirm_new_degradation(self):
        for slow in (10, 11):
            policy = p.PressurePolicy(cfg(), p.ReservationLedger())
            for now in range(10, 18):
                result = policy.decide(
                    now,
                    degraded(
                        now,
                        count=100 + (now - 10) * 100,
                        slow=10 if now == 10 else slow,
                    ),
                    reading("tp4", now=now),
                    [candidate(sampled_at=now, published_at=now)],
                )
                self.assertEqual(result.action, "STAY")

    def test_low_workload_quality_requires_two_independent_windows(self):
        policy = p.PressurePolicy(cfg(), p.ReservationLedger())
        for now in range(10, 17):
            result = policy.decide(
                now,
                degraded(now, load=0.7),
                reading("tp4", now=now),
                [candidate(sampled_at=now, published_at=now)],
            )
            self.assertEqual(result.action, "START_SHADOW" if now == 16 else "STAY")

    def test_missing_or_reset_counters_and_observation_gaps_restart_confirmation(self):
        for bad in (
            None,
            dict(interval_count=1, slow_interval_count=0, window_s=3, slow_budget=0.01),
        ):
            policy = p.PressurePolicy(cfg(), p.ReservationLedger())
            for now in (10, 11, 12):
                policy.decide(now, degraded(now), reading("tp4", now=now), [])
            from dataclasses import replace

            result = policy.decide(
                13,
                replace(degraded(13), service_evidence=bad),
                reading("tp4", now=13),
                [candidate(sampled_at=13, published_at=13)],
            )
            self.assertEqual(result.action, "STAY")
        policy = p.PressurePolicy(cfg(), p.ReservationLedger())
        policy.decide(10, degraded(10), reading("tp4", now=10), [])
        result = policy.decide(
            14,
            degraded(14),
            reading("tp4", now=14),
            [candidate(sampled_at=14, published_at=14)],
        )
        self.assertEqual(result.action, "STAY")

    def test_low_workload_bad_windows_must_be_consecutive_and_disjoint(self):
        policy = p.PressurePolicy(cfg(), p.ReservationLedger())
        for now in range(10, 23):
            # Bad [10,13], healthy [13,16], then bad [16,19] and [19,22].
            elapsed_bad = min(now - 10, 3) + max(0, now - 16)
            result = policy.decide(
                now,
                degraded(now, load=0.7, slow=10 + 2 * elapsed_bad),
                reading("tp4", now=now),
                [candidate(sampled_at=now, published_at=now)],
            )
            self.assertEqual(result.action, "START_SHADOW" if now == 22 else "STAY")

    def test_KV_emergency_does_not_wait_for_quality_or_new_sample_confirmation(self):
        result = p.PressurePolicy(cfg(), p.ReservationLedger()).decide(
            10,
            reading(free_blocks=100, growth_blocks_s=10),
            reading("tp4"),
            [candidate()],
        )
        self.assertTrue(result.capacity_path)
        self.assertEqual(result.action, "START_SHADOW")

    def test_native_service_counts_only_new_completed_intervals(self):
        service = o.NativeServiceWindow()
        service.arrive("r", "tp1", 10)
        service.token("r", 10)
        service.token("r", 10.2)
        service.token("r", 10.25)
        first = service.snapshot("tp1", 10.3)
        later = service.snapshot("tp1", 14)
        self.assertEqual(first["interval_count"], 2)
        self.assertEqual(first["slow_interval_count"], 1)
        self.assertEqual(later["interval_count"], 2)
        self.assertEqual(later["slow_interval_count"], 1)
        self.assertEqual(later["samples"], 0)

    def test_same_prompt_routes_by_projected_pressure_and_KV_risk(self):
        for source, expected in (
            (reading(load=0.1), "tp1"),
            (reading(load=0.9), "tp4"),
            (reading(free_blocks=250, growth_blocks_s=10), "tp4"),
        ):
            result = p.admit(
                "r",
                2048,
                1024,
                (source, reading("tp4")),
                cfg(),
                p.ReservationLedger(),
                10,
                "k",
            )
            self.assertEqual(result.pool_id, expected)
            self.assertIsNotNone(result.routing_pressure)
            self.assertEqual(result.reservation_basis, "none")

    def test_missing_all_telemetry_uses_normalized_inflight_not_always_TP1(self):
        coordinator = r.PressureCoordinator(
            cfg(), lambda _: None, routing_weights=(1, 2)
        )
        counts = dict(tp1=0, tp4=0)
        for index in range(6):
            result = coordinator.admission(
                str(index), 1, 10, (pool(), pool("tp4")), now=10, key=str(index)
            )
            counts[result.pool_id] += 1
        self.assertEqual(counts, dict(tp1=2, tp4=4))

    def test_unpublished_work_is_in_projected_pressure_without_output_reservation(self):
        coordinator = r.PressureCoordinator(cfg(), lambda _: None)
        result = [
            coordinator.admission(
                str(i),
                2048,
                4096,
                (reading(load=0.1), reading("tp4")),
                now=10,
                key=str(i),
            ).pool_id
            for i in range(3)
        ]
        self.assertEqual(result, ["tp1", "tp1", "tp4"])
        self.assertTrue(all(v.blocks == 0 for v in coordinator.ledger.entries.values()))

    def test_workload_cancel_needs_source_quality_and_KV_recovery_for_three_seconds(
        self,
    ):
        c = load("pressure_cancel").ShadowCancellation()
        setting = cfg(shadow_cancel=True)
        for now in (10, 11, 12):
            self.assertIsNone(
                c.decide(
                    now,
                    reading(now=now),
                    reading("tp4", now=now),
                    candidate(),
                    setting,
                    reversible=True,
                )
            )
        self.assertIsNotNone(
            c.decide(
                13,
                reading(now=13),
                reading("tp4", now=13),
                candidate(),
                setting,
                reversible=True,
            )
        )
        for changes in (dict(free_blocks=100), dict(decode_ratio=2), dict(load=0.9)):
            current = load("pressure_cancel").ShadowCancellation()
            for now in (10, 11, 12, 13, 14):
                self.assertIsNone(
                    current.decide(
                        now,
                        reading(now=now, **changes),
                        reading("tp4", now=now),
                        candidate(),
                        setting,
                        reversible=True,
                    )
                )

    def test_inventory_budget_does_not_skip_records_or_accept_future_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.jsonl"
            rows = [
                dict(
                    schema="PRESSURE_INVENTORY_V1",
                    pool_id="tp1",
                    generation=i,
                    sampled_at=i,
                )
                for i in (1, 2, 3)
            ]
            path.write_text("".join(json.dumps(v) + "\n" for v in rows))
            reader = o.InventoryReader(path, "tp1", max_records=1)
            self.assertEqual(reader.poll(2)["generation"], 1)
            self.assertGreater(reader.diagnostics()["backlog_bytes"], 0)
            self.assertEqual(reader.poll(2)["generation"], 2)
            self.assertEqual(reader.poll(2)["generation"], 2)
            self.assertEqual(reader.deferred["generation"], 3)
            self.assertEqual(reader.poll(3)["generation"], 3)
            self.assertEqual(reader.backlog_bytes, 0)


if __name__ == "__main__":
    unittest.main()
