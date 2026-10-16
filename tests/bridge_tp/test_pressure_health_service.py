# SPDX-License-Identifier: Apache-2.0
"""Health mode is optional, causal and independent from physical KV relief."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.bridge_tp.test_pressure_policy import candidate, config, p, pool
from tools.bridge_tp import run_pressure_balanced_pilot as worker
from tools.bridge_tp import run_pressure_comparison as comparison
from tools.bridge_tp.pressure_cpu import load

h = load("pressure_health")
o = load("pressure_observation")
BUNDLE = Path(__file__).resolve().parents[2] / "experiments/phase9/pressure_balanced_v2"


def reference():
    return json.loads((BUNDLE / "health_reference_v2.json").read_text())


class HealthPolicyTests(unittest.TestCase):
    def cfg(self):
        return config(
            health_service=True,
            health_min_efficiency=0.9,
            health_target_load=0.8,
            sustain_s=3,
            capacity_lookahead_s=16.86410117149353,
        )

    def reading(self, name="tp1", now=10, **changes):
        return pool(
            name,
            sampled_at=now,
            published_at=now,
            decode_sampled_at=now,
            health_status=changes.pop("health_status", "SUPPORTED"),
            health_sampled_at=now,
            health_eta=changes.pop("health_eta", 0.7 if name == "tp1" else 1),
            health_load=changes.pop("health_load", 0.7 if name == "tp1" else 0.2),
            unallocated_prompt_blocks=0,
            **changes,
        )

    def evaluate(self, changes=None, target_changes=None, cand_changes=None):
        policy = p.PressurePolicy(self.cfg(), p.ReservationLedger())
        result = None
        for now in (10 + index * 0.2 for index in range(51)):
            result = policy.decide(
                now,
                self.reading(now=now, **(changes or {})),
                self.reading("tp4", now=now, **(target_changes or {})),
                [
                    candidate(
                        sampled_at=now,
                        published_at=now,
                        target_health_after_supported=(cand_changes or {}).get(
                            "target_health_after_supported", True
                        ),
                    )
                ],
            )
        return result

    def test_service_loss_with_ample_KV_starts_after_persistence(self):
        result = self.evaluate()
        self.assertEqual(result.action, "START_SHADOW")
        self.assertFalse(result.capacity_path)
        self.assertEqual(result.service_path, "health")
        self.assertEqual(result.source.zone, "SPARE")

    def test_service_waits_three_seconds_and_restarts_confirmation_after_start(self):
        policy = p.PressurePolicy(self.cfg(), p.ReservationLedger())

        def decide(now):
            return policy.decide(
                now,
                self.reading(now=now),
                self.reading("tp4", now=now),
                [
                    candidate(
                        sampled_at=now,
                        published_at=now,
                        target_health_after_supported=True,
                    )
                ],
            )

        for index in range(15):
            self.assertEqual(decide(10 + index * 0.2).action, "STAY")
        self.assertEqual(decide(13).action, "START_SHADOW")
        policy.note_started(13)
        for index in range(15):
            self.assertEqual(decide(13.2 + index * 0.2).action, "STAY")
        self.assertEqual(decide(16.3).action, "START_SHADOW")

    def test_KV_warning_bypasses_confirmation_and_cooldown_but_not_channel(self):
        ledger = p.ReservationLedger()
        policy = p.PressurePolicy(self.cfg(), ledger)
        policy.note_started(9.9)
        source = self.reading(
            free_blocks=250, growth_blocks_s=10, health_status="UNKNOWN"
        )
        target = self.reading("tp4", health_status="UNKNOWN", decode_ratio=0.9)
        request = candidate(estimated_release_s=self.cfg().capacity_lookahead_s)
        result = policy.decide(10, source, target, [request])
        self.assertEqual(result.action, "START_SHADOW")
        self.assertTrue(result.capacity_path)
        ledger.reserve(
            p.Reservation("busy", "another", "tp4", 10, 0.1, 0, "migration"), target
        )
        result = policy.decide(10.1, source, target, [request])
        self.assertEqual(result.reason, "migration channel occupied")
        self.assertEqual(result.action, "STAY")

    def test_closed_window_persistence_tolerates_subsecond_poll_jitter(self):
        policy = p.PressurePolicy(self.cfg(), p.ReservationLedger())
        for now in (10 + index * 0.7 for index in range(16)):
            result = policy.decide(
                now,
                self.reading(now=now),
                self.reading("tp4", now=now),
                [
                    candidate(
                        sampled_at=now,
                        published_at=now,
                        target_health_after_supported=True,
                    )
                ],
            )
        self.assertEqual(result.action, "START_SHADOW")
        # A real observation gap beyond two seconds resets persistence.
        now += 3
        result = policy.decide(
            now,
            self.reading(now=now),
            self.reading("tp4", now=now),
            [
                candidate(
                    sampled_at=now, published_at=now, target_health_after_supported=True
                )
            ],
        )
        self.assertEqual(result.action, "STAY")

    def test_healthy_both_busy_unknown_or_unsupported_target_stay(self):
        self.assertEqual(self.evaluate(changes=dict(health_eta=1)).action, "STAY")
        for changes in (
            dict(health_load=0.95),
            dict(health_eta=0.7),
            dict(health_status="UNKNOWN"),
            dict(waiting=1),
        ):
            self.assertEqual(self.evaluate(target_changes=changes).action, "STAY")
        self.assertEqual(
            self.evaluate(
                cand_changes=dict(target_health_after_supported=False)
            ).action,
            "STAY",
        )

    def test_KV_path_does_not_require_health_lookup(self):
        cfg = self.cfg()
        policy = p.PressurePolicy(cfg, p.ReservationLedger())
        result = policy.decide(
            10,
            self.reading(free_blocks=250, growth_blocks_s=10, health_status="UNKNOWN"),
            self.reading("tp4", health_status="UNKNOWN"),
            [candidate(estimated_release_s=cfg.capacity_lookahead_s)],
        )
        self.assertEqual(result.action, "START_SHADOW")
        self.assertTrue(result.capacity_path)

    def test_source_recent_prefill_and_stale_health_do_not_trigger_service(self):
        self.assertEqual(
            self.evaluate(
                changes=dict(pending_prompt_tokens=4096, health_eta=1)
            ).action,
            "STAY",
        )
        reading = self.reading()
        self.assertFalse(
            p.health_valid(replace(reading, health_sampled_at=-10), self.cfg(), 10)
        )

    def test_routing_avoids_busy_target_without_admission_gate(self):
        cfg = self.cfg()
        for target, expected in (
            (self.reading("tp4"), "tp4"),
            (self.reading("tp4", health_load=1), "tp1"),
        ):
            ledger = p.ReservationLedger()
            result = p.admit(
                "x", 2048, 1024, (self.reading(), target), cfg, ledger, 10, "k"
            )
            self.assertEqual(result.pool_id, expected)
            self.assertEqual(result.reservation_basis, "none")
            if expected == "tp4":
                second = p.admit(
                    "y", 2048, 1024, (self.reading(), target), cfg, ledger, 10, "k2"
                )
                self.assertEqual(second.pool_id, "tp1")
        source = self.reading(free_blocks=0, health_eta=1)
        target = self.reading("tp4", free_blocks=0, health_load=1)
        self.assertEqual(
            p.admit(
                "x", 1024, 1024, (source, target), cfg, p.ReservationLedger(), 10, "k"
            ).pool_id,
            "tp1",
        )


class HealthObservationTests(unittest.TestCase):
    def test_real_SSE_mirror_includes_active_gap_and_excludes_TTFT(self):
        tokens = h.TokenWindows()
        service = o.NativeServiceWindow(health_observer=tokens)
        service.arrive("a", "tp1", 0)
        for index in range(100):
            service.token("a", 5 + index * 0.02)
        score = tokens.score("tp1", 0, 10)
        self.assertAlmostEqual(score["raw_s"], 10)
        self.assertAlmostEqual(score["goodoutput_s"], 10 / (10 - 6.98))
        service.finish("a")
        self.assertEqual(tokens.score("tp1", 0, 10)["goodoutput_s"], 10)

    def raw(self, generation, now, nd=1, prefill=0):
        return dict(
            pool_id="tp1",
            generation=generation,
            sampled_at=now,
            waiting=0,
            pending_prompt_tokens=0,
            last_scheduled_batch=dict(
                sampled_at=now,
                decode_requests=nd,
                decode_context_sum=nd * 3000,
                prefill_tokens=prefill,
                recompute_tokens=0,
            ),
        )

    def test_closed_window_checks_shape_and_gaps_and_marks_mixed_prefill(self):
        tokens = h.TokenWindows()
        tokens.arrive("a", "tp1", 0)
        for index in range(400):
            tokens.token("a", 0.01 + index * 0.025)
        tracker = h.HealthTracker(reference(), 0, tokens)
        for index in range(21):
            tracker.observe(self.raw(index, index * 0.5), index * 0.5)
        self.assertEqual(tracker.snapshot("tp1", 9)["health_status"], "UNKNOWN")
        value = tracker.snapshot("tp1", 10)
        self.assertEqual(value["health_status"], "SUPPORTED")
        self.assertGreater(value["health_eta"], 1)
        tracker.observe(self.raw(22, 10.1, nd=2), 10.1)
        self.assertEqual(tracker.snapshot("tp1", 10.1)["health_status"], "UNKNOWN")
        self.assertEqual(
            tracker.snapshot("tp4", 10, idle=True)["health_status"], "IDLE"
        )
        for missing, prefill in ((True, 0), (False, 2048)):
            tracker = h.HealthTracker(reference(), 0, tokens)
            for index in range(21):
                if missing and 5 <= index <= 10:
                    continue
                tracker.observe(
                    self.raw(index, index * 0.5, prefill=prefill if index == 10 else 0),
                    index * 0.5,
                )
            value = tracker.snapshot("tp1", 10)
            self.assertEqual(
                value["health_status"], "UNKNOWN" if missing else "SUPPORTED"
            )
            if not missing:
                self.assertTrue(value["health_mixed_window"])

    def test_exact_lookup_and_diagnostic_quarantine(self):
        ref = reference()
        self.assertIsNotNone(h.lookup(ref, "source", 8, 3000))
        self.assertIsNone(h.lookup(ref, "source", 5, 3000))
        self.assertIsNone(h.lookup(ref, "target", 32, 5000))
        self.assertFalse(h.supported_after(ref, 25, 7000))
        self.assertTrue(h.supported_after(ref, 2, 5000))

    def test_Nd2_supplement_is_exact_and_keeps_unknown_shapes_unknown(self):
        ref = reference()
        expected = {
            ("source", 1): 75.75,
            ("source", 2): 74.2,
            ("target", 1): 125.7,
            ("target", 2): 124.05,
            ("target", 3): 124.9,
        }
        self.assertEqual(len(ref["cells"]), 27)
        self.assertEqual(
            sum(c["health_status"] == "REFERENCE" for c in ref["cells"]), 25
        )
        for (pool_name, cb), good in expected.items():
            cell = h.lookup(ref, pool_name, 2, cb * 2048)
            self.assertEqual(cell["runs"], 2)
            self.assertGreaterEqual(cell["windows"], 4)
            self.assertEqual(cell["window_goodoutput"]["median"], good)
            self.assertAlmostEqual(
                h.efficiency(ref, pool_name, 2, cb * 2048, good)["eta"], 1
            )
        self.assertIsNone(h.lookup(ref, "source", 2, 6144))
        self.assertIsNone(h.lookup(ref, "target", 2, 2047))
        self.assertIsNone(h.lookup(ref, "target", 32, 5000))
        self.assertFalse(ref["deployment_ready"])

    def test_Nd2_shape_transition_waits_for_its_own_closed_window(self):
        tokens = h.TokenWindows()
        tracker = h.HealthTracker(reference(), 0, tokens)
        for index in range(21):
            tracker.observe(self.raw(index, index * 0.5), index * 0.5)
        tracker.observe(self.raw(21, 10.1, nd=2), 10.1)
        self.assertEqual(tracker.snapshot("tp1", 10.1)["health_status"], "UNKNOWN")
        for request in ("a", "b"):
            tokens.arrive(request, "tp1", 10)
            for index in range(400):
                tokens.token(request, 10.01 + index * 0.025)
        for index in range(1, 21):
            now = 10 + index * 0.5
            tracker.observe(self.raw(21 + index, now, nd=2), now)
        value = tracker.snapshot("tp1", 20)
        self.assertEqual(value["health_status"], "SUPPORTED")
        self.assertAlmostEqual(value["health_eta"], 80 / 75.75)


class HealthPreflightTests(unittest.TestCase):
    def test_actual_controller_requires_versioned_reference_and_persistence(self):
        inputs = comparison.pilot.read(BUNDLE / "calibration.json")
        recipe = comparison.pilot.read(BUNDLE / "comparison.json")["recipes"][0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "arrivals.json").write_text(
                json.dumps(comparison.arrivals(inputs, recipe))
            )
            policy = comparison.pilot.read(BUNDLE / "health_service_policy.json")
            (root / "policy.json").write_text(json.dumps(policy))
            args = worker.parse_args(
                [
                    "--policy",
                    "pressure-balanced-v1",
                    "--policy-config",
                    str(root / "policy.json"),
                    "--controller-config",
                    str(BUNDLE / "controller_template.json"),
                    "--service-evidence",
                    str(BUNDLE / "service_evidence.json"),
                    "--health-reference",
                    str(BUNDLE / "health_reference_v2.json"),
                    "--arrivals",
                    str(root / "arrivals.json"),
                    "--run-dir",
                    str(root),
                    "--source-inventory",
                    str(root / "source.jsonl"),
                    "--target-inventory",
                    str(root / "target.jsonl"),
                    "--predictor-events",
                    str(root / "predictor.jsonl"),
                    "--predictor-sha256",
                    "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7",
                    "--migration-id",
                    "health-test",
                    "--no-migration",
                ]
            )
            prepared = worker.preflight(args)
            self.assertTrue(prepared[0].health_service)
            self.assertIn(str(args.health_reference), prepared[-1]["input_sha256"])
            args.health_reference = None
            with self.assertRaisesRegex(ValueError, "frozen V2"):
                worker.preflight(args)
            args.health_reference = BUNDLE / "health_reference_v2.json"
            policy["sustain_s"] = 1
            (root / "policy.json").write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError, ">=3s persistence"):
                worker.preflight(args)


if __name__ == "__main__":
    unittest.main()
