# SPDX-License-Identifier: Apache-2.0
"""CPU support boundaries and pressure/persistence ablation contracts."""

import json
import unittest
from dataclasses import replace

from tools.bridge_tp import prefill_cost_reference as r
from tools.bridge_tp import replay_pressure_calibration as replay


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = r.p.PressureConfig(
            **json.loads(
                (
                    replay.ROOT / "experiments/phase9/pressure_balanced_v2/policy.json"
                ).read_text()
            )
        )
        self.pool = r.p.PoolReading(
            pool_id="tp1",
            sampled_at=10,
            published_at=10,
            free_blocks=1000,
            block_size=16,
            context_limit=8192,
            decode_ratio=0.3,
            decode_samples=30,
            decode_sampled_at=10,
            idle=False,
            pending_prompt_tokens=512,
            waiting=1,
            growth_blocks_s=0,
            prefill_batch_tokens=512,
        )

    def batch(self, context, event, latency):
        return dict(
            pool="tp1",
            run_id="train-r1",
            prefill_tokens=512,
            recompute_tokens=0,
            decode_requests=3,
            decode_context_sum=context * 3,
            completion_latency_ms=latency,
            requests=[dict(request_id=event, computed_before=0, prefill_tokens=512)],
        )

    def test_no_extrapolation_between_nd_tokens_or_context(self):
        model = r.PrefillReference.fit(
            [
                self.batch(2200, "a", 90),
                self.batch(2600, "b", 100),
            ]
        )
        query = self.batch(2400, "query", 999)
        estimate = model.estimate("tp1", query)
        self.assertAlmostEqual(estimate["engine_s"], 0.0995)
        self.assertIsNone(model.estimate("tp4", query))
        self.assertIsNone(model.estimate("tp1", dict(query, prefill_tokens=256)))
        self.assertIsNone(model.estimate("tp1", dict(query, decode_requests=4)))
        self.assertIsNone(model.estimate("tp1", self.batch(2800, "x", 0)))

    def test_one_event_is_not_repeated_evidence(self):
        row = self.batch(2200, "same", 90)
        model = r.PrefillReference.fit([row, row])
        self.assertIsNone(model.estimate("tp1", row))

    def test_engine_labels_are_not_query_features(self):
        rows = [self.batch(2200, "a", 90), self.batch(2600, "b", 100)]
        model = r.PrefillReference.fit(rows)
        a = model.estimate("tp1", self.batch(2400, "x", 1))
        b = model.estimate("tp1", self.batch(2400, "x", 999999))
        self.assertEqual(a, b)

    def test_same_tokens_can_be_busy_by_observed_cost(self):
        self.assertEqual(r.baseline_classify(self.pool, self.cfg, 10).zone, "SPARE")
        self.assertEqual(
            r.cost_pressure(self.pool, self.cfg, 10, {"engine_s": 0.2}).zone, "OVERLOAD"
        )

    def test_unknown_cost_cannot_be_spare_or_revive_stale_inventory(self):
        pressure = r.cost_pressure(self.pool, self.cfg, 10, None)
        self.assertEqual(pressure.zone, "UNKNOWN")
        self.assertTrue(pressure.missing)
        stale = r.cost_pressure(self.pool, self.cfg, 20, {"engine_s": 0.2})
        self.assertEqual(stale.zone, "UNKNOWN")
        self.assertIn("pool timestamp", stale.missing)

    def test_known_no_prefill_preserves_decode_pressure(self):
        pool = replace(
            self.pool,
            prefill_batch_tokens=0,
            pending_prompt_tokens=0,
            waiting=0,
            decode_ratio=2,
        )
        self.assertEqual(
            r.cost_pressure(pool, self.cfg, 10, None),
            r.baseline_classify(pool, self.cfg, 10),
        )

    def test_gate_requires_continuity_and_never_fabricates_start(self):
        cfg = replace(self.cfg, sustain_s=1)
        gate = r.ReliefGate(cfg)
        source = r.p.Pressure(2, 0, 0.1, "OVERLOAD", ())
        target = r.p.Pressure(0.3, 0, 0.1, "SPARE", ())
        for timestamp in (0, 0.2, 0.4, 0.6, 0.8):
            self.assertFalse(gate.update(timestamp, source, target))
        self.assertTrue(gate.update(1, source, target))
        self.assertFalse(gate.update(1.2, source, target))
        unknown = r.p.Pressure(None, 0, 0.1, "UNKNOWN", ("decode service",))
        self.assertFalse(gate.update(1.4, source, unknown))
        self.assertFalse(gate.update(1.6, source, target))
        self.assertFalse(gate.update(2.6, source, target))  # tick gap resets

    def test_event_reconstruction_keeps_arrival_token_finish_order(self):
        scene = {
            "tokens": {"a": [10.1, 10.2]},
            "request_results.json": {
                "requests": [
                    dict(
                        id="a",
                        pool="source",
                        max_tokens=2,
                        error=None,
                        submitted_unix_s=10,
                        completed_unix_s=10.3,
                    )
                ]
            },
        }
        events = replay.events(scene)
        self.assertEqual([row[1] for row in events], [0, 1, 1, 2])

    def test_recorded_heartbeat_is_not_available_before_capture(self):
        scene = {
            "tokens": {},
            "request_results.json": {"requests": []},
            "measurement_window.json": {"started_unix_s": 10, "ended_unix_s": 10.4},
            "inventories": {},
            "observations": [],
        }
        for pool, label in (("tp1", "source"), ("tp4", "target")):
            scene["inventories"][pool] = [
                dict(
                    generation=1,
                    sampled_at=0,
                    pool_id=pool,
                    requests=[],
                    waiting=0,
                    pending_prompt_tokens=0,
                    free_blocks=1000,
                    block_size=16,
                    context_limit=8192,
                )
            ]
            scene["observations"].append(
                dict(
                    pool=label,
                    generation=1,
                    captured_unix_s=10.3,
                    observed_at=10.1,
                    metrics=dict(
                        sampled_unix_s=10.3,
                        num_running=0,
                        num_waiting=0,
                        free_kv_blocks=1000,
                    ),
                )
            )
        scene["observations"].insert(
            0, dict(pool="target", captured_unix_s=10.1, error="scrape unavailable")
        )
        rows = replay.timeline(scene, self.cfg, r.PrefillReference([]), None)
        self.assertIsNone(rows[0][1]["tp4"])
        self.assertEqual([row[0] for row in rows], [10.1, 10.3])
        reading, _ = rows[-1][1]["tp4"]
        self.assertTrue(reading.idle)


if __name__ == "__main__":
    unittest.main()
