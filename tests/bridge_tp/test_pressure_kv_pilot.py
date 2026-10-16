# SPDX-License-Identifier: Apache-2.0
"""KV-only pilot contracts, causal growth and matched four-run inputs."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.bridge_tp import test_pressure_comparison as previous
from tests.bridge_tp.test_pressure_policy import candidate, config, p, pool
from tools.bridge_tp import pressure_health_reference as health
from tools.bridge_tp import pressure_kv_pilot as kv
from tools.bridge_tp import run_pressure_comparison as compare

BUNDLE = Path(__file__).resolve().parents[2] / "experiments/phase9/pressure_balanced_v2"


class KvInputTests(previous.ComparisonTests):
    """Reuse actual controller preflight and sampling checks for the new inputs."""

    def setUp(self):
        super().setUp()
        self.config = compare.pilot.read(BUNDLE / "kv_release.json")

    def test_frozen_inputs_preserve_measured_shapes_and_sampling(self):
        kv.validate(self.config, self.inputs)
        for recipe in self.config["recipes"]:
            jobs = compare.arrivals(self.inputs, recipe)
            self.assertEqual(len(jobs), 6)
            self.assertTrue(all(j["request"]["ignore_eos"] for j in jobs))

    def test_only_sustain_can_be_changed(self):
        self.config["policy_overrides"]["sustain_s"] = 0
        with self.assertRaises(ValueError):
            kv.validate(self.config, self.inputs)

    def test_pair_has_identical_arrival_file_and_counterbalanced_order(self):
        def build(spec, scene, root, *_):
            return dict(
                controller_argv=[],
                source_env={},
                target_env={},
                run_dir=root / scene["name"] / "controller",
            )

        def write(path, data):
            path.write_text(json.dumps(data))

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(compare.pilot, "build_scene", side_effect=build),
        ):
            plans = compare.build_plans(
                dict(policy_config=str(BUNDLE / "policy.json")),
                self.config,
                self.inputs,
                Path(temp),
                SimpleNamespace(write_json=write),
                None,
            )
            self.assertEqual(
                [p[1] for p in plans], ["stay", "migrate", "migrate", "stay"]
            )
            for i in (0, 2):
                self.assertEqual(plans[i][2]["arrivals"], plans[i + 1][2]["arrivals"])
            for _, arm, _, plan in plans:
                self.assertEqual(
                    "--no-migration" in plan["controller_argv"], arm == "stay"
                )
            policy = compare.pilot.read(Path(temp) / "comparison_policy.json")
            self.assertEqual(
                policy,
                compare.pilot.read(BUNDLE / "policy.json")
                | self.config["policy_overrides"],
            )

    def test_runtime_capacity_must_allow_one_offload_to_relieve_pressure(self):
        jobs = compare.arrivals(self.inputs, self.config["recipes"][0])
        rows = [
            dict(block_size=16, free_blocks=1962),
            dict(block_size=16, free_blocks=35738),
        ]
        result = kv.capacity_shape(rows, jobs)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["source_after_one_offload_blocks"], 1792)
        for free in (1800, 2500):
            rows[0]["free_blocks"] = free
            self.assertEqual(
                kv.capacity_shape(rows, jobs)["status"], "INVALID_EXPERIMENT_SHAPE"
            )


class GrowthTests(unittest.TestCase):
    def inventory(self, generation, stamp, generated, **changes):
        return dict(
            generation=generation,
            sampled_at=stamp,
            block_size=16,
            requests=[
                dict(
                    request_id="a",
                    generated=generated,
                    owned_blocks=changes.get("owned_blocks", 256),
                )
            ],
        )

    def test_decode_only_deduplicated_past_samples(self):
        reader = health.NativeDecodeGrowth()
        self.assertIsNone(reader.update(self.inventory(1, 10, 0)))
        self.assertEqual(reader.update(self.inventory(2, 11, 160)), 10)
        self.assertEqual(reader.update(self.inventory(2, 11, 160)), 10)
        self.assertEqual(reader.update(self.inventory(3, 12, 320)), 10)
        # A changed prefill allocation with flat generated count adds no growth.
        self.assertEqual(
            reader.update(self.inventory(4, 14, 320, owned_blocks=1000)), 0
        )
        with self.assertRaises(ValueError):
            reader.update(self.inventory(5, 14.5, 300))

    def test_insufficient_span_or_long_idle_gap_is_unknown(self):
        reader = health.NativeDecodeGrowth()
        self.assertIsNone(reader.update(self.inventory(1, 10, 0)))
        self.assertIsNone(reader.update(self.inventory(2, 10.2, 16)))
        self.assertIsNone(reader.update(self.inventory(3, 20, 320)))
        with self.assertRaises(ValueError):
            reader.update(self.inventory(4, 19, 400))


class LookaheadPolicyTests(unittest.TestCase):
    def cfg(self):
        return config(capacity_lookahead_s=16, capacity_only=True)

    def decide(self, source, target=None, selected=None):
        return p.PressurePolicy(self.cfg(), p.ReservationLedger()).decide(
            10,
            source,
            target or pool("tp4"),
            [selected or candidate(releasable_blocks=80, estimated_release_s=16)],
        )

    def test_healthy_service_forecast_crossing_guard_can_start(self):
        source = pool(free_blocks=250, growth_blocks_s=10, unallocated_prompt_blocks=0)
        decision = self.decide(source)
        self.assertEqual(decision.action, "START_SHADOW")
        self.assertTrue(decision.capacity_path)
        self.assertGreater(decision.source.capacity, 1)
        self.assertEqual(decision.candidates[0]["capacity_required_release_blocks"], 35)
        # Legacy modes are unaffected by opt-in lookahead.
        legacy = p.PressurePolicy(config(), p.ReservationLedger()).decide(
            10, source, pool("tp4"), [candidate()]
        )
        self.assertEqual(legacy.action, "STAY")

    def test_no_early_service_trigger_or_impossible_release(self):
        for source in (
            pool(decode_ratio=2, free_blocks=2000, unallocated_prompt_blocks=0),
            pool(free_blocks=138, growth_blocks_s=10, unallocated_prompt_blocks=0),
            pool(free_blocks=100, growth_blocks_s=None, unallocated_prompt_blocks=0),
        ):
            self.assertEqual(self.decide(source).action, "STAY")

    def test_pending_prompt_is_counted_once_and_target_limits_remain(self):
        source = pool(free_blocks=250, growth_blocks_s=0, unallocated_prompt_blocks=130)
        self.assertEqual(self.decide(source).action, "START_SHADOW")
        self.assertEqual(
            self.decide(source, pool("tp4", free_blocks=100)).action, "STAY"
        )
        self.assertEqual(
            self.decide(source, pool("tp4", decode_ratio=2)).action, "STAY"
        )
        self.assertEqual(
            self.decide(source, selected=candidate(target_decode_cost=None)).action,
            "STAY",
        )

    def test_flat_growth_at_physical_guard_can_still_relieve(self):
        self.assertEqual(
            self.decide(
                pool(free_blocks=100, growth_blocks_s=0, unallocated_prompt_blocks=0)
            ).action,
            "START_SHADOW",
        )

    def test_no_start_is_explicit_coverage_gap(self):
        result = kv.assess([], [], "migrate", 1024)
        self.assertEqual(result["coverage"], "COVERAGE_GAP")
        self.assertIsNone(result["start_to_source_release_s"])

    def test_release_receipt_timing_and_preemption_coverage(self):
        source = pool(free_blocks=250, growth_blocks_s=10, unallocated_prompt_blocks=0)
        decision = self.decide(source)
        audit = [
            dict(
                kind="pressure_decision",
                execution="RESERVED",
                decision=asdict(decision),
                source_reading=asdict(source),
            ),
            dict(
                kind="pressure_metrics_sample",
                pool="tp1",
                unix_s=99,
                metrics=dict(preemptions_total=0),
            ),
            dict(kind="pressure_actual_start", unix_s=100),
            dict(
                kind="pressure_source_release_confirmed",
                receipt=dict(released_unix_ns=116_000_000_000),
            ),
        ]
        metrics = [dict(migrated=True, arrival_valid=True)]
        result = kv.assess(audit, metrics, "migrate", 1024)
        self.assertEqual(result["coverage"], "KV_RELEASE_OBSERVED")
        self.assertEqual(result["start_to_source_release_s"], 16)
        self.assertIsNone(result["actual_released_blocks"])
        audit[1]["metrics"]["preemptions_total"] = 1
        self.assertEqual(
            kv.assess(audit, metrics, "migrate", 1024)["coverage"], "COVERAGE_GAP"
        )
        audit[1]["metrics"]["preemptions_total"] = 0
        metrics[0]["arrival_valid"] = False
        self.assertEqual(
            kv.assess(audit, metrics, "migrate", 1024)["coverage"], "COVERAGE_GAP"
        )


if __name__ == "__main__":
    unittest.main()
