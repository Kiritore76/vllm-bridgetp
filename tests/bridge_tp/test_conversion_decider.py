# SPDX-License-Identifier: Apache-2.0
"""Rule-based conversion decider (record-only first) on CPU."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from tools.bridge_tp import conversion_decider as d
from tools.bridge_tp import pressure_conversion as pc
from tools.bridge_tp.pressure_cpu import load

instance_pool = load("instance_pool")

# L1A on A={2,3}, B={0,1}: two TP1s (A) and one TP2 (B).
BUNDLES = {"A": 1, "B": 2}
MEMBERS = {"A": ["A-g1-tp1", "A-g3-tp1"], "B": ["B-g2-tp2"]}
EDGES = {"L1A": [["A", "TP2"], ["B", "TP1"]]}


def status(
    arrivals=0,
    long_arrivals=0,
    capacity=0.2,
    tp1_capacity=None,
    waiting=0,
    starts=(),
    long_on=None,
):
    def row(instance, tp):
        cap = tp1_capacity if tp == 1 and tp1_capacity is not None else capacity
        requests = []
        if instance == long_on:
            requests.append(
                dict(prompt_tokens=6144, max_output_tokens=2048, generated=10)
            )
        return dict(
            lifecycle="ACTIVE",
            pressure=dict(capacity=cap, zone="SPARE"),
            tp_degree=tp,
            inventory=dict(waiting=waiting if tp > 1 else 0),
            requests=requests,
        )

    return dict(
        unix_s=time.time(),
        instances={
            "A-g1-tp1": row("A-g1-tp1", 1),
            "A-g3-tp1": row("A-g3-tp1", 1),
            "B-g2-tp2": row("B-g2-tp2", 2),
        },
        workload=dict(
            window_s=120,
            arrivals=arrivals,
            long_arrivals=long_arrivals,
            migration_starts=[[1.0, s] for s in starts],
        ),
    )


def run(decider, value, seconds, step=1.0, start=1000.0):
    records, now = [], start
    while now <= start + seconds:
        records.append(decider.observe(value, BUNDLES, MEMBERS, now))
        now += step
    return records


class DeciderRuleTests(unittest.TestCase):
    def test_short_load_near_capacity_splits_after_the_hold(self):
        decider = d.ConversionDecider(EDGES)
        # 9 req/s short (72% of the L1A capacity), no long, no KV pressure.
        records = run(decider, status(arrivals=1080), 200)
        proposals = [r for r in records if r["proposal"]]
        self.assertEqual(len(proposals), 1)
        first = proposals[0]
        self.assertEqual(
            (first["proposal"]["bundle"], first["proposal"]["target_mode"]),
            ("B", "TP1"),
        )
        self.assertGreaterEqual(first["proposal"]["held_s"], 180)
        self.assertEqual(first["evaluated_unix_s"] - 1000, 180)
        # 9 req/s short is also more than the merged L2 could carry.
        self.assertEqual(
            first["candidates"]["A->TP2"]["failed"],
            [
                "no long-request pressure on TP1",
                "merged layout cannot carry the short rate",
            ],
        )

    def test_calm_short_load_does_not_split(self):
        decider = d.ConversionDecider(EDGES)
        records = run(decider, status(arrivals=360), 300)  # 3 req/s
        self.assertFalse(any(r["proposal"] for r in records))
        self.assertIn(
            "short load not approaching capacity",
            records[-1]["candidates"]["B->TP1"]["failed"],
        )
        # A queueing TP-k is enough even at a moderate rate.
        decider = d.ConversionDecider(EDGES)
        records = run(decider, status(arrivals=360, waiting=3), 200)
        self.assertEqual(sum(bool(r["proposal"]) for r in records), 1)

    def test_split_needs_no_kv_pressure_and_no_long_request_on_the_bundle(self):
        for value, reason in (
            (status(arrivals=1080, capacity=0.6), "KV pressure somewhere"),
            (
                status(arrivals=1080, long_on="B-g2-tp2"),
                "bundle serves a long request",
            ),
            (status(arrivals=1080, long_arrivals=200), "long share not low"),
            (status(arrivals=5), "long share not low"),
        ):
            decider = d.ConversionDecider(EDGES)
            records = run(decider, value, 200)
            self.assertFalse(any(r["proposal"] for r in records), reason)
            failed = records[-1]["candidates"]["B->TP1"]["failed"]
            self.assertTrue(any(f.startswith(reason) for f in failed), failed)

    def test_long_share_merges_the_tp1_bundle(self):
        decider = d.ConversionDecider(EDGES)
        # 0.9 req/s of which 44% long: the merged L2 carries 0.5 req/s short.
        records = run(decider, status(arrivals=108, long_arrivals=48), 200)
        (first,) = [r for r in records if r["proposal"]]
        self.assertEqual(
            (first["proposal"]["bundle"], first["proposal"]["target_mode"]),
            ("A", "TP2"),
        )

    def test_merge_needs_the_short_rate_to_fit_the_merged_layout(self):
        decider = d.ConversionDecider(EDGES)
        # 11 req/s short: L2 (9.5 req/s) cannot carry it with the margin.
        value = status(arrivals=1800, long_arrivals=540)
        records = run(decider, value, 200)
        self.assertFalse(any(r["proposal"] for r in records))
        self.assertIn(
            "merged layout cannot carry the short rate",
            records[-1]["candidates"]["A->TP2"]["failed"],
        )

    def test_frequent_tp1_migration_or_kv_at_guard_merges_urgently(self):
        decider = d.ConversionDecider(EDGES)
        starts = ["A-g1-tp1", "A-g3-tp1", "A-g1-tp1", "A-g3-tp1"]  # 2/min
        records = run(decider, status(arrivals=120, starts=starts), 200)
        first = next(r for r in records if r["proposal"])
        self.assertEqual(first["evaluated_unix_s"] - 1000, 180)
        self.assertGreater(first["signals"]["tp1_migrations_per_min"], 1.9)
        # A TP1 at its KV guard: the shorter urgent hold (liveness).
        decider = d.ConversionDecider(EDGES)
        records = run(decider, status(arrivals=120, tp1_capacity=1.05), 100)
        first = next(r for r in records if r["proposal"])
        self.assertEqual(first["evaluated_unix_s"] - 1000, 60)
        self.assertTrue(first["candidates"]["A->TP2"]["urgent"])

    def test_a_broken_condition_restarts_the_hold(self):
        decider = d.ConversionDecider(EDGES)
        records = run(decider, status(arrivals=1080), 100)
        records += run(decider, status(arrivals=1080, capacity=0.7), 5, start=1101)
        records += run(decider, status(arrivals=1080), 200, start=1107)
        first = next(r for r in records if r["proposal"])
        self.assertEqual(first["evaluated_unix_s"], 1107 + 180)

    def test_record_only_never_repeats_an_edge_within_the_cooldown(self):
        decider = d.ConversionDecider(EDGES)
        records = run(decider, status(arrivals=1080), 900)
        times = [r["evaluated_unix_s"] - 1000 for r in records if r["proposal"]]
        self.assertEqual(times, [180, 780])  # cooldown 600s per edge
        self.assertEqual(records[0]["mode"], "RECORD_ONLY")

    def test_execute_cools_down_every_edge_after_a_conversion(self):
        decider = d.ConversionDecider(EDGES, execute=True)
        decider.converted(1000)
        records = run(decider, status(arrivals=1080), 700, start=1001)
        times = [r["evaluated_unix_s"] - 1000 for r in records if r["proposal"]]
        # The condition held through the cooldown: proposed as it ends.
        self.assertEqual(times, [600])
        self.assertTrue(records[0]["cooling"])

    def test_layout_names_and_registry_layout(self):
        self.assertEqual(d.layout_name({"A": 1, "B": 1}), "L0")
        self.assertEqual(d.layout_name({"A": 2, "B": 2}), "L2")
        self.assertEqual(d.layout_name(BUNDLES), "L1A")
        registry, _ = instance_pool.fixed_pair_registry(
            source_gpu="2",
            target_gpus=["0", "1"],
            source_url="s",
            target_url="t",
            identity="qwen",
            source_context_limit=32768,
            target_context_limit=32768,
            extras=[
                dict(bundle="A", tp=1, gpus=["3"], endpoint="e", context_limit=32768)
            ],
        )
        self.assertEqual(d.registry_layout(registry), ({}, {}))  # PREPARING
        for row in registry.instances.values():
            registry.transition(row.ref, "ACTIVE")
        bundles, members = d.registry_layout(registry)
        self.assertEqual(bundles, BUNDLES)
        self.assertEqual(members, MEMBERS)


class DeciderDriverTests(unittest.TestCase):
    def driver(self, root, mode):
        registry, aliases = instance_pool.fixed_pair_registry(
            source_gpu="2",
            target_gpus=["0", "1"],
            source_url="s",
            target_url="t",
            identity="qwen",
            source_context_limit=32768,
            target_context_limit=32768,
            extras=[
                dict(bundle="A", tp=1, gpus=["3"], endpoint="e", context_limit=32768)
            ],
        )
        events = []
        backend = pc.ControllerConversionBackend(
            registry,
            instance_pool.registry_manifest(registry, aliases),
            root,
            NS(event=lambda kind, **values: events.append(kind)),
            base_port=8400,
            context_limits=(32768, 32768),
        )
        request = dict(decider=mode, deadline_s=600, base_port=8400, edges=EDGES)
        return pc.ConversionDriver(backend, request, root / "plan.json"), events

    def test_record_only_logs_and_never_holds_the_controller(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver, events = self.driver(root, "RECORD_ONLY")
            driver.start()
            control = json.loads((root / "layout_control.json").read_text())
            self.assertFalse(control["conversion_pending"])
            value = status(arrivals=1080)
            driver.decider.rules["hold_s"] = 0.0
            for _ in range(3):
                value["unix_s"] = time.time()
                (root / "layout_status.json").write_text(json.dumps(value))
                driver.decided_at = -float("inf")
                driver.decide()
            outcome = driver.stop()
            self.assertEqual(outcome["state"], "NOT_TRIGGERED")
            self.assertGreaterEqual(outcome["decisions"], 3)
            self.assertEqual(
                [(p["bundle"], p["target_mode"]) for p in outcome["proposals"]],
                [("B", "TP1")],
            )
            self.assertIn("DECIDER_PROPOSAL", events)
            self.assertNotIn("TRIGGER", events)
            rows = (root / "conversion_decider.jsonl").read_text().splitlines()
            self.assertGreaterEqual(len(rows), 3)
            control = json.loads((root / "layout_control.json").read_text())
            self.assertFalse(control["conversion_pending"])

    def test_a_decider_run_without_conversion_is_a_valid_scene(self):
        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        decided = dict(decider="RECORD_ONLY", deadline_s=600, base_port=8400)
        explicit = dict(bundle="A", target_mode="TP2", trigger_s=30)
        self.assertTrue(launch.conversion_outcome_ok(decided, "NOT_TRIGGERED"))
        self.assertFalse(launch.conversion_outcome_ok(explicit, "NOT_TRIGGERED"))
        for request in (decided, explicit):
            self.assertTrue(launch.conversion_outcome_ok(request, "COMPLETED"))
            self.assertTrue(launch.conversion_outcome_ok(request, "ABORTED"))
            self.assertFalse(launch.conversion_outcome_ok(request, "FENCED"))

    def test_stale_status_is_not_decided_on(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver, _ = self.driver(root, "RECORD_ONLY")
            value = status(arrivals=1080)
            value["unix_s"] = time.time() - 10
            (root / "layout_status.json").write_text(json.dumps(value))
            driver.decide()
            self.assertEqual(driver.decisions, 0)

    def test_launcher_builds_the_start_layout_edges(self):
        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        registry, _ = instance_pool.fixed_pair_registry(
            source_gpu="2",
            target_gpus=["0", "1"],
            source_url="s",
            target_url="t",
            identity="qwen",
            source_context_limit=32768,
            target_context_limit=32768,
            extras=[
                dict(bundle="A", tp=1, gpus=["3"], endpoint="e", context_limit=32768)
            ],
        )
        request = dict(decider="RECORD_ONLY", deadline_s=600, base_port=8400)
        value = launch.conversion_request(
            dict(layout=dict(name="L1A"), conversion=request), registry
        )
        self.assertEqual(value["edges"], EDGES)
        for bad in (
            dict(request, decider="MAYBE"),
            dict(request, rules=dict(nothing=1)),
            dict(request, bundle="A"),
        ):
            with self.assertRaises(ValueError):
                launch.conversion_request(
                    dict(layout=dict(name="L1A"), conversion=bad), registry
                )


class PhaseShiftTests(unittest.TestCase):
    def test_phases_give_each_rule_its_window(self):
        from tools.bridge_tp import qwen3_phase_shift as phase

        root = Path(__file__).resolve().parents[2]
        inputs = json.loads(
            (
                root / "experiments/platforms/qwen3_pressure_observation/inputs.json"
            ).read_text(encoding="utf-8")
        )
        jobs = phase.jobs(inputs)
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
        by = {}
        for job in jobs:
            by.setdefault(job["workload_phase"], []).append(job)
        p1 = [j for j in by["P1_short"] if j["at_s"] >= 60]
        self.assertEqual(len(p1), 2700)
        long = [j for j in by["P2_long"] if j["workload_role"] == "long_input_output"]
        self.assertEqual(len(long), 144)
        self.assertAlmostEqual(len(long) / len(by["P2_long"]), 144 / 324)
        request = long[0]["request"]
        self.assertEqual((len(request["prompt"]), request["max_tokens"]), (6144, 2048))
        self.assertGreaterEqual(
            len(request["prompt"]) + request["max_tokens"], d.LONG_TOKENS
        )
        self.assertLess(max(j["at_s"] for j in jobs), 840)


if __name__ == "__main__":
    unittest.main()
