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
                "no long-request pressure",
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
            row = records[-1]["candidates"]["B->TP1"]
            reasons = row["failed"] + row["waiting"]
            self.assertTrue(any(f.startswith(reason) for f in reasons), reasons)

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
        self.assertEqual(first["evaluated_unix_s"], 1106 + 16 + 180)

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


class PrepareStageTests(unittest.TestCase):
    def test_tilt_starts_shortly_before_the_proposal(self):
        decider = d.ConversionDecider(EDGES)
        records = run(decider, status(arrivals=1080), 180)
        tilted = [r["evaluated_unix_s"] - 1000 for r in records if r["tilt"]]
        self.assertEqual((tilted[0], tilted[-1]), (120, 180))
        self.assertEqual(records[120]["tilt"], ["B"])
        self.assertEqual(records[120]["prepared"]["target_mode"], "TP1")
        self.assertIsNotNone(records[180]["proposal"])

    def test_tilt_needs_the_other_bundle_quiet_but_the_proposal_does_not(self):
        decider = d.ConversionDecider(EDGES)
        value = status(arrivals=108, long_arrivals=48)
        value["instances"]["B-g2-tp2"]["pressure"]["zone"] = "CRITICAL"
        records = run(decider, value, 180)
        self.assertFalse(any(r["tilt"] for r in records))
        self.assertIsNone(records[-1]["prepared"])
        self.assertIsNotNone(records[-1]["proposal"])  # liveness

    def test_a_failed_candidate_ends_the_tilt_at_once(self):
        decider = d.ConversionDecider(EDGES)
        records = run(decider, status(arrivals=1080), 130)
        self.assertEqual(records[-1]["tilt"], ["B"])
        record = decider.observe(status(arrivals=100), BUNDLES, MEMBERS, 1131)
        self.assertEqual(record["tilt"], [])
        self.assertIsNone(record["prepared"])

    def test_tilt_has_a_time_limit(self):
        decider = d.ConversionDecider(EDGES)
        value = status(arrivals=1080, long_on="B-g2-tp2")
        records = run(decider, value, 300)
        tilted = [r["evaluated_unix_s"] - 1000 for r in records if r["tilt"]]
        self.assertEqual((tilted[0], tilted[-1]), (120, 210))

    def test_a_long_request_delays_a_split_for_a_bounded_time(self):
        decider = d.ConversionDecider(EDGES)
        value = status(arrivals=1080, long_on="B-g2-tp2")
        records = run(decider, value, 300)
        (first,) = [r for r in records if r["proposal"]]
        self.assertEqual(first["evaluated_unix_s"] - 1000, 270)  # hold + limit
        self.assertEqual(records[200]["candidates"]["B->TP1"]["failed"], [])
        self.assertEqual(
            records[200]["candidates"]["B->TP1"]["waiting"],
            ["bundle serves a long request"],
        )
        # Once it is free the split goes ahead, without waiting for the limit.
        decider = d.ConversionDecider(EDGES)
        run(decider, value, 199)
        record = decider.observe(status(arrivals=1080), BUNDLES, MEMBERS, 1200)
        self.assertIsNotNone(record["proposal"])

    def test_pressure_moved_to_the_tp2_still_merges_urgently(self):
        decider = d.ConversionDecider(EDGES)
        value = status(arrivals=120, capacity=1.5, tp1_capacity=0.1, long_on="B-g2-tp2")
        records = run(decider, value, 100)
        first = next(r for r in records if r["proposal"])
        self.assertEqual(first["evaluated_unix_s"] - 1000, 60)
        self.assertEqual(first["proposal"]["bundle"], "A")
        self.assertTrue(first["candidates"]["A->TP2"]["urgent"])
        # A TP-k full of short requests is not a reason to merge.
        decider = d.ConversionDecider(EDGES)
        records = run(
            decider, status(arrivals=120, capacity=1.5, tp1_capacity=0.1), 100
        )
        self.assertFalse(any(r["proposal"] for r in records))

    def test_signals_are_smoothed(self):
        decider = d.ConversionDecider(EDGES)
        decider.observe(status(arrivals=120, tp1_capacity=1.4), BUNDLES, MEMBERS, 1000)
        record = decider.observe(status(arrivals=120), BUNDLES, MEMBERS, 1001)
        self.assertEqual(record["signals"]["tp1_capacity_max"], 1.4)
        record = decider.observe(status(arrivals=120), BUNDLES, MEMBERS, 1016)
        self.assertEqual(record["signals"]["tp1_capacity_max"], 0.2)
        # One waiting sample in ten seconds is not a queue; most of them are.
        decider = d.ConversionDecider(EDGES)
        for t in range(10):
            value = status(arrivals=120, waiting=3 if t == 5 else 0)
            record = decider.observe(value, BUNDLES, MEMBERS, 1000 + t)
            self.assertFalse(record["signals"]["tpk_waiting"])
        decider = d.ConversionDecider(EDGES)
        for t in range(10):
            value = status(arrivals=120, waiting=0 if t == 5 else 3)
            record = decider.observe(value, BUNDLES, MEMBERS, 1000 + t)
        self.assertTrue(record["signals"]["tpk_waiting"])


class PrepareDriverTests(unittest.TestCase):
    def driver(self, root, mode, **request):
        driver, events = DeciderDriverTests.driver(DeciderDriverTests(), root, mode)
        driver.request = dict(driver.request, **request)
        driver.decider.rules["hold_s"] = 60.0  # prepare at once, proposal at 60s
        driver.events = events
        return driver

    @staticmethod
    def tick(driver, root, value):
        value["unix_s"] = time.time()
        (root / "layout_status.json").write_text(json.dumps(value))
        driver.decided_at = -float("inf")
        driver.decide()

    def test_tilt_is_published_only_when_asked(self):
        for pre_tilt in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                driver = self.driver(root, "RECORD_ONLY", pre_tilt=pre_tilt)
                self.tick(driver, root, status(arrivals=1080))
                control = root / "layout_control.json"
                if pre_tilt:
                    self.assertEqual(
                        json.loads(control.read_text())["tilt"], ["B-g2-tp2"]
                    )
                    self.assertIn("PRE_TILT", driver.events)
                else:
                    self.assertFalse(control.exists())
                    self.assertNotIn("PRE_TILT", driver.events)

    def test_execute_prestarts_while_prepared_and_drops_when_it_ends(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = self.driver(root, "EXECUTE", pre_tilt=True)
            started, stopped = [], []
            ops = driver.backend.ops
            ops.start = lambda extra: started.append(extra["instance_id"])
            ops.stop = stopped.extend
            driver.backend.prestart_enabled = True
            self.tick(driver, root, status(arrivals=1080))
            self.assertEqual(started, ["B-g4-tp1", "B-g5-tp1"])
            self.tick(driver, root, status(arrivals=1080))
            self.assertEqual(len(started), 2)  # still the same preparation
            self.tick(driver, root, status(arrivals=100))
            self.assertEqual(stopped, ["B-g4-tp1", "B-g5-tp1"])
            self.assertEqual(driver.backend.layout_tilt, [])
            self.assertIsNone(driver.backend.prestarted)

    def test_record_only_never_prestarts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = self.driver(root, "RECORD_ONLY")
            driver.backend.ops.start = lambda extra: self.fail("started an engine")
            driver.backend.prestart_enabled = True
            self.tick(driver, root, status(arrivals=1080))

    def test_a_tilted_proposal_skips_only_the_service_part_of_the_gate(self):
        for pre_tilt in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                driver = self.driver(root, "EXECUTE", pre_tilt=pre_tilt, assist=True)
                driver.decider.rules["hold_s"] = 0.0
                value = status(arrivals=108, long_arrivals=48)
                self.tick(driver, root, value)
                plan = driver.executor.plan
                self.assertEqual(plan["bundle_id"], "A")
                self.assertEqual(plan["method"], "SELECTIVE_ROLL")
                backend = driver.backend
                self.assertEqual(backend.service_gate_passed, pre_tilt)
                # The tilt lasts until quiesce; the controller is held.
                control = json.loads((root / "layout_control.json").read_text())
                self.assertTrue(control["conversion_pending"])
                self.assertEqual(bool(control["tilt"]), pre_tilt)
                # The TP2 got busy from the tilt: only KV headroom is checked.
                busy = status(arrivals=108, long_arrivals=48)
                busy["unix_s"] = time.time()
                busy["instances"]["B-g2-tp2"]["pressure"]["zone"] = "CRITICAL"
                (root / "layout_status.json").write_text(json.dumps(busy))
                self.assertEqual(backend.gate(plan), pre_tilt)
                full = json.loads(json.dumps(busy))
                full["instances"]["B-g2-tp2"]["pressure"]["capacity"] = 0.95
                (root / "layout_status.json").write_text(json.dumps(full))
                self.assertFalse(backend.gate(plan))

    def test_a_decided_split_drains(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = self.driver(root, "EXECUTE", assist=True)
            driver.decider.rules["hold_s"] = 0.0
            self.tick(driver, root, status(arrivals=1080))
            self.assertEqual(driver.executor.plan["method"], "DRAIN")

    def test_the_tilt_time_limit_covers_a_waiting_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = self.driver(root, "RECORD_ONLY", pre_tilt=True)
            self.tick(driver, root, status(arrivals=1080))
            self.assertTrue(driver.backend.layout_tilt)
            driver.end_expired_tilt()
            self.assertTrue(driver.backend.layout_tilt)
            driver.tilt_since = time.monotonic() - 91
            driver.end_expired_tilt()
            self.assertEqual(driver.backend.layout_tilt, [])

    def test_quiesce_clears_the_tilt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = self.driver(root, "RECORD_ONLY", pre_tilt=True)
            self.tick(driver, root, status(arrivals=1080))
            driver.backend.rows["B-g2-tp2"].setdefault("endpoint", "t")
            driver.backend.quiesce(
                dict(
                    plan_id="p",
                    bundle_id="B",
                    target_bundle_mode="TP1",
                    refs=[dict(instance_id="B-g2-tp2", instance_generation=2)],
                )
            )
            self.assertEqual(driver.backend.layout_tilt, [])


class StartGateTests(unittest.TestCase):
    def test_gate_blocks_until_opened_and_times_out(self):
        import os
        import threading
        from unittest.mock import patch

        from vllm.bridge_tp import start_gate

        self.assertIsNone(start_gate.wait_for_start_gate())
        with tempfile.TemporaryDirectory() as folder:
            gate = Path(folder) / "gate"
            with patch.dict(os.environ, {start_gate.ENV: str(gate)}):
                threading.Timer(0.2, gate.touch).start()
                waited = start_gate.wait_for_start_gate(poll_s=0.01)
                self.assertGreaterEqual(waited, 0.15)
            never = {
                start_gate.ENV: str(gate) + "-never",
                start_gate.TIMEOUT_ENV: "0.1",
            }
            with patch.dict(os.environ, never), self.assertRaises(TimeoutError):
                start_gate.wait_for_start_gate(poll_s=0.01)

    def test_worker_waits_at_the_gate_before_any_cuda_use(self):
        import ast

        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "vllm/v1/worker/gpu_worker.py").read_text("utf-8"))
        init = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "init_device"
        )
        calls = [  # the body only (decorators are not part of the call order)
            n.func.id
            for statement in init.body
            for n in ast.walk(statement)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        self.assertEqual(calls[0], "wait_for_start_gate")
        first_if = next(n for n in init.body if isinstance(n, ast.If))
        gate_line = next(
            n.lineno
            for n in ast.walk(init)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "wait_for_start_gate"
        )
        self.assertLess(gate_line, first_if.lineno)


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

    def test_a_failed_result_write_does_not_stop_scene_cleanup(self):
        """Disk full (11-01): the write raised in run_scene's finally, ahead of
        stop_processes, and the engines outlived the scene."""
        import ast

        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        stopped = []
        driver = NS(stop=lambda: stopped.append(1) or dict(state="NOT_TRIGGERED"))
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "gone"
            with self.assertRaises(OSError):
                launch.finish_conversion(driver, missing)
            self.assertIsNone(launch.finish_conversion_best_effort(driver, missing))
        self.assertEqual(len(stopped), 2)
        tree = ast.parse(Path(launch.__file__).read_text("utf-8"))
        run_scene = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "run_scene"
        )
        (handler,) = [
            n for n in ast.walk(run_scene) if isinstance(n, ast.Try) and n.finalbody
        ]
        names = [
            n.func.id if isinstance(n.func, ast.Name) else n.func.attr
            for statement in handler.finalbody
            for n in ast.walk(statement)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
        ]
        self.assertNotIn("finish_conversion", names)
        self.assertIn("finish_conversion_best_effort", names)

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
    def test_scenarios_give_each_rule_its_case(self):
        from tools.bridge_tp import qwen3_phase_shift as phase

        root = Path(__file__).resolve().parents[2]
        inputs = json.loads(
            (
                root / "experiments/platforms/qwen3_pressure_observation/inputs.json"
            ).read_text(encoding="utf-8")
        )

        def share(scenario):
            jobs = [  # what the decider's window sees during the main phase
                j
                for j in phase.jobs(inputs, scenario)
                if j["workload_phase"] in {"long", "peak"}
            ]
            long = [
                j
                for j in jobs
                if len(j["request"]["prompt"]) + j["request"]["max_tokens"]
                >= d.LONG_TOKENS
            ]
            return long, len(long) / len(jobs)

        self.assertGreater(share("merge_heavy")[1], d.RULES["merge_long_share"])
        self.assertGreater(share("merge_moderate")[1], d.RULES["merge_long_share"])
        long, value = share("merge_deep")
        self.assertLess(value, d.RULES["merge_long_share"])
        self.assertEqual(
            (len(long[0]["request"]["prompt"]), long[0]["request"]["max_tokens"]),
            (16384, 8192),
        )
        long, value = share("split_trickle")
        self.assertTrue(long)
        self.assertLess(value, d.RULES["split_long_share"])

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
