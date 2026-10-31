# SPDX-License-Identifier: Apache-2.0
"""S1-b CPU contracts: class-internal routing and one global migration pair."""

import unittest
from dataclasses import asdict, replace

from tests.bridge_tp.test_pressure_policy import candidate, config, p, pool, r
from tests.bridge_tp.test_pressure_workload import cfg as workload_cfg
from tests.bridge_tp.test_pressure_workload import reading
from tools.bridge_tp.pressure_cpu import load

m = load("pressure_multi")

TP = dict(tp1=1, tp4=2, a1=1, a2=1, b1=2, b2=2)


def inst(name, base="tp1", **kwargs):
    return replace(reading(base, **kwargs), pool_id=name)


def coordinator(cfg=None):
    records = []
    return r.PressureCoordinator(cfg or workload_cfg(), records.append), records


class Started:
    def __init__(self):
        self.calls = []

    def bind(self, request_id, session_id):
        return r.BoundAction(
            request_id, session_id, lambda: self.calls.append(request_id)
        )


class ClassRoutingTests(unittest.TestCase):
    def admit(self, readings, prompt=2048, **kwargs):
        co, records = coordinator()
        result = co.admission_multi("q", prompt, 1024, readings, TP, 10, "k", **kwargs)
        return result, records[-1], co

    def test_less_loaded_tp1_is_used_before_tp2(self):
        result, record, _ = self.admit(
            [inst("a1", load=0.9), inst("a2", load=0.1), inst("b1", "tp4")]
        )
        self.assertEqual(result.pool_id, "a2")
        self.assertEqual(
            [row["pool_id"] for row in record["class_ranking"][0]], ["a2", "a1"]
        )
        self.assertEqual(record["unified_choice"]["pool_id"], "a2")
        self.assertFalse(record["unified_choice"]["executed"])

    def test_prompt_fit_ranks_before_pressure(self):
        rows = m.class_ranking(
            [inst("a1", load=0.9), inst("a2", load=0.1, free_blocks=50)],
            2048,
            1024,
            p.ReservationLedger(),
            workload_cfg(),
            10,
        )
        self.assertEqual(
            [(x["pool_id"], x["fit"]) for x in rows],
            [
                ("a1", "FIT"),
                ("a2", "NO_FIT"),
            ],
        )

    def test_equal_instances_break_ties_by_id(self):
        result, _, _ = self.admit([inst("a2"), inst("a1"), inst("b1", "tp4")])
        self.assertEqual(result.pool_id, "a1")

    def test_stale_reading_and_all_stale_fallback(self):
        rows = m.class_ranking(
            [inst("a1", now=10), inst("a2", now=4)],
            512,
            128,
            p.ReservationLedger(),
            workload_cfg(),
            10,
        )
        self.assertEqual(rows[0]["pool_id"], "a1")
        self.assertIsNone(rows[1]["score"])
        ledger = p.ReservationLedger()
        ledger.track_ingress("x", "x", "a1", 10)
        rows = m.class_ranking(
            [inst("a1", now=4), inst("a2", now=4)], 512, 128, ledger, workload_cfg(), 10
        )
        self.assertEqual([x["pool_id"] for x in rows], ["a2", "a1"])
        self.assertIn("no valid projected pressure", rows[0]["basis"])

    def test_single_class_layouts_take_the_class_leader(self):
        for readings, expected in (
            ([inst("a1", load=0.9), inst("a2", load=0.1)], "a2"),
            ([inst("b1", "tp4", load=0.9), inst("b2", "tp4", load=0.2)], "b2"),
        ):
            result, record, co = self.admit(readings)
            self.assertEqual(result.pool_id, expected)
            self.assertTrue(result.routing_reason.startswith("class-only"))
            self.assertEqual(co._owners["q"], expected)
            self.assertEqual(record["reservation_after"][-1]["pool_id"], expected)

    def test_context_infeasible_class_only_raises(self):
        co, _ = coordinator()
        with self.assertRaises(p.RoutingError):
            co.admission_multi("q", 9000, 1024, [inst("a1"), inst("a2")], TP, 10, "k")

    def test_one_instance_per_class_equals_two_pool_admission(self):
        sequence = [(0.1, 0.2), (0.9, 0.2), (0.5, 0.5), (1.4, 0.9), (0.2, 1.5)]
        old, old_records = coordinator()
        new, new_records = coordinator()
        for index, (a, b) in enumerate(sequence):
            pair = (reading(load=a), reading("tp4", load=b))
            args = (f"q{index}", 1024 * (index + 1), 512)
            expected = old.admission(*args, pair, now=10, key=f"k{index}")
            actual = new.admission_multi(*args, list(pair), TP, 10, f"k{index}")
            self.assertEqual(asdict(actual), asdict(expected))
            for key in ("reservation_after", "routing_relief_after"):
                self.assertEqual(new_records[-1][key], old_records[-1][key])
        self.assertEqual(new.ledger.snapshot(), old.ledger.snapshot())
        self.assertIs(new.policy_for("tp1"), new.policy)


class MultiTickTests(unittest.TestCase):
    cfg = staticmethod(lambda: config(sustain_s=0))

    def tick(self, co, readings, candidates, actions=None, now=10):
        tp = {i: (1 if i.startswith(("a", "tp1")) else 2) for i in readings}
        return co.tick_multi(now, readings, tp, candidates, actions)

    def test_one_pair_equals_two_pool_tick(self):
        for source in (
            pool(decode_ratio=1.5),
            pool(free_blocks=100, growth_blocks_s=10),
            pool(),
        ):
            old, old_records = coordinator(self.cfg())
            new, new_records = coordinator(self.cfg())
            started_old, started_new = Started(), Started()
            target = pool("tp4")
            expected = old.tick(
                10, source, target, [candidate()], {"a": started_old.bind("a", "s")}
            )
            self.tick(
                new,
                {"tp1": source, "tp4": target},
                {("tp1", "tp4"): [candidate()]},
                {"a": started_new.bind("a", "s")},
            )
            self.assertEqual(len(new_records), len(old_records))
            for old_row, new_row in zip(old_records, new_records):
                # The fixed pair writes the historical record plus multi fields.
                self.assertEqual({k: new_row[k] for k in old_row}, old_row)
            decision = new_records[0]["evaluations"][0]["pairs"][0]["decision"]
            self.assertEqual(decision, asdict(expected))
            self.assertEqual(started_new.calls, started_old.calls)
            self.assertEqual(new.ledger.snapshot(), old.ledger.snapshot())
            self.assertEqual(
                new.policy.source_state().keys(), old.policy.source_state().keys()
            )
            self.assertEqual(
                (new.policy._since, new.policy._latched, new.policy._last_start),
                (old.policy._since, old.policy._latched, old.policy._last_start),
            )
            self.assertEqual(new.policy.relief.snapshot(), old.policy.relief.snapshot())

    def test_emergency_source_beats_service_source(self):
        co, records = coordinator(self.cfg())
        started = Started()
        readings = {
            "a1": pool("a1", decode_ratio=1.5),
            "a2": pool("a2", free_blocks=100, growth_blocks_s=10),
            "b1": pool("b1"),
        }
        result = self.tick(
            co,
            readings,
            {
                ("a1", "b1"): [candidate("x", owner="a1")],
                ("a2", "b1"): [candidate("y", owner="a2")],
            },
            {k: started.bind(k, "s-" + k) for k in ("x", "y")},
        )
        self.assertEqual(result[:2], ("a2", "b1"))
        self.assertEqual(started.calls, ["y"])
        decision = next(x for x in records if x["kind"] == "pressure_decision_multi")
        self.assertEqual(decision["chosen"], ["a2", "b1"])
        self.assertEqual(decision["deferred"], [["a1", "b1"]])
        self.assertEqual(co.ledger.pending("b1")[0] > 0, True)

    def test_each_source_takes_its_best_target_and_advances_once(self):
        co, records = coordinator(config(sustain_s=3))
        readings = {
            "a1": pool("a1", decode_ratio=1.5),
            "b1": pool("b1", free_blocks=10),
            "b2": pool("b2"),
        }
        cands = {
            ("a1", "b1"): [candidate("x", owner="a1")],
            ("a1", "b2"): [candidate("x", owner="a1")],
        }
        self.assertIsNone(self.tick(co, readings, cands, now=10))
        state = co.policy_for("a1").source_state()
        reference = p.PressurePolicy(config(sustain_s=3), p.ReservationLedger())
        reference.observe_relief(10, readings["a1"])
        reference.decide(10, readings["a1"], readings["b2"], cands[("a1", "b2")])
        self.assertEqual(state["_since"], reference._since)
        self.assertEqual(state["_last_tick"], 10)
        co2, records2 = coordinator(self.cfg())
        started = Started()
        result = self.tick(co2, readings, cands, {"x": started.bind("x", "s")})
        self.assertEqual(result[:2], ("a1", "b2"))
        pairs = records2[0]["evaluations"][0]["pairs"]
        self.assertEqual([x["key"] is None for x in pairs], [True, False])

    def test_active_session_blocks_every_start(self):
        co, records = coordinator(self.cfg())
        co.ledger.reserve(
            p.Reservation("busy", "old", "b1", 10, 0.1, 0, "migration"), pool("b1")
        )
        started = Started()
        readings = {
            "a1": pool("a1", decode_ratio=1.5),
            "b1": pool("b1"),
            "b2": pool("b2"),
        }
        result = self.tick(
            co,
            readings,
            {("a1", "b2"): [candidate("x", owner="a1")]},
            {"x": started.bind("x", "s")},
        )
        self.assertIsNone(result)
        self.assertEqual(started.calls, [])
        self.assertTrue(records[-1]["global_session_active"])
        self.assertEqual(records[-1]["chosen"], ["a1", "b2"])
        self.assertEqual(records[-1]["execution"], "ADVISORY")

    def test_cooldown_is_global_and_trigger_state_is_per_source(self):
        co, co_records = coordinator(config(sustain_s=0, cooldown_s=2))
        started = Started()
        readings = {
            "a1": pool("a1", decode_ratio=1.5),
            "a2": pool("a2", decode_ratio=1.5),
            "b1": pool("b1"),
        }
        cands = {
            ("a1", "b1"): [candidate("x", owner="a1")],
            ("a2", "b1"): [candidate("y", owner="a2")],
        }
        first = self.tick(co, readings, cands, {"x": started.bind("x", "s1")})
        self.assertEqual(first[0], "a1")
        self.assertEqual(co.policy_for("a2")._last_start, 10)
        self.assertIsNot(co.policy_for("a1").relief, co.policy_for("a2").relief)
        co.finish("s1", "x", "CANCELLED", cleanup_confirmed=True)
        readings = {
            k: replace(v, sampled_at=11, published_at=11, decode_sampled_at=11)
            for k, v in readings.items()
        }
        cands = {
            ("a2", "b1"): [candidate("y", owner="a2", sampled_at=11, published_at=11)]
        }
        self.assertIsNone(
            self.tick(co, readings, cands, {"y": started.bind("y", "s2")}, now=11)
        )
        blocked = co_records[-1]["evaluations"][1]["pairs"][0]["decision"]
        self.assertEqual(blocked["reason"], "migration cooldown")
        readings = {
            k: replace(v, sampled_at=12.5, published_at=12.5, decode_sampled_at=12.5)
            for k, v in readings.items()
        }
        cands = {
            ("a2", "b1"): [
                candidate("y", owner="a2", sampled_at=12.5, published_at=12.5)
            ]
        }
        later = self.tick(co, readings, cands, {"y": started.bind("y", "s3")}, now=12.5)
        self.assertEqual(later[0], "a2")
        self.assertEqual(started.calls, ["x", "y"])

    def test_retired_source_generation_starts_fresh(self):
        co, _ = coordinator(config(sustain_s=3))
        readings = {"a1": pool("a1", decode_ratio=1.5), "b1": pool("b1")}
        self.tick(co, readings, {("a1", "b1"): [candidate("x", owner="a1")]})
        self.assertIsNotNone(co.policy_for("a1")._since)
        shared = co.policy.shared
        co.forget_source("a1")
        self.assertIsNone(co.policy_for("a1")._since)
        self.assertIs(co.policy_for("a1").shared, shared)

    def test_target_headroom_counts_pending_promises(self):
        ledger = p.ReservationLedger()
        self.assertEqual(m.target_headroom(pool("b1"), ledger), 2000)
        ledger.reserve(
            p.Reservation("s", "x", "b1", 300, 0.1, 0, "migration"), pool("b1")
        )
        self.assertEqual(m.target_headroom(pool("b1"), ledger), 1700)
        self.assertIsNone(m.target_headroom(pool("b1", free_blocks=None), ledger))


if __name__ == "__main__":
    unittest.main()


class ControllerManifestTests(unittest.TestCase):
    def test_controller_accepts_only_the_fixed_pair(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace as NS

        from tools.bridge_tp import run_pressure_balanced_pilot as pilot

        pool_module = load("instance_pool")
        registry, aliases = pool_module.fixed_pair_registry(
            source_gpu="2",
            target_gpus=["0", "1"],
            source_url="s",
            target_url="t",
            identity="model",
            source_context_limit=32768,
            target_context_limit=32768,
        )
        manifest = pool_module.registry_manifest(registry, aliases)
        controller = NS(target_tp_size=2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "instances.json"
            args = NS(
                instances=path, source_context_limit=32768, target_context_limit=32768
            )
            path.write_text(json.dumps(manifest))
            self.assertEqual(pilot.instance_manifest(args, controller), manifest)
            self.assertIsNone(pilot.instance_manifest(NS(instances=None), controller))
            extra = dict(
                manifest,
                instances=manifest["instances"]
                + [dict(manifest["instances"][0], instance_id="A-g3-tp1")],
            )
            path.write_text(json.dumps(extra))
            with self.assertRaises(ValueError):
                pilot.instance_manifest(args, controller)
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                pilot.instance_manifest(args, NS(target_tp_size=4))


class PreflightOrderTests(unittest.TestCase):
    def test_every_controller_preflight_writes_instances_first(self):
        # G1 (10-31) failed: up-front preflight ran before instances.json existed.
        import ast
        from pathlib import Path

        def called(node):
            func = node.func
            return func.attr if isinstance(func, ast.Attribute) else func.id

        root = Path(__file__).resolve().parents[2] / "tools/bridge_tp"
        for name in (
            "launch_pressure_balanced_pilot.py",
            "run_pressure_comparison.py",
            "run_qwen3_stay_migrate.py",
        ):
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            checked = 0
            for function in ast.walk(tree):
                if not isinstance(function, ast.FunctionDef):
                    continue
                calls = [
                    n
                    for n in ast.walk(function)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, (ast.Attribute, ast.Name))
                ]
                writes = [n.lineno for n in calls if called(n) == "write_instances"]
                for call in calls:
                    if (
                        called(call) == "preflight"
                        and isinstance(call.func, ast.Attribute)
                        and "controller" in ast.unparse(call.func.value)
                    ):
                        checked += 1
                        self.assertTrue(
                            any(line < call.lineno for line in writes),
                            f"{name}:{call.lineno}",
                        )
            self.assertTrue(checked, name)


class InstanceLifecycleTests(unittest.TestCase):
    """S1-c: lifecycle gates, generation-bound receipts, no oversubscription."""

    def started(self, co, readings, cands, request, session, now=10):
        calls = Started()
        tp = {i: (1 if i.startswith("a") else 2) for i in readings}
        result = co.tick_multi(
            now, readings, tp, cands, {request: calls.bind(request, session)}
        )
        return result, calls.calls

    def test_quiescing_instance_closes_admission_and_inbound_migration(self):
        co, records = coordinator()
        co.set_lifecycle("a2", "QUIESCING")
        result = co.admission_multi(
            "q",
            2048,
            1024,
            [inst("a1", load=0.9), inst("a2", load=0.1), inst("b1", "tp4")],
            TP,
            10,
            "k",
        )
        self.assertNotEqual(result.pool_id, "a2")
        self.assertEqual(records[0]["kind"], "pressure_instance_lifecycle")
        with self.assertRaises(p.RoutingError):
            co.admission_multi(
                "r", 128, 64, [inst("a2")], TP, 10, "k2", allowed_pool_id="a2"
            )
        co, _ = coordinator(config(sustain_s=0))
        co.set_lifecycle("b1", "QUIESCING")
        readings = {
            "a1": pool("a1", decode_ratio=1.5),
            "b1": pool("b1"),
            "b2": pool("b2"),
        }
        cands = {
            ("a1", "b1"): [candidate("x", owner="a1")],
            ("a1", "b2"): [candidate("x", owner="a1")],
        }
        result, calls = self.started(co, readings, cands, "x", "s")
        self.assertEqual(result[:2], ("a1", "b2"))
        self.assertEqual(calls, ["x"])

    def test_quiescing_source_still_migrates_out(self):
        co, _ = coordinator(config(sustain_s=0))
        co.set_lifecycle("a1", "QUIESCING")
        readings = {"a1": pool("a1", decode_ratio=1.5), "b1": pool("b1")}
        result, _ = self.started(
            co, readings, {("a1", "b1"): [candidate("x", owner="a1")]}, "x", "s"
        )
        self.assertEqual(result[:2], ("a1", "b1"))

    def test_retirement_waits_for_settlement_and_drops_state(self):
        co, _ = coordinator(config(sustain_s=0))
        readings = {"a1": pool("a1", decode_ratio=1.5), "b1": pool("b1")}
        self.started(
            co, readings, {("a1", "b1"): [candidate("x", owner="a1")]}, "x", "s"
        )
        co.set_lifecycle("a1", "QUIESCING")
        with self.assertRaises(ValueError):
            co.set_lifecycle("a1", "RETIRED")
        co.finish("s", "x", "CANCELLED", cleanup_confirmed=True, target_id="b1")
        co.set_lifecycle("a1", "RETIRED")
        self.assertNotIn("a1", co._policies)
        with self.assertRaises(ValueError):
            co.set_lifecycle("a1", "ACTIVE")
        with self.assertRaises(ValueError):
            co.set_lifecycle("b1", "RETIRED")

    def test_receipts_must_name_the_bound_target_generation(self):
        co, _ = coordinator(config(sustain_s=0))
        readings = {"a1": pool("a1", decode_ratio=1.5), "b1": pool("b1")}
        self.started(
            co, readings, {("a1", "b1"): [candidate("x", owner="a1")]}, "x", "s"
        )
        with self.assertRaises(ValueError):
            co.observe_migration(
                "s", "B-g9-tp2", allocated_blocks=1, service_observed=False
            )
        with self.assertRaises(ValueError):
            co.finish("s", "x", "CANCELLED", cleanup_confirmed=True, target_id="b2")
        co.observe_migration("s", "b1", allocated_blocks=1, service_observed=False)
        self.assertEqual(co.ledger.entries["s"].allocated_blocks, 1)

    def test_two_sources_cannot_oversubscribe_one_target(self):
        co, co_records = coordinator(config(sustain_s=0, cooldown_s=0))
        readings = {
            "a1": pool("a1", decode_ratio=1.5),
            "a2": pool("a2", decode_ratio=1.5),
            "b1": pool("b1", free_blocks=600),
        }
        first, _ = self.started(
            co,
            readings,
            {
                ("a1", "b1"): [candidate("x", owner="a1")],
                ("a2", "b1"): [candidate("y", owner="a2")],
            },
            "x",
            "s1",
        )
        self.assertEqual(first[0], "a1")
        promised = co.ledger.entries["s1"].blocks
        co.finish(
            "s1",
            "x",
            "COMMITTED",
            ownership_confirmed=True,
            source_release_confirmed=True,
            target_id="b1",
        )
        self.assertEqual(co.ledger.pending("b1")[0], promised)
        later = {
            k: replace(v, sampled_at=11, published_at=11, decode_sampled_at=11)
            for k, v in readings.items()
        }
        second, calls = self.started(
            co,
            later,
            {
                ("a2", "b1"): [
                    candidate("y", owner="a2", sampled_at=11, published_at=11)
                ]
            },
            "y",
            "s2",
            now=11,
        )
        self.assertIsNone(second)
        self.assertEqual(calls, [])
        (evaluation,) = [
            e for e in co_records[-1]["evaluations"] if e["source"] == "a2"
        ]
        decision = evaluation["pairs"][0]["decision"]
        rejected = decision["candidates"][0]["rejections"]
        self.assertIn("target physical commitments exceed free blocks", rejected)
        self.assertLessEqual(co.ledger.pending("b1")[0], 600)

    def test_stale_target_evidence_blocks_execution(self):
        co, records = coordinator(config(sustain_s=0))
        readings = {"a1": pool("a1", decode_ratio=1.5), "b1": pool("b1")}
        cands = {
            ("a1", "b1"): [candidate("x", owner="a1", sampled_at=14, published_at=14)]
        }
        readings = {
            "a1": replace(
                readings["a1"], sampled_at=14, published_at=14, decode_sampled_at=14
            ),
            "b1": readings["b1"],
        }
        result, calls = self.started(co, readings, cands, "x", "s", now=14)
        self.assertIsNone(result)
        self.assertEqual(calls, [])
