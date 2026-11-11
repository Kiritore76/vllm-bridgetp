# SPDX-License-Identifier: Apache-2.0
"""Six-card engine reuse: sleep the old form, wake it on the reverse edge."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp.test_pressure_conversion import Reader
from tests.bridge_tp.test_runtime_instance_ids import canonical_plan
from tests.bridge_tp.test_six_card_gate import snapshot
from tools.bridge_tp import pressure_conversion as pc
from tools.bridge_tp import six_conversion as six
from tools.bridge_tp.conversion_channels import TargetChannelGraph
from tools.bridge_tp.conversion_sequence import validate


def reuse_plan(root, layout):
    plan = canonical_plan(root, layout)
    plan["controller_argv"] += ["--maximum-migrations", "0"]
    plan["stager_argv"][1] = "tools/bridge_tp/pressure_sessions.py"
    plan["conversion"] = dict(engine_reuse="sleep")
    six.configure(plan)
    return plan


class Harness:
    """One backend driven through several edges, with fake engines."""

    def __init__(self, test, root, layout, *, fail_sleep=False, residual_mib=900):
        self.test = test
        self.plan = plan = reuse_plan(root, layout)
        self.run = run = plan["run_dir"]
        run.mkdir(parents=True)
        self.calls, self.events = [], []
        self.alive = set(r["instance_id"] for r in plan["instances"]["instances"])
        self.asleep = set()
        self.fail_sleep = fail_sleep
        self.residual_mib = residual_mib

        for target in plan["six_targets"].values():
            Path(target["mechanism_root"]).mkdir(parents=True)
            pc.atomic_json(target["channel_plan"], target["connector_sources"])
        for source in plan["source_links"]:
            pc.atomic_json(
                run / f"gpu_targets_{source}.json",
                six.source_entries(source, plan["six_targets"]),
            )

        def start(extra):
            self.calls.append(("start", extra["instance_id"]))
            self.alive.add(extra["instance_id"])

        def stop(ids):
            self.calls.append(("stop", list(ids)))
            self.alive.difference_update(ids)
            self.asleep.difference_update(ids)

        def sleep(endpoints):
            self.calls.append(("sleep", sorted(endpoints)))
            if self.fail_sleep:
                raise RuntimeError("sleep endpoint refused")
            self.asleep.update(endpoints)
            return dict.fromkeys(endpoints, 1.4)

        def warm(extra):
            if extra.get("wake"):
                self.calls.append(("wake", extra["instance_id"]))
                self.test.assertIn(extra["instance_id"], self.asleep)
                self.asleep.discard(extra["instance_id"])

        self.ops = NS(
            event=lambda kind, **v: self.events.append((kind, v)),
            stop=stop,
            start=start,
            sleep=sleep,
            gpu_memory_used=lambda gpus: dict.fromkeys(gpus, self.residual_mib),
            start_receiver=lambda target: self.calls.append(
                ("stager", target["instance_id"])
            ),
            alive=lambda name: name in self.alive,
            gpus_released=lambda ids: True,
            warm=warm,
            scrape=lambda endpoint, native: dict(
                num_running=0,
                num_waiting=0,
                free_kv_blocks=native["free_blocks"],
                sampled_unix_s=time.time(),
            ),
        )
        self.backend = six.SixBackend(
            plan["registry"],
            plan["instances"],
            run,
            self.ops,
            base_port=8400,
            context_limits=(32768, 32768),
            sequence_context={},
            resources=plan,
        )
        self.backend.engine_reuse = "sleep"
        pc.atomic_json(
            run / "layout_status.json", dict(unix_s=time.time(), instances={})
        )

    def refresh(self):
        backend, run = self.backend, self.run
        now = time.time()
        resources = pc.read_json(run / "layout_control.json")["migration_resources"]
        instances = {}
        for name, row in backend.rows.items():
            lifecycle = backend.layout.get(name, {}).get(
                "lifecycle", backend.registry.instances[name].lifecycle
            )
            native = Reader(name).poll(now)
            Path(row["inventory_path"]).write_text(json.dumps(native) + "\n")
            instances[name] = dict(
                lifecycle=lifecycle,
                sessions=0,
                streams_pending=0,
                open_admissions=0,
                pressure=dict(capacity=0.1, zone="SPARE"),
                unified_fullness=snapshot(now),
                inventory=native,
                requests=[],
            )
        pc.atomic_json(
            run / "layout_status.json",
            dict(
                unix_s=now,
                layout_revision=backend.revision,
                instances=instances,
                migration_resource_targets=list(resources["targets"]),
                migration_resource_sources=list(resources["source_links"]),
            ),
        )
        for target in backend.targets.values():
            receipts = (
                Path(target["mechanism_root"]) / "sessions/cpu/gpu_channel_plan_receipts"
            )
            receipts.mkdir(parents=True, exist_ok=True)
            for rank in range(2):
                pc.atomic_json(
                    receipts / f"tp_rank_{rank}.json",
                    dict(
                        completed_unix_s=now,
                        planned=[
                            r["generation"]
                            for r in pc.read_json(target["channel_plan"])
                        ],
                        closed=list(range(1, backend.alloc["generation"] + 1)),
                    ),
                )
                for gen in range(1001, backend.alloc["generation"] + 1):
                    pc.atomic_json(
                        receipts / f"tp_rank_{rank}-retired-{gen}.json",
                        dict(status="ABORTED"),
                    )
            pre = (
                Path(target["mechanism_root"])
                / "sessions/cpu/gpu_channel_preconnect_receipts"
            )
            pre.mkdir(parents=True, exist_ok=True)
            for c in pc.read_json(target["channel_plan"]):
                for name in [f"source-{c['source']}-{target['instance_id']}.json"] + [
                    f"tp_rank_{r}-{c['source']}.json" for r in range(2)
                ]:
                    pc.atomic_json(
                        pre / name,
                        dict(status="WARMUP_COMPLETE", channel_generation=c["generation"]),
                    )

    def edge(self, edge):
        _, after, bundle, mode = six.EDGES[edge]
        backend = self.backend
        step = dict(bundle=bundle, target_mode=mode)
        backend.begin_step(edge, step)
        executor = pc.conversion.ConversionExecutor(
            backend.registry, backend, self.run / f"plan_{edge}.json", clock=time.time
        )
        executor.plan_create(
            edge,
            backend.registry.registry_revision,
            bundle,
            mode,
            "DRAIN",
            "ACTIVE",
            time.time() + 10,
            "NO_PROFILE",
        )

        def new_target(target):
            Path(target["mechanism_root"]).mkdir(parents=True, exist_ok=True)

        with (
            patch.object(six, "initialize_target", new_target),
            patch.object(six, "initialize_links"),
        ):
            while executor.plan["state"] not in {
                "COMPLETED",
                "RECOVERY_REQUIRED",
                "ABORTED",
            }:
                self.refresh()
                executor.plan_execute(time.time())
                time.sleep(0.002)
        self.test.assertEqual(executor.plan["state"], "COMPLETED", executor.plan)
        self.refresh()
        self.test.assertTrue(backend.step_acknowledged())
        self.test.assertEqual(
            {r.bundle_id: r.tp_degree for r in backend.active().values()},
            six.PATH[after],
        )
        return executor.plan


class SleepReuseTests(unittest.TestCase):
    def test_serving_flags_enable_sleep_mode_only_on_request(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = reuse_plan(Path(folder), "L1AC")
            self.assertIn("--enable-sleep-mode", plan["source_argv"])
            self.assertIn("--enable-sleep-mode", plan["target_argv"])
            self.assertEqual(plan["source_env"]["VLLM_SERVER_DEV_MODE"], "1")
            for extra in plan["extra_engines"]:
                if "receiver_argv" in extra:
                    self.assertIn("--enable-sleep-mode", extra["receiver_argv"])
                    self.assertEqual(extra["receiver_env"]["VLLM_SERVER_DEV_MODE"], "1")
        with tempfile.TemporaryDirectory() as folder:
            plan = canonical_plan(Path(folder), "L1AC")
            six.configure(plan)
            self.assertNotIn("--enable-sleep-mode", plan["source_argv"])
            self.assertNotIn("VLLM_SERVER_DEV_MODE", plan["source_env"])

    def test_request_validation(self):
        base = dict(
            profile="six_card",
            base_port=8400,
            sequence=[
                dict(bundle="A", target_mode="TP2", trigger_s=1, deadline_s=60)
            ],
        )
        validate(dict(base, engine_reuse="sleep"))
        with self.assertRaises(ValueError):
            validate(dict(base, engine_reuse="hibernate"))
        with self.assertRaises(ValueError):
            validate(dict(base, profile="four_card", engine_reuse="sleep"))

    def test_merge_split_merge_wakes_the_same_engines(self):
        with tempfile.TemporaryDirectory() as folder:
            h = Harness(self, Path(folder), "L1AC")
            registry = h.backend.registry
            a_tp1 = sorted(
                i
                for i, r in registry.instances.items()
                if r.bundle_id == "A" and r.tp_degree == 1
            )
            # 1. First A merge: no sleeping TP2 yet, so it cold-starts one;
            # the two A TP1 engines sleep instead of stopping.
            h.edge("L1AC-L2C")
            a_tp2 = [
                i
                for i, r in registry.instances.items()
                if r.bundle_id == "A" and r.tp_degree == 2
            ]
            self.assertEqual(len(a_tp2), 1)
            self.assertIn(("start", a_tp2[0]), h.calls)
            self.assertIn(("sleep", a_tp1), h.calls)
            self.assertFalse(any(k == "stop" for k, _ in h.calls))
            for name in a_tp1:
                self.assertEqual(registry.instances[name].lifecycle, "SLEEPING")
                self.assertEqual(h.backend.layout[name]["lifecycle"], "SLEEPING")
                # Their controller source entries stay (identity never retired).
                self.assertIn(
                    name,
                    pc.read_json(h.run / "layout_control.json")["migration_resources"][
                        "source_links"
                    ],
                )
                self.assertEqual(
                    pc.read_json(h.run / f"gpu_targets_{name}.json"), []
                )
            kinds = [k for k, _ in h.events]
            self.assertLess(kinds.index("SLEEP_START"), kinds.index("SLEPT"))
            self.assertLess(
                kinds.index("RESOURCES_RELEASED"), kinds.index("LOAD_START")
            )
            # 2. A split: the same two TP1 engines wake; A TP2 sleeps.
            h.calls.clear()
            h.events.clear()
            h.edge("L2C-L1AC")
            self.assertFalse(any(k == "start" for k, _ in h.calls), h.calls)
            self.assertEqual(
                sorted(n for k, n in h.calls if k == "wake"), a_tp1
            )
            self.assertIn(("sleep", a_tp2), h.calls)
            for name in a_tp1:
                self.assertEqual(registry.instances[name].lifecycle, "ACTIVE")
                self.assertEqual(registry.instances[name].ref.instance_generation,
                                 int(name.split("-g")[1].split("-")[0]))
                entries = pc.read_json(h.run / f"gpu_targets_{name}.json")
                self.assertEqual({e["target"] for e in entries}, {"B-g2-tp2"})
            self.assertEqual(registry.instances[a_tp2[0]].lifecycle, "SLEEPING")
            resources = pc.read_json(h.run / "layout_control.json")[
                "migration_resources"
            ]
            self.assertIn(a_tp2[0], resources["targets"])
            self.assertEqual(
                pc.read_json(resources["targets"][a_tp2[0]]["channel_plan"]), []
            )
            kinds = [k for k, _ in h.events]
            self.assertLess(kinds.index("SLEPT"), kinds.index("WAKE_START"))
            self.assertLess(
                kinds.index("SOURCE_CHANNELS_PUBLISHED"),
                kinds.index("ROUTE_PUBLISHED"),
            )
            # Woken sources get fresh channel generations at the receiver.
            first = {
                c["source"]: c["generation"]
                for c in h.plan["six_targets"]["B-g2-tp2"]["connector_sources"]
            }
            now = {
                c["source"]: c["generation"]
                for c in h.backend.targets["B-g2-tp2"]["connector_sources"]
            }
            for name in a_tp1:
                self.assertGreater(now[name], first[name])
            # 3. A merges again: the sleeping TP2 wakes; nothing cold-starts.
            h.calls.clear()
            h.events.clear()
            h.edge("L1AC-L2C")
            self.assertFalse(any(k == "start" for k, _ in h.calls), h.calls)
            self.assertEqual([n for k, n in h.calls if k == "wake"], a_tp2)
            self.assertEqual(registry.instances[a_tp2[0]].lifecycle, "ACTIVE")
            kinds = [k for k, _ in h.events]
            self.assertLess(
                kinds.index("CHANNEL_TOPOLOGY_PUBLISHED"),
                kinds.index("ROUTE_PUBLISHED"),
            )
            plan = pc.read_json(h.backend.targets[a_tp2[0]]["channel_plan"])
            active_tp1 = {
                i
                for i, r in registry.instances.items()
                if r.tp_degree == 1 and r.lifecycle == "ACTIVE"
            }
            self.assertEqual(len(active_tp1), 2)
            self.assertEqual({c["source"] for c in plan}, active_tp1)
            self.assertFalse(active_tp1 & set(a_tp1))

    def test_failed_sleep_falls_back_to_stop_and_retirement(self):
        with tempfile.TemporaryDirectory() as folder:
            h = Harness(self, Path(folder), "L1AC", fail_sleep=True)
            registry = h.backend.registry
            a_tp1 = sorted(
                i
                for i, r in registry.instances.items()
                if r.bundle_id == "A" and r.tp_degree == 1
            )
            h.edge("L1AC-L2C")
            self.assertIn(("stop", a_tp1), h.calls)
            for name in a_tp1:
                self.assertEqual(registry.instances[name].lifecycle, "RETIRED")
                self.assertEqual(h.backend.layout[name]["lifecycle"], "RETIRED")
            self.assertFalse(h.backend.sleeping)
            self.assertIn("SLEEP_FALLBACK", [k for k, _ in h.events])

    def test_memory_left_on_gpu_is_a_failed_sleep(self):
        with tempfile.TemporaryDirectory() as folder:
            h = Harness(self, Path(folder), "L1AC", residual_mib=9000)
            h.edge("L1AC-L2C")
            self.assertTrue(any(k == "stop" for k, _ in h.calls))
            self.assertFalse(h.backend.sleeping)

    def test_woken_receiver_reuses_its_name_only_after_retirement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "migration" / "T"
            target.mkdir(parents=True)
            graph = TargetChannelGraph(
                root, {}, {}, clock=time.time, event=lambda *a, **k: None
            )
            graph.add_target(
                "T", mechanism_root=target, channel_plan=target / "plan.json"
            )
            with self.assertRaises(ValueError):
                graph.add_target(
                    "T",
                    mechanism_root=target,
                    channel_plan=target / "plan.json",
                    wake=True,
                )
            del graph.targets["T"]
            with self.assertRaises(ValueError):
                graph.add_target(
                    "T", mechanism_root=target, channel_plan=target / "plan.json"
                )
            graph.add_target(
                "T", mechanism_root=target, channel_plan=target / "plan.json", wake=True
            )
            with self.assertRaises(ValueError):
                graph.add_target(
                    "U",
                    mechanism_root=root / "migration" / "U",
                    channel_plan=root / "migration" / "U" / "plan.json",
                    wake=True,
                )


if __name__ == "__main__":
    unittest.main()
