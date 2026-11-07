# SPDX-License-Identifier: Apache-2.0
"""Six-card unified runtime identity integration, with CPU inventory/HTTP fakes."""

import dataclasses
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp.test_migration_plan import snapshot
from tests.bridge_tp.test_performance_trace import journal
from tests.bridge_tp.test_pressure_policy import config, pool, r
from tests.bridge_tp.test_qwen3_six_card import build
from tools.bridge_tp import migration_plan as mp
from tools.bridge_tp import qwen3_six_card as six
from tools.bridge_tp import run_pressure_balanced_pilot as pilot
from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp.pressure_cpu import load
from tools.bridge_tp.pressure_session import RollingPressureSession
from tools.bridge_tp.pressure_sessions import PressureSessions


def canonical_plan(root, layout):
    plan = build(root, layout)
    plan["scheduler"] = "unified"
    mp.install(plan)
    mp.activate_instance_ids(plan)
    return plan


class PlanTests(unittest.TestCase):
    def test_only_identity_and_path_fields_change_in_the_single_B_plan(self):
        for layout in six.MIXED:
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as folder:
                plan = build(Path(folder), layout)
                plan["scheduler"] = "unified"
                mp.install(plan)
                before = snapshot(plan)
                mp.activate_instance_ids(plan)
                changed = {k for k in before if plan[k] != before[k]}
                self.assertEqual(
                    changed,
                    {
                        "instances",
                        "config",
                        "source_env",
                        "target_env",
                        "controller_argv",
                        "migration_plan",
                        "migration_targets",
                        "migration_sources",
                    },
                )
                self.assertEqual(set(plan) - set(before), {"canonical_instance_ids"})
                for side in ("source", "target"):
                    keys = set(plan[side + "_env"]) | set(before[side + "_env"])
                    differences = {
                        k
                        for k in keys
                        if plan[side + "_env"].get(k) != before[side + "_env"].get(k)
                    }
                    self.assertEqual(
                        differences,
                        {
                            "BRIDGETP_PRESSURE_POOL",
                            "BRIDGETP_PRESSURE_INVENTORY_PATH",
                            "BRIDGETP_PERFORMANCE_TRACE_DIR",
                        },
                    )

    def test_native_capacity_uses_canonical_inventory_and_class_profile_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = canonical_plan(Path(folder), "L2C")
            plan["expected_context_limits"] = (32768, 32768)
            plan["config"]["policy"] = {}
            inventories = {
                row["instance_id"]: dict(
                    pool_id=row["instance_id"],
                    context_limit=32768,
                    block_size=16,
                    total_blocks=2325 if row["tp_degree"] == 1 else 11712,
                    free_blocks=2324 if row["tp_degree"] == 1 else 11711,
                )
                for row in plan["instances"]["instances"]
            }
            writes = {}
            with (
                patch.object(six, "wait_empty", return_value=inventories),
                patch(
                    "tools.bridge_tp.run_pressure_calibration.wait_empty"
                ) as legacy_reader,
            ):
                runner.prepare_native(
                    plan, NS(write_json=lambda p, d: writes.update({str(p): d}))
                )
            legacy_reader.assert_not_called()
            self.assertEqual(plan["config"]["tp1_total_kv_blocks"], 2325)
            self.assertEqual(plan["config"]["tp4_total_kv_blocks"], 11712)
            self.assertTrue(writes)

    def test_warmup_requires_actual_source_ids_not_legacy_alias_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = canonical_plan(Path(folder), "L2C")
            target = plan["instances"]["runtime_pool_ids"]["tp4"]
            receipts = {}
            for channel in plan["channel_sources"]:
                source = channel["source"]
                names = [f"source-{source}-{target}.json"] + [
                    f"tp_rank_{rank}-{source}.json" for rank in range(2)
                ]
                for name in names:
                    receipts[name] = dict(
                        status="WARMUP_COMPLETE",
                        channel_generation=channel["generation"],
                    )
            with patch.object(six.pc, "preconnect_receipts", return_value=receipts):
                self.assertEqual(len(six.warm_channels(plan)), 6)
                source = plan["instances"]["runtime_pool_ids"]["tp1"]
                value = receipts.pop(f"source-{source}-{target}.json")
                receipts[f"source-tp1-{target}.json"] = value
                with (
                    patch.object(six.time, "monotonic", side_effect=[0, 31]),
                    self.assertRaisesRegex(TimeoutError, "proof incomplete"),
                ):
                    six.warm_channels(plan)

    def test_four_card_and_pressure_activation_are_noops(self):
        for cards, scheduler in ((4, "unified"), (6, "pressure")):
            with (
                self.subTest(cards=cards, scheduler=scheduler),
                tempfile.TemporaryDirectory() as folder,
            ):
                plan = build(Path(folder), "L2C")
                plan.update(static_cards=cards, scheduler=scheduler)
                before = snapshot(plan)
                mp.install(plan)
                mp.activate_instance_ids(plan)
                self.assertEqual(plan, before)

    def test_session_binds_actual_owner_endpoint_and_target(self):
        session = object.__new__(RollingPressureSession)
        session.record = None
        session.config = NS(source_pool_id="C-g1-tp1")
        session.adapter = NS(source_url="wrong", target_url="wrong")
        session.ingress = NS(urls={"C-g4-tp1": "http://extra-source/"})
        records = []
        session.audit = NS(write=records.append)
        session.migration_id = "m"
        session.bind_target("B-g2-tp2", "http://target/", "B-g2-tp2")
        self.assertEqual(session.target_instance, "B-g2-tp2")
        self.assertEqual(session.adapter.target_url, "http://target")
        self.assertEqual(PressureSessions._source(session), "C-g1-tp1")
        session.source_instance = session._bind_source(NS(owner="C-g4-tp1"))
        self.assertEqual(session.adapter.source_url, "http://extra-source")
        self.assertEqual(PressureSessions._source(session), "C-g4-tp1")
        session.record = object()
        with self.assertRaises(ValueError):
            session.bind_target("A-g3-tp2", "http://other", "A-g3-tp2")

    def test_every_live_inventory_and_trace_uses_its_instance_id(self):
        for layout in six.LAYOUTS:
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as folder:
                plan = canonical_plan(Path(folder), layout)
                for row in six.instances(plan):
                    name = row["instance_id"]
                    self.assertEqual(row["trace_name"], name)
                    self.assertEqual(
                        Path(row["inventory_path"]).name, f"inventory_{name}.jsonl"
                    )
                for side, alias in (("source", "tp1"), ("target", "tp4")):
                    ids = plan["instances"]["legacy_pool_aliases"]
                    if alias not in ids:
                        continue
                    self.assertEqual(
                        plan[side + "_env"]["BRIDGETP_PRESSURE_POOL"], ids[alias]
                    )
                    argv = plan["controller_argv"]
                    path = argv[argv.index("--" + side + "-inventory") + 1]
                    self.assertEqual(
                        path, plan[side + "_env"]["BRIDGETP_PRESSURE_INVENTORY_PATH"]
                    )
                self.assertEqual(
                    pilot.pair_pools(plan["instances"]),
                    tuple(plan["instances"]["legacy_pool_aliases"].values()),
                )

    def test_existing_transport_and_mechanism_directories_do_not_change(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "L2C")
            argv = list(plan["target_argv"])
            stager = list(plan["stager_argv"])
            directory = plan["source_env"]["BRIDGETP_STREAM_RUN_DIR"]
            plan["scheduler"] = "unified"
            mp.install(plan)
            mp.activate_instance_ids(plan)
            self.assertEqual(plan["target_argv"], argv)
            self.assertEqual(plan["stager_argv"], stager)
            self.assertEqual(plan["source_env"]["BRIDGETP_STREAM_RUN_DIR"], directory)
            self.assertEqual(len(plan["migration_targets"]), 1)
            self.assertEqual(
                plan["migration_targets"][0]["runtime_pool_id"], "B-g2-tp2"
            )

    def test_manifest_rejects_wrong_scheduler_or_runtime_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = canonical_plan(root, "L2C")
            path = root / "manifest.json"
            path.write_text(json.dumps(plan["instances"]))
            args = NS(
                instances=path,
                scheduler="unified",
                no_migration=False,
                source_context_limit=32768,
                target_context_limit=32768,
            )
            controller = NS(target_tp_size=2)
            self.assertEqual(
                pilot.instance_manifest(args, controller), plan["instances"]
            )
            args.scheduler = "pressure"
            with self.assertRaisesRegex(ValueError, "canonical runtime IDs"):
                pilot.instance_manifest(args, controller)
            args.scheduler = "unified"
            plan["instances"]["runtime_pool_ids"]["tp4"] = "A-g3-tp2"
            path.write_text(json.dumps(plan["instances"]))
            with self.assertRaisesRegex(ValueError, "canonical runtime IDs"):
                pilot.instance_manifest(args, controller)

    def test_unknown_pressure_uses_canonical_context_contracts(self):
        manifest = dict(
            legacy_pool_aliases=dict(tp1="C-g1-tp1", tp4="B-g2-tp2"),
            runtime_pool_ids=dict(tp1="C-g1-tp1", tp4="B-g2-tp2"),
        )
        rows = pilot.unknown_routing_pools(
            NS(source_context_limit=32768, target_context_limit=32768),
            16,
            pair=pilot.pair_pools(manifest),
            pool_names=manifest["runtime_pool_ids"],
        )
        self.assertEqual({row.pool_id for row in rows}, {"C-g1-tp1", "B-g2-tp2"})

    def test_identity_change_preserves_tie_selection(self):
        mapping = {"tp1": "C-g1-tp1", "tp4": "B-g2-tp2"}
        choices = []
        for canonical in (False, True):
            ids = mapping if canonical else {k: k for k in mapping}
            co = r.PressureCoordinator(
                config(),
                lambda row: None,
                scheduler="unified",
                slots={v: 16 for v in ids.values()},
            )
            if canonical:
                co.pool_sort_ids = {v: k for k, v in mapping.items()}
            readings = tuple(
                dataclasses.replace(pool("tp1"), pool_id=ids[k]) for k in mapping
            )
            result = co.admission_multi(
                "q", 512, 128, readings, {ids["tp1"]: 1, ids["tp4"]: 2}, 10, "q"
            )
            choices.append(result.pool_id)
        self.assertEqual(choices, ["tp1", "C-g1-tp1"])


class FakeInventory:
    def __init__(self, path, pool_id, run_dir):
        self.pool_id = pool_id
        self.stop = threading.Event()
        self.observed_at = None
        self.generation = 0
        self.error = None
        self.deferred = None
        self.latest = None

    def poll(self, now):
        self.observed_at = now
        self.generation += 1
        self.latest = dict(
            schema="PRESSURE_INVENTORY_V1",
            dropped_before=0,
            writer_error=None,
            pool_id=self.pool_id,
            sampled_at=now,
            generation=self.generation,
            total_blocks=2001,
            usage_denominator_blocks=2000,
            free_blocks=2000,
            block_size=16,
            context_limit=32768,
            pending_prompt_tokens=0,
            waiting=0,
            running=0,
            requests=[],
        )
        return self.latest

    def diagnostics(self):
        return dict(pool_id=self.pool_id)

    def close(self):
        pass


class ControllerTests(unittest.TestCase):
    def test_real_control_loop_routes_records_and_settles_canonical_pools(self):
        ingress = load("pressure_ingress")
        inventory = load("pressure_inventory_process")
        observation = load("pressure_observation")
        actual_ingress = ingress.PressureIngress
        cases = [(layout, False) for layout in six.LAYOUTS] + [("L2C", True)]
        for layout, health in cases:
            with (
                self.subTest(layout=layout, health=health),
                tempfile.TemporaryDirectory() as folder,
            ):
                root = Path(folder)
                plan = canonical_plan(root, layout)
                directory = plan["run_dir"]
                directory.mkdir(parents=True)
                manifest = plan["instances"]
                names = [row["instance_id"] for row in manifest["instances"]]
                args = NS(
                    run_dir=directory,
                    source_inventory=root / "source",
                    target_inventory=root / "target",
                    source_context_limit=32768,
                    target_context_limit=32768,
                    predictor_events=root / "events",
                    predictor_sha256="0" * 64,
                    migration_id="cpu",
                    tick_s=0.005,
                    max_seconds=4,
                    no_progress_s=3,
                    no_migration=True,
                    maximum_migrations=1,
                    routing_strategy="pressure",
                    routing_weights=[1, 1],
                    test_only_cancel_first_shadow=False,
                    scheduler="unified",
                    tp1_max_num_seqs=16,
                    tp2_max_num_seqs=16,
                    health_reference=Path(
                        "experiments/platforms/qwen3_pressure_observation/health_reference.json"
                    ),
                )
                controller = NS(
                    **(
                        plan["config"]
                        | dict(
                            block_size=16,
                            tp1_total_kv_blocks=2001,
                            tp4_total_kv_blocks=2001,
                        )
                    )
                )
                session = NS(
                    sessions=[],
                    record=None,
                    stream=None,
                    target_future=None,
                    settled=True,
                    migration_id="cpu",
                    ready_for_start=True,
                    rotate=lambda: False,
                    cleanup_requires_shutdown=False,
                    cancel_future=None,
                    close=lambda: None,
                )
                jobs = [
                    dict(
                        id=str(n),
                        at_s=n * 0.02,
                        route=name,
                        request=dict(model="cpu", prompt=[1], max_tokens=1),
                    )
                    for n, name in enumerate(names)
                ]
                calls = []

                def post(url, payload, timeout, sink, *args, calls=calls, **kwargs):
                    calls.append(url)
                    started = time.time()
                    sink(0, 42, started)
                    return dict(
                        request_started_unix_s=started, completed_unix_s=time.time()
                    )

                def make_ingress(*args, **kwargs):
                    return actual_ingress(*args, **kwargs, post=post)

                with (
                    patch.dict(sys.modules, {"vllm.bridge_tp.async_journal": journal}),
                    patch(
                        "tools.bridge_tp.pressure_sessions.PressureSessions",
                        return_value=session,
                    ),
                    patch.object(inventory, "ProcessInventory", FakeInventory),
                    patch.object(
                        observation, "scrape_inventory_metrics", return_value=None
                    ),
                    patch.object(ingress, "PressureIngress", side_effect=make_ingress),
                ):
                    pilot._run(
                        args,
                        (
                            dataclasses.replace(config(), health_service=health),
                            controller,
                            NS(release_budget=17),
                            jobs,
                            dict(instance_manifest=manifest),
                        ),
                    )
                results = json.loads(
                    (directory / "pressure_request_results.json").read_text()
                )
                self.assertEqual({row["initial_pool"] for row in results}, set(names))
                self.assertTrue(
                    all(
                        row["source_done"] and not row["source_error"]
                        for row in results
                    )
                )
                audit = [
                    json.loads(line)
                    for line in (directory / "pressure_audit.jsonl")
                    .read_text()
                    .splitlines()
                ]
                scheduler = next(
                    row for row in audit if row["kind"] == "pressure_scheduler"
                )
                self.assertEqual(set(scheduler["slots"]), set(names))
                status = json.loads((directory / "layout_status.json").read_text())
                self.assertEqual(set(status["instances"]), set(names))
                self.assertTrue(
                    all(
                        row["pool_id"] == name
                        for name, row in status["instances"].items()
                    )
                )
