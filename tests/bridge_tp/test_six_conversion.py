# SPDX-License-Identifier: Apache-2.0
"""Canonical six-card engine conversion lifecycle; CPU inventory/process fakes."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp.test_performance_trace import journal
from tests.bridge_tp.test_pressure_conversion import Reader
from tests.bridge_tp.test_pressure_policy import config
from tests.bridge_tp.test_runtime_instance_ids import FakeInventory, canonical_plan
from tests.bridge_tp.test_six_card_gate import snapshot
from tools.bridge_tp import pressure_conversion as pc
from tools.bridge_tp import run_pressure_balanced_pilot as pilot
from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp import six_conversion as six
from tools.bridge_tp.conversion_sequence import validate


class ControllerTests(unittest.TestCase):
    def test_real_loop_handles_dynamic_and_zero_receiver_sets(self):
        ingress = pc.load("pressure_ingress")
        inventory = pc.load("pressure_inventory_process")
        observation = pc.load("pressure_observation")
        actual_ingress = ingress.PressureIngress
        for layout, adding in [(n, False) for n in six.PATH] + [("L1AC", True)]:
            with (
                self.subTest(layout=layout, adding=adding),
                tempfile.TemporaryDirectory() as folder,
            ):
                root = Path(folder)
                plan = plan_at(root, layout)
                directory = plan["run_dir"]
                (directory / "source_active").mkdir(parents=True)
                for name in plan["source_links"]:
                    (directory / "source_active" / name).write_text("CPU entry")
                manifest = plan["instances"]
                argv = plan["controller_argv"]
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
                    no_migration=False,
                    maximum_migrations=0,
                    routing_strategy="pressure",
                    routing_weights=[1, 1],
                    test_only_cancel_first_shadow=False,
                    scheduler="unified",
                    tp1_max_num_seqs=16,
                    tp2_max_num_seqs=16,
                    health_reference=None,
                    migration_targets="all",
                    force_migration_target=None,
                    migration_target=[
                        argv[i + 1]
                        for i, v in enumerate(argv)
                        if v == "--migration-target"
                    ],
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
                # Retire every migration group while keeping serving instances:
                # this exercises the real apply_layout_control reconciliation.
                changes = []
                wanted, links = {}, {}
                retired = set()
                if adding:
                    retired = {
                        r["instance_id"]
                        for r in manifest["instances"]
                        if r["bundle_id"] == "A"
                    }
                    changes = [
                        dict(r, lifecycle="RETIRED")
                        for r in manifest["instances"]
                        if r["instance_id"] in retired
                    ]
                    new = dict(
                        instance_id="A-g100-tp2",
                        tp_degree=2,
                        context_limit=32768,
                        endpoint="http://127.0.0.1:8500",
                        inventory_path=str(directory / "new_inventory.jsonl"),
                        lifecycle="ACTIVE",
                    )
                    changes.append(new)
                    wanted = dict(plan["six_targets"])
                    wanted[new["instance_id"]] = six.receiver(
                        plan["six_state"], new, directory
                    )
                    links = {
                        n: p
                        for n, p in plan["source_links"].items()
                        if n not in retired
                    }
                pc.atomic_json(
                    directory / "layout_control.json",
                    dict(
                        revision=1,
                        instances=changes,
                        conversion_pending=False,
                        migration_resources=dict(targets=wanted, source_links=links),
                    ),
                )
                built = []

                def group(built=built, **kwargs):
                    value = NS(
                        sessions=[],
                        record=None,
                        stream=None,
                        target_future=None,
                        settled=True,
                        migration_id=kwargs["migration_id"],
                        ready_for_start=True,
                        rotate=lambda: False,
                        cleanup_requires_shutdown=False,
                        cancel_future=None,
                        close=lambda: None,
                        root=kwargs["run_dir"],
                        bind_target=lambda *a: None,
                    )
                    built.append(value)
                    return value

                def post(url, payload, timeout, sink, *args, **kwargs):
                    now = time.time()
                    sink(0, 42, now)
                    return dict(
                        request_started_unix_s=now, completed_unix_s=time.time()
                    )

                jobs = [
                    dict(
                        id=str(n),
                        at_s=n * 0.02,
                        route=r["instance_id"],
                        request=dict(model="cpu", prompt=[1], max_tokens=1),
                    )
                    for n, r in enumerate(manifest["instances"])
                    if r["instance_id"] not in retired
                ]
                if adding:
                    jobs.append(
                        dict(
                            id="new",
                            at_s=0.2,
                            route="A-g100-tp2",
                            request=dict(model="cpu", prompt=[1], max_tokens=1),
                        )
                    )
                with (
                    patch.dict(sys.modules, {"vllm.bridge_tp.async_journal": journal}),
                    patch(
                        "tools.bridge_tp.pressure_sessions.PressureSessions",
                        side_effect=group,
                    ),
                    patch.object(inventory, "ProcessInventory", FakeInventory),
                    patch.object(
                        observation, "scrape_inventory_metrics", return_value=None
                    ),
                    patch.object(
                        ingress,
                        "PressureIngress",
                        side_effect=lambda *a, **k: actual_ingress(*a, **k, post=post),
                    ),
                ):
                    pilot._run(
                        args,
                        (
                            config(),
                            controller,
                            NS(release_budget=17),
                            jobs,
                            dict(instance_manifest=manifest),
                        ),
                    )
                status = pc.read_json(directory / "layout_status.json")
                self.assertEqual(set(status["migration_resource_targets"]), set(wanted))
                self.assertEqual(set(status["migration_resource_sources"]), set(links))
                results = pc.read_json(directory / "pressure_request_results.json")
                audit = [
                    json.loads(line)
                    for line in (directory / "pressure_audit.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertFalse(
                    any(e["kind"] == "pressure_actual_start" for e in audit)
                )
                self.assertEqual(len(results), len(jobs))
                self.assertTrue(
                    all(r["source_done"] and not r["source_error"] for r in results)
                )

    def test_cli_accepts_six_edges_only_on_the_matching_platform(self):
        import contextlib
        import io

        for edge, (before, _, _, _) in six.EDGES.items():
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner.main(
                    [
                        "--cards",
                        "6",
                        "--static-layout",
                        before,
                        "--conversion",
                        edge,
                        "--long-burst-pair",
                        "--expanded-context",
                        "--current-machine",
                        "--scheduler",
                        "unified",
                        "--hold-queue",
                        "--kv-short-exempt",
                        "--overload-migrate",
                        "--migrate-current-length",
                        "--repetitions",
                        "1",
                        "--model",
                        "m",
                        "--output",
                        "unused",
                        "--expected-revision",
                        "CPU",
                        "--validate-only",
                    ]
                )
            self.assertEqual(result, 0)


class RunnerVerdictTests(unittest.TestCase):
    """GPU 11-08: a passing L1AC-L2C run ended runner=2 on KeyError('conversion')
    because the four-card and static summaries read receipts it never writes."""

    def args(self, conversion="L1AC-L2C", decider=None):
        return NS(conversion=conversion, conversion_decider=decider)

    def row(self, layout, passed=True, steps=True, status="COMPLETED"):
        return dict(
            status=status,
            conversion_sequence=dict(
                passed=passed,
                all_steps_completed=steps,
                phases=[dict(state="COMPLETED", layout=dict(layout))],
            ),
        )

    def verdict(self, args, ledger, scene=None):
        return runner.six_conversion_result(args, ledger, 1, scene)

    def test_explicit_edge_passes_only_at_its_target_layout(self):
        for edge, (_, after, _, _) in six.EDGES.items():
            result, code = self.verdict(self.args(edge), [self.row(six.PATH[after])])
            self.assertEqual(
                (result, code),
                (dict(status="COMPLETED_CONVERSION", edge=edge, layout=after), 0),
            )
        wrong = self.row(six.PATH["L1AC"])
        self.assertEqual(self.verdict(self.args(), [wrong])[1], 2)

    def test_missing_or_failed_evidence_never_passes(self):
        target = six.PATH["L2C"]
        cases = [
            [dict(status="COMPLETED")],  # no conversion_sequence at all
            [self.row(target, passed=False)],
            [self.row(target, steps=False)],
            [self.row(target, status="FAILED")],
            [],
        ]
        for ledger in cases:
            with self.subTest(ledger=ledger):
                result, code = self.verdict(self.args(), ledger)
                self.assertEqual(
                    (result["status"], code), ("COMPLETED_WITH_COVERAGE_GAPS", 2)
                )

    def test_aborted_conversion_is_reported_as_aborted(self):
        with tempfile.TemporaryDirectory() as folder:
            scene = Path(folder)
            (scene / "conversion_result.json").write_text(
                json.dumps(dict(state="ABORTED"))
            )
            result, code = self.verdict(self.args(), [dict(status="COMPLETED")], scene)
        self.assertEqual((result["status"], code), ("CONVERSION_ABORTED_SAFELY", 2))

    def test_decider_runs(self):
        execute = self.args(conversion=None, decider="execute")
        self.assertEqual(
            self.verdict(execute, [self.row(six.PATH["L2C"])]),
            (dict(status="COMPLETED_CONVERSION_SEQUENCE"), 0),
        )
        self.assertEqual(self.verdict(execute, [dict(status="COMPLETED")])[1], 2)
        record = self.args(conversion=None, decider="record")
        self.assertEqual(
            self.verdict(record, [dict(status="COMPLETED")]),
            (dict(status="COMPLETED_CONVERSION_DECIDER_RECORD"), 0),
        )

    def test_four_card_and_static_summaries_skip_six_card_conversions(self):
        """Their receipts do not exist for these runs; reading them raised."""
        import inspect

        source = inspect.getsource(runner.main)
        for guard in (
            "if args.conversion and not six_conversion:",
            "if args.static_layout and not six_conversion:",
            "if (args.conversion_cycle or args.decider_multi) and not six_conversion:",
            "result, code = six_conversion_result(",
        ):
            self.assertIn(guard, source)
        # The six-card verdict is the last one written.
        self.assertGreater(
            source.index("result, code = six_conversion_result("),
            source.index("if args.static_layout and not six_conversion:"),
        )


def plan_at(root, layout):
    plan = canonical_plan(root, layout)
    plan["controller_argv"] += ["--maximum-migrations", "0"]
    plan["stager_argv"][1] = "tools/bridge_tp/pressure_sessions.py"
    six.configure(plan)
    return plan


class ChannelDiagnosticTests(unittest.TestCase):
    def backend(self, root, now):
        backend = object.__new__(six.SixBackend)
        backend.targets = {
            "B-g2-tp2": dict(
                mechanism_root=str(root),
                connector_sources=[
                    dict(
                        source="A-g5-tp1",
                        base_port=34100,
                        generation=1005,
                    )
                ],
            )
        }
        backend.joined_at = 0
        backend.clock = lambda: now
        backend.channel_graph = NS(acknowledged=lambda: True)
        return backend

    def test_warmup_timeout_includes_the_target_rank_connection_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = (
                root / "sessions/s1/gpu_channel_connection_receipts/tp_rank_1-1005.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    dict(
                        target_tp_rank=1,
                        source_instance="A-g5-tp1",
                        base_port=34100,
                        channel_generation=1005,
                        status="FAILED",
                        stage="bind",
                        last_error="OSError('bind unavailable')",
                        updated_unix_s=30,
                    )
                )
            )
            with (
                patch.object(six.SequenceBackend, "warm_and_verify", return_value=True),
                self.assertRaisesRegex(TimeoutError, "bind unavailable") as raised,
            ):
                six.SixBackend.warm_and_verify(self.backend(root, 781), {})
            self.assertIn('"generation": 1005', str(raised.exception))
            self.assertIn('"target_tp_rank": 1', str(raised.exception))

    def test_diagnostic_complete_never_substitutes_for_warmup_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = (
                root / "sessions/s1/gpu_channel_connection_receipts/tp_rank_1-1005.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(status="COMPLETE", updated_unix_s=20)))
            backend = self.backend(root, 30)
            with patch.object(
                six.SequenceBackend, "warm_and_verify", return_value=True
            ):
                self.assertFalse(six.SixBackend.warm_and_verify(backend, {}))
                receipts = root / "sessions/s1/gpu_channel_preconnect_receipts"
                receipts.mkdir()
                for name in (
                    "source-A-g5-tp1-B-g2-tp2.json",
                    "tp_rank_0-A-g5-tp1.json",
                    "tp_rank_1-A-g5-tp1.json",
                ):
                    (receipts / name).write_text(
                        json.dumps(
                            dict(
                                status="WARMUP_COMPLETE",
                                channel_generation=1005,
                            )
                        )
                    )
                self.assertTrue(six.SixBackend.warm_and_verify(backend, {}))


class PlansTests(unittest.TestCase):
    def test_initial_peer_plans_share_the_reallocated_port_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            original = six.ports.probe_block

            def occupied(first, width):
                if first == 18000:
                    return False, dict(port=first, error="occupied")
                return original(first, width)

            with (
                patch.object(
                    six.ports,
                    "state",
                    return_value=dict(
                        next_port=18000,
                        port_policy=dict(
                            ephemeral_range=[32768, 60999], reserved_ports=""
                        ),
                    ),
                ),
                patch.object(six.ports, "probe_block", side_effect=occupied),
            ):
                plan = plan_at(Path(folder), "L2C")
            blocks = plan["migration_plan"]["port_blocks"]
            self.assertNotIn(18000, [b["first"] for b in blocks])
            self.assertEqual(plan["six_state"]["port_probe_failures"][0]["port"], 18000)
            for source in plan["source_links"]:
                entries = six.source_entries(source, plan["six_targets"])
                for entry in entries:
                    target = plan["six_targets"][entry["target"]]
                    expected = next(
                        c for c in target["connector_sources"] if c["source"] == source
                    )
                    self.assertEqual(
                        (entry["base_port"], entry["generation"]),
                        (expected["base_port"], expected["generation"]),
                    )

    def test_all_layouts_have_owned_full_mesh_and_distinct_ports(self):
        for layout in six.PATH:
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as folder:
                plan = plan_at(Path(folder), layout)
                sources = plan["source_links"]
                targets = plan["six_targets"]
                self.assertEqual(
                    len(sources), 2 * sum(v == 1 for v in six.PATH[layout].values())
                )
                self.assertEqual(
                    len(targets), sum(v == 2 for v in six.PATH[layout].values())
                )
                occupied = set()
                for target in targets.values():
                    for key in (
                        "ready_notification_port",
                        "delta_base_port",
                        "delivery_base_port",
                    ):
                        port = target[key]
                        self.assertTrue(occupied.isdisjoint(range(port, port + 10)))
                        occupied.update(range(port, port + 10))
                    self.assertEqual(
                        {r["source"] for r in target["connector_sources"]}, set(sources)
                    )
                    for channel in target["connector_sources"]:
                        port = channel["base_port"]
                        self.assertTrue(occupied.isdisjoint(range(port, port + 10)))
                        occupied.update(range(port, port + 10))
                    self.assertEqual(
                        Path(target["channel_plan"]).parent,
                        Path(target["mechanism_root"]),
                    )
                self.assertTrue(plan["config"]["dynamic_conversion"])
                self.assertNotIn("--no-migration", plan["controller_argv"])

    def test_owned_entries_initialize_disarmed_and_cleanup_without_symlinks(self):
        with tempfile.TemporaryDirectory() as probe:
            try:
                (Path(probe) / "link").symlink_to(
                    Path(probe) / "target", target_is_directory=True
                )
            except OSError as error:
                if getattr(error, "winerror", None) == 1314:
                    self.skipTest(
                        "Windows symlink privilege unavailable; Linux required"
                    )
                raise
        for layout in six.PATH:
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as folder:
                plan = plan_at(Path(folder), layout)
                plan["run_dir"].mkdir(parents=True)
                try:
                    six.initialize(plan)
                except OSError as error:
                    if getattr(error, "winerror", None) == 1314:
                        self.skipTest(
                            "Windows symlink privilege unavailable; Linux required"
                        )
                    raise
                for source, link in plan["source_links"].items():
                    self.assertTrue(Path(link).is_symlink())
                    self.assertFalse(
                        pc.read_json(Path(link).resolve() / "runtime_control.json")[
                            "armed"
                        ]
                    )
                    self.assertEqual(
                        len(
                            pc.read_json(plan["run_dir"] / f"gpu_targets_{source}.json")
                        ),
                        len(plan["six_targets"]),
                    )
                if layout == "L2C":
                    from tests.bridge_tp.test_multi_target import Audit, Group
                    from tools.bridge_tp.pressure_sessions import TargetSessions

                    groups = {
                        n: Group(Path(t["mechanism_root"]), n)
                        for n, t in plan["six_targets"].items()
                    }
                    manager = TargetSessions(
                        groups, plan["source_links"], Audit(), dynamic=True
                    )
                    retained = {n: g for n, g in groups.items() if n.startswith("A-")}
                    self.assertTrue(manager.reconcile(retained, plan["source_links"]))
                    for link in plan["source_links"].values():
                        control = pc.read_json(
                            Path(link).resolve() / "runtime_control.json"
                        )
                        self.assertEqual(
                            control["target_instance"], next(iter(retained))
                        )
                        self.assertFalse(control["armed"])
                six.cleanup(plan, pc.atomic_json)
                self.assertFalse(
                    any(p.is_symlink() for p in plan["run_dir"].rglob("*"))
                )

    def test_profile_validation_accepts_c_and_rejects_assistance(self):
        step = dict(bundle="C", target_mode="TP1", trigger_s=1, deadline_s=30)
        self.assertEqual(
            validate(dict(profile="six_card", base_port=8400, sequence=[step]))[
                "sequence"
            ],
            [step],
        )
        with self.assertRaises(ValueError):
            validate(dict(base_port=8400, sequence=[step]))
        with self.assertRaises(ValueError):
            validate(
                dict(
                    profile="six_card",
                    base_port=8400,
                    sequence=[dict(step, target_mode="TP2", method="SELECTIVE_ROLL")],
                )
            )

    def test_new_receiver_argv_uses_own_plan_and_mechanism(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_at(Path(folder), "L2C")
            for row in plan["extra_engines"]:
                if row["tp"] != 2:
                    continue
                argv, env = row["receiver_argv"], row["receiver_env"]
                settings = json.loads(argv[argv.index("--kv-transfer-config") + 1])[
                    "kv_connector_extra_config"
                ]
                target = plan["six_targets"][row["instance_id"]]
                self.assertEqual(
                    settings["bridgetp_gpu_direct_channel_plan"], target["channel_plan"]
                )
                self.assertEqual(
                    env["BRIDGETP_TAKEOVER_RUN_DIR"], target["mechanism_dir"]
                )


class LifecycleTests(unittest.TestCase):
    def run_edge(
        self,
        edge,
        cancel_after_retire=False,
        prestart=False,
        cancel_before_retire=False,
    ):
        before, after, bundle, mode = six.EDGES[edge]
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_at(Path(folder), before)
            run = plan["run_dir"]
            run.mkdir(parents=True)
            calls, events = [], []
            alive = set()

            def start(extra):
                calls.append(("start", extra["instance_id"]))
                alive.add(extra["instance_id"])

            def stop(ids):
                calls.append(("stop", ids))
                alive.difference_update(ids)

            for target in plan["six_targets"].values():
                Path(target["mechanism_root"]).mkdir(parents=True)
                pc.atomic_json(target["channel_plan"], target["connector_sources"])
            for source in plan["source_links"]:
                pc.atomic_json(
                    run / f"gpu_targets_{source}.json",
                    six.source_entries(source, plan["six_targets"]),
                )
            ops = NS(
                event=lambda kind, **v: events.append((kind, v)),
                stop=stop,
                start=start,
                start_receiver=lambda target: calls.append(
                    ("stager", target["instance_id"])
                ),
                alive=lambda name: name in alive,
                gpus_released=lambda ids: True,
                warm=lambda extra: None,
                scrape=lambda endpoint, native: dict(
                    num_running=0,
                    num_waiting=0,
                    free_kv_blocks=native["free_blocks"],
                    sampled_unix_s=time.time(),
                ),
            )
            backend = six.SixBackend(
                plan["registry"],
                plan["instances"],
                run,
                ops,
                base_port=8400,
                context_limits=(32768, 32768),
                sequence_context={},
                resources=plan,
                prestart=prestart,
            )
            step = dict(bundle=bundle, target_mode=mode)
            pc.atomic_json(
                run / "layout_status.json", dict(unix_s=time.time(), instances={})
            )
            backend.begin_step("edge", step)
            executor = pc.conversion.ConversionExecutor(
                backend.registry, backend, run / "plan.json", clock=time.time
            )
            executor.plan_create(
                "edge",
                backend.registry.registry_revision,
                bundle,
                mode,
                "DRAIN",
                "ACTIVE",
                time.time() + 10,
                "NO_PROFILE",
            )

            def refresh():
                now = time.time()
                resources = pc.read_json(run / "layout_control.json")[
                    "migration_resources"
                ]
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
                    for rank in range(2):
                        receipts = (
                            Path(target["mechanism_root"])
                            / "sessions/cpu/gpu_channel_plan_receipts"
                        )
                        receipts.mkdir(parents=True, exist_ok=True)
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
                    for c in target["connector_sources"]:
                        for name in [
                            f"source-{c['source']}-{target['instance_id']}.json"
                        ] + [f"tp_rank_{r}-{c['source']}.json" for r in range(2)]:
                            pc.atomic_json(
                                pre / name,
                                dict(
                                    status="WARMUP_COMPLETE",
                                    channel_generation=c["generation"],
                                ),
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
                    refresh()
                    executor.plan_execute(time.time())
                    if executor.plan["state"] == "PREPARING":
                        backend.write_layout({})
                        pending = pc.read_json(run / "layout_control.json")[
                            "migration_resources"
                        ]
                        self.assertFalse(
                            set(backend.started)
                            & (set(pending["targets"]) | set(pending["source_links"]))
                        )
                    if cancel_before_retire and executor.plan["state"] == "DRAINING":
                        reserved = {
                            n for n, _, _, _ in backend.replacement_specs(executor.plan)
                        }
                        self.assertTrue(executor.plan_cancel())
                        self.assertEqual(executor.plan["state"], "ABORTED")
                        self.assertFalse(reserved & alive)
                        backend.begin_step("retry", step)
                        other = {
                            n for n, _, _, _ in backend.replacement_specs(executor.plan)
                        }
                        self.assertTrue(reserved.isdisjoint(other))
                        return
                    if cancel_after_retire and any(
                        backend.layout.get(r.instance_id, {}).get("lifecycle")
                        == "RETIRED"
                        for r in backend.refs(executor.plan)
                    ):
                        self.assertFalse(executor.plan_cancel())
                        self.assertEqual(executor.plan["state"], "RECOVERY_REQUIRED")
                        self.assertFalse(any(k == "stop" for k, _ in calls))
                        return
                    time.sleep(0.002)
            self.assertEqual(executor.plan["state"], "COMPLETED", executor.plan)
            refresh()
            self.assertTrue(backend.step_acknowledged())
            self.assertEqual(
                {r.bundle_id: r.tp_degree for r in backend.active().values()},
                six.PATH[after],
            )
            self.assertTrue(any(k == "stop" for k, _ in calls))
            self.assertEqual(len(backend.started), 2 if mode == "TP1" else 1)
            kinds = [k for k, _ in events]
            self.assertLess(
                kinds.index("CHANNEL_GRAPH_RETIRED"), kinds.index("STOP_RETURN")
            )
            self.assertLess(
                kinds.index("RESOURCES_RELEASED"), kinds.index("LOAD_START")
            )
            self.assertLess(kinds.index("VERIFIED"), kinds.index("ROUTE_PUBLISHED"))

    def test_gated_prestart_merge_and_split_follow_the_same_release_order(self):
        self.run_edge("L1AC-L2C", prestart=True)
        self.run_edge("L2C-L1AC", prestart=True)

    def test_cancelled_prestart_does_not_reuse_instance_identity(self):
        self.run_edge("L1AC-L2C", prestart=True, cancel_before_retire=True)

    def test_abort_cannot_revive_logically_retired_session_identity(self):
        self.run_edge("L1AC-L0", cancel_after_retire=True)

    def test_l0_to_l1ac(self):
        self.run_edge("L0-L1AC")

    def test_l1ac_to_l0(self):
        self.run_edge("L1AC-L0")

    def test_l1ac_to_l2c(self):
        self.run_edge("L1AC-L2C")

    def test_l2c_to_l1ac(self):
        self.run_edge("L2C-L1AC")

    def test_l2c_to_l3(self):
        self.run_edge("L2C-L3")

    def test_l3_to_l2c(self):
        self.run_edge("L3-L2C")
