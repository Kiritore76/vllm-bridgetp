# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for TP2 geometry, atomic commitments and rolling conversion."""

import ast
import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from tools.bridge_tp.pressure_cpu import load

pool = load("instance_pool")
planner = load("reconfiguration_planner")
executor = load("reconfiguration_executor")
geometry = load("tp_geometry")
ROOT = Path(__file__).resolve().parents[2]


def record(name, tp, gpus, bundle="A", generation=1):
    return pool.InstanceRecord(
        pool.InstanceRef(name, generation),
        tp,
        tuple(gpus),
        bundle,
        "http://local",
        "model/config/profile",
        8192,
    )


def active_registry():
    registry = pool.Registry()
    for row in (
        record("s1", 1, ["0"]),
        record("s2", 1, ["1"]),
        record("t", 2, ["2", "3"], "B"),
    ):
        registry.register(row)
        registry.transition(row.ref, "ACTIVE")
    return registry


class GeometryTests(unittest.TestCase):
    def test_tp2_and_tp4_layouts_own_exact_physical_budget(self):
        for k in (2, 4):
            layouts = pool.layouts(k, [str(i) for i in range(k * 2)])
            self.assertEqual(set(layouts), {"L0", "L1A", "L1B", "L2"})
            for rows in layouts.values():
                gpus = [gpu for row in rows for gpu in row["gpu_ids"]]
                self.assertEqual(len(gpus), len(set(gpus)))
                self.assertEqual(len(gpus), 2 * k)

    def test_two_bundle_layouts_keep_historical_names_and_order(self):
        layouts = pool.layouts(2, ["2", "3", "0", "1"])

        def shape(label):
            return [
                (r["bundle_id"], r["tp_degree"], r["gpu_ids"]) for r in layouts[label]
            ]

        self.assertEqual(
            shape("L0"),
            [("A", 1, ["2"]), ("A", 1, ["3"]), ("B", 1, ["0"]), ("B", 1, ["1"])],
        )
        self.assertEqual(
            shape("L1A"), [("A", 1, ["2"]), ("A", 1, ["3"]), ("B", 2, ["0", "1"])]
        )
        self.assertEqual(
            shape("L1B"), [("A", 2, ["2", "3"]), ("B", 1, ["0"]), ("B", 1, ["1"])]
        )
        self.assertEqual(shape("L2"), [("A", 2, ["2", "3"]), ("B", 2, ["0", "1"])])

    def test_any_bundle_count_enumerates_every_assignment(self):
        layouts = pool.layouts(2, [str(i) for i in range(6)])
        self.assertEqual(len(layouts), 8)
        self.assertEqual(
            set(layouts),
            {"L0", "L1AB", "L1AC", "L1BC", "L2A", "L2B", "L2C", "L3"},
        )
        self.assertEqual(
            [(r["bundle_id"], r["tp_degree"]) for r in layouts["L2B"]],
            [("A", 2), ("B", 1), ("B", 1), ("C", 2)],
        )
        for rows in layouts.values():
            gpus = [gpu for row in rows for gpu in row["gpu_ids"]]
            self.assertEqual(sorted(gpus), [str(i) for i in range(6)])
        self.assertEqual(set(pool.layouts(2, ["0", "1"])), {"L0", "L1"})
        for bad in (["0", "1", "2"], ["0", "0", "1", "2"], []):
            with self.assertRaises(ValueError):
                pool.layouts(2, bad)

    def test_fixed_pair_registry_aliases_historical_pools(self):
        registry, aliases = pool.fixed_pair_registry(
            source_gpu="2",
            target_gpus=["0", "1"],
            source_url="http://127.0.0.1:8001",
            target_url="http://127.0.0.1:8200",
            identity="model",
            source_context_limit=32768,
            target_context_limit=32768,
        )
        self.assertEqual(aliases, {"tp1": "A-g1-tp1", "tp4": "B-g2-tp2"})
        manifest = pool.registry_manifest(registry, aliases)
        self.assertEqual(manifest["registry_revision"], 2)
        self.assertEqual(
            [
                (
                    r["instance_id"],
                    r["tp_degree"],
                    r["physical_gpu_ids"],
                    r["legacy_pool"],
                )
                for r in manifest["instances"]
            ],
            [("A-g1-tp1", 1, ["2"], "tp1"), ("B-g2-tp2", 2, ["0", "1"], "tp4")],
        )
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)
        self.assertEqual(pool.resolve_pool(manifest, "tp4"), "B-g2-tp2")
        self.assertEqual(pool.resolve_pool(manifest, "A-g1-tp1"), "A-g1-tp1")
        with self.assertRaises(KeyError):
            pool.resolve_pool(manifest, "tp2")
        with self.assertRaises(ValueError):
            pool.registry_manifest(registry, {"tp1": "A-g1-tp1", "x": "A-g1-tp1"})
        with self.assertRaises(ValueError):
            pool.fixed_pair_registry(
                source_gpu="0",
                target_gpus=["0", "1"],
                source_url="s",
                target_url="t",
                identity="model",
                source_context_limit=1,
                target_context_limit=1,
            )

    def test_session_explicit_tp2_and_legacy_tp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(geometry.session_target_tp(tmp), 4)
            path = Path(tmp) / "session_manifest.json"
            path.write_text(json.dumps(dict(target_tp_size=2)))
            self.assertEqual(geometry.session_target_tp(tmp), 2)
            path.write_text(json.dumps(dict(target_tp_size=3)))
            with self.assertRaises(ValueError):
                geometry.session_target_tp(tmp)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.r = active_registry()
        self.s1, self.s2, self.t = [self.r.instances[k].ref for k in ("s1", "s2", "t")]

    def reserve(self, sid, req, source, blocks):
        return self.r.try_reserve(
            sid,
            req,
            source,
            self.t,
            target_blocks=blocks,
            free_blocks=10,
            sample_id=1,
            now=10,
            sampled_at=10,
        )

    def test_two_sources_cannot_oversubscribe_and_old_inventory_retains_promise(self):
        self.assertIsNotNone(self.reserve("m1", "r1", self.s1, 7))
        self.assertIsNone(self.reserve("m2", "r2", self.s2, 7))
        self.r.materialize("m1", self.t, 3)
        self.r.reconcile_inventory("m1", self.t, 2, True)
        self.assertEqual(self.r.commitments["m1"].state, "ACKED")
        self.assertIsNone(self.reserve("m2", "r2", self.s2, 7))
        self.r.reconcile_inventory("m1", self.t, 3, True)
        self.assertEqual(self.r.commitments["m1"].state, "INVENTORY")

    def test_quiesce_dispatch_race_and_uncertain_no_resubmit(self):
        self.r.acquire_dispatch_lease(self.s1, "r1")
        self.r.transition(self.s1, "QUIESCING")
        self.r.reconcile_dispatch(self.s1, "r1", "UNCERTAIN")
        with self.assertRaises(ValueError):
            self.r.acquire_dispatch_lease(self.t, "r1")
        with self.assertRaises(ValueError):
            self.r.acquire_dispatch_lease(self.s1, "new")
        with self.assertRaises(ValueError):
            self.r.transition(self.s1, "RETIRED")

    def test_cancel_is_not_free_and_stale_generation_does_not_release(self):
        self.reserve("m", "r", self.s1, 3)
        with self.assertRaises(ValueError):
            self.r.release_after_evidence(
                "m", self.t, remote_free=False, fresh_absence=True
            )
        with self.assertRaises(ValueError):
            self.r.release_after_evidence(
                "m",
                replace(self.t, instance_generation=2),
                remote_free=True,
                fresh_absence=True,
            )
        self.assertEqual(self.r.commitments["m"].state, "RESERVED")

    def test_generation_of_retained_bundle_is_unchanged(self):
        for ref in (self.s1, self.s2):
            self.r.transition(ref, "QUIESCING")
            self.r.transition(ref, "RETIRED")
        self.r.register(record("merged", 2, ["0", "1"]))
        self.assertEqual(self.r.get(self.t).ref, self.t)


class PlannerTests(unittest.TestCase):
    def test_delayed_transfer_outside_measured_support_is_unknown(self):
        rows = [replace(r, transfer_supported_delay_s=0) for r in self.requests()]
        result = planner.timeline(rows, ["D", "C"], 7)
        self.assertEqual(result["support_status"], "UNKNOWN")
        self.assertIsNone(result["estimated_ready_time"])

    def requests(self):
        return [
            planner.RemainingRequest(
                name,
                duration,
                ((duration, 1.0),),
                True,
                13,
                100,
                transfer_supported_delay_s=300,
            )
            for name, duration in zip("ABCD", (8, 32, 60, 95))
        ]

    def test_required_serial_fixture(self):
        rows = self.requests()
        for order, expected in (([], 102), (["D"], 67), (["D", "C"], 39)):
            result = planner.timeline(rows, order, 7)
            self.assertEqual(result["estimated_ready_time"], expected)
        result = planner.timeline(rows, ["D", "C"], 7)
        self.assertEqual(result["timeline"][1]["start_s"], 13)

    def plan(self, rows, **kwargs):
        mode = kwargs.pop("objective_mode", "MECHANISM_ONLY")
        manual = planner.ManualPlan("p", "A", 2, mode, "SHADOW", 100, 300)
        return planner.plan_next(
            manual,
            snapshot_id="s",
            now=13,
            requests=rows,
            rebuild_s=7,
            profile_hashes={"test": "fixture"},
            resources_safe=True,
            **kwargs,
        )

    def test_roll_removes_unstarted_c_without_undoing_d(self):
        # D has REALLY committed and freed source; it is absent from R_b.
        rows = [
            replace(
                r,
                natural_s=6 if r.request_id == "C" else r.natural_s - 13,
                duration_pmf=(
                    (6 if r.request_id == "C" else max(0, r.natural_s - 13), 1.0),
                ),
            )
            for r in self.requests()
            if r.request_id in "BC"
        ]
        wait = planner.timeline(rows, [], 7)
        move_c = planner.timeline(rows, ["C"], 7)
        self.assertEqual(wait["estimated_ready_time"], 26)
        self.assertEqual(move_c["estimated_ready_time"], 26)
        self.assertGreater(move_c["transfer_bytes"], wait["transfer_bytes"])
        points, candidates = planner.candidates(rows, 7)
        utilities = {
            tuple(r["ordered_migration_set"]): dict(
                delta_g=1, C_pre=r["transfer_bytes"], C_post=0
            )
            for r in candidates
        }
        result = self.plan(rows, objective_mode="NET_UTILITY", utilities=utilities)
        self.assertEqual(result["ordered_migration_set"], [])

    def test_unknown_nonmigratable_request_blocks_complete_estimate(self):
        rows = self.requests() + [
            planner.RemainingRequest("queued", None, None, False, None, None)
        ]
        result = self.plan(rows)
        self.assertEqual(result["support_status"], "UNKNOWN")
        self.assertIsNone(result["estimated_ready_time"])
        self.assertIsNone(planner.remaining_duration(100, 20, None))

    def test_missing_net_does_not_degrade_to_mechanism(self):
        result = self.plan(self.requests(), objective_mode="NET_UTILITY")
        self.assertEqual(result["support_status"], "UNKNOWN")
        self.assertEqual(result["decision"], "HOLD")
        self.assertIsNone(result["J"])

    def test_deadline_and_destroyed_boundary(self):
        manual = planner.ManualPlan("p", "A", 2, "MECHANISM_ONLY", "ACTIVE", 10, 300)
        result = planner.plan_next(
            manual,
            snapshot_id="s",
            now=11,
            requests=[],
            rebuild_s=7,
            profile_hashes={},
            resources_safe=True,
        )
        self.assertEqual(result["decision"], "PLAN_ABORT")
        result = planner.plan_next(
            manual,
            snapshot_id="s",
            now=11,
            requests=[],
            rebuild_s=7,
            profile_hashes={},
            resources_safe=True,
            destroyed=True,
        )
        self.assertEqual(result["decision"], "RECOVERY_REQUIRED")


class ExecutorTests(unittest.TestCase):
    def test_shadow_does_not_mutate_admission_or_engines(self):
        r, backend = active_registry(), Mock()
        with tempfile.TemporaryDirectory() as tmp:
            ex = executor.ConversionExecutor(r, backend, Path(tmp) / "state.json")
            ex.plan_create(
                "p", r.registry_revision, "A", "TP2", "DRAIN", "SHADOW", 100, "fixture"
            )
            ex.plan_execute(10)
            backend.quiesce.assert_not_called()
            backend.stop_owned_engines.assert_not_called()
            self.assertEqual(r.instances["s1"].lifecycle, "ACTIVE")

    def evidence(self, ref, **overrides):
        values = dict(
            instance_id=ref.instance_id,
            instance_generation=ref.instance_generation,
            sampled_at=10,
            running=0,
            waiting=0,
            submissions=0,
            dispatch_leases=0,
            migration_leases=0,
            sessions=0,
            shadows=0,
            commitments=0,
            ownership_pending=0,
            output_tails=0,
            streams_pending=0,
            channel_retirable=True,
        )
        return executor.DrainEvidence(**dict(values, **overrides))

    def test_all_drain_dimensions_required(self):
        ref = pool.InstanceRef("s", 1)
        self.assertTrue(self.evidence(ref).drained(ref, 10))
        for key in (
            "waiting",
            "submissions",
            "output_tails",
            "streams_pending",
            "commitments",
            "dispatch_leases",
        ):
            self.assertFalse(self.evidence(ref, **{key: None}).drained(ref, 10))
        self.assertFalse(self.evidence(ref).drained(ref, 13))

    def test_live_step_order_and_no_fake_rollback_after_stop(self):
        r = active_registry()
        backend = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            ex = executor.ConversionExecutor(r, backend, Path(tmp) / "state.json")
            ex.plan_create(
                "p", r.registry_revision, "A", "TP2", "DRAIN", "ACTIVE", 100, "fixture"
            )
            backend.precheck.return_value = True
            ex.plan_execute(10)
            backend.kv_emergency.return_value = False
            backend.drain_evidence.return_value = [
                self.evidence(ref) for ref in ex.refs()
            ]
            backend.fence.return_value = True
            self.assertEqual(ex.plan_execute(10)["state"], "FENCED")
            backend.retire_channels.return_value = True
            self.assertEqual(ex.plan_execute(10)["state"], "RECLAIMING")
            self.assertFalse(ex.plan_cancel())
            self.assertEqual(ex.plan_status()["state"], "RECOVERY_REQUIRED")
            backend.restore_admission.assert_not_called()
            backend.stop_owned_engines.assert_called_once()


class TargetRankReceiptTests(unittest.TestCase):
    def test_state_machine_requires_exact_tp2_rank_set(self):
        machine = load("state_machine").MigrationStateMachine(lambda row: None)
        row = machine.create("m", "r", expected_ranks=2)
        machine.mark_rank_ready("m", 0)
        self.assertFalse(row.all_ranks_ready)
        with self.assertRaises(ValueError):
            machine.mark_rank_ready("m", 2)
        machine.mark_rank_ready("m", 1)
        self.assertTrue(row.all_ranks_ready)

    def test_actual_adapter_requires_all_configured_ranks(self):
        source = ROOT / "vllm/bridge_tp/controller/action_adapter.py"
        tree = ast.parse(source.read_text())
        nodes = [
            n for n in tree.body if not isinstance(n, ast.ImportFrom) or not n.level
        ]
        module = types.ModuleType("tp2_adapter_cpu_test")
        sys.modules[module.__name__] = module
        module.__dict__.update(
            session_target_tp=geometry.session_target_tp, RuntimeControl=Mock()
        )
        exec(
            compile(ast.Module(nodes, type_ignores=[]), str(source), "exec"),
            module.__dict__,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            session = dict(
                target_tp_size=2,
                migration_id="m",
                session_token="s",
                source_request_id="r",
            )
            write(root / "session_manifest.json", session)
            adapter = module.ActionAdapter("http://unused", root)
            for rank in range(2):
                row = dict(
                    migration_id="m",
                    status="INITIAL_HISTORY_GPU_RESIDENT",
                    exact_readback=True,
                    end_token=128,
                )
                if rank == 1:
                    self.assertFalse(adapter.poll_initial_history_gpu_ready()[0])
                write(root / "gpu_initial_receipts" / f"tp_rank_{rank}.json", row)
            self.assertTrue(adapter.poll_initial_history_gpu_ready()[0])
            write(root / "session_manifest.json", dict(session, target_tp_size=4))
            self.assertFalse(adapter.poll_initial_history_gpu_ready()[0])


class CalibrationTests(unittest.TestCase):
    def test_tp2_collection_is_bounded_and_no_tp4_runtime_profile_is_imported(self):
        from tools.bridge_tp.run_tp2_calibration import matrix, prepare

        self.assertEqual(len(matrix()) * 2, 10)
        spec = prepare(Path("/model"))
        self.assertEqual(spec["server_settings"]["target_tp_size"], 2)
        self.assertEqual(spec["server_settings"]["tp4_gpus"], "0,1")
        self.assertEqual(spec["server_settings"]["tp1_gpu"], "2")

    def test_window_scoring_preserves_boundary_gap(self):
        from tools.bridge_tp.summarize_tp2_calibration import window_score

        value = window_score([[0, 0.05, 2.05, 2.1]], 2, 3)
        self.assertEqual(value["raw_s"], 2)
        self.assertLess(value["good_s"], value["raw_s"])


class LiveDrainBackendTests(unittest.TestCase):
    def test_native_empty_requires_matching_fresh_metrics_and_no_pending_stream(self):
        from tools.bridge_tp import run_pressure_calibration as collection
        from tools.bridge_tp import run_tp_bundle_drain as driver

        r = active_registry()
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(out_root=Path(tmp), base_port=8500)
            backend = driver.NativeDrainBackend(args, r, Mock(), {})
            ref = r.instances["s1"].ref
            backend.inventories["s1"] = Path(tmp) / "native.jsonl"
            plan = dict(refs=[driver.as_ref(ref)])
            native = dict(
                sampled_at=1,
                requests=[],
                running=0,
                waiting=0,
                block_size=16,
                total_blocks=101,
                usage_denominator_blocks=100,
                free_blocks=100,
            )
            metrics = types.SimpleNamespace(
                num_running=0, num_waiting=0, free_kv_blocks=100, sampled_unix_s=10
            )
            scraper = Mock()
            scraper.scrape.return_value = metrics
            try:
                with (
                    patch.object(collection, "latest_inventory", return_value=native),
                    patch.object(
                        load("telemetry"), "MetricsScraper", return_value=scraper
                    ),
                ):
                    self.assertTrue(backend.drain_evidence(plan)[0].drained(ref, 10))
                    metrics.free_kv_blocks = 99
                    self.assertFalse(backend.drain_evidence(plan)[0].drained(ref, 10))
            finally:
                backend.executor.shutdown()


class DrainClockTests(unittest.TestCase):
    """Evidence scraped after the decision time must be judged when in hand."""

    def run_drain(self, clock, sampled_at, now=100.0):
        r = active_registry()
        backend = Mock()
        backend.precheck.return_value = True
        backend.kv_emergency.return_value = False
        backend.fence.return_value = True
        with tempfile.TemporaryDirectory() as tmp:
            ex = executor.ConversionExecutor(
                r, backend, Path(tmp) / "state.json", clock=clock
            )
            ex.plan_create(
                "p", r.registry_revision, "A", "TP2", "DRAIN", "ACTIVE", 1e9, "fixture"
            )
            ex.plan_execute(now)
            backend.drain_evidence.return_value = [
                ExecutorTests().evidence(ref, sampled_at=sampled_at)
                for ref in ex.refs()
            ]
            return ex.plan_execute(now)

    def test_sample_after_decision_passes_with_post_io_clock(self):
        status = self.run_drain(lambda: 100.2, 100.1)
        self.assertEqual(status["state"], "FENCED")
        check = status["last_drain_check"]
        self.assertEqual((check["decided_at"], check["evaluated_at"]), (100.0, 100.2))
        self.assertTrue(check["complete"])

    def test_pre_io_decision_time_reproduces_the_old_rejection(self):
        status = self.run_drain(None, 100.1)
        self.assertEqual(status["state"], "DRAINING")
        self.assertFalse(status["last_drain_check"]["complete"])

    def test_freshness_and_future_bounds_are_not_relaxed(self):
        self.assertEqual(self.run_drain(lambda: 103.0, 100.1)["state"], "DRAINING")
        self.assertEqual(self.run_drain(lambda: 100.2, 100.5)["state"], "DRAINING")
        self.assertEqual(self.run_drain(lambda: 99.0, 98.9)["state"], "DRAINING")


class DrainDriverTests(unittest.TestCase):
    def backend(self, tmp, **extra):
        from tools.bridge_tp import run_tp_bundle_drain as driver

        args = types.SimpleNamespace(
            out_root=Path(tmp),
            base_port=8700,
            model_path=Path(tmp) / "model",
            platform="qwen3",
            startup_timeout_s=1,
            **extra,
        )
        common = Mock()
        common.server_command.return_value = ["server"]
        common.write_json.side_effect = lambda p, r: p.write_text(json.dumps(r))
        inputs = {"requests": {"2048": {"prompt": [1] * 2048}}}
        return (
            driver,
            driver.NativeDrainBackend(args, pool.Registry(), common, inputs),
            common,
        )

    def test_expanded_context_sets_32k_for_every_engine(self):
        from unittest.mock import PropertyMock

        with tempfile.TemporaryDirectory() as tmp:
            driver, backend, common = self.backend(tmp, expanded_context=True)
            try:
                with patch.object(
                    driver.NativeDrainBackend,
                    "io",
                    new_callable=PropertyMock,
                    return_value=Mock(),
                ):
                    tp1 = backend.start("A", 1, ["2"])
                    tp2 = backend.start("B", 2, ["0", "1"])
                self.assertEqual((tp1.context_limit, tp2.context_limit), (32768, 32768))
                settings = common.server_command.call_args.args[0]
                self.assertEqual(
                    (settings.max_model_len, settings.tp4_max_model_len),
                    (32768, 32768),
                )
            finally:
                backend.executor.shutdown()

    def test_replacements_all_spawn_before_any_readiness_wait(self):
        from unittest.mock import PropertyMock

        with tempfile.TemporaryDirectory() as tmp:
            driver, backend, common = self.backend(tmp, expanded_context=True)
            order = []
            common.start_process.side_effect = lambda name, *a: (
                order.append(("spawn", name)) or Mock()
            )
            common.wait_healthy.side_effect = lambda url, process, timeout: (
                order.append(("wait", url))
            )
            try:
                with patch.object(
                    driver.NativeDrainBackend,
                    "io",
                    new_callable=PropertyMock,
                    return_value=Mock(),
                ):
                    rows = backend.start_parallel(
                        [
                            dict(bundle_id="A", tp_degree=1, gpu_ids=[g])
                            for g in ("2", "3")
                        ],
                        activate=False,
                    )
                kinds = [kind for kind, _ in order]
                self.assertEqual(kinds[:2], ["spawn", "spawn"])
                self.assertEqual(sorted(kinds[2:]), ["wait", "wait"])
                self.assertTrue(all(r.lifecycle == "PREPARING" for r in rows))
                events = [e["kind"] for e in backend.events]
                self.assertEqual(events.count("HEALTHY"), 2)
                self.assertEqual(events.count("READY"), 2)
            finally:
                backend.executor.shutdown()

    def test_verification_is_a_fresh_request_per_replacement(self):
        from unittest.mock import PropertyMock

        with tempfile.TemporaryDirectory() as tmp:
            driver, backend, common = self.backend(tmp, expanded_context=True)
            io = Mock()
            io.post_streaming_completion.return_value = {"token_ids": [1] * 16}
            try:
                with patch.object(
                    driver.NativeDrainBackend,
                    "io",
                    new_callable=PropertyMock,
                    return_value=io,
                ):
                    backend.start_parallel(
                        [dict(bundle_id="A", tp_degree=2, gpu_ids=["2", "3"])],
                        activate=False,
                    )
                    calls = io.post_streaming_completion.call_count
                    plan = dict(bundle_id="A")
                    self.assertTrue(backend.warm_and_verify(plan))
                    self.assertTrue(backend.warm_and_verify(plan))
                    self.assertEqual(io.post_streaming_completion.call_count, calls + 1)
                    io.post_streaming_completion.return_value = {"token_ids": [1]}
                    backend.verified.clear()
                    with self.assertRaises(RuntimeError):
                        backend.warm_and_verify(plan)
            finally:
                backend.executor.shutdown()

    def test_validate_only_reports_requested_bundle_mapping(self):
        import contextlib
        import io as stdio

        from tools.bridge_tp import run_tp_bundle_drain as driver

        stream = stdio.StringIO()
        with contextlib.redirect_stdout(stream):
            code = driver.main(
                [
                    "--edge",
                    "L1A-L2",
                    "--model-path",
                    "unused",
                    "--out-root",
                    "unused",
                    "--platform",
                    "qwen3",
                    "--expanded-context",
                    "--bundle-gpus",
                    "2,3,0,1",
                    "--validate-only",
                ]
            )
        report = json.loads(stream.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(report["bundle"], "A")
        self.assertEqual(
            [row["gpu_ids"] for row in report["initial"]], [["2"], ["3"], ["0", "1"]]
        )
        self.assertEqual(
            [row["gpu_ids"] for row in report["final"]], [["2", "3"], ["0", "1"]]
        )
        with self.assertRaises(ValueError):
            driver.main(
                ["--edge", "L0-L1A", "--model-path", "m", "--out-root", "o"]
                + ["--expanded-context", "--validate-only"]
            )


class LinkProbeTests(unittest.TestCase):
    def test_transport_parsing_and_classification(self):
        from tools.bridge_tp import probe_gpu_links as probe

        log = "\n".join(
            [
                "host:1:2 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] via P2P/CUMEM",
                (
                    "host:1:2 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[0] "
                    "via SHM/direct/direct"
                ),
                "host:1:2 [0] NCCL INFO Connected all rings",
            ]
        )
        self.assertEqual(probe.transports(log), ["P2P/CUMEM", "SHM/direct/direct"])
        self.assertEqual(probe.classify(["P2P/CUMEM"]), "P2P")
        self.assertEqual(probe.classify(["SHM/direct/direct"]), "NON_P2P")
        self.assertEqual(probe.classify(probe.transports(log)), "MIXED")
        self.assertEqual(probe.classify([]), "UNKNOWN")
        self.assertEqual(len(probe.pairs([0, 1, 2, 3])), 6)
        with self.assertRaises(ValueError):
            probe.pairs([0, 0])


if __name__ == "__main__":
    unittest.main()
