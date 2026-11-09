# SPDX-License-Identifier: Apache-2.0
"""Sequence scheduling contracts and live-controller conversion cycles (CPU)."""

import contextlib
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from tests.bridge_tp.test_pressure_conversion import Harness, JoinHarness, jobs
from tools.bridge_tp import conversion_sequence as seq


def step(bundle="A", target="TP2", **extra):
    return dict(bundle=bundle, target_mode=target, trigger_s=0, deadline_s=5, **extra)


class Executor:
    """An effect-free executor; the real executor has separate CPU contracts."""

    def __init__(self, registry, backend, path, **kwargs):
        self.backend, self.path, self.plan = backend, Path(path), None

    def plan_create(self, plan_id, *args):
        self.plan = dict(plan_id=plan_id, state="CREATED")

    def plan_execute(self, now):
        self.plan["state"] = self.backend.states.pop(0)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.plan))
        return self.plan

    def plan_status(self):
        return dict(self.plan)


class DecidedDriverTests(unittest.TestCase):
    def run_driver(self, states, **request):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.events, self.writes, self.begun = [], [], []
        backend = NS(
            registry=NS(registry_revision=1),
            states=list(states),
            revision=0,
            run_dir=root,
            layout_tilt=["A1"],
            service_gate_passed=False,
            status=lambda: None,
            fresh=lambda status: False,
            ops=NS(event=lambda kind, **v: self.events.append(dict(kind=kind, **v))),
            step_acknowledged=lambda: True,
            write_layout=lambda *a, **k: self.writes.append(k),
        )

        def begin(plan_id, step):
            self.begun.append((plan_id, step))
            backend.layout_tilt = []

        backend.begin_step = begin
        request = dict(
            decider="EXECUTE",
            edges={},
            deadline_s=5,
            base_port=8400,
            assist=True,
            multi=True,
            **request,
        )
        with patch.object(seq.pc.conversion, "ConversionExecutor", Executor):
            driver = seq.DecidedSequenceDriver(
                backend, request, root / "conversion_plan.json", interval_s=0.001
            )
            driver.origin = time.monotonic()
            return driver, backend

    def propose(self, driver, *edges):
        with patch.object(seq.pc.conversion, "ConversionExecutor", Executor):
            for bundle, mode in edges:
                driver.start_proposal(
                    dict(bundle=bundle, target_mode=mode, held_s=75.0, layout="L1A")
                )

    def test_it_runs_plans_in_turn_and_frees_the_controller_between_them(self):
        driver, _ = self.run_driver(["COMPLETED", "COMPLETED"])
        self.propose(driver, ("A", "TP2"), ("A", "TP1"))
        self.assertEqual([s["target_mode"] for _, s in self.begun], ["TP2", "TP1"])
        self.assertEqual(
            [p for p, _ in self.begun], ["decided-0001-A", "decided-0002-A"]
        )
        self.assertEqual(len(driver.history), 2)
        self.assertEqual(self.writes[-1]["pending"], False)
        self.assertEqual(driver.sequence_state, "RUNNING")

    def ticking(self, state, remaining=None, short=8.0):
        driver, backend = self.run_driver(["COMPLETED"])
        cancels = []
        driver.executor = NS(
            plan=dict(state=state), plan_cancel=lambda: cancels.append(True)
        )
        driver.step_edge = ("L1A", ("A", "TP2"))
        backend.status = lambda: dict()
        backend.fresh = lambda status: True
        backend.remaining_s = lambda: remaining
        backend.safety = dict(continue_if_remaining_s=60.0)
        backend.registry = NS(instances={})
        driver.decider.signals = lambda *a: dict(short_rate=short, long_blocks=0)
        driver.decider.rules = dict(driver.decider.rules, cancel_hold_s=0.0)
        return driver, cancels

    def test_it_cancels_while_waiting_when_the_load_no_longer_fits(self):
        driver, cancels = self.ticking("CREATED")
        driver.plan_tick()
        self.assertEqual(cancels, [True])
        self.assertEqual(self.events[-1]["kind"], "DECIDER_CANCEL")

    def test_it_cancels_a_drain_with_long_left(self):
        driver, cancels = self.ticking("DRAINING", remaining=120.0)
        driver.plan_tick()
        self.assertEqual(cancels, [True])

    def test_six_committed_drain_does_not_cancel_on_changed_demand(self):
        for remaining in (None, 120.0):
            driver, cancels = self.ticking("DRAINING", remaining=remaining)
            driver.decider.profile = "six_card"
            driver.backend.status = lambda: self.fail("must not reconsider demand")
            driver.plan_tick()
            self.assertEqual(cancels, [])

    def test_it_finishes_a_drain_nearly_done(self):
        driver, cancels = self.ticking("DRAINING", remaining=30.0)
        driver.plan_tick()
        self.assertEqual(cancels, [])

    def test_it_does_not_cancel_after_the_drain(self):
        driver, cancels = self.ticking("FENCED")
        driver.plan_tick()
        self.assertEqual(cancels, [])

    def test_it_keeps_going_while_the_load_still_fits(self):
        driver, cancels = self.ticking("CREATED", short=1.0)
        driver.plan_tick()
        self.assertEqual(cancels, [])

    def test_a_cancelled_conversion_has_no_cooldown(self):
        driver, _ = self.run_driver(["ABORTED"])
        self.propose(driver, ("A", "TP2"))
        self.assertIsNone(driver.decider.converted_at)

    def test_each_edge_gets_its_gate(self):
        driver, backend = self.run_driver(["COMPLETED"] * 3)
        modes = []
        original = driver.after_begin

        def spy(step):
            original(step)
            modes.append(backend.gate_mode)

        driver.after_begin = spy
        self.propose(driver, ("A", "TP2"), ("A", "TP1"), ("B", "TP1"))
        self.assertEqual(modes, ["NONE", "PHYSICAL", "STRICT"])
        self.assertFalse(backend.assist_requires_spare)

    def test_the_prepare_stage_keeps_its_options_after_a_plan(self):
        driver, _ = self.run_driver(["COMPLETED"], pre_tilt=True)
        self.propose(driver, ("A", "TP2"))
        self.assertTrue(driver.request["pre_tilt"])

    def test_only_a_merge_towards_l2_is_assisted(self):
        driver, _ = self.run_driver(["COMPLETED"] * 3)
        self.propose(driver, ("A", "TP2"), ("A", "TP1"), ("B", "TP2"))
        self.assertEqual(
            [s["method"] for _, s in self.begun],
            ["SELECTIVE_ROLL", "DRAIN", "DRAIN"],
        )

    def test_a_safe_abort_leaves_later_proposals_open(self):
        driver, _ = self.run_driver(["ABORTED", "COMPLETED"])
        self.propose(driver, ("A", "TP2"), ("A", "TP2"))
        self.assertEqual(driver.sequence_state, "RUNNING")
        self.assertEqual(len(self.begun), 2)

    def test_recovery_required_ends_all_conversion(self):
        driver, _ = self.run_driver(["RECOVERY_REQUIRED"])
        self.propose(driver, ("A", "TP2"))
        self.assertEqual(driver.sequence_state, "RECOVERY_REQUIRED")
        self.assertEqual(driver.state(), "RECOVERY_REQUIRED")

    def test_the_limit_stops_conversions_but_keeps_the_record(self):
        driver, _ = self.run_driver(["COMPLETED"], max_conversions=1)
        self.propose(driver, ("A", "TP2"), ("A", "TP1"))
        self.assertEqual(len(self.begun), 1)
        self.assertIn("DECIDER_LIMIT", [e["kind"] for e in self.events])

    def test_zero_limit_allows_all_proposals_without_budget_limit_events(self):
        driver, _ = self.run_driver(["COMPLETED"] * 6, max_conversions=0)
        self.propose(driver, *([("A", "TP2"), ("A", "TP1")] * 3))
        self.assertEqual(len(self.begun), 6)
        self.assertNotIn("DECIDER_LIMIT", [e["kind"] for e in self.events])

    def test_a_tilt_survives_the_step_and_passes_the_service_gate(self):
        driver, backend = self.run_driver(["COMPLETED"])
        self.propose(driver, ("A", "TP2"))
        self.assertTrue(backend.service_gate_passed)
        self.assertIn(dict(tilt=["A1"]), self.writes)

    def test_the_decider_cools_down_after_each_plan(self):
        driver, _ = self.run_driver(["COMPLETED"])
        driver.decider.since["A->TP2"] = 1.0
        self.propose(driver, ("A", "TP2"))
        self.assertIsNotNone(driver.decider.converted_at)
        self.assertEqual(driver.decider.since, {})

    def test_it_does_not_prestart_in_the_prepare_stage(self):
        self.assertFalse(seq.DecidedSequenceDriver.prepare_prestart)

    def test_decided_assessment_does_not_expect_a_fixed_sequence(self):
        import inspect

        self.assertIn("decided", inspect.signature(seq.assess).parameters)


class SequenceDriverTests(unittest.TestCase):
    def run_driver(self, steps, states):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        events, changes, begun = [], [], []
        backend = NS(
            registry=NS(registry_revision=1),
            states=list(states),
            revision=0,
            ops=NS(event=lambda kind, **v: events.append(dict(kind=kind, **v))),
            begin_step=lambda *v: begun.append(v),
            step_acknowledged=lambda: True,
            write_layout=lambda *a, **k: changes.append(k),
        )
        request = dict(sequence=steps, base_port=8400)
        with patch.object(seq.pc.conversion, "ConversionExecutor", Executor):
            driver = seq.ConversionSequenceDriver(
                backend, request, Path(self.temp.name) / "legacy.json", interval_s=0.001
            )
            driver.start()
            driver.thread.join(5)
            self.assertFalse(driver.thread.is_alive())
            result = driver.stop()
        self.assertFalse(changes[-1]["pending"])
        self.assertTrue(changes[0]["pending"])
        return driver, result, begun

    def test_four_plans_are_serial_and_keep_separate_journals(self):
        driver, result, begun = self.run_driver(
            [step(), step(target="TP1"), step("B", "TP1"), step("B")],
            ["COMPLETED"] * 4,
        )
        self.assertEqual(result["state"], "COMPLETED")
        self.assertEqual(len(begun), 4)
        self.assertEqual(len({v[0] for v in begun}), 4)
        self.assertEqual(len(list(driver.root.glob("conversions/*/plan.json"))), 4)
        self.assertEqual(len(result["steps"]), 4)

    def test_safe_abort_stops_by_default_but_can_retry_explicitly(self):
        _, result, begun = self.run_driver([step(), step()], ["ABORTED", "COMPLETED"])
        self.assertEqual(result["state"], "ABORTED")
        self.assertEqual(len(begun), 1)
        _, result, begun = self.run_driver(
            [step(on_abort="CONTINUE"), step()], ["ABORTED", "COMPLETED"]
        )
        self.assertEqual(result["state"], "COMPLETED")
        self.assertEqual(len(begun), 2)

    def test_uncertain_effect_stops_even_with_continue(self):
        _, result, begun = self.run_driver(
            [step(on_abort="CONTINUE"), step()], ["RECOVERY_REQUIRED", "COMPLETED"]
        )
        self.assertEqual(result["state"], "RECOVERY_REQUIRED")
        self.assertEqual(len(begun), 1)

    def test_bad_sequence_fails_before_effects(self):
        for request in (
            dict(sequence=[], base_port=8400),
            dict(sequence=[step(bundle="C")], base_port=8400),
            dict(
                sequence=[step(method="SELECTIVE_ROLL", target="TP1")], base_port=8400
            ),
            dict(sequence=[step()], base_port=0),
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                seq.validate(request)


class CycleHarness(Harness):
    """The real controller/executor with CPU engine and NCCL receipt doubles."""

    def __init__(self, root, steps, prestart=False):
        super().__init__(root, "A", "TP2", channel_plan=True)
        self.receipts_after_s = 0.05
        old = self.backend
        for row in self.manifest["instances"]:
            if row["tp_degree"] != 1:
                continue
            source = row["instance_id"]
            channel = dict(
                base_port=30400 if source == "A-g1-tp1" else 30410,
                generation=row["instance_generation"],
                target="B-g2-tp2",
                target_tp=2,
            )
            row.update(
                migration_channel=channel,
                targets_file=str(root / f"gpu_targets_{source}.json"),
                predictor_events=str(root / f"predictor_{source}.jsonl"),
            )
            Path(row["targets_file"]).write_text("[]")
        self.backend = seq.SequenceBackend(
            self.registry,
            self.manifest,
            root,
            old.ops,
            base_port=8400,
            context_limits=(8192, 8192),
            channel_plan=self.channel_plan,
            target_tp_ranks=2,
            prestart=prestart,
            sequence_context=dict(
                target_tp=2,
                target_host="127.0.0.1",
                folder=str(root),
                target_slot=dict(
                    endpoint="http://127.0.0.1:8420",
                    inventory_path=str(root / "new_target"),
                ),
            ),
        )
        self.driver = seq.ConversionSequenceDriver(
            self.backend,
            dict(sequence=steps, base_port=8400, prestart=prestart),
            root / "conversion_plan.json",
            interval_s=0.02,
        )

    def event(self, kind, **values):
        super().event(kind, **values)
        if kind in {"CHANNEL_PLAN_WRITTEN", "CHANNEL_PLAN_JOINED"}:
            directory = self.root / "session-1" / "gpu_channel_plan_receipts"
            directory.mkdir(parents=True, exist_ok=True)
            planned = [
                c["generation"] for c in json.loads(self.channel_plan.read_text())
            ]
            for rank in (0, 1):
                (directory / f"tp_rank_{rank}.json").write_text(
                    json.dumps(
                        dict(
                            planned=planned,
                            closed=values.get("closing", []),
                            completed_unix_s=time.time() + 0.01,
                        )
                    )
                )
        if kind == "CHANNEL_PLAN_JOINED":
            self.join = self.backend.join
            timer = threading.Timer(
                self.receipts_after_s, self.warm_channels, args=(values,)
            )
            timer.daemon = True
            timer.start()

    warm_channels = JoinHarness.warm_channels

    def stopped_endpoints(self):
        return {self.backend.rows[i]["endpoint"] for i in self.stopped}


class LiveCycleTests(unittest.TestCase):
    def test_four_edges_use_new_instances_and_apply_each_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            h = CycleHarness(
                Path(directory),
                [
                    dict(v, deadline_s=20)
                    for v in (
                        step(),
                        step(target="TP1"),
                        step("B", "TP1"),
                        step("B"),
                    )
                ],
            )
            result = h.run(jobs(40, 0.15), max_seconds=40)
            self.assertEqual(result["state"], "COMPLETED", result)
            self.assertIsNone(result["error"], result)
            self.assertEqual(len(result["steps"]), 4)
            self.assertTrue(
                all(r["plan"]["state"] == "COMPLETED" for r in result["steps"])
            )
            active = h.backend.active()
            self.assertEqual(sorted(r.tp_degree for r in active.values()), [1, 1, 2])
            self.assertFalse(set(active) & {"A-g1-tp1", "B-g2-tp2", "A-g3-tp1"})
            status = json.loads((h.root / "layout_status.json").read_text())
            target = next(i for i, r in active.items() if r.tp_degree == 2)
            self.assertEqual(status["migration_target"], target)
            channels = json.loads(h.channel_plan.read_text())
            self.assertEqual(
                {c["source"] for c in channels},
                {i for i, r in active.items() if r.tp_degree == 1},
            )
            self.assertEqual(len({c["generation"] for c in channels}), 2)
            self.assertEqual(
                len(json.loads((h.root / "pressure_request_results.json").read_text())),
                40,
            )

    def test_the_decider_converts_twice_with_requests_held_in_the_controller(self):
        class Scripted:
            """The real driver and controller; only the choice of edge is fixed."""

            execute, rules = True, dict(tilt_max_s=90.0)

            def __init__(self):
                self.edges = [("A", "TP2"), ("A", "TP1")]
                self.converted_times = 0

            def observe(self, status, bundles, members, now):
                bundle, mode = self.edges.pop(0) if self.edges else (None, None)
                return dict(
                    mode="EXECUTE",
                    evaluated_unix_s=now,
                    prepared=None,
                    tilt=[],
                    proposal=dict(bundle=bundle, target_mode=mode, held_s=1.0)
                    if bundle
                    else None,
                )

            def signals(self, *args):
                return dict(short_rate=0.0, long_blocks=0)

            def cancel_reason(self, *args):
                return None

            def converted(self, now):
                self.converted_times += 1

        with tempfile.TemporaryDirectory() as directory:
            h = CycleHarness(Path(directory), [step()])
            request = dict(
                decider="EXECUTE",
                edges={},
                deadline_s=20,
                base_port=8400,
                assist=False,
                multi=True,
            )
            h.driver = seq.DecidedSequenceDriver(
                h.backend, request, h.root / "conversion_plan.json", interval_s=0.02
            )
            h.driver.decider = scripted = Scripted()
            result = h.run(
                jobs(40, 0.15), max_seconds=40, hold_unstarted=2, hold_long_wait_s=30.0
            )
            self.assertEqual(result["state"], "COMPLETED", result)
            self.assertIsNone(result["error"], result)
            self.assertEqual(
                [r["plan"]["state"] for r in result["steps"]],
                ["COMPLETED", "COMPLETED"],
            )
            self.assertEqual(scripted.converted_times, 2)
            self.assertEqual(
                [r["layout"] for r in result["steps"]],
                [{"A": 2, "B": 2}, {"A": 1, "B": 2}],
            )
            self.assertTrue(all(r["controller_acknowledged"] for r in result["steps"]))
            rows = json.loads((h.root / "pressure_request_results.json").read_text())
            self.assertEqual(len(rows), 40)
            self.assertTrue(all("held_s" in r["dispatch_timing"] for r in rows))
            self.assertTrue(
                all(r["source_done"] and not r["source_error"] for r in rows)
            )
            closed, late = {}, []
            audit = (h.root / "pressure_audit.jsonl").read_text().splitlines()
            events = [json.loads(line) for line in audit]
            for event in sorted(
                (e for e in events if "unix_s" in e), key=lambda e: e["unix_s"]
            ):
                if event["kind"] == "pressure_instance_lifecycle":
                    closed[event["instance_id"]] = event["to"] != "ACTIVE"
                elif event["kind"] == "pressure_dispatch" and closed.get(event["pool"]):
                    late.append(event["job_id"])
            self.assertEqual(late, [])
            self.assertTrue(
                all(
                    e["ingress_capacity_gate"]
                    for e in events
                    if e["kind"] == "pressure_dispatch"
                )
            )

    def test_prestart_cycle_and_safe_abort_then_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            h = CycleHarness(
                Path(directory),
                [
                    dict(step(on_abort="CONTINUE"), deadline_s=0.1),
                    dict(step(), deadline_s=20),
                    dict(step(target="TP1"), deadline_s=20),
                ],
                prestart=True,
            )
            gate = h.backend.gate
            h.backend.gate = lambda plan: (
                False if plan["plan_id"] == "sequence-0001-A" else gate(plan)
            )
            result = h.run(jobs(24, 0.15), max_seconds=40)
            self.assertEqual(result["state"], "COMPLETED", result)
            self.assertEqual(
                [r["plan"]["state"] for r in result["steps"]],
                ["ABORTED", "COMPLETED", "COMPLETED"],
            )
            self.assertEqual(sum(e["kind"] == "PRESTART" for e in h.events), 2)
            self.assertEqual(len(h.backend.active()), 3)

    def test_cycle_cli_is_one_run_and_keeps_explicit_steps_separate_from_decider(self):
        from tools.bridge_tp import run_qwen3_stay_migrate as runner

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = runner.main(
                [
                    "--conversion-cycle",
                    "--current-machine",
                    "--repetitions",
                    "1",
                    "--model",
                    "unused",
                    "--output",
                    "unused",
                    "--expected-revision",
                    "CPU",
                    "--validate-only",
                ]
            )
        value = json.loads(stream.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(value["runs"], 1)
        self.assertEqual(value["arms"], ["migrate"])
        self.assertEqual(value["service_context_limits"], dict(tp1=32768, tp2=32768))
        self.assertEqual(value["recipes"][0]["scenario"], "conversion_cycle")


class SequenceEvidenceTests(unittest.TestCase):
    def test_finalization_uses_current_instances_not_retired_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = [
                dict(
                    instance_id="A-g6-tp1",
                    tp_degree=1,
                    lifecycle="ACTIVE",
                    endpoint="http://new-source:8400",
                    inventory_path=str(root / "new-inventory.jsonl"),
                ),
                dict(
                    instance_id="B-g9-tp2",
                    tp_degree=2,
                    lifecycle="ACTIVE",
                    endpoint="http://new-target:8620",
                    inventory_path=str(root / "target-inventory.jsonl"),
                ),
            ]
            (root / "layout_control.json").write_text(
                json.dumps(dict(instances=active))
            )
            (root / "layout_status.json").write_text(
                json.dumps(
                    dict(
                        instances={
                            r["instance_id"]: dict(lifecycle="ACTIVE") for r in active
                        }
                    )
                )
            )
            environment = dict(
                BRIDGETP_PERFORMANCE_FINALIZE_TOKEN="token",
                BRIDGETP_PERFORMANCE_TRACE_DIR=str(root / "trace/old"),
            )
            post, verify = Mock(return_value=dict(status="FINALIZED")), Mock()
            for row in active:
                Path(row["inventory_path"]).write_text(
                    json.dumps(
                        dict(
                            schema="PRESSURE_INVENTORY_V1",
                            sampled_at=time.time(),
                            requests=[],
                            running=0,
                            waiting=0,
                        )
                    )
                    + "\n"
                )
            with (
                patch.dict(
                    sys.modules,
                    {
                        "tools.bridge_tp.run_atomic_takeover": NS(_post_json=post),
                        "tools.bridge_tp.verify_performance_trace": NS(
                            verify_pool=verify
                        ),
                    },
                ),
            ):
                seq.finalize(
                    dict(run_dir=root, source_env=environment, target_env=environment),
                    NS(write_json=Mock()),
                )
                for key in (
                    "writer_error",
                    "inventory_error",
                    "dropped",
                    "dropped_before",
                ):
                    with self.subTest(key=key):
                        Path(active[0]["inventory_path"]).write_text(
                            json.dumps(
                                dict(
                                    schema="PRESSURE_INVENTORY_V1",
                                    sampled_at=time.time(),
                                    requests=[],
                                    running=0,
                                    waiting=0,
                                    **{key: 1},
                                )
                            )
                            + "\n"
                        )
                        with self.assertRaises(RuntimeError):
                            seq.finalize(
                                dict(
                                    run_dir=root,
                                    source_env=environment,
                                    target_env=environment,
                                ),
                                NS(write_json=Mock()),
                            )
            self.assertEqual(
                [c.args[0] for c in post.call_args_list],
                [r["endpoint"] for r in active],
            )
            self.assertEqual([c.args[1] for c in verify.call_args_list], [1, 2])
            self.assertEqual(
                [c.args[0].name for c in verify.call_args_list],
                [r["instance_id"] for r in active],
            )

    def test_step_resets_assist_budget_without_reusing_command_sequence(self):
        from tests.bridge_tp.test_conversion_safety import PLAN, row, status

        with tempfile.TemporaryDirectory() as directory:
            h = CycleHarness(Path(directory), [step()])
            b = h.backend
            previous = {f"m{i}": "COMPLETED" for i in range(4)}
            current = status(
                requests=[row("new-long", 8192)],
                assist_sessions=previous,
                assist_result=dict(seq=7, state="STARTED", request_id="old-long"),
            )
            b.status = lambda: current
            b.assist_seq, b.assisted = 7, ["old-long"]
            b.begin_step("next-plan", step())
            self.assertEqual(b.assist_seq, 7)
            self.assertEqual(b.assisted, [])
            reason, choice = b.assist_choice(PLAN, current)
            self.assertIsNone(reason)
            self.assertEqual(choice["request_id"], "new-long")
            self.assertEqual(b.assisted, [])

    def test_startup_sources_follow_replaceable_target_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            sources = [
                dict(source=name, generation=i + 1, base_port=30400 + 10 * i)
                for i, name in enumerate(("A-g1-tp1", "A-g3-tp1"))
            ]
            plan = dict(
                channel_sources=sources,
                instances=dict(
                    legacy_pool_aliases=dict(tp1="A-g1-tp1", tp4="B-g2-tp2"),
                    instances=[
                        dict(instance_id="A-g1-tp1", tp_degree=1, legacy_pool="tp1"),
                        dict(instance_id="A-g3-tp1", tp_degree=1),
                    ],
                ),
                run_dir=folder / "controller",
                extra_engines=[dict(instance_id="A-g3-tp1", predictor_events="extra")],
                source_env=dict(
                    BRIDGETP_PREDICTOR_LIVE_EVENTS="pair",
                    BRIDGETP_GPU_DIRECT_TARGETS="old-static-list",
                ),
                config=dict(target_url="http://127.0.0.1:8620"),
            )
            seq.prepare_plan(plan)
            self.assertNotIn("BRIDGETP_GPU_DIRECT_TARGETS", plan["source_env"])
            self.assertIn("BRIDGETP_GPU_DIRECT_TARGETS_FILE", plan["source_env"])
            self.assertIn("targets_file", plan["extra_engines"][0])
            self.assertEqual(len(plan["sequence_target_files"]), 2)
            self.assertEqual(list(folder.iterdir()), [])  # metadata only

    def test_channel_identity_includes_receiver_incarnation(self):
        from tools.bridge_tp.pressure_sessions import PressureSessions

        before = NS(source_instance="A-g5-tp1", target_epoch="B-g2-tp2")
        after = NS(source_instance="A-g5-tp1", target_epoch="B-g7-tp2")
        self.assertNotEqual(
            PressureSessions._channel_key(before), PressureSessions._channel_key(after)
        )

    def test_certified_idle_requires_exact_identity_and_two_ranks(self):
        from tools.bridge_tp.pressure_sessions import verified_idle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = dict(
                status="VERIFIED_CHANNEL_IDLE",
                migration_id="m1",
                source_request_id="r1",
                sender=dict(
                    communicator_lifecycle="PERSISTENT_CHANNEL_IDLE",
                    migration_id="m1",
                    source_request_id="r1",
                    channel_session_count=1,
                    channel_create_count=1,
                    channel_destroy_count=0,
                ),
                ranks=[
                    dict(
                        status="PERSISTENT_CHANNEL_IDLE",
                        migration_id="m1",
                        session_request_id="r1",
                        tp_rank=rank,
                        channel_session_count=1,
                        channel_create_count=1,
                        channel_destroy_count=0,
                    )
                    for rank in range(2)
                ],
            )
            path = root / "channel_idle_verified.json"
            path.write_text(json.dumps(value))
            self.assertTrue(verified_idle(root, "m1", "r1"))
            self.assertFalse(verified_idle(root, "m2", "r1"))
            self.assertFalse(verified_idle(root, "m1", "r2"))
            value["sender"]["channel_session_count"] = None
            for rank in value["ranks"]:
                rank["channel_session_count"] = None
            path.write_text(json.dumps(value))
            self.assertFalse(verified_idle(root, "m1", "r1"))

    def test_acceptance_distinguishes_cycle_from_real_migration_coverage(self):
        # Synthetic receipts test the assessor; they are not GPU evidence.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "controller").mkdir()
            events, audit, steps = [], [], []
            layouts = [dict(A=2, B=2), dict(A=1, B=2), dict(A=1, B=1), dict(A=1, B=2)]
            for index, layout in enumerate(layouts):
                base, name = index * 100, f"plan-{index}"
                for offset, kind in enumerate(
                    (
                        "LAYOUT_CONTROL",
                        "FENCED",
                        "RESOURCES_RELEASED",
                        "ROUTE_PUBLISHED",
                    )
                ):
                    events.append(
                        dict(
                            kind=kind,
                            plan_id=name,
                            unix_s=base + offset + 1,
                            changes={"source": "QUIESCING"},
                        )
                    )
                audit.append(
                    dict(
                        kind="pressure_layout_applied",
                        revision=index + 1,
                        unix_s=base + 5,
                    )
                )
                steps.append(
                    dict(
                        plan_id=name,
                        triggered_unix_s=base,
                        completed_unix_s=base + 6,
                        layout=layout,
                        layout_revision=index + 1,
                        controller_acknowledged=True,
                        plan=dict(state="COMPLETED"),
                        active_instances={f"target-{index}": {}},
                    )
                )
            (root / "conversion_result.json").write_text(
                json.dumps(dict(state="COMPLETED", error=None, steps=steps))
            )
            (root / "conversion_events.jsonl").write_text(
                "\n".join(json.dumps(v) for v in events)
            )
            row = dict(
                job_id="r1",
                source_done=True,
                source_error=None,
                request=dict(max_tokens=1),
                dispatch_timing=dict(planned_unix_s=0, http_thread_started_unix_s=2),
                response=dict(emitted=[dict(unix_s=5)]),
            )
            (root / "controller/pressure_request_results.json").write_text(
                json.dumps([row])
            )
            proof = dict(sessions=[])

            def check():
                (root / "controller/pressure_audit.jsonl").write_text(
                    "\n".join(
                        json.dumps(v)
                        # The audit writer's last row has no timestamp.
                        for v in [*audit, dict(kind="evidence_footer", dropped=0)]
                    )
                )
                (root / "closed_loop_acceptance.json").write_text(json.dumps(proof))
                return seq.assess(root, 1)

            result = check()
            self.assertTrue(result["conversion_correct"])
            self.assertFalse(result["passed"])
            self.assertFalse(result["migration_coverage"])
            self.assertEqual(result["planned_latency"]["dispatch_delay_s"]["max"], 2)
            for index in (1, 3):
                audit.append(
                    dict(
                        kind="pressure_actual_start",
                        unix_s=index * 100 + 10,
                        migration_id=f"m{index}",
                        target_epoch=f"target-{index}",
                    )
                )
                proof["sessions"].append(
                    dict(
                        migration_id=f"m{index}", state="COMMITTED", source_release=True
                    )
                )
            self.assertTrue(check()["passed"])
            audit[-1]["target_epoch"] = "retired-target"
            self.assertFalse(check()["migration_coverage"])
            audit.extend(
                [
                    dict(
                        kind="pressure_instance_lifecycle",
                        instance_id="source",
                        to="QUIESCING",
                        unix_s=1,
                    ),
                    dict(
                        kind="pressure_dispatch",
                        pool="source",
                        job_id="late",
                        dispatch_entered_unix_s=3,
                        unix_s=3,
                    ),
                ]
            )
            result = check()
            self.assertFalse(result["conversion_correct"])
            self.assertEqual(result["late_dispatches"], ["late"])
