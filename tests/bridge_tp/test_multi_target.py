# SPDX-License-Identifier: Apache-2.0
"""Stage 3: choosing among receivers, per-target sessions and source entries.

CPU contracts only; no CUDA, NCCL or serving engine.
"""

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp.test_migration_receivers import build as build_b
from tests.bridge_tp.test_runtime_instance_ids import canonical_plan
from tests.bridge_tp.test_unified_scheduler import (
    TP,
    Started,
    coordinator,
    inst,
    source_candidates,
)
from tools.bridge_tp import migration_receivers as receivers
from tools.bridge_tp import qwen3_six_card as six
from tools.bridge_tp import run_pressure_balanced_pilot as pilot
from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp.pressure_sessions import TargetSessions, switch_source_entry


def build(root, execute="all", force=None):
    plan = canonical_plan(root, "L2C")
    plan["controller_argv"] += ["--maximum-migrations", "0"]
    plan["stager_argv"][1] = "tools/bridge_tp/pressure_sessions.py"
    receivers.configure(plan, execute, force)
    return plan


def symlink(test, link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            test.skipTest("Windows symlink privilege unavailable; Linux required")
        raise


class Bindings:
    """START bindings per (request, target), recording which one ran."""

    def __init__(self):
        self.calls = []

    def actions(self, rows, targets, now):
        return {
            (c.request_id, t): dataclasses.replace(
                Started().bind(c.request_id, f"{t}-{now}"),
                start=lambda c=c, t=t: self.calls.append((c.request_id, t)),
            )
            for c in rows
            for t in targets
        }


class SelectionTests(unittest.TestCase):
    """Design 11-06 section 3, on the unified tick with targets b and c."""

    def tick(self, co, now, source, targets, rows, bindings=None):
        source = dataclasses.replace(source, sampled_at=now, published_at=now)
        targets = {
            k: dataclasses.replace(v, sampled_at=now, published_at=now)
            if v.sampled_at == 10.0
            else v
            for k, v in targets.items()
        }
        rows = [
            dataclasses.replace(c, sampled_at=now, published_at=now)
            if c.sampled_at == 10.0
            else c
            for c in rows
        ]
        actions = bindings.actions(rows, list(targets), now) if bindings else None
        return co.tick_multi(
            now,
            {"a1": source} | targets,
            TP,
            {("a1", t): rows for t in targets},
            actions,
        )

    @staticmethod
    def decision(records):
        return [x for x in records if x["kind"] == "pressure_decision_unified"][-1]

    def kv_source(self):
        return inst("a1", total=2325, free=150, requests=3)

    def test_both_fit_the_less_full_one_is_chosen(self):
        co, records = coordinator()
        bindings = Bindings()
        targets = dict(b=inst("b", requests=10), c=inst("c", requests=2))
        rows = source_candidates(16384, 4096)
        self.tick(co, 10.0, self.kv_source(), targets, rows, bindings)
        self.assertEqual(bindings.calls, [("q0", "c")])
        entry = self.decision(records)["evaluations"][0]
        self.assertEqual([t["target"] for t in entry["targets"]], ["c"])
        self.assertEqual(self.decision(records)["chosen"], ["a1", "c", "q0"])

    def test_the_least_full_without_room_falls_back_to_the_next(self):
        co, records = coordinator()
        bindings = Bindings()
        # c is less full (KV 0.91 against b's slots 0.94) but has 975 blocks
        # of room after the guard; the move needs 1030.
        targets = dict(
            b=inst("b", requests=15),
            c=inst("c", requests=2, free=1100),
        )
        rows = source_candidates(16384)
        self.tick(co, 10.0, self.kv_source(), targets, rows, bindings)
        self.assertEqual(bindings.calls, [("q0", "b")])
        entry = self.decision(records)["evaluations"][0]
        self.assertEqual([t["target"] for t in entry["targets"]], ["c", "b"])
        self.assertIn(
            dict(
                request_id="q0",
                reasons=["target room promised to pending prompts"],
                target="c",
            ),
            entry["rejected"],
        )

    def test_no_receiver_with_room_means_no_move(self):
        co, records = coordinator()
        bindings = Bindings()
        targets = dict(
            b=inst("b", free=1200, unallocated=1100),
            c=inst("c", free=1300, unallocated=1200),
        )
        self.tick(
            co, 10.0, self.kv_source(), targets, source_candidates(16384), bindings
        )
        self.assertEqual(bindings.calls, [])
        decision = self.decision(records)
        self.assertIsNone(decision["chosen"])
        rejected = {r["target"] for r in decision["evaluations"][0]["rejected"]}
        self.assertEqual(rejected, {"b", "c"})

    def test_ties_go_to_fewer_promised_prompt_blocks_then_order(self):
        co, records = coordinator()
        bindings = Bindings()
        targets = dict(
            b=inst("b", free=9000, unallocated=300),
            c=inst("c", free=9300, unallocated=600),
        )
        # Same KV share (used + unallocated) and slots: b has fewer promised.
        self.tick(
            co, 10.0, self.kv_source(), targets, source_candidates(4096), bindings
        )
        self.assertEqual(bindings.calls, [("q0", "b")])

    def test_kv_ignores_a_cooling_receiver_and_pressure_does_not(self):
        co, records = coordinator()
        co.migration_state.cooldown_until["c"] = 100.0
        bindings = Bindings()
        targets = dict(b=inst("b", requests=10), c=inst("c", requests=2))
        self.tick(
            co, 10.0, self.kv_source(), targets, source_candidates(4096), bindings
        )
        self.assertEqual(bindings.calls, [("q0", "c")])
        # Pressure on the same source: only b, outside cooling, may receive.
        co, records = coordinator()
        co.migration_state.cooldown_until["c"] = 100.0
        bindings = Bindings()
        source = inst("a1", total=2325, free=1800, requests=15)
        rows = source_candidates(2048)
        self.tick(co, 10.0, source, targets, rows, bindings)
        self.tick(co, 21.0, source, targets, rows, bindings)
        self.assertEqual(bindings.calls, [("q0", "b")])

    def test_a_stale_receiver_is_not_eligible(self):
        co, records = coordinator()
        bindings = Bindings()
        targets = dict(
            b=inst("b", requests=10),
            c=inst("c", requests=2, sampled_at=1.0, published_at=1.0),
        )
        self.tick(
            co, 10.0, self.kv_source(), targets, source_candidates(4096), bindings
        )
        self.assertEqual(bindings.calls, [("q0", "b")])

    def test_a_receiver_without_candidates_is_skipped(self):
        # The pilot passes only warm (source, target) pairs.
        co, records = coordinator()
        started = Started()
        rows = [
            dataclasses.replace(c, sampled_at=10.0, published_at=10.0)
            for c in source_candidates(4096)
        ]
        actions = {
            (c.request_id, "b"): started.bind(c.request_id, "b-10") for c in rows
        }
        co.tick_multi(
            10.0,
            dict(
                a1=dataclasses.replace(self.kv_source(), sampled_at=10.0),
                b=inst("b", requests=10, sampled_at=10.0),
                c=inst("c", requests=2, sampled_at=10.0),
            ),
            TP,
            {("a1", "b"): rows},
            actions,
        )
        self.assertEqual(started.calls, ["q0"])
        entry = self.decision(records)["evaluations"][0]
        self.assertEqual(
            [(t["target"], t["candidates"]) for t in entry["targets"]],
            [("c", 0), ("b", 1)],
        )


class Group:
    """A PressureSessions stand-in: one session that may be in flight."""

    def __init__(self, root, name):
        self.root = root
        self.record = None
        self.cleanup_requires_shutdown = False
        self.migration_id = f"run-{name}-0001"
        self.sessions = [NS(record=None, migration_id=self.migration_id)]
        self.rotations = 0
        self.closed = False
        self.started = []

    def start(self, candidate, snapshot):
        self.record = NS(request_id=candidate.request_id)
        self.started.append(candidate.request_id)

    def rotate(self):
        self.rotations += 1
        return False

    def bind_target(self, *args):
        self.bound = args

    def close(self):
        self.closed = True


class Audit:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append(row)


class SessionOrderTests(unittest.TestCase):
    """Entry switching order, with the link operation replaced (any OS)."""

    def test_the_entry_moves_before_start_and_never_while_in_flight(self):
        events = []
        groups = {
            name: Group(Path("/run/migration") / name, name)
            for name in ("B-g2-tp2", "A-g3-tp2")
        }
        groups["A-g3-tp2"].start = lambda c, s: events.append(("start", c.owner))
        links = {"C-g1-tp1": Path("/run/source_active/C-g1-tp1")}

        def switch(link, active, audit, *, source, target):
            events.append(("switch", link.name, active.parent.name, target))

        sessions = TargetSessions(groups, links, Audit())
        with patch("tools.bridge_tp.pressure_sessions.switch_source_entry", switch):
            sessions.start_on("A-g3-tp2", NS(request_id="q", owner="C-g1-tp1"), None)
            self.assertEqual(
                events,
                [
                    ("switch", "C-g1-tp1", "A-g3-tp2", "A-g3-tp2"),
                    ("start", "C-g1-tp1"),
                ],
            )
            groups["A-g3-tp2"].record = NS(request_id="q")
            with self.assertRaisesRegex(ValueError, "not rotated"):
                sessions.start_on(
                    "B-g2-tp2", NS(request_id="r", owner="C-g1-tp1"), None
                )
            self.assertEqual(len(events), 2)
            self.assertEqual(sessions.current, "A-g3-tp2")


class SessionTests(unittest.TestCase):
    def layout(self, folder):
        run = Path(folder) / "controller"
        groups = {}
        for name in ("B-g2-tp2", "A-g3-tp2"):
            root = run / "migration" / name
            (root / "sessions" / "migration-0001").mkdir(parents=True)
            (root / "sessions" / "migration-0001" / "runtime_control.json").write_text(
                json.dumps(dict(armed=False))
            )
            symlink(
                self, root / "migration_active", root / "sessions" / "migration-0001"
            )
            groups[name] = Group(root, name)
        (run / "source_active").mkdir()
        links = {}
        for source in ("C-g1-tp1", "C-g4-tp1"):
            link = run / "source_active" / source
            symlink(self, link, run / "migration" / "B-g2-tp2" / "migration_active")
            links[source] = link
        return run, groups, links

    def test_a_start_moves_only_the_owner_entry_and_marks_current(self):
        with tempfile.TemporaryDirectory() as folder:
            run, groups, links = self.layout(folder)
            audit = Audit()
            sessions = TargetSessions(groups, links, audit)
            self.assertTrue(sessions.ready_for_start)
            sessions.start_on("A-g3-tp2", NS(request_id="q", owner="C-g4-tp1"), None)
            a_active = run / "migration" / "A-g3-tp2" / "migration_active"
            self.assertEqual(Path(os.readlink(links["C-g4-tp1"])), a_active)
            self.assertNotEqual(Path(os.readlink(links["C-g1-tp1"])), a_active)
            self.assertEqual(sessions.current, "A-g3-tp2")
            self.assertEqual(sessions.migration_id, "run-A-g3-tp2-0001")
            self.assertEqual(groups["A-g3-tp2"].started, ["q"])
            self.assertEqual(audit.rows[-1]["kind"], "source_entry_switched")
            # The entry resolves to the target's own session directory.
            self.assertEqual(
                links["C-g4-tp1"].resolve(),
                (
                    run / "migration" / "A-g3-tp2" / "sessions" / "migration-0001"
                ).resolve(),
            )

    def test_no_start_while_any_target_has_a_session_in_flight(self):
        with tempfile.TemporaryDirectory() as folder:
            run, groups, links = self.layout(folder)
            sessions = TargetSessions(groups, links, Audit())
            groups["B-g2-tp2"].record = NS(request_id="old")
            self.assertFalse(sessions.ready_for_start)
            before = os.readlink(links["C-g1-tp1"])
            with self.assertRaisesRegex(ValueError, "not rotated"):
                sessions.start_on(
                    "A-g3-tp2", NS(request_id="q", owner="C-g1-tp1"), None
                )
            # The entry did not move while B's session is unrotated.
            self.assertEqual(os.readlink(links["C-g1-tp1"]), before)
            groups["B-g2-tp2"].record = None
            groups["B-g2-tp2"].cleanup_requires_shutdown = True
            self.assertFalse(sessions.ready_for_start)
            self.assertTrue(sessions.cleanup_requires_shutdown)

    def test_rotation_reaches_every_target_and_sessions_are_all_listed(self):
        with tempfile.TemporaryDirectory() as folder:
            run, groups, links = self.layout(folder)
            sessions = TargetSessions(groups, links, Audit())
            sessions.rotate()
            self.assertEqual([g.rotations for g in groups.values()], [1, 1])
            self.assertEqual(
                [s.migration_id for s in sessions.sessions],
                ["run-B-g2-tp2-0001", "run-A-g3-tp2-0001"],
            )
            sessions.close()
            self.assertTrue(all(g.closed for g in groups.values()))

    def test_the_entry_never_moves_onto_an_armed_or_uncontrolled_session(self):
        with tempfile.TemporaryDirectory() as folder:
            run, groups, links = self.layout(folder)
            a_active = run / "migration" / "A-g3-tp2" / "migration_active"
            control = a_active.resolve() / "runtime_control.json"
            control.write_text(json.dumps(dict(armed=True)))
            with self.assertRaisesRegex(ValueError, "not explicitly disarmed"):
                switch_source_entry(links["C-g1-tp1"], a_active)
            control.unlink()
            with self.assertRaisesRegex(ValueError, "not explicitly disarmed"):
                switch_source_entry(links["C-g1-tp1"], a_active)
            self.assertNotEqual(Path(os.readlink(links["C-g1-tp1"])), a_active)

    def test_entry_switch_stays_inside_owned_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            run, groups, links = self.layout(folder)
            link = links["C-g1-tp1"]
            b_active = run / "migration" / "B-g2-tp2" / "migration_active"
            self.assertFalse(switch_source_entry(link, b_active))
            outside = Path(folder) / "elsewhere" / "migration_active"
            with self.assertRaisesRegex(ValueError, "escaped"):
                switch_source_entry(link, outside)
            with self.assertRaisesRegex(ValueError, "escaped"):
                switch_source_entry(
                    run / "other" / "C-g1-tp1",
                    run / "migration" / "A-g3-tp2" / "migration_active",
                )
            with self.assertRaisesRegex(ValueError, "no owned entry"):
                TargetSessions(groups, links, Audit()).start_on(
                    "A-g3-tp2", NS(request_id="q", owner="unknown"), None
                )


class PlanTests(unittest.TestCase):
    def test_all_lets_both_receivers_execute_with_their_ready_ports(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder))
            self.assertEqual(plan["receiver_stage"], 3)
            self.assertEqual(
                [t["execution_allowed"] for t in plan["migration_targets"]],
                [True, True],
            )
            argv = plan["controller_argv"]
            self.assertNotIn("--mechanism-root", argv)
            self.assertEqual(argv[argv.index("--migration-targets") + 1], "all")
            receivers_argv = [
                argv[i + 1] for i, v in enumerate(argv) if v == "--migration-target"
            ]
            self.assertEqual(receivers_argv, ["B-g2-tp2,30300", "A-g3-tp2,30301"])
            self.assertTrue(plan["migration_plan"]["multi_target_execution"])
            self.assertEqual(
                plan["migration_plan"]["execution_targets"], ["B-g2-tp2", "A-g3-tp2"]
            )

    def test_a_forced_receiver_is_recorded_and_must_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "all", "A-g3-tp2")
            argv = plan["controller_argv"]
            self.assertEqual(argv[-2:], ["--force-migration-target", "A-g3-tp2"])
            self.assertEqual(plan["migration_plan"]["execution_targets"], ["A-g3-tp2"])
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(ValueError, "not a receiver"),
        ):
            build(Path(folder), "all", "C-g1-tp1")
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(ValueError, "forced id"),
        ):
            build(Path(folder), "B", "A-g3-tp2")

    def test_b_mode_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build_b(Path(folder))
            self.assertEqual(plan["receiver_stage"], 2)
            self.assertEqual(plan["controller_argv"][-2:], ["--migration-targets", "B"])

    def test_controller_receiver_arguments(self):
        args = NS(migration_target=["B-g2-tp2,30300", "A-g3-tp2,30301"])
        self.assertEqual(
            pilot.migration_receivers(args), {"B-g2-tp2": 30300, "A-g3-tp2": 30301}
        )
        for bad in (
            ["B-g2-tp2"],
            ["B-g2-tp2,x"],
            ["B-g2-tp2,1", "B-g2-tp2,2"],
            ["B-g2-tp2,1", "A-g3-tp2,1"],
        ):
            with self.assertRaises(ValueError):
                pilot.migration_receivers(NS(migration_target=bad))

    def test_acceptance_rejects_a_move_outside_the_forced_receiver(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "all", "A-g3-tp2")
            active = {
                r["instance_id"]: dict(lifecycle="ACTIVE", inventory={})
                for r in plan["instances"]["instances"]
            }
            requests = [
                dict(
                    job_id="one",
                    source_done=True,
                    source_error=None,
                    response=dict(emitted=[1]),
                    request=dict(max_tokens=1),
                )
            ]

            def read(path):
                return (
                    dict(instances=active)
                    if path.name == "layout_status.json"
                    else requests
                )

            with patch.object(six.pc, "read_json", side_effect=read):
                result = six.assess(
                    plan,
                    dict(
                        sessions=[
                            dict(
                                state="COMMITTED",
                                target_instance="A-g3-tp2",
                                source_release=True,
                            )
                        ]
                    ),
                    1,
                )
                self.assertEqual(result["migration_targets_used"], ["A-g3-tp2"])
                with self.assertRaisesRegex(ValueError, "outside the allowed"):
                    six.assess(
                        plan,
                        dict(
                            sessions=[
                                dict(state="COMMITTED", target_instance="B-g2-tp2")
                            ]
                        ),
                        1,
                    )

    def test_proof_accepts_sessions_in_either_executing_root(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder))
            plan["run_dir"].mkdir(parents=True)
            rows = []
            for n, target in enumerate(plan["migration_targets"]):
                session = Path(target["mechanism_root"]) / "sessions" / "migration-0001"
                session.mkdir(parents=True)
                (session / "takeover_state.json").write_text(
                    json.dumps(dict(state="CANCELLED"))
                )
                rows.append(
                    dict(
                        started=True,
                        run_dir=str(session),
                        migration_id=f"m{n}",
                        settled=True,
                    )
                )
            audit = [dict(kind="pressure_actual_start")] * 2 + [
                dict(kind="pressure_final", outcome="COMPLETED", sessions=rows),
            ]
            (plan["run_dir"] / "pressure_audit.jsonl").write_text(
                "\n".join(json.dumps(r) for r in audit)
            )
            proof = runner.closed_loop_proof(plan, "migrate")
            self.assertEqual(
                [s["target_instance"] for s in proof["sessions"]],
                ["B-g2-tp2", "A-g3-tp2"],
            )


class ControllerLoopTests(unittest.TestCase):
    """The real pilot loop with all receivers, sessions replaced (no START)."""

    def test_all_receivers_build_one_session_group_each_and_serve(self):
        import sys
        import time

        from tests.bridge_tp.test_performance_trace import journal
        from tests.bridge_tp.test_pressure_policy import config
        from tests.bridge_tp.test_runtime_instance_ids import FakeInventory
        from tools.bridge_tp.pressure_cpu import load

        ingress = load("pressure_ingress")
        inventory = load("pressure_inventory_process")
        observation = load("pressure_observation")
        actual_ingress = ingress.PressureIngress
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = build(root)
            directory = plan["run_dir"]
            (directory / "source_active").mkdir(parents=True)
            for name in plan["source_links"]:
                (directory / "source_active" / name).write_text("entry")
            manifest = plan["instances"]
            names = [row["instance_id"] for row in manifest["instances"]]
            argv = plan["controller_argv"]
            receivers_argv = [
                argv[i + 1] for i, v in enumerate(argv) if v == "--migration-target"
            ]
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
                maximum_migrations=0,
                routing_strategy="pressure",
                routing_weights=[1, 1],
                test_only_cancel_first_shadow=False,
                scheduler="unified",
                tp1_max_num_seqs=16,
                tp2_max_num_seqs=16,
                health_reference=None,
                migration_targets="all",
                migration_target=receivers_argv,
                force_migration_target=None,
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
            built = []

            def group(**kwargs):
                built.append(kwargs)
                return NS(
                    sessions=[],
                    record=None,
                    stream=None,
                    target_future=None,
                    settled=True,
                    migration_id=kwargs["migration_id"] + "-0001",
                    ready_for_start=True,
                    rotate=lambda: False,
                    cleanup_requires_shutdown=False,
                    cancel_future=None,
                    close=lambda: None,
                    root=kwargs["run_dir"],
                    bind_target=lambda *a: None,
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

            def post(url, payload, timeout, sink, *args, **kwargs):
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
                    side_effect=group,
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
                        config(),
                        controller,
                        NS(release_budget=17),
                        jobs,
                        dict(instance_manifest=manifest),
                    ),
                )
            targets = [t["instance_id"] for t in plan["migration_targets"]]
            self.assertEqual(
                [b["run_dir"] for b in built],
                [directory / "migration" / t for t in targets],
            )
            self.assertEqual(
                [b["migration_id"] for b in built], [f"cpu-{t}" for t in targets]
            )
            self.assertEqual(
                [b["config"].ready_notification_port for b in built], [30300, 30301]
            )
            self.assertEqual([b["config"].target_pool_id for b in built], targets)
            results = json.loads(
                (directory / "pressure_request_results.json").read_text()
            )
            self.assertEqual({row["initial_pool"] for row in results}, set(names))
            audit = [
                json.loads(line)
                for line in (directory / "pressure_audit.jsonl")
                .read_text()
                .splitlines()
            ]
            decisions = [
                row for row in audit if row["kind"] == "pressure_decision_unified"
            ]
            self.assertTrue(decisions)
            self.assertTrue(
                any(set(targets) <= set(row["readings"]) for row in decisions)
            )


if __name__ == "__main__":
    unittest.main()
