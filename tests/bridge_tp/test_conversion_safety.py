# SPDX-License-Identifier: Apache-2.0
"""S5a: start gate, drain assist and safety valve of a live conversion (CPU)."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from tests.bridge_tp.test_pressure_conversion import Harness, jobs
from tests.bridge_tp.test_pressure_policy import candidate, config, pool
from tools.bridge_tp.pressure_cpu import load

r = load("pressure_runtime")

PLAN = dict(
    plan_id="manual-A",
    bundle_id="A",
    target_bundle_mode="TP2",
    refs=[
        dict(instance_id="A-g1-tp1", instance_generation=1),
        dict(instance_id="A-g3-tp1", instance_generation=3),
    ],
)


def row(request_id, remaining, tokens_s=20.0, action=True):
    return dict(
        request_id=request_id,
        remaining_bins=[(remaining, 1.0)],
        tokens_s=tokens_s,
        action_available=action,
        transfer_bytes=10**8,
    )


def status(
    *,
    target_zone="SPARE",
    target_capacity=0.1,
    target_sessions=0,
    requests=(),
    assist_result=None,
    target="B-g2-tp2",
    age=0.0,
):
    quiet = dict(zone="SPARE", capacity=0.1)
    return dict(
        unix_s=time.time() - age,
        release_budget_s=8.0,
        migration_target=target,
        assist_result=assist_result,
        instances={
            "A-g1-tp1": dict(
                lifecycle="QUIESCING",
                sessions=0,
                requests=list(requests),
                pressure=quiet,
            ),
            "A-g3-tp1": dict(
                lifecycle="QUIESCING", sessions=0, requests=[], pressure=quiet
            ),
            "B-g2-tp2": dict(
                lifecycle="ACTIVE",
                sessions=target_sessions,
                requests=[],
                pressure=dict(zone=target_zone, capacity=target_capacity),
            ),
        },
    )


class Base(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.harness = Harness(self.root, "A", "TP2")
        self.backend = self.harness.backend
        self.now = [time.time()]
        self.backend.clock = lambda: self.now[0]

    def tearDown(self):
        self.directory.cleanup()

    def write(self, value):
        value["unix_s"] = self.now[0] - (time.time() - value["unix_s"])
        (self.root / "layout_status.json").write_text(json.dumps(value))

    def decisions(self, kind):
        path = self.root / "conversion_decisions.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        return [x for x in rows if x["kind"] == kind]


class GateTests(Base):
    def test_quiet_retained_side_opens_at_once(self):
        self.write(status())
        self.assertTrue(self.backend.gate(PLAN))
        self.assertFalse(self.decisions("GATE")[-1]["relaxed"])

    def test_busy_retained_side_waits_then_relaxes_to_hard_safety(self):
        self.write(status(target_zone="CRITICAL"))
        self.assertFalse(self.backend.gate(PLAN))
        self.assertIn("quieter", self.decisions("GATE")[-1]["reason"])
        self.now[0] += 181
        self.write(status(target_zone="CRITICAL"))
        self.assertTrue(self.backend.gate(PLAN))
        self.assertTrue(self.decisions("GATE")[-1]["relaxed"])

    def test_kv_headroom_is_never_relaxed(self):
        for _ in range(2):
            self.write(status(target_capacity=0.85))
            self.assertFalse(self.backend.gate(PLAN))
            self.assertIn("hard", self.decisions("GATE")[-1]["reason"])
            self.now[0] += 400

    def test_stale_or_missing_status_keeps_it_closed(self):
        self.assertFalse(self.backend.gate(PLAN))
        self.write(status(age=5))
        self.assertFalse(self.backend.gate(PLAN))
        self.assertEqual(self.decisions("GATE")[-1]["reason"], "status stale")


class ValveTests(Base):
    def test_sustained_kv_pressure_on_retained_side_aborts(self):
        self.write(status(target_capacity=1.1))
        self.assertFalse(self.backend.kv_emergency(PLAN))
        self.now[0] += 2
        self.write(status(target_capacity=1.1))
        self.assertFalse(self.backend.kv_emergency(PLAN))
        self.now[0] += 1.5
        self.write(status(target_capacity=1.1))
        self.assertTrue(self.backend.kv_emergency(PLAN))
        self.assertTrue(self.decisions("VALVE_ABORT"))

    def test_recovery_resets_and_service_quality_alone_never_aborts(self):
        self.write(status(target_capacity=1.1))
        self.assertFalse(self.backend.kv_emergency(PLAN))
        self.now[0] += 2
        self.write(status(target_capacity=0.5, target_zone="OVERLOAD"))
        self.assertFalse(self.backend.kv_emergency(PLAN))
        self.now[0] += 5
        self.write(status(target_capacity=1.1))
        self.assertFalse(self.backend.kv_emergency(PLAN))
        self.write(status(target_capacity=None))
        self.now[0] += 10
        self.assertFalse(self.backend.kv_emergency(PLAN))


class AssistChoiceTests(Base):
    def choice(self, value):
        self.write(value)
        return self.backend.assist_choice(PLAN, self.backend.status())

    def test_longest_request_worth_migrating_is_chosen(self):
        reason, choice = self.choice(
            status(requests=[row("a", 2000), row("b", 6000), row("c", 300)])
        )
        self.assertIsNone(reason)
        self.assertEqual(choice["request_id"], "b")
        self.assertAlmostEqual(choice["natural_s"], 6000 / 16)

    def test_short_remaining_is_left_to_finish(self):
        # 3 x 8s budget = 24s; 380 tokens at 16 tok/s = 23.75s.
        reason, choice = self.choice(status(requests=[row("a", 380)]))
        self.assertIsNone(choice)
        self.assertIn("worth", reason)

    def test_target_service_quality_is_not_traded(self):
        for kwargs, text in (
            (dict(target_zone="CRITICAL"), "SPARE"),
            (dict(target_sessions=1), "session"),
            (dict(target=None), "target"),
            (dict(age=5), "stale"),
        ):
            reason, choice = self.choice(status(requests=[row("a", 6000)], **kwargs))
            self.assertIsNone(choice)
            self.assertIn(text, reason)

    def test_unstartable_or_unmeasured_requests_are_skipped(self):
        reason, choice = self.choice(
            status(requests=[row("a", 6000, action=False), row("b", 6000, None)])
        )
        self.assertIsNone(choice)

    def test_one_at_a_time_and_capped_per_conversion(self):
        self.write(status(requests=[row("a", 6000), row("b", 5000)]))
        self.backend.assisted_step(PLAN, self.now[0])
        control = json.loads((self.root / "layout_control.json").read_text())
        self.assertEqual(
            control["assist"], dict(seq=1, request_id="a", source="A-g1-tp1")
        )
        self.now[0] += 1.1
        self.write(status(requests=[row("a", 6000), row("b", 5000)]))
        self.backend.assisted_step(PLAN, self.now[0])
        self.assertEqual(self.backend.assist_seq, 1)  # unanswered: wait
        for seq, rid in ((1, "a"), (2, "b")):
            self.now[0] += 1.1
            self.write(
                status(
                    requests=[row("b", 5000), row("c", 4000)],
                    assist_result=dict(seq=seq, request_id=rid, state="STARTED"),
                )
            )
            self.backend.assisted_step(PLAN, self.now[0])
        self.assertEqual(self.backend.assisted, ["a", "b"])
        self.now[0] += 1.1
        reason, choice = self.choice(
            status(
                requests=[row("c", 4000)],
                assist_result=dict(seq=2, request_id="b", state="STARTED"),
            )
        )
        self.assertIsNone(choice)
        self.assertIn("cap", reason)

    def test_a_refused_assist_is_not_counted(self):
        self.backend.assist_seq = 1
        reason, choice = self.choice(
            status(
                requests=[row("a", 6000)],
                assist_result=dict(seq=1, request_id="a", state="REFUSED"),
            )
        )
        self.assertIsNone(reason)
        self.assertEqual(self.backend.assisted, [])


class CoordinatorAssistTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        self.co = r.PressureCoordinator(config(), self.records.append)
        self.started = []
        for instance, state in (("tp1", "QUIESCING"),):
            self.co.set_lifecycle(instance, state)

    def action(self, request_id="r", session="s1"):
        return r.BoundAction(request_id, session, lambda: self.started.append(session))

    def start(self, **kwargs):
        values = dict(
            now=10.0,
            source="tp1",
            target="tp4",
            candidate=candidate("r"),
            action=self.action(),
            target_reading=pool("tp4", sampled_at=9.5),
            required_blocks=500,
        )
        values.update(kwargs)
        return self.co.assist_start(**values)

    def test_starts_through_the_reservation_path(self):
        self.assertIsNone(self.start())
        self.assertEqual(self.started, ["s1"])
        self.assertEqual(self.co._sessions["s1"], ("r", "tp4"))
        self.assertEqual(self.co._session_source["s1"], "tp1")
        entry = self.co.ledger.entries["s1"]
        self.assertEqual(
            (entry.blocks, entry.kind, entry.pool_id), (500, "migration", "tp4")
        )
        kinds = [x["kind"] for x in self.records]
        self.assertIn("pressure_conversion_assist", kinds)
        self.assertIn("pressure_start_published", kinds)
        # One global session: a second assist is refused.
        self.assertIn(
            "active",
            self.start(candidate=candidate("q"), action=self.action("q", "s2")),
        )

    def test_refusals_leave_no_reservation(self):
        cases = (
            (dict(source="tp9"), "quiescing"),
            (dict(action=None), "binding"),
            (dict(action=self.action("other")), "binding"),
            (dict(target_reading=pool("tp4", sampled_at=1.0)), "stale"),
            (dict(required_blocks=5000), "hold"),
        )
        for kwargs, text in cases:
            self.assertIn(text, self.start(**kwargs))
        self.co.set_lifecycle("tp4", "QUIESCING")
        self.assertIn("target", self.start())
        self.assertEqual(self.co.ledger.entries, {})
        self.assertEqual(self.started, [])


class EndToEndTests(unittest.TestCase):
    def test_rolling_method_runs_the_assist_loop_and_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = Harness(
                Path(directory), "A", "TP2", channel_plan=True, method="SELECTIVE_ROLL"
            )
            outcome = harness.run(jobs(10, 0.3))
            self.assertEqual(outcome["state"], "COMPLETED", outcome)
            self.assertEqual(outcome["plan"]["method"], "SELECTIVE_ROLL")
            rows = [
                json.loads(x)
                for x in (Path(directory) / "conversion_decisions.jsonl")
                .read_text()
                .splitlines()
            ]
            # The harness requests carry no rates: nothing worth migrating.
            assists = [x for x in rows if x["kind"] == "ASSIST"]
            self.assertTrue(assists)
            self.assertTrue(all(x["choice"] is None for x in assists))
            self.assertTrue(any(x["kind"] == "GATE" and x["open"] for x in rows))

    def test_valve_abort_restores_routing_before_anything_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = Harness(
                root,
                "A",
                "TP2",
                channel_plan=True,
                slow={"source"},
                safety=dict(valve_capacity=0.0, valve_sustain_s=0.0),
            )
            outcome = harness.run(jobs(16, 0.3, busy="tp1"))
            self.assertEqual(outcome["state"], "ABORTED", outcome)
            self.assertEqual(harness.stopped, set())
            kinds = [e["kind"] for e in harness.events]
            self.assertIn("SAFETY_ABORT", kinds)
            self.assertIn("ADMISSION_RESTORED", kinds)
            self.assertNotIn("STOP_RETURN", kinds)
            result = json.loads((root / "pressure_request_results.json").read_text())
            self.assertTrue(all(x["source_done"] for x in result))
            status = json.loads((root / "layout_status.json").read_text())
            for instance in ("A-g1-tp1", "A-g3-tp1"):
                self.assertEqual(status["instances"][instance]["lifecycle"], "ACTIVE")
            audit = [
                json.loads(x)
                for x in (root / "pressure_audit.jsonl").read_text().splitlines()
            ]
            restored = max(
                x["unix_s"]
                for x in audit
                if x["kind"] == "pressure_instance_lifecycle" and x["to"] == "ACTIVE"
            )
            later = {
                x["pool"]
                for x in audit
                if x["kind"] == "pressure_dispatch"
                and x["admission_finished_unix_s"] > restored
            }
            self.assertTrue(later & {"tp1", "A-g3-tp1"}, later)


if __name__ == "__main__":
    unittest.main()
