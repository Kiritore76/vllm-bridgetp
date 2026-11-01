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
    assist_sessions=None,
    target="B-g2-tp2",
    age=0.0,
):
    quiet = dict(zone="SPARE", capacity=0.1)
    return dict(
        unix_s=time.time() - age,
        release_budget_s=8.0,
        migration_target=target,
        assist_result=assist_result,
        assist_sessions=assist_sessions or {},
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


class ValveRemainingTests(Base):
    """User 11-01: when the valve trips near the end, finish the conversion."""

    def shadow(self, ready, status="SUPPORTED", age=0.0, error=False):
        self.backend.last_shadow = (
            dict(evaluated_unix_s=self.now[0] - age, error="boom")
            if error
            else dict(
                evaluated_unix_s=self.now[0] - age,
                drain_only=dict(support_status=status, estimated_ready_s=ready),
            )
        )

    def trip(self):
        self.write(status(target_capacity=1.1))
        self.assertFalse(self.backend.kv_emergency(PLAN))
        self.now[0] += 3.5
        self.write(status(target_capacity=1.1))

    def test_nearly_done_finishes_instead_of_cancelling(self):
        self.trip()
        self.shadow(45.0)
        self.assertFalse(self.backend.kv_emergency(PLAN))
        self.assertFalse(self.backend.kv_emergency(PLAN))
        rows = self.decisions("VALVE_CONTINUE")
        self.assertEqual(len(rows), 1)  # logged once per conversion
        self.assertEqual(rows[0]["remaining_s"], 45.0)
        self.assertIn("SAFETY_CONTINUE", [e["kind"] for e in self.harness.events])

    def test_far_from_done_still_cancels(self):
        self.trip()
        self.shadow(150.0)
        self.assertTrue(self.backend.kv_emergency(PLAN))
        self.assertEqual(self.decisions("VALVE_ABORT")[-1]["remaining_s"], 150.0)

    def test_no_usable_estimate_cancels_as_before(self):
        for kwargs in (
            dict(ready=None, status="UNKNOWN"),
            dict(ready=10.0, age=5.0),
            dict(ready=None, error=True),
        ):
            self.trip()
            self.shadow(**kwargs)
            self.assertTrue(self.backend.kv_emergency(PLAN), kwargs)
            self.backend.valve_since = None
        self.backend.last_shadow = None
        self.trip()
        self.assertTrue(self.backend.kv_emergency(PLAN))

    def test_threshold_is_configurable(self):
        self.backend.safety["continue_if_remaining_s"] = 30.0
        self.trip()
        self.shadow(45.0)
        self.assertTrue(self.backend.kv_emergency(PLAN))


class AssistChoiceTests(Base):
    """All-or-nothing evacuation (user 11-01, after S5a GPU run 4)."""

    def choice(self, value):
        self.write(value)
        return self.backend.assist_choice(PLAN, self.backend.status())

    def answer(self, seq, rid, state="STARTED", **kwargs):
        self.now[0] += 1.1
        return self.choice(
            status(assist_result=dict(seq=seq, request_id=rid, state=state), **kwargs)
        )

    def propose(self, requests):
        """Run one decision step and return the published assist."""
        self.write(status(requests=requests))
        self.backend.assisted_step(PLAN, self.now[0])
        return json.loads((self.root / "layout_control.json").read_text())["assist"]

    def test_plans_every_long_request_longest_first(self):
        reason, choice = self.choice(
            status(requests=[row("a", 2000), row("b", 6000), row("c", 300)])
        )
        self.assertIsNone(reason)
        self.assertEqual(choice["request_id"], "b")
        # c: 300 tokens at 16 tok/s = 18.75s <= 3 x 8s; it finishes anyway.
        self.assertEqual(choice["evacuate"], ["b", "a"])

    def test_nothing_long_means_no_evacuation(self):
        reason, choice = self.choice(status(requests=[row("a", 380)]))
        self.assertIsNone(choice)
        self.assertIn("worth", reason)

    def test_a_long_request_that_cannot_move_blocks_the_whole_plan(self):
        for blocked in (row("x", 6000, action=False), row("x", 6000, None)):
            reason, choice = self.choice(status(requests=[row("a", 6000), blocked]))
            self.assertIsNone(choice)
            self.assertIn("all or none", reason)

    def test_more_long_requests_than_the_cap_drain_naturally(self):
        reason, choice = self.choice(
            status(requests=[row(f"r{n}", 5000 + n) for n in range(5)])
        )
        self.assertIsNone(choice)
        self.assertIn("exceed the cap 4", reason)

    def test_the_plan_needs_a_quiet_safe_target(self):
        for kwargs, text in (
            (dict(target_zone="CRITICAL"), "SPARE"),
            (dict(target_sessions=1), "session"),
            (dict(target=None), "target"),
            (dict(age=5), "stale"),
        ):
            reason, choice = self.choice(status(requests=[row("a", 6000)], **kwargs))
            self.assertIsNone(choice)
            self.assertIn(text, reason)

    def test_an_accepted_plan_moves_every_request_to_completion(self):
        assist = self.propose([row("a", 6000), row("b", 5000)])
        self.assertEqual(
            assist,
            dict(seq=1, request_id="a", source="A-g1-tp1", evacuate=["a", "b"]),
        )
        # Accepted. Mid-evacuation the target need not be SPARE any more: the
        # controller's per-START KV and service checks remain the guard.
        reason, choice = self.answer(
            1, "a", requests=[row("b", 5000)], target_zone="CRITICAL"
        )
        self.assertIsNone(reason)
        self.assertEqual(self.backend.evacuation, ["a", "b"])
        self.assertEqual((choice["request_id"], "evacuate" in choice), ("b", False))
        self.backend.assist_seq = 2
        reason, choice = self.answer(2, "b", requests=[row("c", 4000)])
        self.assertIsNone(choice)
        self.assertEqual(reason, "evacuation complete")  # c was not planned

    def test_a_refused_plan_starts_nothing_and_may_be_proposed_again(self):
        self.propose([row("a", 6000), row("b", 5000)])
        reason, choice = self.answer(
            1, "a", state="REFUSED", requests=[row("a", 6000), row("b", 5000)]
        )
        self.assertIsNone(self.backend.evacuation)
        self.assertEqual(self.backend.assisted, [])
        self.assertEqual(choice["evacuate"], ["a", "b"])

    def test_a_refused_step_is_retried_and_ends_only_if_refusals_persist(self):
        """S5a GPU (11-01): a 512-token prompt queued on the target for 0.2s
        refused step 4 of 4 and ended the evacuation."""
        self.propose([row("a", 6000), row("b", 5000)])
        self.answer(1, "a", requests=[row("b", 5000)])
        self.backend.assist_seq = 2
        reason, choice = self.answer(2, "b", state="REFUSED", requests=[row("b", 5000)])
        self.assertIsNone(reason)  # retried at once under the same checks
        self.assertEqual(choice["request_id"], "b")
        self.now[0] += 25
        self.assertIsNotNone(
            self.answer(2, "b", state="REFUSED", requests=[row("b", 5000)])[1]
        )
        self.now[0] += 10  # refused for more than 30s
        reason, choice = self.answer(2, "b", state="REFUSED", requests=[row("b", 5000)])
        self.assertIsNone(choice)
        self.assertIn("too long", reason)

    def test_a_start_resets_the_refusal_clock(self):
        self.propose([row("a", 6000), row("b", 5000), row("c", 4000)])
        self.answer(1, "a", requests=[row("b", 5000), row("c", 4000)])
        self.backend.assist_seq = 2
        self.answer(2, "b", state="REFUSED", requests=[row("b", 5000), row("c", 4000)])
        self.now[0] += 25
        self.answer(2, "b", requests=[row("c", 4000)])  # started after all
        self.assertIsNone(self.backend.refused_since)
        self.backend.assist_seq = 3
        self.now[0] += 25
        reason, choice = self.answer(3, "c", state="REFUSED", requests=[row("c", 4000)])
        self.assertEqual(choice["request_id"], "c")

    def test_committed_and_attempt_caps(self):
        reason, choice = self.choice(
            status(
                requests=[row("c", 4000)],
                assist_sessions=dict(s1="COMMITTED", s2="ACCOUNTED", s3="COMPLETED")
                | {"s4": "COMMITTED"},
            )
        )
        self.assertIn("committed", reason)
        self.backend.assisted = ["a"] * 8
        reason, choice = self.choice(status(requests=[row("c", 4000)]))
        self.assertIn("attempts", reason)


class EvacuationRefusalTests(unittest.TestCase):
    def check(self, free=4000, pending=0, costs=(0.1, 0.1), **kwargs):
        from types import SimpleNamespace as NS

        from tools.bridge_tp import run_pressure_balanced_pilot as pilot

        candidates = [
            NS(
                prompt_tokens=16384,
                max_output_tokens=8192,
                target_decode_cost=c,
                target_health_after_supported=False,
            )
            for c in costs
        ]
        values = dict(
            candidates=candidates,
            block_size=16,
            guard_blocks=125,
            pending_blocks=pending,
            pending_decode=0.0,
            service=True,
        )
        values.update(kwargs)
        return pilot.evacuation_refusal(
            config(workload_pressure=True, health_service=True),
            NS(free_blocks=free, pending_prompt_tokens=0, waiting=0),
            NS(decode=0.3, prefill=0.0),
            **values,
        )

    def test_every_final_length_must_fit_with_the_guard(self):
        # Two 24576-token requests need 3072 blocks; + 125 guard = 3197.
        self.assertIsNone(self.check(free=3197))
        self.assertIn("KV", self.check(free=3196))
        self.assertIn("KV", self.check(free=4000, pending=900))

    def test_the_service_budget_counts_every_added_decode(self):
        self.assertIsNone(self.check(costs=(0.3, 0.3)))  # 0.3 + 0.6 <= 1.0
        self.assertIn("budget", self.check(costs=(0.4, 0.4)))  # 1.1
        self.assertIn("budget", self.check(costs=(0.1, None)))
        self.assertIsNone(self.check(costs=(0.4, 0.4), service=False))


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
        # Marked as a drain assist: relief-cancel rules do not apply to it.
        self.assertEqual(self.co._assist_sessions, {"s1"})
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


class UnsettledTests(unittest.TestCase):
    def test_accounted_admission_keeps_the_pool_unsettled(self):
        """S5a GPU drain arm (11-01): retiring a pool with an ACCOUNTED
        admission stranded it; the controller then never finished."""
        from types import SimpleNamespace as NS

        from tools.bridge_tp import run_pressure_balanced_pilot as pilot

        entries = [
            NS(pool_id="tp1", state="ACCOUNTED"),
            NS(pool_id="tp1", state="MOVED"),
            NS(pool_id="tp1", state="COMPLETED"),
            NS(pool_id="A-g3-tp1", state="ACTIVE"),
            NS(pool_id="A-g3-tp1", state="COMMITTED"),
        ]
        self.assertEqual(pilot.unsettled_on(entries, "tp1"), 1)
        self.assertEqual(pilot.unsettled_on(entries, "A-g3-tp1"), 2)
        self.assertEqual(pilot.unsettled_on(entries, "tp4"), 0)
        # Same states the controller's own completion check waits on.
        source = Path(pilot.__file__).read_text(encoding="utf-8")
        self.assertIn('row.state in {"ACTIVE", "COMMITTED", "ACCOUNTED"}', source)
        self.assertEqual(pilot.UNSETTLED, {"ACTIVE", "COMMITTED", "ACCOUNTED"})


class AssistServiceTests(unittest.TestCase):
    """The controller's post-move service check uses the active policy rule."""

    def refusal(self, cfg=None, after=None, pending=0.0, **kwargs):
        from types import SimpleNamespace as NS

        from tools.bridge_tp import run_pressure_balanced_pilot as pilot

        values = dict(target_decode_cost=0.2, target_health_after_supported=False)
        values.update(kwargs)
        return pilot.assist_service_refusal(
            cfg or config(workload_pressure=True, health_service=True),
            NS(pending_prompt_tokens=0, waiting=0),
            after or NS(decode=0.35, prefill=0.0),
            pending,
            NS(**values),
        )

    def test_long_request_outside_the_reference_passes_under_workload_rule(self):
        """S5a GPU (11-01): all 87 assists were refused on the health flag,
        which is always False for >8K contexts the reference never measured."""
        self.assertIsNone(self.refusal())

    def test_budget_queue_and_unknown_cost_refuse(self):
        from types import SimpleNamespace as NS

        self.assertIn("budget", self.refusal(pending=0.6))
        self.assertIn("budget", self.refusal(target_decode_cost=None))
        self.assertIn("budget", self.refusal(after=NS(decode=None, prefill=0.0)))

    def test_without_workload_pressure_the_health_flag_is_required(self):
        cfg = config(health_service=True)
        self.assertIn("health", self.refusal(cfg=cfg))
        self.assertIsNone(self.refusal(cfg=cfg, target_health_after_supported=True))


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
