# SPDX-License-Identifier: Apache-2.0
"""Dynamic target/session inventory and receiver-scoped close receipts, CPU only."""

import json
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp.test_conversion_safety import PLAN, Base, status
from tests.bridge_tp.test_multi_target import Audit, Group
from tools.bridge_tp import pressure_conversion as pc
from tools.bridge_tp.conversion_channels import TargetChannelGraph
from tools.bridge_tp.conversion_sequence import SequenceBackend
from tools.bridge_tp.pressure_sessions import TargetSessions


class DynamicSessionsTests(unittest.TestCase):
    def setUp(self):
        self.a = Group(Path("/run/migration/A"), "A")
        self.b = Group(Path("/run/migration/B"), "B")
        self.audit = Audit()
        self.links = dict(s=Path("/run/source_active/s"))
        self.sessions = TargetSessions(
            dict(B=self.b), self.links, self.audit, dynamic=True
        )

    def test_add_target_preserves_group_and_current_session_identity(self):
        old_id = self.sessions.migration_id
        self.assertTrue(self.sessions.reconcile(dict(B=self.b, A=self.a), self.links))
        self.assertEqual(self.sessions.current, "B")
        self.assertEqual(self.sessions.migration_id, old_id)
        self.assertTrue(self.sessions.target_ready("A"))

    def test_active_settled_but_unrotated_or_uncertain_session_defers(self):
        for group in (self.a, self.b):
            group.record = NS(request_id="q", settled=True)
            self.assertFalse(
                self.sessions.reconcile(dict(B=self.b, A=self.a), self.links)
            )
            self.assertNotIn("A", self.sessions.groups)
            group.record = None
            group.cleanup_requires_shutdown = True
            self.assertFalse(
                self.sessions.reconcile(dict(B=self.b, A=self.a), self.links)
            )
            group.cleanup_requires_shutdown = False

    def test_retirement_preserves_old_stream_and_late_api_objects(self):
        future = Future()
        old = NS(
            record=NS(request_id="q"),
            target_future=future,
            run_dir=Path("/immutable/old"),
        )
        self.b.sessions.insert(0, old)
        self.assertTrue(self.sessions.reconcile(dict(A=self.a), self.links))
        self.assertTrue(self.b.closed)
        self.assertFalse(self.sessions.target_ready("B"))
        self.assertIn(("B", self.b), self.sessions.items())
        self.assertIn(old, self.sessions.sessions)
        self.assertFalse(future.cancelled())
        future.set_result("old stream finished")
        self.assertEqual(old.target_future.result(), "old stream finished")
        self.assertEqual(old.run_dir, Path("/immutable/old"))

    def test_zero_targets_can_later_gain_a_new_incarnation(self):
        self.assertTrue(self.sessions.reconcile({}, self.links))
        self.assertFalse(self.sessions.ready_for_start)
        self.assertIsNone(self.sessions.record)
        self.assertFalse(self.sessions.rotate())
        self.assertTrue(self.sessions.reconcile(dict(A=self.a), self.links))
        self.assertTrue(self.sessions.ready_for_start)
        self.assertEqual(self.sessions.current, "A")

    def test_initial_zero_targets_is_explicit_and_cannot_start(self):
        with self.assertRaises(ValueError):
            TargetSessions({}, {}, self.audit)
        empty = TargetSessions({}, {}, self.audit, dynamic=True)
        self.assertIsNone(empty.record)
        self.assertFalse(empty.ready_for_start)
        with self.assertRaisesRegex(ValueError, "not rotated"):
            empty.start_on("A", NS(owner="s"), None)

    def test_retired_target_or_source_identity_cannot_return(self):
        self.sessions.reconcile({}, {})
        with self.assertRaisesRegex(ValueError, "target identity"):
            self.sessions.reconcile(dict(B=self.b), {})
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.sessions.reconcile(dict(A=self.a), self.links)

    def test_live_group_or_source_entry_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "group cannot"):
            self.sessions.reconcile(dict(B=self.a), self.links)
        with self.assertRaisesRegex(ValueError, "entry cannot"):
            self.sessions.reconcile(dict(B=self.b), dict(s=Path("/other")))

    def test_next_start_switches_source_before_starting_new_receiver(self):
        self.sessions.reconcile(dict(A=self.a), self.links)
        seen = []

        def switch(*args, **kwargs):
            self.assertFalse(self.a.started)
            seen.append(kwargs["target"])

        with patch("tools.bridge_tp.pressure_sessions.switch_source_entry", switch):
            self.sessions.start_on("A", NS(owner="s", request_id="q"), None)
        self.assertEqual(seen, ["A"])
        self.assertEqual(self.a.started, ["q"])

    def test_static_groups_cannot_silently_enable_dynamic_changes(self):
        fixed = TargetSessions(dict(B=self.b), self.links, self.audit)
        with self.assertRaisesRegex(ValueError, "not enabled"):
            fixed.reconcile({}, {})

    def test_unchanged_topology_is_idempotent_during_a_session(self):
        self.b.record = NS(request_id="q")
        self.assertTrue(self.sessions.reconcile(dict(B=self.b), self.links))
        self.assertFalse(self.sessions.ready_for_start)

    def test_new_groups_and_source_entries_must_stay_in_the_owned_run(self):
        outside = Group(Path("/other/migration/A"), "A")
        with self.assertRaisesRegex(ValueError, "escaped owned run"):
            self.sessions.reconcile(dict(B=self.b, A=outside), self.links)
        with self.assertRaisesRegex(ValueError, "escaped owned run"):
            self.sessions.reconcile(
                dict(B=self.b), dict(other=Path("/other/source_active/other"))
            )
        self.sessions.reconcile({}, {})
        with self.assertRaisesRegex(ValueError, "escaped owned run"):
            self.sessions.reconcile(dict(A=outside), {})


class ChannelGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.now = 100.0
        self.events = []
        self.targets = {}
        for name in ("A", "B"):
            root = self.root / "migration" / name
            root.mkdir(parents=True)
            path = root / "gpu_channel_plan.json"
            self.write(
                path,
                [
                    dict(source="s1", generation=1, base_port=30000),
                    dict(source="s2", generation=2, base_port=30010),
                ],
            )
            self.targets[name] = dict(mechanism_root=root, channel_plan=path)
        self.files = {}
        for n, source in enumerate(("s1", "s2"), 1):
            path = self.root / f"targets_{source}.json"
            self.write(path, [dict(target=t, generation=n) for t in self.targets])
            self.files[source] = path
        self.graph = TargetChannelGraph(
            self.root,
            self.targets,
            self.files,
            clock=lambda: self.now,
            event=lambda k, **v: self.events.append((k, v)),
        )

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def receipt(
        self, target, rank, *, closed=(), planned=(), at=None, session="session1"
    ):
        path = (
            self.targets[target]["mechanism_root"]
            / session
            / "gpu_channel_plan_receipts"
            / f"tp_rank_{rank}.json"
        )
        self.write(
            path,
            dict(
                closed=list(closed),
                planned=list(planned),
                completed_unix_s=self.now if at is None else at,
            ),
        )

    def close_acks(self, target, generations):
        for rank in range(2):
            self.receipt(target, rank, closed=generations)
            for generation in generations:
                path = (
                    self.targets[target]["mechanism_root"]
                    / "session1/gpu_channel_plan_receipts"
                    / f"tp_rank_{rank}-retired-{generation}.json"
                )
                self.write(path, dict(status="ABORTED"))

    def test_precheck_does_not_mix_identical_rank_names_across_receivers(self):
        self.receipt("A", 0, planned=[1, 2])
        self.receipt("B", 1, planned=[1, 2])
        self.assertFalse(self.graph.acknowledged())
        self.receipt("A", 1, planned=[1, 2])
        self.receipt("B", 0, planned=[1, 2])
        self.assertTrue(self.graph.acknowledged())

    def test_latest_receipt_supersedes_old_session_ack(self):
        for target in self.targets:
            for rank in range(2):
                self.receipt(target, rank, planned=[1, 2], at=99, session="old")
        self.receipt("A", 0, planned=[], session="new")
        self.assertFalse(self.graph.acknowledged())

    def test_retiring_receiver_preserves_other_target_in_each_source_file(self):
        self.assertFalse(self.graph.retire("p", {"A"}))
        self.assertEqual(self.graph.rows("A"), [])
        self.assertEqual(len(self.graph.rows("B")), 2)
        for path in self.files.values():
            self.assertEqual([r["target"] for r in json.loads(path.read_text())], ["B"])
        self.close_acks("B", [1, 2])
        self.assertFalse(self.graph.retire("p", {"A"}))
        self.close_acks("A", [1, 2])
        self.assertTrue(self.graph.retire("p", {"A"}))
        self.graph.remove_target("A")
        self.assertEqual(set(self.graph.targets), {"B"})

    def test_retiring_source_closes_only_its_edges_on_every_receiver(self):
        self.assertFalse(self.graph.retire("p", {"s1"}))
        for target in self.targets:
            self.assertEqual([r["source"] for r in self.graph.rows(target)], ["s2"])
        self.assertEqual(json.loads(self.files["s1"].read_text()), [])
        self.assertEqual(len(json.loads(self.files["s2"].read_text())), 2)
        self.close_acks("A", [1])
        self.assertFalse(self.graph.retire("p", {"s1"}))
        self.close_acks("B", [1])
        self.assertTrue(self.graph.retire("p", {"s1"}))

    def test_stale_ack_and_unexpected_rank_never_close_a_channel(self):
        self.graph.retire("p", {"A"})
        self.receipt("A", 0, closed=[1, 2], at=99)
        self.receipt("A", 1, closed=[1, 2])
        self.receipt("A", 2, closed=[1, 2])
        self.assertFalse(self.graph.retire("p", {"A"}))
        self.now += 61
        with self.assertRaisesRegex(RuntimeError, "A.*0"):
            self.graph.retire("p", {"A"})

    def test_teardown_wait_is_bounded_and_audited(self):
        self.graph.retire("p", {"A"})
        for rank in range(2):
            self.receipt("A", rank, closed=[1, 2])
        self.assertFalse(self.graph.retire("p", {"A"}))
        self.now += 10
        self.assertTrue(self.graph.retire("p", {"A"}))
        self.assertFalse(self.events[-1][1]["teardown_complete"])

    def test_session_rotation_does_not_erase_verified_close_proof(self):
        self.graph.retire("p", {"A"})
        self.close_acks("A", [1, 2])
        self.now += 1
        # The new session's receipt is later but has no history of closed edges.
        for rank in range(2):
            self.receipt("A", rank, session="new", closed=[])
        self.assertTrue(self.graph.retire("p", {"A"}))

    def test_pending_plan_cannot_be_overwritten_or_receiver_removed_early(self):
        self.graph.retire("p", {"A"})
        with self.assertRaises(ValueError):
            self.graph.retire("q", {"B"})
        with self.assertRaises(ValueError):
            self.graph.retire("p", {"B"})
        with self.assertRaises(ValueError):
            self.graph.remove_target("A")

    def test_completed_plan_is_idempotent_and_next_plan_uses_remaining_graph(self):
        self.graph.retire("p", {"A"})
        self.close_acks("A", [1, 2])
        self.assertTrue(self.graph.retire("p", {"A"}))
        count = len(self.events)
        self.assertTrue(self.graph.retire("p", {"A"}))
        self.assertEqual(count, len(self.events))
        self.graph.remove_target("A")
        self.assertFalse(self.graph.retire("q", {"s2"}))
        self.close_acks("B", [2])
        self.assertTrue(self.graph.retire("q", {"s2"}))

    def test_owned_paths_and_target_incarnations_are_enforced(self):
        with self.assertRaises(ValueError):
            self.graph.add_target(
                "X",
                mechanism_root=self.root / "outside",
                channel_plan=self.root / "outside/plan.json",
            )
        with self.assertRaises(ValueError):
            self.graph.add_target("A", **self.targets["A"])

    def test_future_and_nonfinite_receipts_cannot_ack_a_plan(self):
        for invalid in (101, float("inf"), float("nan"), True, None):
            for target in self.targets:
                for rank in range(2):
                    self.receipt(target, rank, planned=[1, 2], at=invalid)
            # receipt() uses current time for None; remove the timestamp explicitly.
            if invalid is None:
                path = (
                    self.targets["A"]["mechanism_root"]
                    / "session1/gpu_channel_plan_receipts/tp_rank_0.json"
                )
                self.write(path, dict(planned=[1, 2]))
            self.assertFalse(self.graph.acknowledged(), invalid)

    def test_missing_source_file_fails_before_receiver_plan_is_changed(self):
        self.files["s1"].unlink()
        with self.assertRaisesRegex(ValueError, "source target file"):
            self.graph.retire("p", {"A"})
        self.assertEqual(len(self.graph.rows("A")), 2)

    def test_partial_publication_retry_keeps_original_ack_boundary(self):
        atomic = pc.atomic_json

        def fail_b_once(path, value):
            if Path(path) == self.targets["B"]["channel_plan"]:
                raise OSError("injected second receiver write failure")
            atomic(path, value)

        with patch.object(pc, "atomic_json", fail_b_once), self.assertRaises(OSError):
            self.graph.retire("p", {"s1"})
        self.close_acks("A", [1])
        self.now = 110
        self.assertFalse(self.graph.retire("p", {"s1"}))
        self.assertEqual(self.graph.closing["written_at"], 100)
        self.close_acks("B", [1])
        self.assertTrue(self.graph.retire("p", {"s1"}))

    def test_receiver_cannot_be_added_during_pending_retirement(self):
        self.graph.retire("p", {"A"})
        with self.assertRaisesRegex(ValueError, "during channel retirement"):
            self.graph.add_target(
                "C",
                mechanism_root=self.root / "migration/C",
                channel_plan=self.root / "migration/C/plan.json",
            )


class BackendGraphTests(Base):
    def test_sequence_reuses_scoped_graph_without_touching_legacy_plan(self):
        graph = NS(
            targets=dict(B={}),
            rows=lambda target: [dict(generation=27, base_port=33000)],
            retire=lambda *args: True,
        )
        self.backend = SequenceBackend(
            self.backend.registry,
            self.harness.manifest,
            self.root,
            self.backend.ops,
            base_port=9000,
            context_limits={1: 8192, 2: 8192},
            channel_graph=graph,
            sequence_context={},
        )
        self.assertEqual(self.backend.next_channel_generation, 28)
        self.assertEqual(self.backend.next_channel_port, 33100)
        value = status()
        for source in ("A-g1-tp1", "A-g3-tp1"):
            value["instances"][source]["lifecycle"] = "RETIRED"
        self.write(value)
        with patch.object(graph, "retire", return_value=True) as retire:
            self.assertTrue(self.backend.retire_channels(PLAN))
            retire.assert_called_once()

    def test_retirement_waits_for_controller_ack_before_touching_graph(self):
        graph = NS(retire=lambda *args: True)
        self.backend.channel_graph = graph
        self.write(status())
        with patch.object(graph, "retire", return_value=True) as retire:
            self.assertFalse(self.backend.retire_channels(PLAN))
            retire.assert_not_called()
            value = status()
            for source in ("A-g1-tp1", "A-g3-tp1"):
                value["instances"][source]["lifecycle"] = "RETIRED"
            self.write(value)
            self.assertTrue(self.backend.retire_channels(PLAN))
            retire.assert_called_once_with("manual-A", ["A-g1-tp1", "A-g3-tp1"])

    def test_unacknowledged_graph_never_reaches_service_gate(self):
        self.backend.channel_graph = NS(acknowledged=lambda: False)
        with patch.object(self.backend, "gate") as gate:
            self.assertFalse(self.backend.precheck(PLAN))
            gate.assert_not_called()
