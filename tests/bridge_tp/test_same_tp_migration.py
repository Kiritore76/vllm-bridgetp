# SPDX-License-Identifier: Apache-2.0
"""CPU contracts of migration between instances of the same TP degree.

These do not exercise NCCL or a GPU: they pin who sends what, where, and that a
rank which cannot join stops the session instead of copying a different history.
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path

from vllm.bridge_tp import gpu_direct_targets, source_channels
from vllm.bridge_tp.controller import tp_geometry
from vllm.bridge_tp.rolling_cutover import applied_progress
from vllm.bridge_tp.source_ranks import (
    RANK_FAILURE,
    PeerFailed,
    SourceRole,
    read_failure,
    report_failure,
    report_stage,
    trigger_check,
    wait_for_peers,
)

ROOT = Path(__file__).resolve().parents[2]
PRESSURE_SESSION = ROOT / "tools/bridge_tp/pressure_session.py"
KV_STREAM = ROOT / "vllm/bridge_tp/kv_stream.py"
PHASE8 = ROOT / "vllm/bridge_tp/phase8_source.py"


class RoleTests(unittest.TestCase):
    def test_tp1_splits_for_any_supported_target(self):
        role = SourceRole(1, 0)
        self.assertFalse(role.paired)
        self.assertEqual(role.sender_tp(2), 2)
        self.assertEqual(
            role.target_addresses("h", 100, 2), ["h:100", "h:101"]
        )
        self.assertEqual(role.rank_base(), 0)
        self.assertEqual(role.sender_heads(8), 8)

    def test_tp1_to_tp1_is_one_connection_without_split(self):
        role = SourceRole(1, 0)
        self.assertEqual(role.sender_tp(1), 1)
        self.assertEqual(role.target_addresses("h", 100, 1), ["h:100"])

    def test_each_tp2_rank_only_connects_to_its_own_target_rank(self):
        for rank in range(2):
            role = SourceRole(2, rank)
            self.assertTrue(role.paired)
            self.assertEqual(role.sender_tp(2), 1)
            self.assertEqual(role.rank_base(), rank)
            self.assertEqual(role.target_addresses("h", 100, 2), [f"h:{100 + rank}"])
            self.assertEqual(role.sender_heads(8), 4)

    def test_same_tp_is_general(self):
        role = SourceRole(4, 3)
        self.assertEqual(role.target_addresses("h", 100, 4), ["h:103"])
        self.assertEqual(role.sender_heads(8), 2)

    def test_larger_source_refuses_a_different_tp(self):
        for target in (1, 4):
            with self.assertRaises(ValueError):
                SourceRole(2, 0).check_target(target)
        with self.assertRaises(ValueError):
            SourceRole(2, 0).target_addresses("h", 100, 4)
        with self.assertRaises(ValueError):
            SourceRole(1, 0).check_target(3)

    def test_invalid_rank(self):
        with self.assertRaises(ValueError):
            SourceRole(2, 2)

    def test_rank_zero_keeps_historical_files(self):
        self.assertEqual(SourceRole(1, 0).own_file("a.json"), "a.json")
        self.assertEqual(SourceRole(2, 0).own_file("a.json"), "a.json")
        self.assertEqual(SourceRole(2, 1).own_file("a.json"), "a_rank_1.json")
        root = Path("run")
        self.assertEqual(
            SourceRole(1, 0).delta_receipt_dir(root),
            root / "gpu_direct_delta_sender_receipts",
        )
        self.assertEqual(
            SourceRole(2, 1).delta_receipt_dir(root),
            root / "gpu_direct_delta_sender_receipts" / "rank_1",
        )


class TriggerTests(unittest.TestCase):
    def test_tp1_may_start_late(self):
        role = SourceRole(1, 0)
        self.assertEqual(trigger_check(role, 9, 10), "WAIT")
        self.assertEqual(trigger_check(role, 10, 10), "START")
        self.assertEqual(trigger_check(role, 15, 10), "START")

    def test_ranks_only_start_at_the_exact_token(self):
        role = SourceRole(2, 1)
        self.assertEqual(trigger_check(role, 9, 10), "WAIT")
        self.assertEqual(trigger_check(role, 10, 10), "START")
        self.assertEqual(trigger_check(role, 11, 10), "MISSED")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class PeerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name)
        self.clock = FakeClock()
        self.leader = SourceRole(2, 0)
        self.follower = SourceRole(2, 1)

    def wait(self, stage="start", timeout_s=1.0, **kwargs):
        return wait_for_peers(
            self.run_dir,
            self.leader,
            stage,
            timeout_s=timeout_s,
            clock=self.clock,
            sleep=self.clock.sleep,
            **kwargs,
        )

    def test_all_reported(self):
        report_stage(self.run_dir, self.follower, "start", output_tokens=7)
        self.assertEqual(self.wait(expected=dict(output_tokens=7))[0]["rank"], 1)

    def test_tp1_has_nobody_to_wait_for(self):
        self.assertEqual(
            wait_for_peers(self.run_dir, SourceRole(), "x", timeout_s=0), []
        )

    def test_timeout_fails_closed(self):
        with self.assertRaises(TimeoutError):
            self.wait()

    def test_different_boundary_fails_closed(self):
        report_stage(self.run_dir, self.follower, "start", output_tokens=8)
        with self.assertRaises(PeerFailed):
            self.wait(expected=dict(output_tokens=7))

    def test_stale_session_report_is_refused(self):
        report_stage(self.run_dir, self.follower, "start", migration_id="old")
        with self.assertRaises(PeerFailed):
            self.wait(expected=dict(migration_id="new"))

    def test_failure_stops_the_wait_at_once(self):
        report_failure(self.run_dir, self.follower, "late")
        with self.assertRaisesRegex(PeerFailed, "late"):
            self.wait()
        self.assertEqual(self.clock.now, 0.0)

    def test_abandoned_session_stops_the_wait(self):
        with self.assertRaises(PeerFailed):
            self.wait(abandoned=lambda: True)

    def test_first_failure_stays_the_reason(self):
        report_failure(self.run_dir, self.follower, "first")
        report_failure(self.run_dir, self.leader, "second")
        self.assertEqual(read_failure(self.run_dir)["error"], "first")

    def test_cleanup_does_not_wait_for_a_rank_that_failed(self):
        report_failure(self.run_dir, self.follower, "gone")
        self.assertEqual(self.wait("cleaned"), [])

    def test_cleanup_waits_for_a_healthy_rank(self):
        report_failure(self.run_dir, self.leader, "leader")
        with self.assertRaises(TimeoutError):
            self.wait("cleaned")


def receipt(directory, start, end, migration_id="m"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"delta_{start:012d}_{end:012d}.json").write_text(
        json.dumps(
            dict(
                status="APPLIED_ALL_RANKS",
                migration_id=migration_id,
                start_token=start,
                end_token=end,
            )
        )
    )


class AppliedProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name)
        self.receipts = self.run_dir / "gpu_direct_delta_sender_receipts"

    def manifest(self, source_tp, target_tp=None):
        (self.run_dir / "session_manifest.json").write_text(
            json.dumps(
                dict(source_tp_size=source_tp, target_tp_size=target_tp or source_tp)
            )
        )

    def test_one_sender_is_unchanged(self):
        self.manifest(1, 2)
        receipt(self.receipts, 10, 20)
        receipt(self.receipts, 20, 30)
        self.assertEqual(applied_progress(self.run_dir, "m", 10), 30)

    def test_whole_kv_is_at_the_slowest_rank(self):
        self.manifest(2)
        receipt(self.receipts / "rank_0", 10, 40)
        receipt(self.receipts / "rank_1", 10, 16)
        receipt(self.receipts / "rank_1", 16, 25)
        self.assertEqual(applied_progress(self.run_dir, "m", 10), 25)

    def test_a_rank_without_receipts_means_no_progress(self):
        self.manifest(2)
        receipt(self.receipts / "rank_0", 10, 40)
        self.assertIsNone(applied_progress(self.run_dir, "m", 10))

    def test_a_gap_on_one_rank_means_no_progress(self):
        self.manifest(2)
        receipt(self.receipts / "rank_0", 10, 40)
        receipt(self.receipts / "rank_1", 12, 40)
        self.assertIsNone(applied_progress(self.run_dir, "m", 10))


class GeometryTests(unittest.TestCase):
    def test_session_may_end_on_tp1(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session_manifest.json"
            path.write_text(json.dumps(dict(target_tp_size=1, source_tp_size=1)))
            self.assertEqual(tp_geometry.session_target_tp(tmp), 1)
            self.assertEqual(tp_geometry.session_source_tp(tmp), 1)

    def test_source_tp_defaults_to_the_old_tp1(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(tp_geometry.session_source_tp(tmp), 1)
            path = Path(tmp) / "session_manifest.json"
            path.write_text(json.dumps(dict(target_tp_size=2)))
            self.assertEqual(tp_geometry.session_source_tp(tmp), 1)
            path.write_text(json.dumps(dict(target_tp_size=2, source_tp_size=2)))
            self.assertEqual(tp_geometry.session_source_tp(tmp), 2)

    def test_layout_bundles_still_refuse_tp1(self):
        with self.assertRaises(ValueError):
            tp_geometry.validate_target_tp(1)


class SmokeHelperTests(unittest.TestCase):
    def test_token_comparison(self):
        smoke_source = (ROOT / "tools/bridge_tp/run_tp2_migration_smoke.py").read_text(
            "utf-8"
        )
        tree = ast.parse(smoke_source)
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "compare_tokens"
        )
        namespace = {}
        exec(compile(ast.Module([function], []), "smoke", "exec"), namespace)
        compare = namespace["compare_tokens"]
        self.assertTrue(compare([1, 2], [1, 2])["identical"])
        self.assertEqual(compare([1, 3], [1, 2])["first_difference"], 1)
        self.assertFalse(compare([1], [1, 2])["identical"])


class ChannelDescriptionTests(unittest.TestCase):
    def test_tp2_can_be_a_source_channel(self):
        rows = [dict(source="A-g4-tp2", base_port=31000, generation=4)]
        plan = source_channels.parse_plan(rows, 2)
        self.assertEqual(plan[0].source, "A-g4-tp2")

    def test_tp1_can_be_a_target(self):
        rows = [
            dict(target="A-g5-tp1", host="h", base_port=31000, tp=1, generation=1)
        ]
        targets = gpu_direct_targets.parse_targets(json.dumps(rows), 8)
        self.assertEqual(targets[0].tp, 1)
        self.assertEqual(list(targets[0].ports), [31000, 31001])

    def test_unsupported_target_tp_is_refused(self):
        rows = [
            dict(target="A-g5-tp3", host="h", base_port=31000, tp=3, generation=1)
        ]
        with self.assertRaises(ValueError):
            gpu_direct_targets.parse_targets(json.dumps(rows), 8)


class ControllerContractTests(unittest.TestCase):
    """No GPU here: pin the file-level contract between ranks and controller."""

    def test_controller_reads_the_file_the_ranks_write(self):
        self.assertIn(f'"{RANK_FAILURE}"', PRESSURE_SESSION.read_text("utf-8"))

    def test_only_rank_zero_freezes(self):
        source = PHASE8.read_text("utf-8")
        branch = source.index("if not state.role.leader:\n        # Rank 0 asks")
        self.assertLess(branch, source.index("freeze_request = request_freeze("))

    def test_every_sender_passes_its_rank(self):
        tree = ast.parse(KV_STREAM.read_text("utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "GpuDirectHistorySender"
        ]
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIn("rank_base", [k.arg for k in call.keywords])


if __name__ == "__main__":
    unittest.main()
