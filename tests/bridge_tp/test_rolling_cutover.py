# SPDX-License-Identifier: Apache-2.0
"""Exercise rolling plans, actual source hook and reserved target remapping."""

from __future__ import annotations

import ast
import copy
import json
import threading
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from vllm.bridge_tp.rolling_cutover import (
    MECHANISM,
    RollingPlanner,
    applied_progress,
    finalize_reserved_request,
    rolling_evidence_errors,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class TestOriginalDeltaProgress(unittest.TestCase):
    def test_contiguous_original_acks_and_invalid_gap(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(applied_progress(root, "m1", 120))
            folder = root / "gpu_direct_delta_sender_receipts"
            first = {"migration_id": "m1", "status": "APPLIED_ALL_RANKS",
                     "start_token": 120, "end_token": 136}
            write(folder / "first.json", first)
            self.assertEqual(applied_progress(root, "m1", 120), 136)
            second = {**first, "start_token": 136, "end_token": 150}
            write(folder / "second.json", second)
            self.assertEqual(applied_progress(root, "m1", 120), 150)
            write(folder / "second.json", {**second, "start_token": 137})
            self.assertIsNone(applied_progress(root, "m1", 120))
            write(folder / "second.json", {**second, "status": "RECEIVED"})
            self.assertIsNone(applied_progress(root, "m1", 120))


class TestRollingPlanner(unittest.TestCase):
    def test_late_applied_ack_is_allowed_before_the_actual_boundary(self):
        planner = RollingPlanner(600, minimum_lead_tokens=64, lead_seconds=0.75)
        planner.observe(
            output_tokens=100,
            computed_tokens=199,
            resident_end=195,
            history_ready=True,
            delta_applied=True,
            unix_s=10,
        )
        self.assertIsNone(
            planner.observe(
                output_tokens=148,
                computed_tokens=247,
                resident_end=229,
                history_ready=True,
                delta_applied=True,
                unix_s=11,
            )
        )
        self.assertEqual(planner.boundary, 164)
        self.assertIsNone(
            planner.observe(
                output_tokens=164,
                computed_tokens=263,
                resident_end=259,
                history_ready=True,
                delta_applied=True,
                unix_s=12,
            )
        )
        self.assertEqual(planner.version, 1)

    def test_first_plan_after_history_does_not_wait_for_live_delta(self):
        planner = RollingPlanner(1000)
        self.assertIsNone(
            planner.observe(
                output_tokens=100,
                computed_tokens=199,
                resident_end=None,
                history_ready=False,
                delta_applied=False,
                unix_s=10,
            )
        )
        result = planner.observe(
            output_tokens=133,
            computed_tokens=232,
            resident_end=None,
            history_ready=True,
            delta_applied=False,
            unix_s=11,
        )
        self.assertEqual(result["cutover_output_tokens"], 389)
        self.assertEqual(result["reason"], "FIRST_AFTER_HISTORY_RESIDENT")
        self.assertFalse(result["first_delta_applied"])
        self.assertIsNone(result["delta_lag_tokens"])

    def test_fast_decode_gets_two_seconds_of_lead_with_reservation_cap(self):
        planner = RollingPlanner(1000)
        planner.observe(
            output_tokens=100,
            computed_tokens=199,
            resident_end=None,
            history_ready=False,
            delta_applied=False,
            unix_s=10,
        )
        result = planner.observe(
            output_tokens=300,
            computed_tokens=399,
            resident_end=None,
            history_ready=True,
            delta_applied=False,
            unix_s=11,
        )
        self.assertEqual(result["requested_lead_tokens"], 400)
        self.assertEqual(result["cutover_output_tokens"], 700)
        deferred = planner.observe(
            output_tokens=700,
            computed_tokens=799,
            resident_end=None,
            history_ready=True,
            delta_applied=False,
            unix_s=13,
        )
        self.assertEqual(deferred["cutover_output_tokens"], 999)

    def test_large_backlog_does_not_move_the_actual_boundary(self):
        planner = RollingPlanner(600, minimum_lead_tokens=64, lead_seconds=0.75)
        first = planner.observe(
            output_tokens=100,
            computed_tokens=199,
            resident_end=195,
            history_ready=True,
            delta_applied=True,
            unix_s=10,
        )
        second = planner.observe(
            output_tokens=164,
            computed_tokens=263,
            resident_end=195,
            history_ready=True,
            delta_applied=True,
            unix_s=12,
        )
        self.assertEqual(first["version"], 1)
        self.assertIsNone(second)
        self.assertEqual(first["version"], 1)
        self.assertEqual(planner.version, 1)
        self.assertEqual(planner.boundary, 164)

    def test_exhaustion_has_no_unreserved_boundary(self):
        planner = RollingPlanner(180)
        result = planner.observe(
            output_tokens=179,
            computed_tokens=300,
            resident_end=200,
            history_ready=True,
            delta_applied=True,
            unix_s=10,
        )
        self.assertEqual(result["status"], "RESERVATION_EXHAUSTED")
        self.assertIsNone(planner.boundary)

    def test_does_not_repeat_a_capped_plan_version(self):
        planner = RollingPlanner(180)
        planner.observe(
            output_tokens=140,
            computed_tokens=240,
            resident_end=236,
            history_ready=True,
            delta_applied=True,
            unix_s=10,
        )
        self.assertIsNone(
            planner.observe(
                output_tokens=165,
                computed_tokens=265,
                resident_end=236,
                history_ready=True,
                delta_applied=True,
                unix_s=11,
            )
        )
        self.assertEqual(planner.version, 1)


class TargetRequest:
    def __init__(self):
        self._all_token_ids = [0] * 72
        self.num_computed_tokens = 71
        self.num_output_tokens = 0
        self.max_tokens = 36
        self.sampling_params = types.SimpleNamespace(max_tokens=36)
        self.block_hashes = [1]

    @property
    def num_tokens(self):
        return len(self._all_token_ids)

    def update_block_hashes(self):
        self.block_hashes.append(tuple(self._all_token_ids))


class TestReservedTarget(unittest.TestCase):
    def test_only_actual_prefix_is_exposed_to_decode_and_budget_is_restored(self):
        request = TargetRequest()
        cutover = {
            "all_known_token_ids": list(range(32)),
            "num_prompt_tokens": 8,
            "cutover_num_output_tokens": 24,
            "num_computed_tokens": 31,
        }
        finalize_reserved_request(
            request, cutover, reserved_known_tokens=72, total_output_budget=100
        )
        self.assertEqual(request.num_tokens, 32)
        self.assertEqual(request.num_computed_tokens, 31)
        self.assertEqual(request.max_tokens, 76)
        self.assertEqual(request.sampling_params.max_tokens, 76)
        self.assertEqual(request.prompt_token_ids, list(range(32)))

    def test_rejects_overflow_wrong_computed_prefix_or_already_running_target(self):
        base = {
            "all_known_token_ids": list(range(32)),
            "num_prompt_tokens": 8,
            "cutover_num_output_tokens": 24,
            "num_computed_tokens": 31,
        }
        for patch_value in [
            {"num_computed_tokens": 32},
            {"num_prompt_tokens": 9},
            {"all_known_token_ids": list(range(73))},
        ]:
            with self.assertRaises(ValueError):
                finalize_reserved_request(
                    TargetRequest(),
                    {**base, **patch_value},
                    reserved_known_tokens=72,
                    total_output_budget=100,
                )
        request = TargetRequest()
        request.num_output_tokens = 1
        with self.assertRaises(ValueError):
            finalize_reserved_request(
                request, base, reserved_known_tokens=72, total_output_budget=100
            )


class TestRollingEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.session = {
            "migration_id": "m1",
            "source_request_id": "source1",
            "num_prompt_tokens": 100,
            "num_computed_tokens": 120,
        }
        self.cutover = {
            "cutover_num_output_tokens": 198,
            "num_computed_tokens": 297,
            "all_known_token_ids": list(range(298)),
        }
        common = {
            "mechanism": MECHANISM,
            "migration_id": "m1",
            "source_request_id": "source1",
        }
        write(
            self.root / "rolling_reservation.json",
            {
                **common,
                "initial_end_token": 120,
                "num_prompt_tokens": 100,
                "reservation_output_tokens": 600,
                "source_max_output_tokens": 2000,
                "total_output_budget": 2000,
                "published_unix_s": 10,
            },
        )
        plan = {
            **common,
            "status": "PLANNED",
            "version": 1,
            "cutover_output_tokens": 198,
            "reservation_output_tokens": 600,
            "output_tokens": 134,
            "computed_tokens": 233,
            "resident_end": 229,
            "delta_lag_tokens": 4,
            "history_ready": True,
            "first_delta_applied": True,
            "published_unix_s": 13,
            "reason": "FIRST_AFTER_HISTORY_AND_DELTA_APPLIED",
        }
        self.plan = plan
        (self.root / "rolling_source_plans.jsonl").write_text(json.dumps(plan))
        write(
            self.root / "rolling_freeze_selection.json",
            {
                **common,
                "version": 1,
                "cutover_output_tokens": 198,
                "computed_tokens": 297,
                "resident_end": 293,
                "delta_lag_tokens": 4,
                "history_ready": True,
                "first_delta_applied": True,
                "selected_unix_s": 15,
            },
        )
        write(
            self.root / "rolling_target_finalized.json",
            {
                **common,
                "cutover_output_tokens": 198,
                "reserved_known_tokens": 700,
                "num_tokens": 298,
                "num_computed_tokens": 297,
                "max_tokens": 1802,
            },
        )
        write(
            self.root / "gpu_direct_delta_sender_receipts/first.json",
            {
                "migration_id": "m1",
                "status": "APPLIED_ALL_RANKS",
                "start_token": 120,
                "end_token": 229,
                "completed_unix_s": 12,
            },
        )
        for rank in range(4):
            write(
                self.root / f"gpu_initial_receipts/tp_rank_{rank}.json",
                {
                    "migration_id": "m1",
                    "exact_readback": True,
                    "status": "INITIAL_HISTORY_GPU_RESIDENT",
                    "resident_completed_unix_s": 11,
                },
            )
            write(
                self.root / f"gpu_delta_receipts/tp_rank_{rank}/first.json",
                {
                    "migration_id": "m1",
                    "exact_readback": True,
                    "end_token": 293,
                    "completed_unix_s": 14,
                },
            )

    def errors(self):
        return rolling_evidence_errors(self.root, self.session, self.cutover, 16)

    def test_proved_prefix_and_output_budget_are_accepted(self):
        self.assertEqual(self.errors(), [])

    def test_large_pre_freeze_lag_is_valid_but_missing_applied_proof_is_not(self):
        path = self.root / "rolling_freeze_selection.json"
        proof = json.loads(path.read_text())
        # Existing exact resident proof extends to 293; a claimed watermark
        # of 229 is valid even though the frozen computed watermark is 297.
        write(path, {**proof, "resident_end": 229, "delta_lag_tokens": 68})
        self.assertEqual(self.errors(), [])
        write(path, {**proof, "resident_end": 229, "delta_lag_tokens": 67})
        self.assertTrue(self.errors())
        write(path, {**proof, "first_delta_applied": False})
        self.assertTrue(self.errors())

    def test_history_only_first_plan_accepts_delta_ack_before_freeze(self):
        plan = {
            **self.plan,
            "reason": "FIRST_AFTER_HISTORY_RESIDENT",
            "first_delta_applied": False,
            "resident_end": None,
            "delta_lag_tokens": None,
        }
        (self.root / "rolling_source_plans.jsonl").write_text(json.dumps(plan))
        path = self.root / "gpu_direct_delta_sender_receipts/first.json"
        ack = json.loads(path.read_text())
        write(path, {**ack, "completed_unix_s": 14})
        self.assertEqual(self.errors(), [])
        write(path, {**ack, "completed_unix_s": 15.5})
        self.assertTrue(self.errors())

    def test_missing_ack_late_history_or_wrong_final_budget_are_rejected(self):
        patches = [
            ("gpu_direct_delta_sender_receipts/first.json", {"completed_unix_s": 14}),
            ("gpu_initial_receipts/tp_rank_2.json", {"resident_completed_unix_s": 14}),
            ("rolling_target_finalized.json", {"max_tokens": 1801}),
            ("rolling_freeze_selection.json", {"version": 2}),
        ]
        for name, changes in patches:
            with self.subTest(name=name):
                path = self.root / name
                original = json.loads(path.read_text())
                write(path, {**original, **changes})
                self.assertTrue(self.errors())
                write(path, original)

    def test_nonmonotonic_versions_and_post_freeze_plans_are_rejected(self):
        for changes in (
            {"version": 2},
            {"published_unix_s": 17},
            {"cutover_output_tokens": 601},
        ):
            changed = {**copy.deepcopy(self.plan), **changes}
            (self.root / "rolling_source_plans.jsonl").write_text(json.dumps(changed))
            self.assertTrue(self.errors())


class TestActualTargetCallback(unittest.TestCase):
    def test_scheduler_callback_trims_before_target_can_be_promoted(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = TargetRequest()
            write(
                root / "rolling_reservation.json",
                {
                    "mechanism": MECHANISM,
                    "migration_id": "m1",
                    "num_prompt_tokens": 8,
                    "reservation_output_tokens": 64,
                    "total_output_budget": 100,
                },
            )
            write(
                root / "cutover_manifest.json",
                {
                    "migration_id": "m1",
                    "num_prompt_tokens": 8,
                    "all_known_token_ids": list(range(32)),
                    "num_computed_tokens": 31,
                    "cutover_num_output_tokens": 24,
                },
            )
            path = (
                Path(__file__).resolve().parents[2]
                / "vllm/bridge_tp/streaming_connector.py"
            )
            tree = ast.parse(path.read_text(encoding="utf-8"))
            cls = next(
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef)
                and n.name == "BridgeTPStreamingConnector"
            )
            method = next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "update_connector_output"
            )
            namespace = {
                "_load_json": lambda path: json.loads(path.read_text()),
                "_atomic_json_dump": lambda value, path: write(path, value),
                "_READY_SYNC_STREAM_EVENT": "STREAM_EVENT",
                "time": types.SimpleNamespace(time=lambda: 20),
            }
            future = ast.parse("from __future__ import annotations").body[0]
            exec(
                compile(
                    ast.Module(body=[future, method], type_ignores=[]),
                    str(path),
                    "exec",
                ),
                namespace,
            )
            connector = types.SimpleNamespace(
                _active_requests={"target1": request},
                gpu_resident_shadow=True,
                manifest_path=root / "session_manifest.json",
                ready_sync_mode="STREAM_EVENT",
                _model_wait_pending_requests=set(),
            )
            namespace["update_connector_output"](
                connector, types.SimpleNamespace(finished_recving={"target1"})
            )
            self.assertEqual(request.num_tokens, 32)
            self.assertEqual(request.num_computed_tokens, 31)
            self.assertEqual(request.max_tokens, 76)
            self.assertEqual(connector._model_wait_pending_requests, {"target1"})
            proof = json.loads((root / "rolling_target_finalized.json").read_text())
            self.assertEqual(proof["reserved_known_tokens"], 72)
            self.assertEqual(proof["num_tokens"], 32)


class TestSafeRollingCancel(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.background = self.root / "background"
        write(
            self.background / "background_summary.json",
            {"jobs": 1, "completed": 1, "failed": 0},
        )
        common = {"mechanism": MECHANISM, "migration_id": "m1"}
        write(self.root / "rolling_reservation.json", common)
        write(
            self.root / "rolling_source_plan.json",
            {**common, "status": "RESERVATION_EXHAUSTED"},
        )
        write(
            self.root / "source_response.json",
            {"token_ids": [1, 2, 3], "finish_reason": "stop"},
        )
        write(
            self.root / "target_response.json",
            {"token_ids": [], "finish_reason": "error"},
        )
        write(
            self.root / "response_proxy_stats.json",
            {
                "committed": False,
                "target_origin_tokens": 0,
                "source_origin_tokens": 3,
                "emitted_tokens": 3,
                "emitted": [{"token_id": i, "origin": "source"} for i in (1, 2, 3)],
            },
        )
        write(
            self.root / "takeover_state.json",
            {
                **common,
                "state": "CANCELLED",
                "source_abort_dispatched": False,
                "source_continues_on_tp1": True,
            },
        )
        for name in ("source", "target", "stager"):
            write(
                self.root / f"{name}_cleanup_receipt.json",
                {**common, "status": "CLEANED"},
            )
        rows = [
            {"kind": "transition", "to": "SHADOW"},
            {
                "kind": "abandon",
                "reason": (
                    "rolling reservation exhausted before safe freeze; source continues"
                ),
            },
            {"kind": "transition", "to": "CANCELLED"},
            {
                "kind": "run_end",
                "final_state": "CANCELLED",
                "trigger_path": "MANAGER_M1_START",
            },
        ]
        (self.root / "phase9_audit.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    def accept(self):
        from tools.bridge_tp.run_shadow_strategy_online_validation import (
            accept_shadow_without_cutover,
        )

        return accept_shadow_without_cutover(self.root, self.background, 1, 100)

    def test_complete_source_and_cleaned_shadow_are_a_valid_start_cost(self):
        result = self.accept()
        self.assertEqual(result["status"], "PASS", result["errors"])
        self.assertEqual(result["outcome"], "START_CANCELLED_SOURCE_COMPLETED")

    def test_legacy_stager_must_match_this_sessions_cleanup_request(self):
        reason = "rolling reservation exhausted before safe freeze; source continues"
        write(
            self.root / "session_manifest.json",
            {"migration_id": "m1", "source_request_id": "source1"},
        )
        write(
            self.root / "cleanup_request.json",
            {
                "migration_id": "m1",
                "source_request_id": "source1",
                "reason": reason,
                "abort_source": False,
                "requested_unix_s": 10,
            },
        )
        receipt = {
            "status": "CLEANED",
            "component": "cpu_stager",
            "reason": reason,
            "updated_unix_s": 11,
        }
        path = self.root / "stager_cleanup_receipt.json"
        write(path, receipt)
        self.assertEqual(self.accept()["status"], "PASS")
        for changes in (
            {"updated_unix_s": 9},
            {"reason": "different session"},
            {"migration_id": "wrong"},
            {"status": "FAILED"},
        ):
            write(path, {**receipt, **changes})
            self.assertEqual(self.accept()["status"], "FAIL")

    def test_stager_publishes_the_session_identity(self):
        from tools.bridge_tp.phase8_stager import _cleanup

        write(self.root / "session_manifest.json", {"migration_id": "m1"})
        _cleanup(self.root, "test cleanup", 4, 0)
        receipt = json.loads((self.root / "stager_cleanup_receipt.json").read_text())
        self.assertEqual(receipt["migration_id"], "m1")
        self.assertEqual(receipt["status"], "CLEANED")

    def test_unproved_cleanup_or_partial_source_are_not_accepted(self):
        for name in ("target_cleanup_receipt.json", "rolling_source_plan.json"):
            path = self.root / name
            original = path.read_text()
            path.unlink()
            self.assertEqual(self.accept()["status"], "FAIL")
            path.write_text(original)
        proxy = json.loads((self.root / "response_proxy_stats.json").read_text())
        proxy["emitted"].pop()
        write(self.root / "response_proxy_stats.json", proxy)
        self.assertEqual(self.accept()["status"], "FAIL")

    def test_cancellation_after_freeze_is_not_a_safe_source_only_outcome(self):
        write(self.root / "request_frozen_receipt.json", {})
        self.assertEqual(self.accept()["status"], "FAIL")


@dataclass
class SourceConfig:
    run_dir: Path
    migration_id: str = "m1"
    phase8_cutover_output_tokens: int = 1999
    gpu_direct_delta: bool = True
    gpu_direct_delta_batch_tokens: int = 16
    gpu_direct_delta_flush_ms: float = 25


class TestActualSourceHook(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = 100.0
        config = SourceConfig(self.root)
        self.state = types.SimpleNamespace(
            finalized=False,
            finalizing=False,
            request_id="source1",
            last_computed_token=120,
            block_size=16,
            last_flush_monotonic=0,
            enqueue_gpu_delta=MagicMock(return_value=True),
            lifecycle_lock=threading.Lock(),
            config=config,
        )
        path = Path(__file__).resolve().parents[2] / "vllm/bridge_tp/phase8_source.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "maybe_publish_phase8_delta"
        )
        future = ast.parse("from __future__ import annotations").body[0]
        self.globals = {
            "_state": self.state,
            "time": types.SimpleNamespace(
                time=lambda: self.clock,
                monotonic=lambda: self.clock,
                time_ns=lambda: int(self.clock * 1e9),
                monotonic_ns=lambda: int(self.clock * 1e9),
            ),
            "json": json,
            "threading": threading,
            "_atomic_json_dump": lambda value, path: write(path, value),
            "_get_request_block_ids": lambda *args: (list(range(100)), 16),
            "_background_finalize_cutover": MagicMock(),
        }
        exec(
            compile(
                ast.Module(body=[future, function], type_ignores=[]), str(path), "exec"
            ),
            self.globals,
        )
        self.hook = self.globals["maybe_publish_phase8_delta"]
        self.config = config
        write(
            self.root / "rolling_reservation.json",
            {
                "mechanism": MECHANISM,
                "migration_id": "m1",
                "source_request_id": "source1",
                "reservation_output_tokens": 600,
                "num_prompt_tokens": 100,
                "initial_end_token": 120,
                "source_max_output_tokens": 2000,
            },
        )

    def apply_progress(self, computed):
        for rank in range(4):
            write(
                self.root / "gpu_initial_receipts" / f"tp_rank_{rank}.json",
                {
                    "migration_id": "m1",
                    "status": "INITIAL_HISTORY_GPU_RESIDENT",
                    "exact_readback": True,
                    "end_token": 120,
                },
            )
        write(
            self.root / "gpu_direct_delta_sender_receipts/progress.json",
            {
                "migration_id": "m1",
                "status": "APPLIED_ALL_RANKS",
                "start_token": 120,
                "end_token": computed,
            },
        )

    def step(self, output):
        self.clock = 100 + output * 0.031
        known = 100 + output
        request = types.SimpleNamespace(
            output_token_ids=[0] * output,
            num_tokens=known,
            num_prompt_tokens=100,
            get_token_id=lambda i: i,
        )
        self.hook(
            config=self.config,
            request_id="source1",
            kv_caches=[],
            requests={"source1": request},
            cache_dtype="",
            attn_groups=[],
            input_batch=types.SimpleNamespace(
                req_id_to_index={"source1": 0}, num_computed_tokens_cpu=[known - 2]
            ),
            scheduler_output=types.SimpleNamespace(num_scheduled_tokens={"source1": 1}),
        )

    def test_history_only_plan_keeps_generating_until_applied_boundary(self):
        with patch("vllm.bridge_tp.request_freeze.request_freeze") as freeze:
            for rank in range(4):
                write(
                    self.root / "gpu_initial_receipts" / f"tp_rank_{rank}.json",
                    {
                        "migration_id": "m1",
                        "status": "INITIAL_HISTORY_GPU_RESIDENT",
                        "exact_readback": True,
                        "end_token": 120,
                    },
                )
            self.step(30)
            plan = json.loads((self.root / "rolling_source_plan.json").read_text())
            self.assertEqual(plan["cutover_output_tokens"], 286)
            self.assertFalse(plan["first_delta_applied"])
            self.step(285)
            self.assertFalse(freeze.called)
            self.step(286)
            self.assertGreater(self.state.rolling_planner.boundary, 286)
            self.assertFalse(freeze.called)

    def test_large_backlog_freezes_at_first_boundary_and_enqueues_tail(self):
        with (
            patch("vllm.bridge_tp.request_freeze.enabled_from_env", return_value=True),
            patch(
                "vllm.bridge_tp.request_freeze.request_freeze", return_value={}
            ) as freeze,
            patch.object(threading, "Thread") as thread,
        ):
            for output in range(21, 134):
                self.step(output)
            self.assertFalse(freeze.called)
            self.assertFalse((self.root / "rolling_source_plan.json").exists())
            self.apply_progress(100 + 134 - 5)
            self.step(134)
            first = self.state.rolling_planner.boundary
            self.step(first - 16)
            self.assertEqual(self.state.rolling_planner.boundary, first)
            self.assertFalse(freeze.called)
            self.step(first)
            freeze.assert_called_once()
            self.assertEqual(freeze.call_args.kwargs["output_tokens"], first)
            self.assertEqual(self.state.rolling_planner.boundary, first)
            self.assertTrue(self.state.finalizing)
            thread.return_value.start.assert_called_once()
            proof = json.loads(
                (self.root / "rolling_freeze_selection.json").read_text()
            )
            self.assertEqual(proof["version"], 1)
            self.assertGreater(proof["delta_lag_tokens"], 16)
            self.assertEqual(proof["freeze_policy"], "FREEZE_AT_BOUNDARY_DRAIN_TAIL")
            self.assertIsNone(proof["maximum_pre_freeze_delta_lag_tokens"])
            final = self.state.enqueue_gpu_delta.call_args.kwargs
            self.assertEqual(final["end_token"], proof["computed_tokens"])
            self.assertFalse((self.root / "cutover_manifest.json").exists())

    def test_reservation_exhaustion_never_freezes_or_enqueues_out_of_range(self):
        with patch("vllm.bridge_tp.request_freeze.request_freeze") as freeze:
            self.step(599)
            self.step(600)
            self.assertTrue(self.state.rolling_exhausted)
            self.assertFalse(freeze.called)
            self.assertFalse(self.state.enqueue_gpu_delta.called)
