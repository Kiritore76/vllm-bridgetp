# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual scheduler waiting pass without importing CUDA workers."""

import ast
import collections
import json
import tempfile
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Queue(collections.deque):
    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.popleft()

    def prepend_request(self, request):
        self.appendleft(request)

    def prepend_requests(self, requests):
        self.extendleft(requests)


class Request:
    def __init__(self, name):
        self.request_id = name
        self.status = "WAITING"
        self.num_computed_tokens = 0
        self.num_tokens = 17
        self.num_prompt_tokens = 17
        self.prefill_stats = None
        self.lora_request = None
        self.has_encoder_inputs = False
        self.mm_features = []


class TestAsyncShadowAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / "vllm/v1/core/sched/scheduler.py").read_text())
        scheduler = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
        )
        schedule = next(
            n
            for n in scheduler.body
            if isinstance(n, ast.FunctionDef) and n.name == "schedule"
        )
        waiting = next(
            n
            for n in schedule.body
            if isinstance(n, ast.If) and "step_skipped_waiting" in ast.unparse(n)
        )
        cls.code = compile(
            ast.Module(body=[waiting], type_ignores=[]),
            "actual_scheduler_waiting_pass",
            "exec",
        )

    def run_pass(
        self,
        *,
        enabled=True,
        allocate=True,
        paused=False,
        preempted=False,
        full=False,
        async_result=True,
    ):
        ordinary = [Request("ordinary-a"), Request("ordinary-b")]
        shadow = Request("shadow")
        waiting = Queue(ordinary + [shadow])
        probes, allocations, receipts, updates = [], [], [], []
        connector = types.SimpleNamespace(
            budget_free_async_load_admission=enabled,
            can_load_kv_without_token_budget=lambda r: r is shadow,
            get_num_new_matched_tokens=lambda r, local: (
                probes.append(r.request_id) or (16, async_result)
            ),
            record_async_load_admission=lambda *args: receipts.append(args),
            update_state_after_alloc=lambda *args: updates.append(args),
        )

        def allocate_slots(request, tokens, **kwargs):
            allocations.append((request.request_id, tokens, kwargs))
            return object() if allocate else None

        cache = types.SimpleNamespace(
            get_computed_blocks=lambda r: ([], 0),
            allocate_slots=allocate_slots,
            get_blocks=lambda r: [],
        )
        state = types.SimpleNamespace(
            waiting=waiting,
            skipped_waiting=Queue(),
            running=[object()] if full else [],
            policy=None,
            connector=connector,
            max_num_running_reqs=1 if full else 4,
            _pause_state="PAUSED_ALL" if paused else "UNPAUSED",
            _select_waiting_queue_for_scheduling=lambda: waiting,
            _is_blocked_waiting_status=lambda status: False,
            lora_config=None,
            kv_cache_manager=cache,
            ec_connector=None,
            need_mamba_block_aligned_split=False,
            use_eagle=False,
            num_lookahead_tokens=0,
            is_encoder_decoder=False,
            _inflight_prefill_reserved_blocks=lambda: 7,
            scheduler_reserve_full_isl=True,
            _inflight_prefills=set(),
            connector_prefix_cache_stats=None,
        )
        namespace = {
            "self": state,
            "token_budget": 0,
            "preempted_reqs": [object()] if preempted else [],
            "PauseState": types.SimpleNamespace(UNPAUSED="UNPAUSED"),
            "RequestStatus": types.SimpleNamespace(
                WAITING_FOR_REMOTE_KVS="WAITING_FOR_REMOTE_KVS"
            ),
            "create_request_queue": lambda policy: Queue(),
            "scheduled_loras": set(),
            "encoder_compute_budget": 0,
        }
        exec(self.code, namespace)
        return state, ordinary, shadow, probes, allocations, receipts, updates

    def test_zero_budget_load_bypasses_ordinary_compute_and_preserves_order(self):
        state, ordinary, shadow, probes, allocations, receipts, updates = (
            self.run_pass()
        )
        self.assertEqual(list(state.waiting), ordinary)
        self.assertEqual(probes, ["shadow"])
        self.assertEqual(allocations[0][1], 0)
        self.assertEqual(allocations[0][2]["reserved_blocks"], 7)
        self.assertTrue(allocations[0][2]["full_sequence_must_fit"])
        self.assertTrue(allocations[0][2]["delay_cache_blocks"])
        self.assertEqual(shadow.status, "WAITING_FOR_REMOTE_KVS")
        self.assertIn(shadow, state._inflight_prefills)
        self.assertEqual(list(state.skipped_waiting), [shadow])
        self.assertEqual(receipts[0][1:], (0, 7, 16))
        self.assertEqual(len(updates), 1)

    def test_insufficient_kv_never_admits_or_records_success(self):
        state, ordinary, shadow, _, allocations, receipts, updates = self.run_pass(
            allocate=False
        )
        self.assertEqual(list(state.waiting), ordinary + [shadow])
        self.assertEqual(allocations[0][2]["reserved_blocks"], 7)
        self.assertEqual(receipts, [])
        self.assertEqual(updates, [])
        self.assertEqual(shadow.status, "WAITING")

    def test_false_async_probe_cannot_schedule_local_work_at_zero_budget(self):
        state, ordinary, shadow, _, allocations, receipts, _ = self.run_pass(
            async_result=False
        )
        self.assertEqual(list(state.waiting), ordinary + [shadow])
        self.assertEqual(allocations, [])
        self.assertEqual(receipts, [])

    def test_legacy_connector_retains_zero_budget_behavior(self):
        state, ordinary, shadow, probes, allocations, _, _ = self.run_pass(
            enabled=False
        )
        self.assertEqual(list(state.waiting), ordinary + [shadow])
        self.assertEqual(probes, [])
        self.assertEqual(allocations, [])

    def test_pause_preemption_and_sequence_limit_remain_effective(self):
        for settings in ({"paused": True}, {"preempted": True}, {"full": True}):
            with self.subTest(settings=settings):
                state, ordinary, shadow, probes, allocations, _, _ = self.run_pass(
                    **settings
                )
                self.assertEqual(list(state.waiting), ordinary + [shadow])
                self.assertEqual(probes, [])
                self.assertEqual(allocations, [])

    def test_connector_opt_in_is_specific_to_gpu_resident_shadow(self):
        methods = self.connector_methods()
        state = types.SimpleNamespace(
            gpu_resident_shadow=False,
            _active_requests={},
            _request_matches=lambda r: r.request_id == "shadow",
        )
        predicate = methods["can_load_kv_without_token_budget"]
        self.assertFalse(predicate(state, Request("shadow")))
        state.gpu_resident_shadow = True
        self.assertTrue(predicate(state, Request("shadow")))
        self.assertFalse(predicate(state, Request("ordinary")))

    def test_loaded_shadow_is_not_revalidated_after_final_length_changes(self):
        methods = self.connector_methods()
        request = Request("shadow")
        request.num_tokens = 409
        request.num_computed_tokens = 408

        def invalid_first_admission_probe(request):
            raise ValueError("final continuation differs from initial reservation")

        state = types.SimpleNamespace(
            gpu_resident_shadow=True,
            _active_requests={request.request_id: request},
            _request_matches=invalid_first_admission_probe,
        )
        self.assertFalse(methods["can_load_kv_without_token_budget"](state, request))
        # Identity remains authoritative even if a preemption resets computed KV.
        request.num_computed_tokens = 0
        self.assertFalse(methods["can_load_kv_without_token_budget"](state, request))

    def test_remote_kv_resume_does_not_repeat_initial_admission_probe(self):
        methods = self.connector_methods()
        request = Request("shadow")
        request.num_computed_tokens = 16
        state = types.SimpleNamespace(
            gpu_resident_shadow=True,
            _active_requests={},
            _request_matches=lambda r: self.fail("resume probed as initial admission"),
        )
        self.assertFalse(methods["can_load_kv_without_token_budget"](state, request))

    def test_admission_receipt_keeps_budget_and_reservation_evidence(self):
        methods = self.connector_methods()
        with tempfile.TemporaryDirectory() as temp:
            state = types.SimpleNamespace(
                manifest_path=Path(temp) / "manifest.json",
                _load_manifest=lambda: {"migration_id": "migration"},
            )
            request = Request("shadow")
            methods["record_async_load_admission"](state, request, 0, 7, 16)
            receipt = json.loads(
                (Path(temp) / "async_load_admission_receipt.json").read_text()
            )
            self.assertTrue(receipt["budget_free_admission"])
            self.assertEqual(receipt["remaining_compute_token_budget"], 0)
            self.assertEqual(receipt["inflight_prefill_reserved_blocks"], 7)
            self.assertEqual(receipt["target_request_id"], "shadow")

    @staticmethod
    def connector_methods():
        tree = ast.parse((ROOT / "vllm/bridge_tp/streaming_connector.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "BridgeTPStreamingConnector"
        )
        names = {"can_load_kv_without_token_budget", "record_async_load_admission"}
        methods = [
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        future = ast.parse("from __future__ import annotations").body[0]
        namespace = {
            "time": time,
            "_atomic_json_dump": lambda value, path: path.write_text(json.dumps(value)),
        }
        exec(
            compile(
                ast.Module(body=[future] + methods, type_ignores=[]),
                "actual_connector_admission_methods",
                "exec",
            ),
            namespace,
        )
        return namespace


if __name__ == "__main__":
    unittest.main()
