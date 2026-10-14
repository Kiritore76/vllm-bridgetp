# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for real batch boundaries and nonblocking CUDA event evidence."""

import ast
import asyncio
import collections
import hmac
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from tools.bridge_tp.verify_performance_trace import read_journal, verify, verify_pool

ROOT = Path(__file__).resolve().parents[2]
NS = types.SimpleNamespace


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


journal = load("performance_journal_test", "vllm/bridge_tp/async_journal.py")
performance = load("performance_trace_test", "vllm/bridge_tp/performance_trace.py")
BatchSampler = performance.BatchSampler
GpuTrace = performance.GpuTrace
HostTrace = performance.HostTrace
batch_composition = performance.batch_composition


def request(computed, prompt, total):
    return NS(num_computed_tokens=computed, num_prompt_tokens=prompt, num_tokens=total)


def actual_method(path, class_name, name, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    method.decorator_list = []
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    scope = {
        "cast": lambda _, value: value,
        "Future": Future,
        "ModelRunnerOutput": object,
        "Any": object,
    }
    scope.update(namespace or {})
    exec(compile(ast.Module(body=[method], type_ignores=[]), name, "exec"), scope)
    return scope[name]


class Event:
    def __init__(self, ready=True):
        self.ready = ready
        self.recorded = False

    def record(self, stream):
        self.recorded = True

    def query(self):
        return self.ready

    def elapsed_time(self, end):
        if not end.ready:
            raise AssertionError("read an unfinished event")
        return 12.5

    def synchronize(self):
        raise AssertionError("must not synchronize events")


class PerformanceTraceTests(unittest.TestCase):
    def setUp(self):
        modules = patch.dict(sys.modules, {"vllm.bridge_tp.async_journal": journal})
        modules.start()
        self.addCleanup(modules.stop)

    def test_disabled_does_not_construct_writer_or_import_torch(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(HostTrace.from_env())
            self.assertIsNone(GpuTrace.from_env(0, "cuda:0"))

    def test_mixed_batch_prefix_cache_and_recomputed_history(self):
        data = {
            "chunk": request(1024, 4096, 4096),
            "decode": request(2048, 1024, 2049),
            "resumed": request(4096, 4096, 4200),
        }
        result = batch_composition(data, {"chunk": 512, "decode": 1, "resumed": 104})
        self.assertEqual(result["prefill_tokens"], 512)
        self.assertEqual(result["recompute_tokens"], 103)
        self.assertEqual(result["decode_requests"], 2)
        self.assertEqual(result["decode_tokens"], 2)
        self.assertEqual(result["decode_context_sum"], 2048 + 4199)
        self.assertEqual(result["prefill_attention_work"], 512 * 1024 + 512 * 513 // 2)
        self.assertEqual(data["chunk"].num_computed_tokens, 1024)

    def test_sampler_keeps_prefill_between_periodic_decode_samples(self):
        output = NS(
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={"r": 1},
            scheduled_spec_decode_tokens={},
            preempted_req_ids=None,
        )
        data = {"r": request(20, 10, 21)}
        with patch.dict(os.environ, {"BRIDGETP_PERFORMANCE_SAMPLE_EVERY": "8"}):
            sampler = BatchSampler()
        snapshots = [
            sampler.snapshot(data, output, kv_usage=0.4, running=1, waiting=0)
            for _ in range(9)
        ]
        self.assertEqual([s["step_id"] for s in snapshots if s], [1, 9])
        data["r"] = request(0, 10, 10)
        self.assertEqual(
            sampler.snapshot(data, output, kv_usage=0.5, running=1, waiting=3)[
                "step_id"
            ],
            10,
        )

    def test_actual_scheduler_counts_target_without_source_freeze_gate(self):
        update = actual_method(
            "vllm/v1/core/sched/scheduler.py", "Scheduler", "_update_after_schedule"
        )
        scheduler = NS(
            _bridgetp_batch_sampler=None,
            _bridgetp_count_scheduled=True,
            _bridgetp_request_freeze=None,
            _bridgetp_prefill_scheduled_tokens=0,
            _bridgetp_decode_scheduled_tokens=0,
            requests={"p": request(0, 12, 12)},
            _inflight_prefills=set(),
            enable_return_routed_experts=False,
        )
        scheduler.requests["p"].num_output_placeholders = 0
        scheduler.requests["p"].use_structured_output = False
        out = NS(num_scheduled_tokens={"p": 8}, has_structured_output_requests=False)
        update(scheduler, out)
        self.assertEqual(scheduler._bridgetp_prefill_scheduled_tokens, 8)
        self.assertEqual(scheduler.requests["p"].num_computed_tokens, 8)

    def test_unready_event_is_retained_then_read_without_synchronizing(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"BRIDGETP_PERFORMANCE_TRACE_DIR": temporary}),
        ):
            events = [Event(), Event(False)]
            trace = GpuTrace(0, "cuda:0", lambda: events.pop(0), lambda: None)
            ticket = trace.begin({"step_id": 7})
            trace.end(ticket)
            trace.poll()
            self.assertEqual(len(trace.pending), 1)
            ticket[2].ready = True
            trace.poll()
            self.assertEqual(len(trace.pending), 0)
            trace.close()
            rows = read_journal(Path(temporary) / "gpu_rank0.jsonl")
            gpu = next(r for r in rows if r["kind"] == "gpu_batch")
            self.assertEqual(gpu["gpu_forward_logits_ms"], 12.5)

    def test_event_buffer_overflow_is_explicit_and_finalization_fails(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"BRIDGETP_PERFORMANCE_TRACE_DIR": temporary}),
        ):
            trace = GpuTrace(0, "cuda:0", lambda: Event(False), lambda: None)
            trace.capacity = 1
            trace.end(trace.begin({"step_id": 1}))
            self.assertIsNone(trace.begin({"step_id": 2}))
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                trace.close(timeout_s=0)
            self.assertEqual(trace.dropped, 1)

    def test_rank_composition_mismatch_is_rejected(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"BRIDGETP_PERFORMANCE_TRACE_DIR": temporary}),
        ):
            metadata = {
                "step_id": 1,
                "requests": [],
                "scheduled_tokens": 0,
                "prefill_tokens": 0,
                "decode_tokens": 0,
                "recompute_tokens": 0,
                "speculative": False,
            }
            host = HostTrace()
            host.submitted(metadata)
            host.completed(metadata)
            host.close()
            gpu = GpuTrace(0, "cuda:0", Event, lambda: None)
            gpu.end(gpu.begin(metadata))
            gpu.close()
            self.assertEqual(verify_pool(Path(temporary), 1)["samples"], 1)
            path = Path(temporary) / "gpu_rank0.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            next(r for r in rows if r["kind"] == "gpu_batch")["step_id"] = 2
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "ids differ"):
                verify_pool(Path(temporary), 1)

    def engine(self, trace, queue_size=1):
        output = NS(
            bridgetp_performance_sample={"step_id": 3},
            total_num_scheduled_tokens=1,
            pending_structured_output_tokens=False,
        )
        future = Future()
        future.set_result(object())
        scheduler = NS(
            has_requests=Mock(side_effect=[True, False]),
            schedule=lambda: output,
            get_grammar_bitmask=lambda _: None,
            update_from_output=lambda *args: {"done": True},
        )
        return NS(
            _bridgetp_performance_trace=trace,
            scheduler=scheduler,
            model_executor=NS(
                execute_model=lambda *a, **kw: future,
                sample_tokens=lambda *a, **kw: future,
            ),
            log_error_detail=lambda _: nullcontext(),
            log_iteration_details=lambda _: nullcontext(),
            _process_aborts_queue=lambda: None,
            is_ec_consumer=True,
            is_pooling_model=False,
            use_spec_decode=False,
            batch_queue=collections.deque() if queue_size > 1 else None,
            batch_queue_size=queue_size,
        )

    def test_actual_sync_engine_trace_wraps_result_completion(self):
        method = actual_method("vllm/v1/engine/core.py", "EngineCore", "step")
        trace = Mock()
        engine = self.engine(trace)
        self.assertEqual(method(engine), ({"done": True}, True))
        self.assertEqual([c[0] for c in trace.method_calls], ["submitted", "completed"])

    def test_actual_async_engine_does_not_finish_newly_submitted_batch(self):
        method = actual_method(
            "vllm/v1/engine/core.py", "EngineCore", "step_with_batch_queue"
        )
        trace = Mock()
        engine = self.engine(trace, queue_size=2)
        self.assertEqual(method(engine), (None, True))
        trace.completed.assert_not_called()
        self.assertEqual(method(engine), ({"done": True}, False))
        trace.completed.assert_called_once_with({"step_id": 3})

    def test_finalization_refuses_active_engine(self):
        method = actual_method(
            "vllm/v1/engine/core.py",
            "EngineCore",
            "bridge_tp_finalize_performance_trace",
        )
        engine = self.engine(Mock())
        engine.scheduler.has_unfinished_requests = lambda: True
        with self.assertRaisesRegex(RuntimeError, "idle"):
            method(engine)

    def test_actual_finalization_drains_all_workers_and_engine(self):
        method = actual_method(
            "vllm/v1/engine/core.py",
            "EngineCore",
            "bridge_tp_finalize_performance_trace",
        )
        trace = Mock()
        trace.close.return_value = {"status": "FINALIZED"}
        rpc = Mock(return_value=[{"status": "FINALIZED", "rank": 0}])
        engine = NS(
            _bridgetp_performance_trace=trace,
            batch_queue=None,
            scheduler=NS(has_unfinished_requests=lambda: False),
            model_executor=NS(collective_rpc=rpc),
        )
        self.assertEqual(method(engine)["workers"][0]["rank"], 0)
        rpc.assert_called_once_with(
            "bridge_tp_finalize_performance_trace", timeout=30.0
        )
        trace.close.assert_called_once()

    def test_authenticated_route_only_calls_named_idle_finalizer(self):
        class HttpError(Exception):
            def __init__(self, status_code, detail):
                self.status_code = status_code

        tree = ast.parse((ROOT / "vllm/bridge_tp/takeover_api.py").read_text())
        route = next(
            n
            for n in tree.body
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "finalize_performance_trace"
        )
        route.decorator_list = []
        route.returns = None
        route.args.args[0].annotation = None
        scope = {
            "os": os,
            "hmac": hmac,
            "HTTPException": HttpError,
            "_predictor_finalize_lock": asyncio.Lock(),
        }
        exec(compile(ast.Module(body=[route], type_ignores=[]), "route", "exec"), scope)
        rpc = AsyncMock(return_value={"status": "FINALIZED"})

        def invoke(token):
            req = NS(
                json=AsyncMock(return_value={"token": token}),
                app=NS(
                    state=NS(engine_client=NS(engine_core=NS(call_utility_async=rpc)))
                ),
            )
            coroutine = scope[route.name](req)
            try:
                coroutine.send(None)
            except StopIteration as completed:
                return completed.value
            self.fail("mock route unexpectedly suspended")

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HttpError) as error:
                invoke("secret")
            self.assertEqual(error.exception.status_code, 404)
        with patch.dict(
            os.environ,
            {
                "BRIDGETP_PERFORMANCE_TRACE_DIR": "trace",
                "BRIDGETP_PERFORMANCE_FINALIZE_TOKEN": "secret",
            },
        ):
            with self.assertRaises(HttpError) as error:
                invoke("wrong")
            self.assertEqual(error.exception.status_code, 403)
            rpc.assert_not_awaited()
            self.assertEqual(invoke("secret"), {"status": "FINALIZED"})
        rpc.assert_awaited_once_with("bridge_tp_finalize_performance_trace")

    def test_pilot_registers_only_two_diagnostic_pairs(self):
        from tools.bridge_tp.run_performance_trace_pilot import design

        rows = design()
        self.assertEqual([row["target_state"] for row in rows], ["idle", "prefill"])
        self.assertTrue(all(row["diagnostic_use_only"] for row in rows))
        self.assertTrue(all(row["threshold"] == 0 for row in rows))
        self.assertEqual(len({row["source_family_group"] for row in rows}), 2)

    def synthetic_arm(self, root, *, target_scheduled_unix_s):
        arm = root / "performance-pilot-prefill-20262401/case/prob_0_START/r01"
        controller = arm / "controller"
        provenance = arm / "provenance"
        provenance.mkdir(parents=True)
        for pool, ranks in (("source", 1), ("target", 4)):
            directory = controller / "performance_trace" / pool
            metadata = {
                "step_id": 1,
                "scheduled_unix_s": 1001
                if pool == "source"
                else target_scheduled_unix_s,
                "scheduled_tokens": 1,
                "prefill_tokens": int(pool == "target"),
                "decode_tokens": int(pool == "source"),
                "recompute_tokens": 0,
                "speculative": False,
                "requests": [{"query_tokens": 1}],
            }
            with patch.dict(
                os.environ, {"BRIDGETP_PERFORMANCE_TRACE_DIR": str(directory)}
            ):
                host = HostTrace()
                host.submitted(metadata)
                host.completed(metadata)
                workers = []
                for rank in range(ranks):
                    gpu = GpuTrace(rank, f"cuda:{rank}", Event, lambda: None)
                    gpu.end(gpu.begin(metadata))
                    workers.append(gpu.close())
                receipt = {
                    "status": "FINALIZED",
                    "engine": host.close(),
                    "workers": workers,
                }
            (provenance / f"performance_trace_{pool}_finalization.json").write_text(
                json.dumps(receipt)
            )
        (controller / "response_proxy_stats.json").write_text(
            json.dumps({"emitted": [{"unix_s": 1000}]})
        )

    def test_full_arm_finalization_and_rank_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.synthetic_arm(root, target_scheduled_unix_s=1001)
            report = verify(root, 1)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(
                report["arms"][0]["pools"]["target"]["after_anchor_first_token_300s"][
                    "prefill_samples"
                ],
                1,
            )

    def test_target_warmup_cannot_stand_in_for_measured_prefill(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.synthetic_arm(root, target_scheduled_unix_s=999)
            with self.assertRaisesRegex(ValueError, "lacks measured prefill"):
                verify(root, 1)


if __name__ == "__main__":
    unittest.main()
