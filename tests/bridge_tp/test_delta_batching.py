# SPDX-License-Identifier: Apache-2.0
"""Exercise real queue coalescing and boundary propagation without CUDA."""

import ast
import importlib.util
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "delta_batching_cpu", ROOT / "vllm/bridge_tp/delta_batching.py"
)
batching = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batching)


class ObservedQueue(queue.Queue):
    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()

    def get(self, block=True, timeout=None):
        if timeout is not None:
            self.waiting.set()
        return super().get(block, timeout)


def work(start, end, final=None):
    return dict(
        start_token=start,
        end_token=end,
        final_end_token=final,
        block_ids=[end // 16],
        ready_event=f"ready-{end}",
    )


class DeltaBatchingTests(unittest.TestCase):
    def delayed_merge(self, first, additions):
        jobs = ObservedQueue()
        jobs.put(first)
        initial = jobs.get_nowait()
        result, errors = [], []

        def consume():
            try:
                result.append(batching.coalesce_delta(initial, jobs))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        self.assertTrue(jobs.waiting.wait(1), "small tail did not enter bounded wait")
        for addition in additions:
            jobs.put(addition)
        thread.join(1)
        self.assertFalse(thread.is_alive(), "coalescing did not finish")
        self.assertEqual(errors, [])
        # join() must still wait for the physical send/ACK owned by the caller.
        self.assertGreaterEqual(jobs.unfinished_tasks, 1)
        jobs.task_done()
        return result[0], jobs

    def test_actual_13_plus_1_tail_becomes_one_batch_with_latest_cuda_event(self):
        (merged, count), jobs = self.delayed_merge(
            work(2284, 2297, 2298), [work(2297, 2298, 2298)]
        )
        self.assertEqual((merged["start_token"], merged["end_token"]), (2284, 2298))
        self.assertEqual(count, 2)
        self.assertEqual(merged["ready_event"], "ready-2298")
        self.assertEqual(merged["block_ids"], [2298 // 16])
        self.assertEqual(merged["tail_coalescing"]["outcome"], "FINAL_BOUNDARY")
        self.assertEqual(jobs.unfinished_tasks, 0)

    def test_small_previous_batch_can_absorb_all_four_remaining_tokens(self):
        (merged, count), jobs = self.delayed_merge(
            work(0, 16, 20), [work(16, 18, 20), work(18, 20, 20)]
        )
        self.assertEqual((merged["end_token"], count), (20, 3))
        self.assertEqual(merged["tail_coalescing"]["outcome"], "FINAL_BOUNDARY")
        self.assertEqual(jobs.unfinished_tasks, 0)

    def test_absent_final_tail_times_out_without_losing_work(self):
        jobs = queue.Queue()
        jobs.put(work(2284, 2297, 2298))
        first = jobs.get_nowait()
        merged, count = batching.coalesce_delta(first, jobs, max_wait_s=0.005)
        self.assertEqual((merged["end_token"], count), (2297, 1))
        self.assertEqual(merged["tail_coalescing"]["outcome"], "TIMEOUT")
        self.assertEqual(jobs.unfinished_tasks, 1)
        jobs.task_done()

    def test_normal_far_or_large_batches_never_wait(self):
        for first in (work(0, 13), work(0, 13, 30), work(0, 17, 18)):
            with self.subTest(first=first):
                jobs = ObservedQueue()
                merged, count = batching.coalesce_delta(first, jobs)
                self.assertFalse(jobs.waiting.is_set())
                self.assertNotIn("tail_coalescing", merged)
                self.assertEqual(count, 1)

    def test_stop_wakes_waiter_and_keeps_stop_marker_accounted(self):
        (merged, count), jobs = self.delayed_merge(work(0, 13, 14), [None])
        self.assertEqual(count, 1)
        self.assertEqual(merged["tail_coalescing"]["outcome"], "STOP_REQUESTED")
        self.assertIsNone(jobs.get_nowait())
        jobs.task_done()
        self.assertEqual(jobs.unfinished_tasks, 0)

    def test_queue_gap_is_rejected_without_leaking_extra_queue_tasks(self):
        jobs = queue.Queue()
        jobs.put(work(0, 13, 14))
        jobs.put(work(14, 15, 15))
        first = jobs.get_nowait()
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            batching.coalesce_delta(first, jobs)
        jobs.task_done()
        self.assertEqual(jobs.unfinished_tasks, 0)

    def test_started_batch_is_not_rewritten_by_a_later_tail(self):
        jobs = queue.Queue()
        first = work(0, 13)
        sent, _ = batching.coalesce_delta(first, jobs)
        jobs.put(work(13, 14, 14))
        final = jobs.get_nowait()
        tail, _ = batching.coalesce_delta(final, jobs)
        self.assertEqual(sent["end_token"], 13)
        self.assertEqual(tail["start_token"], 13)
        jobs.task_done()

    def test_boundary_extension_uses_latest_hint_without_resetting_deadline(self):
        (merged, count), jobs = self.delayed_merge(
            work(0, 13, 14), [work(13, 14, 16), work(14, 16, 16)]
        )
        self.assertEqual((merged["end_token"], merged["final_end_token"]), (16, 16))
        self.assertEqual(count, 3)
        self.assertEqual(jobs.unfinished_tasks, 0)

    def test_actual_source_hook_propagates_exact_final_computed_boundary(self):
        path = ROOT / "vllm/bridge_tp/phase8_source.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hook = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "maybe_publish_phase8_delta"
        )
        future = ast.parse("from __future__ import annotations").body[0]
        with tempfile.TemporaryDirectory() as folder:
            state = NS(
                finalized=False,
                finalizing=False,
                request_id="r",
                last_computed_token=2296,
                block_size=16,
                last_flush_monotonic=0,
                enqueue_gpu_delta=Mock(return_value=True),
            )
            namespace = dict(
                _state=state,
                time=NS(monotonic=lambda: 1),
                _get_request_block_ids=lambda *args: (list(range(200)), 16),
            )
            exec(
                compile(
                    ast.Module(body=[future, hook], type_ignores=[]), str(path), "exec"
                ),
                namespace,
            )
            namespace[hook.name](
                config=NS(
                    run_dir=Path(folder),
                    phase8_cutover_output_tokens=251,
                    gpu_direct_delta=True,
                    gpu_direct_delta_batch_tokens=16,
                    gpu_direct_delta_flush_ms=25,
                ),
                request_id="r",
                kv_caches=[],
                requests={"r": NS(output_token_ids=[0] * 250, num_prompt_tokens=2048)},
                input_batch=NS(
                    req_id_to_index={"r": 0}, num_computed_tokens_cpu=[2296]
                ),
                scheduler_output=NS(num_scheduled_tokens={"r": 1}),
                cache_dtype="auto",
                attn_groups=[],
            )
            payload = state.enqueue_gpu_delta.call_args.kwargs
            self.assertEqual(payload["end_token"], 2297)
            self.assertEqual(payload["final_end_token"], 2298)


if __name__ == "__main__":
    unittest.main()
