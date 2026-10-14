# SPDX-License-Identifier: Apache-2.0
"""Opt-in, sampled batch telemetry with no device synchronization in generation."""

import os
import time
from collections import deque
from pathlib import Path

SCHEMA = "BRIDGETP_BATCH_PERFORMANCE_V1"


def trace_enabled():
    return bool(os.environ.get("BRIDGETP_PERFORMANCE_TRACE_DIR"))


def batch_composition(requests, scheduled):
    """Describe scheduled work before advancing computed-token counters.

    Rebuilding already generated history is separate from prompt prefill and
    ordinary decode. Context lengths exclude the current query token. Speculative
    decode is flagged by the caller and must not enter the ordinary-decode fit.
    """
    rows = []
    totals = dict.fromkeys(
        (
            "decode_requests",
            "decode_tokens",
            "prefill_tokens",
            "recompute_tokens",
            "decode_context_sum",
            "prefill_attention_work",
            "attention_work",
        ),
        0,
    )
    for req_id, count in scheduled.items():
        request = requests[req_id]
        q = int(count)
        c = int(request.num_computed_tokens)
        prompt = int(request.num_prompt_tokens)
        total = int(request.num_tokens)
        if q <= 0 or c < 0 or prompt < 0 or total < prompt:
            raise ValueError("invalid scheduled batch state")
        p = min(q, max(0, prompt - c))
        r = min(q - p, max(0, total - 1 - max(c, prompt)))
        d = q - p - r
        totals["prefill_tokens"] += p
        totals["recompute_tokens"] += r
        totals["decode_tokens"] += d
        totals["decode_requests"] += int(d > 0)
        totals["decode_context_sum"] += c + p + r if d else 0
        totals["prefill_attention_work"] += p * c + p * (p + 1) // 2
        totals["attention_work"] += q * c + q * (q + 1) // 2
        rows.append(
            {
                "request_id": req_id,
                "query_tokens": q,
                "computed_before": c,
                "prompt_tokens": prompt,
                "known_tokens": total,
                "prefill_tokens": p,
                "recompute_tokens": r,
                "decode_tokens": d,
                "preemptions": int(getattr(request, "num_preemptions", 0)),
            }
        )
    return {
        **totals,
        "scheduled_requests": len(rows),
        "scheduled_tokens": sum(scheduled.values()),
        "requests": rows,
    }


class BatchSampler:
    """Select every eighth nonempty batch plus every prompt/recompute batch."""

    def __init__(self):
        self.step = 0
        self.every = int(os.environ.get("BRIDGETP_PERFORMANCE_SAMPLE_EVERY", "8"))
        if self.every < 1:
            raise ValueError("performance sampling interval must be positive")

    def snapshot(self, requests, output, *, kv_usage, running, waiting):
        self.step += 1
        if not output.total_num_scheduled_tokens:
            return None
        composition = batch_composition(requests, output.num_scheduled_tokens)
        special = composition["prefill_tokens"] or composition["recompute_tokens"]
        if (self.step - 1) % self.every and not special:
            return None
        return {
            "schema": SCHEMA,
            "step_id": self.step,
            "scheduled_unix_s": time.time(),
            "scheduled_monotonic_s": time.monotonic(),
            "sample_every": self.every,
            "kv_usage": float(kv_usage),
            "running_requests": running,
            "waiting_requests": waiting,
            "speculative": bool(output.scheduled_spec_decode_tokens),
            "preempted_request_ids": sorted(output.preempted_req_ids or ()),
            **composition,
        }


class HostTrace:
    def __init__(self):
        from vllm.bridge_tp.async_journal import AsyncJournal

        directory = Path(os.environ["BRIDGETP_PERFORMANCE_TRACE_DIR"])
        self.journal = AsyncJournal(directory / "engine.jsonl", capacity=2048)
        self.closed = False
        self.journal.submit(
            {
                "kind": "trace_header",
                "schema": SCHEMA,
                "role": "engine",
                "pid": os.getpid(),
            }
        )

    @classmethod
    def from_env(cls):
        return cls() if trace_enabled() else None

    def submitted(self, metadata):
        if metadata is not None:
            metadata["submitted_monotonic_s"] = time.monotonic()
            metadata["submitted_unix_s"] = time.time()

    def completed(self, metadata):
        if metadata is not None and not self.closed:
            now = time.monotonic()
            self.journal.submit(
                {
                    "kind": "engine_batch",
                    **metadata,
                    "completed_unix_s": time.time(),
                    "completion_latency_ms": 1000
                    * (now - metadata["submitted_monotonic_s"]),
                    "scope": "submit_to_model_and_sampling_result",
                }
            )

    def close(self):
        self.closed = True
        self.journal.close()
        return {"status": "FINALIZED", "saved_records": self.journal.saved}


class GpuTrace:
    """CUDA event spans are queried later; no per-batch synchronize or allreduce."""

    def __init__(self, rank, device, event_factory, stream_factory):
        from vllm.bridge_tp.async_journal import AsyncJournal

        directory = Path(os.environ["BRIDGETP_PERFORMANCE_TRACE_DIR"])
        self.rank = rank
        self.device = device
        self.event_factory = event_factory
        self.stream_factory = stream_factory
        self.pending = deque()
        self.capacity = 64
        self.dropped = 0
        self.error = None
        self.closed = False
        self.journal = AsyncJournal(directory / f"gpu_rank{rank}.jsonl", capacity=2048)
        self.journal.submit(
            {
                "kind": "trace_header",
                "schema": SCHEMA,
                "role": "gpu",
                "rank": rank,
                "pid": os.getpid(),
                "device": str(device),
            }
        )

    @classmethod
    def from_env(cls, rank, device):
        if not trace_enabled():
            return None
        import torch

        return cls(
            rank,
            device,
            lambda: torch.cuda.Event(enable_timing=True),
            lambda: torch.cuda.current_stream(device),
        )

    def poll(self):
        """Only read elapsed time after the recorded end event reports ready."""
        try:
            while self.pending and self.pending[0][2].query():
                metadata, start, end, extra = self.pending.popleft()
                self.journal.submit(
                    {
                        "kind": "gpu_batch",
                        **metadata,
                        **extra,
                        "rank": self.rank,
                        "gpu_forward_logits_ms": start.elapsed_time(end),
                        "scope": "forward_and_logits_current_stream",
                        "observed_unix_s": time.time(),
                    }
                )
        except Exception as error:
            self.error = str(error)

    def begin(self, metadata, **extra):
        self.poll()
        if metadata is None or self.closed or self.error:
            return None
        if len(self.pending) >= self.capacity:
            self.dropped += 1
            return None
        try:
            start, end = self.event_factory(), self.event_factory()
            start.record(self.stream_factory())
            return dict(metadata), start, end, extra
        except Exception as error:
            self.error = str(error)
            return None

    def end(self, ticket):
        if ticket is not None:
            try:
                ticket[2].record(self.stream_factory())
                self.pending.append(ticket)
            except Exception as error:
                self.error = str(error)

    def close(self, timeout_s=5.0):
        if not self.closed:
            # Called only after the measured workload drains, never in decode.
            deadline = time.monotonic() + timeout_s
            self.poll()
            while self.pending and not self.error and time.monotonic() < deadline:
                time.sleep(0.001)
                self.poll()
            self.closed = True
            self.journal.submit(
                {
                    "kind": "gpu_trace_receipt",
                    "rank": self.rank,
                    "pending": len(self.pending),
                    "dropped_samples": self.dropped,
                    "timing_error": self.error,
                }
            )
            self.journal.close()
        if self.pending or self.dropped or self.error:
            raise RuntimeError("GPU performance evidence incomplete")
        return {
            "status": "FINALIZED",
            "rank": self.rank,
            "saved_records": self.journal.saved,
        }
