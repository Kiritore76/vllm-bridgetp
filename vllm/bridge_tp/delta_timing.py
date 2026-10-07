# SPDX-License-Identifier: Apache-2.0
"""Opt-in delta timings without adding CUDA synchronization to restoration."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import torch


class DeltaTiming:
    """Record CPU submission and CUDA stream intervals for one delta batch.

    CUDA intervals include stream contention and CPU enqueue gaps.
    CPU wall intervals include scheduling; thread CPU reports active CPU time.
    The caller's existing exact-readback synchronization completes the events.
    No event is synchronized or waited on by this observer.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.thread_began = time.thread_time()
        self.wall_began = time.perf_counter()
        self.rows: list[tuple[str, str, float, Any, float]] = []

    def mark(self, stage: str, layer: str = "", *, cuda: bool = True) -> None:
        event = None
        if cuda and self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
        self.rows.append((stage, layer, time.perf_counter(), event, time.thread_time()))

    def result(self) -> dict[str, Any]:
        cpu = defaultdict(float)
        gpu = defaultdict(float)
        active = defaultdict(float)
        incomplete = 0
        # Attribute each interval to its beginning marker. These are parallel
        # views of the same execution and must not be added together.
        for left, right in zip(self.rows, self.rows[1:]):
            stage, _, began, event, thread_began = left
            cpu[stage] += (right[2] - began) * 1000
            active[stage] += (right[4] - thread_began) * 1000
            end = right[3]
            if event is not None and end is not None:
                if event.query() and end.query():
                    gpu[stage] += event.elapsed_time(end)
                else:
                    incomplete += 1
        return {
            "cpu_stage_ms": dict(cpu),
            "thread_cpu_stage_ms": dict(active),
            "observer_wall_ms": (time.perf_counter() - self.wall_began) * 1000,
            "observer_thread_cpu_ms": (time.thread_time() - self.thread_began) * 1000,
            "cuda_stream_stage_ms": dict(gpu),
            "incomplete_cuda_intervals": incomplete,
            "marker_count": len(self.rows),
            "extra_cuda_synchronize": False,
            "note": (
                "CPU and CUDA overlap; CUDA includes stream waits and enqueue gaps."
            ),
        }


def delta_timing(device: torch.device) -> DeltaTiming | None:
    """Enable diagnostics only for an explicitly configured pilot."""
    if os.environ.get("BRIDGETP_DELTA_STAGE_TIMING", "") != "1":
        return None
    return DeltaTiming(device)
