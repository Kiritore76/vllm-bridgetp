# SPDX-License-Identifier: Apache-2.0
"""Dispatch arrivals at fixed epochs, independently of response lifetimes."""

from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace

SCHEDULE_VERSION = "FIXED_EVENT_EPOCH_DUE_TIME_DISPATCH_V2"


def output_threshold(event):
    match = re.fullmatch(r"ANCHOR_OUTPUT_([1-9][0-9]*)", event)
    return int(match.group(1)) if match else None


def event_matches(row, event):
    if event == "M2_INITIAL_RATE":
        return row.get("kind") == "manager_m2_initial_rate"
    threshold = 1 if event == "ANCHOR_FIRST_OUTPUT" else output_threshold(event)
    if threshold is None:
        raise ValueError(f"unknown controller event: {event}")
    return row.get("kind") == "telemetry" and (
        int(row.get("output_tokens") or 0) >= threshold
    )


def event_monotonic(row, mono_origin, unix_origin):
    """Convert recorded occurrence time, never its delayed observation time."""
    stamp = row.get("unix_s")
    if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp <= 0:
        raise ValueError("controller event lacks a valid unix_s")
    return mono_origin + stamp - unix_origin


class ControllerEpochs:
    """Read audit once incrementally and freeze each first event occurrence."""

    def __init__(self, path, events, mono_origin, unix_origin):
        self.path = path
        self.events = set(events)
        self.mono_origin = mono_origin
        self.unix_origin = unix_origin
        self.epochs = {None: mono_origin}
        self.offset = 0

    def poll(self):
        if (
            self.events <= self.epochs.keys()
            or not self.path
            or not self.path.is_file()
        ):
            return
        with self.path.open(encoding="utf-8") as handle:
            handle.seek(self.offset)
            while True:
                line = handle.readline()
                if not line or not line.endswith("\n"):
                    break
                self.offset = handle.tell()
                row = json.loads(line)
                for event in self.events - self.epochs.keys():
                    if event_matches(row, event):
                        self.epochs[event] = event_monotonic(
                            row, self.mono_origin, self.unix_origin
                        )


@dataclass(frozen=True)
class ScheduledJob:
    job: dict
    epoch_monotonic: float
    scheduled_monotonic: float
    epoch_unix_s: float
    scheduled_unix_s: float
    dispatch_monotonic: float | None = None
    dispatch_unix_s: float | None = None
    dispatch_thread_cpu_s: float | None = None
    since_previous_dispatch_wall_s: float | None = None
    since_previous_dispatch_cpu_s: float | None = None


def due_jobs(jobs, audit, mono_origin, unix_origin, event_timeout_s=120.0):
    """Yield due arrivals without occupying response workers while sleeping."""
    pending = list(jobs)
    epochs = ControllerEpochs(
        audit,
        {j["start_after_event"] for j in jobs if j.get("start_after_event")},
        mono_origin,
        unix_origin,
    )
    while pending:
        epochs.poll()
        known = [j for j in pending if j.get("start_after_event") in epochs.epochs]
        unknown = [
            j for j in pending if j.get("start_after_event") not in epochs.epochs
        ]
        now = time.monotonic()
        if unknown and now - mono_origin > event_timeout_s:
            raise TimeoutError("background controller epoch was not observed")
        if known:
            job = min(
                known,
                key=lambda j: (
                    epochs.epochs[j.get("start_after_event")]
                    + float(j["start_after_s"])
                ),
            )
            epoch = epochs.epochs[job.get("start_after_event")]
            due = epoch + float(job["start_after_s"])
            if due <= now:
                pending.remove(job)
                epoch_unix = unix_origin + epoch - mono_origin
                yield ScheduledJob(
                    job,
                    epoch,
                    due,
                    epoch_unix,
                    epoch_unix + float(job["start_after_s"]),
                )
                continue
            delay = due - now
        else:
            delay = 0.02
        time.sleep(min(delay, 0.02 if unknown else 0.1))


def execute_schedule(
    jobs, run_job, audit, mono_origin, unix_origin, maximum_inflight=512,
    diagnostic_event=None,
):
    """Bound active HTTP workers; overload fails instead of silently queuing."""
    if not jobs or maximum_inflight < 1:
        raise ValueError("invalid background dispatch capacity")
    with ThreadPoolExecutor(max_workers=min(maximum_inflight, len(jobs))) as executor:
        futures, active = [], set()
        previous_wall, previous_cpu = time.monotonic(), time.thread_time()
        for planned in due_jobs(jobs, audit, mono_origin, unix_origin):
            active = {f for f in active if not f.done()}
            if len(active) >= maximum_inflight:
                raise RuntimeError("collector HTTP dispatch capacity exhausted")
            dispatch_wall, dispatch_cpu = time.monotonic(), time.thread_time()
            planned = replace(
                planned, dispatch_monotonic=dispatch_wall,
                dispatch_unix_s=time.time(), dispatch_thread_cpu_s=dispatch_cpu,
                since_previous_dispatch_wall_s=dispatch_wall - previous_wall,
                since_previous_dispatch_cpu_s=dispatch_cpu - previous_cpu,
            )
            previous_wall, previous_cpu = dispatch_wall, dispatch_cpu
            future = executor.submit(run_job, planned)
            if diagnostic_event is not None:
                diagnostic_event({
                    "kind": "collector_dispatch_submit_finished",
                    "job_id": planned.job["job_id"],
                    "dispatch_monotonic_s": dispatch_wall,
                    "submit_wall_s": time.monotonic() - dispatch_wall,
                    "submit_thread_cpu_s": time.thread_time() - dispatch_cpu,
                })
            futures.append(future)
            active.add(future)
        return [future.result() for future in as_completed(futures)]
