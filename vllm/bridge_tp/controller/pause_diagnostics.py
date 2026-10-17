# SPDX-License-Identifier: Apache-2.0
"""Opt-in timing evidence; producers aggregate without doing disk I/O."""

import contextlib
import gc
import json
import os
import threading
import time

_observer = None
_gc_monitor = None


class GCMonitor:
    """Record collections without logging, locking or doing I/O in callbacks."""

    def __init__(self, observer):
        from collections import deque

        self.observer = observer
        self.events = deque()
        self.active = None
        self.overflow = False
        self.callback = self._callback

    def _callback(self, phase, info):
        if phase == "start":
            self.active = (
                time.time(),
                time.monotonic(),
                time.thread_time(),
                info["generation"],
                threading.current_thread().name,
            )
        elif self.active is not None:
            wall, start, cpu, generation, thread = self.active
            self.active = None
            if len(self.events) >= 4096:
                self.overflow = True
                return
            self.events.append(
                (
                    wall,
                    time.monotonic() - start,
                    time.thread_time() - cpu,
                    generation,
                    thread,
                )
            )

    def flush(self):
        if self.overflow:
            raise RuntimeError("GC timing evidence overflow")
        while self.events:
            wall, duration, cpu, generation, thread = self.events.popleft()
            self.observer.record(f"gc.generation_{generation}", wall, duration, cpu)
            if duration >= 0.1:
                self.observer.journal.submit(
                    dict(
                        kind="gc_collection",
                        pid=os.getpid(),
                        generation=generation,
                        started_unix_s=wall,
                        duration_s=duration,
                        thread_cpu_s=cpu,
                        thread=thread,
                    )
                )

    def __enter__(self):
        global _gc_monitor
        self.previous = _gc_monitor
        _gc_monitor = self
        gc.callbacks.append(self.callback)
        return self

    def __exit__(self, *_):
        global _gc_monitor
        gc.callbacks.remove(self.callback)
        _gc_monitor = self.previous
        self.flush()


class TimingObserver:
    def __init__(self, journal):
        self.journal = journal
        self.lock = threading.Lock()
        self.buckets = {}
        self.last_emit = time.monotonic()
        self.last_tick = None
        self.closed = False
        journal.submit(
            dict(
                kind="timing_header",
                schema="PRESSURE_PAUSE_TIMING_V1",
                slow_s=0.1,
                slow_tick_interval_s=0.5,
                summary_interval_s=1,
                pid=os.getpid(),
            )
        )

    def record(self, name, start, duration, cpu_s):
        now = time.monotonic()
        with self.lock:
            if self.closed:
                return
            row = self.buckets.setdefault(
                name, dict(count=0, total_s=0.0, max_s=0.0, cpu_s=0.0)
            )
            row["count"] += 1
            row["total_s"] += duration
            row["max_s"] = max(row["max_s"], duration)
            row["cpu_s"] += cpu_s
            if duration >= (0.5 if name == "controller.tick_interval" else 0.1):
                self.journal.submit(
                    dict(
                        kind="slow_operation",
                        name=name,
                        started_unix_s=start,
                        completed_unix_s=time.time(),
                        duration_s=duration,
                        thread_cpu_s=cpu_s,
                        thread=threading.current_thread().name,
                    )
                )
            if now - self.last_emit >= 1:
                self._flush(now)

    def _flush(self, now):
        if self.buckets:
            self.journal.submit(
                dict(
                    kind="timing_summary",
                    unix_s=time.time(),
                    monotonic_s=now,
                    operations=self.buckets,
                )
            )
            self.buckets = {}
        self.last_emit = now

    def tick(self):
        now = time.monotonic()
        if self.last_tick is not None:
            self.record(
                "controller.tick_interval",
                time.time() - (now - self.last_tick),
                now - self.last_tick,
                0.0,
            )
        self.last_tick = now

    def close(self):
        with self.lock:
            self.closed = True
            self._flush(time.monotonic())
        self.journal.close()


@contextlib.contextmanager
def installed(observer):
    global _observer
    previous = _observer
    _observer = observer
    try:
        yield observer
    finally:
        _observer = previous


@contextlib.contextmanager
def span(name):
    observer = _observer
    if observer is None:
        yield
        return
    wall, start, cpu = time.time(), time.monotonic(), time.thread_time()
    try:
        yield
    finally:
        observer.record(name, wall, time.monotonic() - start, time.thread_time() - cpu)


def call(name, function, *args, **kwargs):
    with span(name):
        return function(*args, **kwargs)


def tick():
    if _gc_monitor is not None:
        _gc_monitor.flush()
    if _observer is not None:
        _observer.tick()


def verify(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if (
        len(rows) < 3
        or rows[0].get("schema") != "PRESSURE_PAUSE_TIMING_V1"
        or rows[-1].get("kind") != "evidence_footer"
        or rows[-1].get("dropped") != 0
        or rows[-1].get("writer_error") is not None
        or rows[-1].get("saved_records") != len(rows) - 1
    ):
        raise RuntimeError("incomplete pause timing evidence; preserve all data")
    slow = [r for r in rows if r.get("kind") == "slow_operation"]
    return dict(
        status="COMPLETED",
        records=len(rows) - 1,
        slow_operations=len(slow),
        maximum_operation_s=max((r["duration_s"] for r in slow), default=0),
        note="inclusive timings overlap; never add nested durations",
    )
