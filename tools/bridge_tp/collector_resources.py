# SPDX-License-Identifier: Apache-2.0
"""Low-frequency Linux host observations; no GPU calls or controller inputs."""

import gc
import sys
import threading
import time
from collections import deque
from pathlib import Path


class ResourceSampler:
    """Save raw CPU/memory counters for attributing collector dispatch delays."""

    def __init__(self, journal, interval_s=2.0, diagnostics=False):
        self.journal = journal
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.diagnostics = diagnostics
        self.gc_started = {}
        self.gc_events = deque()
        self.gc_dropped = 0
        self.gc_observer = self._gc_event
        if diagnostics:
            gc.callbacks.append(self.gc_observer)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _gc_event(self, phase, info):
        key = (threading.get_ident(), info["generation"])
        if phase == "start":
            self.gc_started[key] = (time.monotonic(), time.thread_time())
        elif phase == "stop" and key in self.gc_started:
            wall, cpu = self.gc_started.pop(key)
            elapsed = time.monotonic() - wall
            if elapsed >= 0.02:
                record = {
                    "kind": "collector_gc_pause", "unix_s": time.time(),
                    "generation": key[1], "thread_id": key[0],
                    "wall_s": elapsed, "thread_cpu_s": time.thread_time() - cpu,
                    "collected": info.get("collected"),
                    "uncollectable": info.get("uncollectable"),
                }
                if len(self.gc_events) < 4096:
                    self.gc_events.append(record)
                else:
                    self.gc_dropped += 1

    def drain_gc(self):
        while self.gc_events:
            self.journal.submit(self.gc_events.popleft())

    @staticmethod
    def thread_locations():
        """Bounded locations after a stall; no locals or response content."""
        result = []
        frames = sys._current_frames()
        main = threading.main_thread().ident
        ordered = sorted(frames, key=lambda thread_id: thread_id != main)
        for thread_id in ordered[:32]:
            frame = frames[thread_id]
            stack = []
            for _ in range(6):
                if frame is None:
                    break
                stack.append({"file": frame.f_code.co_filename,
                              "function": frame.f_code.co_name,
                              "line": frame.f_lineno})
                frame = frame.f_back
            result.append({"thread_id": thread_id, "stack": stack})
        return result

    def _run(self):
        paths = (
            "/proc/self/status", "/proc/self/stat", "/proc/self/schedstat",
            "/proc/self/sched", "/proc/loadavg",
            "/proc/pressure/cpu", "/proc/pressure/memory", "/proc/self/cgroup",
        )
        # cgroup mounts differ between containers; retain unavailable paths.
        groups = (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/cpu"))
        expected = time.monotonic()
        while not self.stop.is_set():
            self.drain_gc()
            lateness = max(0.0, time.monotonic() - expected)
            if self.diagnostics and lateness >= 0.5:
                self.journal.submit({
                    "kind": "collector_sampler_resumed_after_stall",
                    "unix_s": time.time(), "lateness_s": lateness,
                    "stack_basis": "AFTER_RESUMPTION_NOT_PROOF_OF_BLOCKER",
                    "thread_locations": self.thread_locations(),
                })
            values = {}
            for path in [Path(p) for p in paths] + [
                group / name for group in groups
                for name in ("cpu.stat", "cpu.max", "memory.current",
                             "memory.events", "cpu.cfs_quota_us",
                             "cpu.cfs_period_us")
            ]:
                try:
                    values[str(path)] = path.read_text()[:8192]
                except OSError as error:
                    values[str(path)] = {"unavailable": type(error).__name__}
            self.journal.submit({
                "kind": "collector_resources", "unix_s": time.time(),
                "monotonic_s": time.monotonic(), "process_cpu_s": time.process_time(),
                "counters": values,
            })
            expected = time.monotonic() + self.interval_s
            self.stop.wait(self.interval_s)

    def close(self):
        self.stop.set()
        self.worker.join()
        if self.diagnostics:
            gc.callbacks.remove(self.gc_observer)
        self.drain_gc()
        if self.gc_dropped:
            self.journal.submit({"kind": "collector_gc_diagnostic_loss",
                                 "dropped": self.gc_dropped})
