# SPDX-License-Identifier: Apache-2.0
"""Low-frequency Linux host observations; no GPU calls or controller inputs."""

import threading
import time
from pathlib import Path


class ResourceSampler:
    """Save raw CPU/memory counters for attributing collector dispatch delays."""

    def __init__(self, journal, interval_s=2.0):
        self.journal = journal
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        paths = (
            "/proc/self/status", "/proc/self/stat", "/proc/loadavg",
            "/proc/pressure/cpu", "/proc/pressure/memory", "/proc/self/cgroup",
        )
        # cgroup mounts differ between containers; retain unavailable paths.
        groups = (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/cpu"))
        while not self.stop.is_set():
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
            self.stop.wait(self.interval_s)

    def close(self):
        self.stop.set()
        self.worker.join()
