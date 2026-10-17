# SPDX-License-Identifier: Apache-2.0
"""Arrival deadlines independent of observation and migration control I/O."""

import threading
import time


class ScheduledDispatch:
    """One arrival producer; publish only completed submissions to readers.

    Pool readings retain their original timestamps. The ingress applies its
    normal stale-data fallback; publishing a cache never makes evidence fresh.
    This isolates control-loop waits, not whole-process/GIL or host stalls.
    """

    def __init__(self, jobs, submit, pools, origin_monotonic, max_seconds):
        self.jobs = sorted(jobs, key=lambda job: job["at_s"])
        self.submit = submit
        self.pools = tuple(pools)
        self.origin = origin_monotonic
        self.deadline = origin_monotonic + max_seconds
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.dispatched = {}
        self.error = None
        self.thread = threading.Thread(
            target=self._run, name="pressure-arrivals", daemon=True
        )

    def publish(self, pools):
        # PoolReading is frozen; callers must replace, never mutate its dicts.
        with self.lock:
            self.pools = tuple(pools)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            for job in self.jobs:
                due = self.origin + job["at_s"]
                if due >= self.deadline:
                    break
                while True:
                    remaining = due - time.monotonic()
                    if remaining <= 0:
                        break
                    if self.stop.wait(remaining):
                        return
                if self.stop.is_set() or time.monotonic() >= self.deadline:
                    return
                with self.lock:
                    pools = self.pools
                value = self.submit(job, pools)
                with self.lock:
                    self.dispatched[job["id"]] = value
        except BaseException as error:
            with self.lock:
                self.error = error
            self.stop.set()

    def snapshot(self, *, check=True):
        with self.lock:
            error, values = self.error, dict(self.dispatched)
        if check and error is not None:
            raise RuntimeError("arrival dispatcher failed") from error
        return values

    def close(self):
        self.stop.set()
        self.thread.join(3)
        if self.thread.is_alive():
            raise RuntimeError("arrival dispatcher did not stop")
