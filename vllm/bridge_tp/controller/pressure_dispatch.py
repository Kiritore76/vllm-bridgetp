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

    def __init__(self, jobs, submit, pools, origin_monotonic, max_seconds, hold=None):
        self.jobs = sorted(jobs, key=lambda job: job["at_s"])
        self.submit = submit
        self.pools = tuple(pools)
        self.origin = origin_monotonic
        self.deadline = origin_monotonic + max_seconds
        self.hold = hold
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
            if self.hold is not None:
                self._run_held()
                return
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

    def _send(self, job, pools):
        value = self.submit(job, pools)
        with self.lock:
            self.dispatched[job["id"]] = value
        return value

    def _run_held(self):
        """Arrivals join the hold queue on time and leave when a pool has room."""
        hold, index = self.hold, 0
        while not self.stop.is_set():
            now = time.monotonic()
            if now >= self.deadline:
                return
            while index < len(self.jobs):
                job = self.jobs[index]
                due = self.origin + job["at_s"]
                if due > now:
                    break
                index += 1
                if job.get("route") is not None:
                    with self.lock:
                        pools = self.pools
                    self._send(job, pools)
                else:
                    hold.add(job, due)
            with self.lock:
                pools = self.pools
            while not self.stop.is_set():
                released = hold.next(pools, time.monotonic(), time.time())
                if released is None:
                    break
                job, allowed = released
                hold.sent(job, self._send(job, allowed))
            if index == len(self.jobs) and not hold.waiting():
                return
            wait = hold.poll_s
            if index < len(self.jobs):
                due = self.origin + self.jobs[index]["at_s"]
                wait = min(wait, max(0.0, due - time.monotonic()))
            self.stop.wait(wait)

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
