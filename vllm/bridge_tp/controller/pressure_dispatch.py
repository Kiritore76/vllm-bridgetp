# SPDX-License-Identifier: Apache-2.0
"""Arrival deadlines independent of observation and migration control I/O."""

import threading
import time

from .pressure_policy import AdmissionDeferred


class ScheduledDispatch:
    """One arrival producer; publish only completed submissions to readers.

    Pool readings retain their original timestamps. The ingress applies its
    normal stale-data fallback; publishing a cache never makes evidence fresh.
    This isolates control-loop waits, not whole-process/GIL or host stalls.
    """

    def __init__(
        self,
        jobs,
        submit,
        pools,
        origin_monotonic,
        max_seconds,
        hold=None,
        observe_held=False,
    ):
        self.jobs = sorted(jobs, key=lambda job: job["at_s"])
        self.submit = submit
        self.pools = tuple(pools)
        self.origin = origin_monotonic
        self.deadline = origin_monotonic + max_seconds
        self.hold = hold
        self.observe_held = observe_held
        self.held_demand = (0, 0)
        # (oldest held short arrival, oldest held long arrival, held long
        # prompt tokens); arrivals are monotonic times, None when none is held.
        self.held_detail = (None, None, ())
        self.held_demand_at = -float("inf")
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
        # Fixed-route arrivals skip the hold queue; one that does not fit yet
        # (AdmissionDeferred) waits here, in arrival order, and is retried on
        # every poll before the held arrivals (11-12 directkv run: the second
        # pinned C long request crashed the dispatcher instead of waiting).
        pinned = []
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
                    pinned.append(job)
                else:
                    hold.add(job, due)
            if pinned:
                with self.lock:
                    pools = self.pools
                waiting = []
                for job in pinned:
                    if waiting:  # keep the arrival order per fixed route
                        waiting.append(job)
                        continue
                    try:
                        self._send(job, pools)
                    except AdmissionDeferred:
                        waiting.append(job)
                pinned = waiting
            self._publish_held_demand()
            with self.lock:
                pools = self.pools
            deferred = []
            while not self.stop.is_set():
                released_at = time.monotonic()
                released = hold.next(pools, released_at, time.time())
                if released is None:
                    break
                job, allowed = released
                try:
                    value = self._send(job, allowed)
                except AdmissionDeferred:
                    # Do not let one stale-fit long request starve later shorts.
                    # Restore after this scan so it is retried only next poll.
                    deferred.append((job, released_at))
                    continue
                hold.sent(job, value)
            for job, released_at in deferred:
                hold.retry(job, released_at)
            self._publish_held_demand()
            if index == len(self.jobs) and not hold.waiting() and not pinned:
                return
            wait = hold.poll_s
            if index < len(self.jobs):
                due = self.origin + self.jobs[index]["at_s"]
                wait = min(wait, max(0.0, due - time.monotonic()))
            self.stop.wait(wait)

    def _publish_held_demand(self):
        """Only the arrival writer iterates deques; readers get one atomic tuple."""
        if not self.observe_held:
            return
        now = time.monotonic()
        if not self.hold.short and not self.hold.long:
            self.held_demand, self.held_demand_at = (0, 0), now
            self.held_detail = (None, None, ())
            return
        if now - self.held_demand_at < 0.5:
            return
        short, long = 0, 0
        oldest_short = oldest_long = None
        prompts = []
        for queue in (self.hold.short, self.hold.long):
            for job, arrived in queue:
                body = job["request"]
                if len(body["prompt"]) + body["max_tokens"] >= 8192:
                    long += 1
                    prompts.append(len(body["prompt"]))
                    if oldest_long is None or arrived < oldest_long:
                        oldest_long = arrived
                else:
                    short += 1
                    if oldest_short is None or arrived < oldest_short:
                        oldest_short = arrived
        self.held_demand = (short, long)
        self.held_detail = (oldest_short, oldest_long, tuple(prompts))
        self.held_demand_at = now

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
