# SPDX-License-Identifier: Apache-2.0
"""Keep arrivals in the controller until a pool can start them.

A request sent to an engine queue stays there even after another pool goes idle,
and its queued prompt stops long requests from migrating off (mixed static L1A,
GPU 11-03). Room is counted from the controller's own streams, so stale telemetry
cannot open or close it. Only the prompt decides whether a request fits: output
length is not known here, and growth is migration's job (user 11-02, 11-04). Fresh
telemetry showing too little KV for the prompt keeps a request here. Without fresh
evidence a small prompt is sent and a long one waits, bounded by ``long_wait_s``.
"""

import bisect
import math
from collections import deque


class HoldQueue:
    """Small and long prompts each leave in arrival order.

    A long prompt without KV room waits alone so small prompts keep flowing. Past
    ``long_wait_s`` it is sent first, to a pool it fits or one with no fresh
    evidence; while every pool measures too little KV it keeps waiting. One thread
    calls this.
    """

    poll_s = 0.05

    def __init__(self, unstarted_limit, long_wait_s, long_tokens, max_age_s, active):
        if unstarted_limit <= 0 or long_wait_s < 0 or long_tokens <= 0:
            raise ValueError("invalid hold settings")
        self.limit, self.long_wait_s = unstarted_limit, long_wait_s
        self.long_tokens, self.max_age_s = long_tokens, max_age_s
        self.active = active
        self.short, self.long = deque(), deque()
        self.unstarted = {}
        # Read from the status thread: only appended to or replaced whole.
        self.arrived_at = []
        self.held_long = ()

    def is_long(self, job):
        return len(job["request"]["prompt"]) >= self.long_tokens

    def add(self, job, arrived):
        (self.long if self.is_long(job) else self.short).append((job, arrived))
        self.arrived_at.append(arrived)
        if self.is_long(job):
            self._publish_long()

    def _publish_long(self):
        self.held_long = tuple(len(job["request"]["prompt"]) for job, _ in self.long)

    def arrivals_since(self, cut):
        """Arrivals, not sends: a held queue caps what the engines receive."""
        times = self.arrived_at
        return len(times) - bisect.bisect_left(times, cut)

    def waiting(self):
        return len(self.short) + len(self.long)

    def _refresh(self):
        for pool_id, values in self.unstarted.items():
            self.unstarted[pool_id] = [
                v for v in values if not (v.recorder.tokens_seen or v.future.done())
            ]

    def _fresh(self, pool, wall):
        return (
            pool.free_blocks is not None
            and type(pool.unallocated_prompt_blocks) is int
            and pool.block_size > 0
            and 0 <= wall - pool.sampled_at <= self.max_age_s
        )

    def _prompt_fits(self, pool, job):
        return pool.free_blocks - pool.unallocated_prompt_blocks >= math.ceil(
            len(job["request"]["prompt"]) / pool.block_size
        )

    def _release(self, queue, now, pools):
        job, arrived = queue.popleft()
        if queue is self.long:
            self._publish_long()
        return dict(job, held_s=now - arrived), tuple(pools)

    def next(self, pools, now, wall):
        """The next request to send and the pools it may go to, or None."""
        self._refresh()
        pools = [p for p in pools if self.active(p.pool_id)]
        room = [p for p in pools if len(self.unstarted.get(p.pool_id, ())) < self.limit]
        if self.long and pools:
            job, arrived = self.long[0]
            fits = [
                p for p in room if self._fresh(p, wall) and self._prompt_fits(p, job)
            ]
            if fits:
                return self._release(self.long, now, fits)
            if now - arrived >= self.long_wait_s:
                allowed = [
                    p
                    for p in pools
                    if not self._fresh(p, wall) or self._prompt_fits(p, job)
                ]
                if allowed:
                    return self._release(self.long, now, allowed)
        if self.short:
            job = self.short[0][0]
            allowed = [
                p
                for p in room
                if not self._fresh(p, wall) or self._prompt_fits(p, job)
            ]
            if allowed:
                return self._release(self.short, now, allowed)
        return None

    def sent(self, job, value):
        self.unstarted.setdefault(value.pool_id, []).append(value)
