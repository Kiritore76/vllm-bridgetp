# SPDX-License-Identifier: Apache-2.0
"""Keep arrivals in the controller until a pool can start them.

A request sent to an engine queue stays there even after another pool goes idle,
and its queued prompt stops long requests from migrating off (mixed static L1A,
GPU 11-03). Room is counted from the controller's own streams, so stale telemetry
cannot open or close it. Fresh telemetry showing too little KV also keeps a request
here: sent into a full engine it only waits there and pushes others out (GPU 11-03,
user decision). Without fresh evidence a short request is sent and a long request
waits, bounded by ``long_wait_s``.
"""

import math
from collections import deque


class HoldQueue:
    """Short and long requests each leave in arrival order.

    A long request without KV room waits alone so short requests keep flowing.
    Past ``long_wait_s`` it is sent first, to a pool it fits or one with no fresh
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
        self.running_long = []

    def is_long(self, job):
        request = job["request"]
        return len(request["prompt"]) + int(request["max_tokens"]) >= self.long_tokens

    def add(self, job, arrived):
        (self.long if self.is_long(job) else self.short).append((job, arrived))

    def waiting(self):
        return len(self.short) + len(self.long)

    def _refresh(self):
        for pool_id, values in self.unstarted.items():
            self.unstarted[pool_id] = [
                v for v in values if not (v.recorder.tokens_seen or v.future.done())
            ]
        self.running_long = [v for v in self.running_long if not v.future.done()]

    def _reserved_blocks(self, pool_id, block_size):
        """What the long requests already sent here will still allocate."""
        total = 0
        for value in self.running_long:
            recorder = value.recorder
            owner = recorder.target_pool if recorder.target_arrived else value.pool_id
            if owner == pool_id:
                left = int(value.payload["max_tokens"]) - recorder.tokens_seen
                if not recorder.tokens_seen:
                    left += len(value.payload["prompt"])
                total += math.ceil(max(0, left) / block_size)
        return total

    def _fresh(self, pool, wall):
        return (
            pool.free_blocks is not None
            and type(pool.unallocated_prompt_blocks) is int
            and pool.block_size > 0
            and 0 <= wall - pool.sampled_at <= self.max_age_s
        )

    @staticmethod
    def _final_blocks(pool, job):
        request = job["request"]
        return math.ceil(
            (len(request["prompt"]) + int(request["max_tokens"])) / pool.block_size
        )

    def _long_fits(self, pool, job, wall):
        return (
            self._fresh(pool, wall)
            and pool.free_blocks
            - pool.unallocated_prompt_blocks
            - self._reserved_blocks(pool.pool_id, pool.block_size)
            >= self._final_blocks(pool, job)
        )

    def _short_fits(self, pool, job, wall):
        # Short requests end in seconds, so long requests' later growth is not
        # charged, and stale telemetry must not stop them.
        return (
            not self._fresh(pool, wall)
            or pool.free_blocks - pool.unallocated_prompt_blocks
            >= self._final_blocks(pool, job)
        )

    def _release(self, queue, now, pools):
        job, arrived = queue.popleft()
        return dict(job, held_s=now - arrived), tuple(pools)

    def next(self, pools, now, wall):
        """The next request to send and the pools it may go to, or None."""
        self._refresh()
        pools = [p for p in pools if self.active(p.pool_id)]
        room = [p for p in pools if len(self.unstarted.get(p.pool_id, ())) < self.limit]
        if self.long and pools:
            job, arrived = self.long[0]
            fits = [p for p in room if self._long_fits(p, job, wall)]
            if fits:
                return self._release(self.long, now, fits)
            if now - arrived >= self.long_wait_s:
                allowed = [
                    p
                    for p in pools
                    if not self._fresh(p, wall) or self._long_fits(p, job, wall)
                ]
                if allowed:
                    return self._release(self.long, now, allowed)
        if self.short:
            allowed = [p for p in room if self._short_fits(p, self.short[0][0], wall)]
            if allowed:
                return self._release(self.short, now, allowed)
        return None

    def sent(self, job, value):
        self.unstarted.setdefault(value.pool_id, []).append(value)
        if self.is_long(job):
            self.running_long.append(value)
