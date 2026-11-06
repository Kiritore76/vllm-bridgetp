# SPDX-License-Identifier: Apache-2.0
"""Keep arrivals in the controller until a pool can start them.

A request sent to an engine queue stays there even after another pool goes idle,
and its queued prompt stops long requests from migrating off (mixed static L1A,
GPU 11-03). Room is counted from the controller's own streams, so stale telemetry
cannot open or close it. Only the prompt decides whether a request fits: output
length is not known here, and growth is migration's job (user 11-02, 11-04). Fresh
telemetry showing too little KV for the prompt keeps a request here. Without fresh
evidence a small prompt is sent and a long one waits, bounded by ``long_wait_s``.

With ``tp1_long_kv``, a TP1 takes no long prompt while it already serves a request
past short size or its KV is used this far (user 11-04): two long requests on one
TP1 preempt each other and stop its queue. Short requests are not affected.
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
    short_blocks = 64

    def __init__(
        self,
        unstarted_limit,
        long_wait_s,
        long_tokens,
        max_age_s,
        active,
        degree=None,
        tp1_long_kv=0.0,
        unified_guard_blocks=None,
    ):
        if unstarted_limit <= 0 or long_wait_s < 0 or long_tokens <= 0:
            raise ValueError("invalid hold settings")
        # Unified (user 11-06): one queue and two rules, the engine queue limit
        # and a prompt that fits after the guard; no long/short distinction.
        self.unified = unified_guard_blocks is not None
        if self.unified and (unified_guard_blocks < 0 or tp1_long_kv):
            raise ValueError("unified hold takes a guard and no TP1 long rule")
        self.guard_blocks = unified_guard_blocks
        self.limit, self.long_wait_s = unstarted_limit, long_wait_s
        self.long_tokens, self.max_age_s = long_tokens, max_age_s
        self.active = active
        self.short, self.long = deque(), deque()
        self.unstarted = {}
        self.running = []
        self.degree, self.tp1_long_kv = degree, tp1_long_kv
        # Read from the status thread: only appended to or replaced whole.
        self.arrived_at = []
        self.held_long = ()

    def is_long(self, job):
        return len(job["request"]["prompt"]) >= self.long_tokens

    def add(self, job, arrived):
        if self.unified:
            self.short.append((job, arrived))
            self.arrived_at.append(arrived)
            return
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
        self.running = [v for v in self.running if not v.future.done()]

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

    def _takes_long(self, pool):
        if not self.tp1_long_kv or self.degree(pool.pool_id) != 1:
            return True
        limit = self.short_blocks * pool.block_size
        for value in self.running:
            recorder = value.recorder
            owner = recorder.target_pool if recorder.target_arrived else value.pool_id
            if (
                owner == pool.pool_id
                and len(value.payload["prompt"]) + recorder.tokens_seen > limit
            ):
                return False
        total = getattr(pool, "total_blocks", None)
        return not (
            total
            and pool.free_blocks is not None
            and 1 - pool.free_blocks / total >= self.tp1_long_kv
        )

    # Arrivals scanned per call; one that fits nowhere does not stop later ones.
    unified_scan = 256

    def _unified_fits(self, pool, job):
        """Prompt blocks fit after unallocated, sent-but-unstarted and the guard."""
        size = pool.block_size
        need = math.ceil(len(job["request"]["prompt"]) / size)
        sent = sum(
            math.ceil(len(v.payload["prompt"]) / size)
            for v in self.unstarted.get(pool.pool_id, ())
        )
        return (
            pool.free_blocks - pool.unallocated_prompt_blocks - sent - self.guard_blocks
            >= need
        )

    def _never_fits(self, pools, job):
        """No active instance could hold the prompt even when empty."""
        known = [p for p in pools if p.total_blocks and p.block_size > 0]
        if not known or len(known) != len(pools):
            return False
        return all(
            p.total_blocks - self.guard_blocks
            < math.ceil(len(job["request"]["prompt"]) / p.block_size)
            for p in known
        )

    def _next_unified(self, pools, room, now, wall):
        """First arrival that fits somewhere; safety for the two dead ends.

        A prompt no instance could ever hold fails the run at once instead of
        waiting forever. Without any fresh reading for ``long_wait_s`` the first
        arrival goes to an instance with engine room, where admission picks the
        least full one (the old hold's bounded wait, user 11-06).
        """
        fresh = [p for p in room if self._fresh(p, wall)]
        for index, (job, arrived) in enumerate(self.short):
            if index >= self.unified_scan:
                break
            if self._never_fits(pools, job):
                size = len(job["request"]["prompt"])
                raise ValueError(
                    f"prompt of {size} tokens can never fit any active instance"
                )
            fits = [p for p in fresh if self._unified_fits(p, job)]
            if fits:
                del self.short[index]
                return dict(job, held_s=now - arrived), tuple(fits)
        if self.short and room and not any(self._fresh(p, wall) for p in pools):
            job, arrived = self.short[0]
            if now - arrived >= self.long_wait_s:
                self.short.popleft()
                return dict(job, held_s=now - arrived), tuple(room)
        return None

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
        if self.unified:
            return self._next_unified(pools, room, now, wall)
        if self.long and pools:
            job, arrived = self.long[0]
            fits = [
                p
                for p in room
                if self._fresh(p, wall)
                and self._prompt_fits(p, job)
                and self._takes_long(p)
            ]
            if fits:
                return self._release(self.long, now, fits)
            if now - arrived >= self.long_wait_s:
                allowed = [
                    p
                    for p in pools
                    if self._takes_long(p)
                    and (not self._fresh(p, wall) or self._prompt_fits(p, job))
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
        self.running.append(value)
