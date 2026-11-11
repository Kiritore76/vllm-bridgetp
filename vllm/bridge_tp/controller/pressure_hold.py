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

Release order (unified hold; user 11-11): ``fifo`` scans arrivals oldest first.
``attainable`` serves the requests that can still meet their first-token budget
before those already past it: under overload FIFO makes every arrival wait the
whole backlog, so none meets it (six-card split-back: TTFT p50 7s, 0/433). Order:
(1) waiting at least ``starve_s``, oldest first, so a late request is not passed
over for long; (2) still attainable, oldest first; (3) the rest, oldest first.
Every request is still sent; only the order of held ones changes.
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
        order="fifo",
        starve_s=10.0,
    ):
        if unstarted_limit <= 0 or long_wait_s < 0 or long_tokens <= 0:
            raise ValueError("invalid hold settings")
        if order not in {"fifo", "attainable"} or not starve_s > 0:
            raise ValueError("invalid hold release order")
        if order != "fifo" and unified_guard_blocks is None:
            raise ValueError("the attainable order is for the unified hold")
        self.order, self.starve_s = order, starve_s
        # Engine side of TTFT (send to first token) of recent short requests:
        # under overload the engine queue alone takes seconds (GPU 11-11).
        self.engine_delays = deque(maxlen=20)
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
            kept = []
            for v in values:
                if v.recorder.tokens_seen and self.order != "fifo":
                    self._observe_start(v)
                if not (v.recorder.tokens_seen or v.future.done()):
                    kept.append(v)
            self.unstarted[pool_id] = kept
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
    # First-token budgets of the strict SLO (short 2s, long 10s). After release
    # the engine still needs the median send-to-first-token of recent short
    # requests (at least 0.5s), plus 0.16 ms per extra prompt token (GPU 11-10).
    # A fixed 0.58s instead kept few more in time than FIFO: offline 11-11,
    # L2C +160 against +1979 with this online estimate.
    ttft_budget_s = dict(short=2.0, long=10.0)
    release_margin_s = 0.5
    prefill_s_per_token = 1.6e-4

    # The engine delay is measured on short prompts; a longer one adds its
    # extra prefill on top.
    short_prompt_tokens = 512

    def _observe_start(self, value):
        """Record send-to-first-token of a started short request."""
        if len(value.payload["prompt"]) >= self.long_tokens:
            return
        try:
            first = value.recorder.proxy.emitted[0].unix_s
        except (AttributeError, IndexError):
            return
        delay = first - value.dispatched_at
        if math.isfinite(delay) and delay >= 0:
            self.engine_delays.append(delay)

    def engine_delay_s(self):
        """Median recent engine delay, never below the fixed margin."""
        if not self.engine_delays:
            return self.release_margin_s
        ordered = sorted(self.engine_delays)
        return max(self.release_margin_s, ordered[len(ordered) // 2])

    def _attainable(self, job, waited, engine_s):
        prompt = len(job["request"]["prompt"])
        budget = self.ttft_budget_s["long" if prompt >= self.long_tokens else "short"]
        extra = max(0, prompt - self.short_prompt_tokens) * self.prefill_s_per_token
        return waited + engine_s + extra <= budget

    def _scan_order(self, now):
        """Indexes of held arrivals in the order they are offered a pool."""
        count = len(self.short)
        if self.order == "fifo":
            return range(min(count, self.unified_scan))
        starved, attainable, late = [], [], []
        # The queue is in arrival order: starved ones lead, attainable ones are
        # among the newest (no budget exceeds 10s), the late are in between.
        newest = max(self.ttft_budget_s.values())
        index = 0
        while index < count and now - self.short[index][1] >= self.starve_s:
            starved.append(index)
            index += 1
        first_recent = index
        while first_recent < count and now - self.short[first_recent][1] > newest:
            first_recent += 1
        engine_s = self.engine_delay_s()
        for i in range(first_recent, count):
            job, arrived = self.short[i]
            fits = self._attainable(job, now - arrived, engine_s)
            (attainable if fits else late).append(i)
        late = list(range(index, first_recent)) + late
        late.sort()
        return (starved + attainable + late)[: self.unified_scan]

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
        for index in self._scan_order(now):
            job, arrived = self.short[index]
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
                p for p in room if not self._fresh(p, wall) or self._prompt_fits(p, job)
            ]
            if allowed:
                return self._release(self.short, now, allowed)
        return None

    def sent(self, job, value):
        self.unstarted.setdefault(value.pool_id, []).append(value)
        self.running.append(value)

    def retry(self, job, released_at):
        """Restore a final-admission refusal without inventing a new arrival."""
        if not self.unified:
            raise ValueError("capacity retry is for unified hold only")
        job = dict(job)
        arrived = released_at - job.pop("held_s")
        index = bisect.bisect_right([at for _, at in self.short], arrived)
        self.short.insert(index, (job, arrived))
