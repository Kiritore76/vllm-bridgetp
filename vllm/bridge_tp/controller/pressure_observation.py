# SPDX-License-Identifier: Apache-2.0
"""Decision-before inventory ingestion and live streaming ITL windows."""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from pathlib import Path

from .pressure_policy import PoolReading, PressureConfig


def scrape_inventory_metrics(scraper, inventory):
    """Use the allocator's explicit usage denominator, never its raw pool size."""
    total = inventory["total_blocks"]
    usable = inventory.get("usage_denominator_blocks")
    if (
        type(usable) is not int
        or usable != total - 1
        or not 0 <= inventory["free_blocks"] <= usable
    ):
        raise ValueError("inventory usage denominator missing/invalid")
    scraper.total_kv_blocks = usable
    return scraper.scrape().__dict__


class InventoryReader:
    def __init__(self, path: Path, pool_id: str):
        self.path, self.pool_id = path, pool_id
        self.offset = 0
        self.latest = None
        self.observed_at = None
        self.error = None
        self.error_detail = None
        self.deferred = None

    def poll(self, now: float) -> dict | None:
        self.deferred = None
        if not self.path.exists():
            return None
        if self.path.stat().st_size < self.offset:
            raise ValueError("inventory journal was truncated")
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    self.offset = start
                    break
                self.offset = stream.tell()
                row = json.loads(line)
                if row.get("kind") == "evidence_footer":
                    if row.get("dropped") or row.get("writer_error"):
                        self.error = "inventory evidence incomplete"
                        if self.error_detail is None:
                            self.error_detail = self._row_diagnostic(row, now)
                    continue
                if (
                    row.get("schema") != "PRESSURE_INVENTORY_V1"
                    or row.get("pool_id") != self.pool_id
                    or row.get("dropped_before")
                    or row.get("writer_error")
                    or row.get("inventory_error")
                ):
                    self.error = "inventory identity/integrity invalid"
                    if self.error_detail is None:
                        self.error_detail = self._row_diagnostic(row, now)
                    continue
                if not math.isfinite(row["sampled_at"]):
                    self.error = "inventory timestamp is not finite"
                    if self.error_detail is None:
                        self.error_detail = self._row_diagnostic(row, now)
                    continue
                if row["sampled_at"] > now:
                    # A producer can append while this poll is reading. Keep
                    # that line unread until the caller's time reaches it;
                    # never use future evidence or poison subsequent polls.
                    self.offset = start
                    self.deferred = self._row_diagnostic(row, now)
                    break
                if self.latest and row["generation"] <= self.latest["generation"]:
                    raise ValueError("inventory generation did not increase")
                self.latest = row
                self.observed_at = now
        # A future line does not invalidate the already accepted past prefix.
        # The policy still checks its original timestamp/age before migration.
        return None if self.error else self.latest

    def _row_diagnostic(self, row, now):
        return dict(
            pool_id=self.pool_id,
            row_pool_id=row.get("pool_id"),
            schema=row.get("schema"),
            generation=row.get("generation"),
            sampled_at=row.get("sampled_at"),
            poll_unix_s=now,
            dropped_before=row.get("dropped_before"),
            dropped=row.get("dropped"),
            writer_error=row.get("writer_error"),
            inventory_error=row.get("inventory_error"),
        )

    def diagnostics(self):
        return dict(
            pool_id=self.pool_id,
            path=str(self.path),
            error=self.error,
            error_detail=self.error_detail,
            deferred=self.deferred,
            last_generation=self.latest.get("generation") if self.latest else None,
            last_sampled_at=self.latest.get("sampled_at") if self.latest else None,
        )


class NativeServiceWindow:
    """Updated by the real SSE token callback, before the controller reads it.

    Active-request gaps are retained as right-censored diagnostic evidence.
    Completed-request averages and final quality weights never enter decisions.
    """

    def __init__(self, window_s=3.0, itl_reference_s=0.1, slow_budget=0.01):
        if window_s <= 0 or itl_reference_s <= 0 or not 0 < slow_budget <= 1:
            raise ValueError("invalid native service window")
        self.window_s, self.reference, self.slow_budget = (
            window_s,
            itl_reference_s,
            slow_budget,
        )
        self._intervals = {name: deque() for name in ("tp1", "tp4")}
        self._active = {}
        self._arrivals = {"tp1": -math.inf, "tp4": -math.inf}
        self._lock = threading.RLock()

    def arrive(self, request_id, pool_id, now):
        with self._lock:
            if request_id in self._active or pool_id not in self._intervals:
                raise ValueError("duplicate request or unknown pool")
            self._active[request_id] = (pool_id, now, None)
            self._arrivals[pool_id] = max(now, self._arrivals[pool_id])

    def token(self, request_id, now):
        with self._lock:
            pool, arrived, last = self._active[request_id]
            if now < (last if last is not None else arrived):
                raise ValueError("stream token time regressed")
            if last is not None:
                self._intervals[pool].append((now, now - last, request_id))
            self._active[request_id] = (pool, arrived, now)

    def finish(self, request_id):
        with self._lock:
            self._active.pop(request_id, None)

    def snapshot(self, pool_id, now):
        with self._lock:
            intervals = deque(
                v for v in self._intervals[pool_id] if v[0] >= now - self.window_s
            )
            self._intervals[pool_id] = intervals
            values = sorted(v[1] for v in intervals if v[0] <= now)
            gaps = [
                now - last
                for pool, _, last in self._active.values()
                if pool == pool_id and last is not None
            ]
            pending_ttft = [
                now - arrived
                for pool, arrived, last in self._active.values()
                if pool == pool_id and last is None
            ]
            slow = (
                sum(v > self.reference for v in values) / len(values)
                if values
                else None
            )
            p95 = None
            if values:
                position = 0.95 * (len(values) - 1)
                lo = int(position)
                hi = min(lo + 1, len(values) - 1)
                p95 = values[lo] + (position - lo) * (values[hi] - values[lo])
            request_intervals = {}
            for captured, interval, request_id in intervals:
                if captured <= now:
                    request_intervals.setdefault(request_id, []).append(interval)
            return dict(
                samples=len(values),
                latest_at=max(v[0] for v in intervals if v[0] <= now)
                if values
                else None,
                p95_s=p95,
                slow_fraction=slow,
                reference_s=self.reference,
                ratio=max(p95 / self.reference, slow / self.slow_budget)
                if values
                else None,
                max_active_gap_s=max(gaps, default=0),
                max_pending_ttft_s=max(pending_ttft, default=0),
                last_arrival=self._arrivals[pool_id],
                active=sum(pool == pool_id for pool, _, _ in self._active.values()),
                request_rates={
                    request_id: len(v) / sum(v)
                    for request_id, v in request_intervals.items()
                    if len(v) >= 5 and sum(v) > 0
                },
            )


def pool_reading(
    inventory: dict,
    observed_at: float,
    service: dict,
    cfg: PressureConfig,
    now: float,
    *,
    growth_blocks_s=None,
    heartbeat: dict | None = None,
):
    """Build the pure policy reading only after reading published inventory.

    An old empty scheduler report may describe idle while an independently live
    endpoint is serving nothing; callers must refresh/liveness-check that report
    separately. This adapter deliberately does not invent a heartbeat.
    """
    sampled_at = inventory["sampled_at"]
    if (
        heartbeat is not None
        and not inventory["requests"]
        and service["active"] == 0
        and service["last_arrival"] <= sampled_at
        and 0 <= now - heartbeat.get("sampled_unix_s", -1e30) <= cfg.max_age_s
        and heartbeat.get("num_running") == 0
        and heartbeat.get("num_waiting") == 0
        and heartbeat.get("free_kv_blocks") == inventory["free_blocks"]
    ):
        sampled_at = heartbeat["sampled_unix_s"]
        observed_at = max(observed_at, sampled_at)
    fresh = 0 <= now - sampled_at <= cfg.max_age_s
    idle = (
        fresh
        and not inventory["requests"]
        and inventory["waiting"] == 0
        and service["active"] == 0
        and service["last_arrival"] <= inventory["sampled_at"]
    )
    ratio = service["ratio"]
    # A still-open slow interval is not silently absent from pressure. Preserve
    # the true completed sample count/freshness; a gap cannot fabricate samples.
    if ratio is not None:
        ratio = max(ratio, service["max_active_gap_s"] / service["reference_s"])
    scheduled = inventory.get("last_scheduled_batch")
    step = 0 if idle else None
    if (
        scheduled
        and not scheduled.get("error")
        and (0 <= now - scheduled["sampled_at"] <= cfg.max_age_s)
        and not idle
    ):
        step = scheduled["prefill_tokens"]
    return PoolReading(
        pool_id=inventory["pool_id"],
        sampled_at=sampled_at,
        published_at=observed_at,
        free_blocks=inventory["free_blocks"],
        block_size=inventory["block_size"],
        context_limit=inventory["context_limit"],
        decode_ratio=ratio,
        decode_samples=service["samples"],
        decode_sampled_at=service["latest_at"],
        idle=idle,
        pending_prompt_tokens=inventory["pending_prompt_tokens"],
        waiting=inventory["waiting"],
        growth_blocks_s=growth_blocks_s,
        native_itl_p95_s=service["p95_s"],
        slow_interval_fraction=service["slow_fraction"],
        prefill_batch_tokens=step,
    )
