# SPDX-License-Identifier: Apache-2.0
"""Bounded observation I/O; ownership changes retain synchronous durability."""

import json
import math
import threading
import time

from .audit import AuditLog
from .telemetry import TelemetryError


class PressureAudit:
    """Keep the complete async timeline and a separate durable critical journal."""

    # Admission tracks identity/unpublished prompt demand, with zero reserved
    # KV. It does not transfer ownership. Migration START/settlement still fsync.
    OBSERVATIONS = {
        "pressure_metrics_sample",
        "pressure_progress",
        "pressure_admission",
        "pressure_dispatch",
    }

    def __init__(self, path, journal, metadata):
        self.path, self.journal = path, journal
        self.durable = AuditLog(path.with_name("pressure_audit_durable.jsonl"))
        self.critical_records = 0
        self.records = 0
        self.closed = False
        self.lock = threading.Lock()
        self.critical_lock = threading.Lock()
        self.write(
            dict(
                kind="run_metadata",
                **metadata,
                observation_io="bounded_async_with_durable_critical_v2",
            )
        )

    def write(self, record):
        payload = dict(record)
        payload.setdefault("unix_s", time.time())
        # Freeze mutable caller data before handing it to a background writer.
        payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
        observation = payload.get("kind") in self.OBSERVATIONS or (
            payload.get("kind") == "pressure_ingress_reconcile"
            and payload.get("state_changed") is False
        )
        if not observation:
            # Do not hold the observation queue lock across fsync. A slow
            # migration journal must not delay unrelated arrivals.
            with self.critical_lock:
                if self.closed:
                    raise RuntimeError("pressure audit is closed")
                self.durable.write(payload)
                self.critical_records += 1
        with self.lock:
            if self.closed:
                raise RuntimeError("pressure audit is closed")
            self.journal.submit(payload)
            self.records += 1

    def close(self):
        with self.critical_lock, self.lock:
            if self.closed:
                return
            self.closed = True
        try:
            self.journal.close()  # Overflow/write errors fail the run, never hide loss.
            if self.journal.saved != self.records:
                raise RuntimeError("pressure audit observation count differs")
            receipt = dict(
                status="COMPLETED",
                records=self.records,
                critical_fsync_records=self.critical_records,
                dropped=self.journal.dropped,
                batch_flush_interval_s=self.journal.flush_interval_s,
            )
            self.path.with_name("pressure_io_receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            self.durable.close()


class BackgroundInventory:
    """Single journal reader with bounded work and a nonblocking cached poll.

    Reading/parsing takes place outside the cache lock. Integrity failures are
    sticky and raised to the controller; records are never skipped to catch up.
    """

    def __init__(self, reader, poll, interval_s=0.05):
        self.reader, self.read, self.interval_s = reader, poll, interval_s
        self.pool_id = reader.pool_id
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.cached = (None, None, reader.diagnostics(), None)
        self.observed_at = None
        self.error = None
        self.diagnostic = reader.diagnostics()
        self.thread = threading.Thread(
            target=self._run, name=f"pressure-inventory-{self.pool_id}", daemon=True
        )
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                value = self.read(time.time())
                state = (
                    value,
                    self.reader.observed_at,
                    self.reader.diagnostics(),
                    None,
                )
            except BaseException as error:
                state = (None, None, self.reader.diagnostics(), error)
                self.stop.set()
            with self.lock:
                self.cached = state
            self.stop.wait(self.interval_s)

    def poll(self, now):
        with self.lock:
            value, observed, diagnostic, error = self.cached
        if error is not None:
            raise RuntimeError("background inventory failed") from error
        self.observed_at, self.diagnostic = observed, diagnostic
        self.error = diagnostic["error"]
        self.deferred = diagnostic["deferred"]
        self.latest = value
        return value if value is None or value["sampled_at"] <= now else None

    def diagnostics(self):
        return dict(self.diagnostic)

    def close(self):
        self.stop.set()
        self.thread.join(3)
        if self.thread.is_alive():
            raise RuntimeError("background inventory did not stop")


class BackgroundMetrics:
    """At most one in-flight HTTP call per pool; cached reads never renew age."""

    def __init__(self, scrape, interval_s=1.0):
        if interval_s <= 0:
            raise ValueError("metrics interval must be positive")
        self.scrape, self.interval_s = scrape, interval_s
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.inventory = None
        self.value = None
        self.error = "background metrics not ready"
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="pressure-metrics"
        )
        self.thread.start()

    def update_inventory(self, inventory):
        total, usable = (
            inventory["total_blocks"],
            inventory.get("usage_denominator_blocks"),
        )
        if (
            type(usable) is not int
            or usable != total - 1
            or not 0 <= inventory["free_blocks"] <= usable
        ):
            raise ValueError("inventory usage denominator missing/invalid")
        with self.lock:
            self.inventory = {
                k: inventory[k]
                for k in ("total_blocks", "usage_denominator_blocks", "free_blocks")
            }

    def _run(self):
        while not self.stop.is_set():
            with self.lock:
                inventory = self.inventory
            if inventory is not None:
                started = time.time()
                try:
                    value = dict(self.scrape(inventory))
                    sampled = value.get("sampled_unix_s")
                    if (
                        not isinstance(sampled, (float, int))
                        or not math.isfinite(sampled)
                        or sampled > time.time()
                    ):
                        raise ValueError("invalid metrics timestamp")
                    # A slow response cannot be made fresh merely by arriving late.
                    value["sampled_unix_s"] = min(sampled, started)
                    with self.lock:
                        self.value, self.error = value, None
                except Exception as error:
                    with self.lock:
                        self.value, self.error = None, repr(error)
            self.stop.wait(self.interval_s)

    def snapshot(self, now, max_age_s):
        with self.lock:
            if (
                self.value is None
                or not 0 <= now - self.value["sampled_unix_s"] <= max_age_s
            ):
                raise TelemetryError(self.error or "background metrics stale/future")
            return dict(self.value)

    def close(self):
        self.stop.set()
        self.thread.join(3)
        if self.thread.is_alive():
            raise RuntimeError("background metrics did not stop after HTTP timeout")
