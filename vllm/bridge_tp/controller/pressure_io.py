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

    OBSERVATIONS = {"pressure_metrics_sample", "pressure_progress"}

    def __init__(self, path, journal, metadata):
        self.path, self.journal = path, journal
        self.durable = AuditLog(path.with_name("pressure_audit_durable.jsonl"))
        self.critical_records = 0
        self.records = 0
        self.closed = False
        self.lock = threading.Lock()
        self.write(
            dict(
                kind="run_metadata",
                **metadata,
                observation_io="bounded_async_with_durable_critical_v1",
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
        with self.lock:
            if self.closed:
                raise RuntimeError("pressure audit is closed")
            if not observation:
                self.durable.write(payload)
                self.critical_records += 1
            self.journal.submit(payload)
            self.records += 1

    def close(self):
        with self.lock:
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
