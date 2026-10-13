# SPDX-License-Identifier: Apache-2.0
"""Bounded JSON evidence writer; producers never perform disk I/O."""

import hashlib
import json
import queue
import threading
import time


class AsyncJournal:
    """Drain immutable event records and expose losses instead of hiding them."""

    def __init__(self, path, capacity=65536, batch_size=512, flush_interval_s=0.05):
        if capacity < 1 or batch_size < 1 or flush_interval_s <= 0:
            raise ValueError("invalid journal buffering parameters")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.error = None
        self.saved = 0
        self.write_batches = 0
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.closed = False
        self.handle = path.open("x", encoding="utf-8", buffering=65536)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, value):
        """Return False on overflow; the footer makes evidence loss auditable."""
        if self.closed:
            raise RuntimeError("evidence journal is closed")
        try:
            self.queue.put_nowait(value)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def submit_batch(self, records):
        """One queue acquisition per bounded batch; preserve every record."""
        if self.closed:
            raise RuntimeError("evidence journal is closed")
        if not 1 <= len(records) <= 64:
            raise ValueError("journal batches must contain 1..64 records")
        try:
            self.queue.put_nowait(("records", tuple(records)))
            return True
        except queue.Full:
            self.dropped += len(records)
            return False

    def _run(self):
        last_flush = time.monotonic()
        while True:
            try:
                value = self.queue.get(timeout=self.flush_interval_s)
            except queue.Empty:
                if self.error is None:
                    try:
                        self.handle.flush()
                    except Exception as error:
                        self.error = error
                last_flush = time.monotonic()
                continue
            values = [value]
            while values[-1] is not None and len(values) < self.batch_size:
                try:
                    values.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            stopping = values[-1] is None
            entries = values[:-1] if stopping else values
            records = []
            for entry in entries:
                if isinstance(entry, tuple) and entry[0] == "records":
                    records.extend(entry[1])
                else:
                    records.append(entry)
            try:
                if records and self.error is None:
                    payload = "".join(
                        json.dumps(record, allow_nan=False) + "\n"
                        for record in records
                    )
                    self.handle.write(payload)
                    self.saved += len(records)
                    self.write_batches += 1
                if self.error is None and (
                    stopping
                    or time.monotonic() - last_flush >= self.flush_interval_s
                ):
                    self.handle.flush()
                    last_flush = time.monotonic()
            except Exception as error:
                self.error = error
            finally:
                for _ in values:
                    self.queue.task_done()
            if stopping:
                return

    def close(self):
        """Drain at shutdown, writing a final evidence-integrity receipt."""
        if self.closed:
            if self.error or self.dropped:
                raise RuntimeError("asynchronous evidence incomplete") from self.error
            return
        self.closed = True
        self.queue.put(None)
        self.worker.join()
        try:
            self.handle.write(
                json.dumps(
                    {
                        "kind": "evidence_footer",
                        "dropped": self.dropped,
                        "writer_error": str(self.error) if self.error else None,
                        "saved_records": self.saved,
                        "write_batches": self.write_batches,
                        "queue_capacity": self.queue.maxsize,
                        "flush_interval_s": self.flush_interval_s,
                    }
                )
                + "\n"
            )
        finally:
            self.handle.close()
        if self.error or self.dropped:
            raise RuntimeError("asynchronous evidence incomplete") from self.error


class FeatureJournal:
    """Persist CPU feature batches in a separate bounded disk worker."""

    def __init__(self, directory, capacity=128):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.dropped = 0
        self.saved = 0
        self.sha256 = {}
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, feature_id, payload):
        try:
            self.queue.put_nowait((feature_id, payload))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self):
        import torch

        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    feature_id, payload = item
                    path = self.directory / f"{feature_id}.pt"
                    torch.save(payload, path)
                    self.sha256[path.name] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    self.saved += 1
            except Exception as error:
                self.error = error
            finally:
                self.queue.task_done()

    def close(self):
        self.queue.put(None)
        self.worker.join()
        receipt = {
            "format_version": 1,
            "saved_batches": self.saved,
            "dropped_batches": self.dropped,
            "sha256": self.sha256,
            "writer_error": str(self.error) if self.error else None,
        }
        (self.directory / "receipt.json").write_text(
            json.dumps(receipt) + "\n", encoding="utf-8"
        )
        if self.error or self.dropped:
            raise RuntimeError("predictor features incomplete") from self.error
