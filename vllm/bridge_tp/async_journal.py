# SPDX-License-Identifier: Apache-2.0
"""Bounded JSON evidence writer; producers never perform disk I/O."""

import hashlib
import json
import queue
import threading


class AsyncJournal:
    """Drain immutable event records and expose losses instead of hiding them."""

    def __init__(self, path, capacity=4096):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.error = None
        self.handle = path.open("x", encoding="utf-8", buffering=1)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, value):
        """Return False on overflow; the footer makes evidence loss auditable."""
        try:
            self.queue.put_nowait(value)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self):
        while True:
            value = self.queue.get()
            try:
                if value is None:
                    return
                if self.error is None:
                    self.handle.write(json.dumps(value, allow_nan=False) + "\n")
            except Exception as error:
                self.error = error
            finally:
                self.queue.task_done()

    def close(self):
        """Drain at shutdown, writing a final evidence-integrity receipt."""
        self.queue.put(None)
        self.worker.join()
        try:
            self.handle.write(
                json.dumps(
                    {
                        "kind": "evidence_footer",
                        "dropped": self.dropped,
                        "writer_error": str(self.error) if self.error else None,
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
