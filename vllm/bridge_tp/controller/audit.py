# SPDX-License-Identifier: Apache-2.0
"""Append-only audit trail.

Phase 9 pass condition 3 requires that every decision records its inputs,
benefit, cost, and resulting action. This writer is deliberately dumb: one
JSON object per line, flushed and fsynced on every record, never rewritten.
An inspector must be able to reconstruct the run from this file alone even if
the controller process dies mid-run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: str | os.PathLike[str], run_metadata: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8")
        if run_metadata is not None:
            self.write({"kind": "run_metadata", **run_metadata})

    def write(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("unix_s", time.time())
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def read_audit(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Stream records back, skipping a torn final line if the run was killed."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                return
