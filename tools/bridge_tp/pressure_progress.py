# SPDX-License-Identifier: Apache-2.0
"""Small, atomic progress snapshots and a workload-aware no-progress deadline."""

import json
import os
from pathlib import Path


def save_progress(path: Path, **status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


class ProgressWatch:
    def __init__(self, timeout_s):
        self.timeout_s = timeout_s
        self.marker = None
        self.changed_at = None

    def update(self, now, marker, *, work_due, reason):
        if self.changed_at is None or marker != self.marker or not work_due:
            self.marker, self.changed_at = marker, now
        if work_due and now - self.changed_at >= self.timeout_s:
            raise TimeoutError(
                f"no workload progress for {self.timeout_s:g}s: {reason}"
            )
