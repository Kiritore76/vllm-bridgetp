# SPDX-License-Identifier: Apache-2.0
"""Small, atomic progress snapshots and a workload-aware no-progress deadline."""

import json
import os
from pathlib import Path


def save_progress(path: Path, **status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def stream_progress(dispatched, target_future=None):
    """Count actual output/completion progress even without engine inventory."""
    return (
        tuple(
            (key, stream.future.done(), stream.recorder.progress())
            for key, stream in dispatched.items()
        ),
        target_future.done() if target_future is not None else None,
    )


def completed_streams(dispatched, target_stream=None, target_future=None):
    """A migrated request completes when its target stream also completes."""
    return sum(
        stream.future.done()
        and (
            stream is not target_stream or target_future is None or target_future.done()
        )
        for stream in dispatched.values()
    )


def inventory_progress(inventories):
    """Include prefill computation even before a request emits its first token."""
    return tuple(
        (
            pool["pool_id"],
            row["request_id"],
            row["generated"],
            row.get("computed_tokens", 0),
        )
        for pool in inventories
        for row in pool["requests"]
    )


def workload_progress(dispatched, last_inventory, session, reservations):
    """One marker for healthy and unavailable inventory, without poll timestamps."""
    return (
        len(dispatched),
        completed_streams(dispatched, session.stream, session.target_future),
        last_inventory,
        session.record.state.value if session.record else "no_session",
        session.settled,
        tuple(
            (
                row.key,
                row.state,
                row.allocated_blocks,
                row.prefill_observed,
                row.service_observed,
            )
            for row in reservations
        ),
        stream_progress(dispatched, session.target_future),
    )


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
