# SPDX-License-Identifier: Apache-2.0
"""Small, atomic progress snapshots and a workload-aware no-progress deadline."""

import collections
import itertools
import json
import os
from pathlib import Path

from tools.bridge_tp.pressure_cpu import load


def save_progress(path: Path, **status):
    temporary = path.with_suffix(".tmp")
    with load("pause_diagnostics").span("progress.write_replace"):
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
    sessions = getattr(target_stream, "sessions", None)
    if sessions is not None:
        pending = {
            id(s.stream)
            for s in sessions
            if s.target_future is not None and not s.target_future.done()
        }
        return sum(
            s.future.done() and id(s) not in pending for s in dispatched.values()
        )
    return sum(
        stream.future.done()
        and (
            stream is not target_stream or target_future is None or target_future.done()
        )
        for stream in dispatched.values()
    )


class StreamTracker:
    """What each control tick still has to look at among all dispatched streams.

    Scanning every past stream each tick grew with run length (GPU 11-03). A
    stream leaves once it can no longer change: its source ended without error,
    no migration session is waiting on it, and nothing was committed to a target,
    because a committed stream keeps emitting after its source ends.
    """

    def __init__(self):
        self.open = {}
        self.retired = 0
        self.seen = 0
        self.arrivals = collections.deque()

    def update(self, dispatched, session=None):
        for key in itertools.islice(dispatched, self.seen, None):
            self.open[key] = dispatched[key]
            self.arrivals.append(dispatched[key])
        self.seen = len(dispatched)
        pending = {
            id(s.stream)
            for s in getattr(session, "sessions", ())
            if s.target_future is not None and not s.target_future.done()
        }
        for key, stream in tuple(self.open.items()):
            if not stream.future.done():
                continue
            stream.future.result()  # a failed stream stops the run, as before
            committed = stream.recorder.progress()[2]
            if not committed and id(stream) not in pending:
                del self.open[key]
                self.retired += 1

    def completed(self, session):
        return self.retired + completed_streams(self.open, session)

    def all_done(self):
        return all(stream.future.done() for stream in self.open.values())

    def recent(self, now, window):
        """Streams dispatched in the last ``window`` seconds."""
        while self.arrivals and now - self.arrivals[0].dispatched_at > window:
            self.arrivals.popleft()
        return [v for v in self.arrivals if 0 <= now - v.dispatched_at <= window]


class DueJobs:
    """Jobs due and not yet dispatched; the dispatcher works in this same order."""

    def __init__(self, jobs):
        self.jobs = sorted(jobs, key=lambda job: job["at_s"])
        self.next = 0

    def queued(self, elapsed, dispatched):
        while self.next < len(self.jobs) and self.jobs[self.next]["id"] in dispatched:
            self.next += 1
        out = []
        for job in itertools.islice(self.jobs, self.next, None):
            if elapsed < job["at_s"]:
                break
            if job["id"] not in dispatched:
                out.append(job["id"])
        return out


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
        completed_streams(dispatched, session)
        if hasattr(session, "sessions")
        else completed_streams(dispatched, session.stream, session.target_future),
        last_inventory,
        "CANCELLING"
        if getattr(session, "cancel_future", None) is not None and not session.settled
        else session.record.state.value
        if session.record
        else "no_session",
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
        tuple(
            (
                s.migration_id,
                s.target_future.done() if s.target_future is not None else None,
            )
            for s in getattr(session, "sessions", [])
        ),
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
