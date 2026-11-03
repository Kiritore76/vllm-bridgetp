# SPDX-License-Identifier: Apache-2.0
"""Tracking only open streams answers what scanning every stream would."""

import unittest
from concurrent.futures import Future
from types import SimpleNamespace as NS

from tools.bridge_tp.pressure_progress import (
    DueJobs,
    StreamTracker,
    completed_streams,
)


def stream(at=0.0, committed=False, tokens=0):
    return NS(
        future=Future(),
        dispatched_at=at,
        recorder=NS(progress=lambda: (tokens, 0, committed)),
    )


def finish(value, error=None):
    if error:
        value.future.set_exception(error)
    else:
        value.future.set_result({})


class StreamTrackerTests(unittest.TestCase):
    def test_finished_streams_retire_and_completed_matches_a_full_scan(self):
        session = NS(sessions=[])
        streams = {f"j{i}": stream() for i in range(6)}
        tracker = StreamTracker()
        tracker.update(streams, session)
        for key in ("j0", "j3"):
            finish(streams[key])
        tracker.update(streams, session)
        self.assertEqual(tracker.retired, 2)
        self.assertEqual(sorted(tracker.open), ["j1", "j2", "j4", "j5"])
        self.assertEqual(
            tracker.completed(session), completed_streams(streams, session)
        )
        self.assertFalse(tracker.all_done())
        for key in ("j1", "j2", "j4", "j5"):
            finish(streams[key])
        tracker.update(streams, session)
        self.assertTrue(tracker.all_done())
        self.assertEqual(tracker.completed(session), 6)

    def test_new_streams_are_picked_up_between_ticks(self):
        session = NS(sessions=[])
        streams = {"a": stream()}
        tracker = StreamTracker()
        tracker.update(streams, session)
        finish(streams["a"])
        tracker.update(streams, session)
        streams["b"] = stream()
        tracker.update(streams, session)
        self.assertEqual(list(tracker.open), ["b"])
        self.assertEqual(tracker.retired, 1)

    def test_committed_stream_stays_open_after_its_source_ends(self):
        session = NS(sessions=[])
        value = stream(committed=True)
        streams = {"m": value}
        tracker = StreamTracker()
        finish(value)
        tracker.update(streams, session)
        self.assertEqual(list(tracker.open), ["m"])
        self.assertEqual(tracker.retired, 0)

    def test_stream_a_session_waits_on_stays_open_and_uncounted(self):
        value = stream()
        target = Future()
        session = NS(sessions=[NS(stream=value, target_future=target)])
        streams = {"m": value}
        tracker = StreamTracker()
        finish(value)
        tracker.update(streams, session)
        self.assertEqual(tracker.retired, 0)
        self.assertEqual(tracker.completed(session), 0)
        self.assertEqual(
            tracker.completed(session), completed_streams(streams, session)
        )
        target.set_result({})
        tracker.update(streams, session)
        self.assertEqual(tracker.retired, 1)
        self.assertEqual(tracker.completed(session), 1)

    def test_a_failed_stream_still_stops_the_run(self):
        value = stream()
        finish(value, RuntimeError("stream failed"))
        with self.assertRaises(RuntimeError):
            StreamTracker().update({"m": value}, NS(sessions=[]))

    def test_recent_keeps_only_the_window(self):
        streams = {f"j{i}": stream(at=float(i)) for i in range(10)}
        tracker = StreamTracker()
        tracker.update(streams, NS(sessions=[]))
        self.assertEqual(len(tracker.recent(9.0, 3.0)), 4)
        self.assertEqual(len(tracker.arrivals), 4)


class DueJobsTests(unittest.TestCase):
    def test_matches_a_full_scan(self):
        jobs = [dict(id=f"j{i}", at_s=float(i // 2)) for i in range(40)]
        due = DueJobs(jobs)
        dispatched = {}
        for tick in range(25):
            elapsed = tick * 0.7
            for job in sorted(jobs, key=lambda j: j["at_s"])[:tick]:
                dispatched[job["id"]] = True
            self.assertEqual(
                due.queued(elapsed, dispatched),
                [
                    j["id"]
                    for j in sorted(jobs, key=lambda j: j["at_s"])
                    if elapsed >= j["at_s"] and j["id"] not in dispatched
                ],
            )


if __name__ == "__main__":
    unittest.main()
