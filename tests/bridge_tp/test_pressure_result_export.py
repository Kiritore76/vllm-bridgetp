# SPDX-License-Identifier: Apache-2.0
"""Post-workload evidence export, failure preservation and GC diagnostics."""

import json
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from tools.bridge_tp import pressure_result_export as export
from tools.bridge_tp.pressure_cpu import load

diagnostics = load("pause_diagnostics")


def completed(index, stats):
    future = Future()
    future.set_result(dict(request_started_unix_s=100 + index, finish_reason="length"))
    return NS(
        client_id=f"client-{index}",
        server_id=f"server-{index}",
        pool_id="pool",
        payload=dict(prompt=[1, 2], max_tokens=1),
        future=future,
        recorder=NS(stats=stats),
        dispatch_entered_at=100 + index,
        admission_finished_at=100 + index,
        executor_submitted_at=100 + index,
        http_thread_started_at=100 + index,
        held_s=0,
    )


class ResultExportTests(unittest.TestCase):
    def test_more_than_one_gc_buffer_is_drained_without_changing_gc_policy(self):
        monitor = diagnostics.GCMonitor(Mock())
        calls = []

        def stats():
            for _ in range(64):
                monitor._callback("start", dict(generation=0))
                monitor._callback("stop", {})
            calls.append(1)
            return dict(emitted=[dict(index=0, token_id=7, unix_s=101)], token_ids=[7])

        values = {str(i): completed(i, stats) for i in range(80)}
        jobs = [dict(id=str(i), at_s=i) for i in range(80)]
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(diagnostics, "_gc_monitor", monitor),
        ):
            path = Path(folder) / "results.json"
            receipt = export.write_request_results(path, values, jobs, 100)
            rows = json.loads(path.read_text())
            self.assertEqual(len(rows), 80)
            self.assertEqual(rows[1]["dispatch_timing"]["planned_unix_s"], 101)
            self.assertEqual(rows[1]["response"]["token_ids"], [7])
            self.assertEqual(receipt["requests"], 80)
            self.assertEqual(len(monitor.events), 0)
            self.assertFalse(monitor.overflow)
            self.assertEqual(monitor.observer.record.call_count, 5120)

    def test_missing_consumption_still_rejects_diagnostic_overflow(self):
        monitor = diagnostics.GCMonitor(Mock())
        for _ in range(4097):
            monitor._callback("start", dict(generation=0))
            monitor._callback("stop", {})
        with (
            patch.object(diagnostics, "_gc_monitor", monitor),
            self.assertRaisesRegex(RuntimeError, "GC timing evidence overflow"),
        ):
            diagnostics.flush_gc()

    def test_only_one_request_stats_object_stays_materialized(self):
        counts = dict(live=0, max=0)

        class Tracked(dict):
            def __init__(self):
                super().__init__(emitted=[dict(index=0, token_id=7, unix_s=101)])
                counts["live"] += 1
                counts["max"] = max(counts["live"], counts["max"])

            def __del__(self):
                counts["live"] -= 1

        values = {str(i): completed(i, Tracked) for i in range(20)}
        with tempfile.TemporaryDirectory() as folder:
            export.write_request_results(
                Path(folder) / "results.json",
                values,
                [dict(id=str(i), at_s=i) for i in range(20)],
                100,
            )
        self.assertEqual(counts, dict(live=0, max=1))

    def test_export_failure_keeps_existing_final_and_partial_evidence(self):
        bad = completed(0, lambda: dict(unserializable={1}))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.json"
            path.write_text("previous")
            with self.assertRaises(TypeError):
                export.write_request_results(
                    path, {"one": bad}, [dict(id="one", at_s=0)], 100
                )
            self.assertEqual(path.read_text(), "previous")
            self.assertTrue(path.with_name(path.name + ".partial").is_file())

    def test_rename_failure_is_not_silently_accepted(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(export.os, "replace", side_effect=OSError("replace failed")),
        ):
            path = Path(folder) / "results.json"
            with self.assertRaisesRegex(OSError, "replace failed"):
                export.write_request_results(path, {}, [], 100)
            self.assertFalse(path.exists())
            self.assertEqual(
                json.loads(path.with_name(path.name + ".partial").read_text()), []
            )


if __name__ == "__main__":
    unittest.main()
