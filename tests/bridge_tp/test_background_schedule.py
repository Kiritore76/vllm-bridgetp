# SPDX-License-Identifier: Apache-2.0
"""Regression checks for burst evidence loss and delayed arrival epochs."""

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bridge_tp.background_schedule import ControllerEpochs, execute_schedule

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "background_journal_test", ROOT / "vllm/bridge_tp/async_journal.py"
)
journal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(journal)


class BackgroundScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_late_reader_uses_original_event_and_measures_lag(self):
        path = self.root / "audit.jsonl"
        stamp = time.time() - 30
        path.write_text(
            json.dumps({"kind": "telemetry", "output_tokens": 1, "unix_s": stamp})
            + "\n"
        )
        mono, unix = time.monotonic(), time.time()
        jobs = [
            {
                "job_id": "source",
                "start_after_s": 2,
                "start_after_event": "ANCHOR_FIRST_OUTPUT",
            }
        ]
        records = execute_schedule(jobs, lambda row: row, path, mono, unix)
        self.assertAlmostEqual(records[0].epoch_unix_s, stamp, places=4)
        self.assertGreater(time.monotonic() - records[0].scheduled_monotonic, 27)

    def test_first_epoch_is_shared_and_not_replaced_by_later_telemetry(self):
        path = self.root / "audit.jsonl"
        path.write_text(
            json.dumps({"kind": "telemetry", "output_tokens": 1, "unix_s": 100}) + "\n"
        )
        epochs = ControllerEpochs(path, {"ANCHOR_FIRST_OUTPUT"}, 1000, 100)
        epochs.poll()
        with path.open("a") as handle:
            handle.write(
                json.dumps({"kind": "telemetry", "output_tokens": 10, "unix_s": 105})
                + "\n"
            )
        epochs.poll()
        self.assertEqual(epochs.epochs["ANCHOR_FIRST_OUTPUT"], 1000)

    def test_partial_event_waits_for_complete_line(self):
        path = self.root / "audit.jsonl"
        path.write_text('{"kind":"telemetry","output_tokens":1,')
        epochs = ControllerEpochs(path, {"ANCHOR_FIRST_OUTPUT"}, 1000, 100)
        epochs.poll()
        self.assertNotIn("ANCHOR_FIRST_OUTPUT", epochs.epochs)
        with path.open("a") as handle:
            handle.write('"unix_s":100}\n')
        epochs.poll()
        self.assertEqual(epochs.epochs["ANCHOR_FIRST_OUTPUT"], 1000)

    def test_missing_event_timestamp_fails_instead_of_resetting_epoch(self):
        path = self.root / "audit.jsonl"
        path.write_text('{"kind":"telemetry","output_tokens":1}\n')
        epochs = ControllerEpochs(path, {"ANCHOR_FIRST_OUTPUT"}, 1000, 100)
        with self.assertRaisesRegex(ValueError, "valid unix_s"):
            epochs.poll()

    def test_more_than_128_responses_do_not_block_due_arrivals(self):
        count = 140
        jobs = [{"job_id": str(i), "start_after_s": 0} for i in range(count)]
        starts, lock, release = [], threading.Lock(), threading.Event()

        def run(row):
            with lock:
                starts.append(row.job["job_id"])
                if len(starts) == count:
                    release.set()
            self.assertTrue(release.wait(3), "due arrival was blocked by responses")
            return row.job["job_id"]

        results = execute_schedule(jobs, run, None, time.monotonic(), time.time())
        self.assertEqual(set(results), {str(i) for i in range(count)})

    def test_dispatch_limit_is_explicit_not_a_hidden_queue(self):
        jobs = [{"job_id": str(i), "start_after_s": 0} for i in range(3)]
        release = threading.Event()

        def run(_):
            release.wait(0.1)

        with self.assertRaisesRegex(RuntimeError, "dispatch capacity exhausted"):
            execute_schedule(
                jobs, run, None, time.monotonic(), time.time(), maximum_inflight=2
            )

    def test_target_dispatch_does_not_wait_for_source_epoch(self):
        path = self.root / "audit.jsonl"
        jobs = [
            {
                "job_id": "source",
                "start_after_s": 0,
                "start_after_event": "ANCHOR_FIRST_OUTPUT",
            },
            {"job_id": "target", "start_after_s": 0},
        ]

        def run(row):
            if row.job["job_id"] == "target":
                path.write_text(
                    json.dumps(
                        {"kind": "telemetry", "output_tokens": 1, "unix_s": time.time()}
                    )
                    + "\n"
                )
            return row.job["job_id"]

        results = execute_schedule(jobs, run, path, time.monotonic(), time.time())
        self.assertEqual(set(results), {"source", "target"})


class BufferedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "native.jsonl"

    def test_parallel_burst_preserves_every_identity_and_batches_writes(self):
        writer = journal.AsyncJournal(self.path)

        def produce(rank):
            for index in range(5000):
                writer.submit({"job_id": rank, "token": index, "unix_s": index / 100})

        threads = [threading.Thread(target=produce, args=(rank,)) for rank in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        writer.close()
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(rows[-1]["dropped"], 0)
        self.assertEqual(rows[-1]["saved_records"], 40000)
        self.assertLess(rows[-1]["write_batches"], 4000)
        identities = {(row["job_id"], row["token"]) for row in rows[:-1]}
        self.assertEqual(len(identities), 40000)

    def test_active_journal_flushes_without_close(self):
        writer = journal.AsyncJournal(self.path)
        self.addCleanup(writer.close)
        writer.submit({"kind": "native_arrival", "unix_s": 1})
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not self.path.read_text():
            time.sleep(0.01)
        self.assertIn("native_arrival", self.path.read_text())

    def test_overflow_still_fails_with_an_explicit_loss_footer(self):
        release = threading.Event()
        original = journal.AsyncJournal._run

        def delayed(writer):
            release.wait(1)
            original(writer)

        with patch.object(journal.AsyncJournal, "_run", delayed):
            writer = journal.AsyncJournal(self.path, capacity=1)
            self.assertTrue(writer.submit({"id": 1}))
            self.assertFalse(writer.submit({"id": 2}))
            release.set()
            with self.assertRaisesRegex(RuntimeError, "evidence incomplete"):
                writer.close()
        footer = json.loads(self.path.read_text().splitlines()[-1])
        self.assertEqual(footer["dropped"], 1)
        self.assertEqual(footer["saved_records"], 1)

    def test_serialization_error_does_not_produce_success_receipt(self):
        writer = journal.AsyncJournal(self.path)
        writer.submit({"bad": float("nan")})
        with self.assertRaisesRegex(RuntimeError, "evidence incomplete"):
            writer.close()
        footer = json.loads(self.path.read_text().splitlines()[-1])
        self.assertIsNotNone(footer["writer_error"])
        self.assertEqual(footer["saved_records"], 0)

    def test_closed_journal_cannot_accept_unwritten_records(self):
        writer = journal.AsyncJournal(self.path)
        writer.close()
        writer.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            writer.submit({"id": 1})


class BackgroundDriverTests(unittest.TestCase):
    def setUp(self):
        from tools.bridge_tp import run_phase9_capacity_background

        self.driver = run_phase9_capacity_background
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, jobs):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"format_version": 1, "jobs": jobs}))
        return SimpleNamespace(
            manifest=path,
            validate_only=False,
            controller_audit_path=None,
            out_dir=self.root / "run",
            source_url="http://source",
            target_url="http://target",
            request_timeout_s=10,
        )

    def jobs(self, count):
        return [
            {
                "job_id": f"target_{i}",
                "pool": "target",
                "start_after_s": 0,
                "request": {"model": "test", "prompt": [1], "max_tokens": 2},
            }
            for i in range(count)
        ]

    def test_real_driver_dispatches_262_before_long_responses_complete(self):
        count = 262
        arguments = self.args(self.jobs(count))
        starts, lock, release = [], threading.Lock(), threading.Event()

        def post(_url, request, _timeout, record):
            with lock:
                starts.append(request["request_id"])
                if len(starts) == count:
                    release.set()
            if not release.wait(5):
                raise RuntimeError("response workers blocked due arrivals")
            stamp = time.time()
            record(0, 1, stamp)
            record(1, 2, stamp + 0.001)
            return {
                "response_id": request["request_id"],
                "token_ids": [1, 2],
                "finish_reason": "stop",
            }

        with (
            patch.object(self.driver, "parse_args", return_value=arguments),
            patch.object(self.driver, "post_streaming_completion", post),
            patch.dict("os.environ", {"BRIDGETP_TARGET_OBSERVATION": "1"}),
        ):
            self.driver.main()
        summary = json.loads(
            (arguments.out_dir / "background_summary.json").read_text()
        )
        self.assertEqual(summary["completed"], count)
        self.assertEqual(summary["failed"], 0)
        self.assertTrue(summary["background_event_integrity"]["complete"])
        for row in summary["results"]:
            self.assertLessEqual(row["dispatch_unix_s"],
                                 row["worker_started_unix_s"])
            self.assertLessEqual(row["worker_started_unix_s"],
                                 row["http_call_started_unix_s"])
            self.assertGreaterEqual(row["due_to_dispatch_s"], 0)
            self.assertGreaterEqual(row["dispatch_to_worker_s"], 0)
        self.assertTrue(summary["native_evidence_integrity"]["complete"])
        native = [
            json.loads(line)
            for line in (arguments.out_dir / "target_native_events.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(native[-1]["saved_records"], count * 2)
        self.assertEqual(native[-1]["dropped"], 0)

    def test_driver_keeps_diagnostic_summary_when_native_evidence_fails(self):
        arguments = self.args(self.jobs(1))
        from vllm.bridge_tp.async_journal import AsyncJournal

        class FailedJournal(AsyncJournal):
            def close(self):
                self.dropped = 1
                super().close()

        def post(_url, request, _timeout, record):
            record(0, 1, time.time())
            return {
                "response_id": request["request_id"],
                "token_ids": [1],
                "finish_reason": "stop",
            }

        with (
            patch.object(self.driver, "parse_args", return_value=arguments),
            patch.object(self.driver, "post_streaming_completion", post),
            patch("vllm.bridge_tp.async_journal.AsyncJournal", FailedJournal),
            patch.dict("os.environ", {"BRIDGETP_TARGET_OBSERVATION": "1"}),
            self.assertRaisesRegex(RuntimeError, "native evidence incomplete"),
        ):
            self.driver.main()
        summary = json.loads(
            (arguments.out_dir / "background_summary.json").read_text()
        )
        self.assertEqual(summary["completed"], 1)
        self.assertFalse(summary["native_evidence_integrity"]["complete"])
        self.assertEqual(summary["native_evidence_integrity"]["dropped"], 1)


if __name__ == "__main__":
    unittest.main()
