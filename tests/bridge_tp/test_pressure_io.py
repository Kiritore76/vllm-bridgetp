# SPDX-License-Identifier: Apache-2.0
"""Async observations must not weaken audit durability or freshness checks."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.bridge_tp.test_performance_trace import journal
from tools.bridge_tp.pressure_cpu import load

m = load("pressure_io")
ROOT = Path(__file__).resolve().parents[2]


class PressureIOTests(unittest.TestCase):
    def test_observations_async_critical_events_durable_and_payload_frozen(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "pressure_audit.jsonl"
            writer = journal.AsyncJournal(path, flush_interval_s=1)
            with patch.object(load("audit").os, "fsync") as fsync:
                audit = m.PressureAudit(path, writer, {})
                mutable = dict(kind="pressure_metrics_sample", value=dict(count=1))
                audit.write(mutable)
                mutable["value"]["count"] = 99
                audit.write(dict(kind="pressure_decision"))
                audit.write(
                    dict(kind="pressure_ingress_reconcile", state_changed=False)
                )
                audit.write(dict(kind="pressure_ingress_reconcile", state_changed=True))
                audit.write(dict(kind="unknown_event_defaults_durable"))
                audit.close()
                self.assertEqual(fsync.call_count, 4)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[1]["value"]["count"], 1)
            self.assertEqual(rows[-1]["dropped"], 0)
            critical = [
                json.loads(line)
                for line in path.with_name("pressure_audit_durable.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(critical), 4)
            self.assertEqual(
                json.loads(path.with_name("pressure_io_receipt.json").read_text())[
                    "records"
                ],
                6,
            )

    def test_observation_failure_does_not_erase_durable_critical_event(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "pressure_audit.jsonl"
            writer = Mock()
            writer.submit.return_value = False
            writer.close.side_effect = RuntimeError("overflow")
            audit = m.PressureAudit(path, writer, {})
            audit.write(dict(kind="pressure_source_release_confirmed"))
            with self.assertRaisesRegex(RuntimeError, "overflow"):
                audit.close()
            text = path.with_name("pressure_audit_durable.jsonl").read_text()
            self.assertIn("pressure_source_release_confirmed", text)
            self.assertFalse(path.with_name("pressure_io_receipt.json").exists())

    def test_slow_http_never_blocks_snapshot_and_worker_stops(self):
        entered, release = threading.Event(), threading.Event()

        def scrape(inventory):
            entered.set()
            release.wait(2)
            return dict(sampled_unix_s=time.time())

        sampler = m.BackgroundMetrics(scrape, interval_s=0.01)
        try:
            sampler.update_inventory(
                dict(total_blocks=101, usage_denominator_blocks=100, free_blocks=80)
            )
            self.assertTrue(entered.wait(1))
            with self.assertRaises(m.TelemetryError):
                sampler.snapshot(time.time(), 2)
        finally:
            release.set()
            sampler.close()
        self.assertFalse(sampler.thread.is_alive())

    def test_cached_snapshot_never_refreshes_time_and_failures_invalidate(self):
        with patch.object(m.threading.Thread, "start"):
            sampler = m.BackgroundMetrics(Mock())
        sampler.value = dict(sampled_unix_s=100, num_running=0)
        self.assertEqual(sampler.snapshot(101, 2)["sampled_unix_s"], 100)
        self.assertEqual(sampler.snapshot(101.5, 2)["sampled_unix_s"], 100)
        for now in (99, 103):
            with self.assertRaises(m.TelemetryError):
                sampler.snapshot(now, 2)
        with self.assertRaises(ValueError):
            sampler.update_inventory(
                dict(total_blocks=101, usage_denominator_blocks=101, free_blocks=80)
            )

        def fail(inventory):
            sampler.stop.set()
            raise OSError("HTTP failed")

        sampler.scrape = fail
        sampler.update_inventory(
            dict(total_blocks=101, usage_denominator_blocks=100, free_blocks=80)
        )
        sampler._run()
        with self.assertRaisesRegex(m.TelemetryError, "HTTP failed"):
            sampler.snapshot(101, 2)

    def test_delayed_http_response_uses_request_start_for_freshness(self):
        clock = [100.0]

        def scrape(inventory):
            clock[0] = 105.0
            sampler.stop.set()
            return dict(sampled_unix_s=105.0)

        with patch.object(m.threading.Thread, "start"):
            sampler = m.BackgroundMetrics(scrape)
        sampler.update_inventory(
            dict(total_blocks=101, usage_denominator_blocks=100, free_blocks=80)
        )
        with patch.object(m.time, "time", side_effect=lambda: clock[0]):
            sampler._run()
        self.assertEqual(sampler.value["sampled_unix_s"], 100)
        with self.assertRaises(m.TelemetryError):
            sampler.snapshot(105, 2)


if __name__ == "__main__":
    unittest.main()
