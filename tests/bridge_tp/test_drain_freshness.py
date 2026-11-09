# SPDX-License-Identifier: Apache-2.0
"""Growing journals and slow collectors must not strand an empty conversion."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from tools.bridge_tp import pressure_conversion as pc
from tools.bridge_tp import run_pressure_calibration as calibration


class JournalTests(unittest.TestCase):
    def test_large_history_reads_only_the_tail(self):
        row = dict(schema="PRESSURE_INVENTORY_V1", generation=50)
        data = b"old history\n" * 400000 + json.dumps(row).encode() + b"\n"
        stream = io.BytesIO(data)
        source = Mock(wraps=stream)
        source.__enter__ = Mock(return_value=source)
        source.__exit__ = Mock()
        path = Mock()
        path.open.return_value = source
        self.assertEqual(calibration.latest_inventory(path), row)
        self.assertEqual(source.read.call_args_list, [unittest.mock.call(65536)])

    def test_long_row_partial_append_and_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inventory.jsonl"
            self.assertIsNone(calibration.latest_inventory(path))
            path.write_bytes(b'{"partial":')
            self.assertIsNone(calibration.latest_inventory(path))
            row = dict(schema="PRESSURE_INVENTORY_V1", requests=["x" * 150000])
            path.write_bytes(json.dumps(row).encode() + b'\n{"partial":')
            self.assertEqual(calibration.latest_inventory(path), row)
            for key in ("writer_error", "inventory_error", "dropped", "dropped_before"):
                with self.subTest(key=key):
                    path.write_text(json.dumps(dict(row, **{key: 1})) + "\n")
                    with self.assertRaises(RuntimeError):
                        calibration.latest_inventory(path)
            path.write_text('{"schema":"unknown"}\n')
            self.assertIsNone(calibration.latest_inventory(path))
            path.write_text("{broken}\n")
            with self.assertRaises(json.JSONDecodeError):
                calibration.latest_inventory(path)


class DrainTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.now = 100.0
        self.backend = b = object.__new__(pc.ControllerConversionBackend)
        b.run_dir = Path(self.directory.name)
        b.clock = lambda: self.now
        b.revision = 4
        self.plan = dict(
            plan_id="drain", refs=[dict(instance_id="A-g6-tp2", instance_generation=6)]
        )
        b.rows = {"A-g6-tp2": dict(inventory_path="unused", endpoint="unused")}
        b.status = self.status
        b.shadow = Mock(side_effect=self.slow_shadow)
        b.ops = NS(scrape=self.scrape, event=Mock())
        self.native = dict(
            sampled_at=50, requests=[], running=0, waiting=0, free_blocks=100
        )

    def status(self):
        return dict(
            unix_s=self.now,
            layout_revision=4,
            instances={
                "A-g6-tp2": dict(
                    lifecycle="QUIESCING",
                    open_admissions=0,
                    streams_pending=0,
                    sessions=0,
                )
            },
        )

    def slow_shadow(self, *_):
        self.now += 3

    def scrape(self, *_):
        self.now += 0.1
        return dict(
            num_running=0,
            num_waiting=0,
            free_kv_blocks=100,
            sampled_unix_s=self.now,
        )

    def collect(self):
        with patch.object(pc, "read_latest", return_value=self.native):
            return self.backend.drain_evidence(self.plan)

    def drained(self, rows):
        return bool(rows) and rows[0].drained(self.backend.refs(self.plan)[0], self.now)

    def test_slow_shadow_then_fresh_idle_proof_advances_without_retimestamp(self):
        self.assertTrue(self.drained(self.collect()))
        self.assertEqual(self.native["sampled_at"], 50)
        self.assertTrue(self.backend.fence(self.plan))
        log = json.loads(
            (self.backend.run_dir / "conversion_drain_observations.jsonl").read_text()
        )
        self.assertEqual(log["controller_sampled_at"], 103.1)
        self.assertEqual(log["instances"]["A-g6-tp2"]["native_sampled_at"], 50)

    def test_slow_native_read_does_not_pin_proof_to_pre_io_controller(self):
        def slow_read(_):
            self.now += 3
            return self.native

        with patch.object(pc, "read_latest", side_effect=slow_read):
            rows = self.backend.drain_evidence(self.plan)
        self.assertTrue(self.drained(rows))

    def test_slow_second_source_cannot_refresh_the_first_http_sample(self):
        self.plan["refs"].append(
            dict(
                instance_id="A-g7-tp2",
                instance_generation=7,
            )
        )
        self.backend.rows["A-g7-tp2"] = self.backend.rows["A-g6-tp2"]

        def status():
            value = self.status()
            value["instances"]["A-g7-tp2"] = value["instances"]["A-g6-tp2"]
            return value

        calls = 0

        def scrape(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.now += 3
            return self.scrape(*args)

        self.backend.status = status
        self.backend.ops.scrape = scrape
        self.assertFalse(self.drained(self.collect()))

    def test_stale_controller_or_old_revision_cannot_drain_or_fence(self):
        for change in (
            dict(unix_s=90),
            dict(unix_s=200),
            dict(layout_revision=3),
        ):
            with self.subTest(change=change):
                self.backend.status = lambda change=change: dict(
                    self.status(), **change
                )
                self.assertFalse(self.drained(self.collect()))
                self.assertFalse(self.backend.fence(self.plan))

    def test_new_activity_or_reopened_instance_blocks_fence(self):
        self.assertTrue(self.drained(self.collect()))
        self.backend.shadow = Mock()
        for key, value in (
            ("lifecycle", "ACTIVE"),
            ("open_admissions", 1),
            ("streams_pending", 1),
            ("sessions", 1),
        ):
            with self.subTest(key=key):
                state = self.status()
                state["instances"]["A-g6-tp2"][key] = value
                self.backend.status = lambda state=state: state
                self.assertFalse(self.drained(self.collect()))
                self.assertFalse(self.backend.fence(self.plan))
        self.backend.ops.event.assert_not_called()

    def test_missing_instance_or_native_does_not_become_idle(self):
        state = self.status()
        state["instances"] = {"A-g7-tp2": state["instances"]["A-g6-tp2"]}
        self.backend.status = lambda: state
        self.assertFalse(self.drained(self.collect()))
        self.assertFalse(self.backend.fence(self.plan))
        self.backend.status = self.status
        self.native = None
        self.assertFalse(self.drained(self.collect()))

    def test_stale_or_missing_http_cannot_refresh_old_native(self):
        for observed in (
            {},
            dict(
                num_running=0,
                num_waiting=0,
                free_kv_blocks=100,
                sampled_unix_s=90,
            ),
        ):
            self.backend.ops.scrape = lambda *_, observed=observed: observed
            self.assertFalse(self.drained(self.collect()))

    def test_future_native_or_http_is_not_hidden_by_minimum_timestamp(self):
        self.native["sampled_at"] = 200
        self.assertFalse(self.drained(self.collect()))
        self.native["sampled_at"] = 50
        self.backend.ops.scrape = lambda *_: dict(
            num_running=0,
            num_waiting=0,
            free_kv_blocks=100,
            sampled_unix_s=200,
        )
        self.assertFalse(self.drained(self.collect()))

    def test_nonempty_native_and_mismatched_free_blocks_do_not_refresh(self):
        for change in (dict(requests=["live"]), dict(running=1), dict(free_blocks=99)):
            with self.subTest(change=change):
                original = self.native
                self.native = dict(original, **change)
                self.assertFalse(self.drained(self.collect()))
                self.native = original

    def test_retirement_ack_must_be_fresh_and_apply_current_revision(self):
        for change in (dict(unix_s=90), dict(layout_revision=3)):
            self.backend.status = lambda change=change: dict(self.status(), **change)
            self.assertIsNone(self.backend.controller_states(["A-g6-tp2"]))


if __name__ == "__main__":
    unittest.main()
