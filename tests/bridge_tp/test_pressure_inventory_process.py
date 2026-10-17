# SPDX-License-Identifier: Apache-2.0
"""Real CPU subprocess isolation, journal integrity, IPC age and owner cleanup."""

import gc
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

from tests.bridge_tp.test_pressure_observation import inv, scheduler
from tests.bridge_tp.test_pressure_policy import config, pool
from tools.bridge_tp.pressure_cpu import load

m, d = (load(name) for name in ("pressure_inventory_process", "pause_diagnostics"))


def sample(generation=1, now=None):
    raw = inv.inventory_snapshot(
        scheduler(), "tp1", time.time() if now is None else now
    )
    return dict(raw, generation=generation, dropped_before=0, writer_error=None)


def append(path, raw):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(raw) + "\n")


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    raise AssertionError("CPU inventory subprocess condition timed out")


class InventoryProcessTests(unittest.TestCase):
    def test_compact_snapshot_preserves_workload_and_prefill_geometry(self):
        from tests.bridge_tp.test_pressure_workload import profiles
        from tools.bridge_tp.pressure_health_reference import unallocated_prompt_blocks

        raw = sample(now=10)
        raw["last_scheduled_batch"] = dict(
            sampled_at=10,
            decode_requests=1,
            decode_context_sum=2048,
            prefill_tokens=0,
            recompute_tokens=0,
            requests=[
                dict(
                    request_id="a",
                    computed_before=2048,
                    decode_tokens=1,
                    prefill_tokens=0,
                    recompute_tokens=0,
                    ignored_history=[1] * 100,
                )
            ],
        )
        compact = m.compact_inventory(raw)
        workload = load("pressure_workload")
        self.assertEqual(
            workload.observe(profiles()["tp1"], raw, 10),
            workload.observe(profiles()["tp1"], compact, 10),
        )
        self.assertEqual(
            unallocated_prompt_blocks(raw["requests"], 16),
            unallocated_prompt_blocks(compact["requests"], 16),
        )
        self.assertNotIn(
            "ignored_history", compact["last_scheduled_batch"]["requests"][0]
        )

    def test_early_timer_wakeup_rechecks_the_absolute_deadline(self):
        from unittest.mock import patch

        dispatch = load("pressure_dispatch")
        clock, sends = [100.0], []
        scheduler = dispatch.ScheduledDispatch(
            [dict(id="a", at_s=0.1)],
            lambda job, pools: sends.append(clock[0]),
            (),
            100.0,
            3,
        )
        scheduler.stop = Mock()
        scheduler.stop.is_set.return_value = False
        waits = []

        def wait(delay):
            waits.append(delay)
            clock[0] += delay / 2 if len(waits) == 1 else delay
            return False

        scheduler.stop.wait.side_effect = wait
        with patch.object(dispatch.time, "monotonic", side_effect=lambda: clock[0]):
            scheduler._run()
        self.assertEqual(sends, [100.1])
        self.assertEqual(len(waits), 2)

    def test_actual_worker_compacts_without_refreshing_or_loading_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = sample()
            raw["unused_token_history"] = list(range(20000))
            path = root / "inventory.jsonl"
            append(path, raw)
            reader = m.ProcessInventory(path, "tp1", root)
            try:
                value = wait_for(lambda: reader.poll(time.time()))
                self.assertEqual(value["sampled_at"], raw["sampled_at"])
                self.assertNotIn("unused_token_history", value)
                self.assertNotIn("blocks", value["requests"][0])
                self.assertEqual(value["requests"][0]["releasable_blocks"], 1)
                stamp = reader.observed_at
                self.assertEqual(
                    reader.poll(time.time() + 60)["sampled_at"], raw["sampled_at"]
                )
                self.assertEqual(reader.observed_at, stamp)
                self.assertNotEqual(reader.worker_pid, os.getpid())
            finally:
                reader.close()
            receipt = json.loads(
                (root / "inventory_tp1_process_receipt.json").read_text()
            )
            self.assertEqual(receipt["status"], "COMPLETED")
            self.assertFalse(receipt["forced"])
            self.assertTrue(receipt["receiver_stopped"])
            self.assertEqual(reader.process.poll(), 0)
            self.assertLess(receipt["max_message_bytes"], 4096)
            records = [
                json.loads(line) for line in reader.timing_path.read_text().splitlines()
            ]
            identity = next(
                r for r in records if r["kind"] == "inventory_worker_identity"
            )
            self.assertFalse(identity["cuda_package_imported"])
            names = set().union(
                *(r["operations"] for r in records if r["kind"] == "timing_summary")
            )
            self.assertTrue(
                {"inventory.poll", "inventory.read_record", "inventory.parse_record"}
                <= names
            )

    def test_partial_records_and_future_evidence_keep_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "inventory.jsonl"
            append(path, sample())
            future = sample(2, time.time() + 60)
            append(path, future)
            append(path, sample(3))
            reader = m.ProcessInventory(path, "tp1", root)
            try:
                value = wait_for(lambda: reader.poll(time.time()))
                self.assertEqual(value["generation"], 1)
                self.assertEqual(reader.deferred["generation"], 2)
            finally:
                reader.close()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "inventory.jsonl"
            append(path, sample())
            reader = m.ProcessInventory(path, "tp1", root)
            try:
                wait_for(lambda reader=reader: reader.poll(time.time()))
                payload = json.dumps(sample(2)) + "\n"
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(payload[: len(payload) // 2])
                self.assertEqual(reader.poll(time.time())["generation"], 1)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(payload[len(payload) // 2 :])
                wait_for(lambda: reader.poll(time.time())["generation"] == 2)
            finally:
                reader.close()

    def test_bad_record_cannot_be_overwritten_by_a_later_valid_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "inventory.jsonl"
            append(path, sample())
            reader = m.ProcessInventory(path, "tp1", root)
            wait_for(lambda: reader.poll(time.time()))
            append(path, dict(sample(2), dropped_before=1))
            append(path, sample(3))
            wait_for(lambda: reader.failure)
            with self.assertRaisesRegex(RuntimeError, "subprocess failed"):
                reader.poll(time.time())
            with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
                reader.close()
            self.assertFalse(reader.thread.is_alive())
            self.assertIsNotNone(reader.process.poll())
            receipt = json.loads(
                (root / "inventory_tp1_process_receipt.json").read_text()
            )
            self.assertEqual(receipt["status"], "FAILED")

    def test_truncation_and_unexpected_exit_fail_and_reap_owned_worker(self):
        for truncate in (True, False):
            with (
                self.subTest(truncate=truncate),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                path = root / "inventory.jsonl"
                append(path, sample())
                reader = m.ProcessInventory(path, "tp1", root)
                wait_for(lambda reader=reader: reader.poll(time.time()))
                if truncate:
                    path.write_text("")
                    wait_for(lambda reader=reader: reader.failure)
                else:
                    reader.process.terminate()
                    reader.process.wait(timeout=3)
                with self.assertRaisesRegex(RuntimeError, "subprocess failed"):
                    reader.poll(time.time())
                with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
                    reader.close()
                self.assertFalse(reader.thread.is_alive())
                self.assertIsNotNone(reader.process.poll())

    def test_worker_holding_its_gil_does_not_block_parent_ingress_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, busy, finished = (
                root / "inventory.jsonl",
                root / "busy",
                root / "finished",
            )
            append(path, sample())
            script = """
import sys, time
from pathlib import Path
from tools.bridge_tp.pressure_cpu import load
m=load('pressure_inventory_process')
base=m.InventoryReader
class BusyReader(base):
    paused=False
    def poll(self, now):
        if not self.paused:
            self.paused=True
            Path(sys.argv[2]).write_text('busy')
            sys.setswitchinterval(10)
            end=time.monotonic()+0.8
            while time.monotonic()<end:
                pass
            sys.setswitchinterval(0.005)
            Path(sys.argv[3]).write_text('finished')
        return super().poll(time.time())
m.InventoryReader=BusyReader
m.worker(Path(sys.argv[1]),'tp1',Path(sys.argv[4]))
"""
            reader = m.ProcessInventory(
                path,
                "tp1",
                root,
                command=[
                    sys.executable,
                    "-B",
                    "-c",
                    script,
                    str(path),
                    str(busy),
                    str(finished),
                    str(root / "inventory_tp1_timing.jsonl"),
                ],
            )
            try:
                wait_for(busy.exists)
                ingress = load("pressure_ingress")
                co = load("pressure_runtime").PressureCoordinator(
                    config(), lambda row: None
                )
                window = load("pressure_observation").NativeServiceWindow()
                sent = threading.Event()
                tokens = []

                def post(url, payload, timeout, sink):
                    sink(0, 42, time.time())
                    tokens.append(42)
                    if len(tokens) == 3:
                        sent.set()
                    return dict(completed_unix_s=time.time())

                with ThreadPoolExecutor(max_workers=4) as executor:
                    router = ingress.PressureIngress(
                        co, window, executor, "source", "target", post=post
                    )
                    scheduler = load("pressure_dispatch").ScheduledDispatch(
                        [dict(id=str(n), at_s=n * 0.02) for n in range(3)],
                        lambda job, pools: router.submit(
                            dict(prompt=[1], max_tokens=8), pools
                        ),
                        (pool(), pool("tp4")),
                        time.monotonic(),
                        3,
                    )
                    try:
                        scheduler.start()
                        self.assertTrue(sent.wait(0.5))
                        self.assertFalse(finished.exists())
                        self.assertIsNone(reader.poll(time.time()))
                    finally:
                        scheduler.close()
                    self.assertEqual(len(scheduler.snapshot()), 3)
            finally:
                reader.close()

    def test_ipc_identity_sequence_and_size_are_checked(self):
        reader = m.ProcessInventory.__new__(m.ProcessInventory)
        reader.process = Mock(pid=os.getpid())
        reader.pool_id, reader.sequence, reader.worker_pid = "tp1", 0, os.getpid()
        reader.max_message_bytes = 0
        for mismatch in (dict(seq=2), dict(pool_id="tp4"), dict(pid=1)):
            with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                reader._accept(
                    dict(schema=m.SCHEMA, pool_id="tp1", pid=os.getpid(), seq=1)
                    | mismatch,
                    1,
                )
        reader.failure = None
        reader.lock = threading.Lock()
        reader.process.stdout = io.BytesIO(b"x" * (m.MAX_MESSAGE_BYTES + 1))
        reader._receive()
        self.assertIn("oversized", reader.failure)

    def test_gc_callback_records_outside_collection_and_restores_callbacks(self):
        observer = Mock()
        old = list(gc.callbacks)
        with d.GCMonitor(observer) as monitor:
            gc.collect(2)
            observer.record.assert_not_called()
            monitor.flush()
            self.assertTrue(
                any(
                    c.args[0] == "gc.generation_2"
                    for c in observer.record.call_args_list
                )
            )
        self.assertEqual(gc.callbacks, old)


if __name__ == "__main__":
    unittest.main()
