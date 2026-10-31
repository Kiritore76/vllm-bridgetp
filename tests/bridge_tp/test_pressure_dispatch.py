# SPDX-License-Identifier: Apache-2.0
"""Exercise blocked control I/O while real ingress keeps sending CPU arrivals."""

import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp.test_performance_trace import journal
from tests.bridge_tp.test_pressure_policy import candidate, config, pool
from tests.bridge_tp.test_pressure_workload import cfg as workload_config
from tools.bridge_tp import run_pressure_balanced_pilot as pilot
from tools.bridge_tp.pressure_cpu import load

d, i, o, r, io = (
    load(name)
    for name in (
        "pressure_dispatch",
        "pressure_ingress",
        "pressure_observation",
        "pressure_runtime",
        "pressure_io",
    )
)


class DispatchTests(unittest.TestCase):
    def test_blocked_control_does_not_delay_arrivals(self):
        sent = threading.Event()
        rows, calls = [], []
        cfg = config(sustain_s=0)
        co = r.PressureCoordinator(cfg, rows.append)
        with ThreadPoolExecutor(max_workers=4) as executor:

            def post(url, payload, timeout, sink):
                calls.append(payload["request_id"])
                if len(calls) == 3:
                    sent.set()
                return dict(completed_unix_s=time.time())

            ingress = i.PressureIngress(
                co,
                o.NativeServiceWindow(),
                executor,
                "source",
                "target",
                post=post,
            )
            jobs = [
                dict(
                    id=str(n),
                    at_s=n * 0.03,
                    request=dict(model="qwen", prompt=[1], max_tokens=8),
                )
                for n in range(3)
            ]
            origin, unix = time.monotonic(), time.time()

            def submit(job, pools):
                dispatched = {}
                pilot.dispatch_due(
                    [job],
                    dispatched,
                    time.monotonic() - origin,
                    ingress,
                    pools,
                    journal_sink,
                    audit,
                    origin_unix_s=unix,
                )
                return dispatched[job["id"]]

            class Sink:
                def submit(self, row):
                    pass

            class Audit:
                def write(self, row):
                    rows.append(row)

            journal_sink, audit = Sink(), Audit()
            dispatcher = d.ScheduledDispatch(
                jobs,
                submit,
                (pool(), pool("tp4")),
                origin,
                3,
            )
            try:
                dispatcher.start()
                # The test/control thread waits instead of dispatching.
                arrived = sent.wait(1)
                dispatcher.snapshot()
                self.assertTrue(arrived)
                self.assertEqual(len(calls), len(set(calls)))
            finally:
                dispatcher.close()
            streams = dispatcher.snapshot()
            self.assertEqual(len(streams), 3)
            for value in streams.values():
                value.future.result()
                self.assertLessEqual(
                    value.dispatch_entered_at, value.admission_finished_at
                )
                self.assertLessEqual(
                    value.executor_submitted_at, value.http_thread_started_at
                )

    def test_migration_start_does_not_hold_routing_lock(self):
        entered, release = threading.Event(), threading.Event()
        co = r.PressureCoordinator(config(sustain_s=0), lambda row: None)

        def start():
            entered.set()
            release.wait(3)

        errors = []

        def control():
            try:
                co.tick(
                    10,
                    pool(decode_ratio=1.5),
                    pool("tp4"),
                    [candidate()],
                    {"a": r.BoundAction("a", "migration", start)},
                )
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=control)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            acquired = co._lock.acquire(timeout=0.5)
            try:
                self.assertTrue(acquired)
                self.assertEqual(co.ledger.entries["migration"].state, "ACTIVE")
                admission = co.admission(
                    "new",
                    32,
                    64,
                    (pool(), pool("tp4")),
                    now=10,
                    key="new-admission",
                )
                self.assertIn(admission.pool_id, {"tp1", "tp4"})
            finally:
                if acquired:
                    co._lock.release()
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(errors)

    def test_critical_fsync_does_not_hold_arrival_journal_lock(self):
        entered, release, observed = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pressure_audit.jsonl"
            audit = io.PressureAudit(path, journal.AsyncJournal(path), {})
            original = audit.durable.write

            def blocked_write(row):
                entered.set()
                release.wait(3)
                original(row)

            def arrival():
                audit.write(dict(kind="pressure_admission"))
                audit.write(dict(kind="pressure_dispatch"))
                observed.set()

            with patch.object(audit.durable, "write", side_effect=blocked_write):
                control = threading.Thread(
                    target=audit.write, args=(dict(kind="pressure_decision"),)
                )
                control.start()
                producer = threading.Thread(target=arrival)
                try:
                    self.assertTrue(entered.wait(1))
                    producer.start()
                    self.assertTrue(observed.wait(1))
                finally:
                    release.set()
                    control.join(2)
                    producer.join(2)
                    audit.close()
            self.assertEqual(audit.records, 4)
            self.assertEqual(audit.critical_records, 2)

    def test_cached_reading_keeps_original_age_and_fallback(self):
        rows = []
        co = r.PressureCoordinator(workload_config(), rows.append)
        # Idle flags must not let a stale cache regain freshness.
        old = (pool(idle=True), pool("tp4", idle=True))
        co.admission("a", 32, 64, old, now=100, key="a")
        evidence = rows[-1]
        self.assertEqual(evidence["pool_readings"][0]["sampled_at"], 10)
        self.assertIn("fallback", evidence["admission"]["routing_reason"])

    def test_dispatch_failure_propagates_and_stop_prevents_later_arrivals(self):
        called = threading.Event()

        def fail(job, pools):
            called.set()
            raise ValueError("transport submission failed")

        dispatcher = d.ScheduledDispatch(
            [dict(id="a", at_s=0)],
            fail,
            (),
            time.monotonic(),
            3,
        )
        dispatcher.start()
        self.assertTrue(called.wait(1))
        dispatcher.thread.join(1)
        with self.assertRaisesRegex(RuntimeError, "dispatcher failed"):
            dispatcher.snapshot()
        dispatcher.close()
        self.assertEqual(dispatcher.snapshot(check=False), {})
        dispatcher = d.ScheduledDispatch(
            [dict(id="future", at_s=100)],
            fail,
            (),
            time.monotonic(),
            200,
        )
        dispatcher.start()
        dispatcher.close()
        self.assertFalse(dispatcher.thread.is_alive())

    def test_pilot_wiring_drains_and_saves_timing_after_control_pause(self):
        calls, all_sent, control_paused = [], threading.Event(), threading.Event()

        class Reader:
            def __init__(self, path, pool_id):
                self.pool_id, self.observed_at = pool_id, None
                self.generation = 0

            def diagnostics(self):
                return dict(error=None, deferred=None)

            def poll(self, now):
                self.observed_at = now
                self.generation += 1
                return dict(
                    schema="PRESSURE_INVENTORY_V1",
                    dropped_before=0,
                    writer_error=None,
                    pool_id=self.pool_id,
                    sampled_at=now,
                    generation=self.generation,
                    total_blocks=2001,
                    usage_denominator_blocks=2000,
                    free_blocks=2000,
                    block_size=16,
                    context_limit=8192,
                    pending_prompt_tokens=0,
                    waiting=0,
                    running=0,
                    requests=[],
                )

        session = NS(
            sessions=[],
            record=None,
            stream=None,
            target_future=None,
            settled=True,
            migration_id="test",
            ready_for_start=True,
            rotate=lambda: False,
            cleanup_requires_shutdown=False,
            cancel_future=None,
            close=lambda: None,
        )

        write_lock = threading.Lock()

        def publish(url, after=-float("inf")):
            while time.time() <= after:
                time.sleep(0.001)
            value = inventory_readers[url].poll(time.time())
            with (root / url).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value) + "\n")

        def post(url, payload, timeout, sink):
            calls.append(payload["request_id"])
            if len(calls) == 3:
                all_sent.set()
            sink(0, 42, time.time())
            completed = time.time()
            with write_lock:
                publish(url, after=completed)
            return dict(request_started_unix_s=completed, completed_unix_s=completed)

        actual_tick = r.PressureCoordinator.tick_multi

        def slow_tick(co, *args, **kwargs):
            control_paused.set()
            if not all_sent.wait(1):
                raise AssertionError("control pause delayed arrival submission")
            return actual_tick(co, *args, **kwargs)

        actual_ingress = i.PressureIngress

        def make_ingress(*args, **kwargs):
            return actual_ingress(*args, **kwargs, post=post)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory_readers = {
                "source": Reader(None, "tp1"),
                "target": Reader(None, "tp4"),
            }
            publish("source")
            publish("target")
            args = NS(
                run_dir=root,
                source_inventory=root / "source",
                target_inventory=root / "target",
                source_context_limit=8192,
                target_context_limit=8192,
                predictor_events=root / "predictor",
                predictor_sha256="0" * 64,
                migration_id="test",
                tick_s=0.005,
                max_seconds=3,
                no_progress_s=2,
                no_migration=True,
                maximum_migrations=1,
                routing_strategy="pressure",
                routing_weights=[1, 1],
                test_only_cancel_first_shadow=False,
            )
            controller = NS(
                source_url="source",
                target_url="target",
                block_size=16,
                tp1_total_kv_blocks=2001,
                tp4_total_kv_blocks=2001,
            )
            jobs = [
                dict(
                    id=str(n),
                    at_s=n * 0.03,
                    request=dict(model="qwen", prompt=[1], max_tokens=8),
                )
                for n in range(3)
            ]
            with (
                patch.dict(sys.modules, {"vllm.bridge_tp.async_journal": journal}),
                patch(
                    "tools.bridge_tp.pressure_sessions.PressureSessions",
                    return_value=session,
                ),
                patch.object(o, "scrape_inventory_metrics", return_value=None),
                patch.object(i, "PressureIngress", side_effect=make_ingress),
                patch.object(r.PressureCoordinator, "tick_multi", slow_tick),
            ):
                pilot._run(
                    args, (config(), controller, NS(release_budget=17), jobs, {})
                )
            self.assertTrue(control_paused.is_set())
            result = json.loads((root / "pressure_request_results.json").read_text())
            self.assertEqual(len(result), 3)
            self.assertEqual(len(set(calls)), 3)
            self.assertTrue(all(row["source_done"] for row in result))
            self.assertTrue(
                all(
                    row["dispatch_timing"]["http_thread_started_unix_s"] is not None
                    for row in result
                )
            )
            progress = json.loads((root / "pressure_progress.json").read_text())
            self.assertEqual(progress["stage"], "COMPLETED")

    def test_routing_only_instance_serves_routed_requests(self):
        """S1-d1a: an extra TP1 outside the migration pair gets its own
        inventory, scrape and URL, and receives routed requests."""
        calls = []

        class Reader:
            def __init__(self, pool_id):
                self.pool_id, self.generation = pool_id, 0

            def poll(self, now):
                self.generation += 1
                return dict(
                    schema="PRESSURE_INVENTORY_V1",
                    dropped_before=0,
                    writer_error=None,
                    pool_id=self.pool_id,
                    sampled_at=now,
                    generation=self.generation,
                    total_blocks=2001,
                    usage_denominator_blocks=2000,
                    free_blocks=2000,
                    block_size=16,
                    context_limit=8192,
                    pending_prompt_tokens=0,
                    waiting=0,
                    running=0,
                    requests=[],
                )

        session = NS(
            sessions=[],
            record=None,
            stream=None,
            target_future=None,
            settled=True,
            migration_id="test",
            ready_for_start=True,
            rotate=lambda: False,
            cleanup_requires_shutdown=False,
            cancel_future=None,
            close=lambda: None,
        )
        write_lock = threading.Lock()

        def publish(url, after=-float("inf")):
            while time.time() <= after:
                time.sleep(0.001)
            value = readers[url].poll(time.time())
            with (root / url).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value) + "\n")

        def post(url, payload, timeout, sink):
            calls.append(url)
            sink(0, 42, time.time())
            completed = time.time()
            with write_lock:
                publish(url, after=completed)
            return dict(request_started_unix_s=completed, completed_unix_s=completed)

        actual_ingress = i.PressureIngress

        def make_ingress(*args, **kwargs):
            return actual_ingress(*args, **kwargs, post=post)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            readers = {
                "source": Reader("tp1"),
                "target": Reader("tp4"),
                "extra": Reader("A-g3-tp1"),
            }
            for url in readers:
                publish(url)
            row = dict(tp_degree=1, context_limit=8192, physical_gpu_ids=["3"])
            manifest = dict(
                schema="BRIDGETP_INSTANCE_MANIFEST_V1",
                legacy_pool_aliases={"tp1": "A-g1-tp1", "tp4": "B-g2-tp2"},
                instances=[
                    dict(row, instance_id="A-g1-tp1", endpoint="source"),
                    dict(
                        row,
                        instance_id="B-g2-tp2",
                        endpoint="target",
                        tp_degree=2,
                    ),
                    dict(
                        row,
                        instance_id="A-g3-tp1",
                        endpoint="extra",
                        inventory_path=str(root / "extra"),
                    ),
                ],
            )
            args = NS(
                run_dir=root,
                source_inventory=root / "source",
                target_inventory=root / "target",
                source_context_limit=8192,
                target_context_limit=8192,
                predictor_events=root / "predictor",
                predictor_sha256="0" * 64,
                migration_id="test",
                tick_s=0.005,
                max_seconds=3,
                no_progress_s=2,
                no_migration=True,
                maximum_migrations=1,
                routing_strategy="pressure",
                routing_weights=[1, 1],
                test_only_cancel_first_shadow=False,
                instances=root / "instances.json",
            )
            (root / "instances.json").write_text(json.dumps(manifest))
            controller = NS(
                source_url="source",
                target_url="target",
                block_size=16,
                tp1_total_kv_blocks=2001,
                tp4_total_kv_blocks=2001,
                target_tp_size=2,
            )
            checked = pilot.instance_manifest(args, controller)
            jobs = [
                dict(
                    id=str(n),
                    at_s=n * 0.03,
                    request=dict(model="qwen", prompt=[1], max_tokens=8),
                )
                for n in range(3)
            ]
            with (
                patch.dict(sys.modules, {"vllm.bridge_tp.async_journal": journal}),
                patch(
                    "tools.bridge_tp.pressure_sessions.PressureSessions",
                    return_value=session,
                ),
                patch.object(o, "scrape_inventory_metrics", return_value=None),
                patch.object(i, "PressureIngress", side_effect=make_ingress),
            ):
                pilot._run(
                    args,
                    (
                        config(),
                        controller,
                        NS(release_budget=17),
                        jobs,
                        {"instance_manifest": checked},
                    ),
                )
            result = json.loads((root / "pressure_request_results.json").read_text())
            self.assertEqual(len(result), 3)
            self.assertTrue(all(row["source_done"] for row in result))
            self.assertIn("A-g3-tp1", {row["initial_pool"] for row in result})
            self.assertIn("extra", calls)
            self.assertTrue((root / "inventory_A-g3-tp1_process_receipt.json").exists())
            progress = json.loads((root / "pressure_progress.json").read_text())
            self.assertEqual(progress["stage"], "COMPLETED")
            bad = dict(manifest)
            bad["instances"] = manifest["instances"][:2] + [
                dict(manifest["instances"][2], inventory_path="")
            ]
            (root / "instances.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                pilot.instance_manifest(args, controller)


if __name__ == "__main__":
    unittest.main()
