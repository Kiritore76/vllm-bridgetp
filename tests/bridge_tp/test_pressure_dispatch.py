# SPDX-License-Identifier: Apache-2.0
"""Exercise blocked control I/O while real ingress keeps sending CPU arrivals."""

import json
import os
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


h = load("pressure_hold")


def hold_job(name, prompt=1, output=8, at_s=0.0):
    return dict(
        id=name, at_s=at_s, request=dict(prompt=[1] * prompt, max_tokens=output)
    )


def stream(pool_id, tokens=0, done=False, prompt=1, output=8):
    return NS(
        pool_id=pool_id,
        recorder=NS(tokens_seen=tokens, target_arrived=False, target_pool="tp4"),
        future=NS(done=lambda: done),
        payload=dict(prompt=[1] * prompt, max_tokens=output),
    )


def hold_pool(name, free=2000, unallocated=0, sampled_at=100.0):
    return NS(
        pool_id=name,
        free_blocks=free,
        unallocated_prompt_blocks=unallocated,
        block_size=16,
        sampled_at=sampled_at,
    )


class HoldQueueTests(unittest.TestCase):
    def queue(self, limit=2, wait=60, active=lambda _: True):
        return h.HoldQueue(limit, wait, 8192, 2.0, active)

    def names(self, released):
        return released[0]["id"], [p.pool_id for p in released[1]]

    def test_short_requests_leave_in_order_and_only_while_a_pool_has_room(self):
        queue = self.queue(limit=1)
        pools = [hold_pool("tp1"), hold_pool("tp4")]
        for n in "abc":
            queue.add(hold_job(n), 0.0)
        first = queue.next(pools, 1.0, 100.0)
        self.assertEqual(self.names(first), ("a", ["tp1", "tp4"]))
        self.assertEqual(first[0]["held_s"], 1.0)
        waiting = stream("tp1")
        queue.sent(first[0], waiting)
        self.assertEqual(self.names(queue.next(pools, 1.0, 100.0)), ("b", ["tp4"]))
        queue.sent(hold_job("b"), stream("tp4"))
        self.assertIsNone(queue.next(pools, 1.0, 100.0))
        # The first token, not completion, frees the place in line.
        waiting.recorder.tokens_seen = 1
        self.assertEqual(self.names(queue.next(pools, 1.0, 100.0)), ("c", ["tp1"]))
        self.assertEqual(queue.waiting(), 0)

    def test_stale_telemetry_does_not_stop_short_requests(self):
        queue = self.queue()
        queue.add(hold_job("a"), 0.0)
        pools = [hold_pool("tp1", free=None, unallocated=None, sampled_at=1.0)]
        self.assertEqual(self.names(queue.next(pools, 1.0, 100.0))[0], "a")

    def test_a_long_request_without_room_waits_alone(self):
        queue = self.queue()
        queue.add(hold_job("long", 16384, 8192), 0.0)
        queue.add(hold_job("short"), 0.5)
        tight = [hold_pool("tp1", free=1500), hold_pool("tp4", unallocated=600)]
        self.assertEqual(self.names(queue.next(tight, 1.0, 100.0))[0], "short")
        self.assertIsNone(queue.next(tight, 1.0, 100.0))
        self.assertEqual(queue.waiting(), 1)
        roomy = [hold_pool("tp1"), hold_pool("tp4", free=1000)]
        self.assertEqual(self.names(queue.next(roomy, 1.0, 100.0)), ("long", ["tp1"]))

    def test_a_long_request_needs_fresh_evidence_until_its_limit(self):
        queue = self.queue(wait=60)
        queue.add(hold_job("long", 16384, 8192), 0.0)
        queue.add(hold_job("short"), 0.0)
        stale = [hold_pool("tp1", sampled_at=90.0)]
        self.assertEqual(self.names(queue.next(stale, 59.0, 100.0))[0], "short")
        queue.sent(hold_job("short"), stream("tp1"))
        queue.sent(hold_job("other"), stream("tp1"))
        self.assertIsNone(queue.next(stale, 59.0, 100.0))
        # Past the limit it goes first, even with no room or evidence.
        self.assertEqual(self.names(queue.next(stale, 60.0, 100.0))[0], "long")

    def test_a_measured_full_pool_takes_no_short_request(self):
        queue = self.queue()
        queue.add(hold_job("a"), 0.0)
        full = hold_pool("tp4", free=20, unallocated=20)
        pools = [hold_pool("tp1"), full]
        self.assertEqual(self.names(queue.next(pools, 1.0, 100.0)), ("a", ["tp1"]))
        queue.add(hold_job("b"), 0.0)
        self.assertIsNone(queue.next([full], 1.0, 100.0))
        full.free_blocks = 21
        self.assertEqual(self.names(queue.next([full], 1.0, 100.0))[0], "b")

    def test_a_short_request_ignores_the_growth_of_long_requests(self):
        queue = self.queue(limit=8)
        running = stream("tp4", tokens=800, prompt=16384, output=8192)
        queue.sent(hold_job("old", 16384, 8192), running)
        queue.add(hold_job("a"), 0.0)
        self.assertEqual(
            self.names(queue.next([hold_pool("tp4", free=10)], 1.0, 100.0))[0], "a"
        )

    def test_a_long_request_past_its_limit_waits_while_kv_is_measured_short(self):
        queue = self.queue(wait=60)
        queue.add(hold_job("long", 16384, 8192), 0.0)
        tight = [hold_pool("tp1", free=1500), hold_pool("tp4", free=1500)]
        self.assertIsNone(queue.next(tight, 500.0, 100.0))
        self.assertEqual(queue.waiting(), 1)
        tight[1].free_blocks = 1536
        self.assertEqual(self.names(queue.next(tight, 500.0, 100.0)), ("long", ["tp4"]))

    def test_past_its_limit_a_long_request_only_passes_unmeasured_pools(self):
        queue = self.queue(wait=60)
        queue.add(hold_job("long", 16384, 8192), 0.0)
        pools = [hold_pool("tp1", free=1500), hold_pool("tp4", sampled_at=90.0)]
        self.assertIsNone(queue.next(pools, 59.0, 100.0))
        self.assertEqual(self.names(queue.next(pools, 60.0, 100.0)), ("long", ["tp4"]))

    def test_past_its_limit_a_long_request_may_exceed_the_unstarted_limit(self):
        queue = self.queue(limit=1, wait=60)
        queue.sent(hold_job("a"), stream("tp1"))
        queue.add(hold_job("long", 16384, 8192), 0.0)
        pools = [hold_pool("tp1")]
        self.assertIsNone(queue.next(pools, 59.0, 100.0))
        self.assertEqual(self.names(queue.next(pools, 60.0, 100.0))[0], "long")

    def test_long_requests_already_sent_keep_their_growth_reserved(self):
        queue = self.queue(limit=8)
        running = stream("tp4", tokens=800, prompt=16384, output=8192)
        queue.sent(hold_job("old", 16384, 8192), running)
        queue.add(hold_job("new", 16384, 8192), 0.0)
        self.assertIsNone(queue.next([hold_pool("tp4", free=1800)], 1.0, 100.0))
        self.assertEqual(
            self.names(queue.next([hold_pool("tp4", free=2000)], 1.0, 100.0))[0],
            "new",
        )

    def test_a_migrated_long_request_is_charged_to_its_target(self):
        queue = self.queue(limit=8)
        moved = stream("tp1", tokens=800, prompt=16384, output=8192)
        moved.recorder.target_arrived = True
        queue.sent(hold_job("old", 16384, 8192), moved)
        queue.add(hold_job("new", 16384, 8192), 0.0)
        self.assertEqual(
            self.names(
                queue.next(
                    [hold_pool("tp1", free=1600), hold_pool("tp4", free=1600)],
                    1.0,
                    100.0,
                )
            ),
            ("new", ["tp1"]),
        )

    def test_retired_instances_are_not_offered(self):
        queue = self.queue(active=lambda pool_id: pool_id != "tp1")
        queue.add(hold_job("a"), 0.0)
        pools = [hold_pool("tp1"), hold_pool("tp4")]
        self.assertEqual(self.names(queue.next(pools, 1.0, 100.0)), ("a", ["tp4"]))

    def test_dispatcher_sends_held_requests_in_order_as_engines_start(self):
        sent, open_streams = [], []

        def submit(job, pools):
            value = stream(pools[0].pool_id)
            sent.append((job["id"], job.get("held_s")))
            open_streams.append(value)
            return value

        jobs = [hold_job(str(n), at_s=0.0) for n in range(3)]
        now = time.time()
        dispatcher = d.ScheduledDispatch(
            jobs,
            submit,
            [hold_pool("tp1", sampled_at=now)],
            time.monotonic(),
            5.0,
            hold=self.queue(limit=1),
        )
        dispatcher.start()
        time.sleep(0.2)
        self.assertEqual([name for name, _ in sent], ["0"])
        self.assertEqual(sorted(dispatcher.snapshot()), ["0"])
        for _ in range(2):
            open_streams[-1].recorder.tokens_seen = 1
            time.sleep(0.2)
        open_streams[-1].recorder.tokens_seen = 1
        dispatcher.thread.join(2)
        self.assertFalse(dispatcher.thread.is_alive())
        self.assertEqual([name for name, _ in sent], ["0", "1", "2"])
        self.assertGreater(sent[-1][1], sent[0][1])

    def test_fixed_routes_bypass_the_hold(self):
        sent = []
        job = dict(hold_job("fixed"), route="tp4")

        def submit(job, pools):
            sent.append(job["id"])
            return stream("tp4")

        hold = self.queue(limit=1, active=lambda _: False)
        dispatcher = d.ScheduledDispatch(
            [job], submit, [], time.monotonic(), 5.0, hold=hold
        )
        dispatcher.start()
        dispatcher.thread.join(2)
        self.assertEqual(sent, ["fixed"])
        self.assertFalse(dispatcher.thread.is_alive())


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

        actual_tick = r.PressureCoordinator.tick_multi
        evaluated = []

        def recording_tick(co, now, readings, tp_of, candidates, actions=None):
            evaluated.append(sorted(source for source, _ in candidates))
            return actual_tick(co, now, readings, tp_of, candidates, actions)

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
                        migration_channel=dict(base_port=30410, generation=3),
                        predictor_events=str(root / "predictor-A-g3-tp1"),
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
                patch.object(r.PressureCoordinator, "tick_multi", recording_tick),
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
            # S1-d2c: both channel-owning TP1 instances are migration sources.
            self.assertTrue(evaluated)
            self.assertTrue(all(row == ["A-g3-tp1", "tp1"] for row in evaluated))
            progress = json.loads((root / "pressure_progress.json").read_text())
            self.assertEqual(progress["stage"], "COMPLETED")
            bad = dict(manifest)
            bad["instances"] = manifest["instances"][:2] + [
                dict(manifest["instances"][2], inventory_path="")
            ]
            (root / "instances.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                pilot.instance_manifest(args, controller)

    def test_single_class_layouts_route_without_the_absent_pair_side(self):
        """S3b-2: L0 (TP1 only) and L2 (TP2 only) serve with migration off."""

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
        actual_ingress = i.PressureIngress
        actual_tick = r.PressureCoordinator.tick_multi
        for pair_pool, pair_id, extra_id, tp, url in (
            ("tp1", "A-g1-tp1", "A-g3-tp1", 1, "source"),
            ("tp4", "B-g2-tp2", "A-g3-tp2", 2, "target"),
        ):
            with self.subTest(pair_pool), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                calls, evaluated = [], []
                write_lock = threading.Lock()
                readers = {url: Reader(pair_pool), "extra": Reader(extra_id)}

                def publish(name, after=-float("inf"), root=root, readers=readers):
                    while time.time() <= after:
                        time.sleep(0.001)
                    value = readers[name].poll(time.time())
                    with (root / name).open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(value) + "\n")

                def post(
                    name,
                    payload,
                    timeout,
                    sink,
                    calls=calls,
                    publish=publish,
                    lock=write_lock,
                ):
                    calls.append(name)
                    sink(0, 42, time.time())
                    completed = time.time()
                    with lock:
                        publish(name, after=completed)
                    return dict(
                        request_started_unix_s=completed, completed_unix_s=completed
                    )

                def make_ingress(*args, post=post, **kwargs):
                    return actual_ingress(*args, **kwargs, post=post)

                def recording_tick(
                    co, now, readings, tp_of, candidates, actions=None, seen=evaluated
                ):
                    seen.append(sorted(source for source, _ in candidates))
                    return actual_tick(co, now, readings, tp_of, candidates, actions)

                for name in readers:
                    publish(name)
                row = dict(tp_degree=tp, context_limit=8192, physical_gpu_ids=["3"])
                manifest = dict(
                    schema="BRIDGETP_INSTANCE_MANIFEST_V1",
                    legacy_pool_aliases={pair_pool: pair_id},
                    instances=[
                        dict(row, instance_id=pair_id, endpoint=url),
                        dict(
                            row,
                            instance_id=extra_id,
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
                self.assertEqual(pilot.pair_pools(checked), (pair_pool,))
                jobs = [
                    dict(
                        id=str(n),
                        at_s=n * 0.03,
                        request=dict(model="qwen", prompt=[1], max_tokens=8),
                    )
                    for n in range(4)
                ]
                with (
                    patch.dict(sys.modules, {"vllm.bridge_tp.async_journal": journal}),
                    patch(
                        "tools.bridge_tp.pressure_sessions.PressureSessions",
                        return_value=session,
                    ),
                    patch.object(o, "scrape_inventory_metrics", return_value=None),
                    patch.object(i, "PressureIngress", side_effect=make_ingress),
                    patch.object(r.PressureCoordinator, "tick_multi", recording_tick),
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
                result = json.loads(
                    (root / "pressure_request_results.json").read_text()
                )
                self.assertEqual(len(result), 4)
                self.assertTrue(all(row["source_done"] for row in result))
                self.assertEqual(
                    {row["initial_pool"] for row in result}, {pair_pool, extra_id}
                )
                self.assertEqual(set(calls), {url, "extra"})
                absent = "tp4" if pair_pool == "tp1" else "tp1"
                self.assertFalse(
                    (root / f"inventory_{absent}_process_receipt.json").exists()
                )
                # No pair to migrate between: L0 never evaluates; L2 has a
                # target but no migration source.
                self.assertTrue(all(row == [] for row in evaluated))
                self.assertEqual(bool(evaluated), pair_pool == "tp4")
                progress = json.loads((root / "pressure_progress.json").read_text())
                self.assertEqual(progress["stage"], "COMPLETED")
                args.no_migration = False
                with self.assertRaises(ValueError):
                    pilot.instance_manifest(args, controller)

    def test_layout_control_adds_and_retires_instances_mid_run(self):
        """S3b-1: the conversion executor's layout file changes membership."""
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

        extra = dict(
            instance_id="A-g3-tp1",
            tp_degree=1,
            context_limit=8192,
            endpoint="extra",
            inventory_path="",
        )

        def layout(revision, *rows):
            path = root / "layout_control.json.tmp"
            path.write_text(json.dumps(dict(revision=revision, instances=list(rows))))
            os.replace(path, root / "layout_control.json")

        def post(url, payload, timeout, sink):
            with write_lock:
                calls.append(url)
                if len(calls) == 1:
                    publish("extra")
                    layout(1, dict(extra, lifecycle="ACTIVE"))
                if len(calls) == 5:
                    layout(
                        2,
                        dict(extra, lifecycle="RETIRED"),
                        dict(
                            instance_id="tp4",
                            tp_degree=4,
                            context_limit=8192,
                            endpoint="target",
                            inventory_path=str(root / "target"),
                            lifecycle="RETIRED",
                        ),
                    )
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
            extra["inventory_path"] = str(root / "extra")
            readers = {
                "source": Reader("tp1"),
                "target": Reader("tp4"),
                "extra": Reader("A-g3-tp1"),
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
                max_seconds=8,
                no_progress_s=4,
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
                    at_s=n * 0.3,
                    request=dict(model="qwen", prompt=[1], max_tokens=8),
                )
                for n in range(8)
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
                    args, (config(), controller, NS(release_budget=17), jobs, {})
                )
            result = json.loads((root / "pressure_request_results.json").read_text())
            self.assertEqual(len(result), 8)
            self.assertTrue(all(row["source_done"] for row in result))
            pools = [row["initial_pool"] for row in result]
            # Joined after its first inventory, served requests, then retired:
            # nothing is routed to it (or to the retired target) afterwards.
            self.assertIn("A-g3-tp1", pools)
            self.assertEqual(pools[-1], "tp1")
            self.assertTrue((root / "inventory_A-g3-tp1_process_receipt.json").exists())
            audit = [
                json.loads(line)
                for line in (root / "pressure_audit.jsonl").read_text().splitlines()
            ]
            applied = [r for r in audit if r["kind"] == "pressure_layout_applied"]
            changes = {}
            for row in applied:
                for c in row["changes"]:
                    changes.setdefault(c["pool_id"], []).append(c["change"])
            self.assertEqual(
                changes,
                {
                    "A-g3-tp1": ["ADDED", "QUIESCING", "RETIRED"],
                    "tp4": ["QUIESCING", "RETIRED"],
                },
            )
            # Request 4 was still open on the extra instance when the layout
            # said RETIRED: it stays QUIESCING until its inventory settles it.
            self.assertIn(["A-g3-tp1"], [r["retire_deferred"] for r in applied])
            progress = json.loads((root / "pressure_progress.json").read_text())
            self.assertEqual(progress["stage"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
