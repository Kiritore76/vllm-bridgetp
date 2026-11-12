# SPDX-License-Identifier: Apache-2.0
"""Drain recompute on the real ingress, ledger and response seam (CPU only).

Only the HTTP transport is fake: the source emits tokens, then blocks until the
controller closes its response (the client disconnect vLLM aborts on); the
continuation engine emits the rest. No GPU or vLLM engine claim.
"""

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

from tools.bridge_tp import pressure_sustained as sustained
from tools.bridge_tp import six_conversion as six
from tools.bridge_tp.pressure_conversion import SAFETY
from tools.bridge_tp.pressure_cpu import load
from tools.bridge_tp.pressure_result_export import write_request_results

p, r, o, ingress_mod, rc = (
    load(name)
    for name in (
        "pressure_policy",
        "pressure_runtime",
        "pressure_observation",
        "pressure_ingress",
        "pressure_recompute",
    )
)


def reading(name, ratio=0.4):
    return p.PoolReading(name, 10, 10, 500, 16, 8192, ratio, 30, 10, False, 0, 0, 1)


class FakeResponse:
    def __init__(self):
        self.closed = threading.Event()

    def close(self):
        self.closed.set()


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


class Transport:
    """Source stops after ``emit`` tokens; ``finish`` lets it end on its own."""

    def __init__(self, emit=5):
        self.emit = emit
        self.calls = []
        self.finish = threading.Event()

    def __call__(self, url, payload, timeout, sink, on_response=None):
        self.calls.append((url, dict(payload)))
        if url == "source" and len(payload["prompt"]) == 64:  # not a continuation
            response = FakeResponse()
            if on_response is not None:
                on_response(response)
            for index in range(self.emit):
                sink(index, 100 + index, time.time())
            while not (response.closed.is_set() or self.finish.is_set()):
                time.sleep(0.01)
            if self.finish.is_set() and not response.closed.is_set():
                for index in range(self.emit, payload["max_tokens"]):
                    sink(index, 100 + index, time.time())
                return dict(
                    request_started_unix_s=time.time(),
                    completed_unix_s=time.time(),
                    finish_reason="length",
                )
            raise RuntimeError("streaming response ended before completion")
        for index in range(payload["max_tokens"]):
            sink(index, 200 + index, time.time())
        return dict(
            response_id="cmpl-new",
            completed_unix_s=time.time(),
            finish_reason="length",
        )


class DrainRecomputeTests(unittest.TestCase):
    def setUp(self):
        cfg = p.PressureConfig(
            prompt_boundary=2048,
            prefill_pending_budget=2048,
            prefill_waiting_budget=4,
            capacity_guard_blocks=125,
            sustain_s=0,
            survival_horizon=64,
        )
        self.rows = []
        self.co = r.PressureCoordinator(cfg, self.rows.append)
        self.window = o.NativeServiceWindow()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(self.executor.shutdown, wait=False, cancel_futures=True)
        self.transport = Transport()
        self.ingress = ingress_mod.PressureIngress(
            self.co,
            self.window,
            self.executor,
            "source",
            "target",
            post=self.transport,
        )
        self.ingress.urls["new"] = "new"
        self.tp_of = {"tp1": 1, "tp4": 2}
        self.manager = rc.DrainRecompute(
            ingress=self.ingress,
            coordinator=self.co,
            audit=self.rows.append,
            tp_of=self.tp_of,
            target_of=lambda bundle, known: next(
                (i for i in self.tp_of if i not in known and bundle == "A"), None
            ),
        )
        self.addCleanup(self.release_threads)
        self.job = {"id": "long-1", "at_s": 0.0}
        self.stream = self.ingress.submit(
            dict(model="qwen", prompt=[1] * 64, max_tokens=12),
            (reading("tp1"), reading("tp4")),
        )
        self.assertEqual(self.stream.pool_id, "tp1")
        wait_for(lambda: self.stream.recorder.progress()[0] == 5)
        self.server_id = f"cmpl-{self.stream.client_id}-0-abcdef"
        self.ingress.reconcile(self.inventory("tp1", generated=5), time.time())
        self.co.set_lifecycle("tp1", "QUIESCING")

    def release_threads(self):
        """No source or parked thread may outlive its test (exit would wait)."""
        self.transport.finish.set()
        for job in self.manager.jobs.values():
            job.fail("test cleanup")
        try:
            self.stream.future.exception(timeout=5)
        except Exception:
            pass

    def inventory(self, pool, *, generated=None, sampled_at=None):
        rows = []
        if generated is not None:
            rows.append(
                dict(
                    request_id=self.server_id,
                    owner=pool,
                    max_output_tokens=12,
                    prompt_tokens=64,
                    generated=generated,
                    ignore_eos=False,
                    owned_blocks=5,
                )
            )
        stamp = time.time() if sampled_at is None else sampled_at
        return dict(pool_id=pool, sampled_at=stamp, generation=1, requests=rows)

    def request(self):
        return self.manager.request(
            dict(
                seq=1,
                bundle="A",
                requests=[dict(request_id=self.server_id, instance="tp1")],
            )
        )

    def test_abort_park_and_continue_on_the_new_instance(self):
        result = self.request()
        self.assertEqual(result["state"], "STARTED")
        job = self.stream.recompute
        wait_for(lambda: job.state == "SOURCE_ENDED")
        self.assertEqual(job.cutover, 5)
        self.assertEqual(self.co._owners[self.server_id], rc.PARKED_OWNER)
        self.assertFalse(self.stream.future.done())  # one client stream

        # A sample taken before the end is no release evidence.
        early = self.inventory("tp1", sampled_at=job.source_ended_at - 1)
        self.manager.tick(time.time(), [reading("tp1")], {"tp1": early})
        self.assertEqual(job.state, "SOURCE_ENDED")
        later = self.inventory("tp1", sampled_at=job.source_ended_at + 0.5)
        self.manager.tick(time.time(), [reading("tp1")], {"tp1": later})
        self.assertEqual(job.state, "PARKED")
        entry = self.co.ledger.entries[self.stream.reservation_key]
        self.assertEqual(entry.state, "MOVED")

        # No new instance yet: the request waits instead of going elsewhere.
        self.manager.tick(
            time.time(), [reading("tp1"), reading("tp4")], {"tp1": later}
        )
        self.assertEqual(job.state, "PARKED")

        self.tp_of["new"] = 2
        self.window.add_pool("new")  # the pilot adds it with the layout row
        self.manager.tick(
            time.time(),
            [reading("tp1"), reading("tp4"), reading("new", 0.0)],
            {"tp1": later},
        )
        self.assertIn(job.state, {"RESUMED", "COMPLETED"})
        transport = self.stream.future.result(timeout=5)
        self.assertEqual(job.state, "COMPLETED")
        url, payload = self.transport.calls[-1]
        self.assertEqual(url, "new")
        self.assertEqual(payload["prompt"], [1] * 64 + [100, 101, 102, 103, 104])
        self.assertEqual(payload["max_tokens"], 7)
        self.assertEqual(payload["request_id"], self.stream.client_id)
        emitted = self.stream.recorder.proxy.emitted
        self.assertEqual([t.index for t in emitted], list(range(12)))
        self.assertEqual(
            [t.token_id for t in emitted],
            [100, 101, 102, 103, 104] + [200 + i for i in range(7)],
        )
        self.assertEqual(
            [t.origin for t in emitted], ["source"] * 5 + ["target"] * 7
        )
        self.assertEqual(transport["finish_reason"], "length")
        self.assertEqual(transport["recompute"]["state"], "COMPLETED")
        self.assertEqual(self.stream.pool_id, "new")
        self.assertEqual(self.stream.initial_pool_id, "tp1")
        self.assertEqual(self.stream.original_payload["max_tokens"], 12)
        new_entry = self.co.ledger.entries["recompute-" + self.stream.client_id]
        self.assertEqual(new_entry.pool_id, "new")

        # Reconcile binds the continuation's server request on the new pool.
        resumed = f"cmpl-{self.stream.client_id}-0-fedcba"
        self.ingress.reconcile(
            dict(
                pool_id="new",
                sampled_at=time.time(),
                generation=1,
                requests=[
                    dict(
                        request_id=resumed,
                        owner="new",
                        max_output_tokens=7,
                        prompt_tokens=69,
                        generated=1,
                        ignore_eos=False,
                        owned_blocks=5,
                    )
                ],
            ),
            time.time(),
        )
        self.assertEqual(self.stream.server_id, resumed)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            write_request_results(
                path, {self.job["id"]: self.stream}, [self.job], time.time()
            )
            row = json.loads(path.read_text())[0]
        self.assertEqual(row["initial_pool"], "tp1")
        self.assertEqual(row["request"]["max_tokens"], 12)
        self.assertEqual(row["request"]["prompt"], [1] * 64)
        self.assertEqual(row["recompute"]["cutover"], 5)
        self.assertTrue(row["response"]["committed"])
        self.assertTrue(sustained.valid_output(row, row["request"], True))
        self.assertTrue(sustained.valid_output(row, row["request"], False))

    def test_idle_release_sample_before_the_stream_end_settles(self):
        # GPU 11-12: the engine published its empty inventory once, a few ms
        # before the source stream thread recorded its end, then stayed idle.
        self.request()
        job = self.stream.recompute
        wait_for(lambda: job.state == "SOURCE_ENDED")
        before = self.inventory("tp1", sampled_at=job.requested_at - 0.01)
        self.manager.tick(time.time(), [reading("tp1")], {"tp1": before})
        self.assertEqual(job.state, "SOURCE_ENDED")
        listed = self.inventory(
            "tp1", generated=5, sampled_at=job.source_ended_at + 1
        )
        self.manager.tick(time.time(), [reading("tp1")], {"tp1": listed})
        self.assertEqual(job.state, "SOURCE_ENDED")
        between = (job.requested_at + job.source_ended_at) / 2
        self.assertLess(between, job.source_ended_at)
        released = self.inventory("tp1", sampled_at=between)
        self.manager.tick(time.time(), [reading("tp1")], {"tp1": released})
        self.assertEqual(job.state, "PARKED")
        entry = self.co.ledger.entries[self.stream.reservation_key]
        self.assertEqual(entry.state, "MOVED")

    def test_source_that_finished_first_is_not_resubmitted(self):
        stream = self.stream
        # Let the source finish before the disconnect can take effect.
        stream._abort_requested = False
        original = stream.abort_source
        stream.abort_source = lambda: None
        self.request()
        self.transport.finish.set()
        result = stream.future.result(timeout=5)
        self.assertEqual(stream.recompute.state, "NATURAL")
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(stream.recorder.progress()[0], 12)
        self.assertEqual(len(self.transport.calls), 1)
        stream.abort_source = original

    def test_refusals_leave_the_stream_untouched(self):
        self.co.set_lifecycle("tp1", "ACTIVE")
        result = self.request()
        self.assertEqual(result["state"], "REFUSED")
        self.assertIsNone(self.stream.recompute)
        self.assertFalse(self.stream._abort_requested)
        # The same sequence number is not applied twice.
        self.co.set_lifecycle("tp1", "QUIESCING")
        self.assertEqual(self.request()["state"], "REFUSED")
        unknown = self.manager.request(
            dict(seq=2, bundle="A", requests=[dict(request_id="x", instance="tp1")])
        )
        self.assertEqual(unknown["state"], "REFUSED")
        self.transport.finish.set()
        self.stream.future.result(timeout=5)

    def test_parked_request_falls_back_to_any_active_instance(self):
        self.manager.fallback_after_s = 0.0
        self.request()
        job = self.stream.recompute
        wait_for(lambda: job.state == "SOURCE_ENDED")
        later = self.inventory("tp1", sampled_at=job.source_ended_at + 0.5)
        self.manager.tick(time.time(), [reading("tp1"), reading("tp4")], {"tp1": later})
        self.stream.future.result(timeout=5)
        self.assertEqual(job.target_pool, "tp4")
        self.assertEqual(self.transport.calls[-1][0], "target")

    def test_aborted_conversion_resumes_without_waiting(self):
        self.request()
        job = self.stream.recompute
        wait_for(lambda: job.state == "SOURCE_ENDED")
        later = self.inventory("tp1", sampled_at=job.source_ended_at + 0.5)
        self.manager.tick(time.time(), [reading("tp1")], {"tp1": later})
        self.assertEqual(job.state, "PARKED")
        self.co.set_lifecycle("tp1", "ACTIVE")  # the conversion reopened it
        self.manager.tick(time.time(), [reading("tp1")], {"tp1": later})
        self.stream.future.result(timeout=5)
        self.assertEqual(job.target_pool, "tp1")
        self.assertEqual(self.transport.calls[-1][1]["max_tokens"], 7)



class RealSocketAbortTests(unittest.TestCase):
    """The disconnect is real: a local SSE server sees its writes fail."""

    def test_closing_the_response_ends_the_read_and_disconnects(self):
        import http.server

        online = load("online_io")
        disconnected = threading.Event()

        class Slow(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for index in range(400):
                        chunk = dict(id="x", choices=[dict(token_ids=[index])])
                        line = "data: " + json.dumps(chunk) + chr(10) * 2
                        self.wfile.write(line.encode())
                        self.wfile.flush()
                        time.sleep(0.02)
                except OSError:
                    disconnected.set()

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        responses, tokens = [], []
        outcome = {}

        def read():
            try:
                online.post_streaming_completion(
                    f"http://127.0.0.1:{server.server_port}",
                    dict(prompt=[1], max_tokens=400),
                    30,
                    lambda i, t, u: tokens.append(t),
                    on_response=responses.append,
                )
            except Exception as error:
                outcome["error"] = error

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        wait_for(lambda: len(tokens) >= 5)
        started = time.monotonic()
        rc.close_response(responses[0])
        reader.join(5)
        self.assertFalse(reader.is_alive())
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIn("error", outcome)
        self.assertLess(len(tokens), 400)
        self.assertTrue(disconnected.wait(5))


class ContinuationPayloadTests(unittest.TestCase):
    def test_continuation_payload_contract(self):
        payload = dict(prompt=[1, 2], max_tokens=4, request_id="c")
        self.assertEqual(
            rc.continuation_payload(payload, [7, 8]),
            dict(prompt=[1, 2, 7, 8], max_tokens=2, request_id="c"),
        )
        self.assertEqual(rc.continuation_payload(payload, [])["max_tokens"], 4)
        with self.assertRaises(ValueError):
            rc.continuation_payload(payload, [1, 2, 3, 4])


def status_instances(rows, *, pending=None, sessions=0):
    return {
        "A-g1": dict(
            tp_degree=1,
            sessions=sessions,
            streams_pending=len(rows) if pending is None else pending,
            requests=rows,
        ),
        "A-g2": dict(tp_degree=1, sessions=0, streams_pending=0, requests=[]),
    }


def row(request_id, prompt=16000, generated=1000):
    return dict(request_id=request_id, prompt_tokens=prompt, generated=generated)


class RecomputeChoiceTests(unittest.TestCase):
    def test_waits_for_the_delay_then_takes_every_request_on_the_group(self):
        instances = status_instances([row("a"), row("b")])
        refs = ["A-g1", "A-g2"]
        reason, choice = six.recompute_choice(instances, refs, 44.9, SAFETY)
        self.assertIsNone(choice)
        self.assertIn("delay", reason)
        reason, choice = six.recompute_choice(instances, refs, 45.0, SAFETY)
        self.assertIsNone(reason)
        self.assertEqual([r["request_id"] for r in choice["requests"]], ["a", "b"])
        self.assertEqual(choice["context_tokens"], 34000)

    def test_caps_and_unknowns_keep_the_natural_drain(self):
        refs = ["A-g1", "A-g2"]
        many = status_instances([row(str(i), 1000, 10) for i in range(4)])
        self.assertIn("cap", six.recompute_choice(many, refs, 60, SAFETY)[0])
        big = status_instances([row("a", 32000, 2000), row("b", 30000, 2000)])
        self.assertIn("budget", six.recompute_choice(big, refs, 60, SAFETY)[0])
        hidden = status_instances([row("a")], pending=2)
        self.assertIn("differ", six.recompute_choice(hidden, refs, 60, SAFETY)[0])
        moving = status_instances([row("a")], sessions=1)
        self.assertIn("session", six.recompute_choice(moving, refs, 60, SAFETY)[0])
        empty = status_instances([])
        self.assertIn("empty", six.recompute_choice(empty, refs, 60, SAFETY)[0])

    def test_stalled_requests_go_but_count_toward_neither_cap_nor_budget(self):
        """GPU 11-11: four on A at 45s, two had stood still ~130s."""
        refs = ["A-g1", "A-g2"]
        rows = [row("a"), row("b"), row("c", 20000), row("d", 20000)]
        instances = status_instances(rows)
        reason, choice = six.recompute_choice(instances, refs, 45, SAFETY)
        self.assertIn("4 progressing requests exceed", reason)
        reason, choice = six.recompute_choice(
            instances, refs, 45, SAFETY, frozenset({"c", "d"})
        )
        self.assertIsNone(reason)
        self.assertEqual([r["request_id"] for r in choice["requests"]], list("abcd"))
        self.assertEqual(choice["stalled"], 2)
        self.assertEqual(choice["counted_tokens"], 34000)  # only a and b
        self.assertEqual(choice["context_tokens"], 34000 + 2 * 21000)
        # Progressing requests alone still respect the token budget.
        big = status_instances([row("a", 32000, 2000), row("b", 30000, 2000)])
        self.assertIn("budget", six.recompute_choice(big, refs, 45, SAFETY)[0])

    def test_step_marks_a_request_stalled_after_10s_without_output(self):
        instances = status_instances(
            [row("a"), row("b"), row("c"), row("d")]
        )
        backend, clock = self.backend(1000.0, instances)
        plan = dict(plan_id="p", bundle_id="A")
        for t in range(0, 46):
            clock[0] = 1000.0 + t
            for r in instances["A-g1"]["requests"]:
                if r["request_id"] in ("a", "b") or t < 30:
                    r["generated"] = 1000 + t  # c and d stop at 30s
            backend.recompute_step(plan, clock[0])
        # At 45s c and d have been still for 15s: two progressing, so go.
        self.assertEqual(len(backend.layouts), 1)
        self.assertEqual(backend.layouts[0]["stalled"], 2)

    def test_output_growth_restarts_the_stall_clock(self):
        instances = status_instances([row(x) for x in "abcd"])
        backend, clock = self.backend(1000.0, instances)
        plan = dict(plan_id="p", bundle_id="A")
        for t in range(0, 60):
            clock[0] = 1000.0 + t
            for r in instances["A-g1"]["requests"]:
                # c and d grow once every 9s: slow, never stalled.
                if r["request_id"] in ("a", "b") or t % 9 == 0:
                    r["generated"] = 1000 + t
            backend.recompute_step(plan, clock[0])
        self.assertEqual(backend.layouts, [])

    def backend(self, now, instances):
        backend = six.SixBackend.__new__(six.SixBackend)
        clock = [now]
        backend.clock = lambda: clock[0]
        backend.recompute_enabled = True
        backend.layout_recompute = None
        backend.recompute_seq = 0
        backend.recompute_since = None
        backend.last_recompute_reason = None
        backend.recompute_progress = {}
        backend.assist_seq = backend.assist_step_start_seq = 0
        backend.safety = dict(SAFETY)
        backend.decisions, backend.events, backend.layouts = [], [], []
        backend.decision = lambda kind, **v: backend.decisions.append((kind, v))
        backend.write_layout = lambda changes: backend.layouts.append(
            backend.layout_recompute
        )
        backend.ops = NS(event=lambda kind, **v: backend.events.append(kind))
        backend.refs = lambda plan: [NS(instance_id=i) for i in instances]
        backend.status = lambda: dict(unix_s=clock[0], instances=instances)
        return backend, clock

    def test_step_requests_once_after_the_drain_delay(self):
        instances = status_instances([row("a")])
        backend, clock = self.backend(1000.0, instances)
        plan = dict(plan_id="p", bundle_id="A")
        backend.recompute_step(plan, 1000.0)
        self.assertEqual(backend.layouts, [])
        clock[0] = 1045.0
        backend.recompute_step(plan, 1045.0)
        self.assertEqual(len(backend.layouts), 1)
        control = backend.layouts[0]
        self.assertEqual(control["seq"], 1)
        self.assertEqual(control["bundle"], "A")
        self.assertEqual(control["requests"][0]["request_id"], "a")
        self.assertEqual(backend.events, ["RECOMPUTE_REQUESTED"])
        clock[0] = 1100.0
        backend.recompute_step(plan, 1100.0)
        self.assertEqual(len(backend.layouts), 1)

    def test_step_is_off_by_default_and_only_for_tp1_merges(self):
        instances = status_instances([row("a")])
        backend, clock = self.backend(1000.0, instances)
        backend.recompute_enabled = False
        clock[0] = 2000.0
        backend.recompute_step(dict(plan_id="p", bundle_id="A"), 2000.0)
        self.assertEqual(backend.layouts, [])
        backend.recompute_enabled = True
        instances["A-g1"]["tp_degree"] = 2
        backend.recompute_step(dict(plan_id="p", bundle_id="A"), 2000.0)
        self.assertEqual(backend.layouts, [])
        backend.layout_recompute = dict(seq=1)
        self.assertEqual(
            backend.assist_choice(dict(), dict(unix_s=2000.0, instances={})),
            ("group requests are being recomputed", None),
        )



class SmokeCheckTests(unittest.TestCase):
    """The GPU smoke's checker on synthetic records of both outcomes."""

    def scene(self, root, *, requested):
        from tools.bridge_tp import drain_recompute_smoke as smoke

        scene = root / "trial" / "scene"
        (scene / "controller").mkdir(parents=True)
        plan = dict(state="COMPLETED", bundle_id="A")
        (scene / "conversion_sequence_result.json").write_text(
            json.dumps(dict(steps=[dict(plan=plan)]))
        )
        events = [
            dict(kind="TRIGGER", unix_s=110.0),
            dict(kind="PLAN_STATE", state="DRAINING", unix_s=111.0),
        ]
        decisions = [dict(kind="RECOMPUTE", unix_s=120.0, reason="drain younger")]
        rows = [
            dict(
                job_id="s",
                request=dict(max_tokens=2),
                response=dict(
                    emitted=[
                        dict(index=0, origin="source", unix_s=1.0),
                        dict(index=1, origin="source", unix_s=2.0),
                    ]
                ),
                source_error=None,
            )
        ]
        if requested:
            events.append(dict(kind="RECOMPUTE_REQUESTED", unix_s=156.0))
            decisions.append(
                dict(kind="RECOMPUTE", unix_s=156.0, reason=None, choice=dict(seq=1))
            )
            emitted = [
                dict(index=i, origin="source" if i < 2 else "target", unix_s=float(i))
                for i in range(4)
            ]
            rows.append(
                dict(
                    job_id="l",
                    request=dict(max_tokens=4),
                    response=dict(emitted=emitted, handoff_stall_s=30.0),
                    source_error=None,
                    source_transport=dict(finish_reason="length"),
                    recompute=dict(
                        state="COMPLETED",
                        source_pool="A-g1",
                        target_pool="A-g3-tp2",
                        cutover=2,
                        requested_at=156.0,
                        source_ended_at=156.2,
                        parked_at=157.0,
                        resumed_at=180.0,
                    ),
                )
            )
        for name, items in (
            ("conversion_events.jsonl", events),
            ("controller/conversion_decisions.jsonl", decisions),
            ("controller/pressure_audit.jsonl", []),
        ):
            (scene / name).write_text("".join(json.dumps(x) + "\n" for x in items))
        (scene / "controller/layout_status.json").write_text(
            json.dumps(dict(workload_origin_unix_s=0.0))
        )
        (scene / "controller/pressure_request_results.json").write_text(
            json.dumps(rows)
        )
        return smoke, root / "trial"

    def test_exercised_and_missing_coverage_codes(self):
        with tempfile.TemporaryDirectory() as folder:
            smoke, trial = self.scene(Path(folder), requested=True)
            value, code = smoke.check(trial)
            self.assertEqual(code, 0, value["errors"])
            self.assertEqual(value["timeline_s"]["recompute_requested"], 156.0)
            self.assertTrue(value["jobs"][0]["seam"])
        with tempfile.TemporaryDirectory() as folder:
            smoke, trial = self.scene(Path(folder), requested=False)
            self.assertEqual(smoke.check(trial)[1], 2)


if __name__ == "__main__":
    unittest.main()
