# SPDX-License-Identifier: Apache-2.0
"""Real spawned processes; no GPU inference or external network."""

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.isolated_collector import IsolatedCollector

ROOT = Path(__file__).resolve().parents[2]


def stub_vllm_packages():
    for name in ("vllm", "vllm.bridge_tp", "vllm.bridge_tp.controller"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(ROOT / name.replace(".", "/"))]
            sys.modules[name] = package
spec = importlib.util.spec_from_file_location(
    "isolated_journal_test", ROOT / "vllm/bridge_tp/async_journal.py")
journal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(journal)


def busy_handler(planned, config, event, native):
    if config.get("crash"):
        os._exit(3)
    deadline = time.monotonic() + config.get("cpu_s", 0)
    while time.monotonic() < deadline:
        pass
    if native is not None:
        native.submit({"kind": "native_arrival", "job_id": planned.job["job_id"],
                       "unix_s": 123.0})
        for index in range(100):
            native.submit({"kind": "native_itl", "job_id": planned.job["job_id"],
                           "unix_s": 124 + index / 100, "index": index})
    return {"job_id": planned.job["job_id"],
            "dispatch_lag_s": planned.dispatch_monotonic - planned.scheduled_monotonic,
            "scheduled_unix_s": planned.scheduled_unix_s}


def real_background_handler(planned, config, event, native):
    stub_vllm_packages()
    from tools.bridge_tp.isolated_collector import background_handler

    return background_handler(planned, config, event, native)


class IsolatedCollectorTests(unittest.TestCase):
    def test_real_http_streams_keep_tokens_and_drain_batched_receipts(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                packet = {"id": "test", "choices": [{"token_ids": [1, 2, 3],
                                                         "finish_reason": "stop"}]}
                self.wfile.write(("data: " + json.dumps(packet)
                                  + "\n\ndata: [DONE]\n\n").encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                url = f"http://127.0.0.1:{server.server_port}"
                config = {"out_dir": Path(directory), "source_url": url,
                          "target_url": url, "request_timeout_s": 5,
                          "collector_retain_chunks": False}
                native = journal.AsyncJournal(Path(directory) / "native.jsonl")
                collector = IsolatedCollector(config, lambda row: None, native,
                                              handler=real_background_handler,
                                              processes=2, threads=2)
                try:
                    jobs = [{"job_id": str(i), "pool": "target",
                             "start_after_s": i * 0.05,
                             "request": {"prompt": [10], "max_tokens": 3}}
                            for i in range(4)]
                    collector.prepare(jobs, None)
                    rows = collector.execute(time.monotonic(), time.time())
                    collector.close()
                    for row in rows:
                        self.assertEqual(row["status"], "COMPLETED", row.get("error"))
                        self.assertEqual(row["token_ids"], [1, 2, 3])
                        self.assertEqual(len(row["token_times_unix_s"]), 3)
                        self.assertEqual(row["status"], "COMPLETED")
                finally:
                    collector.close(abort=True)
                    native.close()
                footer = json.loads((Path(directory) / "native.jsonl")
                                    .read_text().splitlines()[-1])
                self.assertEqual(footer["saved_records"], 12)
                self.assertEqual(footer["dropped"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_spawned_stream_cpu_does_not_run_in_dispatcher_and_no_events_lost(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "native.jsonl"
            native = journal.AsyncJournal(path)
            collector = IsolatedCollector({"cpu_s": 0.4}, lambda row: None, native,
                                          handler=busy_handler, processes=2, threads=2)
            try:
                jobs = [{"job_id": str(i), "start_after_s": i * 0.05} for i in range(4)]
                collector.prepare(jobs, None)
                mono, unix = time.monotonic(), time.time()
                results = collector.execute(mono, unix)
                collector.close()
                self.assertEqual(len(results), 4)
                self.assertLess(max(r["dispatch_lag_s"] for r in results), 1.0)
                for i, row in enumerate(results):
                    self.assertAlmostEqual(row["scheduled_unix_s"], unix + i * 0.05)
                self.assertTrue(all(not p.is_alive() for p in collector.children))
                self.assertEqual(sum(r["native_dropped"]
                                     for r in collector.receipts.values()), 0)
            finally:
                collector.close(abort=True)
                native.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[-1]["saved_records"], 404)
            self.assertEqual(records[-1]["dropped"], 0)
            for job_id in map(str, range(4)):
                tokens = [r["index"] for r in records[:-1]
                          if r.get("job_id") == job_id and "index" in r]
                self.assertEqual(tokens, list(range(100)))

    def test_dead_worker_fails_and_all_children_are_reaped(self):
        collector = IsolatedCollector({"crash": True}, lambda row: None,
                                      handler=busy_handler, processes=1, threads=1)
        try:
            collector.prepare([{"job_id": "crash", "start_after_s": 0}], None)
            with self.assertRaisesRegex(RuntimeError, "exited"):
                collector.execute(time.monotonic(), time.time())
        finally:
            collector.close(abort=True)
        self.assertTrue(all(not p.is_alive() for p in collector.children))

    def test_batch_overflow_counts_records_not_messages(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "native.jsonl"
            with patch.object(journal.AsyncJournal, "_run", lambda self: None):
                writer = journal.AsyncJournal(path, capacity=1)
                self.assertTrue(writer.submit_batch([{"i": i} for i in range(64)]))
                self.assertFalse(writer.submit_batch([{"i": i} for i in range(64)]))
                self.assertEqual(writer.dropped, 64)
                writer.handle.close()

    def test_optional_chunk_retention_preserves_ids_and_token_timestamps(self):
        spec = importlib.util.spec_from_file_location(
            "vllm.bridge_tp.controller.isolated_online_io_test",
            ROOT / "vllm/bridge_tp/controller/online_io.py")
        stub_vllm_packages()
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        packet = {"id": "x", "choices": [{"token_ids": [1, 2],
                                              "finish_reason": "stop"}]}

        class Response:
            def __enter__(self):
                return iter([("data: " + json.dumps(packet)).encode(), b"data: [DONE]"])

            def __exit__(self, *args):
                return False

        for retain in (True, False):
            seen = []
            with patch.object(module.urllib.request, "urlopen",
                              return_value=Response()):
                result = module.post_streaming_completion(
                    "http://local-test", {}, 1,
                    lambda *row, sink=seen: sink.append(row),
                    retain_chunks=retain)
            self.assertEqual(result["token_ids"], [1, 2])
            self.assertEqual(len(seen), 2)
            self.assertEqual(len(result["chunks"]), int(retain))
