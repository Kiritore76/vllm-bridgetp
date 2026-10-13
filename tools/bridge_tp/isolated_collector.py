# SPDX-License-Identifier: Apache-2.0
"""Spawned HTTP process groups; the parent never parses response streams."""

import json
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path


def _dispatcher(jobs, audit, epoch_input, inputs, output, diagnostic_dir=None):
    """Dedicated process: no SSE parsing, native journal or response objects."""
    from tools.bridge_tp.background_schedule import due_jobs

    sampler = None
    handle = None
    timings = []
    if diagnostic_dir is not None:
        from tools.bridge_tp.collector_resources import ResourceSampler

        handle = (Path(diagnostic_dir) / "dispatcher_resources.jsonl").open(
            "w", encoding="utf-8")

        class Sink:
            def submit(self, row):
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                return True

        sampler = ResourceSampler(Sink(), diagnostics=True)
    output.put(("dispatcher_ready", -1, {}), timeout=5)
    mono_origin, unix_origin = epoch_input.get()
    previous_wall, previous_cpu = time.monotonic(), time.thread_time()
    try:
        arrivals = due_jobs(jobs, audit, mono_origin, unix_origin)
        for planned in arrivals:
            wall, cpu = time.monotonic(), time.thread_time()
            planned = replace(
                planned, dispatch_monotonic=wall, dispatch_unix_s=time.time(),
                dispatch_thread_cpu_s=cpu,
                since_previous_dispatch_wall_s=wall - previous_wall,
                since_previous_dispatch_cpu_s=cpu - previous_cpu,
            )
            previous_wall, previous_cpu = wall, cpu
            # HTTP workers already own all prompts and encoded request bodies.
            ticket = replace(planned, job={"job_id": planned.job["job_id"]})
            owner = inputs["ownership"][ticket.job["job_id"]]
            inputs["queues"][owner].put_nowait(("dispatch", ticket))
            if diagnostic_dir is not None:
                timings.append({"job_id": ticket.job["job_id"],
                                "scheduled_unix_s": ticket.scheduled_unix_s,
                                "dispatch_unix_s": ticket.dispatch_unix_s,
                                "due_to_dispatch_s": wall - ticket.scheduled_monotonic,
                                "enqueue_wall_s": time.monotonic() - wall,
                                "enqueue_thread_cpu_s": time.thread_time() - cpu})
        output.put(("dispatch_done", -1, {}), timeout=5)
    except BaseException as error:
        detail = f"dispatch: {type(error).__name__}: {error}"
        output.put(("fatal", -1, detail), timeout=5)
    finally:
        if sampler is not None:
            sampler.close()
            handle.close()
            path = Path(diagnostic_dir) / "dispatcher_timing.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in timings),
                            encoding="utf-8")


class EvidenceChannel:
    """Per-request native buffers: producers never contend on one journal lock."""

    def __init__(self, output, index, batch_size=64, capacity=1024):
        self.output, self.index = output, index
        self.batch_size, self.capacity = batch_size, capacity
        self.buffers, self.lock = {}, threading.Lock()
        self.dropped = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.thread.start()

    def send(self, kind, payload):
        self.output.put((kind, self.index, payload), timeout=5)

    def register(self):
        buffer = deque()
        with self.lock:
            self.buffers[id(buffer)] = buffer
        return buffer

    def submit(self, buffer, record):
        if len(buffer) >= self.capacity:
            with self.lock:
                self.dropped += 1
            return False
        buffer.append(record)
        return True

    def _flush_loop(self):
        while not self.stop.wait(0.05):
            self.flush()

    def flush(self):
        with self.lock:
            buffers = list(self.buffers.values())
        for buffer in buffers:
            records = []
            while len(records) < self.batch_size:
                try:
                    records.append(buffer.popleft())
                except IndexError:
                    break
            if records:
                try:
                    self.output.put_nowait(("native", self.index, records))
                except queue.Full:
                    with self.lock:
                        self.dropped += len(records)

    def close(self):
        self.stop.set()
        self.thread.join()
        while any(self.buffers.values()):
            self.flush()
        self.send("worker_receipt", {"native_dropped": self.dropped,
                                     "pid": os.getpid(),
                                     "process_cpu_s": time.process_time()})


class NativeSink:
    def __init__(self, channel):
        self.channel, self.buffer = channel, channel.register()

    def submit(self, record):
        if record.get("kind") == "native_arrival":
            try:
                self.channel.output.put_nowait(("native", self.channel.index, [record]))
            except queue.Full:
                with self.channel.lock:
                    self.channel.dropped += 1
                return False
            return True
        return self.channel.submit(self.buffer, record)


def background_handler(planned, config, event, native):
    from argparse import Namespace

    from tools.bridge_tp.run_phase9_capacity_background import run_background_job

    return run_background_job(planned, Namespace(**config), event, native)


def _worker(index, incoming, output, config, handler, native_enabled, threads):
    channel = EvidenceChannel(output, index)
    sampler = None
    try:
        # Import the HTTP implementation before announcing readiness.
        if handler is background_handler:
            from tools.bridge_tp import run_phase9_capacity_background  # noqa: F401

        if os.environ.get("BRIDGETP_COLLECTOR_STALL_DIAGNOSTICS") == "1":
            from tools.bridge_tp.collector_resources import ResourceSampler

            class ResourceSink:
                def submit(self, row):
                    row = {**row, "kind": "worker_" + row["kind"],
                           "collector_process_index": index}
                    channel.send("resource", row)
                    return True

            sampler = ResourceSampler(ResourceSink(), diagnostics=True)

        with ThreadPoolExecutor(max_workers=threads) as pool:
            active = set()
            prepared = {}
            warm_gate, warm_started = threading.Event(), queue.Queue()

            def warm():
                warm_started.put(None)
                warm_gate.wait()

            warming = [pool.submit(warm) for _ in range(threads)]
            for _ in warming:
                warm_started.get(timeout=30)
            warm_gate.set()
            for future in warming:
                future.result()

            def run(planned):
                try:
                    result = handler(
                        planned, config,
                        lambda row: channel.send("event", row),
                        NativeSink(channel) if native_enabled else None,
                    )
                    channel.send("result", (planned.job["job_id"], result, None))
                except Exception as error:
                    channel.send("result", (planned.job["job_id"], None,
                                            f"{type(error).__name__}: {error}"))

            channel.send("ready", {})
            while True:
                message = incoming.get()
                if message is None:
                    break
                kind, value = message
                if kind == "load":
                    job = dict(value)
                    if config.get("collector_preencode_requests"):
                        from tools.bridge_tp.run_phase9_capacity_background import (
                            prepare_background_request,
                        )

                        request = prepare_background_request(job)
                        job["_prepared_request"] = request
                        job["_prepared_body"] = json.dumps(request).encode()
                    prepared[job["job_id"]] = job
                    continue
                if kind == "loaded":
                    channel.send("prepared", {"jobs": len(prepared)})
                    continue
                if kind != "dispatch":
                    raise ValueError("unknown isolated dispatch message")
                planned = replace(value, job=prepared.pop(value.job["job_id"]))
                planned.job["_ticket_received_monotonic"] = time.monotonic()
                planned.job["_ticket_received_unix_s"] = time.time()
                active = {future for future in active if not future.done()}
                if len(active) >= threads:
                    raise RuntimeError("isolated HTTP worker capacity exhausted")
                active.add(pool.submit(run, planned))
        if sampler is not None:
            sampler.close()
            sampler = None
        channel.close()
    except BaseException as error:
        try:
            channel.send("fatal", f"{type(error).__name__}: {error}")
        finally:
            channel.stop.set()


class IsolatedCollector:
    """Bounded IPC, explicit failures, and joined children on every exit."""

    def __init__(self, config, event, native=None, handler=background_handler,
                 processes=4, threads=64, startup_timeout_s=120, resources=None):
        self.event, self.native = event, native
        self.resources = resources
        self.pending, self.lock = {}, threading.Lock()
        self.ready, self.receipts = set(), {}
        self.prepared = {}
        self.config = config
        self.dispatcher = None
        self.dispatched = False
        self.dispatcher_ready = False
        self.error = None
        self.closed = False
        self.context = mp.get_context("spawn")
        self.output = self.context.Queue(maxsize=4096)
        self.epoch_input = self.context.Queue(maxsize=1)
        self.inputs = [self.context.Queue(maxsize=512) for _ in range(processes)]
        self.children = [self.context.Process(
            target=_worker,
            args=(i, inbox, self.output, config, handler, native is not None, threads),
        ) for i, inbox in enumerate(self.inputs)]
        self.stop = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            for child in self.children:
                child.start()
            deadline = time.monotonic() + startup_timeout_s
            while len(self.ready) != processes:
                self.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("isolated collector startup timed out")
                time.sleep(0.02)
        except BaseException:
            self.close(abort=True)
            raise

    def _read(self):
        try:
            while not self.stop.is_set():
                try:
                    kind, index, payload = self.output.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "ready":
                    self.ready.add(index)
                elif kind == "prepared":
                    self.prepared[index] = payload["jobs"]
                elif kind == "worker_receipt":
                    self.receipts[index] = payload
                elif kind == "dispatch_done":
                    self.dispatched = True
                elif kind == "dispatcher_ready":
                    self.dispatcher_ready = True
                elif kind == "fatal":
                    raise RuntimeError(payload)
                elif kind == "event":
                    self.event(payload)
                elif kind == "resource" and self.resources is not None:
                    self.resources.submit(payload)
                elif kind == "native":
                    if self.native is not None:
                        self.native.submit_batch(payload)
                elif kind == "result":
                    key, result, error = payload
                    with self.lock:
                        future = self.pending.pop(key)
                    if error:
                        future.set_exception(RuntimeError(error))
                    else:
                        future.set_result(result)
        except Exception as error:
            self.error = error

    def check(self):
        if self.error is not None:
            raise RuntimeError("isolated evidence receiver failed") from self.error
        if any(child.pid and child.exitcode is not None for child in self.children):
            raise RuntimeError("isolated HTTP worker exited unexpectedly")

    def prepare(self, jobs, audit):
        with self.lock:
            self.pending = {job["job_id"]: Future() for job in jobs}
            if len(self.pending) != len(jobs):
                raise ValueError("duplicate isolated job id")
        counts = [0] * len(self.inputs)
        for index, job in enumerate(jobs):
            worker = index % len(self.inputs)
            self.inputs[worker].put(("load", job), timeout=5)
            counts[worker] += 1
        for inbox in self.inputs:
            inbox.put(("loaded", None), timeout=5)
        deadline = time.monotonic() + 120
        while len(self.prepared) != len(self.inputs):
            self.check()
            if time.monotonic() > deadline:
                raise TimeoutError("isolated request preparation timed out")
            time.sleep(0.02)
        if any(self.prepared[i] != count for i, count in enumerate(counts)):
            raise RuntimeError("isolated prepared request count differs")
        # Stable manifest order determines worker ownership, independent of due order.
        compact = [{key: job[key] for key in
                    ("job_id", "start_after_s", "start_after_event") if key in job}
                   for job in jobs]
        ownership = {job["job_id"]: i % len(self.inputs)
                     for i, job in enumerate(jobs)}
        self.dispatcher = self.context.Process(
            target=_dispatcher,
            args=(compact, audit, self.epoch_input,
                  {"queues": self.inputs, "ownership": ownership}, self.output,
                  self.config.get("out_dir") if os.environ.get(
                      "BRIDGETP_COLLECTOR_STALL_DIAGNOSTICS") == "1" else None),
        )
        self.dispatcher.start()
        deadline = time.monotonic() + 120
        while not self.dispatcher_ready:
            self.check()
            if self.dispatcher.exitcode is not None:
                raise RuntimeError("isolated dispatcher exited before readiness")
            if time.monotonic() > deadline:
                raise TimeoutError("isolated dispatcher readiness timed out")
            time.sleep(0.02)

    def execute(self, mono_origin, unix_origin):
        if not self.dispatcher_ready:
            raise RuntimeError("prepare dispatcher before selecting arrival epoch")
        with self.lock:
            futures = list(self.pending.values())
        self.epoch_input.put_nowait((mono_origin, unix_origin))
        while not all(future.done() for future in futures) or not self.dispatched:
            self.check()
            if self.dispatcher.exitcode not in (None, 0):
                raise RuntimeError("isolated dispatcher exited unexpectedly")
            time.sleep(0.02)
        return [future.result() for future in futures]

    def close(self, abort=False):
        if self.closed:
            return
        self.closed = True
        try:
            if not abort:
                for inbox in self.inputs:
                    inbox.put(None, timeout=2)
                deadline = time.monotonic() + 30
                while len(self.receipts) < len(self.children):
                    if self.error:
                        raise RuntimeError("isolated evidence receiver failed")
                    if time.monotonic() > deadline:
                        raise TimeoutError("isolated evidence drain timed out")
                    time.sleep(0.02)
                if any(r["native_dropped"] for r in self.receipts.values()):
                    raise RuntimeError("isolated native evidence dropped")
        finally:
            if self.dispatcher is not None and self.dispatcher.pid:
                self.dispatcher.join(timeout=2)
                if self.dispatcher.is_alive():
                    self.dispatcher.terminate()
                    self.dispatcher.join(timeout=2)
            for child in self.children:
                if child.pid:
                    child.join(timeout=2)
                    if child.is_alive():
                        child.terminate()
                        child.join(timeout=2)
            self.stop.set()
            self.reader.join(timeout=2)
            for inbox in self.inputs:
                inbox.cancel_join_thread()
                inbox.close()
            self.output.cancel_join_thread()
            self.output.close()
            self.epoch_input.cancel_join_thread()
            self.epoch_input.close()
