# SPDX-License-Identifier: Apache-2.0
"""CPU-only inventory subprocess and bounded latest-snapshot IPC cache."""

import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import pause_diagnostics as timing
from .pressure_observation import InventoryReader

MAX_MESSAGE_BYTES = 262144
SCHEMA = "PRESSURE_INVENTORY_IPC_V1"
ROOT = Path(__file__).resolve().parents[3]
TOP_FIELDS = (
    "schema",
    "pool_id",
    "sampled_at",
    "generation",
    "free_blocks",
    "total_blocks",
    "usage_denominator_blocks",
    "block_size",
    "context_limit",
    "running",
    "waiting",
    "pending_prompt_tokens",
    "prefill_generation",
    "dropped_before",
    "writer_error",
    "inventory_error",
)
REQUEST_FIELDS = (
    "request_id",
    "owner",
    "running",
    "generated",
    "prompt_tokens",
    "computed_tokens",
    "max_output_tokens",
    "ignore_eos",
    "owned_blocks",
    "releasable_blocks",
    "block_group_supported",
)
BATCH_FIELDS = (
    "sampled_at",
    "error",
    "decode_requests",
    "decode_context_sum",
    "prefill_tokens",
    "recompute_tokens",
)
BATCH_REQUEST_FIELDS = (
    "request_id",
    "computed_before",
    "decode_tokens",
    "prefill_tokens",
    "recompute_tokens",
)


def compact_inventory(raw):
    """Discard physical block lists/history only after validating the raw record."""
    if raw is None:
        return None
    result = {key: raw[key] for key in TOP_FIELDS if key in raw}
    result["requests"] = [
        {key: row[key] for key in REQUEST_FIELDS if key in row}
        for row in raw["requests"]
    ]
    batch = raw.get("last_scheduled_batch")
    if batch is not None:
        value = {key: batch[key] for key in BATCH_FIELDS if key in batch}
        value["requests"] = [
            {key: row[key] for key in BATCH_REQUEST_FIELDS if key in row}
            for row in batch.get("requests", [])
        ]
        batch = value
    result["last_scheduled_batch"] = batch
    return result


def worker(path, pool_id, timing_path, interval_s=0.05):
    """Read every raw record in an exec'd interpreter, never importing CUDA."""
    # Loading this file directly avoids vllm/__init__ and GPU initialization.
    spec = importlib.util.spec_from_file_location(
        "inventory_async_journal",
        Path(__file__).resolve().parents[1] / "async_journal.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observer = timing.TimingObserver(module.AsyncJournal(timing_path))
    observer.journal.submit(
        dict(
            kind="inventory_worker_identity",
            pid=os.getpid(),
            parent_pid=os.getppid(),
            pool_id=pool_id,
            cuda_package_imported=any(
                name in sys.modules for name in ("vllm", "torch")
            ),
        )
    )
    reader = InventoryReader(path, pool_id)
    stop = threading.Event()

    def parent_commands():
        # Parent exit closes stdin as well. Children inherit its process group
        # so owner shutdown can also reap them if parsing/file I/O is stuck.
        sys.stdin.buffer.read(1)
        stop.set()

    threading.Thread(target=parent_commands, daemon=True).start()
    seq = 0

    def send(status, value=None, error=None):
        nonlocal seq
        message = dict(
            schema=SCHEMA,
            pool_id=pool_id,
            pid=os.getpid(),
            seq=seq + 1,
            status=status,
            inventory=compact_inventory(value),
            observed_at=reader.observed_at,
            diagnostic=reader.diagnostics(),
            error=error,
        )
        payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
        if len(payload) + 1 > MAX_MESSAGE_BYTES:
            raise ValueError("compact inventory exceeds IPC byte budget")
        sys.stdout.buffer.write(payload + b"\n")
        sys.stdout.buffer.flush()
        seq += 1

    try:
        with timing.installed(observer), timing.GCMonitor(observer) as gc_monitor:
            previous, last_sent = None, -float("inf")
            while not stop.is_set():
                value = timing.call("inventory.poll", reader.poll, time.time())
                gc_monitor.flush()
                if reader.error:
                    raise ValueError(reader.error)
                key = (
                    reader.offset,
                    reader.backlog_bytes,
                    reader.deferred.get("generation") if reader.deferred else None,
                )
                if key != previous or time.monotonic() - last_sent >= 1:
                    with timing.span("inventory.compact_and_send"):
                        send("OK", value)
                    previous, last_sent = key, time.monotonic()
                stop.wait(interval_s)
        send("STOPPED")
        return 0
    except Exception as error:
        with contextlib.suppress(OSError, ValueError):
            send("ERROR", error=repr(error))
        raise
    finally:
        observer.close()


class ProcessInventory:
    """Cache compact snapshots; the controller never waits for inventory I/O.

    A small receiving thread checks every IPC message and retains only the
    latest parsed snapshot. Raw journal parsing, validation and GC are isolated
    in an exec'd CPU-only process; receipt errors and unexpected exits are fatal.
    """

    def __init__(self, path, pool_id, run_dir, *, command=None):
        if pool_id not in {"tp1", "tp4"}:
            raise ValueError("invalid inventory pool")
        self.path, self.pool_id, self.run_dir = Path(path), pool_id, Path(run_dir)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.closed = False
        self.failure = None
        self.sequence = 0
        self.worker_pid = None
        self.max_message_bytes = 0
        self.acknowledged = False
        self.observed_at = self.latest = self.deferred = self.error = None
        self.generation = None
        self.diagnostic = dict(
            pool_id=pool_id,
            path=str(path),
            error=None,
            error_detail=None,
            deferred=None,
            last_generation=None,
            last_sampled_at=None,
            backlog_bytes=0,
        )
        self.cached = (None, None, self.diagnostic)
        self.timing_path = self.run_dir / f"inventory_{pool_id}_timing.jsonl"
        argv = command or [
            sys.executable,
            "-B",
            str(ROOT / "tools/bridge_tp/observe_pressure_inventory.py"),
            "--inventory",
            str(path),
            "--pool",
            pool_id,
            "--timing",
            str(self.timing_path),
        ]
        with (self.run_dir / f"inventory_{pool_id}_worker.log").open("xb") as log:
            self.process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                cwd=ROOT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        self.thread = threading.Thread(
            target=self._receive, name=f"inventory-cache-{pool_id}", daemon=True
        )
        self.thread.start()

    def _receive(self):
        try:
            while True:
                line = self.process.stdout.readline(MAX_MESSAGE_BYTES + 1)
                if not line:
                    if not self.acknowledged:
                        raise RuntimeError(
                            "inventory process exited without STOPPED receipt"
                        )
                    break
                if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
                    raise ValueError("invalid/oversized inventory IPC frame")
                with timing.span("inventory.ipc_decode"):
                    message = json.loads(line)
                self._accept(message, len(line))
        except Exception as error:
            with self.lock:
                self.failure = repr(error)

    def _accept(self, message, size):
        if self.worker_pid is None:
            # Windows virtualenv python.exe may be a redirector. The private
            # pipe authenticates its worker; Linux exec retains Popen's PID.
            worker_pid = message.get("pid")
            if (
                type(worker_pid) is not int
                or worker_pid <= 0
                or (os.name != "nt" and worker_pid != self.process.pid)
            ):
                raise ValueError("inventory IPC worker PID differs")
            self.worker_pid = worker_pid
        if (
            message["schema"] != SCHEMA
            or message["pool_id"] != self.pool_id
            or message["pid"] != self.worker_pid
            or message["seq"] != self.sequence + 1
        ):
            raise ValueError("inventory IPC identity/sequence differs")
        self.sequence += 1
        self.max_message_bytes = max(self.max_message_bytes, size)
        status = message["status"]
        if status == "ERROR":
            raise RuntimeError(
                "inventory worker: "
                + str(message["error"])
                + "; "
                + repr(message["diagnostic"])
            )
        if status == "STOPPED":
            if not self.stop.is_set():
                raise RuntimeError("inventory worker stopped unexpectedly")
            self.acknowledged = True
            return
        if status != "OK":
            raise ValueError("unknown inventory IPC status")
        value, diagnostic = message["inventory"], message["diagnostic"]
        if diagnostic.get("error"):
            raise RuntimeError("inventory integrity failed: " + repr(diagnostic))
        if value is not None:
            if (
                value["pool_id"] != self.pool_id
                or value["schema"] != "PRESSURE_INVENTORY_V1"
                or self.generation is not None
                and value["generation"] < self.generation
            ):
                raise ValueError("cached inventory identity/generation differs")
            self.generation = value["generation"]
        with self.lock:
            self.cached = (value, message["observed_at"], diagnostic)

    def poll(self, now):
        with self.lock:
            value, observed, diagnostic = self.cached
            failure = self.failure
        if failure or self.process.poll() is not None:
            raise RuntimeError("inventory subprocess failed: " + str(failure))
        self.latest, self.observed_at, self.diagnostic = value, observed, diagnostic
        self.error, self.deferred = diagnostic["error"], diagnostic["deferred"]
        # Never renew an original observation on receipt or a cached read.
        return value if value is None or value["sampled_at"] <= now else None

    def diagnostics(self):
        return dict(
            self.diagnostic,
            observer_pid=self.process.pid,
            observer_mode="isolated_process",
            ipc_messages=self.sequence,
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        forced = False
        try:
            self.process.stdin.close()  # EOF is a graceful stop, no filesystem polling.
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                forced = True
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            self.thread.join(2)
            error = self.failure
            if (
                forced
                or self.process.returncode
                or self.thread.is_alive()
                or not self.acknowledged
            ):
                error = error or "inventory subprocess did not close cleanly"
            timing_receipt = None
            if not error:
                try:
                    timing_receipt = timing.verify(self.timing_path)
                except (OSError, ValueError, RuntimeError) as cause:
                    error = repr(cause)
            receipt = dict(
                status="FAILED" if error else "COMPLETED",
                pid=self.process.pid,
                worker_pid=self.worker_pid,
                pool_id=self.pool_id,
                parent_pid=os.getpid(),
                returncode=self.process.returncode,
                forced=forced,
                receiver_stopped=not self.thread.is_alive(),
                acknowledged=self.acknowledged,
                ipc_messages=self.sequence,
                max_message_bytes=self.max_message_bytes,
                last_generation=self.generation,
                error=error,
                timing=timing_receipt,
            )
            (
                self.run_dir / f"inventory_{self.pool_id}_process_receipt.json"
            ).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            if error:
                raise RuntimeError("inventory subprocess shutdown failed: " + error)
        finally:
            self.process.stdout.close()
