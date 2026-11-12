# SPDX-License-Identifier: Apache-2.0
"""Drain recompute: finish a merge without waiting for its longest requests.

A TP1 group that merges into one TP2 cannot hand its last long requests to
another instance when no retained instance has KV room (six-card guard40
11-10: 236 drain assists refused, drain 310 s behind three long requests).
Recompute ends those requests on the source and continues them on the new
instance with ``prompt + visible output`` as the prompt:

1. ``ABORTING``: the controller closes the source HTTP stream; vLLM aborts the
   request on client disconnect. The visible stream stops at index K.
2. ``SOURCE_ENDED``: the source thread returned; K is final, because the
   source reader is the only writer of source tokens.
3. ``PARKED``: a source inventory sampled after the end no longer lists the
   request, so its admission is retired as ``MOVED`` (source KV released).
4. ``RESUMED``: a new admission on the target was accepted; the same stream
   thread posts the continuation and the recorder maps its token ``i`` to the
   global index ``K + i`` (cutover K, committed: late source tokens are void).
5. ``COMPLETED`` or ``FAILED``.

A source that finished on its own before the abort took effect is
``NATURAL``: its result stands and nothing is resubmitted. Sampling is the
pilot's strict greedy contract, so the continuation equals the uninterrupted
output in exact arithmetic; the run does not claim bitwise equality.
"""

from __future__ import annotations

import socket
import threading
import time

PARKED_OWNER = "recompute-parked"
TERMINAL = frozenset({"NATURAL", "COMPLETED", "FAILED"})


def close_response(response):
    """End a blocking SSE read from another thread (client disconnect)."""
    raw = getattr(getattr(response, "fp", None), "raw", None)
    sock = getattr(raw, "_sock", None)
    try:
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
        else:
            response.close()
    except OSError:
        pass


def continuation_payload(payload, output_ids):
    """Same request and stop contract, resumed after ``output_ids``."""
    cap = int(payload["max_tokens"])
    if not 0 <= len(output_ids) < cap:
        raise ValueError("continuation needs at least one remaining token")
    resumed = dict(payload)
    resumed["prompt"] = list(payload["prompt"]) + [int(t) for t in output_ids]
    resumed["max_tokens"] = cap - len(output_ids)
    return resumed


class RecomputeJob:
    """One request's recompute, shared by the controller and its stream thread."""

    def __init__(self, seq, stream, source_pool, *, clock=time.time):
        self.seq, self.stream, self.source_pool = seq, stream, source_pool
        self.clock = clock
        self.client_id = stream.client_id
        self.source_server_id = stream.server_id
        self.state = "ABORTING"
        self.requested_at = clock()
        self.source_ended_at = None
        self.source_error = None
        self.cutover = None
        self.parked_at = None
        self.resumed_at = None
        self.completed_at = None
        self.target_pool = None
        self.target_url = None
        self.payload = None
        self.reason = None
        self._lock = threading.Lock()
        self._resume = threading.Event()

    # ---- stream thread --------------------------------------------------
    def finish_source(self, result, error, started_unix_s, post, timeout_s):
        """Called by the stream thread once the source call returned or raised.

        Returns the stream's final transport result (blocks while parked).
        """
        recorder = self.stream.recorder
        emitted = recorder.progress()[0]
        cap = int(self.stream.payload["max_tokens"])
        with self._lock:
            self.source_ended_at = self.clock()
            self.source_error = None if error is None else repr(error)
            if result is not None and emitted == cap:
                self.state = "NATURAL"
                self.reason = "source completed before the abort took effect"
                self._resume.set()
                return result
            if result is not None and result.get("finish_reason") == "stop":
                self.state = "NATURAL"
                self.reason = "source stopped naturally before the abort"
                self._resume.set()
                return result
            self.cutover = emitted
            self.state = "SOURCE_ENDED"
        source_first = recorder.proxy.emitted[0].unix_s if emitted else None
        while not self._resume.wait(1.0):
            pass
        if self.state == "FAILED":
            raise RuntimeError(f"recompute of {self.client_id} failed: {self.reason}")
        try:
            target = post(
                self.target_url,
                self.payload,
                timeout_s,
                recorder.on_target_token,
            )
        except Exception as failure:
            with self._lock:
                self.state, self.reason = "FAILED", repr(failure)
            raise
        with self._lock:
            self.state, self.completed_at = "COMPLETED", self.clock()
        first = source_first
        if first is None and recorder.progress()[0]:
            first = recorder.proxy.emitted[0].unix_s
        completed = target["completed_unix_s"]
        return {
            "response_id": target.get("response_id"),
            "finish_reason": target.get("finish_reason"),
            "request_started_unix_s": started_unix_s,
            "first_token_unix_s": first,
            "completed_unix_s": completed,
            "ttft_ms": None if first is None else (first - started_unix_s) * 1000,
            "e2e_ms": (completed - started_unix_s) * 1000,
            "chunks": [],
            "chunks_retained": False,
            "recompute": self.summary(),
        }

    # ---- controller -----------------------------------------------------
    def park(self, now):
        with self._lock:
            if self.state != "SOURCE_ENDED":
                raise ValueError("only an ended source can be parked")
            self.state, self.parked_at = "PARKED", now

    def resume(self, pool_id, url, payload, now):
        with self._lock:
            if self.state != "PARKED":
                raise ValueError("only a parked request can resume")
            self.target_pool, self.target_url, self.payload = pool_id, url, payload
            self.state, self.resumed_at = "RESUMED", now
        self._resume.set()

    def fail(self, reason):
        with self._lock:
            if self.state in TERMINAL:
                return
            self.state, self.reason = "FAILED", reason
        self._resume.set()

    def summary(self):
        return dict(
            seq=self.seq,
            client_id=self.client_id,
            source_server_id=self.source_server_id,
            source_pool=self.source_pool,
            state=self.state,
            cutover=self.cutover,
            requested_at=self.requested_at,
            source_ended_at=self.source_ended_at,
            parked_at=self.parked_at,
            resumed_at=self.resumed_at,
            completed_at=self.completed_at,
            target_pool=self.target_pool,
            source_error=self.source_error,
            reason=self.reason,
        )


class DrainRecompute:
    """Controller side: start, account and resume requested recomputes.

    ``target_of(bundle, known)`` names the ACTIVE pool that should take the
    continuation: the bundle's new instance (absent from ``known``, the pools
    that existed at the request). After ``fallback_after_s`` parked, any ACTIVE
    pool the scheduler admits it to is used instead, so a failed conversion
    never strands a request.
    """

    def __init__(
        self,
        *,
        ingress,
        coordinator,
        audit,
        tp_of,
        target_of,
        clock=time.time,
        fallback_after_s=180.0,
    ):
        self.ingress, self.co, self.audit = ingress, coordinator, audit
        self.tp_of, self.target_of, self.clock = tp_of, target_of, clock
        self.fallback_after_s = fallback_after_s
        self.jobs = {}
        self.requests = {}  # seq -> dict(bundle, known, result)
        self.last_seq = 0

    def request(self, control):
        """Apply one recompute request from the layout file (idempotent)."""
        seq = int(control["seq"])
        if seq <= self.last_seq:
            return self.requests.get(seq, {}).get("result")
        self.last_seq = seq
        rows = control["requests"]
        result = dict(seq=seq, state="STARTED", requests=[], reason=None)
        streams, refusal = [], None
        for row in rows:
            try:
                stream = self.ingress.for_server_id(row["request_id"])
            except ValueError:
                refusal = f"{row['request_id']} has no unique live stream"
                break
            owner = self.co._owners.get(stream.server_id, stream.pool_id)
            if (
                stream.future is None
                or stream.future.done()
                or getattr(stream, "recompute", None) is not None
                or stream.recorder.proxy.cutover_index is not None
                or stream.recorder.proxy.committed
            ):
                refusal = f"{row['request_id']} ended, migrating or recomputed"
                break
            if self.co.lifecycle(owner) != "QUIESCING":
                refusal = f"{row['request_id']} owner {owner} is not quiescing"
                break
            if any(
                r.kind == "migration" and r.request_id == stream.server_id
                for r in self.co.ledger.live_rows()
            ):
                refusal = f"{row['request_id']} has a live migration reservation"
                break
            streams.append((stream, owner))
        if refusal is not None or not streams:
            result.update(state="REFUSED", reason=refusal or "no requests")
        else:
            for stream, owner in streams:
                job = RecomputeJob(seq, stream, owner, clock=self.clock)
                self.jobs[stream.client_id] = job
                stream.recompute = job
                with self.co._lock:
                    self.co._owners[stream.server_id] = PARKED_OWNER
                stream.abort_source()
                result["requests"].append(stream.client_id)
        self.requests[seq] = dict(
            bundle=control.get("bundle"),
            # Pools serving now. A sleeping engine of the bundle that wakes
            # later keeps its id and must still count as the new instance.
            known=frozenset(
                i for i in self.tp_of if self.co.lifecycle(i) != "SLEEPING"
            ),
            result=result,
        )
        self.audit(dict(kind="drain_recompute_request", control=control, **result))
        return result

    def tick(self, now, pools, inventories):
        """Advance every job one step; ``inventories`` maps pool id -> raw."""
        readings = {p.pool_id: p for p in pools}
        for job in tuple(self.jobs.values()):
            before = job.state
            if job.state == "SOURCE_ENDED":
                self._settle_source(job, inventories, now)
            if job.state == "PARKED":
                self._resume(job, readings, now)
            if job.state != before:
                self.audit(dict(kind="drain_recompute_state", **job.summary()))

    def _settle_source(self, job, inventories, now):
        raw = inventories.get(job.source_pool)
        stream = job.stream
        if raw is None:
            # The source was retired: its inventory ended with it, and the
            # retirement itself required no unsettled admission there.
            if self.co.lifecycle(job.source_pool) != "RETIRED":
                return
        else:
            # The request was live on the source when its abort was requested,
            # so any later sample without it proves the engine released it.
            # Native inventory is event driven: an engine left idle by the
            # abort publishes once, possibly before the source stream thread
            # records its end (GPU 11-12: waiting for a sample after
            # source_ended_at kept a drain open forever).
            prefix = f"cmpl-{job.client_id}-0-"
            if raw["sampled_at"] <= job.requested_at or any(
                r["request_id"].startswith(prefix) for r in raw["requests"]
            ):
                return
        with self.co._lock:
            entry = self.co.ledger.entries[stream.reservation_key]
            if entry.state in {"ACTIVE", "COMMITTED", "ACCOUNTED"}:
                self.co.ledger.retire_admission(entry.key, "MOVED", evidence=True)
        job.park(now)

    def _resume(self, job, readings, now):
        request = self.requests[job.seq]
        target = self.target_of(request["bundle"], request["known"])
        # A conversion that aborted reopens its sources: no new instance comes.
        fallback = (
            now - job.parked_at >= self.fallback_after_s
            or self.co.lifecycle(job.source_pool) == "ACTIVE"
        )
        if target is None and not fallback:
            return
        allowed = target if target is not None and not fallback else None
        candidates = [
            r
            for pool_id, r in readings.items()
            if self.co.lifecycle(pool_id) == "ACTIVE"
            and allowed in (None, pool_id)
        ]
        if not candidates:
            return
        stream = job.stream
        output = [t.token_id for t in stream.recorder.proxy.emitted]
        if len(output) != job.cutover:
            job.fail("visible output changed after the source ended")
            return
        payload = continuation_payload(stream.payload, output)
        key = "recompute-" + stream.client_id
        try:
            # The ledger tracks each identity once; the continuation's server
            # request (same client id) rebinds this row on reconcile.
            admission = self.co.admission_multi(
                stream.client_id + ":recompute",
                len(payload["prompt"]),
                int(payload["max_tokens"]),
                candidates,
                self.tp_of,
                now,
                key,
                allowed_pool_id=allowed,
            )
        except ValueError as refusal:  # RoutingError / AdmissionDeferred
            if getattr(job, "_last_refusal", None) != str(refusal):
                job._last_refusal = str(refusal)
                self.audit(
                    dict(
                        kind="drain_recompute_deferred",
                        client_id=stream.client_id,
                        reason=str(refusal),
                    )
                )
            return
        if admission.max_output_tokens != payload["max_tokens"]:
            job.fail("admission changed the continuation's stop contract")
            return
        self.ingress.rebind(stream, admission.pool_id, key, payload)
        with self.co._lock:
            # Drain evidence counts an unbound stream under its client id.
            self.co._owners[stream.client_id] = admission.pool_id
        recorder = stream.recorder
        recorder.target_pool = admission.pool_id
        recorder.set_cutover(job.cutover, now)
        recorder.on_commit(now)
        job.resume(
            admission.pool_id, self.ingress.urls[admission.pool_id], payload, now
        )

    def snapshot(self):
        return dict(
            last_seq=self.last_seq,
            results={
                str(seq): dict(
                    value["result"],
                    jobs=[
                        j.summary() for j in self.jobs.values() if j.seq == seq
                    ],
                )
                for seq, value in self.requests.items()
            },
        )
