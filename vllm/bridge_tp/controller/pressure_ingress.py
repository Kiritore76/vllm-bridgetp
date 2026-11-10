# SPDX-License-Identifier: Apache-2.0
"""Route streaming requests; observe engine queueing without ingress KV promises."""

from __future__ import annotations

import inspect
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import partial

from .online_io import ProxyRecorder, post_streaming_completion
from .pressure_recompute import close_response
from .response_proxy import ProxyMode
from .sampling_contract import freeze_strict_greedy_sampling


class ObservedRecorder(ProxyRecorder):
    def __init__(self, external_id, service, pool_id, clock, emission_sink=None):
        super().__init__(
            external_id,
            ProxyMode.HOLD_BACK,
            emission_sink=emission_sink,
            compact_history=True,
        )
        self.service, self.pool_id, self.clock = service, pool_id, clock
        self.service_id = external_id
        self.target_service_id = external_id + ":target"
        self.target_arrived = False
        self.target_pool = "tp4"
        self.tokens_seen = 0

    def on_source_token(self, index, token_id, unix_s):
        self.tokens_seen += 1
        self.service.token(self.service_id, unix_s)
        super().on_source_token(index, token_id, unix_s)

    def on_target_token(self, index, token_id, unix_s):
        self.tokens_seen += 1
        if not self.target_arrived:
            self.service.arrive(self.target_service_id, self.target_pool, unix_s)
            self.target_arrived = True
        self.service.token(self.target_service_id, unix_s)
        super().on_target_token(index, token_id, unix_s)


@dataclass
class IngressStream:
    client_id: str
    pool_id: str
    payload: dict
    reservation_key: str
    recorder: ObservedRecorder
    dispatched_at: float
    future: object = None
    server_id: str | None = None
    dispatch_entered_at: float | None = None
    admission_finished_at: float | None = None
    executor_submitted_at: float | None = None
    http_thread_started_at: float | None = None
    held_s: float = 0.0
    # Drain recompute (pressure_recompute): the job, the client's original
    # request/pool (export keeps them), and the source response to close.
    recompute: object = None
    original_payload: dict | None = None
    initial_pool_id: str | None = None
    _response: object = field(default=None, repr=False)
    _abort_requested: bool = field(default=False, repr=False)
    _abort_lock: object = field(default_factory=threading.Lock, repr=False)

    def bind_response(self, response):
        with self._abort_lock:
            self._response = response
            abort = self._abort_requested
        if abort:
            close_response(response)

    def abort_source(self):
        """Close the source stream; vLLM aborts on client disconnect."""
        with self._abort_lock:
            self._abort_requested = True
            response = self._response
        if response is not None:
            close_response(response)


class PressureIngress:
    """Route HTTP submission and bind only published server inventory.

    The caller owns executor shutdown and audits final stream/proxy results.
    Requests use token-ID prompts and n=1; arbitrary batching/beam/multimodal
    inputs are outside this pilot ingress. No source stream is relaunched when
    its request is selected for migration.
    """

    def __init__(
        self,
        coordinator,
        service,
        executor,
        source_url,
        target_url,
        timeout_s=1800,
        *,
        post=None,
        clock=time.time,
        reconcile_log_interval_s=0,
        tp_of=None,
        pair_ids=None,
    ):
        self.coordinator, self.service, self.executor = coordinator, service, executor
        pair_ids = pair_ids or {"tp1": "tp1", "tp4": "tp4"}
        self.urls = {pair_ids["tp1"]: source_url, pair_ids["tp4"]: target_url}
        # With TP degrees per pool id, route through the multi-instance entry.
        self.tp_of = tp_of
        self.timeout_s, self.clock = timeout_s, clock
        # Complete visible tokens already live in the recorder and journal.
        # Keeping nested SSE chunk dicts in every Future duplicates that data.
        self.post = (
            partial(post_streaming_completion, retain_chunks=False)
            if post is None
            else post
        )
        try:
            self._post_takes_response = (
                "on_response" in inspect.signature(self.post).parameters
            )
        except (TypeError, ValueError):
            self._post_takes_response = False
        self.streams = {}
        self._open_streams = {}  # what reconcile still has to look at
        self._lock = threading.RLock()
        self.reconcile_log_interval_s = reconcile_log_interval_s
        self._reconcile_logged_at = {}
        self._reconcile_suppressed = {}

    def submit(
        self,
        request: dict,
        pools,
        *,
        emission_sink=None,
        allowed_pool_id=None,
    ) -> IngressStream:
        entered = self.clock()
        prompt = request.get("prompt")
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(v) is not int or v < 0 for v in prompt)
            or request.get("n", 1) != 1
        ):
            raise ValueError("pilot ingress requires one token-ID prompt and n=1")
        payload = freeze_strict_greedy_sampling(dict(request))
        client_id = "pressure-" + uuid.uuid4().hex
        payload.update(request_id=client_id, stream=True, return_token_ids=True)
        payload.setdefault("ignore_eos", False)
        key = "admission-" + client_id
        with self._lock:
            now = self.clock()
            if self.tp_of is None:
                admission = self.coordinator.admission(
                    client_id,
                    len(prompt),
                    int(payload["max_tokens"]),
                    pools,
                    now=now,
                    key=key,
                    allowed_pool_id=allowed_pool_id,
                )
            else:
                admission = self.coordinator.admission_multi(
                    client_id,
                    len(prompt),
                    int(payload["max_tokens"]),
                    list(pools),
                    self.tp_of,
                    now,
                    key,
                    allowed_pool_id=allowed_pool_id,
                )
            payload["max_tokens"] = admission.max_output_tokens
            recorder = ObservedRecorder(
                client_id, self.service, admission.pool_id, self.clock, emission_sink
            )
            self.service.arrive(client_id, admission.pool_id, now)
            value = IngressStream(
                client_id, admission.pool_id, payload, key, recorder, now
            )
            value.dispatch_entered_at = entered
            value.admission_finished_at = self.clock()
            self.streams[client_id] = value
            self._open_streams[client_id] = value

            def consume():
                value.http_thread_started_at = self.clock()
                started_unix_s = time.time()
                options = (
                    dict(on_response=value.bind_response)
                    if self._post_takes_response
                    else {}
                )
                result = error = None
                try:
                    result = self.post(
                        self.urls[value.pool_id],
                        value.payload,
                        self.timeout_s,
                        recorder.on_source_token,
                        **options,
                    )
                except Exception as failure:
                    # Only a recompute's deliberate disconnect is not an error.
                    if value.recompute is None:
                        raise
                    error = failure
                finally:
                    self.service.finish(client_id)
                if value.recompute is None:
                    return result
                try:
                    return value.recompute.finish_source(
                        result, error, started_unix_s, self.post, self.timeout_s
                    )
                finally:
                    self.service.finish(recorder.target_service_id)

            # A transport error can follow acceptance. Retain identity tracking
            # until completion/cleanup; this record reserves no KV capacity.
            value.executor_submitted_at = self.clock()
            value.future = self.executor.submit(consume)
            return value

    def reconcile(self, inventory: dict, now: float) -> None:
        cfg = self.coordinator.policy.cfg
        if not 0 <= now - inventory["sampled_at"] <= cfg.max_age_s:
            raise ValueError("cannot reconcile stale/future inventory")
        records = []
        with self._lock, self.coordinator._lock:
            for client_id, value in tuple(self._open_streams.items()):
                entry = self.coordinator.ledger.entries[value.reservation_key]
                if entry.state in {"MOVED", "COMPLETED", "CANCELLED", "FAILED"}:
                    del self._open_streams[client_id]
                    continue
                if value.pool_id != inventory["pool_id"]:
                    continue
                before = (value.server_id, dict(vars(entry)))
                # vLLM completion creates cmpl-<client-id>-<prompt-index>-<suffix>.
                # Random client IDs prevent prefix collisions; require one match
                # and verify its original length/stop contract before rebinding.
                prefix = f"cmpl-{value.client_id}-0-"
                matches = [
                    r
                    for r in inventory["requests"]
                    if r["request_id"].startswith(prefix)
                ]
                if len(matches) > 1:
                    raise ValueError("ambiguous server request identity")
                if matches:
                    row = matches[0]
                    if (
                        row["max_output_tokens"] != value.payload["max_tokens"]
                        or row["prompt_tokens"] != len(value.payload["prompt"])
                        or row["ignore_eos"] != value.payload["ignore_eos"]
                    ):
                        raise ValueError("server request stop contract differs")
                    if value.server_id not in {None, row["request_id"]}:
                        raise ValueError("server request identity changed")
                    value.server_id = row["request_id"]
                    if entry.request_id != value.server_id:
                        self.coordinator._owners.pop(entry.request_id, None)
                        entry.request_id = value.server_id
                        self.coordinator._owners[value.server_id] = value.pool_id
                    # Waiting and running requests both appear in inventory.
                    # Native pending_prompt_tokens now owns this demand; stop
                    # adding the unpublished prompt count. There is no ingress
                    # block or decode promise to reconcile during preemption.
                    if entry.state != "ACCOUNTED":
                        self.coordinator.ledger.observe(
                            entry.key,
                            allocated_blocks=0,
                            service_observed=True,
                            prefill_observed=True,
                        )
                    if entry.state == "ACTIVE":
                        self.coordinator.ledger.settle(
                            entry.key, "COMMITTED", evidence=True
                        )
                elif (
                    value.future is not None
                    and value.future.done()
                    and value.future.exception() is None
                    and inventory["sampled_at"]
                    > value.future.result()["completed_unix_s"]
                ):
                    self.coordinator.ledger.retire_admission(
                        entry.key, "COMPLETED", evidence=True
                    )
                changed = before != (value.server_id, dict(vars(entry)))
                if (
                    not changed
                    and now
                    - self._reconcile_logged_at.get(value.client_id, -float("inf"))
                    < self.reconcile_log_interval_s
                ):
                    self._reconcile_suppressed[value.client_id] = (
                        self._reconcile_suppressed.get(value.client_id, 0) + 1
                    )
                    continue
                self._reconcile_logged_at[value.client_id] = now
                records.append(
                    {
                        "kind": "pressure_ingress_reconcile",
                        "unix_s": now,
                        "client_id": value.client_id,
                        "server_id": value.server_id,
                        "inventory_generation": inventory["generation"],
                        "state_changed": changed,
                        "suppressed_unchanged": self._reconcile_suppressed.pop(
                            value.client_id, 0
                        ),
                        "reservation": dict(vars(entry)),
                    }
                )
        for record in records:
            self.coordinator.audit(record)

    def rebind(self, value, pool_id, key, payload):
        """Point a parked stream at its continuation's admission and engine.

        Reconcile then matches the continuation's server request (same client
        id) against ``payload``; the export keeps the original request/pool.
        """
        with self._lock:
            if value.original_payload is None:
                value.original_payload = value.payload
                value.initial_pool_id = value.pool_id
            value.payload = payload
            value.pool_id = pool_id
            value.reservation_key = key
            value.server_id = None
            self._open_streams[value.client_id] = value

    def for_server_id(self, request_id: str) -> IngressStream:
        with self._lock:
            matches = [
                v for v in self._open_streams.values() if v.server_id == request_id
            ] or [v for v in self.streams.values() if v.server_id == request_id]
        if len(matches) != 1:
            raise ValueError("selected request is not bound to one existing stream")
        return matches[0]

    def refuse_restart_if_active(self):
        """First pilot does not recover ownership from an in-memory ledger."""
        if any(
            r.state in {"ACTIVE", "COMMITTED", "ACCOUNTED"}
            for r in self.coordinator.ledger.live_rows()
        ):
            raise RuntimeError(
                "active commitments require reconciliation before restart"
            )
