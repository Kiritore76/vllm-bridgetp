# SPDX-License-Identifier: Apache-2.0
"""Experimental real streaming ingress backed by the pressure reservation ledger."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

from .online_io import ProxyRecorder, post_streaming_completion
from .response_proxy import ProxyMode
from .sampling_contract import freeze_strict_greedy_sampling


class ObservedRecorder(ProxyRecorder):
    def __init__(self, external_id, service, pool_id, clock, emission_sink=None):
        super().__init__(external_id, ProxyMode.HOLD_BACK, emission_sink=emission_sink)
        self.service, self.pool_id, self.clock = service, pool_id, clock
        self.service_id = external_id
        self.target_service_id = external_id + ":target"
        self.target_arrived = False

    def on_source_token(self, index, token_id, unix_s):
        self.service.token(self.service_id, unix_s)
        super().on_source_token(index, token_id, unix_s)

    def on_target_token(self, index, token_id, unix_s):
        if not self.target_arrived:
            self.service.arrive(self.target_service_id, "tp4", unix_s)
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


class PressureIngress:
    """Reserve before HTTP submission and bind only published server inventory.

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
        post=post_streaming_completion,
        clock=time.time,
    ):
        self.coordinator, self.service, self.executor = coordinator, service, executor
        self.urls = {"tp1": source_url, "tp4": target_url}
        self.timeout_s, self.post, self.clock = timeout_s, post, clock
        self.streams = {}
        self._lock = threading.RLock()

    def submit(
        self,
        request: dict,
        pools,
        decode_costs: dict,
        *,
        emission_sink=None,
        allowed_pool_id=None,
    ) -> IngressStream:
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
            admission = self.coordinator.admission(
                client_id,
                len(prompt),
                int(payload["max_tokens"]),
                pools,
                now=now,
                key=key,
                decode_costs=decode_costs,
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
            self.streams[client_id] = value

            def consume():
                try:
                    return self.post(
                        self.urls[value.pool_id],
                        value.payload,
                        self.timeout_s,
                        recorder.on_source_token,
                    )
                finally:
                    self.service.finish(client_id)

            # An exception may happen after a transport accepted work; keep the
            # reservation until independently confirmed completion/cleanup.
            value.future = self.executor.submit(consume)
            return value

    def reconcile(self, inventory: dict, now: float) -> None:
        cfg = self.coordinator.policy.cfg
        if not 0 <= now - inventory["sampled_at"] <= cfg.max_age_s:
            raise ValueError("cannot reconcile stale/future inventory")
        with self._lock, self.coordinator._lock:
            for value in self.streams.values():
                if value.pool_id != inventory["pool_id"]:
                    continue
                entry = self.coordinator.ledger.entries[value.reservation_key]
                if entry.state in {"MOVED", "COMPLETED", "CANCELLED", "FAILED"}:
                    continue
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
                    if row["owned_blocks"] is not None:
                        allocated = min(row["owned_blocks"], entry.blocks)
                        if allocated < entry.allocated_blocks:
                            # Preemption/recompute can physically free allocation:
                            # restore its FUTURE commitment instead of losing it.
                            entry.allocated_blocks = allocated
                            if entry.state == "ACCOUNTED":
                                entry.state = "COMMITTED"
                        visible = entry.service_observed or row["generated"] >= 2
                        if entry.state != "ACCOUNTED":
                            self.coordinator.ledger.observe(
                                entry.key,
                                allocated_blocks=allocated,
                                service_observed=visible,
                                prefill_observed=True,
                            )
                    if entry.state == "ACTIVE":
                        self.coordinator.ledger.settle(
                            entry.key, "COMMITTED", evidence=True
                        )
                elif (
                    value.server_id
                    and value.future is not None
                    and value.future.done()
                    and value.future.exception() is None
                    and inventory["sampled_at"]
                    > value.future.result()["completed_unix_s"]
                ):
                    self.coordinator.ledger.retire_admission(
                        entry.key, "COMPLETED", evidence=True
                    )
                self.coordinator.audit(
                    {
                        "kind": "pressure_ingress_reconcile",
                        "unix_s": now,
                        "client_id": value.client_id,
                        "server_id": value.server_id,
                        "inventory_generation": inventory["generation"],
                        "reservation": dict(vars(entry)),
                    }
                )

    def for_server_id(self, request_id: str) -> IngressStream:
        matches = [v for v in self.streams.values() if v.server_id == request_id]
        if len(matches) != 1:
            raise ValueError("selected request is not bound to one existing stream")
        return matches[0]

    def refuse_restart_if_active(self):
        """First pilot does not recover ownership from an in-memory ledger."""
        if any(
            r.state in {"ACTIVE", "COMMITTED", "ACCOUNTED"}
            for r in self.coordinator.ledger.entries.values()
        ):
            raise RuntimeError(
                "active commitments require reconciliation before restart"
            )
