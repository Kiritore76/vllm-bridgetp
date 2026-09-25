# SPDX-License-Identifier: Apache-2.0
"""Adapter from policy actions to the validated Phase 7/8 mechanism.

This module contains NO KV tensor logic. Everything it does is either writing
a runtime control block or POSTing to an endpoint that Phase 7/8 already
validated. Keeping that boundary is what lets Phase 9 inherit the Phase 5-8
correctness evidence instead of re-earning it.

Endpoints (from ``vllm/bridge_tp/takeover_api.py``):

    POST {source}/bridge_tp/v1/takeover  {"action": "commit"|"rollback", ...}
    POST {source}/bridge_tp/v1/cleanup   {"reason": ..., "abort_source": bool}

Both require the session-binding triple, which the server cross-checks against
``session_manifest.json`` and refuses with 403 if it does not match:

    migration_id, session_token, source_request_id

Readiness is evidence on disk, not a return value: the four sender receipts in
``stage_delivery_receipts/`` (Phase 8) or ``sender_receipts/`` (Phase 7) plus
the four receiver receipts under ``receiver_receipts/<target_request_id>/``.
``poll_target_ready`` reproduces exactly the checks the server performs in
``_validate_target_ready`` so the controller never issues a commit that the
server will reject.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..runtime_control import RuntimeControl


class ActionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionBinding:
    run_dir: Path
    migration_id: str
    session_token: str
    source_request_id: str

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> SessionBinding:
        run = Path(run_dir)
        session = json.loads(
            (run / "session_manifest.json").read_text(encoding="utf-8")
        )
        return cls(
            run_dir=run,
            migration_id=str(session["migration_id"]),
            session_token=str(session["session_token"]),
            source_request_id=str(session["source_request_id"]),
        )

    def body(self, **extra: Any) -> dict[str, Any]:
        return {
            "migration_id": self.migration_id,
            "session_token": self.session_token,
            "source_request_id": self.source_request_id,
            **extra,
        }


def _post(url: str, payload: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:  # pragma: no cover - network path
        detail = error.read().decode("utf-8", errors="replace")
        raise ActionError(f"{url} -> HTTP {error.code}: {detail}") from error
    except OSError as error:  # pragma: no cover - network path
        raise ActionError(f"{url} unreachable: {error}") from error


class ActionAdapter:
    """The only component allowed to mutate migration state on the servers."""

    def __init__(
        self,
        source_url: str,
        run_dir: str | Path | SessionBinding,
        expected_migration_id: str | None = None,
        target_url: str | None = None,
        ready_notification_mode: str = "FILE_POLL",
        ready_notification_host: str = "127.0.0.1",
        ready_notification_port: int = 0,
        ready_latch_poll_ms: float = 5.0,
    ) -> None:
        self.source_url = source_url.rstrip("/")
        self.target_url = target_url.rstrip("/") if target_url else None
        self.expected_migration_id = expected_migration_id
        if isinstance(run_dir, SessionBinding):
            self.run_dir = run_dir.run_dir
            self._binding: SessionBinding | None = run_dir
        else:
            self.run_dir = Path(run_dir)
            self._binding = None
        self.ready_notification_mode = ready_notification_mode.upper()
        if self.ready_notification_mode not in {"FILE_POLL", "UDP"}:
            raise ValueError(
                "ready_notification_mode must be FILE_POLL or UDP"
            )
        self._ready_socket: socket.socket | None = None
        self._ready_notification_ranks: set[int] = set()
        self._ready_notification_count = 0
        self._last_ready_notification: dict[str, Any] | None = None
        if ready_latch_poll_ms < 0:
            raise ValueError("ready_latch_poll_ms cannot be negative")
        self.ready_latch_poll_s = ready_latch_poll_ms / 1000
        self._ready_latch_armed_unix_s: float | None = None
        self._ready_latch_poll_count = 0
        self._authoritative_ready_unix_s: float | None = None
        if self.ready_notification_mode == "UDP":
            if not 0 < ready_notification_port <= 65535:
                raise ValueError("UDP ready notification port is invalid")
            ready_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            ready_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            ready_socket.bind((ready_notification_host, ready_notification_port))
            self._ready_socket = ready_socket

    @property
    def binding(self) -> SessionBinding:
        if self._binding is None:
            try:
                self._binding = SessionBinding.from_run_dir(self.run_dir)
            except (OSError, KeyError, TypeError, ValueError) as error:
                raise ActionError(
                    "migration session is not bound yet; session_manifest.json "
                    "has not been published or is invalid"
                ) from error
        return self._binding

    def refresh_binding(self) -> SessionBinding | None:
        """Bind after the dynamically armed source publishes its manifest."""
        if self._binding is not None:
            return self._binding
        try:
            self._binding = SessionBinding.from_run_dir(self.run_dir)
        except (OSError, KeyError, TypeError, ValueError):
            return None
        if (
            self.expected_migration_id
            and self._binding.migration_id != self.expected_migration_id
        ):
            actual = self._binding.migration_id
            self._binding = None
            raise ActionError(
                "session migration ID differs from controller: "
                f"{actual} != {self.expected_migration_id}"
            )
        return self._binding

    def wait_for_preparing_binding(
        self,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.02,
    ) -> SessionBinding | None:
        """Wait until a dynamically armed snapshot can accept cleanup.

        Snapshot preparation is synchronous inside the source engine.  A
        capacity CLEAR can therefore arrive after the trigger was honored but
        before ``session_manifest.json`` and the PREPARING takeover state are
        published.  Cleanup is safe only after both files describe the same
        migration.
        """
        if timeout_s < 0 or poll_interval_s <= 0:
            raise ValueError("binding wait durations must be positive")
        deadline = time.monotonic() + timeout_s
        while True:
            binding = self.refresh_binding()
            takeover = self.read_takeover_state()
            if (
                binding is not None
                and takeover is not None
                and takeover.get("state") == "PREPARING"
                and takeover.get("migration_id") == binding.migration_id
                and takeover.get("source_request_id") == binding.source_request_id
            ):
                return binding
            if time.monotonic() >= deadline:
                return None
            time.sleep(poll_interval_s)

    # ---- actuation via the runtime control block -----------------------
    def arm_shadow(
        self,
        trigger_output_tokens: int,
        rate_gib_s: float,
        cutover_output_tokens: int | None = None,
        note: str = "",
    ) -> RuntimeControl:
        """Enter Shadow: publish the trigger boundary and the initial rate.

        Requires the P9-0 patch described in ``vllm/bridge_tp/runtime_control.py``;
        without it the server keeps the env-frozen boundary and this call has no
        observable effect.
        """
        current = RuntimeControl.load(self.run_dir) or RuntimeControl()
        return RuntimeControl(
            generation=current.generation,
            armed=True,
            trigger_output_tokens=int(trigger_output_tokens),
            cutover_output_tokens=(
                int(cutover_output_tokens)
                if cutover_output_tokens is not None
                else current.cutover_output_tokens
            ),
            rate_gib_s=float(rate_gib_s),
            target_request_admitted=current.target_request_admitted,
            migration_id=current.migration_id,
            source_request_id_prefix=current.source_request_id_prefix,
            note=note or "shadow armed by Phase 9 controller",
        ).write(self.run_dir)

    def set_rate(self, rate_gib_s: float, note: str = "") -> RuntimeControl:
        current = RuntimeControl.load(self.run_dir) or RuntimeControl()
        return RuntimeControl(
            generation=current.generation,
            armed=current.armed,
            trigger_output_tokens=current.trigger_output_tokens,
            cutover_output_tokens=current.cutover_output_tokens,
            rate_gib_s=float(rate_gib_s),
            target_request_admitted=current.target_request_admitted,
            migration_id=current.migration_id,
            source_request_id_prefix=current.source_request_id_prefix,
            note=note or "rate update",
        ).write(self.run_dir)

    def set_cutover(self, cutover_output_tokens: int, note: str = "") -> RuntimeControl:
        current = RuntimeControl.load(self.run_dir) or RuntimeControl()
        if (
            current.trigger_output_tokens is not None
            and cutover_output_tokens <= current.trigger_output_tokens
        ):
            raise ActionError(
                f"cutover {cutover_output_tokens} must be after trigger "
                f"{current.trigger_output_tokens}"
            )
        return RuntimeControl(
            generation=current.generation,
            armed=current.armed,
            trigger_output_tokens=current.trigger_output_tokens,
            cutover_output_tokens=int(cutover_output_tokens),
            rate_gib_s=current.rate_gib_s,
            target_request_admitted=current.target_request_admitted,
            migration_id=current.migration_id,
            source_request_id_prefix=current.source_request_id_prefix,
            note=note or "cutover boundary set",
        ).write(self.run_dir)

    def mark_target_request_admitted(self, note: str = "") -> RuntimeControl:
        current = RuntimeControl.load(self.run_dir) or RuntimeControl()
        return RuntimeControl(
            generation=current.generation,
            armed=current.armed,
            trigger_output_tokens=current.trigger_output_tokens,
            cutover_output_tokens=current.cutover_output_tokens,
            rate_gib_s=current.rate_gib_s,
            target_request_admitted=True,
            migration_id=current.migration_id,
            source_request_id_prefix=current.source_request_id_prefix,
            note=note or "target request admitted",
        ).write(self.run_dir)

    def disarm(self, note: str = "") -> RuntimeControl:
        current = RuntimeControl.load(self.run_dir) or RuntimeControl()
        return RuntimeControl(
            generation=current.generation,
            armed=False,
            trigger_output_tokens=current.trigger_output_tokens,
            cutover_output_tokens=current.cutover_output_tokens,
            rate_gib_s=current.rate_gib_s,
            target_request_admitted=current.target_request_admitted,
            migration_id=current.migration_id,
            source_request_id_prefix=current.source_request_id_prefix,
            note=note or "migration disarmed",
        ).write(self.run_dir)

    # ---- readiness evidence -------------------------------------------
    def poll_initial_history_gpu_buffered(self) -> tuple[bool, set[int], str]:
        """Return whether all four initial history shards reached TP4 GPU memory.

        This is only an admission gate for the dormant target request. The
        source cutover remains the sentinel until exact resident readback.
        """
        binding = self.refresh_binding()
        if binding is None:
            return False, set(), "session manifest not created yet"
        receipt_dir = self.run_dir / "gpu_initial_receipts"
        if not receipt_dir.is_dir():
            return False, set(), "initial GPU-history receipts not created yet"
        buffered: set[int] = set()
        for rank in range(4):
            path = receipt_dir / f"tp_rank_{rank}.json"
            if not path.is_file():
                continue
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if receipt.get("migration_id") != binding.migration_id:
                return False, buffered, f"rank {rank} initial receipt migration ID differs"
            status = receipt.get("status")
            if status == "INITIAL_HISTORY_GPU_BUFFERED":
                if receipt.get("gpu_ready") is True:
                    buffered.add(rank)
            elif status == "INITIAL_HISTORY_GPU_RESIDENT":
                if receipt.get("exact_readback") is not True:
                    return False, buffered, f"rank {rank} initial GPU readback FAILED"
                buffered.add(rank)
        return (
            len(buffered) == 4,
            buffered,
            "all four ranks initial history GPU-buffered"
            if len(buffered) == 4
            else f"{len(buffered)}/4 ranks initial history GPU-buffered",
        )

    def poll_initial_history_gpu_ready(self) -> tuple[bool, set[int], str]:
        """Return whether the initial Shadow history is safely ready on TP4.

        This is deliberately *not* the final ``TARGET_READY`` commit gate.
        During an earliest-ready experiment the source still owns generation,
        so final readiness cannot exist until the source freezes and sends its
        final delta. Buffered receipts admit a dormant target request, then
        these exact TP4 readback receipts authorize the final source cutover.
        A temporary ``INITIAL_HISTORY_GPU_BUFFERED`` receipt is deliberately
        not enough to freeze the source: it only proves that a receive buffer
        exists, not that history is resident in the target KV cache.
        """
        binding = self.refresh_binding()
        if binding is None:
            return False, set(), "session manifest not created yet"
        receipt_dir = self.run_dir / "gpu_initial_receipts"
        if not receipt_dir.is_dir():
            return False, set(), "initial GPU-history receipts not created yet"
        ready: set[int] = set()
        for rank in range(4):
            path = receipt_dir / f"tp_rank_{rank}.json"
            if not path.is_file():
                continue
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if receipt.get("migration_id") != binding.migration_id:
                return False, ready, f"rank {rank} initial receipt migration ID differs"
            if receipt.get("status") != "INITIAL_HISTORY_GPU_RESIDENT":
                continue
            if receipt.get("exact_readback") is not True:
                return False, ready, f"rank {rank} initial GPU history readback FAILED"
            ready.add(rank)
        return (
            len(ready) == 4,
            ready,
            "all four ranks initial history GPU-ready"
            if len(ready) == 4
            else f"{len(ready)}/4 ranks initial history GPU-ready",
        )

    def poll_delta_gpu_resident_progress(
        self,
    ) -> tuple[bool, dict[int, int], str]:
        """Read each rank's exact GPU-resident delta watermark.

        The initial history receipt is the baseline until that rank applies its
        first delta. A STREAMING watermark is published only after delta
        injection and exact readback, before the transport ACK is sent.
        """
        binding = self.refresh_binding()
        if binding is None:
            return False, {}, "session manifest not created yet"
        progress: dict[int, int] = {}
        for rank in range(4):
            initial_path = (
                self.run_dir / "gpu_initial_receipts" / f"tp_rank_{rank}.json"
            )
            try:
                initial = json.loads(initial_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False, progress, f"rank {rank} initial receipt unavailable"
            if (
                initial.get("migration_id") != binding.migration_id
                or initial.get("status") != "INITIAL_HISTORY_GPU_RESIDENT"
                or initial.get("exact_readback") is not True
            ):
                return False, progress, f"rank {rank} initial history not exact"
            try:
                initial_end = int(initial["end_token"])
            except (KeyError, TypeError, ValueError):
                return False, progress, f"rank {rank} initial end token missing"
            watermark_path = (
                self.run_dir / "gpu_watermarks" / f"tp_rank_{rank}.json"
            )
            try:
                watermark = json.loads(watermark_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                progress[rank] = initial_end
                continue
            except (OSError, json.JSONDecodeError):
                return False, progress, f"rank {rank} delta watermark unreadable"
            if (
                watermark.get("migration_id") != binding.migration_id
                or watermark.get("status") != "STREAMING"
                or watermark.get("exact_readback") is not True
            ):
                return False, progress, f"rank {rank} delta watermark not exact"
            try:
                end_token = int(watermark["end_token"])
            except (KeyError, TypeError, ValueError):
                return False, progress, f"rank {rank} delta end token missing"
            if end_token < initial_end:
                return False, progress, f"rank {rank} delta watermark regressed"
            progress[rank] = end_token
        return True, progress, "all four exact GPU-resident watermarks available"

    def poll_target_ready(self) -> tuple[bool, set[int], str]:
        """Mirror the server's ``_validate_target_ready`` gate.

        Returns ``(ready, ranks_ready, detail)``.
        """
        binding = self.refresh_binding()
        if binding is None:
            return False, set(), "session manifest not created yet"
        run = self.run_dir
        phase8 = (run / "staging_manifest.json").exists()
        sender_dir = run / ("stage_delivery_receipts" if phase8 else "sender_receipts")
        receiver_root = run / "receiver_receipts"
        if not sender_dir.is_dir() or not receiver_root.is_dir():
            return False, set(), "receipt directories not created yet"

        target_dirs = sorted(p for p in receiver_root.iterdir() if p.is_dir())
        if len(target_dirs) != 1:
            return False, set(), f"expected 1 target dir, found {len(target_dirs)}"

        ready: set[int] = set()
        for rank in range(4):
            sender_path = sender_dir / f"tp_rank_{rank}.json"
            receiver_path = target_dirs[0] / f"tp_rank_{rank}.json"
            if not (sender_path.exists() and receiver_path.exists()):
                continue
            try:
                sender = json.loads(sender_path.read_text(encoding="utf-8"))
                receiver = json.loads(receiver_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if sender.get("migration_id") != binding.migration_id:
                return False, ready, f"rank {rank} sender migration ID differs"
            if receiver.get("migration_id") != binding.migration_id:
                return False, ready, f"rank {rank} receiver migration ID differs"
            if sender.get("status") != "READY":
                continue
            if receiver.get("status") != "TARGET_READY":
                continue
            if not receiver.get("exact_readback"):
                return False, ready, f"rank {rank} exact readback FAILED"
            if sender.get("payload_sha256") != receiver.get("payload_sha256"):
                return False, ready, f"rank {rank} payload digest mismatch"
            if int(sender.get("payload_bytes", -1)) != int(
                receiver.get("payload_bytes", -2)
            ):
                return False, ready, f"rank {rank} payload byte count mismatch"
            ready.add(rank)
        return (
            len(ready) == 4,
            ready,
            "all four ranks ready"
            if len(ready) == 4
            else (f"{len(ready)}/4 ranks ready"),
        )

    def wait_for_target_ready(
        self,
        timeout_s: float,
    ) -> tuple[bool, set[int], str]:
        """Wait for a rank-ready notification, then verify disk evidence.

        UDP is only a wake-up hint.  The authoritative readiness result still
        comes from ``poll_target_ready``, which revalidates all sender and
        receiver receipts.  A lost or malformed datagram therefore falls back
        to the same fail-closed file gate used before P1.
        """
        ready, ranks, detail = self.poll_target_ready()
        if ready:
            self._authoritative_ready_unix_s = time.time()
            self._drain_ready_notifications()
            return ready, ranks, detail
        if self._ready_socket is None or timeout_s <= 0:
            return ready, ranks, detail

        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result = self.poll_target_ready()
                if result[0]:
                    self._authoritative_ready_unix_s = time.time()
                return result
            latch_armed = len(self._ready_notification_ranks) == 4
            if latch_armed and self._ready_latch_armed_unix_s is None:
                self._ready_latch_armed_unix_s = time.time()
            socket_timeout = remaining
            if latch_armed and self.ready_latch_poll_s > 0:
                socket_timeout = min(remaining, self.ready_latch_poll_s)
            self._ready_socket.settimeout(socket_timeout)
            try:
                payload, _address = self._ready_socket.recvfrom(65536)
            except TimeoutError:
                if latch_armed and self.ready_latch_poll_s > 0:
                    self._ready_latch_poll_count += 1
                    ready, ranks, detail = self.poll_target_ready()
                    if ready:
                        self._authoritative_ready_unix_s = time.time()
                        self._drain_ready_notifications()
                        return ready, ranks, detail
                    continue
                result = self.poll_target_ready()
                if result[0]:
                    self._authoritative_ready_unix_s = time.time()
                return result
            except OSError:
                result = self.poll_target_ready()
                if result[0]:
                    self._authoritative_ready_unix_s = time.time()
                return result
            if not self._record_ready_notification(payload):
                continue
            ready, ranks, detail = self.poll_target_ready()
            if ready:
                self._authoritative_ready_unix_s = time.time()
                self._drain_ready_notifications()
                return ready, ranks, detail

    def _record_ready_notification(self, payload: bytes) -> bool:
        try:
            message = json.loads(payload.decode("utf-8"))
            rank = int(message["tp_rank"])
            migration_id = str(message["migration_id"])
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            return False
        if not 0 <= rank < 4:
            return False
        if (
            self.expected_migration_id
            and migration_id != self.expected_migration_id
        ):
            return False
        message["controller_received_unix_s"] = time.time()
        self._last_ready_notification = message
        self._ready_notification_ranks.add(rank)
        self._ready_notification_count += 1
        return True

    def _drain_ready_notifications(self) -> None:
        if self._ready_socket is None:
            return
        self._ready_socket.setblocking(False)
        while True:
            try:
                payload, _address = self._ready_socket.recvfrom(65536)
            except BlockingIOError:
                return
            except OSError:
                return
            self._record_ready_notification(payload)

    def ready_notification_evidence(self) -> dict[str, Any]:
        """Return diagnostic evidence without making it a safety gate."""
        last = self._last_ready_notification or {}
        sent = last.get("sent_unix_s")
        received = last.get("controller_received_unix_s")
        return {
            "mode": self.ready_notification_mode,
            "notification_count": self._ready_notification_count,
            "notified_ranks": sorted(self._ready_notification_ranks),
            "last_notification_sent_unix_s": sent,
            "last_notification_received_unix_s": received,
            "last_notification_delivery_ms": (
                (float(received) - float(sent)) * 1000
                if sent is not None and received is not None
                else None
            ),
            "ready_latch_poll_ms": self.ready_latch_poll_s * 1000,
            "ready_latch_armed_unix_s": self._ready_latch_armed_unix_s,
            "ready_latch_poll_count": self._ready_latch_poll_count,
            "authoritative_ready_unix_s": self._authoritative_ready_unix_s,
            "ready_latch_to_authoritative_ready_ms": (
                (
                    self._authoritative_ready_unix_s
                    - self._ready_latch_armed_unix_s
                )
                * 1000
                if self._authoritative_ready_unix_s is not None
                and self._ready_latch_armed_unix_s is not None
                else None
            ),
        }

    def close(self) -> None:
        """Release the optional persistent P1 notification socket."""
        if self._ready_socket is not None:
            self._ready_socket.close()
            self._ready_socket = None

    # ---- terminal actions ----------------------------------------------
    def commit(self) -> dict[str, Any]:
        ready, ranks, detail = self.poll_target_ready()
        if not ready:
            raise ActionError(f"refusing commit: {detail} (ranks={sorted(ranks)})")
        return _post(
            f"{self.source_url}/bridge_tp/v1/takeover",
            self.binding.body(action="commit"),
        )

    def rollback(self, reason: str) -> dict[str, Any]:
        return _post(
            f"{self.source_url}/bridge_tp/v1/takeover",
            self.binding.body(action="rollback", reason=reason),
        )

    def cancel(self, reason: str, *, abort_source: bool = True) -> dict[str, Any]:
        """Drain pre-cutover staging and optionally abort the TP1 request.

        The default preserves the Phase 8 cancellation experiment.  Phase 9
        policy abandonment passes ``abort_source=False`` because ownership has
        not moved and the user's request must continue on TP1.
        """
        return _post(
            f"{self.source_url}/bridge_tp/v1/cleanup",
            self.binding.body(reason=reason, abort_source=abort_source),
        )

    def cancel_shadow_target(self, reason: str) -> dict[str, Any] | None:
        """Abort and release an admitted dormant TP4 Shadow request."""
        if self.target_url is None:
            return None
        control = RuntimeControl.load(self.run_dir)
        if control is None or not control.target_request_admitted:
            return None
        target_request_id = f"bridgetp-phase9-target-{self.run_dir.name}"
        return _post(
            f"{self.target_url}/bridge_tp/v1/shadow_target_cleanup",
            self.binding.body(
                reason=reason,
                target_request_id=target_request_id,
            ),
        )

    def read_takeover_state(self) -> dict[str, Any] | None:
        path = self.run_dir / "takeover_state.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
