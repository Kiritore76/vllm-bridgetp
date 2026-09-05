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
    ) -> None:
        self.source_url = source_url.rstrip("/")
        self.expected_migration_id = expected_migration_id
        if isinstance(run_dir, SessionBinding):
            self.run_dir = run_dir.run_dir
            self._binding: SessionBinding | None = run_dir
        else:
            self.run_dir = Path(run_dir)
            self._binding = None

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
            note=note or "migration disarmed",
        ).write(self.run_dir)

    # ---- readiness evidence -------------------------------------------
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

    def read_takeover_state(self) -> dict[str, Any] | None:
        path = self.run_dir / "takeover_state.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
