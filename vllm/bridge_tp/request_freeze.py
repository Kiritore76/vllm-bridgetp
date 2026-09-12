# SPDX-License-Identifier: Apache-2.0
"""Experimental per-request freeze gate for BridgeTP migration experiments.

The gate is intentionally file based.  The model worker publishes an exact
token-boundary freeze request while the EngineCore scheduler owns the request
and its KV blocks.  The scheduler then omits only that request from subsequent
steps; other requests remain schedulable and the frozen request keeps its KV
allocation until takeover aborts it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_TRUE = {"1", "true", "yes", "on"}


def enabled_from_env() -> bool:
    return os.getenv("BRIDGETP_REQUEST_FREEZE_ENABLED", "0").strip().lower() in _TRUE


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def request_freeze(
    run_dir: str | Path,
    request_id: str,
    *,
    output_tokens: int,
    num_computed_tokens: int,
) -> dict[str, Any]:
    """Publish a freeze request at a completed model-step boundary."""
    value = {
        "format_version": 1,
        "action": "FREEZE",
        "request_id": request_id,
        "output_tokens": int(output_tokens),
        "num_computed_tokens": int(num_computed_tokens),
        "requested_unix_ns": time.time_ns(),
        "requested_monotonic_ns": time.monotonic_ns(),
    }
    _atomic_json_dump(value, Path(run_dir) / "request_freeze_control.json")
    from vllm.bridge_tp.experiment_timeline import emit_event

    emit_event(
        run_dir,
        "source_worker",
        "FREEZE_REQUESTED",
        request_id=request_id,
        output_tokens=int(output_tokens),
        num_computed_tokens=int(num_computed_tokens),
    )
    return value


class RequestFreezeGate:
    """Cheap scheduler-side cache for one experimental frozen request."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.control_path = self.run_dir / "request_freeze_control.json"
        self.frozen_receipt_path = self.run_dir / "request_frozen_receipt.json"
        self.release_receipt_path = self.run_dir / "source_kv_release_receipt.json"
        self._mtime_ns = -1
        self._request_id: str | None = None
        self._recorded_frozen = False
        self._request_prefix = os.getenv(
            "BRIDGETP_STREAM_SOURCE_REQUEST_ID_PREFIX", ""
        ).strip()

    @classmethod
    def from_env(cls) -> RequestFreezeGate | None:
        if not enabled_from_env():
            return None
        run_dir = os.getenv("BRIDGETP_STREAM_RUN_DIR", "").strip()
        if not run_dir:
            raise ValueError(
                "BRIDGETP_REQUEST_FREEZE_ENABLED requires BRIDGETP_STREAM_RUN_DIR"
            )
        return cls(run_dir)

    def _refresh(self) -> None:
        try:
            stat = self.control_path.stat()
        except FileNotFoundError:
            return
        if stat.st_mtime_ns == self._mtime_ns:
            return
        value = json.loads(self.control_path.read_text(encoding="utf-8"))
        self._mtime_ns = stat.st_mtime_ns
        if value.get("action") == "FREEZE":
            self._request_id = str(value["request_id"])
        else:
            self._request_id = None

    def is_frozen(self, request_id: str) -> bool:
        self._refresh()
        return self._request_id == request_id

    def record_frozen(self, request: Any, scheduler_step: int) -> None:
        if self._recorded_frozen:
            return
        self._recorded_frozen = True
        _atomic_json_dump(
            {
                "format_version": 1,
                "status": "FROZEN",
                "request_id": request.request_id,
                "scheduler_step": int(scheduler_step),
                "num_prompt_tokens": int(request.num_prompt_tokens),
                "num_output_tokens": len(request.output_token_ids),
                "num_computed_tokens": int(request.num_computed_tokens),
                "frozen_unix_ns": time.time_ns(),
                "frozen_monotonic_ns": time.monotonic_ns(),
                "scope": "single request; KV retained; peer scheduling continues",
            },
            self.frozen_receipt_path,
        )
        from vllm.bridge_tp.experiment_timeline import emit_event

        emit_event(
            self.run_dir,
            "source_scheduler",
            "REQUEST_FROZEN",
            request_id=request.request_id,
            scheduler_step=int(scheduler_step),
            num_output_tokens=len(request.output_token_ids),
        )

    def record_released(self, request: Any) -> None:
        self._refresh()
        if self._request_id != request.request_id and not (
            self._request_prefix and request.request_id.startswith(self._request_prefix)
        ):
            return
        _atomic_json_dump(
            {
                "format_version": 1,
                "status": "SOURCE_KV_RELEASED",
                "request_id": request.request_id,
                "num_prompt_tokens": int(request.num_prompt_tokens),
                "num_output_tokens": len(request.output_token_ids),
                "num_computed_tokens": int(request.num_computed_tokens),
                "released_unix_ns": time.time_ns(),
                "released_monotonic_ns": time.monotonic_ns(),
                "evidence": "emitted after KVCacheManager.free returned",
            },
            self.release_receipt_path,
        )
        from vllm.bridge_tp.experiment_timeline import emit_event

        emit_event(
            self.run_dir,
            "source_scheduler",
            "SOURCE_KV_RELEASED",
            request_id=request.request_id,
            num_output_tokens=len(request.output_token_ids),
        )
