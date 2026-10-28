# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Phase 7/8 control-plane API for commit, source abort, and rollback."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request

from vllm.bridge_tp.controller.tp_geometry import session_target_tp
from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()
_takeover_lock = asyncio.Lock()
_predictor_finalize_lock = asyncio.Lock()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _configured_session() -> tuple[Path, str]:
    run_dir_value = os.getenv("BRIDGETP_TAKEOVER_RUN_DIR", "").strip()
    migration_id = os.getenv("BRIDGETP_TAKEOVER_MIGRATION_ID", "").strip()
    persistent = os.getenv(
        "BRIDGETP_TAKEOVER_PERSISTENT_CHANNEL",
        os.getenv("BRIDGETP_PERSISTENT_CHANNEL", ""),
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not run_dir_value or (not migration_id and not persistent):
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="BridgeTP takeover session is not configured",
        )
    run_dir = Path(run_dir_value)
    if persistent:
        try:
            # Pin this API operation to the immutable active session directory.
            # A later channel reuse must not redirect its receipt or request ID.
            run_dir = run_dir.resolve(strict=True)
            active = _load_json(run_dir / "session_manifest.json")
            migration_id = str(active["migration_id"])
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Persistent BridgeTP session is not active: {error}",
            ) from error
    return run_dir, migration_id


def _validate_body(
    body: dict[str, Any], run_dir: Path, migration_id: str
) -> dict[str, Any]:
    session = _load_json(run_dir / "session_manifest.json")
    staging_path = run_dir / "staging_manifest.json"
    manifest = _load_json(staging_path) if staging_path.exists() else session
    expected = {
        "migration_id": migration_id,
        "session_token": session["session_token"],
        "source_request_id": session["source_request_id"],
    }
    for key, value in expected.items():
        if body.get(key) != value:
            raise HTTPException(
                status_code=HTTPStatus.FORBIDDEN,
                detail=f"BridgeTP takeover field {key} does not match the session",
            )
    return manifest


def _validate_target_ready(
    run_dir: Path, migration_id: str
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    if session_target_tp(run_dir) != int(os.getenv("BRIDGETP_STREAM_TARGET_TP", "4")):
        raise ValueError("session target geometry differs from configured transport")
    phase8 = (run_dir / "staging_manifest.json").exists()
    sender_dir = (
        run_dir / "stage_delivery_receipts" if phase8 else run_dir / "sender_receipts"
    )
    senders = [
        _load_json(sender_dir / f"tp_rank_{rank}.json")
        for rank in range(session_target_tp(run_dir))
    ]
    receiver_root = run_dir / "receiver_receipts"
    target_dirs = sorted(path for path in receiver_root.iterdir() if path.is_dir())
    if len(target_dirs) != 1:
        raise ValueError("Phase 7 requires exactly one target request directory")
    receivers = [
        _load_json(target_dirs[0] / f"tp_rank_{rank}.json")
        for rank in range(session_target_tp(run_dir))
    ]
    target_request_ids = {str(receipt["target_request_id"]) for receipt in receivers}
    if len(target_request_ids) != 1:
        raise ValueError("TP4 ranks disagree on the target request ID")
    target_request_id = next(iter(target_request_ids))
    for rank, (sender, receiver) in enumerate(zip(senders, receivers)):
        for receipt in (sender, receiver):
            if receipt.get("migration_id") != migration_id:
                raise ValueError(f"TP rank {rank} migration ID differs")
        if sender.get("status") != "READY":
            raise ValueError(f"TP rank {rank} sender is not READY")
        if receiver.get("status") != "TARGET_READY":
            raise ValueError(f"TP rank {rank} receiver is not TARGET_READY")
        if not receiver.get("exact_readback"):
            raise ValueError(f"TP rank {rank} exact readback failed")
        if sender.get("payload_sha256") != receiver.get("payload_sha256"):
            raise ValueError(f"TP rank {rank} payload SHA256 differs")
        if int(sender["payload_bytes"]) != int(receiver["payload_bytes"]):
            raise ValueError(f"TP rank {rank} payload byte count differs")
    return target_request_id, senders, receivers


def _external_request_id(source_request_id: str) -> str:
    if "-" not in source_request_id:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail="Source internal request ID cannot be mapped to external ID",
        )
    return source_request_id.rsplit("-", 1)[0]


def _target_completion_engine_request_id(target_request_id: str) -> str:
    """Map a single-prompt completion payload ID to its engine request ID."""
    return f"cmpl-{target_request_id}-0"


@router.post("/bridge_tp/v1/cleanup")
async def cleanup(raw_request: Request) -> dict[str, Any]:
    """Cancel pre-cutover staging, optionally aborting the source request.

    ``abort_source`` defaults to ``True`` for compatibility with the Phase 8
    cancellation experiment.  An online controller abandoning only the
    migration must send ``False`` so TP1 remains the sole response owner.
    """
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"JSON decode error: {error}",
        ) from error
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Request body must be a JSON object",
        )
    abort_source = body.get("abort_source", True)
    if not isinstance(abort_source, bool):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="abort_source must be a JSON boolean",
        )
    run_dir, migration_id = _configured_session()
    manifest = _validate_body(body, run_dir, migration_id)
    if manifest.get("phase") != "BridgeTP D3 Phase 8":
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail="The cleanup endpoint is restricted to BridgeTP Phase 8",
        )
    async with _takeover_lock:
        state_path = run_dir / "takeover_state.json"
        state = _load_json(state_path)
        if state.get("state") != "PREPARING":
            raise HTTPException(
                status_code=HTTPStatus.CONFLICT,
                detail=f"Cannot clean up takeover state {state.get('state')}",
            )
        cleanup_request = {
            "format_version": 1,
            "phase": "BridgeTP D3 Phase 8",
            "migration_id": migration_id,
            "source_request_id": manifest["source_request_id"],
            "reason": str(body.get("reason", "controller cancelled before cutover")),
            "abort_source": abort_source,
            "requested_unix_s": time.time(),
        }
        _atomic_json_dump(cleanup_request, run_dir / "cleanup_request.json")
        source_external_request_id: str | None = None
        if cleanup_request["abort_source"]:
            source_external_request_id = _external_request_id(
                str(manifest["source_request_id"])
            )
            await raw_request.app.state.engine_client.abort(source_external_request_id)
        elif (run_dir / "cutover_manifest.json").exists():
            # A previously frozen source must resume if TP4 cannot take over.
            # Before manifest publication the background finalizer owns this
            # transition and writes RESUME after its delta worker has stopped.
            _atomic_json_dump(
                {
                    "format_version": 1,
                    "action": "RESUME",
                    "request_id": manifest["source_request_id"],
                    "reason": "migration cancelled before commit",
                },
                run_dir / "request_freeze_control.json",
            )
        cancelled = {
            **state,
            "state": "CANCELLED",
            "source_external_request_id": source_external_request_id,
            "source_abort_dispatched": cleanup_request["abort_source"],
            "source_continues_on_tp1": not cleanup_request["abort_source"],
            "reason": cleanup_request["reason"],
            "updated_unix_s": time.time(),
        }
        _atomic_json_dump(cancelled, state_path)
        return cancelled


@router.post("/bridge_tp/v1/takeover")
async def takeover(raw_request: Request) -> dict[str, Any]:
    """Commit a ready target and abort source, or roll back before commit."""
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"JSON decode error: {error}",
        ) from error
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Request body must be a JSON object",
        )
    action = body.get("action")
    if action not in {"commit", "rollback"}:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="action must be 'commit' or 'rollback'",
        )

    run_dir, migration_id = _configured_session()
    manifest = _validate_body(body, run_dir, migration_id)
    state_path = run_dir / "takeover_state.json"

    async with _takeover_lock:
        existing = _load_json(state_path) if state_path.exists() else None
        if existing is not None:
            existing_state = existing.get("state")
            if action == "commit" and existing_state == "COMMITTED":
                return existing
            if action == "rollback" and existing_state == "ROLLED_BACK":
                return existing
            if existing_state != "PREPARING":
                raise HTTPException(
                    status_code=HTTPStatus.CONFLICT,
                    detail=f"Takeover already reached state {existing_state}",
                )

        base_state = {
            "format_version": 1,
            "phase": str(manifest.get("phase", "BridgeTP D3 Phase 7")),
            "scope": ("application-level atomic handoff; no crash-consensus claim"),
            "migration_id": migration_id,
            "source_request_id": manifest["source_request_id"],
            "snapshot_num_output_tokens": manifest["snapshot_num_output_tokens"],
            "updated_unix_s": time.time(),
        }

        if action == "rollback":
            state = {
                **base_state,
                "state": "ROLLED_BACK",
                "source_abort_dispatched": False,
                "reason": str(body.get("reason", "controller requested rollback")),
            }
            _atomic_json_dump(state, state_path)
            logger.warning(
                "BridgeTP takeover rolled back migration %s; source remains owner",
                migration_id,
            )
            return state

        try:
            target_request_id, senders, receivers = _validate_target_ready(
                run_dir, migration_id
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=HTTPStatus.CONFLICT,
                detail=f"TP4 target is not ready: {error}",
            ) from error

        source_request_id = str(manifest["source_request_id"])
        source_external_request_id = _external_request_id(source_request_id)
        committing = {
            **base_state,
            "state": "COMMITTING",
            "target_request_id": target_request_id,
            "source_external_request_id": source_external_request_id,
            "source_abort_dispatched": False,
            "ready_sender_ranks": [receipt["target_tp_rank"] for receipt in senders],
            "ready_receiver_ranks": [receipt["tp_rank"] for receipt in receivers],
        }
        _atomic_json_dump(committing, state_path)

        await raw_request.app.state.engine_client.abort(source_external_request_id)
        committed = {
            **committing,
            "state": "COMMITTED",
            "source_abort_dispatched": True,
            "updated_unix_s": time.time(),
        }
        _atomic_json_dump(committed, state_path)
        from vllm.bridge_tp.experiment_timeline import emit_event

        emit_event(
            run_dir,
            "source_api",
            "TAKEOVER_COMMITTED_SOURCE_ABORT_DISPATCHED",
            request_id=source_external_request_id,
            migration_id=migration_id,
        )
        logger.warning(
            "BridgeTP takeover committed migration %s and aborted source %s",
            migration_id,
            source_external_request_id,
        )
        return committed


@router.post("/bridge_tp/v1/shadow_target_cleanup")
async def shadow_target_cleanup(raw_request: Request) -> dict[str, Any]:
    """Abort a dormant target request when reversible Shadow is abandoned."""
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"JSON decode error: {error}",
        ) from error
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Request body must be a JSON object",
        )
    run_dir, migration_id = _configured_session()
    _validate_body(body, run_dir, migration_id)
    target_request_id = str(body.get("target_request_id", "")).strip()
    expected_request_id = f"bridgetp-phase9-target-{run_dir.name}"
    if target_request_id != expected_request_id:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail="Shadow target request ID differs from configured session",
        )
    # OpenAI completions prefixes the supplied request_id and appends the
    # prompt index before handing it to the engine. Aborting the bare payload
    # ID leaves the dormant target request in the scheduler indefinitely.
    await raw_request.app.state.engine_client.abort(
        _target_completion_engine_request_id(target_request_id)
    )
    receipt = {
        "format_version": 1,
        "phase": "BridgeTP D3 Phase 8",
        "status": "CLEANED",
        "migration_id": migration_id,
        "target_request_id": target_request_id,
        "reason": str(body.get("reason", "Shadow migration abandoned")),
        "updated_unix_s": time.time(),
    }
    _atomic_json_dump(receipt, run_dir / "target_cleanup_receipt.json")
    return receipt


@router.post("/bridge_tp/v1/predictor_features/finalize")
async def finalize_predictor_features(raw_request: Request) -> dict[str, Any]:
    """Confirm feature persistence before the experiment stops its source.

    This experiment-only route accepts a per-run secret, exposes only one named
    worker operation, and is called after controller and background completion.
    It does not enable the unrestricted development RPC router.
    """
    token = os.environ.get("BRIDGETP_PREDICTOR_FINALIZE_TOKEN", "")
    if not token or not os.environ.get("BRIDGETP_PREDICTOR_FEATURE_JOURNAL"):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Feature journal is disabled"
        )
    try:
        body = await raw_request.json()
    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Expected a JSON object"
        ) from error
    provided = body.get("token") if isinstance(body, dict) else None
    if not isinstance(provided, str) or not hmac.compare_digest(provided, token):
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN, detail="Finalization token differs"
        )
    async with _predictor_finalize_lock:
        results = await raw_request.app.state.engine_client.collective_rpc(
            "bridge_tp_finalize_predictor_features", timeout=60.0
        )
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
        or results[0].get("status") != "FINALIZED"
        or not isinstance(results[0].get("saved_batches"), int)
        or results[0]["saved_batches"] <= 0
    ):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Source feature journal did not confirm persistence",
        )
    return results[0]


@router.post("/bridge_tp/v1/performance_trace/finalize")
async def finalize_performance_trace(raw_request: Request) -> dict[str, Any]:
    """Expose one authenticated, idle-only experiment evidence operation."""
    token = os.environ.get("BRIDGETP_PERFORMANCE_FINALIZE_TOKEN", "")
    if not token or not os.environ.get("BRIDGETP_PERFORMANCE_TRACE_DIR"):
        raise HTTPException(status_code=404, detail="Performance trace disabled")
    try:
        body = await raw_request.json()
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail="Expected JSON object") from error
    provided = body.get("token") if isinstance(body, dict) else None
    if not isinstance(provided, str) or not hmac.compare_digest(provided, token):
        raise HTTPException(status_code=403, detail="Finalization token differs")
    async with _predictor_finalize_lock:
        return await raw_request.app.state.engine_client.engine_core.call_utility_async(
            "bridge_tp_finalize_performance_trace"
        )


@router.post("/bridge_tp/v1/diagnostic/retire_channel")
async def retire_diagnostic_channel(raw_request: Request) -> dict[str, Any]:
    """Experimental one-way channel retirement after native empty/source release."""
    token = os.environ.get("BRIDGETP_CHANNEL_DIAGNOSTIC_TOKEN", "")
    role = os.environ.get("BRIDGETP_CHANNEL_DIAGNOSTIC_ROLE", "")
    root = os.environ.get("BRIDGETP_CHANNEL_DIAGNOSTIC_DIR", "")
    if not token or role not in {"source", "target"} or not root:
        raise HTTPException(status_code=404, detail="Channel diagnostic disabled")
    body = await raw_request.json()
    provided = body.get("token") if isinstance(body, dict) else None
    if not isinstance(provided, str) or not hmac.compare_digest(provided, token):
        raise HTTPException(status_code=403, detail="Diagnostic token differs")
    from vllm.bridge_tp.channel_diagnostics import validate_arm

    async with _takeover_lock:
        try:
            validate_arm(Path(root), role, os.getpid())
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        results = await raw_request.app.state.engine_client.collective_rpc(
            "bridge_tp_retire_diagnostic_channel", timeout=60.0
        )
    expected = 1 if role == "source" else session_target_tp(Path(root))
    if any(r.get("status") == "REJECTED" for r in results):
        raise HTTPException(status_code=409, detail=dict(role=role, ranks=results))
    if len(results) != expected or any(
        r.get("status") != "RETIRED" or not r.get("identity_unchanged") for r in results
    ):
        raise HTTPException(status_code=500, detail="Retirement rank proof incomplete")
    return dict(status="RETIRED", role=role, ranks=results)


def attach_router(app: FastAPI, *, diagnostics_only: bool = False) -> None:
    if diagnostics_only:
        diagnostics = APIRouter()
        diagnostics.add_api_route(
            "/bridge_tp/v1/performance_trace/finalize",
            finalize_performance_trace,
            methods=["POST"],
        )
        app.include_router(diagnostics)
    else:
        app.include_router(router)
