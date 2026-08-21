# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Phase 7/8 control-plane API for commit, source abort, and rollback."""

from __future__ import annotations

import asyncio
import json
import os
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request

from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()
_takeover_lock = asyncio.Lock()


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
    if not run_dir_value or not migration_id:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="BridgeTP takeover session is not configured",
        )
    return Path(run_dir_value).resolve(), migration_id


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
    phase8 = (run_dir / "staging_manifest.json").exists()
    sender_dir = (
        run_dir / "stage_delivery_receipts"
        if phase8
        else run_dir / "sender_receipts"
    )
    senders = [
        _load_json(sender_dir / f"tp_rank_{rank}.json")
        for rank in range(4)
    ]
    receiver_root = run_dir / "receiver_receipts"
    target_dirs = sorted(path for path in receiver_root.iterdir() if path.is_dir())
    if len(target_dirs) != 1:
        raise ValueError("Phase 7 requires exactly one target request directory")
    receivers = [
        _load_json(target_dirs[0] / f"tp_rank_{rank}.json")
        for rank in range(4)
    ]
    target_request_ids = {
        str(receipt["target_request_id"]) for receipt in receivers
    }
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


@router.post("/bridge_tp/v1/cleanup")
async def cleanup(raw_request: Request) -> dict[str, Any]:
    """Cancel a pre-cutover source and release Phase 8 staging resources."""
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
            "requested_unix_s": time.time(),
        }
        _atomic_json_dump(cleanup_request, run_dir / "cleanup_request.json")
        source_external_request_id = _external_request_id(
            str(manifest["source_request_id"])
        )
        await raw_request.app.state.engine_client.abort(source_external_request_id)
        cancelled = {
            **state,
            "state": "CANCELLED",
            "source_external_request_id": source_external_request_id,
            "source_abort_dispatched": True,
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
            "scope": (
                "application-level atomic handoff; no crash-consensus claim"
            ),
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
        logger.warning(
            "BridgeTP takeover committed migration %s and aborted source %s",
            migration_id,
            source_external_request_id,
        )
        return committed


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
