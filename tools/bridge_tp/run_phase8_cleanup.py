#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate explicit Phase 8 cancellation and resource cleanup."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_atomic_takeover import (
    _choice,
    _dump_json,
    _load_json,
    _post_completion,
    _post_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default="http://127.0.0.1:8001")
    parser.add_argument("--source-request", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=600)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    request = _load_json(args.source_request)
    request.update(
        {
            "request_id": f"bridgetp-phase8-cancel-{args.run_dir.name}",
            "max_tokens": max(288, int(request.get("max_tokens", 0))),
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
            "return_token_ids": True,
        }
    )
    _dump_json(request, args.run_dir / "source_request.json")
    deadline = time.monotonic() + args.timeout_s
    with ThreadPoolExecutor(max_workers=1) as executor:
        source_future = executor.submit(
            _post_completion, args.source_url, request, args.timeout_s
        )
        manifest_path = args.run_dir / "session_manifest.json"
        delta_dir = args.run_dir / "delta_sender_receipts"
        while not manifest_path.exists() or not list(delta_dir.glob("*.json")):
            if source_future.done():
                raise RuntimeError("Source ended before cancellation point")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for staged Phase 8 deltas")
            time.sleep(0.02)
        manifest = _load_json(manifest_path)
        cleanup_request = {
            "migration_id": manifest["migration_id"],
            "session_token": manifest["session_token"],
            "source_request_id": manifest["source_request_id"],
            "reason": "explicit controller cancellation before cutover",
        }
        cleanup_response = _post_json(
            args.source_url,
            "/bridge_tp/v1/cleanup",
            cleanup_request,
            args.timeout_s,
        )
        _dump_json(cleanup_request, args.run_dir / "cleanup_api_request.json")
        _dump_json(cleanup_response, args.run_dir / "cleanup_api_response.json")
        response = source_future.result()
    _dump_json(response, args.run_dir / "source_response.json")

    source_receipt_path = args.run_dir / "source_cleanup_receipt.json"
    stager_receipt_path = args.run_dir / "stager_cleanup_receipt.json"
    while not source_receipt_path.exists() or not stager_receipt_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for Phase 8 cleanup receipts")
        time.sleep(0.02)
    source_receipt = _load_json(source_receipt_path)
    stager_receipt = _load_json(stager_receipt_path)
    initial_ready = all(
        _load_json(
            args.run_dir / "sender_receipts" / f"tp_rank_{rank}.json"
        ).get("status") == "READY"
        for rank in range(4)
    )
    result = {
        "status": "PASS"
        if (
            cleanup_response.get("state") == "CANCELLED"
            and cleanup_response.get("source_abort_dispatched") is True
            and _choice(response).get("finish_reason") == "abort"
            and source_receipt.get("status") == "CLEANED"
            and stager_receipt.get("status") == "CLEANED"
            and initial_ready
            and int(source_receipt.get("delta_tokens_drained", 0)) > 0
            and not (args.run_dir / "staging_manifest.json").exists()
            and not (args.run_dir / "target_request.json").exists()
        )
        else "FAIL",
        "phase": "BridgeTP D3 Phase 8",
        "mode": "pre_cutover_controller_cancellation",
        "takeover_state": cleanup_response.get("state"),
        "source_abort_dispatched": cleanup_response.get(
            "source_abort_dispatched"
        ),
        "source_finish_reason": _choice(response).get("finish_reason"),
        "source_status": source_receipt.get("status"),
        "stager_status": stager_receipt.get("status"),
        "initial_old_kv_staged": initial_ready,
        "delta_tokens_drained": source_receipt.get("delta_tokens_drained"),
        "released_rank_buffers": stager_receipt.get("released_rank_buffers"),
        "target_request_created": False,
        "takeover_committed": False,
    }
    _dump_json(result, args.run_dir / "phase8_cleanup_result.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
