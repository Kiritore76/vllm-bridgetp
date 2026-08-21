#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Drive one Phase 8 old-KV/new-KV staged atomic takeover."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from run_atomic_takeover import (
    _choice,
    _dump_json,
    _load_json,
    _post_completion,
    _post_json,
    _post_streaming_completion,
    _token_ids,
)
from vllm.bridge_tp.stream_protocol import MIGRATION_PARAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default="http://127.0.0.1:8001")
    parser.add_argument("--target-url", default="http://127.0.0.1:8200")
    parser.add_argument("--source-request", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ready-timeout-s", type=float, default=600)
    parser.add_argument("--request-timeout-s", type=float, default=900)
    return parser.parse_args()


def _wait_for_path(
    path: Path, source_future: Future[dict[str, Any]], deadline: float
) -> dict[str, Any]:
    while not path.exists():
        if source_future.done():
            response = source_future.result()
            _dump_json(response, path.parent / "source_response_before_staging.json")
            raise RuntimeError("Source finished before Phase 8 staging was ready")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.02)
    return _load_json(path)


def _ready(run_dir: Path) -> bool:
    delivery = run_dir / "stage_delivery_receipts"
    receiver_root = run_dir / "receiver_receipts"
    if not receiver_root.exists():
        return False
    target_dirs = [path for path in receiver_root.iterdir() if path.is_dir()]
    if len(target_dirs) != 1:
        return False
    for rank in range(4):
        sender_path = delivery / f"tp_rank_{rank}.json"
        receiver_path = target_dirs[0] / f"tp_rank_{rank}.json"
        if not sender_path.exists() or not receiver_path.exists():
            return False
        sender = _load_json(sender_path)
        receiver = _load_json(receiver_path)
        if sender.get("status") != "READY":
            return False
        if receiver.get("status") != "TARGET_READY":
            return False
        if not receiver.get("exact_readback"):
            return False
    return True


def _wait_ready(
    run_dir: Path,
    source_future: Future[dict[str, Any]],
    target_future: Future[dict[str, Any]],
    deadline: float,
) -> None:
    while not _ready(run_dir):
        if source_future.done():
            raise RuntimeError("Source finished before Phase 8 target became ready")
        if target_future.done():
            target_future.result()
            raise RuntimeError("Target finished before Phase 8 commit")
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for Phase 8 TARGET_READY")
        time.sleep(0.02)


def _overlap_evidence(run_dir: Path) -> tuple[float, float, bool]:
    initial_completed = [
        float(
            _load_json(
                run_dir / "initial_stage_receipts" / f"tp_rank_{rank}.json"
            )["completed_unix_s"]
        )
        for rank in range(4)
    ]
    delta_times = [
        float(_load_json(path)["staged_unix_s"])
        for path in (run_dir / "delta_sender_receipts").glob("*.json")
    ]
    if not delta_times:
        raise ValueError("Phase 8 produced no new-KV delta receipts")
    first_delta = min(delta_times)
    last_initial = max(initial_completed)
    return first_delta, last_initial, first_delta < last_initial


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    source_request = _load_json(args.source_request)
    source_request.update(
        {
            "request_id": f"bridgetp-phase8-{args.run_dir.name}",
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
            "return_token_ids": True,
        }
    )
    _dump_json(source_request, args.run_dir / "source_request.json")
    deadline = time.monotonic() + args.ready_timeout_s

    with ThreadPoolExecutor(max_workers=2) as executor:
        source_future = executor.submit(
            _post_completion,
            args.source_url,
            source_request,
            args.request_timeout_s,
        )
        staging = _wait_for_path(
            args.run_dir / "staging_manifest.json", source_future, deadline
        )
        if staging.get("phase") != "BridgeTP D3 Phase 8":
            raise ValueError("Stager did not publish a Phase 8 manifest")
        cutover_output = int(staging["snapshot_num_output_tokens"])
        remaining_tokens = int(source_request["max_tokens"]) - cutover_output
        if remaining_tokens < 64:
            raise ValueError("Phase 8 requires at least 64 post-cutover tokens")
        target_request = {
            "model": source_request["model"],
            "request_id": f"bridgetp-phase8-target-{args.run_dir.name}",
            "prompt": staging["all_known_token_ids"],
            "max_tokens": remaining_tokens,
            "temperature": 0,
            "stream": True,
            "return_token_ids": True,
            "kv_transfer_params": {
                MIGRATION_PARAM: staging["migration_id"],
            },
        }
        _dump_json(target_request, args.run_dir / "target_request.json")
        target_future = executor.submit(
            _post_streaming_completion,
            args.target_url,
            target_request,
            args.request_timeout_s,
        )
        _wait_ready(args.run_dir, source_future, target_future, deadline)
        ready_monotonic = time.monotonic()
        takeover_request = {
            "migration_id": staging["migration_id"],
            "session_token": staging["session_token"],
            "source_request_id": staging["source_request_id"],
            "action": "commit",
            "reason": "Phase 8 staged old-KV/new-KV cutover",
        }
        commit_started = time.monotonic()
        decision = _post_json(
            args.source_url,
            "/bridge_tp/v1/takeover",
            takeover_request,
            args.request_timeout_s,
        )
        commit_completed = time.monotonic()
        _dump_json(takeover_request, args.run_dir / "takeover_request.json")
        _dump_json(decision, args.run_dir / "takeover_response.json")
        target_response = target_future.result()
        source_response = source_future.result()

    _dump_json(source_response, args.run_dir / "source_response.json")
    _dump_json(target_response, args.run_dir / "target_response.json")
    control_request = dict(source_request)
    control_request.pop("request_id", None)
    control_response = _post_completion(
        args.source_url, control_request, args.request_timeout_s
    )
    _dump_json(control_request, args.run_dir / "control_request.json")
    _dump_json(control_response, args.run_dir / "control_response.json")

    prompt_tokens = int(staging["num_prompt_tokens"])
    cutover_ids = list(staging["all_known_token_ids"])[prompt_tokens:]
    # Streaming target responses are normalized by
    # _post_streaming_completion() and expose token_ids at the top level,
    # unlike ordinary completion responses which store them in choices[0].
    target_token_ids = target_response.get("token_ids")
    if not isinstance(target_token_ids, list):
        raise ValueError("Streaming target response contains no token_ids")
    target_ids = [int(token_id) for token_id in target_token_ids]
    source_ids = _token_ids(source_response)
    control_ids = _token_ids(control_response)
    assembled_ids = cutover_ids + target_ids
    first_delta, last_initial, overlapped = _overlap_evidence(args.run_dir)
    first_token_time = target_response.get("first_token_monotonic")
    exact = assembled_ids == control_ids
    result = {
        "status": "PASS"
        if (
            decision.get("state") == "COMMITTED"
            and decision.get("source_abort_dispatched")
            and _choice(source_response).get("finish_reason") == "abort"
            and source_ids[:len(cutover_ids)] == cutover_ids
            and len(target_ids) == remaining_tokens
            and int(staging["new_kv_delta_tokens"]) > 0
            and overlapped
            and exact
        )
        else "FAIL",
        "phase": "BridgeTP D3 Phase 8",
        "migration_id": staging["migration_id"],
        "takeover_state": decision.get("state"),
        "source_abort_dispatched": decision.get("source_abort_dispatched"),
        "source_finish_reason": _choice(source_response).get("finish_reason"),
        "old_kv_num_computed_tokens": staging["old_kv_num_computed_tokens"],
        "cutover_num_output_tokens": cutover_output,
        "cutover_num_computed_tokens": staging["num_computed_tokens"],
        "new_kv_delta_tokens": staging["new_kv_delta_tokens"],
        "new_kv_delta_batches": staging["new_kv_delta_batches"],
        "old_kv_new_kv_overlap_proven": overlapped,
        "first_delta_staged_unix_s": first_delta,
        "last_initial_rank_staged_unix_s": last_initial,
        "source_tokens_computed_before_abort": len(source_ids),
        "remaining_generation_budget_transferred": remaining_tokens,
        "target_tokens": len(target_ids),
        "assembled_token_ids": assembled_ids,
        "control_token_ids": control_ids,
        "exact_end_to_end_token_continuity": exact,
        "target_ready_to_commit_response_ms": (
            commit_completed - ready_monotonic
        ) * 1000,
        "commit_api_ms": (commit_completed - commit_started) * 1000,
        "commit_to_target_first_token_ms": (
            (float(first_token_time) - commit_completed) * 1000
            if first_token_time is not None
            else None
        ),
    }
    _dump_json(result, args.run_dir / "phase8_result.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
