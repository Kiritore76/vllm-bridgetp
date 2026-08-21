# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Drive one Phase 7 commit or rollback run on a live TP1/TP4 pair."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from vllm.bridge_tp.stream_protocol import MIGRATION_PARAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default="http://127.0.0.1:8001")
    parser.add_argument("--target-url", default="http://127.0.0.1:8200")
    parser.add_argument("--source-request", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ready-timeout-s", type=float, default=900)
    parser.add_argument("--request-timeout-s", type=float, default=1800)
    parser.add_argument(
        "--force-rollback-after-ready",
        action="store_true",
        help="Exercise pre-commit rollback; the source must remain owner.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _dump_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def _post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(
            f"POST {path} on {base_url} failed: HTTP {error.code}: {body}"
        ) from error
    if not isinstance(value, dict):
        raise TypeError(f"POST {path} did not return a JSON object")
    return value


def _post_completion(
    base_url: str, payload: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    return _post_json(base_url, "/v1/completions", payload, timeout_s)


def _post_streaming_completion(
    base_url: str, payload: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    token_ids: list[int] = []
    chunks: list[dict[str, Any]] = []
    first_token_monotonic: float | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    saw_done = False
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    saw_done = True
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(f"Target streaming error: {chunk['error']}")
                chunks.append(chunk)
                response_id = chunk.get("id", response_id)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta_ids = choice.get("token_ids") or []
                if delta_ids and first_token_monotonic is None:
                    first_token_monotonic = time.monotonic()
                token_ids.extend(int(token_id) for token_id in delta_ids)
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(
            f"Target completion failed: HTTP {error.code}: {body}"
        ) from error
    if not saw_done or finish_reason is None:
        raise RuntimeError("Target streaming response ended before completion")
    return {
        "response_id": response_id,
        "token_ids": token_ids,
        "finish_reason": finish_reason,
        "first_token_monotonic": first_token_monotonic,
        "chunks": chunks,
    }


def _choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Completion response contains no choice: {response}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise TypeError("Completion choice must be a JSON object")
    return choice


def _token_ids(response: dict[str, Any]) -> list[int]:
    token_ids = _choice(response).get("token_ids")
    if not isinstance(token_ids, list):
        raise ValueError("Completion response contains no token_ids")
    return [int(token_id) for token_id in token_ids]


def _ready_receipts(run_dir: Path) -> bool:
    try:
        senders = [
            _load_json(run_dir / "sender_receipts" / f"tp_rank_{rank}.json")
            for rank in range(4)
        ]
        receiver_root = run_dir / "receiver_receipts"
        target_dirs = [path for path in receiver_root.iterdir() if path.is_dir()]
        if len(target_dirs) != 1:
            return False
        receivers = [
            _load_json(target_dirs[0] / f"tp_rank_{rank}.json")
            for rank in range(4)
        ]
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return all(receipt.get("status") == "READY" for receipt in senders) and all(
        receipt.get("status") == "TARGET_READY" for receipt in receivers
    )


def _wait_for_manifest(
    path: Path, source_future: Future[dict[str, Any]], deadline: float
) -> dict[str, Any]:
    while not path.exists():
        if source_future.done():
            source_future.result()
            raise RuntimeError("Source request ended before publishing its snapshot")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.05)
    return _load_json(path)


def _wait_for_ready(
    run_dir: Path,
    source_future: Future[dict[str, Any]],
    target_future: Future[dict[str, Any]],
    deadline: float,
) -> None:
    while not _ready_receipts(run_dir):
        if source_future.done():
            source_future.result()
            raise RuntimeError("Source request ended before TP4 became ready")
        if target_future.done():
            target_future.result()
            raise RuntimeError("Target request ended before takeover decision")
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for four TARGET_READY ranks")
        time.sleep(0.02)


def main() -> None:
    args = parse_args()
    manifest_path = args.run_dir / "session_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Phase 7 run directory was already used: {args.run_dir}")
    source_request = _load_json(args.source_request)
    source_request.update(
        {
            "request_id": f"bridgetp-phase7-{args.run_dir.name}",
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
        manifest = _wait_for_manifest(manifest_path, source_future, deadline)
        if manifest.get("phase") != "BridgeTP D3 Phase 7":
            raise ValueError("Source did not publish a Phase 7 manifest")
        minimum_source_tokens = int(manifest["snapshot_num_output_tokens"]) + 64
        if int(source_request.get("max_tokens", 0)) < minimum_source_tokens:
            raise ValueError(
                "Source max_tokens must leave at least 64 post-snapshot tokens: "
                f"{source_request.get('max_tokens')} < {minimum_source_tokens}"
            )
        remaining_tokens = int(source_request["max_tokens"]) - int(
            manifest["snapshot_num_output_tokens"]
        )

        target_request = {
            "model": source_request["model"],
            "request_id": f"bridgetp-phase7-target-{args.run_dir.name}",
            "prompt": manifest["all_known_token_ids"],
            "max_tokens": remaining_tokens,
            "temperature": 0,
            "stream": True,
            "return_token_ids": True,
            "kv_transfer_params": {
                MIGRATION_PARAM: manifest["migration_id"],
            },
        }
        _dump_json(target_request, args.run_dir / "target_request.json")
        target_future = executor.submit(
            _post_streaming_completion,
            args.target_url,
            target_request,
            args.request_timeout_s,
        )
        takeover_identity = {
            "migration_id": manifest["migration_id"],
            "session_token": manifest["session_token"],
            "source_request_id": manifest["source_request_id"],
        }
        try:
            _wait_for_ready(args.run_dir, source_future, target_future, deadline)
        except Exception as ready_error:
            rollback_request = {
                **takeover_identity,
                "action": "rollback",
                "reason": (
                    "automatic pre-commit rollback: "
                    f"{type(ready_error).__name__}: {ready_error}"
                ),
            }
            rollback_response = _post_json(
                args.source_url,
                "/bridge_tp/v1/takeover",
                rollback_request,
                args.request_timeout_s,
            )
            _dump_json(
                rollback_request, args.run_dir / "automatic_rollback_request.json"
            )
            _dump_json(
                rollback_response,
                args.run_dir / "automatic_rollback_response.json",
            )
            try:
                target_future.result()
            except Exception:
                pass
            try:
                source_after_rollback = source_future.result()
                _dump_json(
                    source_after_rollback,
                    args.run_dir / "source_response_after_automatic_rollback.json",
                )
            except Exception as source_error:
                (args.run_dir / "source_error_after_automatic_rollback.txt").write_text(
                    f"{type(source_error).__name__}: {source_error}\n"
                )
            raise RuntimeError(
                "Phase 7 did not reach TARGET_READY; automatic rollback was applied"
            ) from ready_error
        ready_monotonic = time.monotonic()

        action = "rollback" if args.force_rollback_after_ready else "commit"
        takeover_request = {
            **takeover_identity,
            "action": action,
            "reason": "forced pre-commit rollback validation",
        }
        decision_started = time.monotonic()
        decision = _post_json(
            args.source_url,
            "/bridge_tp/v1/takeover",
            takeover_request,
            args.request_timeout_s,
        )
        decision_completed = time.monotonic()
        _dump_json(takeover_request, args.run_dir / "takeover_request.json")
        _dump_json(decision, args.run_dir / "takeover_response.json")

        target_error: str | None = None
        try:
            target_response = target_future.result()
        except Exception as error:
            target_response = {}
            target_error = f"{type(error).__name__}: {error}"
        source_response = source_future.result()

    _dump_json(source_response, args.run_dir / "source_response.json")
    _dump_json(target_response, args.run_dir / "target_response.json")
    if target_error:
        (args.run_dir / "target_error.txt").write_text(target_error + "\n")

    snapshot_output = list(manifest["all_known_token_ids"])[
        int(manifest["num_prompt_tokens"]) :
    ]
    if len(snapshot_output) != int(manifest["snapshot_num_output_tokens"]):
        raise ValueError("Manifest output-token boundary is inconsistent")
    control_request = dict(source_request)
    control_request.pop("request_id", None)
    control_request["max_tokens"] = int(source_request["max_tokens"])
    control_response = _post_completion(
        args.source_url, control_request, args.request_timeout_s
    )
    _dump_json(control_request, args.run_dir / "control_request.json")
    _dump_json(control_response, args.run_dir / "control_response.json")
    control_ids = _token_ids(control_response)
    source_ids = _token_ids(source_response)

    if action == "commit":
        target_ids = [int(token_id) for token_id in target_response["token_ids"]]
        assembled_ids = snapshot_output + target_ids
        source_finish_reason = _choice(source_response).get("finish_reason")
        source_prefix_matches_snapshot = (
            source_ids[: len(snapshot_output)] == snapshot_output
        )
        exact = assembled_ids == control_ids
        first_token_time = target_response.get("first_token_monotonic")
        result = {
            "status": "PASS"
            if (
                decision.get("state") == "COMMITTED"
                and decision.get("source_abort_dispatched")
                and source_finish_reason == "abort"
                and source_prefix_matches_snapshot
                and len(target_ids) == remaining_tokens
                and exact
            )
            else "FAIL",
            "phase": "BridgeTP D3 Phase 7",
            "mode": "commit",
            "migration_id": manifest["migration_id"],
            "takeover_state": decision.get("state"),
            "source_abort_dispatched": decision.get("source_abort_dispatched"),
            "source_finish_reason": source_finish_reason,
            "source_tokens_computed_before_abort": len(source_ids),
            "source_prefix_matches_snapshot": source_prefix_matches_snapshot,
            "discarded_source_tokens_after_snapshot": max(
                0, len(source_ids) - len(snapshot_output)
            ),
            "remaining_generation_budget_transferred": remaining_tokens,
            "target_tokens": len(target_ids),
            "assembled_token_ids": assembled_ids,
            "control_token_ids": control_ids,
            "exact_end_to_end_token_continuity": exact,
            "target_ready_to_commit_response_ms": (
                decision_completed - ready_monotonic
            )
            * 1000,
            "commit_api_ms": (decision_completed - decision_started) * 1000,
            "commit_to_target_first_token_ms": (
                (float(first_token_time) - decision_completed) * 1000
                if first_token_time is not None
                else None
            ),
            "rollback_proven": False,
        }
    else:
        source_finish_reason = _choice(source_response).get("finish_reason")
        exact = source_ids[: len(control_ids)] == control_ids
        result = {
            "status": "PASS"
            if (
                decision.get("state") == "ROLLED_BACK"
                and not decision.get("source_abort_dispatched")
                and source_finish_reason != "abort"
                and target_error is not None
                and exact
            )
            else "FAIL",
            "phase": "BridgeTP D3 Phase 7",
            "mode": "rollback",
            "migration_id": manifest["migration_id"],
            "takeover_state": decision.get("state"),
            "source_abort_dispatched": decision.get("source_abort_dispatched"),
            "source_finish_reason": source_finish_reason,
            "source_tokens_after_rollback": len(source_ids),
            "target_failed_after_rollback": target_error is not None,
            "target_error": target_error,
            "source_prefix_equals_control": exact,
            "rollback_proven": True,
        }
    _dump_json(result, args.run_dir / "takeover_result.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
