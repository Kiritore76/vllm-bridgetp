# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Drive one live TP1-to-TP4 Phase 6 shadow-continuation run."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from vllm.bridge_tp.stream_protocol import MIGRATION_PARAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default="http://127.0.0.1:8001")
    parser.add_argument("--target-url", default="http://127.0.0.1:8200")
    parser.add_argument("--source-request", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--continuation-tokens", type=int, default=32)
    parser.add_argument("--manifest-timeout-s", type=float, default=900)
    parser.add_argument("--request-timeout-s", type=float, default=1800)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return value


def _dump_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def _post_completion(
    base_url: str, payload: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
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
            f"Completion request to {base_url} failed: HTTP {error.code}: {body}"
        ) from error
    if not isinstance(value, dict):
        raise TypeError("Completion response must be a JSON object")
    return value


def _token_ids(response: dict[str, Any]) -> list[int]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Completion response has no choices: {response}")
    token_ids = choices[0].get("token_ids")
    if not isinstance(token_ids, list):
        raise ValueError(
            "Completion response has no token_ids; start requests with "
            "return_token_ids=true"
        )
    return [int(token_id) for token_id in token_ids]


def main() -> None:
    args = parse_args()
    if args.continuation_tokens <= 0:
        raise ValueError("--continuation-tokens must be positive")
    source_request = _load_json(args.source_request)
    source_request.update(
        {
            "temperature": 0,
            "stream": False,
            "return_token_ids": True,
        }
    )
    manifest_path = args.run_dir / "session_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Phase 6 run directory was already used: {args.run_dir}")

    with ThreadPoolExecutor(max_workers=1) as executor:
        source_future = executor.submit(
            _post_completion,
            args.source_url,
            source_request,
            args.request_timeout_s,
        )
        deadline = time.monotonic() + args.manifest_timeout_s
        while not manifest_path.exists():
            if source_future.done():
                source_future.result()
                raise RuntimeError("Source request ended before publishing a snapshot")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {manifest_path}")
            time.sleep(0.1)

        manifest = _load_json(manifest_path)
        required_source_tokens = int(manifest["snapshot_num_output_tokens"]) + int(
            args.continuation_tokens
        )
        if int(source_request.get("max_tokens", 0)) < required_source_tokens:
            raise ValueError(
                "Source max_tokens is too small for snapshot plus comparison: "
                f"{source_request.get('max_tokens')} < {required_source_tokens}"
            )
        target_request = {
            "model": source_request["model"],
            "prompt": manifest["all_known_token_ids"],
            "max_tokens": args.continuation_tokens,
            "temperature": 0,
            "stream": False,
            "return_token_ids": True,
            "kv_transfer_params": {
                MIGRATION_PARAM: manifest["migration_id"],
            },
        }
        _dump_json(source_request, args.run_dir / "source_request.json")
        _dump_json(target_request, args.run_dir / "target_request.json")
        target_response = _post_completion(
            args.target_url, target_request, args.request_timeout_s
        )
        source_response = source_future.result()

    _dump_json(source_response, args.run_dir / "source_response.json")
    _dump_json(target_response, args.run_dir / "target_response.json")
    source_ids = _token_ids(source_response)
    target_ids = _token_ids(target_response)
    start = int(manifest["snapshot_num_output_tokens"])
    expected = source_ids[start : start + args.continuation_tokens]
    exact = expected == target_ids
    result = {
        "status": "PASS" if exact else "FAIL",
        "phase": "BridgeTP D3 Phase 6",
        "scope": "live TP1-to-TP4 shadow continuation; no ownership takeover",
        "migration_id": manifest["migration_id"],
        "source_request_id": manifest["source_request_id"],
        "source_response_id": source_response.get("id"),
        "target_response_id": target_response.get("id"),
        "snapshot_num_output_tokens": start,
        "continuation_tokens": args.continuation_tokens,
        "source_continuation_token_ids": expected,
        "target_continuation_token_ids": target_ids,
        "exact_token_continuity": exact,
    }
    _dump_json(result, args.run_dir / "continuity_result.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
