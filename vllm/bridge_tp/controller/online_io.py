# SPDX-License-Identifier: Apache-2.0
"""Online request and response plumbing for the Phase 9 controller."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .response_proxy import EmittedToken, ProxyMode, ResponseProxy
from .sampling_contract import (
    freeze_strict_greedy_sampling,
    strict_greedy_sampling_errors,
)

TokenSink = Callable[[int, int, float], None]
MIGRATION_PARAM = "bridgetp_migration_id"


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def atomic_json_dump(value: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def post_streaming_completion(
    base_url: str,
    payload: dict[str, Any],
    timeout_s: float,
    token_sink: TokenSink,
) -> dict[str, Any]:
    """Consume one OpenAI-compatible SSE completion and expose each token."""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    token_ids: list[int] = []
    chunks: list[dict[str, Any]] = []
    response_id: str | None = None
    finish_reason: str | None = None
    first_token_monotonic: float | None = None
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
                    raise RuntimeError(f"streaming completion error: {chunk['error']}")
                chunks.append(chunk)
                response_id = chunk.get("id", response_id)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                for raw_token_id in choice.get("token_ids") or []:
                    token_id = int(raw_token_id)
                    if first_token_monotonic is None:
                        first_token_monotonic = time.monotonic()
                    index = len(token_ids)
                    token_ids.append(token_id)
                    token_sink(index, token_id, time.time())
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
    except urllib.error.HTTPError as error:  # pragma: no cover - server path
        body = error.read().decode(errors="replace")
        raise RuntimeError(
            f"completion request to {base_url} failed: HTTP {error.code}: {body}"
        ) from error
    except OSError as error:  # pragma: no cover - server path
        raise RuntimeError(
            f"completion request to {base_url} failed: {error}"
        ) from error
    if not saw_done or finish_reason is None:
        raise RuntimeError("streaming response ended before completion")
    return {
        "response_id": response_id,
        "token_ids": token_ids,
        "finish_reason": finish_reason,
        "first_token_monotonic": first_token_monotonic,
        "chunks": chunks,
    }


class ProxyRecorder:
    """Thread-safe bridge from two SSE readers to one response seam."""

    def __init__(
        self,
        external_request_id: str,
        mode: ProxyMode,
        emission_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.proxy = ResponseProxy(external_request_id, mode)
        self._emission_sink = emission_sink
        self._lock = threading.Lock()
        self._target_offset: int | None = None
        self._pending_target: list[tuple[int, int, float]] = []

    def on_source_token(self, index: int, token_id: int, unix_s: float) -> None:
        with self._lock:
            self._publish(self.proxy.on_source_token(index, token_id, unix_s))

    def set_cutover(self, cutover_index: int, unix_s: float) -> None:
        with self._lock:
            self.proxy.set_cutover(cutover_index, unix_s)
            self._target_offset = cutover_index

    def on_target_token(self, index: int, token_id: int, unix_s: float) -> None:
        with self._lock:
            if self._target_offset is None:
                raise RuntimeError("target stream started before cutover was set")
            global_index = self._target_offset + index
            if not self.proxy.committed:
                # The target observes the server-side COMMITTED barrier before
                # the controller receives the commit HTTP response. Buffer this
                # narrow acknowledgement race, then replay after on_commit().
                self._pending_target.append((global_index, token_id, unix_s))
                return
            self._publish(self.proxy.on_target_token(global_index, token_id, unix_s))

    def on_commit(self, unix_s: float) -> None:
        with self._lock:
            self.proxy.on_commit(unix_s)
            for index, token_id, _token_unix_s in self._pending_target:
                self._publish(self.proxy.on_target_token(index, token_id, unix_s))
            self._pending_target.clear()

    def on_rollback(self, unix_s: float, reason: str) -> None:
        with self._lock:
            self._pending_target.clear()
            self._publish(self.proxy.on_rollback(unix_s, reason))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return self.proxy.stats()

    def _publish(self, emitted: list[EmittedToken]) -> None:
        if self._emission_sink is None:
            return
        for token in emitted:
            self._emission_sink(
                {
                    "index": token.index,
                    "token_id": token.token_id,
                    "unix_s": token.unix_s,
                    "origin": token.origin,
                }
            )


def build_target_request(
    source_request: dict[str, Any],
    staging: dict[str, Any],
    run_name: str,
) -> tuple[dict[str, Any], int]:
    source_errors = strict_greedy_sampling_errors(source_request)
    if source_errors:
        raise ValueError(
            "source request does not carry the Phase 9 sampling contract: "
            + "; ".join(source_errors)
        )
    cutover = int(staging["snapshot_num_output_tokens"])
    remaining = int(source_request["max_tokens"]) - cutover
    if remaining <= 0:
        raise ValueError("source max_tokens leaves no post-cutover target tokens")
    target = freeze_strict_greedy_sampling(
        {
            "model": source_request["model"],
            "request_id": f"bridgetp-phase9-target-{run_name}",
            "prompt": staging["all_known_token_ids"],
            "max_tokens": remaining,
            "ignore_eos": bool(source_request.get("ignore_eos", False)),
            "stream": True,
            "return_token_ids": True,
            "kv_transfer_params": {
                MIGRATION_PARAM: staging["migration_id"],
            },
        }
    )
    if "logprobs" in source_request:
        target["logprobs"] = int(source_request["logprobs"])
    return target, cutover


def honored_generation(path: str | Path) -> int | None:
    try:
        value = load_json(path)
        if int(value.get("format_version", -1)) != 1:
            return None
        return int(value["generation"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
