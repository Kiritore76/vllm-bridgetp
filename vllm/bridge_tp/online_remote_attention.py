# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fail-closed online split-KV attention used by the Bridge experiment.

The TP1 worker keeps the suffix of the anchor request and sends only the
already-rotated query-head shard to each TP4 worker.  TP4 evaluates stable
softmax statistics over the contiguous, GPU-resident prefix.  TP1 evaluates
the suffix, merges both partitions, and returns that value to the model's
normal output projection.  Consequently this is part of the token-generation
data path; it is not a sidecar timing workload.

The first implementation is deliberately narrow: one decode token, one
explicit anchor request, standard BF16/FP16 paged KV, TP1 -> TP4, and a
block-aligned prefix.  Any violation fails closed when strict mode is enabled.
"""

from __future__ import annotations

import json
import math
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.bridge_tp.controller.anchor_selector import select_source_request_id
from vllm.bridge_tp.stream_protocol import (
    deserialize_rank_payload,
    recv_json,
    recv_payload_frames,
    send_json,
    send_payload_frames,
    serialize_rank_payload,
    sha256_bytes,
)


def gather_paged_kv(
    cache: torch.Tensor,
    block_ids: list[int],
    start_token: int,
    end_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ``[start, end)`` as ``[kv_heads, tokens, head_dim]``."""
    if cache.ndim != 5 or int(cache.shape[1]) != 2:
        raise ValueError(f"unsupported paged-KV shape {tuple(cache.shape)}")
    if not 0 <= start_token <= end_token:
        raise ValueError("invalid paged-KV token range")
    block_size = int(cache.shape[2])
    if math.ceil(end_token / block_size) > len(block_ids):
        raise ValueError("block table does not cover requested KV range")
    heads = int(cache.shape[3])
    dim = int(cache.shape[4])
    if start_token == end_token:
        empty = cache.new_empty((heads, 0, dim))
        return empty, empty
    logical_tokens = torch.arange(
        start_token, end_token, dtype=torch.long, device=cache.device
    )
    logical_blocks = torch.div(logical_tokens, block_size, rounding_mode="floor")
    block_table = torch.tensor(block_ids, dtype=torch.long, device=cache.device)
    physical_blocks = block_table[logical_blocks]
    offsets = logical_tokens.remainder(block_size)
    key = cache[physical_blocks, 0, offsets].permute(1, 0, 2)
    value = cache[physical_blocks, 1, offsets].permute(1, 0, 2)
    return key, value


def attention_stats(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-head maximum, denominator and unnormalised numerator."""
    if query.ndim != 2 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("split attention requires Q[H,D], KV[Hkv,T,D]")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and KV head dimensions differ")
    if query.shape[0] % key.shape[0]:
        raise ValueError("query heads are not divisible by KV heads")
    if key.shape[1] == 0:
        heads = query.shape[0]
        return (
            torch.full((heads,), -torch.inf, device=query.device),
            torch.zeros((heads,), dtype=torch.float32, device=query.device),
            torch.zeros_like(query, dtype=torch.float32),
        )
    repeats = query.shape[0] // key.shape[0]
    expanded_key = key.repeat_interleave(repeats, dim=0).float()
    expanded_value = value.repeat_interleave(repeats, dim=0).float()
    scores = torch.einsum("hd,htd->ht", query.float(), expanded_key)
    scores.mul_(float(scale))
    maximum = scores.max(dim=-1).values
    exponentials = torch.exp(scores - maximum.unsqueeze(-1))
    denominator = exponentials.sum(dim=-1)
    numerator = torch.einsum("ht,htd->hd", exponentials, expanded_value)
    return maximum, denominator, numerator


def merge_attention_stats(
    local: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    remote: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    local_m, local_l, local_o = local
    remote_m, remote_l, remote_o = remote
    maximum = torch.maximum(local_m, remote_m)
    local_weight = torch.exp(local_m - maximum)
    remote_weight = torch.exp(remote_m - maximum)
    denominator = local_weight * local_l + remote_weight * remote_l
    if bool(torch.any(denominator <= 0)):
        raise ValueError("split attention produced a non-positive denominator")
    numerator = local_weight.unsqueeze(-1) * local_o
    numerator.add_(remote_weight.unsqueeze(-1) * remote_o)
    return numerator / denominator.unsqueeze(-1)


def _send_tensor_payload(connection: socket.socket, value: dict[str, Any]) -> int:
    payload = serialize_rank_payload(value)
    header = {
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "num_frames": math.ceil(len(payload) / (1024 * 1024)),
    }
    send_json(connection, header)
    send_payload_frames(connection, payload, chunk_bytes=1024 * 1024)
    return len(payload)


def _recv_tensor_payload(connection: socket.socket) -> tuple[dict[str, Any], int]:
    header = recv_json(connection)
    payload, _ = recv_payload_frames(
        connection,
        payload_bytes=int(header["payload_bytes"]),
        payload_sha256=str(header["payload_sha256"]),
        num_frames=int(header["num_frames"]),
    )
    return deserialize_rank_payload(payload), len(payload)


@dataclass(frozen=True)
class RemoteAttentionConfig:
    run_dir: Path
    host: str
    base_port: int
    timeout_s: float
    source_request_id_prefix: str
    strict: bool = True
    target_tp_size: int = 4

    @classmethod
    def from_env(cls) -> RemoteAttentionConfig | None:
        if os.getenv("BRIDGETP_ONLINE_REMOTE_ATTENTION", "0").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return None
        run_dir = os.getenv("BRIDGETP_STREAM_RUN_DIR", "").strip()
        prefix = os.getenv("BRIDGETP_STREAM_SOURCE_REQUEST_ID_PREFIX", "").strip()
        if not run_dir or not prefix:
            raise ValueError("online remote attention requires run dir and anchor ID")
        return cls(
            run_dir=Path(run_dir),
            host=os.getenv("BRIDGETP_REMOTE_ATTENTION_HOST", "127.0.0.1"),
            base_port=int(os.getenv("BRIDGETP_REMOTE_ATTENTION_BASE_PORT", "30200")),
            timeout_s=float(os.getenv("BRIDGETP_REMOTE_ATTENTION_TIMEOUT_S", "30")),
            source_request_id_prefix=prefix,
            strict=os.getenv("BRIDGETP_REMOTE_ATTENTION_STRICT", "1").lower()
            not in {"0", "false", "no", "off"},
        )


class RemoteAttentionRankServer:
    """Serve one TP4 rank's real GPU-resident KV prefix."""

    def __init__(
        self,
        *,
        config: RemoteAttentionConfig,
        rank: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        kv_lock: threading.RLock,
    ) -> None:
        self.config = config
        self.rank = rank
        self.kv_caches = kv_caches
        self.block_ids = block_ids
        self.kv_lock = kv_lock
        self.error: BaseException | None = None
        self.calls = 0
        self.thread = threading.Thread(
            target=self._run,
            name=f"bridgetp-remote-attention-rank-{rank}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.config.host, self.config.base_port + self.rank))
            listener.listen(64)
            listener.settimeout(0.2)
            while True:
                control = self.config.run_dir / "takeover_state.json"
                if control.is_file():
                    state = json.loads(control.read_text(encoding="utf-8")).get("state")
                    if state in {"COMMITTED", "ROLLED_BACK", "CANCELLED"}:
                        return
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(self.config.timeout_s)
                    while True:
                        try:
                            self._serve_one(connection)
                        except ConnectionError:
                            break
        except BaseException as error:
            self.error = error
        finally:
            listener.close()

    def _serve_one(self, connection: socket.socket) -> None:
        request, request_bytes = _recv_tensor_payload(connection)
        if int(request.get("rank", -1)) != self.rank:
            raise ValueError("remote-attention request reached the wrong TP4 rank")
        layer_name = str(request["layer_name"])
        boundary = int(request["boundary"])
        if boundary <= 0 or boundary % int(request["block_size"]):
            raise ValueError("remote-attention boundary is not a positive block edge")
        cache = self.kv_caches.get(layer_name)
        if cache is None:
            raise KeyError(f"unknown target KV layer {layer_name!r}")
        query = request["query"].to(device=cache.device, dtype=cache.dtype)
        started = time.perf_counter()
        with self.kv_lock:
            key, value = gather_paged_kv(cache, self.block_ids, 0, boundary)
            stats = attention_stats(query, key, value, float(request["scale"]))
            if cache.device.type == "cuda":
                torch.cuda.synchronize(cache.device)
        response_bytes = _send_tensor_payload(
            connection,
            {
                "rank": self.rank,
                "boundary": boundary,
                "maximum": stats[0].cpu(),
                "denominator": stats[1].cpu(),
                "numerator": stats[2].cpu(),
            },
        )
        self.calls += 1
        send_json(
            connection,
            {
                "status": "PASS",
                "compute_ms": (time.perf_counter() - started) * 1000,
                "request_bytes": request_bytes,
                "response_bytes": response_bytes,
            },
        )


class RemoteAttentionClient:
    def __init__(self, config: RemoteAttentionConfig) -> None:
        self.config = config
        self._metrics_lock = threading.Lock()
        self._connections: dict[int, socket.socket] = {}
        self._connection_locks = [
            threading.Lock() for _ in range(config.target_tp_size)
        ]
        self.executor = ThreadPoolExecutor(max_workers=config.target_tp_size)
        self.verified_layers: set[str] = set()

    def _connection(self, rank: int) -> socket.socket:
        existing = self._connections.get(rank)
        if existing is not None:
            return existing
        deadline = time.monotonic() + self.config.timeout_s
        while True:
            try:
                connection = socket.create_connection(
                    (self.config.host, self.config.base_port + rank), timeout=1.0
                )
                connection.settimeout(self.config.timeout_s)
                self._connections[rank] = connection
                return connection
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out connecting to remote-attention rank {rank}"
                    ) from error
                time.sleep(0.005)

    def common_boundary(self) -> int:
        boundaries: list[int] = []
        for rank in range(self.config.target_tp_size):
            rank_dir = self.config.run_dir / "gpu_block_receipts" / f"tp_rank_{rank}"
            logical = 0
            end = 0
            while (rank_dir / f"block_{logical:012d}.json").is_file():
                record = json.loads(
                    (rank_dir / f"block_{logical:012d}.json").read_text(
                        encoding="utf-8"
                    )
                )
                end = int(record["end_token"])
                logical += 1
            watermark = self.config.run_dir / "gpu_watermarks" / f"tp_rank_{rank}.json"
            if watermark.is_file():
                end = max(
                    end,
                    int(json.loads(watermark.read_text(encoding="utf-8"))["end_token"]),
                )
            boundaries.append(end)
        if len(boundaries) != self.config.target_tp_size:
            return 0
        manifest = self.config.run_dir / "session_manifest.json"
        if not manifest.is_file():
            return 0
        block_size = int(json.loads(manifest.read_text(encoding="utf-8"))["block_size"])
        return min(boundaries) // block_size * block_size

    def rank_stats(
        self,
        *,
        rank: int,
        layer_name: str,
        query: torch.Tensor,
        boundary: int,
        block_size: int,
        scale: float,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, Any]]:
        started = time.perf_counter()
        with self._connection_locks[rank]:
            connection = self._connection(rank)
            connection.settimeout(self.config.timeout_s)
            request_bytes = _send_tensor_payload(
                connection,
                {
                    "rank": rank,
                    "layer_name": layer_name,
                    "boundary": boundary,
                    "block_size": block_size,
                    "scale": scale,
                    "query": query.detach().cpu(),
                },
            )
            response, response_bytes = _recv_tensor_payload(connection)
            trailer = recv_json(connection)
        if trailer.get("status") != "PASS" or int(response["boundary"]) != boundary:
            raise RuntimeError("TP4 rejected the remote-attention request")
        stats = (
            response["maximum"].to(query.device),
            response["denominator"].to(query.device),
            response["numerator"].to(query.device),
        )
        return stats, {
            "rank": rank,
            "round_trip_ms": (time.perf_counter() - started) * 1000,
            "target_compute_ms": float(trailer["compute_ms"]),
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
        }

    def record(self, value: dict[str, Any]) -> None:
        path = self.config.run_dir / "online_remote_attention.jsonl"
        with self._metrics_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(value, separators=(",", ":")) + "\n")


_client: RemoteAttentionClient | None | bool = False


def get_remote_attention_client() -> RemoteAttentionClient | None:
    global _client
    if _client is False:
        config = RemoteAttentionConfig.from_env()
        _client = RemoteAttentionClient(config) if config is not None else None
    return _client if isinstance(_client, RemoteAttentionClient) else None


def maybe_run_online_remote_attention(
    *,
    layer_name: str,
    layer: Any,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
) -> bool:
    """Replace the anchor row with a real TP1-suffix/TP4-prefix result."""
    client = get_remote_attention_client()
    if client is None:
        return False
    marker = client.config.run_dir / "remote_attention_bridge.json"
    if not marker.is_file():
        return False
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    req_ids = context.additional_kwargs.get("bridgetp_request_ids", [])
    request_ids = [str(request_id) for request_id in req_ids]
    selected_request_id = select_source_request_id(
        request_ids,
        client.config.source_request_id_prefix,
    )
    matches = (
        [request_ids.index(selected_request_id)]
        if selected_request_id is not None
        else []
    )
    if len(matches) != 1:
        if client.config.strict:
            raise RuntimeError(
                "expected one Bridge anchor row; "
                f"configured_prefix={client.config.source_request_id_prefix!r}, "
                f"request_ids={request_ids!r}, matched_rows={matches!r}"
            )
        return False
    if len(req_ids) != 1:
        raise RuntimeError(
            "online remote attention requires an anchor-only TP1 decode batch"
        )
    request_index = matches[0]
    query_starts = attn_metadata.query_start_loc.tolist()
    start = int(query_starts[request_index])
    end = int(query_starts[request_index + 1])
    if end - start != 1:
        raise RuntimeError("online remote attention supports one-token decode only")
    boundary = client.common_boundary()
    sequence_length = int(attn_metadata.seq_lens[request_index].item())
    if boundary <= 0 or boundary >= sequence_length:
        return False
    block_ids = [int(value) for value in attn_metadata.block_table[request_index].tolist()]
    block_size = int(kv_cache.shape[2])
    anchor_query = query[start]
    heads = int(anchor_query.shape[0])
    if heads % client.config.target_tp_size:
        raise ValueError("query heads do not divide over TP4")
    rank_queries = list(torch.chunk(anchor_query, client.config.target_tp_size, dim=0))
    started = time.perf_counter()
    futures = [
        client.executor.submit(
            client.rank_stats,
            rank=rank,
            layer_name=layer_name,
            query=rank_query,
            boundary=boundary,
            block_size=block_size,
            scale=float(layer.impl.scale),
        )
        for rank, rank_query in enumerate(rank_queries)
    ]
    responses = [future.result() for future in futures]
    remote = tuple(
        torch.cat([response[0][index] for response in responses], dim=0)
        for index in range(3)
    )
    local_key, local_value = gather_paged_kv(
        kv_cache, block_ids, boundary, sequence_length
    )
    local = attention_stats(anchor_query, local_key, local_value, float(layer.impl.scale))
    merged = merge_attention_stats(local, remote)
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    cosine_similarity: float | None = None
    if layer_name not in client.verified_layers:
        full_key, full_value = gather_paged_kv(
            kv_cache, block_ids, 0, sequence_length
        )
        full = attention_stats(
            anchor_query, full_key, full_value, float(layer.impl.scale)
        )
        reference = full[2] / full[1].unsqueeze(-1)
        difference = (merged.float() - reference).abs()
        max_abs_error = float(difference.max().item())
        mean_abs_error = float(difference.mean().item())
        cosine_similarity = float(
            torch.nn.functional.cosine_similarity(
                merged.float().reshape(1, -1), reference.reshape(1, -1)
            ).item()
        )
        max_tolerance = float(
            os.getenv("BRIDGETP_REMOTE_ATTENTION_MAX_ABS_TOLERANCE", "0.02")
        )
        cosine_tolerance = float(
            os.getenv("BRIDGETP_REMOTE_ATTENTION_MIN_COSINE", "0.999")
        )
        if (
            max_abs_error > max_tolerance
            or cosine_similarity < cosine_tolerance
        ):
            raise RuntimeError(
                "online remote attention differs from full local reference: "
                f"max_abs={max_abs_error}, cosine={cosine_similarity}"
            )
        client.verified_layers.add(layer_name)
    output[start].copy_(merged.to(dtype=output.dtype))
    elapsed_ms = (time.perf_counter() - started) * 1000
    client.record(
        {
            "format_version": 1,
            "status": "PASS",
            "unix_s": time.time(),
            "layer_name": layer_name,
            "request_id": req_ids[request_index],
            "sequence_tokens": sequence_length,
            "remote_prefix_tokens": boundary,
            "local_suffix_tokens": sequence_length - boundary,
            "total_ms": elapsed_ms,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "cosine_similarity": cosine_similarity,
            "ranks": [response[1] for response in responses],
        }
    )
    return True
