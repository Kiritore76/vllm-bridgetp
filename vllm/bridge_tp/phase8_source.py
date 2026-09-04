# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TP1 new-KV mirror used by BridgeTP Phase 8.

The Phase 6/7 publisher moves the historical block image once.  This module
then mirrors only newly-computed token slots to a CPU stager while TP1 keeps
decoding.  At the configured cutover boundary it waits for every delta ACK and
publishes an immutable cutover manifest.
"""

from __future__ import annotations

import json
import math
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from vllm.bridge_tp.kv_export import _get_request_block_ids
from vllm.bridge_tp.runtime_control import RuntimeControl
from vllm.bridge_tp.stream_protocol import (
    PROTOCOL_VERSION,
    recv_json,
    send_json,
    send_payload_frames,
    serialize_rank_payload,
    sha256_bytes,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class _DeltaWork:
    start_token: int
    end_token: int
    header: dict[str, Any]
    payload: bytes


@dataclass
class _Phase8SourceState:
    config: Any
    request_id: str
    session_token: str
    last_computed_token: int
    block_size: int
    block_axis: int
    layer_names: list[str]
    queues: list[queue.Queue[_DeltaWork | None]] = field(default_factory=list)
    workers: list[threading.Thread] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    delta_batches: int = 0
    delta_tokens: int = 0
    delta_payload_bytes: int = 0
    d2h_ms: float = 0.0
    finalized: bool = False
    stopped: bool = False
    lifecycle_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def start(self) -> None:
        for rank in range(self.config.target_tp_size):
            work_queue: queue.Queue[_DeltaWork | None] = queue.Queue()
            worker = threading.Thread(
                target=self._worker,
                args=(rank, work_queue),
                name=f"bridgetp-phase8-delta-rank-{rank}",
                daemon=True,
            )
            self.queues.append(work_queue)
            self.workers.append(worker)
            worker.start()
        threading.Thread(
            target=self._watch_cleanup,
            name="bridgetp-phase8-source-cleanup",
            daemon=True,
        ).start()

    def _worker(
        self, rank: int, work_queue: queue.Queue[_DeltaWork | None]
    ) -> None:
        while True:
            work = work_queue.get()
            try:
                if work is None:
                    return
                self._send_delta(rank, work)
            except Exception as error:
                message = (
                    f"rank={rank} tokens=[{getattr(work, 'start_token', '?')},"
                    f"{getattr(work, 'end_token', '?')}): "
                    f"{type(error).__name__}: {error}"
                )
                self.errors.append(message)
                logger.exception("BridgeTP Phase 8 delta publisher failed")
            finally:
                work_queue.task_done()

    def _send_delta(self, rank: int, work: _DeltaWork) -> None:
        deadline = time.monotonic() + self.config.socket_timeout_s
        connection: socket.socket | None = None
        while connection is None:
            try:
                connection = socket.create_connection(
                    (
                        self.config.phase8_delta_host,
                        self.config.phase8_delta_base_port + rank,
                    ),
                    timeout=min(5.0, self.config.socket_timeout_s),
                )
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out connecting to Phase 8 stager rank {rank}"
                    ) from error
                time.sleep(0.02)
        with connection:
            connection.settimeout(self.config.socket_timeout_s)
            send_json(connection, work.header)
            rate_provider = None
            if RuntimeControl.path(self.config.run_dir).exists():
                rate_provider = self._current_per_rank_rate_bytes_s
            transfer = send_payload_frames(
                connection,
                work.payload,
                chunk_bytes=self.config.chunk_bytes,
                rate_provider=rate_provider,
            )
            acknowledgement = recv_json(connection)
            if acknowledgement.get("status") != "STAGED":
                raise RuntimeError(
                    f"Phase 8 stager rank {rank} rejected delta: "
                    f"{acknowledgement}"
                )
        _atomic_json_dump(
            {
                "format_version": 1,
                "phase": "BridgeTP D3 Phase 8",
                "status": "STAGED",
                "migration_id": self.config.migration_id,
                "target_tp_rank": rank,
                "start_token": work.start_token,
                "end_token": work.end_token,
                "staged_unix_s": time.time(),
                **transfer,
            },
            self.config.run_dir
            / "delta_sender_receipts"
            / f"tp_rank_{rank}_{work.start_token}_{work.end_token}.json",
        )

    def _current_per_rank_rate_bytes_s(self) -> float:
        control = RuntimeControl.load(self.config.run_dir)
        aggregate_rate = self.config.aggregate_rate_gib_s
        if control is not None and control.rate_gib_s is not None:
            aggregate_rate = control.rate_gib_s
        if not aggregate_rate:
            return 0.0
        return aggregate_rate * 1024**3 / self.config.target_tp_size

    def enqueue(
        self,
        *,
        start_token: int,
        end_token: int,
        rank_payloads: list[dict[str, torch.Tensor]],
    ) -> bool:
        with self.lifecycle_lock:
            if self.finalized or self.stopped:
                return False
            for rank, layers in enumerate(rank_payloads):
                payload = serialize_rank_payload(
                    {
                        "format_version": 1,
                        "phase": "BridgeTP D3 Phase 8",
                        "migration_id": self.config.migration_id,
                        "source_request_id": self.request_id,
                        "target_tp_rank": rank,
                        "start_token": start_token,
                        "end_token": end_token,
                        "layers": layers,
                    }
                )
                header = {
                    "protocol_version": PROTOCOL_VERSION,
                    "phase": "BridgeTP D3 Phase 8",
                    "migration_id": self.config.migration_id,
                    "session_token": self.session_token,
                    "source_request_id": self.request_id,
                    "target_tp_rank": rank,
                    "start_token": start_token,
                    "end_token": end_token,
                    "payload_bytes": len(payload),
                    "payload_sha256": sha256_bytes(payload),
                    "num_frames": math.ceil(len(payload) / self.config.chunk_bytes),
                    "chunk_bytes": self.config.chunk_bytes,
                }
                self.delta_payload_bytes += len(payload)
                self.queues[rank].put(
                    _DeltaWork(start_token, end_token, header, payload)
                )
            self.delta_batches += 1
            self.delta_tokens += end_token - start_token
            return True

    def wait_for_acks(self) -> None:
        for work_queue in self.queues:
            work_queue.join()
        if self.errors:
            raise RuntimeError("; ".join(self.errors))

    def stop_workers(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        for work_queue in self.queues:
            work_queue.put(None)

    def _watch_cleanup(self) -> None:
        cleanup_path = self.config.run_dir / "cleanup_request.json"
        while not self.finalized and not cleanup_path.exists():
            time.sleep(0.05)
        if not cleanup_path.exists() or self.finalized:
            return
        try:
            # Prevent later decode iterations from enqueueing deltas after the
            # worker queues have been drained.  This matters when Phase 9
            # abandons only the migration and deliberately leaves TP1 serving
            # the request.
            with self.lifecycle_lock:
                self.finalized = True
            self.wait_for_acks()
            self.stop_workers()
            request = json.loads(cleanup_path.read_text(encoding="utf-8"))
            _atomic_json_dump(
                {
                    "format_version": 1,
                    "phase": "BridgeTP D3 Phase 8",
                    "status": "CLEANED",
                    "migration_id": self.config.migration_id,
                    "component": "source_delta_mirror",
                    "reason": request.get("reason", "source ended before cutover"),
                    "delta_batches_drained": self.delta_batches,
                    "delta_tokens_drained": self.delta_tokens,
                    "updated_unix_s": time.time(),
                },
                self.config.run_dir / "source_cleanup_receipt.json",
            )
        except Exception:
            logger.exception("BridgeTP Phase 8 source cleanup failed")


_state: _Phase8SourceState | None = None


def start_phase8_source(
    *,
    config: Any,
    request_id: str,
    session_token: str,
    initial_num_computed_tokens: int,
    block_size: int,
    block_axis: int,
    layer_names: list[str],
) -> None:
    global _state
    if _state is not None:
        raise RuntimeError("BridgeTP Phase 8 source session already exists")
    _state = _Phase8SourceState(
        config=config,
        request_id=request_id,
        session_token=session_token,
        last_computed_token=initial_num_computed_tokens,
        block_size=block_size,
        block_axis=block_axis,
        layer_names=list(layer_names),
    )
    _state.start()


def _token_axis(cache: torch.Tensor, block_axis: int, block_size: int) -> int:
    candidates = [
        axis
        for axis, size in enumerate(cache.shape)
        if axis != block_axis and int(size) == block_size
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Phase 8 requires one unambiguous token axis in each KV cache; "
            f"shape={tuple(cache.shape)}, block_axis={block_axis}, "
            f"block_size={block_size}, candidates={candidates}"
        )
    return candidates[0]


def _copy_delta_rank_shards(
    *,
    state: _Phase8SourceState,
    kv_caches: list[torch.Tensor],
    block_ids: list[int],
    start_token: int,
    end_token: int,
) -> tuple[list[dict[str, torch.Tensor]], float]:
    if len(kv_caches) != len(state.layer_names):
        raise ValueError("Phase 8 layer count differs from the initial snapshot")
    device = kv_caches[0].device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    rank_layers: list[dict[str, torch.Tensor]] = [
        {} for _ in range(state.config.target_tp_size)
    ]
    for layer_name, cache in zip(state.layer_names, kv_caches):
        token_axis = _token_axis(cache, state.block_axis, state.block_size)
        token_slices: list[torch.Tensor] = []
        for token_index in range(start_token, end_token):
            logical_block = token_index // state.block_size
            token_offset = token_index % state.block_size
            physical_block = block_ids[logical_block]
            index: list[int | slice] = [slice(None)] * cache.ndim
            index[state.block_axis] = physical_block
            index[token_axis] = token_offset
            token_slices.append(cache[tuple(index)].detach().cpu())
        delta = torch.stack(token_slices, dim=0)
        delta_head_axis = state.config.head_axis + 1
        for removed_axis in sorted((state.block_axis, token_axis)):
            if removed_axis < state.config.head_axis:
                delta_head_axis -= 1
        if int(delta.shape[delta_head_axis]) != state.config.expected_kv_heads:
            raise ValueError(
                f"Phase 8 source KV-head count differs for {layer_name}: "
                f"{delta.shape[delta_head_axis]} != "
                f"{state.config.expected_kv_heads}"
            )
        shards = torch.chunk(
            delta, state.config.target_tp_size, dim=delta_head_axis
        )
        if len(shards) != state.config.target_tp_size:
            raise ValueError("Phase 8 could not split all target TP ranks")
        for rank, shard in enumerate(shards):
            rank_layers[rank][layer_name] = shard.contiguous()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return rank_layers, (time.perf_counter() - started) * 1000


def maybe_publish_phase8_delta(
    *,
    config: Any,
    request_id: str,
    kv_caches: list[torch.Tensor],
    requests: dict[str, Any],
    input_batch: Any,
    scheduler_output: Any,
    cache_dtype: str,
    attn_groups: list[list[Any]],
) -> None:
    del cache_dtype, attn_groups
    state = _state
    if state is None or state.finalized or state.request_id != request_id:
        return
    request_index = input_batch.req_id_to_index[request_id]
    request = requests[request_id]
    output_tokens = len(request.output_token_ids)
    num_scheduled = int(scheduler_output.num_scheduled_tokens.get(request_id, 0))
    num_computed = (
        int(input_batch.num_computed_tokens_cpu[request_index]) + num_scheduled
    )
    if num_computed <= state.last_computed_token:
        return
    if output_tokens > config.phase8_cutover_output_tokens:
        raise ValueError("Phase 8 skipped its exact cutover output-token boundary")
    block_ids, block_size = _get_request_block_ids(
        input_batch, request_index, num_computed
    )
    if block_size != state.block_size:
        raise ValueError("Phase 8 block size changed during generation")
    start_token = state.last_computed_token
    rank_payloads, d2h_ms = _copy_delta_rank_shards(
        state=state,
        kv_caches=kv_caches,
        block_ids=block_ids,
        start_token=start_token,
        end_token=num_computed,
    )
    state.d2h_ms += d2h_ms
    enqueued = state.enqueue(
        start_token=start_token,
        end_token=num_computed,
        rank_payloads=rank_payloads,
    )
    if not enqueued:
        return
    state.last_computed_token = num_computed
    if output_tokens != config.phase8_cutover_output_tokens:
        return

    num_known = int(request.num_tokens)
    pending = num_known - num_computed
    if pending != 1:
        raise ValueError(
            f"Phase 8 cutover requires one pending token, observed {pending}"
        )
    state.wait_for_acks()
    known_token_ids = [request.get_token_id(i) for i in range(num_known)]
    state.finalized = True
    state.stop_workers()
    _atomic_json_dump(
        {
            "format_version": 1,
            "phase": "BridgeTP D3 Phase 8",
            "scope": "old-KV snapshot plus acknowledged new-KV deltas",
            "protocol_version": PROTOCOL_VERSION,
            "migration_id": config.migration_id,
            "session_token": state.session_token,
            "source_request_id": request_id,
            "cutover_num_output_tokens": output_tokens,
            "num_prompt_tokens": int(request.num_prompt_tokens),
            "num_computed_tokens": num_computed,
            "pending_known_tokens": pending,
            "computed_token_ids": known_token_ids[:num_computed],
            "pending_token_ids": known_token_ids[num_computed:],
            "all_known_token_ids": known_token_ids,
            "num_blocks": math.ceil(num_computed / state.block_size),
            "block_size": state.block_size,
            "block_axis": state.block_axis,
            "delta_start_token": int(
                json.loads(
                    (config.run_dir / "session_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["num_computed_tokens"]
            ),
            "delta_end_token": num_computed,
            "delta_batches": state.delta_batches,
            "delta_tokens": state.delta_tokens,
            "delta_payload_bytes": state.delta_payload_bytes,
            "delta_d2h_ms": state.d2h_ms,
            "updated_unix_s": time.time(),
        },
        config.run_dir / "cutover_manifest.json",
    )
    logger.warning(
        "BridgeTP Phase 8 cutover prepared at output=%d, computed=%d, "
        "delta_tokens=%d",
        output_tokens,
        num_computed,
        state.delta_tokens,
    )
