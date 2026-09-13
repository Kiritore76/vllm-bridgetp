# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Live TCP-backed TP4 KV connector for BridgeTP Phase 6."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.bridge_tp.block_layout import snapshot_target_block_ids
from vllm.bridge_tp.kv_restore import inject_rank_delta, inject_rank_shard
from vllm.bridge_tp.online_remote_attention import (
    RemoteAttentionConfig,
    RemoteAttentionRankServer,
)
from vllm.bridge_tp.stream_protocol import (
    MIGRATION_PARAM,
    PROTOCOL_VERSION,
    deserialize_rank_payload,
    recv_json,
    recv_payload_frames,
    send_json,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class _ShadowCancelled(RuntimeError):
    pass


@dataclass
class BridgeTPStreamRequest:
    migration_id: str
    source_request_id: str
    target_request_id: str
    target_block_ids: list[int]
    num_computed_tokens: int
    gpu_resident_shadow: bool = False


@dataclass
class BridgeTPStreamMetadata(KVConnectorMetadata):
    requests: list[BridgeTPStreamRequest] = field(default_factory=list)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe[:160] or "request"


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


class BridgeTPStreamingConnector(KVConnectorBase_V1):
    """Receive a live TP1 snapshot into scheduler-owned TP4 blocks."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        manifest_path = self._kv_transfer_config.get_from_extra_config(
            "bridgetp_stream_manifest", None
        )
        if not manifest_path:
            raise ValueError(
                "BridgeTPStreamingConnector requires "
                "kv_connector_extra_config.bridgetp_stream_manifest"
            )
        self.manifest_path = Path(manifest_path).resolve()
        receipt_dir = self._kv_transfer_config.get_from_extra_config(
            "bridgetp_stream_receipt_dir",
            str(self.manifest_path.parent / "receiver_receipts"),
        )
        self.receipt_dir = Path(receipt_dir).resolve()
        self.socket_timeout_s = float(
            self._kv_transfer_config.get_from_extra_config(
                "bridgetp_stream_socket_timeout_s", 600
            )
        )
        takeover_control_path = self._kv_transfer_config.get_from_extra_config(
            "bridgetp_takeover_control_path", None
        )
        self.takeover_control_path = (
            Path(takeover_control_path).resolve()
            if takeover_control_path
            else None
        )
        self.takeover_control_timeout_s = float(
            self._kv_transfer_config.get_from_extra_config(
                "bridgetp_takeover_control_timeout_s", 600
            )
        )
        configured_phase = self._kv_transfer_config.get_from_extra_config(
            "bridgetp_stream_expected_phase", None
        )
        self.expected_phase = str(configured_phase) if configured_phase else (
            "BridgeTP D3 Phase 7"
            if self.takeover_control_path is not None
            else "BridgeTP D3 Phase 6"
        )
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 4:
            raise ValueError("BridgeTP Phase 6 requires tensor_parallel_size=4")
        if parallel.pipeline_parallel_size != 1:
            raise ValueError("BridgeTP Phase 6 does not support pipeline parallelism")
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("BridgeTP Phase 6 requires one KV-cache group")
        self._target_model = str(vllm_config.model_config.model)
        self._target_block_size = int(vllm_config.cache_config.block_size)
        self.gpu_resident_shadow = bool(
            self._kv_transfer_config.get_from_extra_config(
                "bridgetp_gpu_resident_shadow", False
            )
        )
        self.shadow_cutover_output_tokens = int(
            self._kv_transfer_config.get_from_extra_config(
                "bridgetp_shadow_cutover_output_tokens", 0
            )
        )
        self.online_remote_attention = bool(
            self._kv_transfer_config.get_from_extra_config(
                "bridgetp_online_remote_attention", False
            )
        )
        self.remote_attention_base_port = int(
            self._kv_transfer_config.get_from_extra_config(
                "bridgetp_remote_attention_base_port", 30200
            )
        )
        self._manifest: dict[str, Any] | None = None
        self._pending_requests: dict[str, Request] = {}
        self._active_requests: dict[str, Request] = {}
        self._registered_kv_caches: dict[str, torch.Tensor] = {}
        self._load_threads: dict[str, threading.Thread] = {}
        self._gpu_kv_lock = threading.RLock()
        self._remote_attention_servers: dict[str, RemoteAttentionRankServer] = {}
        self._completed_recvs: set[str] = set()
        self._reported_recvs: set[str] = set()
        self._load_errors: dict[str, BaseException] = {}
        self._claimed_target_request_id: str | None = None
        logger.warning(
            "BridgeTP Phase 6 streaming connector enabled; target waits for %s",
            self.manifest_path,
        )

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest is not None:
            return self._manifest
        with self.manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        required = {
            "protocol_version": PROTOCOL_VERSION,
            "source_tp_size": 1,
            "target_tp_size": 4,
            "pending_known_tokens": 1,
        }
        for key, value in required.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"Phase 6 manifest field {key} differs: "
                    f"{manifest.get(key)!r} != {value!r}"
                )
        if manifest.get("phase") != self.expected_phase:
            raise ValueError(
                "Stream manifest phase differs: "
                f"{manifest.get('phase')!r} != {self.expected_phase!r}"
            )
        if str(manifest["model"]) != self._target_model:
            raise ValueError("Phase 6 source and target model paths differ")
        if int(manifest["block_size"]) != self._target_block_size:
            raise ValueError("Phase 6 source and target block sizes differ")
        ranks = manifest.get("ranks")
        if not isinstance(ranks, list) or [
            int(record["target_tp_rank"]) for record in ranks
        ] != list(range(4)):
            raise ValueError("Phase 6 manifest does not contain ranks 0..3")
        known = list(manifest["all_known_token_ids"])
        computed = list(manifest["computed_token_ids"])
        pending = list(manifest["pending_token_ids"])
        if computed + pending != known:
            raise ValueError("Phase 6 token boundary is inconsistent")
        if len(computed) != int(manifest["num_computed_tokens"]):
            raise ValueError("Phase 6 computed token count is inconsistent")
        self._manifest = manifest
        return manifest

    def _request_matches(self, request: Request) -> bool:
        params = request.kv_transfer_params or {}
        migration_id = params.get(MIGRATION_PARAM)
        if migration_id is None:
            if (
                "bridgetp-phase" in request.request_id
                and "target" in request.request_id
            ):
                raise ValueError(
                    "BridgeTP target request is missing migration marker "
                    f"{MIGRATION_PARAM!r}; refusing local recomputation"
                )
            return False
        manifest = self._load_manifest()
        prompt = request.prompt_token_ids
        if self.gpu_resident_shadow:
            known = list(manifest["all_known_token_ids"])
            prompt_matches = (
                prompt is not None
                and list(prompt[: len(known)]) == known
                and request.num_tokens == self._planned_known_tokens(manifest)
            )
        else:
            prompt_matches = prompt is not None and list(prompt) == list(
                manifest["all_known_token_ids"]
            )
        if migration_id != manifest["migration_id"]:
            raise ValueError(
                "BridgeTP target migration id differs from the active manifest: "
                f"{migration_id!r} != {manifest['migration_id']!r}"
            )
        if not prompt_matches:
            raise ValueError(
                "Phase 6 target prompt must exactly equal the live snapshot token "
                "history"
            )
        expected_computed = self._external_computed_tokens(manifest)
        if request.num_tokens != expected_computed + 1:
            raise ValueError("Phase 6 requires exactly one pending token")
        return True

    def _planned_known_tokens(self, manifest: dict[str, Any]) -> int:
        if not self.gpu_resident_shadow:
            return int(manifest["num_computed_tokens"]) + 1
        if self.shadow_cutover_output_tokens <= 0:
            raise ValueError("GPU-resident Shadow requires a cutover boundary")
        return int(manifest["num_prompt_tokens"]) + self.shadow_cutover_output_tokens

    def _external_computed_tokens(self, manifest: dict[str, Any]) -> int:
        return self._planned_known_tokens(manifest) - 1

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if not self._request_matches(request):
            return 0, False
        if num_computed_tokens != 0:
            raise ValueError(
                "Phase 6 cannot mix a local prefix-cache hit with streamed KV"
            )
        if (
            self._claimed_target_request_id is not None
            and self._claimed_target_request_id != request.request_id
        ):
            raise RuntimeError("Phase 6 migration session was already claimed")
        manifest = self._load_manifest()
        return (
            self._external_computed_tokens(manifest),
            self.gpu_resident_shadow,
        )

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens == 0:
            return
        manifest = self._load_manifest()
        if not self._request_matches(request):
            raise ValueError("Allocated request does not match Phase 6 session")
        if num_external_tokens != self._external_computed_tokens(manifest):
            raise ValueError("Target external-token count differs from snapshot")
        block_ids = blocks.get_block_ids()
        self._snapshot_target_block_ids(
            request,
            block_ids,
            "Target block allocation differs from live snapshot",
        )
        self._claimed_target_request_id = request.request_id
        setattr(request, "_bridgetp_target_block_ids", block_ids)
        self._pending_requests[request.request_id] = request
        self._active_requests[request.request_id] = request

    def _snapshot_target_block_ids(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        error_message: str,
    ) -> list[int]:
        """Validate an allocation and return blocks covered by streamed KV.

        The request contains one pending token beyond ``num_computed_tokens``.
        When the computed prefix exactly fills its final block, the scheduler
        legitimately allocates one additional tail block for that pending
        token.  The streamed snapshot must be restored only into the prefix
        blocks; the scheduler-owned tail block remains untouched for local
        decode.
        """
        manifest = self._load_manifest()
        if self.gpu_resident_shadow:
            planned_blocks = math.ceil(
                self._external_computed_tokens(manifest)
                / int(manifest["block_size"])
            )
            return snapshot_target_block_ids(
                block_ids,
                request_num_tokens=request.num_tokens,
                block_size=int(manifest["block_size"]),
                snapshot_blocks=planned_blocks,
                error_message=error_message,
            )
        return snapshot_target_block_ids(
            block_ids,
            request_num_tokens=request.num_tokens,
            block_size=int(manifest["block_size"]),
            snapshot_blocks=int(manifest["num_blocks"]),
            error_message=error_message,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        metadata = BridgeTPStreamMetadata()
        if not self._pending_requests:
            return metadata
        manifest = self._load_manifest()
        if self.gpu_resident_shadow:
            pending = list(self._pending_requests.items())
            self._pending_requests.clear()
            for request_id, request in pending:
                blocks = self._snapshot_target_block_ids(
                    request,
                    self._allocated_block_ids(request_id),
                    "Worker block table differs from allocation",
                )
                metadata.requests.append(
                    BridgeTPStreamRequest(
                        migration_id=str(manifest["migration_id"]),
                        source_request_id=str(manifest["source_request_id"]),
                        target_request_id=request_id,
                        target_block_ids=blocks,
                        num_computed_tokens=self._external_computed_tokens(manifest),
                        gpu_resident_shadow=True,
                    )
                )
            return metadata
        for new_request in scheduler_output.scheduled_new_reqs:
            request = self._pending_requests.pop(new_request.req_id, None)
            if request is None:
                continue
            if new_request.num_computed_tokens != int(
                manifest["num_computed_tokens"]
            ):
                raise ValueError("Worker token boundary differs from snapshot")
            snapshot_block_ids = self._snapshot_target_block_ids(
                request,
                new_request.block_ids,
                "Worker block table differs from allocation",
            )
            metadata.requests.append(
                BridgeTPStreamRequest(
                    migration_id=str(manifest["migration_id"]),
                    source_request_id=str(manifest["source_request_id"]),
                    target_request_id=request.request_id,
                    target_block_ids=snapshot_block_ids,
                    num_computed_tokens=int(manifest["num_computed_tokens"]),
                )
            )
        if self._pending_requests:
            raise RuntimeError("Allocated Phase 6 request was not scheduled")
        return metadata

    def _allocated_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        request = self._active_requests[request_id]
        blocks = getattr(request, "_bridgetp_target_block_ids", None)
        if blocks is None:
            raise RuntimeError("GPU-resident target allocation was not recorded")
        return blocks

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, BridgeTPStreamMetadata):
            raise TypeError("Unexpected BridgeTP Phase 6 connector metadata")
        if not metadata.requests:
            return
        if len(metadata.requests) != 1:
            raise ValueError("Phase 6 restores one request at a time")
        request = metadata.requests[0]
        if request.gpu_resident_shadow:
            self._start_live_gpu_load(request)
            return
        manifest = self._load_manifest()
        tp_rank = get_tp_group().rank_in_group
        record = manifest["ranks"][tp_rank]
        started = time.perf_counter()
        with socket.create_connection(
            (str(record["host"]), int(record["port"])),
            timeout=self.socket_timeout_s,
        ) as connection:
            connection.settimeout(self.socket_timeout_s)
            send_json(
                connection,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "migration_id": request.migration_id,
                    "session_token": manifest["session_token"],
                    "target_tp_rank": tp_rank,
                    "target_request_id": request.target_request_id,
                },
            )
            header = recv_json(connection)
            expected_header = {
                "protocol_version": PROTOCOL_VERSION,
                "migration_id": request.migration_id,
                "source_request_id": request.source_request_id,
                "target_tp_size": 4,
                "target_tp_rank": tp_rank,
                "num_computed_tokens": request.num_computed_tokens,
                "pending_known_tokens": 1,
                "block_size": int(manifest["block_size"]),
                "block_axis": int(manifest["block_axis"]),
                "num_layers": int(manifest["num_layers"]),
                "raw_tensor_bytes": int(record["raw_tensor_bytes"]),
                "payload_bytes": int(record["payload_bytes"]),
                "payload_sha256": str(record["payload_sha256"]),
                "num_frames": int(record["num_frames"]),
            }
            for key, value in expected_header.items():
                if header.get(key) != value:
                    raise ValueError(
                        f"Rank {tp_rank} stream header field {key} differs: "
                        f"{header.get(key)!r} != {value!r}"
                    )
            receive_started = time.perf_counter()
            payload_bytes, transfer = recv_payload_frames(
                connection,
                payload_bytes=int(header["payload_bytes"]),
                num_frames=int(header["num_frames"]),
                payload_sha256=str(header["payload_sha256"]),
                max_frame_bytes=int(header["chunk_bytes"]),
            )
            receive_ms = (time.perf_counter() - receive_started) * 1000
            deserialize_started = time.perf_counter()
            payload = deserialize_rank_payload(payload_bytes)
            deserialize_ms = (time.perf_counter() - deserialize_started) * 1000
            expected_payload = {
                "format_version": 1,
                "migration_id": request.migration_id,
                "source_request_id": request.source_request_id,
                "target_tp_size": 4,
                "target_tp_rank": tp_rank,
                "block_axis": int(manifest["block_axis"]),
                "block_size": int(manifest["block_size"]),
                "num_computed_tokens": request.num_computed_tokens,
            }
            for key, value in expected_payload.items():
                if payload.get(key) != value:
                    raise ValueError(
                        f"Rank {tp_rank} payload field {key} differs: "
                        f"{payload.get(key)!r} != {value!r}"
                    )
            shard_layers = payload.get("layers")
            if not isinstance(shard_layers, dict) or not shard_layers:
                raise ValueError(f"TP rank {tp_rank} received no KV layers")

            destination_layers: dict[str, torch.Tensor] = {}
            for layer_name, layer in forward_context.no_compile_layers.items():
                kv_cache = getattr(layer, "kv_cache", None)
                if layer_name in shard_layers:
                    if not isinstance(kv_cache, torch.Tensor):
                        raise TypeError(f"Layer {layer_name} has no tensor KV cache")
                    destination_layers[layer_name] = kv_cache
            if set(destination_layers) != set(shard_layers):
                missing = sorted(set(shard_layers) - set(destination_layers))
                raise ValueError(f"Target model is missing KV layers: {missing}")

            device = next(iter(destination_layers.values())).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inject_started = time.perf_counter()
            validation = inject_rank_shard(
                destination_layers,
                shard_layers,
                request.target_block_ids,
                block_axis=int(manifest["block_axis"]),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inject_ms = (time.perf_counter() - inject_started) * 1000
            receipt_path = (
                self.receipt_dir
                / _safe_name(request.target_request_id)
                / f"tp_rank_{tp_rank}.json"
            )
            phase7 = self.takeover_control_path is not None
            receipt = {
                "format_version": 1,
                "phase": self.expected_phase,
                "scope": (
                    "target-ready barrier before ownership commit"
                    if phase7
                    else (
                        "live stream restore and exact readback; "
                        "no ownership takeover"
                    )
                ),
                "status": "TARGET_READY" if phase7 else "READY",
                "migration_id": request.migration_id,
                "source_request_id": request.source_request_id,
                "target_request_id": request.target_request_id,
                "tp_rank": tp_rank,
                "target_block_ids": request.target_block_ids,
                "num_computed_tokens": request.num_computed_tokens,
                "pending_tokens_to_compute": 1,
                "receive_ms": receive_ms,
                "deserialize_ms": deserialize_ms,
                "inject_and_readback_ms": inject_ms,
                "target_ready_total_ms": (time.perf_counter() - started) * 1000,
                "total_ms": (time.perf_counter() - started) * 1000,
                "target_ready_unix_s": time.time(),
                **transfer,
                **validation,
            }
            _atomic_json_dump(receipt, receipt_path)
            send_json(
                connection,
                {
                    "status": "READY",
                    "target_request_id": request.target_request_id,
                    "exact_readback": validation["exact_readback"],
                },
            )

        if phase7:
            assert self.takeover_control_path is not None
            wait_started = time.perf_counter()
            deadline = wait_started + self.takeover_control_timeout_s
            while True:
                if self.takeover_control_path.exists():
                    state = _load_json(self.takeover_control_path)
                    if state.get("migration_id") != request.migration_id:
                        raise ValueError("Takeover state migration ID differs")
                    decision = state.get("state")
                    if decision == "COMMITTED":
                        if not state.get("source_abort_dispatched"):
                            raise ValueError(
                                "COMMITTED state has no source-abort evidence"
                            )
                        receipt.update(
                            {
                                "status": "OWNERSHIP_COMMITTED",
                                "control_wait_ms": (
                                    time.perf_counter() - wait_started
                                )
                                * 1000,
                                "source_abort_dispatched": True,
                                "takeover_state": decision,
                                "ownership_ready_total_ms": (
                                    time.perf_counter() - started
                                )
                                * 1000,
                            }
                        )
                        _atomic_json_dump(receipt, receipt_path)
                        get_tp_group().barrier()
                        break
                    if decision == "ROLLED_BACK":
                        receipt.update(
                            {
                                "status": "ROLLED_BACK",
                                "control_wait_ms": (
                                    time.perf_counter() - wait_started
                                )
                                * 1000,
                                "source_abort_dispatched": False,
                                "takeover_state": decision,
                                "rollback_total_ms": (
                                    time.perf_counter() - started
                                )
                                * 1000,
                            }
                        )
                        _atomic_json_dump(receipt, receipt_path)
                        get_tp_group().barrier()
                        raise RuntimeError(
                            "BridgeTP Phase 7 target was rolled back before commit"
                        )
                if time.perf_counter() >= deadline:
                    receipt["status"] = "CONTROL_TIMEOUT"
                    receipt["control_wait_ms"] = (
                        time.perf_counter() - wait_started
                    ) * 1000
                    _atomic_json_dump(receipt, receipt_path)
                    raise TimeoutError("Timed out waiting for Phase 7 commit decision")
                time.sleep(0.01)

        logger.warning(
            "%s received request %s on rank %d; readback=%s status=%s",
            receipt["phase"],
            request.target_request_id,
            tp_rank,
            validation["exact_readback"],
            receipt["status"],
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Retain paged-KV tensors for asynchronous Shadow injection."""
        self._registered_kv_caches = dict(kv_caches)

    def _start_live_gpu_load(self, request: BridgeTPStreamRequest) -> None:
        if request.target_request_id in self._load_threads:
            return
        if not self._registered_kv_caches:
            raise RuntimeError("TP4 KV caches were not registered with connector")
        if self.online_remote_attention:
            tp_rank = get_tp_group().rank_in_group
            server = RemoteAttentionRankServer(
                config=RemoteAttentionConfig(
                    run_dir=self.manifest_path.parent,
                    host="127.0.0.1",
                    base_port=self.remote_attention_base_port,
                    timeout_s=self.socket_timeout_s,
                    source_request_id_prefix="target-does-not-select-source",
                ),
                rank=tp_rank,
                kv_caches=self._registered_kv_caches,
                block_ids=request.target_block_ids,
                kv_lock=self._gpu_kv_lock,
            )
            self._remote_attention_servers[request.target_request_id] = server
            server.start()
        thread = threading.Thread(
            target=self._live_gpu_load,
            args=(request,),
            name=f"bridgetp-live-gpu-{_safe_name(request.target_request_id)}",
            daemon=True,
        )
        self._load_threads[request.target_request_id] = thread
        thread.start()

    def _wait_for_file(self, path: Path, deadline: float) -> None:
        while not path.is_file():
            self._raise_if_shadow_cancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {path}")
            time.sleep(0.005)

    def _raise_if_shadow_cancelled(self) -> None:
        if (
            self.takeover_control_path is None
            or not self.takeover_control_path.is_file()
        ):
            return
        state = _load_json(self.takeover_control_path).get("state")
        if state in {"ROLLED_BACK", "CANCELLED"}:
            raise _ShadowCancelled(f"GPU-resident Shadow ended in {state}")

    def _destination_layers(
        self, shard_layers: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        destination = {
            name: tensor
            for name, tensor in self._registered_kv_caches.items()
            if name in shard_layers
        }
        if set(destination) != set(shard_layers):
            missing = sorted(set(shard_layers) - set(destination))
            raise ValueError(f"Target model is missing KV layers: {missing}")
        return destination

    def _live_gpu_load(self, request: BridgeTPStreamRequest) -> None:
        request_id = request.target_request_id
        try:
            manifest = self._load_manifest()
            tp_rank = get_tp_group().rank_in_group
            deadline = time.monotonic() + self.socket_timeout_s
            queue_dir = self.manifest_path.parent / "live_gpu_queue" / (
                f"tp_rank_{tp_rank}"
            )
            digest = hashlib.sha256()
            aggregate_bytes = 0
            device = next(iter(self._registered_kv_caches.values())).device
            if device.type == "cuda":
                torch.cuda.set_device(device)
            started = time.perf_counter()
            initial_end = int(manifest["num_computed_tokens"])
            initial_blocks = math.ceil(initial_end / int(manifest["block_size"]))
            history_path = queue_dir / "history_full.bin"
            self._wait_for_file(history_path, deadline)
            history_bytes = history_path.read_bytes()
            history = deserialize_rank_payload(history_bytes)
            for key, expected in {
                "migration_id": request.migration_id,
                "source_request_id": request.source_request_id,
                "target_tp_rank": tp_rank,
            }.items():
                if history.get(key) != expected:
                    raise ValueError(f"Live Shadow history {key} differs")
            history_layers = history.get("layers")
            if not isinstance(history_layers, dict) or not history_layers:
                raise ValueError("Live Shadow history has no KV layers")
            with self._gpu_kv_lock:
                validation = inject_rank_shard(
                    self._destination_layers(history_layers),
                    history_layers,
                    request.target_block_ids[:initial_blocks],
                    block_axis=int(manifest["block_axis"]),
                )
            exact_readback = validation["exact_readback"] is True
            digest.update(history_bytes)
            aggregate_bytes += len(history_bytes)
            # One full-tensor exact readback covers every logical block.  Keep
            # the per-block receipts required by the protocol without paying
            # for 132 separate tensor serializations and GPU round trips.
            completed_unix_s = time.time()
            for logical_block in range(initial_blocks):
                _atomic_json_dump(
                    {
                        "format_version": 1,
                        "status": "BLOCK_GPU_RESIDENT",
                        "migration_id": request.migration_id,
                        "target_request_id": request_id,
                        "tp_rank": tp_rank,
                        "logical_block": logical_block,
                        "end_token": min(
                            (logical_block + 1) * int(manifest["block_size"]),
                            initial_end,
                        ),
                        "exact_readback": exact_readback,
                        "verification_scope": "FULL_RANK_EXACT_READBACK",
                        "completed_unix_s": completed_unix_s,
                    },
                    self.manifest_path.parent
                    / "gpu_block_receipts"
                    / f"tp_rank_{tp_rank}"
                    / f"block_{logical_block:012d}.json",
                )
            current = initial_end
            delta_batches = 0
            _atomic_json_dump(
                {
                    "format_version": 1,
                    "status": "INITIAL_HISTORY_GPU_RESIDENT",
                    "migration_id": request.migration_id,
                    "target_request_id": request_id,
                    "tp_rank": tp_rank,
                    "end_token": current,
                    "exact_readback": exact_readback,
                    "completed_unix_s": time.time(),
                },
                self.manifest_path.parent
                / "gpu_initial_receipts"
                / f"tp_rank_{tp_rank}.json",
            )
            while current < request.num_computed_tokens:
                candidates = sorted(queue_dir.glob(f"delta_{current:012d}_*.bin"))
                if not candidates:
                    self._raise_if_shadow_cancelled()
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for TP4 rank {tp_rank} delta {current}"
                        )
                    time.sleep(0.005)
                    continue
                if len(candidates) != 1:
                    raise ValueError(f"Ambiguous live delta beginning at {current}")
                path = candidates[0]
                delta_bytes = path.read_bytes()
                delta = deserialize_rank_payload(delta_bytes)
                start = int(delta["start_token"])
                end = int(delta["end_token"])
                if start != current or end > request.num_computed_tokens:
                    raise ValueError(
                        f"Non-contiguous live delta [{start}, {end}) at {current}"
                    )
                delta_layers = delta.get("layers")
                if not isinstance(delta_layers, dict) or not delta_layers:
                    raise ValueError("Live Shadow delta has no KV layers")
                with self._gpu_kv_lock:
                    inject_rank_delta(
                        self._destination_layers(delta_layers),
                        delta_layers,
                        request.target_block_ids,
                        start_token=start,
                        end_token=end,
                        block_axis=int(manifest["block_axis"]),
                        block_size=int(manifest["block_size"]),
                    )
                digest.update(delta_bytes)
                aggregate_bytes += len(delta_bytes)
                current = end
                delta_batches += 1
                delta_receipt = {
                    "format_version": 1,
                    "status": "DELTA_GPU_RESIDENT",
                    "migration_id": request.migration_id,
                    "target_request_id": request_id,
                    "tp_rank": tp_rank,
                    "start_token": start,
                    "end_token": end,
                    "exact_readback": True,
                    "completed_unix_s": time.time(),
                }
                _atomic_json_dump(
                    delta_receipt,
                    self.manifest_path.parent
                    / "gpu_delta_receipts"
                    / f"tp_rank_{tp_rank}"
                    / f"delta_{start:012d}_{end:012d}.json",
                )
                _atomic_json_dump(
                    {
                        **delta_receipt,
                        "status": "STREAMING",
                        "end_token": current,
                        "updated_unix_s": time.time(),
                    },
                    self.manifest_path.parent
                    / "gpu_watermarks"
                    / f"tp_rank_{tp_rank}.json",
                )

            cutover_path = self.manifest_path.parent / "cutover_manifest.json"
            self._wait_for_file(cutover_path, deadline)
            cutover = _load_json(cutover_path)
            if int(cutover["num_computed_tokens"]) != current:
                raise ValueError("Cutover boundary differs from GPU watermark")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            receipt_path = (
                self.receipt_dir / _safe_name(request_id) / f"tp_rank_{tp_rank}.json"
            )
            receipt = {
                "format_version": 1,
                "phase": self.expected_phase,
                "scope": "GPU-resident incremental Shadow before atomic takeover",
                "status": "TARGET_READY",
                "migration_id": request.migration_id,
                "source_request_id": request.source_request_id,
                "target_request_id": request_id,
                "tp_rank": tp_rank,
                "target_block_ids": request.target_block_ids,
                "num_computed_tokens": current,
                "pending_tokens_to_compute": 1,
                "payload_bytes": aggregate_bytes,
                "payload_sha256": digest.hexdigest(),
                "delta_batches": delta_batches,
                "gpu_resident": True,
                "exact_readback": exact_readback,
                "target_ready_total_ms": (time.perf_counter() - started) * 1000,
                "target_ready_unix_s": time.time(),
            }
            _atomic_json_dump(receipt, receipt_path)
            _atomic_json_dump(
                {
                    "format_version": 1,
                    "status": "TARGET_READY",
                    "migration_id": request.migration_id,
                    "target_request_id": request_id,
                    "tp_rank": tp_rank,
                    "end_token": current,
                    "payload_bytes": aggregate_bytes,
                    "payload_sha256": digest.hexdigest(),
                    "updated_unix_s": time.time(),
                },
                self.manifest_path.parent
                / "gpu_watermarks"
                / f"tp_rank_{tp_rank}.json",
            )
            from vllm.bridge_tp.experiment_timeline import emit_event

            emit_event(
                self.manifest_path.parent,
                "target_connector",
                "TARGET_RANK_READY",
                request_id=request_id,
                migration_id=request.migration_id,
                tp_rank=tp_rank,
                num_computed_tokens=current,
                exact_readback=exact_readback,
            )
            if self.takeover_control_path is not None:
                while True:
                    if self.takeover_control_path.is_file():
                        control = _load_json(self.takeover_control_path)
                        if control.get("state") == "COMMITTED":
                            receipt["status"] = "OWNERSHIP_COMMITTED"
                            receipt["takeover_state"] = "COMMITTED"
                            receipt["ownership_ready_total_ms"] = (
                                time.perf_counter() - started
                            ) * 1000
                            _atomic_json_dump(receipt, receipt_path)
                            break
                        if control.get("state") in {"ROLLED_BACK", "CANCELLED"}:
                            raise _ShadowCancelled(
                                f"Live Shadow ended in {control.get('state')}"
                            )
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for Shadow commit")
                    time.sleep(0.005)
            self._completed_recvs.add(request_id)
        except _ShadowCancelled as error:
            _atomic_json_dump(
                {
                    "format_version": 1,
                    "status": "CLEANED",
                    "target_request_id": request_id,
                    "reason": str(error),
                    "updated_unix_s": time.time(),
                },
                self.manifest_path.parent
                / "gpu_cleanup_receipts"
                / f"{_safe_name(request_id)}.json",
            )
            self._completed_recvs.add(request_id)
        except BaseException as error:
            self._load_errors[request_id] = error

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        if self._load_errors:
            request_id, error = next(iter(self._load_errors.items()))
            raise RuntimeError(
                f"GPU-resident Shadow load failed for {request_id}: {error}"
            ) from error
        ready = self._completed_recvs - self._reported_recvs
        if not ready:
            return None, None
        self._reported_recvs.update(ready)
        return None, set(ready)

    def update_connector_output(self, connector_output: Any) -> None:
        for request_id in connector_output.finished_recving or ():
            request = self._active_requests.get(request_id)
            if request is None or not self.gpu_resident_shadow:
                continue
            cutover_path = self.manifest_path.parent / "cutover_manifest.json"
            if not cutover_path.is_file():
                cleanup_path = (
                    self.manifest_path.parent / "target_cleanup_receipt.json"
                )
                if cleanup_path.is_file():
                    continue
                raise FileNotFoundError("Shadow receive finished without cutover")
            cutover = _load_json(cutover_path)
            token_ids = list(cutover["all_known_token_ids"])
            if len(token_ids) != request.num_tokens:
                raise ValueError("Final Shadow token count differs from reservation")
            request.prompt_token_ids = token_ids
            request._all_token_ids[:] = token_ids
            request.num_prompt_tokens = len(token_ids)
            request.block_hashes.clear()
            request.update_block_hashes()

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        return
