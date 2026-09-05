# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Live TCP-backed TP4 KV connector for BridgeTP Phase 6."""

from __future__ import annotations

import json
import os
import re
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.bridge_tp.block_layout import snapshot_target_block_ids
from vllm.bridge_tp.kv_restore import inject_rank_shard
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


@dataclass
class BridgeTPStreamRequest:
    migration_id: str
    source_request_id: str
    target_request_id: str
    target_block_ids: list[int]
    num_computed_tokens: int


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
        self._manifest: dict[str, Any] | None = None
        self._pending_requests: dict[str, Request] = {}
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
        if request.num_tokens != int(manifest["num_computed_tokens"]) + 1:
            raise ValueError("Phase 6 requires exactly one pending token")
        return True

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
        return int(self._load_manifest()["num_computed_tokens"]), False

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
        if num_external_tokens != int(manifest["num_computed_tokens"]):
            raise ValueError("Target external-token count differs from snapshot")
        block_ids = blocks.get_block_ids()
        self._snapshot_target_block_ids(
            request,
            block_ids,
            "Target block allocation differs from live snapshot",
        )
        self._claimed_target_request_id = request.request_id
        self._pending_requests[request.request_id] = request

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
