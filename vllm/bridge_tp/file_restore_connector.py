# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Synchronous file-backed TP4 restore connector for BridgeTP Phase 5."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.bridge_tp.block_layout import snapshot_target_block_ids
from vllm.bridge_tp.kv_restore import (
    RESTORE_PARAM,
    RestoreArtifact,
    inject_rank_shard,
    load_rank_shard,
    load_restore_artifact,
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
class BridgeTPRestoreRequest:
    """Worker metadata for one target request and its allocated TP4 blocks."""

    target_request_id: str
    source_request_id: str
    target_block_ids: list[int]
    num_computed_tokens: int


@dataclass
class BridgeTPRestoreMetadata(KVConnectorMetadata):
    """Serializable scheduler-to-worker restore metadata."""

    requests: list[BridgeTPRestoreRequest] = field(default_factory=list)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe[:160] or "request"


def _atomic_json_dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_path, path)


class BridgeTPFileRestoreConnector(KVConnectorBase_V1):
    """Load one authenticated Phase 4 shard into scheduler-owned TP4 blocks.

    This connector is deliberately synchronous and single-request. It proves
    destination allocation, per-rank KV injection, and exact device readback.
    It is not the final online transfer or ownership-takeover mechanism.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        artifact_dir = self._kv_transfer_config.get_from_extra_config(
            "bridgetp_reshard_dir", None
        )
        if not artifact_dir:
            raise ValueError(
                "BridgeTPFileRestoreConnector requires "
                "kv_connector_extra_config.bridgetp_reshard_dir"
            )
        self.artifact: RestoreArtifact = load_restore_artifact(Path(artifact_dir))
        receipt_dir = self._kv_transfer_config.get_from_extra_config(
            "bridgetp_restore_receipt_dir",
            str(self.artifact.root / "restore_receipts"),
        )
        self.receipt_dir = Path(receipt_dir).resolve()

        parallel_config = vllm_config.parallel_config
        if parallel_config.tensor_parallel_size != 4:
            raise ValueError("BridgeTP Phase 5 requires tensor_parallel_size=4")
        if parallel_config.pipeline_parallel_size != 1:
            raise ValueError(
                "BridgeTP Phase 5 MVP does not support pipeline parallelism"
            )
        target_model = str(vllm_config.model_config.model)
        source_model = str(self.artifact.manifest["source_dump"]["model"])
        if target_model != source_model:
            raise ValueError(
                f"Target model differs from Phase 4 source: "
                f"{target_model!r} != {source_model!r}"
            )
        if vllm_config.cache_config.block_size != self.artifact.block_size:
            raise ValueError(
                "Target vLLM block size differs from Phase 4 artifact: "
                f"{vllm_config.cache_config.block_size} != {self.artifact.block_size}"
            )
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("BridgeTP Phase 5 requires one KV-cache group")

        self._pending_requests: dict[str, Request] = {}
        self._claimed_target_request_id: str | None = None
        logger.warning(
            "BridgeTP Phase 5 file restore enabled for source request %s; "
            "this is a synchronous validation connector, not online takeover",
            self.artifact.source_request_id,
        )

    def _request_matches(self, request: Request) -> bool:
        params = request.kv_transfer_params or {}
        if params.get(RESTORE_PARAM) != self.artifact.source_request_id:
            return False
        prompt = request.prompt_token_ids
        if prompt is None or list(prompt) != self.artifact.all_known_token_ids:
            raise ValueError(
                "BridgeTP restore prompt must exactly equal computed plus pending "
                "token history from generated_tokens.json"
            )
        if request.num_tokens != self.artifact.num_computed_tokens + 1:
            raise ValueError("BridgeTP Phase 5 requires exactly one pending token")
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
                "BridgeTP Phase 5 cannot mix local prefix-cache hits with the "
                "restored artifact; start with --no-enable-prefix-caching"
            )
        if (
            self._claimed_target_request_id is not None
            and self._claimed_target_request_id != request.request_id
        ):
            raise RuntimeError("BridgeTP Phase 5 artifact has already been claimed")
        return self.artifact.num_computed_tokens, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens == 0:
            return
        if not self._request_matches(request):
            raise ValueError("Allocated request does not match BridgeTP artifact")
        if num_external_tokens != self.artifact.num_computed_tokens:
            raise ValueError("Scheduler external-token count differs from artifact")
        block_ids = blocks.get_block_ids()
        snapshot_target_block_ids(
            block_ids,
            request_num_tokens=request.num_tokens,
            block_size=self.artifact.block_size,
            snapshot_blocks=self.artifact.num_blocks,
            error_message=(
                "Allocated target block count differs from source request blocks"
            ),
        )
        self._claimed_target_request_id = request.request_id
        self._pending_requests[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        metadata = BridgeTPRestoreMetadata()
        for new_request in scheduler_output.scheduled_new_reqs:
            request = self._pending_requests.pop(new_request.req_id, None)
            if request is None:
                continue
            if new_request.num_computed_tokens != self.artifact.num_computed_tokens:
                raise ValueError("Worker request boundary differs from artifact")
            snapshot_block_ids = snapshot_target_block_ids(
                new_request.block_ids,
                request_num_tokens=request.num_tokens,
                block_size=self.artifact.block_size,
                snapshot_blocks=self.artifact.num_blocks,
                error_message="Worker target block table differs from allocation",
            )
            metadata.requests.append(
                BridgeTPRestoreRequest(
                    target_request_id=request.request_id,
                    source_request_id=self.artifact.source_request_id,
                    target_block_ids=snapshot_block_ids,
                    num_computed_tokens=self.artifact.num_computed_tokens,
                )
            )
        if self._pending_requests:
            raise RuntimeError(
                "Allocated BridgeTP request was not present in scheduled_new_reqs"
            )
        return metadata

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, BridgeTPRestoreMetadata):
            raise TypeError("Unexpected BridgeTP connector metadata type")
        if not metadata.requests:
            return
        if len(metadata.requests) != 1:
            raise ValueError("BridgeTP Phase 5 restores one request at a time")

        request = metadata.requests[0]
        if request.source_request_id != self.artifact.source_request_id:
            raise ValueError("Worker restore source request ID differs")
        if request.num_computed_tokens != self.artifact.num_computed_tokens:
            raise ValueError("Worker restore token boundary differs")

        tp_rank = get_tp_group().rank_in_group
        shard_layers = load_rank_shard(self.artifact, tp_rank)
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

        if destination_layers:
            device = next(iter(destination_layers.values())).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        started = time.perf_counter()
        validation = inject_rank_shard(
            destination_layers,
            shard_layers,
            request.target_block_ids,
            block_axis=self.artifact.block_axis,
        )
        if destination_layers and device.type == "cuda":
            torch.cuda.synchronize(device)
        restore_ms = (time.perf_counter() - started) * 1000

        receipt = {
            "format_version": 1,
            "phase": "BridgeTP D3 Phase 5",
            "scope": "TP4 file restore and exact readback; no online takeover",
            "source_request_id": request.source_request_id,
            "target_request_id": request.target_request_id,
            "tp_rank": tp_rank,
            "target_block_ids": request.target_block_ids,
            "num_computed_tokens": request.num_computed_tokens,
            "pending_tokens_to_compute": 1,
            "restore_ms": restore_ms,
            **validation,
        }
        receipt_path = (
            self.receipt_dir
            / _safe_name(request.target_request_id)
            / f"tp_rank_{tp_rank}.json"
        )
        _atomic_json_dump(receipt, receipt_path)
        logger.warning(
            "BridgeTP restored request %s on TP rank %d into blocks %s; "
            "readback=%s restore_ms=%.3f",
            request.target_request_id,
            tp_rank,
            request.target_block_ids,
            validation["exact_readback"],
            restore_ms,
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
