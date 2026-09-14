# SPDX-License-Identifier: Apache-2.0
"""NCCL point-to-point transport for BridgeTP Shadow history.

The source and target vLLM engines are independent distributed worlds.  This
module therefore creates a dedicated two-rank NCCL communicator for each TP4
rank instead of borrowing either engine's tensor-parallel process group.
Only small connection metadata travels through ZMQ; KV tensors remain on CUDA.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine,
)

TRANSPORT = "NCCL_P2P_GPU_DIRECT"
INTEGRITY = "NCCL_COMPLETION_PLUS_GPU_EXACT_READBACK"


def _engine(*, device_index: int, host: str, port: int) -> P2pNcclEngine:
    config = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_both",
        kv_ip=host,
        kv_port=port,
        # History is consumed one layer at a time, so it never needs a large
        # host-side or device-side retention pool.
        kv_buffer_size=float(4 * 1024**3),
        kv_connector_extra_config={
            "send_type": "PUT",
            # P2pNcclEngine always constructs its fallback pinned-memory pool.
            # Keep that unused pool tiny: this path consumes each CUDA tensor
            # immediately and the 4 GiB buffer threshold prevents spilling.
            "mem_pool_size_gb": 0.001,
            "nccl_num_channels": "8",
        },
    )
    return P2pNcclEngine(
        local_rank=device_index,
        config=config,
        hostname=host,
    )


def tensor_id(migration_id: str, rank: int, layer_index: int) -> str:
    """Return the deterministic rendezvous key for one rank/layer tensor."""
    return f"{migration_id}#history#rank{rank}#layer{layer_index}"


@dataclass
class ReceiveResult:
    layers: dict[str, torch.Tensor]
    raw_tensor_bytes: int
    receive_ms: float


class GpuDirectHistoryReceiver:
    """Receive one TP4 rank's historical shard directly into CUDA tensors."""

    def __init__(self, *, device: torch.device, host: str, port: int) -> None:
        if device.type != "cuda":
            raise ValueError("GPU-direct history requires a CUDA device")
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        self.device = torch.device("cuda", device_index)
        self.engine = _engine(device_index=device_index, host=host, port=port)

    def receive(
        self,
        *,
        migration_id: str,
        rank: int,
        layer_names: list[str],
    ) -> ReceiveResult:
        started = time.perf_counter()
        layers: dict[str, torch.Tensor] = {}
        raw_bytes = 0
        for layer_index, layer_name in enumerate(layer_names):
            value = self.engine.recv_tensor(
                tensor_id(migration_id, rank, layer_index)
            )
            if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
                raise RuntimeError(
                    f"rank {rank} layer {layer_name} was not received on CUDA"
                )
            layers[layer_name] = value
            raw_bytes += value.numel() * value.element_size()
        return ReceiveResult(
            layers=layers,
            raw_tensor_bytes=raw_bytes,
            receive_ms=(time.perf_counter() - started) * 1000,
        )


class GpuDirectHistorySender:
    """Reshard TP1 KV on GPU and send each shard to its TP4 rank."""

    def __init__(self, *, device: torch.device, host: str, port: int) -> None:
        if device.type != "cuda":
            raise ValueError("GPU-direct history requires a CUDA device")
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        self.device = torch.device("cuda", device_index)
        self.engine = _engine(device_index=device_index, host=host, port=port)

    def send(
        self,
        *,
        migration_id: str,
        kv_caches: list[torch.Tensor],
        layer_names: list[str],
        block_ids: list[int],
        block_axis: int,
        head_axis: int,
        target_tp_size: int,
        target_addresses: list[str],
    ) -> list[dict[str, Any]]:
        if len(kv_caches) != len(layer_names):
            raise ValueError("KV cache and layer-name counts differ")
        if len(target_addresses) != target_tp_size:
            raise ValueError("target address count differs from TP size")
        started = time.perf_counter()
        bytes_by_rank = [0] * target_tp_size
        block_index = torch.tensor(
            block_ids, dtype=torch.long, device=self.device
        )
        # A dedicated stream lets the Python model thread return immediately.
        # The first operation is naturally ordered after the trigger iteration
        # because PyTorch's allocator records cross-stream tensor use.
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            for layer_index, cache in enumerate(kv_caches):
                if cache.device != self.device:
                    raise ValueError("all source KV caches must share one CUDA device")
                selected = cache.index_select(block_axis, block_index)
                if int(selected.shape[head_axis]) % target_tp_size:
                    raise ValueError("source KV heads are not divisible by target TP")
                shards = [
                    shard.contiguous()
                    for shard in selected.chunk(target_tp_size, dim=head_axis)
                ]
                # P2pNcclEngine owns a separate send stream.  Complete this
                # producer stream before handing over pointers to avoid a
                # cross-stream read-before-write race.
                stream.synchronize()
                for rank, contiguous in enumerate(shards):
                    ok = self.engine.send_tensor(
                        tensor_id(migration_id, rank, layer_index),
                        contiguous,
                        target_addresses[rank],
                    )
                    if not ok:
                        raise RuntimeError(
                            f"NCCL sender rejected rank {rank} layer {layer_index}"
                        )
                    bytes_by_rank[rank] += (
                        contiguous.numel() * contiguous.element_size()
                    )
        stream.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return [
            {
                "target_tp_rank": rank,
                "raw_tensor_bytes": count,
                "transfer_ms": elapsed_ms,
                "observed_gib_s": (
                    count / 1024**3 / (elapsed_ms / 1000)
                    if elapsed_ms > 0
                    else None
                ),
            }
            for rank, count in enumerate(bytes_by_rank)
        ]
