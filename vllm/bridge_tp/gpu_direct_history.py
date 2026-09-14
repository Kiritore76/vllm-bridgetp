# SPDX-License-Identifier: Apache-2.0
"""Dependency-free NCCL point-to-point transport for Shadow history.

The source and target vLLM engines are independent distributed worlds. A
dedicated two-rank NCCL communicator is negotiated over a small TCP control
socket for each TP4 rank. Only JSON metadata and acknowledgements use the
socket; every KV tensor remains on CUDA.
"""

from __future__ import annotations

import base64
import socket
import time
from dataclasses import dataclass
from typing import Any

import torch

from vllm.bridge_tp.stream_protocol import recv_json, send_json
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclDataTypeEnum,
)

TRANSPORT = "NCCL_P2P_GPU_DIRECT"
INTEGRITY = "NCCL_COMPLETION_PLUS_GPU_EXACT_READBACK"
_TIMEOUT_S = 600.0


def tensor_id(migration_id: str, rank: int, layer_index: int) -> str:
    """Return the deterministic rendezvous key for one rank/layer tensor."""
    return f"{migration_id}#history#rank{rank}#layer{layer_index}"


def _cuda_device(device: torch.device) -> torch.device:
    if device.type != "cuda":
        raise ValueError("GPU-direct history requires a CUDA device")
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return torch.device("cuda", index)


def _connect(address: str) -> socket.socket:
    host, port_text = address.rsplit(":", 1)
    deadline = time.monotonic() + _TIMEOUT_S
    while True:
        try:
            connection = socket.create_connection(
                (host, int(port_text)), timeout=5.0
            )
            connection.settimeout(_TIMEOUT_S)
            return connection
        except OSError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out connecting to GPU-direct receiver {address}"
                ) from error
            time.sleep(0.02)


def _send_tensor(
    nccl: NCCLLibrary,
    comm: Any,
    tensor: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    with torch.cuda.stream(stream):
        nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            1,
            comm,
            cudaStream_t(stream.cuda_stream),
        )
    stream.synchronize()


def _recv_tensor(
    nccl: NCCLLibrary,
    comm: Any,
    tensor: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    with torch.cuda.stream(stream):
        nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            0,
            comm,
            cudaStream_t(stream.cuda_stream),
        )
    stream.synchronize()


@dataclass
class ReceiveResult:
    layers: dict[str, torch.Tensor]
    raw_tensor_bytes: int
    receive_ms: float


class GpuDirectHistoryReceiver:
    """Receive one TP4 rank's historical shard directly into CUDA tensors."""

    def __init__(self, *, device: torch.device, host: str, port: int) -> None:
        self.device = _cuda_device(device)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((host, port))
        self.listener.listen(1)
        self.listener.settimeout(_TIMEOUT_S)

    def receive(
        self,
        *,
        migration_id: str,
        rank: int,
        layer_names: list[str],
    ) -> ReceiveResult:
        started = time.perf_counter()
        nccl = NCCLLibrary()
        connection, _ = self.listener.accept()
        connection.settimeout(_TIMEOUT_S)
        comm = None
        try:
            hello = recv_json(connection)
            if hello.get("migration_id") != migration_id:
                raise ValueError("GPU-direct migration ID differs")
            if int(hello.get("target_tp_rank", -1)) != rank:
                raise ValueError("GPU-direct target rank differs")
            unique_bytes = base64.b64decode(str(hello["nccl_unique_id_b64"]))
            unique_id = nccl.unique_id_from_bytes(unique_bytes)
            with torch.cuda.device(self.device):
                comm = nccl.ncclCommInitRank(2, unique_id, 1)
                stream = torch.cuda.Stream(device=self.device)
            send_json(connection, {"status": "COMM_READY"})

            layers: dict[str, torch.Tensor] = {}
            raw_bytes = 0
            for layer_index, layer_name in enumerate(layer_names):
                header = recv_json(connection)
                expected_id = tensor_id(migration_id, rank, layer_index)
                if header.get("tensor_id") != expected_id:
                    raise ValueError("GPU-direct tensor order differs")
                if header.get("layer_name") != layer_name:
                    raise ValueError("GPU-direct layer name differs")
                dtype = getattr(torch, str(header["dtype"]))
                shape = [int(value) for value in header["shape"]]
                with torch.cuda.device(self.device):
                    value = torch.empty(shape, dtype=dtype, device=self.device)
                send_json(connection, {"status": "READY_TO_RECV"})
                _recv_tensor(nccl, comm, value, stream)
                send_json(connection, {"status": "RECEIVED"})
                layers[layer_name] = value
                raw_bytes += value.numel() * value.element_size()
            send_json(connection, {"status": "COMPLETE"})
            return ReceiveResult(
                layers=layers,
                raw_tensor_bytes=raw_bytes,
                receive_ms=(time.perf_counter() - started) * 1000,
            )
        finally:
            if comm is not None:
                nccl.ncclCommDestroy(comm)
            connection.close()
            self.listener.close()


class GpuDirectHistorySender:
    """Reshard TP1 KV on GPU and send each shard to its TP4 rank."""

    def __init__(self, *, device: torch.device, host: str, port: int) -> None:
        del host, port
        self.device = _cuda_device(device)

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

        nccl = NCCLLibrary()
        connections: list[socket.socket] = []
        comms: list[Any] = []
        try:
            for rank, address in enumerate(target_addresses):
                connection = _connect(address)
                unique_id = nccl.ncclGetUniqueId()
                send_json(
                    connection,
                    {
                        "migration_id": migration_id,
                        "target_tp_rank": rank,
                        "nccl_unique_id_b64": base64.b64encode(
                            bytes(unique_id.internal)
                        ).decode("ascii"),
                    },
                )
                with torch.cuda.device(self.device):
                    comm = nccl.ncclCommInitRank(2, unique_id, 0)
                ready = recv_json(connection)
                if ready.get("status") != "COMM_READY":
                    raise RuntimeError(
                        f"GPU-direct rank {rank} did not become ready"
                    )
                connections.append(connection)
                comms.append(comm)

            started = time.perf_counter()
            bytes_by_rank = [0] * target_tp_size
            block_index = torch.tensor(
                block_ids, dtype=torch.long, device=self.device
            )
            with torch.cuda.device(self.device):
                producer_stream = torch.cuda.Stream(device=self.device)
                send_streams = [
                    torch.cuda.Stream(device=self.device)
                    for _ in range(target_tp_size)
                ]
            with torch.cuda.stream(producer_stream):
                for layer_index, (layer_name, cache) in enumerate(
                    zip(layer_names, kv_caches)
                ):
                    if _cuda_device(cache.device) != self.device:
                        raise ValueError("all source KV caches must share one GPU")
                    selected = cache.index_select(block_axis, block_index)
                    if int(selected.shape[head_axis]) % target_tp_size:
                        raise ValueError(
                            "source KV heads are not divisible by target TP"
                        )
                    shards = [
                        shard.contiguous()
                        for shard in selected.chunk(
                            target_tp_size, dim=head_axis
                        )
                    ]
                    producer_stream.synchronize()
                    for rank, value in enumerate(shards):
                        send_json(
                            connections[rank],
                            {
                                "tensor_id": tensor_id(
                                    migration_id, rank, layer_index
                                ),
                                "layer_name": layer_name,
                                "shape": list(value.shape),
                                "dtype": str(value.dtype).removeprefix("torch."),
                            },
                        )
                        ready = recv_json(connections[rank])
                        if ready.get("status") != "READY_TO_RECV":
                            raise RuntimeError(
                                f"rank {rank} rejected layer {layer_index}"
                            )
                        _send_tensor(
                            nccl, comms[rank], value, send_streams[rank]
                        )
                        received = recv_json(connections[rank])
                        if received.get("status") != "RECEIVED":
                            raise RuntimeError(
                                f"rank {rank} missed layer {layer_index}"
                            )
                        bytes_by_rank[rank] += (
                            value.numel() * value.element_size()
                        )
            for rank, connection in enumerate(connections):
                final = recv_json(connection)
                if final.get("status") != "COMPLETE":
                    raise RuntimeError(f"GPU-direct rank {rank} is incomplete")
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
        finally:
            for comm in comms:
                nccl.ncclCommDestroy(comm)
            for connection in connections:
                connection.close()
