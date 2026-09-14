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


def _send_group(
    nccl: NCCLLibrary,
    comms: list[Any],
    tensors: list[torch.Tensor],
    streams: list[torch.cuda.Stream],
) -> None:
    nccl.ncclGroupStart()
    try:
        for comm, tensor, stream in zip(comms, tensors, streams):
            with torch.cuda.stream(stream):
                nccl.ncclSend(
                    buffer_type(tensor.data_ptr()),
                    tensor.numel(),
                    ncclDataTypeEnum.from_torch(tensor.dtype),
                    1,
                    comm,
                    cudaStream_t(stream.cuda_stream),
                )
    finally:
        nccl.ncclGroupEnd()
    for stream in streams:
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
        layer_records: list[dict[str, Any]],
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

            if not layer_records:
                raise ValueError("GPU-direct history has no layer records")
            dtype_names = {
                str(row["dtype"]).removeprefix("torch.")
                for row in layer_records
            }
            if len(dtype_names) != 1:
                raise ValueError("GPU-direct packed history requires one dtype")
            dtype = getattr(torch, dtype_names.pop())
            shapes = [
                [int(value) for value in row["rank_shape"]]
                for row in layer_records
            ]
            counts = [
                int(torch.tensor(shape).prod().item()) for shape in shapes
            ]
            with torch.cuda.device(self.device):
                packed = torch.empty(sum(counts), dtype=dtype, device=self.device)
            send_json(
                connection,
                {
                    "status": "READY_TO_RECV",
                    "tensor_id": tensor_id(migration_id, rank, 0),
                    "numel": packed.numel(),
                },
            )
            _recv_tensor(nccl, comm, packed, stream)
            send_json(connection, {"status": "RECEIVED"})
            layers: dict[str, torch.Tensor] = {}
            offset = 0
            for record, shape, count in zip(layer_records, shapes, counts):
                layers[str(record["layer_name"])] = packed.narrow(
                    0, offset, count
                ).view(shape)
                offset += count
            raw_bytes = packed.numel() * packed.element_size()
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
            block_index = torch.tensor(
                block_ids, dtype=torch.long, device=self.device
            )
            dtype = kv_caches[0].dtype
            if any(cache.dtype != dtype for cache in kv_caches):
                raise ValueError("GPU-direct packed history requires one dtype")
            rank_elements = sum(
                cache.numel()
                // int(cache.shape[block_axis])
                * len(block_ids)
                // target_tp_size
                for cache in kv_caches
            )
            with torch.cuda.device(self.device):
                producer_stream = torch.cuda.Stream(device=self.device)
                send_streams = [
                    torch.cuda.Stream(device=self.device)
                    for _ in range(target_tp_size)
                ]
                packed_by_rank = [
                    torch.empty(rank_elements, dtype=dtype, device=self.device)
                    for _ in range(target_tp_size)
                ]
            offsets = [0] * target_tp_size
            with torch.cuda.stream(producer_stream):
                for cache in kv_caches:
                    if _cuda_device(cache.device) != self.device:
                        raise ValueError("all source KV caches must share one GPU")
                    selected = cache.index_select(block_axis, block_index)
                    if int(selected.shape[head_axis]) % target_tp_size:
                        raise ValueError(
                            "source KV heads are not divisible by target TP"
                        )
                    for rank, shard in enumerate(
                        selected.chunk(target_tp_size, dim=head_axis)
                    ):
                        flat = shard.contiguous().view(-1)
                        packed_by_rank[rank].narrow(
                            0, offsets[rank], flat.numel()
                        ).copy_(flat)
                        offsets[rank] += flat.numel()
            producer_stream.synchronize()
            if offsets != [rank_elements] * target_tp_size:
                raise RuntimeError("GPU-direct packed history size differs")
            for rank, connection in enumerate(connections):
                ready = recv_json(connection)
                if (
                    ready.get("status") != "READY_TO_RECV"
                    or int(ready.get("numel", -1)) != rank_elements
                ):
                    raise RuntimeError(
                        f"GPU-direct rank {rank} rejected packed history"
                    )
            _send_group(nccl, comms, packed_by_rank, send_streams)
            for rank, connection in enumerate(connections):
                received = recv_json(connection)
                if received.get("status") != "RECEIVED":
                    raise RuntimeError(
                        f"GPU-direct rank {rank} missed packed history"
                    )
            for rank, connection in enumerate(connections):
                final = recv_json(connection)
                if final.get("status") != "COMPLETE":
                    raise RuntimeError(f"GPU-direct rank {rank} is incomplete")
            elapsed_ms = (time.perf_counter() - started) * 1000
            bytes_by_rank = [
                value.numel() * value.element_size()
                for value in packed_by_rank
            ]
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
