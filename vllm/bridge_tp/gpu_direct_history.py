# SPDX-License-Identifier: Apache-2.0
"""Dependency-free NCCL point-to-point transport for Shadow history.

The source and target vLLM engines are independent distributed worlds. A
dedicated two-rank NCCL communicator is negotiated over a small TCP control
socket for each TP4 rank. Only JSON metadata and acknowledgements use the
socket; every KV tensor remains on CUDA.
"""

from __future__ import annotations

import base64
import math
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

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
) -> torch.cuda.Event:
    with torch.cuda.stream(stream):
        nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            0,
            comm,
            cudaStream_t(stream.cuda_stream),
        )
        receive_done = torch.cuda.Event(enable_timing=False)
        receive_done.record(stream)
    return receive_done


@dataclass
class ReceiveResult:
    layers: dict[str, torch.Tensor]
    raw_tensor_bytes: int
    receive_ms: float
    receive_done_event: torch.cuda.Event | None = None
    start_token: int | None = None
    end_token: int | None = None


class GpuDirectHistoryReceiver:
    """Receive one TP4 rank's historical shard directly into CUDA tensors."""

    def __init__(
        self,
        *,
        device: torch.device,
        host: str,
        port: int,
        defer_communicator_destroy: bool = False,
    ) -> None:
        self.device = _cuda_device(device)
        self.defer_communicator_destroy = defer_communicator_destroy
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((host, port))
        self.listener.listen(1)
        self.listener.settimeout(_TIMEOUT_S)
        self.nccl: NCCLLibrary | None = None
        self.connection: socket.socket | None = None
        self.comm: Any | None = None
        self.stream: torch.cuda.Stream | None = None
        self.terminal_close_received_unix_s: float | None = None
        self.control_closed_unix_s: float | None = None
        self.destroy_started_unix_s: float | None = None
        self.destroy_completed_unix_s: float | None = None
        self.destroy_error: str | None = None
        self._destroy_thread: threading.Thread | None = None

    def _open(self, *, migration_id: str, rank: int) -> None:
        if self.connection is not None:
            return
        self.nccl = NCCLLibrary()
        connection, _ = self.listener.accept()
        connection.settimeout(_TIMEOUT_S)
        hello = recv_json(connection)
        if hello.get("migration_id") != migration_id:
            raise ValueError("GPU-direct migration ID differs")
        if int(hello.get("target_tp_rank", -1)) != rank:
            raise ValueError("GPU-direct target rank differs")
        unique_bytes = base64.b64decode(str(hello["nccl_unique_id_b64"]))
        unique_id = self.nccl.unique_id_from_bytes(unique_bytes)
        with torch.cuda.device(self.device):
            self.comm = self.nccl.ncclCommInitRank(2, unique_id, 1)
            self.stream = torch.cuda.Stream(device=self.device)
        self.connection = connection
        send_json(connection, {"status": "COMM_READY"})

    @staticmethod
    def _dtype(name: str) -> torch.dtype:
        value = getattr(torch, name.removeprefix("torch."), None)
        if not isinstance(value, torch.dtype):
            raise ValueError(f"unsupported GPU-direct dtype {name!r}")
        return value

    def _receive_packed(
        self,
        *,
        shapes: list[list[int]],
        dtype: torch.dtype,
        tensor_key: str,
        synchronize: bool,
    ) -> tuple[torch.Tensor, float, torch.cuda.Event]:
        if self.connection is None or self.nccl is None:
            raise RuntimeError("GPU-direct receiver is not connected")
        if self.comm is None or self.stream is None:
            raise RuntimeError("GPU-direct receiver communicator is missing")
        counts = [math.prod(shape) for shape in shapes]
        with torch.cuda.device(self.device):
            packed = torch.empty(sum(counts), dtype=dtype, device=self.device)
        send_json(
            self.connection,
            {"status": "READY_TO_RECV", "tensor_id": tensor_key,
             "numel": packed.numel()},
        )
        started = time.perf_counter()
        receive_done = _recv_tensor(self.nccl, self.comm, packed, self.stream)
        if synchronize:
            receive_done.synchronize()
        return (
            packed,
            (time.perf_counter() - started) * 1000,
            receive_done,
        )

    @staticmethod
    def _unpack(
        packed: torch.Tensor,
        layer_names: list[str],
        shapes: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        layers: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape in zip(layer_names, shapes):
            count = math.prod(shape)
            layers[name] = packed.narrow(0, offset, count).view(shape)
            offset += count
        return layers

    def receive(
        self,
        *,
        migration_id: str,
        rank: int,
        layer_records: list[dict[str, Any]],
        keep_open: bool = False,
        synchronize: bool = True,
    ) -> ReceiveResult:
        started = time.perf_counter()
        self._open(migration_id=migration_id, rank=rank)
        try:
            if not layer_records:
                raise ValueError("GPU-direct history has no layer records")
            dtype_names = {
                str(row["dtype"]).removeprefix("torch.")
                for row in layer_records
            }
            if len(dtype_names) != 1:
                raise ValueError("GPU-direct packed history requires one dtype")
            dtype = self._dtype(dtype_names.pop())
            shapes = [
                [int(value) for value in row["rank_shape"]]
                for row in layer_records
            ]
            packed, _, receive_done = self._receive_packed(
                shapes=shapes,
                dtype=dtype,
                tensor_key=tensor_id(migration_id, rank, 0),
                synchronize=synchronize,
            )
            assert self.connection is not None
            send_json(self.connection, {"status": "RECEIVED"})
            layers = self._unpack(
                packed,
                [str(row["layer_name"]) for row in layer_records],
                shapes,
            )
            raw_bytes = packed.numel() * packed.element_size()
            send_json(self.connection, {"status": "COMPLETE"})
            result = ReceiveResult(
                layers=layers,
                raw_tensor_bytes=raw_bytes,
                receive_ms=(time.perf_counter() - started) * 1000,
                receive_done_event=receive_done,
            )
            if not keep_open:
                self.close()
            return result
        except Exception:
            self.close()
            raise

    def receive_delta(
        self,
        *,
        migration_id: str,
        rank: int,
        synchronize: bool = True,
    ) -> ReceiveResult | None:
        """Receive the next packed delta, or ``None`` after source close."""
        if self.connection is None:
            raise RuntimeError("history must be received before GPU delta")
        header = recv_json(self.connection)
        if header.get("op") == "CLOSE":
            self.terminal_close_received_unix_s = time.time()
            send_json(self.connection, {"status": "CLOSED"})
            if self.defer_communicator_destroy:
                # TARGET_READY is a data-dependency statement, not a resource-
                # reclamation statement.  Destroying a blocking NCCL
                # communicator here used to put communicator finalization on
                # the handoff critical path.  Close the control sockets now;
                # the connector starts destruction only after publishing the
                # authoritative TARGET_READY receipt.
                self._close_control()
            else:
                self.close()
            return None
        if header.get("op") != "DELTA":
            raise ValueError(f"unexpected GPU-direct operation {header.get('op')!r}")
        if header.get("migration_id") != migration_id:
            raise ValueError("GPU-direct delta migration ID differs")
        if int(header.get("target_tp_rank", -1)) != rank:
            raise ValueError("GPU-direct delta rank differs")
        shapes = [[int(value) for value in row] for row in header["shapes"]]
        names = [str(value) for value in header["layer_names"]]
        started = time.perf_counter()
        packed, _, receive_done = self._receive_packed(
            shapes=shapes,
            dtype=self._dtype(str(header["dtype"])),
            tensor_key=str(header["tensor_id"]),
            synchronize=synchronize,
        )
        return ReceiveResult(
            layers=self._unpack(packed, names, shapes),
            raw_tensor_bytes=packed.numel() * packed.element_size(),
            receive_ms=(time.perf_counter() - started) * 1000,
            receive_done_event=receive_done,
            start_token=int(header["start_token"]),
            end_token=int(header["end_token"]),
        )

    def acknowledge_delta(self, *, start_token: int, end_token: int) -> None:
        if self.connection is None:
            raise RuntimeError("GPU-direct receiver is closed")
        send_json(
            self.connection,
            {"status": "APPLIED", "start_token": start_token,
             "end_token": end_token},
        )

    def _close_control(self) -> None:
        closed = False
        if self.connection is not None:
            self.connection.close()
            closed = True
        self.connection = None
        try:
            if self.listener.fileno() >= 0:
                self.listener.close()
                closed = True
        except OSError:
            pass
        if closed:
            self.control_closed_unix_s = time.time()

    def destroy_async(
        self,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Destroy a quiesced communicator outside the ready critical path."""
        if self._destroy_thread is not None:
            return
        nccl = self.nccl
        comm = self.comm
        if nccl is None or comm is None:
            return
        self.nccl = None
        self.comm = None
        # Keep the Python stream alive until destroy has returned.  It owns no
        # new work after CLOSE, but retaining it avoids premature wrapper
        # reclamation while NCCL finalizes the communicator.
        stream = self.stream
        self.stream = None

        def publish(status: str) -> None:
            if on_update is None:
                return
            try:
                on_update(
                    {
                        "status": status,
                        "terminal_close_received_unix_s": (
                            self.terminal_close_received_unix_s
                        ),
                        "control_closed_unix_s": self.control_closed_unix_s,
                        "destroy_started_unix_s": self.destroy_started_unix_s,
                        "destroy_completed_unix_s": (
                            self.destroy_completed_unix_s
                        ),
                        "destroy_ms": (
                            (
                                self.destroy_completed_unix_s
                                - self.destroy_started_unix_s
                            )
                            * 1000
                            if self.destroy_started_unix_s is not None
                            and self.destroy_completed_unix_s is not None
                            else None
                        ),
                        "error": self.destroy_error,
                    }
                )
            except Exception:
                # Cleanup must not be skipped because optional diagnostic
                # evidence could not be persisted during server shutdown.
                pass

        def destroy(held_stream: torch.cuda.Stream | None = stream) -> None:
            self.destroy_started_unix_s = time.time()
            publish("DESTROYING")
            try:
                nccl.ncclCommDestroy(comm)
            except BaseException as error:
                self.destroy_error = f"{type(error).__name__}: {error}"
            finally:
                self.destroy_completed_unix_s = time.time()
                publish("ERROR" if self.destroy_error else "DESTROYED")
                del held_stream

        self._destroy_thread = threading.Thread(
            target=destroy,
            name="bridgetp-nccl-destroy",
            daemon=True,
        )
        self._destroy_thread.start()

    def close(self) -> None:
        if self._destroy_thread is None and self.comm is not None:
            assert self.nccl is not None
            self.nccl.ncclCommDestroy(self.comm)
            self.comm = None
            self.nccl = None
        self.stream = None
        self._close_control()

    def abort(self) -> None:
        """Release a pooled communicator during worker shutdown."""
        if self.comm is not None:
            assert self.nccl is not None
            self.nccl.ncclCommAbort(self.comm)
            self.comm = None
            self.nccl = None
        self.stream = None
        self._close_control()


class GpuDirectHistorySender:
    """Reshard TP1 KV on GPU and send each shard to its TP4 rank."""

    def __init__(self, *, device: torch.device, host: str, port: int) -> None:
        del host, port
        self.device = _cuda_device(device)
        self.nccl: NCCLLibrary | None = None
        self.connections: list[socket.socket] = []
        self.comms: list[Any] = []
        self.streams: list[torch.cuda.Stream] = []
        self.producer_stream: torch.cuda.Stream | None = None
        self.target_tp_size = 0

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
        keep_open: bool = False,
    ) -> list[dict[str, Any]]:
        if len(kv_caches) != len(layer_names):
            raise ValueError("KV cache and layer-name counts differ")
        if len(target_addresses) != target_tp_size:
            raise ValueError("target address count differs from TP size")

        nccl = NCCLLibrary()
        connections: list[socket.socket] = []
        comms: list[Any] = []
        succeeded = False
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
            self.streams = send_streams
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
            result = [
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
            succeeded = True
            if keep_open:
                self.nccl = nccl
                self.connections = connections
                self.comms = comms
                self.producer_stream = producer_stream
                self.target_tp_size = target_tp_size
            return result
        finally:
            if not (keep_open and succeeded):
                for comm in comms:
                    nccl.ncclCommDestroy(comm)
                for connection in connections:
                    connection.close()

    def send_delta(
        self,
        *,
        migration_id: str,
        kv_caches: list[torch.Tensor],
        layer_names: list[str],
        block_ids: list[int],
        block_axis: int,
        block_size: int,
        head_axis: int,
        expected_kv_heads: int,
        start_token: int,
        end_token: int,
    ) -> dict[str, Any]:
        """Pack and send one contiguous token range over retained NCCL links."""
        if self.nccl is None or not self.connections or not self.comms:
            raise RuntimeError("GPU-direct history session is not retained")
        if not start_token < end_token:
            raise ValueError("GPU-direct delta range is empty")
        started = time.perf_counter()
        pack_started = time.perf_counter()
        rank_layers: list[list[torch.Tensor]] = [
            [] for _ in range(self.target_tp_size)
        ]
        shapes_by_rank: list[list[list[int]]] = [
            [] for _ in range(self.target_tp_size)
        ]
        producer = self.producer_stream
        if producer is None:
            with torch.cuda.device(self.device):
                producer = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(producer):
            for cache in kv_caches:
                normalized_block_axis = (
                    block_axis if block_axis >= 0 else cache.ndim + block_axis
                )
                normalized_head_axis = (
                    head_axis if head_axis >= 0 else cache.ndim + head_axis
                )
                token_axes = [
                    axis for axis, size in enumerate(cache.shape)
                    if axis != normalized_block_axis and int(size) == block_size
                ]
                if len(token_axes) != 1:
                    raise ValueError("GPU-direct delta token axis is ambiguous")
                token_axis = token_axes[0]
                first_block = start_token // block_size
                final_block = (end_token - 1) // block_size
                physical_blocks = torch.tensor(
                    block_ids[first_block:final_block + 1],
                    dtype=torch.long,
                    device=self.device,
                )
                selected = cache.index_select(
                    normalized_block_axis, physical_blocks
                )
                remaining_axes = [
                    axis for axis in range(cache.ndim)
                    if axis not in (normalized_block_axis, token_axis)
                ]
                block_major = selected.permute(
                    normalized_block_axis, token_axis, *remaining_axes
                )
                flattened = block_major.reshape(
                    block_major.shape[0] * block_size,
                    *(cache.shape[axis] for axis in remaining_axes),
                )
                delta = flattened.narrow(
                    0,
                    start_token % block_size,
                    end_token - start_token,
                )
                delta_head_axis = 1 + remaining_axes.index(
                    normalized_head_axis
                )
                if int(delta.shape[delta_head_axis]) != expected_kv_heads:
                    raise ValueError("GPU-direct delta KV-head count differs")
                shards = delta.chunk(self.target_tp_size, dim=delta_head_axis)
                for rank, shard in enumerate(shards):
                    contiguous = shard.contiguous()
                    rank_layers[rank].append(contiguous)
                    shapes_by_rank[rank].append(list(contiguous.shape))
            packed_by_rank = [
                torch.cat([value.view(-1) for value in layers])
                for layers in rank_layers
            ]
        producer.synchronize()
        pack_ms = (time.perf_counter() - pack_started) * 1000
        receiver_ready_started = time.perf_counter()
        for rank, connection in enumerate(self.connections):
            send_json(
                connection,
                {
                    "op": "DELTA",
                    "migration_id": migration_id,
                    "target_tp_rank": rank,
                    "tensor_id": (
                        f"{migration_id}#delta#rank{rank}#"
                        f"{start_token}:{end_token}"
                    ),
                    "start_token": start_token,
                    "end_token": end_token,
                    "layout": "BLOCK_MAJOR_TOKEN_CONTIGUOUS_V1",
                    "layer_names": layer_names,
                    "shapes": shapes_by_rank[rank],
                    "dtype": str(packed_by_rank[rank].dtype),
                    "numel": packed_by_rank[rank].numel(),
                },
            )
        for rank, connection in enumerate(self.connections):
            ready = recv_json(connection)
            if (
                ready.get("status") != "READY_TO_RECV"
                or int(ready.get("numel", -1)) != packed_by_rank[rank].numel()
            ):
                raise RuntimeError(f"GPU-direct rank {rank} rejected delta")
        receiver_ready_ms = (
            time.perf_counter() - receiver_ready_started
        ) * 1000
        nccl_started = time.perf_counter()
        _send_group(self.nccl, self.comms, packed_by_rank, self.streams)
        nccl_send_ms = (time.perf_counter() - nccl_started) * 1000
        apply_ack_started = time.perf_counter()
        for rank, connection in enumerate(self.connections):
            applied = recv_json(connection)
            if (
                applied.get("status") != "APPLIED"
                or int(applied.get("start_token", -1)) != start_token
                or int(applied.get("end_token", -1)) != end_token
            ):
                raise RuntimeError(f"GPU-direct rank {rank} did not apply delta")
        target_apply_ack_ms = (
            time.perf_counter() - apply_ack_started
        ) * 1000
        elapsed_ms = (time.perf_counter() - started) * 1000
        total_bytes = sum(
            value.numel() * value.element_size() for value in packed_by_rank
        )
        return {
            "start_token": start_token,
            "end_token": end_token,
            "tokens": end_token - start_token,
            "payload_bytes": total_bytes,
            "transfer_ms": elapsed_ms,
            "pack_ms": pack_ms,
            "receiver_ready_ms": receiver_ready_ms,
            "nccl_send_ms": nccl_send_ms,
            "target_apply_ack_ms": target_apply_ack_ms,
            "layout": "BLOCK_MAJOR_TOKEN_CONTIGUOUS_V1",
            "observed_aggregate_gib_s": (
                total_bytes / 1024**3 / (elapsed_ms / 1000)
                if elapsed_ms > 0 else None
            ),
        }

    def close_control(self) -> None:
        """Complete the wire protocol without destroying communicators."""
        for connection in self.connections:
            try:
                send_json(connection, {"op": "CLOSE"})
            except OSError:
                pass
        for connection in self.connections:
            try:
                recv_json(connection)
            except (OSError, EOFError):
                pass
        for connection in self.connections:
            connection.close()
        self.connections = []

    def close(self) -> None:
        self.close_control()
        if self.nccl is not None:
            for comm in self.comms:
                self.nccl.ncclCommDestroy(comm)
        self.producer_stream = None
        self.comms = []
        self.streams = []
        self.nccl = None

    def abort(self) -> None:
        """Release pooled communicators without peer-finalize coordination."""
        self.close_control()
        if self.nccl is not None:
            for comm in self.comms:
                self.nccl.ncclCommAbort(comm)
        self.producer_stream = None
        self.comms = []
        self.streams = []
        self.nccl = None
