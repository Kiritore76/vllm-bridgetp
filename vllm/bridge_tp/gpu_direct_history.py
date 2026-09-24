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

from vllm.bridge_tp.stream_protocol import (
    PayloadType,
    PersistentChannelLifecycle,
    ProtocolViolation,
    SessionEnvelope,
    recv_json,
    send_json,
)
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclDataTypeEnum,
)

TRANSPORT = "NCCL_P2P_GPU_DIRECT"
INTEGRITY = "NCCL_COMPLETION_PLUS_GPU_EXACT_READBACK"
_TIMEOUT_S = 600.0
_WARMUP_ELEMENTS = 256


def _session_message(
    operation: str,
    envelope: SessionEnvelope,
    **extra: Any,
) -> dict[str, Any]:
    return {"op": operation, **envelope.to_wire(), **extra}


def _ack_envelope(envelope: SessionEnvelope) -> SessionEnvelope:
    return SessionEnvelope(
        channel_generation=envelope.channel_generation,
        migration_id=envelope.migration_id,
        request_id=envelope.request_id,
        sequence_number=envelope.sequence_number,
        payload_type=PayloadType.ACK,
        token_start=envelope.token_start,
        token_end=envelope.token_end,
        rank=envelope.rank,
    )


def _validate_ack(
    value: dict[str, Any],
    *,
    expected_status: str,
    expected: SessionEnvelope,
) -> None:
    if value.get("status") != expected_status:
        raise ProtocolViolation(
            f"persistent-channel rank {expected.rank} returned "
            f"{value.get('status')!r}, expected {expected_status!r}"
        )
    received = SessionEnvelope.from_wire(value)
    expected_ack = _ack_envelope(expected)
    if received != expected_ack:
        raise ProtocolViolation("persistent-channel ACK identity differs")


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
        self.persistent_channel = False
        self.channel_generation = 0
        self.topology_key = ""
        self.lifecycle: PersistentChannelLifecycle | None = None
        self._active_request_id: str | None = None
        self._last_delta_envelope: SessionEnvelope | None = None
        self._packed_buffer: torch.Tensor | None = None
        self.buffer_capacity_elements = 0
        self.buffer_high_water_bytes = 0
        self.last_session_payload_released_bytes = 0
        self.allow_idle_wait = False
        self.warmup_completed_unix_s: float | None = None

    def warmup_preconnected_channel(self, *, rank: int) -> int:
        """Verify a tiny NCCL receive before starting any KV session."""
        if (
            not self.persistent_channel
            or self.lifecycle is None
            or self.lifecycle.state.value != "IDLE"
            or self.lifecycle.active_session is not None
            or self.connection is None
            or self.nccl is None
            or self.comm is None
            or self.stream is None
            or self.warmup_completed_unix_s is not None
        ):
            raise ProtocolViolation("warmup requires an idle preconnected channel")
        header = recv_json(self.connection)
        if (
            header.get("op") != "WARMUP"
            or int(header.get("channel_generation", -1)) != self.channel_generation
            or int(header.get("target_tp_rank", -1)) != rank
            or int(header.get("numel", -1)) != _WARMUP_ELEMENTS
            or header.get("dtype") != "float16"
        ):
            raise ProtocolViolation("GPU-direct warmup header differs")
        with torch.cuda.device(self.device):
            payload = torch.empty(
                _WARMUP_ELEMENTS, dtype=torch.float16, device=self.device
            )
        send_json(self.connection, {"status": "WARMUP_READY", "target_tp_rank": rank})
        done = _recv_tensor(self.nccl, self.comm, payload, self.stream)
        done.synchronize()
        if not bool(torch.all(payload == rank + 1).item()):
            raise ProtocolViolation("GPU-direct warmup payload differs")
        send_json(
            self.connection,
            {"status": "WARMUP_COMPLETE", "target_tp_rank": rank},
        )
        self.warmup_completed_unix_s = time.time()
        return payload.numel() * payload.element_size()

    def _open(self, *, migration_id: str, rank: int) -> None:
        if self.connection is not None:
            return
        self.nccl = NCCLLibrary()
        connection, _ = self.listener.accept()
        connection.settimeout(_TIMEOUT_S)
        hello = recv_json(connection)
        persistent_channel = bool(hello.get("persistent_channel", False))
        if not persistent_channel and hello.get("migration_id") != migration_id:
            raise ValueError("GPU-direct migration ID differs")
        if int(hello.get("target_tp_rank", -1)) != rank:
            raise ValueError("GPU-direct target rank differs")
        unique_bytes = base64.b64decode(str(hello["nccl_unique_id_b64"]))
        unique_id = self.nccl.unique_id_from_bytes(unique_bytes)
        with torch.cuda.device(self.device):
            self.comm = self.nccl.ncclCommInitRank(2, unique_id, 1)
            self.stream = torch.cuda.Stream(device=self.device)
        self.connection = connection
        if persistent_channel:
            generation = int(hello.get("channel_generation", -1))
            topology_key = str(hello.get("topology_key", ""))
            if generation < 0 or not topology_key:
                raise ProtocolViolation(
                    "persistent-channel hello is missing channel identity"
                )
            self.persistent_channel = True
            self.channel_generation = generation
            self.topology_key = topology_key
            self.lifecycle = PersistentChannelLifecycle(
                topology_key=topology_key,
                expected_ranks=frozenset({rank}),
                channel_generation=generation,
            )
            self.lifecycle.mark_open()
            send_json(
                connection,
                {
                    "status": "CHANNEL_READY",
                    "channel_generation": generation,
                    "target_tp_rank": rank,
                },
            )
        else:
            send_json(connection, {"status": "COMM_READY"})

    def _start_persistent_session(
        self,
        *,
        migration_id: str,
        request_id: str,
    ) -> SessionEnvelope:
        if self.connection is None or self.lifecycle is None:
            raise RuntimeError("persistent GPU-direct channel is not open")
        message = recv_json(self.connection)
        if self.allow_idle_wait:
            self.connection.settimeout(_TIMEOUT_S)
        if message.get("op") != "START_SESSION":
            raise ProtocolViolation("expected START_SESSION")
        envelope = SessionEnvelope.from_wire(message)
        if envelope.payload_type is not PayloadType.START_SESSION:
            raise ProtocolViolation("START_SESSION payload type differs")
        if envelope.migration_id != migration_id:
            raise ProtocolViolation("START_SESSION migration_id differs")
        if envelope.request_id != request_id:
            raise ProtocolViolation("START_SESSION request_id differs")
        session = self.lifecycle.start_session(
            migration_id=migration_id,
            request_id=request_id,
        )
        self.last_session_payload_released_bytes = 0
        session.accept_inbound(envelope)
        self._active_request_id = request_id
        send_json(
            self.connection,
            {
                "status": "SESSION_READY",
                **_ack_envelope(envelope).to_wire(),
            },
        )
        return envelope

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
        envelope: SessionEnvelope | None = None,
    ) -> tuple[torch.Tensor, float, torch.cuda.Event]:
        if self.connection is None or self.nccl is None:
            raise RuntimeError("GPU-direct receiver is not connected")
        if self.comm is None or self.stream is None:
            raise RuntimeError("GPU-direct receiver communicator is missing")
        counts = [math.prod(shape) for shape in shapes]
        required_elements = sum(counts)
        with torch.cuda.device(self.device):
            reusable_buffer = (
                self.persistent_channel
                and self._packed_buffer is not None
                and self._packed_buffer.dtype == dtype
                and self.buffer_capacity_elements >= required_elements
            )
            if not reusable_buffer:
                capacity = max(required_elements, self.buffer_capacity_elements)
                self._packed_buffer = torch.empty(
                    capacity,
                    dtype=dtype,
                    device=self.device,
                )
                self.buffer_capacity_elements = capacity
                self.buffer_high_water_bytes = max(
                    self.buffer_high_water_bytes,
                    capacity * self._packed_buffer.element_size(),
                )
            assert self._packed_buffer is not None
            packed = self._packed_buffer.narrow(0, 0, required_elements)
        ready: dict[str, Any] = {
            "status": "READY_TO_RECV",
            "tensor_id": tensor_key,
            "numel": packed.numel(),
        }
        if envelope is not None:
            ready.update(_ack_envelope(envelope).to_wire())
        send_json(self.connection, ready)
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
        request_id: str | None = None,
        rank: int,
        layer_records: list[dict[str, Any]],
        keep_open: bool = False,
        synchronize: bool = True,
    ) -> ReceiveResult:
        started = time.perf_counter()
        self._open(migration_id=migration_id, rank=rank)
        try:
            history_envelope: SessionEnvelope | None = None
            if self.persistent_channel:
                if not keep_open:
                    raise ProtocolViolation(
                        "persistent receiver must keep channel open"
                    )
                if not request_id:
                    raise ProtocolViolation(
                        "persistent history requires request_id"
                    )
                self._start_persistent_session(
                    migration_id=migration_id,
                    request_id=request_id,
                )
                assert self.connection is not None
                history_message = recv_json(self.connection)
                if history_message.get("op") != "HISTORY":
                    raise ProtocolViolation("expected HISTORY")
                history_envelope = SessionEnvelope.from_wire(history_message)
                if history_envelope.payload_type is not PayloadType.HISTORY:
                    raise ProtocolViolation("HISTORY payload type differs")
                assert self.lifecycle is not None
                assert self.lifecycle.active_session is not None
                self.lifecycle.active_session.accept_inbound(history_envelope)
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
                envelope=history_envelope,
            )
            assert self.connection is not None
            if history_envelope is None:
                send_json(self.connection, {"status": "RECEIVED"})
            else:
                send_json(
                    self.connection,
                    {
                        "status": "RECEIVED",
                        **_ack_envelope(history_envelope).to_wire(),
                    },
                )
            layers = self._unpack(
                packed,
                [str(row["layer_name"]) for row in layer_records],
                shapes,
            )
            raw_bytes = packed.numel() * packed.element_size()
            if history_envelope is None:
                send_json(self.connection, {"status": "COMPLETE"})
            else:
                send_json(
                    self.connection,
                    {
                        "status": "COMPLETE",
                        **_ack_envelope(history_envelope).to_wire(),
                    },
                )
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
        if (
            getattr(self, "persistent_channel", False)
            and header.get("op") == "END_SESSION"
        ):
            envelope = SessionEnvelope.from_wire(header)
            if envelope.payload_type is not PayloadType.END_SESSION:
                raise ProtocolViolation("END_SESSION payload type differs")
            if self.lifecycle is None or self.lifecycle.active_session is None:
                raise ProtocolViolation("persistent channel has no active session")
            session = self.lifecycle.active_session
            session.accept_inbound(envelope)
            session.mark_rank_ready(rank)
            session.commit()
            send_json(
                self.connection,
                {
                    "status": "SESSION_ENDED",
                    **_ack_envelope(envelope).to_wire(),
                },
            )
            self.lifecycle.finish_session()
            self._active_request_id = None
            self._last_delta_envelope = None
            self.last_session_payload_released_bytes = (
                self.release_session_payload_buffer()
            )
            if self.allow_idle_wait:
                self.connection.settimeout(None)
            return None
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
        delta_envelope: SessionEnvelope | None = None
        if getattr(self, "persistent_channel", False):
            delta_envelope = SessionEnvelope.from_wire(header)
            if delta_envelope.payload_type is not PayloadType.DELTA:
                raise ProtocolViolation("DELTA payload type differs")
            if self.lifecycle is None or self.lifecycle.active_session is None:
                raise ProtocolViolation("persistent channel has no active session")
            self.lifecycle.active_session.accept_inbound(delta_envelope)
            self._last_delta_envelope = delta_envelope
        shapes = [[int(value) for value in row] for row in header["shapes"]]
        names = [str(value) for value in header["layer_names"]]
        started = time.perf_counter()
        packed, _, receive_done = self._receive_packed(
            shapes=shapes,
            dtype=self._dtype(str(header["dtype"])),
            tensor_key=str(header["tensor_id"]),
            synchronize=synchronize,
            envelope=delta_envelope,
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
        if getattr(self, "persistent_channel", False):
            envelope = self._last_delta_envelope
            if envelope is None:
                raise ProtocolViolation("persistent delta ACK has no message")
            if (
                envelope.token_start != start_token
                or envelope.token_end != end_token
            ):
                raise ProtocolViolation("persistent delta ACK range differs")
            send_json(
                self.connection,
                {
                    "status": "APPLIED",
                    **_ack_envelope(envelope).to_wire(),
                },
            )
            self._last_delta_envelope = None
        else:
            send_json(
                self.connection,
                {
                    "status": "APPLIED",
                    "start_token": start_token,
                    "end_token": end_token,
                },
            )

    def release_session_payload_buffer(self) -> int:
        """Drop old-session KV payload storage while retaining the channel.

        The persistent communicator is process-lifetime state, whereas the
        packed receive buffer contains a previous request's KV payload.  Do
        not let the latter look like a request-level KV leak.  CUDA's caching
        allocator may keep the released bytes in ``memory_reserved`` for
        reuse, but ``memory_allocated`` will no longer include this buffer.
        """
        if self.lifecycle is None or self.lifecycle.state.value != "IDLE":
            raise RuntimeError(
                "persistent payload buffer can only be released while IDLE"
            )
        released = (
            self._packed_buffer.numel() * self._packed_buffer.element_size()
            if self._packed_buffer is not None
            else 0
        )
        self._packed_buffer = None
        self.buffer_capacity_elements = 0
        return released

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
        if self.lifecycle is not None:
            if self.lifecycle.active_session is not None:
                self.lifecycle.active_session.cancel()
                self.lifecycle.finish_session()
            self.lifecycle.shutdown()
        if self._destroy_thread is None and self.comm is not None:
            assert self.nccl is not None
            self.nccl.ncclCommDestroy(self.comm)
            self.comm = None
            self.nccl = None
        self.stream = None
        self._packed_buffer = None
        self.buffer_capacity_elements = 0
        self._close_control()

    def abort(self) -> None:
        """Release a pooled communicator during worker shutdown."""
        if self.lifecycle is not None:
            if self.lifecycle.active_session is not None:
                self.lifecycle.active_session.cancel()
                self.lifecycle.finish_session()
            self.lifecycle.shutdown()
        if self.comm is not None:
            assert self.nccl is not None
            self.nccl.ncclCommAbort(self.comm)
            self.comm = None
            self.nccl = None
        self.stream = None
        self._packed_buffer = None
        self.buffer_capacity_elements = 0
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
        self.target_addresses: tuple[str, ...] = ()
        self.persistent_channel = False
        self.channel_generation = 0
        self.topology_key = ""
        self.lifecycle: PersistentChannelLifecycle | None = None
        self._active_migration_id: str | None = None
        self._active_request_id: str | None = None
        self._next_sequence_by_rank: dict[int, int] = {}
        self._packed_buffers: list[torch.Tensor] = []
        self.buffer_capacity_elements = 0
        self.buffer_high_water_bytes = 0
        self.last_session_payload_released_bytes = 0
        self.preconnect_completed_unix_s: float | None = None
        self.warmup_completed_unix_s: float | None = None

    def warmup_preconnected_channel(self) -> int:
        """Exercise all preconnected ranks with a tiny NCCL payload."""
        if (
            not self.persistent_channel
            or self.lifecycle is None
            or self.lifecycle.state.value != "IDLE"
            or self.lifecycle.active_session is not None
            or self.nccl is None
            or self.target_tp_size <= 0
            or len(self.connections) != self.target_tp_size
            or len(self.comms) != self.target_tp_size
            or self.warmup_completed_unix_s is not None
        ):
            raise ProtocolViolation("warmup requires an idle preconnected channel")
        with torch.cuda.device(self.device):
            payloads = [
                torch.full(
                    (_WARMUP_ELEMENTS,), rank + 1,
                    dtype=torch.float16, device=self.device,
                )
                for rank in range(self.target_tp_size)
            ]
            streams = [
                torch.cuda.Stream(device=self.device)
                for _ in range(self.target_tp_size)
            ]
            # The test payload is created on the current stream, while NCCL
            # uses separate streams. Finish the fill before publishing it.
            torch.cuda.current_stream(self.device).synchronize()
        for rank, connection in enumerate(self.connections):
            send_json(
                connection,
                {
                    "op": "WARMUP",
                    "channel_generation": self.channel_generation,
                    "target_tp_rank": rank,
                    "numel": _WARMUP_ELEMENTS,
                    "dtype": "float16",
                },
            )
        for rank, connection in enumerate(self.connections):
            ready = recv_json(connection)
            if ready != {"status": "WARMUP_READY", "target_tp_rank": rank}:
                raise ProtocolViolation(f"GPU-direct rank {rank} warmup not ready")
        _send_group(self.nccl, self.comms, payloads, streams)
        for rank, connection in enumerate(self.connections):
            complete = recv_json(connection)
            if complete != {"status": "WARMUP_COMPLETE", "target_tp_rank": rank}:
                raise ProtocolViolation(f"GPU-direct rank {rank} warmup failed")
        self.streams = streams
        self.warmup_completed_unix_s = time.time()
        return sum(value.numel() * value.element_size() for value in payloads)

    def preconnect(
        self,
        *,
        channel_generation: int,
        target_addresses: list[str],
    ) -> None:
        """Open a persistent topology channel without starting a KV session."""
        if self.nccl is not None:
            raise RuntimeError("GPU-direct sender channel is already open")
        if not target_addresses or channel_generation < 0:
            raise ValueError("invalid GPU-direct preconnect topology")
        topology_key = "|".join(target_addresses)
        nccl = NCCLLibrary()
        connections: list[socket.socket] = []
        comms: list[Any] = []
        try:
            for rank, address in enumerate(target_addresses):
                connection = _connect(address)
                connections.append(connection)
                unique_id = nccl.ncclGetUniqueId()
                send_json(
                    connection,
                    {
                        "migration_id": "",
                        "target_tp_rank": rank,
                        "nccl_unique_id_b64": base64.b64encode(
                            bytes(unique_id.internal)
                        ).decode("ascii"),
                        "persistent_channel": True,
                        "channel_generation": channel_generation,
                        "topology_key": topology_key,
                    },
                )
                with torch.cuda.device(self.device):
                    comm = nccl.ncclCommInitRank(2, unique_id, 0)
                comms.append(comm)
                ready = recv_json(connection)
                if (
                    ready.get("status") != "CHANNEL_READY"
                    or int(ready.get("channel_generation", -1))
                    != channel_generation
                    or int(ready.get("target_tp_rank", -1)) != rank
                ):
                    raise ProtocolViolation(
                        f"GPU-direct rank {rank} preconnect identity differs"
                    )
        except BaseException:
            for comm in comms:
                nccl.ncclCommDestroy(comm)
            for connection in connections:
                connection.close()
            raise
        self.nccl = nccl
        self.connections = connections
        self.comms = comms
        self.target_tp_size = len(target_addresses)
        self.target_addresses = tuple(target_addresses)
        self.persistent_channel = True
        self.channel_generation = channel_generation
        self.topology_key = topology_key
        self.lifecycle = PersistentChannelLifecycle(
            topology_key=topology_key,
            expected_ranks=frozenset(range(len(target_addresses))),
            channel_generation=channel_generation,
        )
        self.lifecycle.mark_open()
        self.preconnect_completed_unix_s = time.time()

    def _next_envelope(
        self,
        *,
        rank: int,
        payload_type: PayloadType,
        token_start: int,
        token_end: int,
    ) -> SessionEnvelope:
        if self._active_migration_id is None or self._active_request_id is None:
            raise ProtocolViolation("persistent sender has no active session")
        sequence = self._next_sequence_by_rank[rank]
        self._next_sequence_by_rank[rank] = sequence + 1
        return SessionEnvelope(
            channel_generation=self.channel_generation,
            migration_id=self._active_migration_id,
            request_id=self._active_request_id,
            sequence_number=sequence,
            payload_type=payload_type,
            token_start=token_start,
            token_end=token_end,
            rank=rank,
        )

    def send(
        self,
        *,
        migration_id: str,
        request_id: str | None = None,
        channel_generation: int = 0,
        persistent_channel: bool = False,
        history_end_token: int | None = None,
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
        if persistent_channel and not request_id:
            raise ValueError("persistent GPU-direct history requires request_id")
        if persistent_channel and history_end_token is None:
            raise ValueError(
                "persistent GPU-direct history requires history_end_token"
            )
        if persistent_channel and not keep_open:
            raise ValueError("persistent GPU-direct history must keep channel open")

        channel_setup_started = time.perf_counter()
        topology_key = "|".join(target_addresses)
        reuse_channel = persistent_channel and self.nccl is not None
        if reuse_channel:
            if not self.persistent_channel or self.lifecycle is None:
                raise ProtocolViolation("sender does not own a persistent channel")
            if channel_generation != self.channel_generation:
                raise ProtocolViolation("sender channel generation differs")
            if topology_key != self.topology_key:
                raise ProtocolViolation("sender topology differs")
            if target_tp_size != self.target_tp_size:
                raise ProtocolViolation("sender target TP size differs")
            nccl = self.nccl
            connections = self.connections
            comms = self.comms
        else:
            nccl = NCCLLibrary()
            connections = []
            comms = []
        succeeded = False
        try:
            if not reuse_channel:
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
                            "persistent_channel": persistent_channel,
                            "channel_generation": channel_generation,
                            "topology_key": topology_key,
                        },
                    )
                    with torch.cuda.device(self.device):
                        comm = nccl.ncclCommInitRank(2, unique_id, 0)
                    ready = recv_json(connection)
                    expected_status = (
                        "CHANNEL_READY" if persistent_channel else "COMM_READY"
                    )
                    if ready.get("status") != expected_status:
                        raise RuntimeError(
                            f"GPU-direct rank {rank} did not become ready"
                        )
                    if persistent_channel and (
                        int(ready.get("channel_generation", -1))
                        != channel_generation
                        or int(ready.get("target_tp_rank", -1)) != rank
                    ):
                        raise ProtocolViolation(
                            f"GPU-direct rank {rank} channel identity differs"
                        )
                    connections.append(connection)
                    comms.append(comm)
                if persistent_channel:
                    self.persistent_channel = True
                    self.channel_generation = channel_generation
                    self.topology_key = topology_key
                    self.lifecycle = PersistentChannelLifecycle(
                        topology_key=topology_key,
                        expected_ranks=frozenset(range(target_tp_size)),
                        channel_generation=channel_generation,
                    )
                    self.lifecycle.mark_open()

            channel_setup_ms = (
                time.perf_counter() - channel_setup_started
            ) * 1000
            session_handshake_started = time.perf_counter()
            if persistent_channel:
                assert request_id is not None
                assert self.lifecycle is not None
                self.lifecycle.start_session(
                    migration_id=migration_id,
                    request_id=request_id,
                )
                self.last_session_payload_released_bytes = 0
                self._active_migration_id = migration_id
                self._active_request_id = request_id
                self._next_sequence_by_rank = {
                    rank: 0 for rank in range(target_tp_size)
                }
                for rank, connection in enumerate(connections):
                    envelope = self._next_envelope(
                        rank=rank,
                        payload_type=PayloadType.START_SESSION,
                        token_start=0,
                        token_end=0,
                    )
                    send_json(
                        connection,
                        _session_message("START_SESSION", envelope),
                    )
                    _validate_ack(
                        recv_json(connection),
                        expected_status="SESSION_READY",
                        expected=envelope,
                    )

            session_handshake_ms = (
                time.perf_counter() - session_handshake_started
            ) * 1000
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
                producer_stream = self.producer_stream
                if producer_stream is None:
                    producer_stream = torch.cuda.Stream(device=self.device)
                send_streams = self.streams
                if len(send_streams) != target_tp_size:
                    send_streams = [
                        torch.cuda.Stream(device=self.device)
                        for _ in range(target_tp_size)
                    ]
                reusable_buffers = (
                    persistent_channel
                    and len(self._packed_buffers) == target_tp_size
                    and self.buffer_capacity_elements >= rank_elements
                    and all(value.dtype == dtype for value in self._packed_buffers)
                )
                if reusable_buffers:
                    packed_by_rank = [
                        value.narrow(0, 0, rank_elements)
                        for value in self._packed_buffers
                    ]
                else:
                    capacity = max(rank_elements, self.buffer_capacity_elements)
                    self._packed_buffers = [
                        torch.empty(capacity, dtype=dtype, device=self.device)
                        for _ in range(target_tp_size)
                    ]
                    self.buffer_capacity_elements = capacity
                    packed_by_rank = [
                        value.narrow(0, 0, rank_elements)
                        for value in self._packed_buffers
                    ]
                    self.buffer_high_water_bytes = max(
                        self.buffer_high_water_bytes,
                        capacity
                        * kv_caches[0].element_size()
                        * target_tp_size,
                    )
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
            pack_ms = (time.perf_counter() - started) * 1000
            if offsets != [rank_elements] * target_tp_size:
                raise RuntimeError("GPU-direct packed history size differs")
            receiver_ready_started = time.perf_counter()
            history_envelopes: dict[int, SessionEnvelope] = {}
            if persistent_channel:
                assert history_end_token is not None
                assert self.lifecycle is not None
                assert self.lifecycle.active_session is not None
                for rank, connection in enumerate(connections):
                    envelope = self._next_envelope(
                        rank=rank,
                        payload_type=PayloadType.HISTORY,
                        token_start=0,
                        token_end=history_end_token,
                    )
                    history_envelopes[rank] = envelope
                    self.lifecycle.active_session.expect_ack(
                        rank=rank,
                        sequence_number=envelope.sequence_number,
                    )
                    send_json(
                        connection,
                        _session_message("HISTORY", envelope),
                    )
            for rank, connection in enumerate(connections):
                ready = recv_json(connection)
                if (
                    ready.get("status") != "READY_TO_RECV"
                    or int(ready.get("numel", -1)) != rank_elements
                ):
                    raise RuntimeError(
                        f"GPU-direct rank {rank} rejected packed history"
                    )
                if persistent_channel:
                    _validate_ack(
                        ready,
                        expected_status="READY_TO_RECV",
                        expected=history_envelopes[rank],
                    )
            receiver_ready_ms = (
                time.perf_counter() - receiver_ready_started
            ) * 1000
            nccl_send_started = time.perf_counter()
            _send_group(nccl, comms, packed_by_rank, send_streams)
            nccl_send_ms = (time.perf_counter() - nccl_send_started) * 1000
            completion_ack_started = time.perf_counter()
            for rank, connection in enumerate(connections):
                received = recv_json(connection)
                if received.get("status") != "RECEIVED":
                    raise RuntimeError(
                        f"GPU-direct rank {rank} missed packed history"
                    )
                if persistent_channel:
                    _validate_ack(
                        received,
                        expected_status="RECEIVED",
                        expected=history_envelopes[rank],
                    )
            for rank, connection in enumerate(connections):
                final = recv_json(connection)
                if final.get("status") != "COMPLETE":
                    raise RuntimeError(f"GPU-direct rank {rank} is incomplete")
                if persistent_channel:
                    _validate_ack(
                        final,
                        expected_status="COMPLETE",
                        expected=history_envelopes[rank],
                    )
                    assert self.lifecycle is not None
                    assert self.lifecycle.active_session is not None
                    self.lifecycle.active_session.acknowledge(
                        rank=rank,
                        sequence_number=(
                            history_envelopes[rank].sequence_number
                        ),
                    )
            completion_ack_ms = (
                time.perf_counter() - completion_ack_started
            ) * 1000
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
                    "channel_reused": reuse_channel,
                    "preconnect_completed_unix_s": (
                        self.preconnect_completed_unix_s
                    ),
                    "warmup_completed_unix_s": self.warmup_completed_unix_s,
                    "channel_setup_ms": channel_setup_ms,
                    "session_handshake_ms": session_handshake_ms,
                    "pack_ms": pack_ms,
                    "receiver_ready_ms": receiver_ready_ms,
                    "nccl_send_ms": nccl_send_ms,
                    "completion_ack_ms": completion_ack_ms,
                    "channel_generation": (
                        self.channel_generation if persistent_channel else None
                    ),
                    "channel_create_count": (
                        self.lifecycle.create_count
                        if persistent_channel and self.lifecycle is not None
                        else None
                    ),
                    "channel_session_count": (
                        self.lifecycle.session_count
                        if persistent_channel and self.lifecycle is not None
                        else None
                    ),
                    "buffer_high_water_bytes": (
                        self.buffer_high_water_bytes
                        if persistent_channel
                        else None
                    ),
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
                self.target_addresses = tuple(target_addresses)
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
        if self.persistent_channel and migration_id != self._active_migration_id:
            raise ProtocolViolation("persistent delta migration_id differs")
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
        delta_envelopes: dict[int, SessionEnvelope] = {}
        for rank, connection in enumerate(self.connections):
            identity: dict[str, Any] = {}
            if self.persistent_channel:
                if self.lifecycle is None or self.lifecycle.active_session is None:
                    raise ProtocolViolation(
                        "persistent sender has no active session"
                    )
                envelope = self._next_envelope(
                    rank=rank,
                    payload_type=PayloadType.DELTA,
                    token_start=start_token,
                    token_end=end_token,
                )
                delta_envelopes[rank] = envelope
                self.lifecycle.active_session.expect_ack(
                    rank=rank,
                    sequence_number=envelope.sequence_number,
                )
                identity = envelope.to_wire()
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
                    **identity,
                },
            )
        for rank, connection in enumerate(self.connections):
            ready = recv_json(connection)
            if (
                ready.get("status") != "READY_TO_RECV"
                or int(ready.get("numel", -1)) != packed_by_rank[rank].numel()
            ):
                raise RuntimeError(f"GPU-direct rank {rank} rejected delta")
            if self.persistent_channel:
                _validate_ack(
                    ready,
                    expected_status="READY_TO_RECV",
                    expected=delta_envelopes[rank],
                )
        receiver_ready_ms = (
            time.perf_counter() - receiver_ready_started
        ) * 1000
        nccl_started = time.perf_counter()
        _send_group(self.nccl, self.comms, packed_by_rank, self.streams)
        nccl_send_ms = (time.perf_counter() - nccl_started) * 1000
        apply_ack_started = time.perf_counter()
        for rank, connection in enumerate(self.connections):
            applied = recv_json(connection)
            if self.persistent_channel:
                _validate_ack(
                    applied,
                    expected_status="APPLIED",
                    expected=delta_envelopes[rank],
                )
                assert self.lifecycle is not None
                assert self.lifecycle.active_session is not None
                self.lifecycle.active_session.acknowledge(
                    rank=rank,
                    sequence_number=delta_envelopes[rank].sequence_number,
                )
            elif (
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

    def end_session(self) -> None:
        """End one request while preserving the persistent channel."""
        if not self.persistent_channel or self.lifecycle is None:
            raise ProtocolViolation("sender does not own a persistent channel")
        session = self.lifecycle.active_session
        if session is None:
            raise ProtocolViolation("persistent sender has no active session")
        end_envelopes: dict[int, SessionEnvelope] = {}
        for rank, connection in enumerate(self.connections):
            envelope = self._next_envelope(
                rank=rank,
                payload_type=PayloadType.END_SESSION,
                token_start=session.delta_watermarks.get(
                    rank, session.history_watermarks.get(rank, 0)
                ),
                token_end=session.delta_watermarks.get(
                    rank, session.history_watermarks.get(rank, 0)
                ),
            )
            end_envelopes[rank] = envelope
            send_json(connection, _session_message("END_SESSION", envelope))
        for rank, connection in enumerate(self.connections):
            _validate_ack(
                recv_json(connection),
                expected_status="SESSION_ENDED",
                expected=end_envelopes[rank],
            )
            session.mark_rank_ready(rank)
        session.commit()
        self.lifecycle.finish_session()
        self._active_migration_id = None
        self._active_request_id = None
        self._next_sequence_by_rank.clear()
        self.last_session_payload_released_bytes = (
            self.release_session_payload_buffers()
        )

    def release_session_payload_buffers(self) -> int:
        """Drop completed-request payload buffers without closing NCCL."""
        if self.lifecycle is None or self.lifecycle.state.value != "IDLE":
            raise RuntimeError(
                "persistent payload buffers can only be released while IDLE"
            )
        released = sum(
            value.numel() * value.element_size() for value in self._packed_buffers
        )
        self._packed_buffers = []
        self.buffer_capacity_elements = 0
        return released

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
        if self.lifecycle is not None:
            if self.lifecycle.active_session is not None:
                self.lifecycle.active_session.cancel()
                self.lifecycle.finish_session()
            self.lifecycle.shutdown()
        if self.nccl is not None:
            for comm in self.comms:
                self.nccl.ncclCommDestroy(comm)
        self.producer_stream = None
        self._packed_buffers = []
        self.buffer_capacity_elements = 0
        self.comms = []
        self.streams = []
        self.nccl = None

    def abort(self) -> None:
        """Release pooled communicators without peer-finalize coordination."""
        self.close_control()
        if self.lifecycle is not None:
            if self.lifecycle.active_session is not None:
                self.lifecycle.active_session.cancel()
                self.lifecycle.finish_session()
            self.lifecycle.shutdown()
        if self.nccl is not None:
            for comm in self.comms:
                self.nccl.ncclCommAbort(comm)
        self.producer_stream = None
        self._packed_buffers = []
        self.buffer_capacity_elements = 0
        self.comms = []
        self.streams = []
        self.nccl = None
