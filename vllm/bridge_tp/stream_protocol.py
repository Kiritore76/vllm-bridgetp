# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Framed TCP protocol used by the BridgeTP Phase 6 live transfer."""

from __future__ import annotations

import hashlib
import io
import json
import socket
import struct
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

PROTOCOL_VERSION = 1
PERSISTENT_CHANNEL_PROTOCOL_VERSION = 1
MIGRATION_PARAM = "bridgetp_migration_id"
_JSON_LENGTH = struct.Struct("!I")
_FRAME_HEADER = struct.Struct("!QI32s")
_MAX_JSON_BYTES = 1024 * 1024
_MAX_FRAME_BYTES = 16 * 1024 * 1024


class ProtocolViolation(RuntimeError):
    """Raised when a persistent-channel message violates session identity."""


class ChannelState(str, Enum):
    CREATING = "CREATING"
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    DESTROYED = "DESTROYED"


class SessionState(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    READY = "READY"
    COMMITTED = "COMMITTED"
    CANCELLED = "CANCELLED"
    RESET = "RESET"


class PayloadType(str, Enum):
    START_SESSION = "START_SESSION"
    HISTORY = "HISTORY"
    DELTA = "DELTA"
    ACK = "ACK"
    RANK_READY = "RANK_READY"
    END_SESSION = "END_SESSION"
    CANCEL_SESSION = "CANCEL_SESSION"


def _required_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolViolation(f"{field_name} must be an integer")
    if value < minimum:
        raise ProtocolViolation(f"{field_name} must be >= {minimum}")
    return value


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolViolation(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class SessionEnvelope:
    """Identity carried by every persistent-channel message and ACK."""

    channel_generation: int
    migration_id: str
    request_id: str
    sequence_number: int
    payload_type: PayloadType
    token_start: int
    token_end: int
    rank: int
    protocol_version: int = PERSISTENT_CHANNEL_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _required_int(self.protocol_version, "protocol_version", minimum=1)
        _required_int(self.channel_generation, "channel_generation")
        _required_string(self.migration_id, "migration_id")
        _required_string(self.request_id, "request_id")
        _required_int(self.sequence_number, "sequence_number")
        _required_int(self.token_start, "token_start")
        _required_int(self.token_end, "token_end")
        _required_int(self.rank, "rank")
        if self.token_end < self.token_start:
            raise ProtocolViolation("token_end must be >= token_start")
        if not isinstance(self.payload_type, PayloadType):
            raise ProtocolViolation("payload_type must be a PayloadType")

    def to_wire(self) -> dict[str, int | str]:
        return {
            "protocol_version": self.protocol_version,
            "channel_generation": self.channel_generation,
            "migration_id": self.migration_id,
            "request_id": self.request_id,
            "sequence_number": self.sequence_number,
            "payload_type": self.payload_type.value,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "rank": self.rank,
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "SessionEnvelope":
        try:
            payload_type = PayloadType(value["payload_type"])
            return cls(
                protocol_version=_required_int(
                    value["protocol_version"],
                    "protocol_version",
                    minimum=1,
                ),
                channel_generation=_required_int(
                    value["channel_generation"], "channel_generation"
                ),
                migration_id=_required_string(
                    value["migration_id"], "migration_id"
                ),
                request_id=_required_string(value["request_id"], "request_id"),
                sequence_number=_required_int(
                    value["sequence_number"], "sequence_number"
                ),
                payload_type=payload_type,
                token_start=_required_int(value["token_start"], "token_start"),
                token_end=_required_int(value["token_end"], "token_end"),
                rank=_required_int(value["rank"], "rank"),
            )
        except KeyError as error:
            raise ProtocolViolation(
                f"persistent-channel message is missing {error.args[0]}"
            ) from error
        except ValueError as error:
            raise ProtocolViolation(
                f"unknown payload_type {value.get('payload_type')!r}"
            ) from error


@dataclass
class MigrationSession:
    """Request-local state hosted by a long-lived persistent channel."""

    channel_generation: int
    migration_id: str
    request_id: str
    expected_ranks: frozenset[int]
    state: SessionState = SessionState.NEW
    next_inbound_sequences: dict[int, int] = field(default_factory=dict)
    history_watermarks: dict[int, int] = field(default_factory=dict)
    delta_watermarks: dict[int, int] = field(default_factory=dict)
    ready_ranks: set[int] = field(default_factory=set)
    pending_ack_sequences: set[tuple[int, int]] = field(default_factory=set)
    _last_token_end: dict[tuple[int, PayloadType], int] = field(
        default_factory=dict
    )

    def start(self) -> None:
        if self.state is not SessionState.NEW:
            raise ProtocolViolation(f"cannot start session in {self.state.value}")
        self.next_inbound_sequences = {
            rank: 0 for rank in self.expected_ranks
        }
        self.history_watermarks = {rank: 0 for rank in self.expected_ranks}
        self.delta_watermarks = {rank: 0 for rank in self.expected_ranks}
        self.state = SessionState.ACTIVE

    def validate_identity(self, envelope: SessionEnvelope) -> None:
        if envelope.protocol_version != PERSISTENT_CHANNEL_PROTOCOL_VERSION:
            raise ProtocolViolation("persistent-channel protocol version differs")
        if envelope.channel_generation != self.channel_generation:
            raise ProtocolViolation("stale channel generation")
        if envelope.migration_id != self.migration_id:
            raise ProtocolViolation("stale migration_id")
        if envelope.request_id != self.request_id:
            raise ProtocolViolation("request_id differs")
        if envelope.rank not in self.expected_ranks:
            raise ProtocolViolation(f"unexpected rank {envelope.rank}")

    def accept_inbound(self, envelope: SessionEnvelope) -> None:
        if self.state not in {SessionState.ACTIVE, SessionState.READY}:
            raise ProtocolViolation(
                f"cannot accept a message in {self.state.value}"
            )
        self.validate_identity(envelope)
        expected_sequence = self.next_inbound_sequences[envelope.rank]
        if envelope.sequence_number < expected_sequence:
            raise ProtocolViolation("duplicate or replayed sequence_number")
        if envelope.sequence_number > expected_sequence:
            raise ProtocolViolation("out-of-order sequence_number")

        if envelope.payload_type in {PayloadType.HISTORY, PayloadType.DELTA}:
            range_key = (envelope.rank, envelope.payload_type)
            previous_end = self._last_token_end.get(range_key, 0)
            if envelope.token_start < previous_end:
                raise ProtocolViolation("overlapping or replayed token range")
            self._last_token_end[range_key] = envelope.token_end
            if envelope.payload_type is PayloadType.HISTORY:
                self.history_watermarks[envelope.rank] = max(
                    self.history_watermarks[envelope.rank], envelope.token_end
                )
            else:
                self.delta_watermarks[envelope.rank] = max(
                    self.delta_watermarks[envelope.rank], envelope.token_end
                )

        self.next_inbound_sequences[envelope.rank] += 1

    def expect_ack(self, *, rank: int, sequence_number: int) -> None:
        rank = _required_int(rank, "rank")
        if rank not in self.expected_ranks:
            raise ProtocolViolation(f"unexpected rank {rank}")
        self.pending_ack_sequences.add(
            (rank, _required_int(sequence_number, "sequence_number"))
        )

    def acknowledge(self, *, rank: int, sequence_number: int) -> None:
        rank = _required_int(rank, "rank")
        sequence_number = _required_int(sequence_number, "sequence_number")
        key = (rank, sequence_number)
        if key not in self.pending_ack_sequences:
            raise ProtocolViolation("stale or duplicate ACK")
        self.pending_ack_sequences.remove(key)

    def mark_rank_ready(self, rank: int) -> None:
        rank = _required_int(rank, "rank")
        if rank not in self.expected_ranks:
            raise ProtocolViolation(f"unexpected rank {rank}")
        if rank in self.ready_ranks:
            raise ProtocolViolation(f"duplicate ready notification from rank {rank}")
        self.ready_ranks.add(rank)
        if self.ready_ranks == set(self.expected_ranks):
            self.state = SessionState.READY

    def commit(self) -> None:
        if self.state is not SessionState.READY:
            raise ProtocolViolation("session must be READY before commit")
        if self.pending_ack_sequences:
            raise ProtocolViolation("session has pending ACKs")
        self.state = SessionState.COMMITTED

    def cancel(self) -> None:
        if self.state not in {
            SessionState.NEW,
            SessionState.ACTIVE,
            SessionState.READY,
        }:
            raise ProtocolViolation(f"cannot cancel session in {self.state.value}")
        self.state = SessionState.CANCELLED

    def reset(self) -> None:
        if self.state not in {
            SessionState.COMMITTED,
            SessionState.CANCELLED,
        }:
            raise ProtocolViolation("only a terminal session can be reset")
        self.next_inbound_sequences.clear()
        self.history_watermarks.clear()
        self.delta_watermarks.clear()
        self.ready_ranks.clear()
        self.pending_ack_sequences.clear()
        self._last_token_end.clear()
        self.state = SessionState.RESET


class PersistentChannelLifecycle:
    """Thread-safe lifecycle for one reusable fixed-topology channel."""

    def __init__(
        self,
        *,
        topology_key: str,
        expected_ranks: frozenset[int],
        channel_generation: int = 0,
    ) -> None:
        if not topology_key:
            raise ValueError("topology_key cannot be empty")
        if not expected_ranks:
            raise ValueError("expected_ranks cannot be empty")
        self.topology_key = topology_key
        self.expected_ranks = expected_ranks
        self.channel_generation = _required_int(
            channel_generation, "channel_generation"
        )
        self.state = ChannelState.CREATING
        self.active_session: MigrationSession | None = None
        self.create_count = 0
        self.destroy_count = 0
        self.rebuild_count = 0
        self.session_count = 0
        self._lock = threading.RLock()

    def mark_open(self) -> None:
        with self._lock:
            if self.state is not ChannelState.CREATING:
                raise ProtocolViolation(
                    f"cannot open channel in {self.state.value}"
                )
            self.create_count += 1
            self.state = ChannelState.IDLE

    def start_session(
        self, *, migration_id: str, request_id: str
    ) -> MigrationSession:
        with self._lock:
            if self.state is not ChannelState.IDLE:
                raise ProtocolViolation(
                    f"channel is not idle: {self.state.value}"
                )
            session = MigrationSession(
                channel_generation=self.channel_generation,
                migration_id=_required_string(migration_id, "migration_id"),
                request_id=_required_string(request_id, "request_id"),
                expected_ranks=self.expected_ranks,
            )
            session.start()
            self.active_session = session
            self.session_count += 1
            self.state = ChannelState.ACTIVE
            return session

    def finish_session(self) -> None:
        with self._lock:
            if self.state is not ChannelState.ACTIVE:
                raise ProtocolViolation(
                    f"cannot finish session in {self.state.value}"
                )
            if self.active_session is None:
                raise ProtocolViolation("channel has no active session")
            if self.active_session.state not in {
                SessionState.COMMITTED,
                SessionState.CANCELLED,
            }:
                raise ProtocolViolation("active session is not terminal")
            self.state = ChannelState.DRAINING
            self.active_session.reset()
            self.active_session = None
            self.state = ChannelState.IDLE

    def begin_rebuild(self, *, topology_key: str) -> None:
        with self._lock:
            if self.state is not ChannelState.IDLE:
                raise ProtocolViolation("channel must be IDLE before rebuild")
            if not topology_key:
                raise ValueError("topology_key cannot be empty")
            self.state = ChannelState.DRAINING
            self.destroy_count += 1
            self.rebuild_count += 1
            self.channel_generation += 1
            self.topology_key = topology_key
            self.state = ChannelState.CREATING

    def shutdown(self) -> None:
        with self._lock:
            if self.state is ChannelState.DESTROYED:
                return
            if self.state not in {ChannelState.IDLE, ChannelState.CREATING}:
                raise ProtocolViolation(
                    f"cannot shut down channel in {self.state.value}"
                )
            self.state = ChannelState.SHUTTING_DOWN
            self.destroy_count += 1
            self.state = ChannelState.DESTROYED


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError(
                f"TCP stream ended with {remaining} bytes still expected"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_json(connection: socket.socket, value: Mapping[str, Any]) -> None:
    data = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    if len(data) > _MAX_JSON_BYTES:
        raise ValueError("BridgeTP JSON frame exceeds 1 MiB")
    connection.sendall(_JSON_LENGTH.pack(len(data)))
    connection.sendall(data)


def recv_json(connection: socket.socket) -> dict[str, Any]:
    (length,) = _JSON_LENGTH.unpack(_recv_exact(connection, _JSON_LENGTH.size))
    if length > _MAX_JSON_BYTES:
        raise ValueError(f"BridgeTP JSON frame is too large: {length}")
    value = json.loads(_recv_exact(connection, length))
    if not isinstance(value, dict):
        raise TypeError("BridgeTP JSON frame must contain an object")
    return value


def serialize_rank_payload(payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return buffer.getvalue()


def deserialize_rank_payload(data: bytes) -> dict[str, Any]:
    value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("BridgeTP rank payload must be a dictionary")
    return value


def send_payload_frames(
    connection: socket.socket,
    payload: bytes,
    *,
    chunk_bytes: int,
    rate_bytes_per_second: float = 0.0,
    rate_provider: Callable[[], float] | None = None,
) -> dict[str, int | float | str]:
    """Send an ordered, independently hashed sequence of payload frames.

    ``rate_provider`` is evaluated once per frame and takes precedence over
    ``rate_bytes_per_second``. This lets the Phase 9 controller adjust an
    in-flight transfer without rebuilding the sender. A returned value of zero
    preserves the existing unlimited-rate convention.
    """
    if chunk_bytes <= 0 or chunk_bytes > _MAX_FRAME_BYTES:
        raise ValueError("chunk_bytes must be in [1, 16 MiB]")
    if rate_bytes_per_second < 0:
        raise ValueError("rate_bytes_per_second cannot be negative")

    started = time.perf_counter()
    last_frame_completed = started
    frame_count = 0
    for sequence, offset in enumerate(range(0, len(payload), chunk_bytes)):
        chunk = payload[offset : offset + chunk_bytes]
        digest = hashlib.sha256(chunk).digest()
        connection.sendall(_FRAME_HEADER.pack(sequence, len(chunk), digest))
        connection.sendall(chunk)
        frame_count += 1
        current_rate = (
            float(rate_provider())
            if rate_provider is not None
            else rate_bytes_per_second
        )
        if current_rate < 0:
            raise ValueError("dynamic rate_bytes_per_second cannot be negative")
        if rate_provider is not None and current_rate:
            target_frame_seconds = len(chunk) / current_rate
            frame_elapsed = time.perf_counter() - last_frame_completed
            delay = target_frame_seconds - frame_elapsed
            if delay > 0:
                time.sleep(delay)
            last_frame_completed = time.perf_counter()
        elif current_rate:
            target_elapsed = (offset + len(chunk)) / rate_bytes_per_second
            delay = target_elapsed - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
    elapsed = time.perf_counter() - started
    return {
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "num_frames": frame_count,
        "send_seconds": elapsed,
        "observed_gib_s": (len(payload) / 1024**3 / elapsed) if elapsed else 0.0,
    }


def recv_payload_frames(
    connection: socket.socket,
    *,
    payload_bytes: int,
    num_frames: int,
    payload_sha256: str,
    max_frame_bytes: int = _MAX_FRAME_BYTES,
) -> tuple[bytes, dict[str, int | float | str]]:
    """Receive frames, rejecting gaps, reordering, and digest mismatches."""
    if payload_bytes < 0 or num_frames < 0:
        raise ValueError("Negative payload metadata is invalid")
    if max_frame_bytes <= 0 or max_frame_bytes > _MAX_FRAME_BYTES:
        raise ValueError("max_frame_bytes must be in [1, 16 MiB]")

    started = time.perf_counter()
    chunks: list[bytes] = []
    received = 0
    for expected_sequence in range(num_frames):
        sequence, length, expected_digest = _FRAME_HEADER.unpack(
            _recv_exact(connection, _FRAME_HEADER.size)
        )
        if sequence != expected_sequence:
            raise ValueError(
                f"BridgeTP frame sequence differs: {sequence} != "
                f"{expected_sequence}"
            )
        if length > max_frame_bytes:
            raise ValueError(f"BridgeTP frame is too large: {length}")
        if received + length > payload_bytes:
            raise ValueError("BridgeTP frames exceed declared payload length")
        chunk = _recv_exact(connection, length)
        if hashlib.sha256(chunk).digest() != expected_digest:
            raise ValueError(f"BridgeTP frame {sequence} SHA256 mismatch")
        chunks.append(chunk)
        received += length

    if received != payload_bytes:
        raise ValueError(
            f"BridgeTP payload length differs: {received} != {payload_bytes}"
        )
    payload = b"".join(chunks)
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != payload_sha256:
        raise ValueError(
            "BridgeTP full payload SHA256 mismatch: "
            f"{actual_sha256} != {payload_sha256}"
        )
    elapsed = time.perf_counter() - started
    return payload, {
        "payload_bytes": received,
        "payload_sha256": actual_sha256,
        "num_frames": num_frames,
        "receive_seconds": elapsed,
        "observed_gib_s": (received / 1024**3 / elapsed) if elapsed else 0.0,
    }
