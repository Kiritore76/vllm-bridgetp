# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Framed TCP protocol used by the BridgeTP Phase 6 live transfer."""

from __future__ import annotations

import hashlib
import io
import json
import socket
import struct
import time
from collections.abc import Mapping
from typing import Any

import torch

PROTOCOL_VERSION = 1
MIGRATION_PARAM = "bridgetp_migration_id"
_JSON_LENGTH = struct.Struct("!I")
_FRAME_HEADER = struct.Struct("!QI32s")
_MAX_JSON_BYTES = 1024 * 1024
_MAX_FRAME_BYTES = 16 * 1024 * 1024


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
) -> dict[str, int | float | str]:
    """Send an ordered, independently hashed sequence of payload frames."""
    if chunk_bytes <= 0 or chunk_bytes > _MAX_FRAME_BYTES:
        raise ValueError("chunk_bytes must be in [1, 16 MiB]")
    if rate_bytes_per_second < 0:
        raise ValueError("rate_bytes_per_second cannot be negative")

    started = time.perf_counter()
    frame_count = 0
    for sequence, offset in enumerate(range(0, len(payload), chunk_bytes)):
        chunk = payload[offset : offset + chunk_bytes]
        digest = hashlib.sha256(chunk).digest()
        connection.sendall(_FRAME_HEADER.pack(sequence, len(chunk), digest))
        connection.sendall(chunk)
        frame_count += 1
        if rate_bytes_per_second:
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
