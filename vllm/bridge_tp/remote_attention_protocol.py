# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contracts for BridgeTP split-KV remote-attention validation.

This module deliberately contains no CUDA or transport implementation.  It
defines the geometry, partition, wire-volume, and fail-closed acceptance
contracts shared by the GPU validation runner and the future online Bridge
implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

EvidenceClass = Literal[
    "GPU_SYNTHETIC_KV_TRANSFER",
    "GPU_CAPTURED_MODEL_TENSORS",
]
BridgeState = Literal[
    "LOCAL_TP1",
    "SHADOW",
    "BRIDGE",
    "TAKEOVER_TP4",
    "CANCELLED",
    "FAILED_BRIDGE",
]
BlockKind = Literal["HISTORY", "NEW_KV"]


@dataclass(frozen=True)
class BlockTransferRecord:
    """Auditable ownership record for one logical KV block.

    ``target_verified`` means every required target rank has validated the
    block identity and digest.  Releasing the TP1 copy before that point is a
    protocol violation.
    """

    request_id: str
    migration_epoch: str
    block_id: int
    token_start: int
    token_end: int
    kind: BlockKind
    state: BridgeState
    source_present: bool
    target_verified: bool
    source_released: bool
    queued_unix_s: float | None = None
    sent_unix_s: float | None = None
    verified_unix_s: float | None = None
    released_unix_s: float | None = None

    def validate(self) -> None:
        if not self.request_id or not self.migration_epoch:
            raise ValueError("request_id and migration_epoch are required")
        if self.block_id < 0:
            raise ValueError("block_id cannot be negative")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("invalid block token range")
        if self.state == "SHADOW" and self.source_released:
            raise ValueError("Shadow cannot release TP1 KV")
        if self.source_released and not self.target_verified:
            raise ValueError("TP1 release requires verified TP4 ownership")
        if self.source_released and self.source_present:
            raise ValueError("released TP1 block cannot remain source-present")
        times = [
            self.queued_unix_s,
            self.sent_unix_s,
            self.verified_unix_s,
            self.released_unix_s,
        ]
        observed = [value for value in times if value is not None]
        if observed != sorted(observed):
            raise ValueError("block timestamps are out of order")
        if self.target_verified and self.verified_unix_s is None:
            raise ValueError("verified ownership requires a verified timestamp")
        if self.source_released and self.released_unix_s is None:
            raise ValueError("released ownership requires a released timestamp")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def validate_block_records(
    records: list[BlockTransferRecord],
) -> dict[str, Any]:
    """Validate block identity and ownership invariants fail-closed."""
    errors: list[str] = []
    identities: set[tuple[str, str, int]] = set()
    for index, record in enumerate(records):
        try:
            record.validate()
        except ValueError as exc:
            errors.append(f"record[{index}]: {exc}")
        identity = (
            record.request_id,
            record.migration_epoch,
            record.block_id,
        )
        if identity in identities:
            errors.append(f"record[{index}]: duplicate block identity {identity}")
        identities.add(identity)
    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "records": len(records),
        "errors": errors,
    }


@dataclass(frozen=True)
class RemoteAttentionGeometry:
    """Model and TP geometry used by one validation run."""

    num_layers: int = 48
    num_query_heads: int = 40
    num_kv_heads: int = 8
    head_dim: int = 128
    target_tp_size: int = 4
    block_size: int = 16
    dtype_bytes: int = 2

    def validate(self) -> None:
        values = asdict(self)
        if any(int(value) <= 0 for value in values.values()):
            raise ValueError("remote-attention geometry values must be positive")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.num_query_heads % self.target_tp_size:
            raise ValueError("query heads must divide across target TP ranks")
        if self.num_kv_heads % self.target_tp_size:
            raise ValueError("KV heads must divide across target TP ranks")

    @property
    def query_heads_per_rank(self) -> int:
        self.validate()
        return self.num_query_heads // self.target_tp_size

    @property
    def kv_heads_per_rank(self) -> int:
        self.validate()
        return self.num_kv_heads // self.target_tp_size

    @property
    def query_heads_per_kv_head(self) -> int:
        self.validate()
        return self.num_query_heads // self.num_kv_heads

    @property
    def kv_bytes_per_token_per_layer(self) -> int:
        self.validate()
        return 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def aggregate_kv_bytes_per_token(self) -> int:
        return self.num_layers * self.kv_bytes_per_token_per_layer

    def to_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class KVPartition:
    """One contiguous local-prefix/remote-suffix KV partition."""

    context_tokens: int
    local_tokens: int
    remote_tokens: int
    block_size: int

    def validate(self) -> None:
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.local_tokens < 0 or self.remote_tokens < 0:
            raise ValueError("partition token counts cannot be negative")
        if self.local_tokens + self.remote_tokens != self.context_tokens:
            raise ValueError("local and remote tokens must cover the context")
        if self.remote_tokens and self.remote_tokens % self.block_size:
            raise ValueError("remote suffix must end on a KV block boundary")

    @property
    def remote_fraction(self) -> float:
        self.validate()
        return self.remote_tokens / self.context_tokens

    @property
    def ownership_boundary(self) -> int:
        """First token owned by TP4; TP1 owns ``[0, boundary)``."""
        self.validate()
        return self.local_tokens

    def to_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "remote_fraction": self.remote_fraction,
            "ownership_boundary": self.ownership_boundary,
        }


def make_partition(
    context_tokens: int,
    requested_remote_fraction: float,
    block_size: int,
) -> KVPartition:
    """Round a requested TP4 suffix to a complete-block partition."""
    if context_tokens <= 0 or block_size <= 0:
        raise ValueError("context_tokens and block_size must be positive")
    if not 0 <= requested_remote_fraction <= 1:
        raise ValueError("requested remote fraction must be in [0, 1]")
    requested = round(context_tokens * requested_remote_fraction)
    remote_tokens = requested // block_size * block_size
    if requested_remote_fraction > 0 and remote_tokens == 0:
        remote_tokens = min(block_size, context_tokens)
    remote_tokens = min(remote_tokens, context_tokens)
    partition = KVPartition(
        context_tokens=context_tokens,
        local_tokens=context_tokens - remote_tokens,
        remote_tokens=remote_tokens,
        block_size=block_size,
    )
    partition.validate()
    return partition


def history_transfer_bytes(
    partition: KVPartition,
    geometry: RemoteAttentionGeometry,
) -> int:
    """Return aggregate K+V bytes already owned by TP4."""
    partition.validate()
    geometry.validate()
    return partition.remote_tokens * geometry.aggregate_kv_bytes_per_token


def one_layer_rank_transfer_bytes(
    partition: KVPartition,
    geometry: RemoteAttentionGeometry,
) -> int:
    """Return K+V bytes staged to one TP4 rank for one layer."""
    partition.validate()
    geometry.validate()
    return (
        2
        * geometry.kv_heads_per_rank
        * partition.remote_tokens
        * geometry.head_dim
        * geometry.dtype_bytes
    )


def one_layer_query_wire_bytes(geometry: RemoteAttentionGeometry) -> int:
    """Return TP1-to-TP4 Q bytes across all TP4 ranks for one layer."""
    geometry.validate()
    return geometry.num_query_heads * geometry.head_dim * geometry.dtype_bytes


def one_layer_statistics_wire_bytes(
    geometry: RemoteAttentionGeometry,
    statistics_dtype_bytes: int = 4,
) -> int:
    """Return TP4-to-TP1 ``m``, ``l``, and ``o`` bytes for one layer."""
    geometry.validate()
    if statistics_dtype_bytes <= 0:
        raise ValueError("statistics_dtype_bytes must be positive")
    scalars_per_query_head = 2 + geometry.head_dim
    return geometry.num_query_heads * scalars_per_query_head * statistics_dtype_bytes


def projected_token_wire_bytes(
    geometry: RemoteAttentionGeometry,
    statistics_dtype_bytes: int = 4,
) -> int:
    """Return Q plus statistics traffic for all layers of one token."""
    per_layer = one_layer_query_wire_bytes(geometry)
    per_layer += one_layer_statistics_wire_bytes(geometry, statistics_dtype_bytes)
    return geometry.num_layers * per_layer


def validate_measurements(
    rows: list[dict[str, Any]],
    *,
    expected_cases: int,
    max_abs_tolerance: float,
    mean_abs_tolerance: float,
) -> dict[str, Any]:
    """Build a fail-closed acceptance record for measured GPU cases."""
    errors: list[str] = []
    if expected_cases <= 0:
        errors.append("expected_cases must be positive")
    if len(rows) != expected_cases:
        errors.append(f"recorded {len(rows)} cases, expected {expected_cases}")
    for index, row in enumerate(rows):
        prefix = f"case[{index}]"
        if row.get("status") != "PASS":
            errors.append(f"{prefix} status is not PASS")
        if not bool(row.get("finite", False)):
            errors.append(f"{prefix} contains non-finite output")
        max_abs = row.get("max_abs_error")
        mean_abs = row.get("mean_abs_error")
        if not isinstance(max_abs, (int, float)):
            errors.append(f"{prefix} has no max_abs_error")
        elif float(max_abs) > max_abs_tolerance:
            errors.append(
                f"{prefix} max_abs_error {max_abs} exceeds {max_abs_tolerance}"
            )
        if not isinstance(mean_abs, (int, float)):
            errors.append(f"{prefix} has no mean_abs_error")
        elif float(mean_abs) > mean_abs_tolerance:
            errors.append(
                f"{prefix} mean_abs_error {mean_abs} exceeds {mean_abs_tolerance}"
            )
        staged = row.get("staged_kv_bytes")
        expected_staged = row.get("expected_staged_kv_bytes")
        if staged != expected_staged:
            errors.append(f"{prefix} staged bytes {staged} != {expected_staged}")
        if float(row.get("step_p50_ms", 0)) <= 0:
            errors.append(f"{prefix} has no positive measured step latency")
    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "expected_cases": expected_cases,
        "recorded_cases": len(rows),
        "max_abs_tolerance": max_abs_tolerance,
        "mean_abs_tolerance": mean_abs_tolerance,
        "errors": errors,
    }
