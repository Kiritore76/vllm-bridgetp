# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline KV-head resharding utilities for BridgeTP D3 Phase 4."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch

LayerTensors = Mapping[str, torch.Tensor]


def normalize_axis(ndim: int, axis: int) -> int:
    """Return a nonnegative tensor axis or raise for an invalid axis."""
    normalized = axis if axis >= 0 else ndim + axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"Axis {axis} is invalid for a {ndim}-D tensor")
    return normalized


def validate_tp1_layers(
    layers: LayerTensors,
    *,
    head_axis: int,
    target_tp_size: int,
    expected_source_kv_heads: int,
) -> tuple[int, int]:
    """Validate the Phase 4 source layout.

    Args:
        layers: Request-scoped TP1 KV tensors keyed by layer name.
        head_axis: Axis containing the source KV heads.
        target_tp_size: Number of target tensor-parallel ranks.
        expected_source_kv_heads: Expected TP1 KV-head count.

    Returns:
        The normalized head axis and KV heads per target rank.
    """
    if not layers:
        raise ValueError("The TP1 dump contains no layer tensors")
    if target_tp_size <= 0:
        raise ValueError("target_tp_size must be positive")
    if expected_source_kv_heads <= 0:
        raise ValueError("expected_source_kv_heads must be positive")
    if expected_source_kv_heads % target_tp_size:
        raise ValueError(
            "Source KV heads must be divisible by target TP size: "
            f"{expected_source_kv_heads} % {target_tp_size} != 0"
        )

    first = next(iter(layers.values()))
    normalized_axis = normalize_axis(first.ndim, head_axis)
    reference_shape = tuple(first.shape)
    reference_dtype = first.dtype
    for layer_name, tensor in layers.items():
        if tensor.ndim != first.ndim:
            raise ValueError(
                f"Layer {layer_name} rank differs from the first layer: "
                f"{tensor.ndim} != {first.ndim}"
            )
        if tuple(tensor.shape) != reference_shape:
            raise ValueError(
                f"Layer {layer_name} shape differs from the first layer: "
                f"{tuple(tensor.shape)} != {reference_shape}"
            )
        if tensor.dtype != reference_dtype:
            raise ValueError(
                f"Layer {layer_name} dtype differs from the first layer: "
                f"{tensor.dtype} != {reference_dtype}"
            )
        if tensor.shape[normalized_axis] != expected_source_kv_heads:
            raise ValueError(
                f"Layer {layer_name} has {tensor.shape[normalized_axis]} heads "
                f"on axis {normalized_axis}; expected "
                f"{expected_source_kv_heads}"
            )

    return normalized_axis, expected_source_kv_heads // target_tp_size


def iter_tp_rank_shards(
    layers: LayerTensors,
    *,
    head_axis: int,
    target_tp_size: int,
    expected_source_kv_heads: int,
) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """Yield contiguous per-rank KV-head shards in TP-rank order."""
    normalized_axis, heads_per_rank = validate_tp1_layers(
        layers,
        head_axis=head_axis,
        target_tp_size=target_tp_size,
        expected_source_kv_heads=expected_source_kv_heads,
    )

    for rank in range(target_tp_size):
        start = rank * heads_per_rank
        rank_layers = {
            layer_name: tensor.narrow(
                normalized_axis, start, heads_per_rank
            ).contiguous()
            for layer_name, tensor in layers.items()
        }
        yield rank, rank_layers


def validate_exact_roundtrip(
    source_layers: LayerTensors,
    rank_layers: Sequence[LayerTensors],
    *,
    head_axis: int,
) -> dict[str, int | bool]:
    """Require exact reconstruction after concatenating rank shards."""
    if not rank_layers:
        raise ValueError("No rank shards were provided")

    reconstructed_elements = 0
    for layer_name, source in source_layers.items():
        normalized_axis = normalize_axis(source.ndim, head_axis)
        pieces: list[torch.Tensor] = []
        for rank, shard in enumerate(rank_layers):
            if layer_name not in shard:
                raise KeyError(f"Rank {rank} is missing layer {layer_name}")
            piece = shard[layer_name]
            if piece.dtype != source.dtype:
                raise ValueError(
                    f"Rank {rank} layer {layer_name} dtype mismatch: "
                    f"{piece.dtype} != {source.dtype}"
                )
            pieces.append(piece)

        reconstructed = torch.cat(pieces, dim=normalized_axis)
        if reconstructed.shape != source.shape:
            raise ValueError(
                f"Reconstructed layer {layer_name} has shape "
                f"{tuple(reconstructed.shape)}, expected {tuple(source.shape)}"
            )
        if not torch.equal(reconstructed, source):
            mismatches = int(torch.count_nonzero(reconstructed != source))
            raise ValueError(
                f"Reconstructed layer {layer_name} differs in {mismatches} elements"
            )
        reconstructed_elements += source.numel()

    source_names = set(source_layers)
    for rank, shard in enumerate(rank_layers):
        unexpected = set(shard) - source_names
        if unexpected:
            raise KeyError(
                f"Rank {rank} contains unexpected layers: {sorted(unexpected)}"
            )

    return {
        "exact_roundtrip": True,
        "num_layers": len(source_layers),
        "reconstructed_elements": reconstructed_elements,
    }
