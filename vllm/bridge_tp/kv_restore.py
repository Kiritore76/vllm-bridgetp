# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strict artifact loading and KV-block injection for BridgeTP Phase 5."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

RESTORE_PARAM = "bridgetp_restore_request_id"


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


@dataclass(frozen=True)
class RestoreArtifact:
    """Validated metadata for one TP1-to-TP4 reshard artifact."""

    root: Path
    manifest: dict[str, Any]
    tokens: dict[str, Any]

    @property
    def source_request_id(self) -> str:
        return str(self.manifest["request_id"])

    @property
    def computed_token_ids(self) -> list[int]:
        return list(self.tokens["computed_token_ids"])

    @property
    def pending_token_ids(self) -> list[int]:
        return list(self.tokens["known_not_computed_token_ids"])

    @property
    def all_known_token_ids(self) -> list[int]:
        return self.computed_token_ids + self.pending_token_ids

    @property
    def num_computed_tokens(self) -> int:
        return int(self.manifest["source_dump"]["num_computed_tokens"])

    @property
    def block_axis(self) -> int:
        return int(self.manifest["source_dump"]["block_axis"])

    @property
    def block_size(self) -> int:
        return int(self.manifest["source_dump"]["block_size"])

    @property
    def num_blocks(self) -> int:
        return len(self.manifest["source_dump"]["physical_block_ids"])

    def rank_record(self, rank: int) -> dict[str, Any]:
        for record in self.manifest["reshard"]["rank_files"]:
            if int(record["target_tp_rank"]) == rank:
                return record
        raise KeyError(f"Phase 5 artifact has no shard for TP rank {rank}")


def load_restore_artifact(root: Path) -> RestoreArtifact:
    """Load and cross-check one Phase 4 artifact without loading tensors."""
    resolved_root = root.resolve()
    manifest_path = resolved_root / "reshard_manifest.json"
    tokens_path = resolved_root / "generated_tokens.json"
    manifest = _load_json(manifest_path)
    tokens = _load_json(tokens_path)

    if manifest.get("phase") != "BridgeTP D3 Phase 4":
        raise ValueError("Restore input is not a BridgeTP D3 Phase 4 artifact")
    if manifest["reshard"]["target_tp_size"] != 4:
        raise ValueError("Phase 5 requires a TP4 reshard artifact")
    if manifest["source_dump"]["tp_size"] != 1:
        raise ValueError("Phase 5 source topology must be TP1")
    if manifest["request_id"] != tokens["request_id"]:
        raise ValueError("Reshard manifest and token history request IDs differ")
    if sha256_file(tokens_path) != manifest["source_dump"]["generated_tokens_sha256"]:
        raise ValueError("generated_tokens.json SHA256 does not match manifest")

    computed = list(tokens["computed_token_ids"])
    pending = list(tokens["known_not_computed_token_ids"])
    known = list(tokens["prompt_token_ids"]) + list(tokens["output_token_ids"])
    if computed + pending != known:
        raise ValueError(
            "Computed and pending token histories do not reconstruct history"
        )
    if len(computed) != manifest["source_dump"]["num_computed_tokens"]:
        raise ValueError("Computed token count differs from reshard manifest")
    if len(pending) != manifest["source_dump"]["pending_known_tokens"]:
        raise ValueError("Pending token count differs from reshard manifest")
    if len(pending) != 1:
        raise ValueError("Phase 5 MVP requires exactly one pending token")

    block_size = int(manifest["source_dump"]["block_size"])
    num_blocks = len(manifest["source_dump"]["physical_block_ids"])
    required_blocks = (len(computed) + block_size - 1) // block_size
    if num_blocks != required_blocks:
        raise ValueError(
            f"Source block count differs from computed-token boundary: "
            f"{num_blocks} != {required_blocks}"
        )

    expected_ranks = list(range(4))
    observed_ranks = [
        int(record["target_tp_rank"]) for record in manifest["reshard"]["rank_files"]
    ]
    if observed_ranks != expected_ranks:
        raise ValueError(f"Phase 4 rank records are incomplete: {observed_ranks}")
    return RestoreArtifact(resolved_root, manifest, tokens)


def load_rank_shard(artifact: RestoreArtifact, rank: int) -> dict[str, torch.Tensor]:
    """Load and authenticate the CPU tensor shard for one TP4 rank."""
    record = artifact.rank_record(rank)
    shard_path = (artifact.root / record["relative_path"]).resolve()
    if not shard_path.is_relative_to(artifact.root):
        raise ValueError(f"Shard path escapes artifact directory: {shard_path}")
    if sha256_file(shard_path) != record["sha256"]:
        raise ValueError(f"Shard SHA256 mismatch: {shard_path}")

    payload = torch.load(shard_path, map_location="cpu", weights_only=True)
    expected = {
        "request_id": artifact.source_request_id,
        "target_tp_size": 4,
        "target_tp_rank": rank,
        "block_axis": artifact.block_axis,
        "block_size": artifact.block_size,
        "num_computed_tokens": artifact.num_computed_tokens,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Rank {rank} shard field {key} differs: "
                f"{payload.get(key)!r} != {value!r}"
            )
    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"Rank {rank} shard contains no layer tensors")
    return layers


def inject_rank_shard(
    destination_layers: Mapping[str, torch.Tensor],
    shard_layers: Mapping[str, torch.Tensor],
    target_block_ids: list[int],
    *,
    block_axis: int,
) -> dict[str, int | bool]:
    """Synchronously write one rank shard and require an exact readback."""
    if not target_block_ids:
        raise ValueError("No destination blocks were allocated")
    if len(set(target_block_ids)) != len(target_block_ids):
        raise ValueError("Destination block IDs are not unique")
    if min(target_block_ids) < 0:
        raise ValueError("Destination block IDs must be nonnegative")
    if set(destination_layers) != set(shard_layers):
        missing = sorted(set(shard_layers) - set(destination_layers))
        extra = sorted(set(destination_layers) - set(shard_layers))
        raise ValueError(
            f"Destination/shard layer names differ; missing={missing}, extra={extra}"
        )

    raw_tensor_bytes = 0
    for layer_name, source_cpu in shard_layers.items():
        destination = destination_layers[layer_name]
        normalized_axis = (
            block_axis if block_axis >= 0 else destination.ndim + block_axis
        )
        if normalized_axis < 0 or normalized_axis >= destination.ndim:
            raise ValueError(f"Invalid block axis {block_axis} for {layer_name}")
        if source_cpu.ndim != destination.ndim:
            raise ValueError(f"Layer {layer_name} tensor ranks differ")
        if source_cpu.shape[normalized_axis] != len(target_block_ids):
            raise ValueError(
                f"Layer {layer_name} shard block count differs from allocation"
            )
        if max(target_block_ids) >= destination.shape[normalized_axis]:
            raise IndexError(f"Layer {layer_name} destination block is out of range")
        for axis, (source_size, destination_size) in enumerate(
            zip(source_cpu.shape, destination.shape)
        ):
            if axis != normalized_axis and source_size != destination_size:
                raise ValueError(
                    f"Layer {layer_name} shape mismatch on axis {axis}: "
                    f"{source_size} != {destination_size}"
                )
        if source_cpu.dtype != destination.dtype:
            raise ValueError(
                f"Layer {layer_name} dtype mismatch: "
                f"{source_cpu.dtype} != {destination.dtype}"
            )

        index = torch.tensor(
            target_block_ids, dtype=torch.long, device=destination.device
        )
        source = source_cpu.to(device=destination.device)
        destination.index_copy_(normalized_axis, index, source)
        restored = destination.index_select(normalized_axis, index).cpu()
        if not torch.equal(restored, source_cpu):
            mismatches = int(torch.count_nonzero(restored != source_cpu))
            raise ValueError(
                f"Layer {layer_name} restore readback differs in {mismatches} elements"
            )
        raw_tensor_bytes += source_cpu.numel() * source_cpu.element_size()

    return {
        "exact_readback": True,
        "num_layers": len(shard_layers),
        "num_target_blocks": len(target_block_ids),
        "raw_tensor_bytes": raw_tensor_bytes,
    }
