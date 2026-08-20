# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from vllm.bridge_tp.kv_reshard import (
    iter_tp_rank_shards,
    validate_exact_roundtrip,
    validate_tp1_layers,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _atomic_json_dump(data: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_path, path)


def _prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _validate_source_manifest(
    manifest: dict[str, Any], payload: dict[str, Any]
) -> dict[str, torch.Tensor]:
    if manifest["tp_world_size"] != 1 or manifest["tp_rank"] != 0:
        raise ValueError("Phase 4 source must be the rank-0 dump from TP1")
    if payload["request_id"] != manifest["request_id"]:
        raise ValueError("Source tensor and manifest request IDs differ")
    if payload["physical_block_ids"] != manifest["physical_block_ids"]:
        raise ValueError("Source tensor and manifest block IDs differ")

    layers = payload["layers"]
    manifest_layers = manifest["layers"]
    if len(layers) != manifest["num_layers"]:
        raise ValueError("Source tensor layer count differs from manifest")
    if len(manifest_layers) != manifest["num_layers"]:
        raise ValueError("Manifest layer records are incomplete")

    expected_names = {record["layer_name"] for record in manifest_layers}
    if set(layers) != expected_names:
        raise ValueError("Source tensor layer names differ from manifest")
    for record in manifest_layers:
        tensor = layers[record["layer_name"]]
        if list(tensor.shape) != record["dump_shape"]:
            raise ValueError(
                f"Layer {record['layer_name']} shape differs from manifest"
            )
        if str(tensor.dtype) != record["dtype"]:
            raise ValueError(
                f"Layer {record['layer_name']} dtype differs from manifest"
            )
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline BridgeTP TP1-to-TPN KV-head resharding."
    )
    parser.add_argument(
        "source_dump_dir",
        type=Path,
        help="Phase 1-3 directory containing kv_blocks.pt and manifest.json.",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-tp-size", type=int, required=True)
    parser.add_argument("--head-axis", type=int, required=True)
    parser.add_argument("--expected-source-kv-heads", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_tp_size != 4:
        raise ValueError("BridgeTP D3 Phase 4 requires --target-tp-size 4")
    source_dir = args.source_dump_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_manifest_path = source_dir / "manifest.json"
    source_tokens_path = source_dir / "generated_tokens.json"
    source_tensor_path = source_dir / "kv_blocks.pt"

    source_manifest = _load_json(source_manifest_path)
    source_payload = torch.load(
        source_tensor_path, map_location="cpu", weights_only=True
    )
    source_layers = _validate_source_manifest(source_manifest, source_payload)

    normalized_axis, heads_per_rank = validate_tp1_layers(
        source_layers,
        head_axis=args.head_axis,
        target_tp_size=args.target_tp_size,
        expected_source_kv_heads=args.expected_source_kv_heads,
    )
    _prepare_output_dir(output_dir)
    shutil.copy2(source_tokens_path, output_dir / "generated_tokens.json")

    started = time.perf_counter()
    rank_file_records: list[dict[str, Any]] = []
    rank_layers_for_validation: list[dict[str, torch.Tensor]] = []
    for rank, rank_layers in iter_tp_rank_shards(
        source_layers,
        head_axis=normalized_axis,
        target_tp_size=args.target_tp_size,
        expected_source_kv_heads=args.expected_source_kv_heads,
    ):
        rank_dir = output_dir / f"tp_rank_{rank}"
        rank_dir.mkdir()
        rank_path = rank_dir / "kv_shard.pt"
        temporary_path = rank_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "format_version": 1,
                "phase": "BridgeTP D3 Phase 4",
                "scope": "offline KV-head reshard; no restore or takeover",
                "request_id": source_manifest["request_id"],
                "source_tp_size": 1,
                "target_tp_size": args.target_tp_size,
                "target_tp_rank": rank,
                "head_axis": normalized_axis,
                "source_physical_block_ids": source_manifest["physical_block_ids"],
                "layers": rank_layers,
            },
            temporary_path,
        )
        os.replace(temporary_path, rank_path)
        rank_layers_for_validation.append(rank_layers)
        rank_file_records.append(
            {
                "target_tp_rank": rank,
                "relative_path": rank_path.relative_to(output_dir).as_posix(),
                "file_bytes": rank_path.stat().st_size,
                "sha256": _sha256(rank_path),
            }
        )

    validation = validate_exact_roundtrip(
        source_layers,
        rank_layers_for_validation,
        head_axis=normalized_axis,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    first_source = next(iter(source_layers.values()))
    first_rank = next(iter(rank_layers_for_validation[0].values()))
    manifest = {
        "format_version": 1,
        "phase": "BridgeTP D3 Phase 4",
        "scope": "offline TP1-to-TP4 KV-head reshard; no restore or takeover",
        "request_id": source_manifest["request_id"],
        "source_dump": {
            "manifest_sha256": _sha256(source_manifest_path),
            "generated_tokens_sha256": _sha256(source_tokens_path),
            "kv_blocks_sha256": _sha256(source_tensor_path),
            "tp_size": 1,
            "physical_block_ids": source_manifest["physical_block_ids"],
            "num_computed_tokens": source_manifest["num_computed_tokens"],
            "pending_known_tokens": source_manifest["pending_known_tokens"],
            "tensor_shape": list(first_source.shape),
            "tensor_dtype": str(first_source.dtype),
            "raw_tensor_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in source_layers.values()
            ),
        },
        "reshard": {
            "target_tp_size": args.target_tp_size,
            "head_axis": normalized_axis,
            "source_kv_heads": args.expected_source_kv_heads,
            "kv_heads_per_rank": heads_per_rank,
            "rank_tensor_shape": list(first_rank.shape),
            "rank_files": rank_file_records,
            "elapsed_ms": elapsed_ms,
        },
        "validation": validation,
    }
    _atomic_json_dump(manifest, output_dir / "reshard_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
