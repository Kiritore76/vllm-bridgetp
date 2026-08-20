# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from vllm.bridge_tp.kv_reshard import validate_exact_roundtrip


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate BridgeTP Phase 4 offline KV-head shards."
    )
    parser.add_argument("source_dump_dir", type=Path)
    parser.add_argument("reshard_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dump_dir.resolve()
    reshard_dir = args.reshard_dir.resolve()
    manifest = _load_json(reshard_dir / "reshard_manifest.json")
    source_payload = torch.load(
        source_dir / "kv_blocks.pt", map_location="cpu", weights_only=True
    )
    source_layers = source_payload["layers"]

    if manifest["reshard"]["target_tp_size"] != 4:
        raise ValueError("Phase 4 manifest target TP size must be 4")

    if (
        _sha256(source_dir / "kv_blocks.pt")
        != manifest["source_dump"]["kv_blocks_sha256"]
    ):
        raise ValueError("Source kv_blocks.pt SHA256 does not match manifest")

    rank_layers = []
    total_shard_bytes = 0
    for record in manifest["reshard"]["rank_files"]:
        rank_path = (reshard_dir / record["relative_path"]).resolve()
        if not rank_path.is_relative_to(reshard_dir):
            raise ValueError(f"Shard path escapes reshard directory: {rank_path}")
        if _sha256(rank_path) != record["sha256"]:
            raise ValueError(f"Shard SHA256 mismatch: {rank_path}")
        payload = torch.load(rank_path, map_location="cpu", weights_only=True)
        if payload["target_tp_size"] != 4:
            raise ValueError(f"Shard target TP size mismatch: {rank_path}")
        if payload["target_tp_rank"] != record["target_tp_rank"]:
            raise ValueError(f"Shard rank mismatch: {rank_path}")
        if payload["request_id"] != manifest["request_id"]:
            raise ValueError(f"Shard request ID mismatch: {rank_path}")
        rank_layers.append(payload["layers"])
        total_shard_bytes += sum(
            tensor.numel() * tensor.element_size()
            for tensor in payload["layers"].values()
        )

    expected_ranks = list(range(4))
    observed_ranks = [
        record["target_tp_rank"] for record in manifest["reshard"]["rank_files"]
    ]
    if observed_ranks != expected_ranks:
        raise ValueError(
            f"Shard ranks are incomplete or out of order: {observed_ranks}"
        )

    validation = validate_exact_roundtrip(
        source_layers,
        rank_layers,
        head_axis=manifest["reshard"]["head_axis"],
    )
    source_bytes = manifest["source_dump"]["raw_tensor_bytes"]
    if total_shard_bytes != source_bytes:
        raise ValueError(
            f"Raw shard bytes differ from source: {total_shard_bytes} != {source_bytes}"
        )

    summary = {
        "status": "PASS",
        "request_id": manifest["request_id"],
        "target_tp_size": manifest["reshard"]["target_tp_size"],
        "head_axis": manifest["reshard"]["head_axis"],
        "source_kv_heads": manifest["reshard"]["source_kv_heads"],
        "kv_heads_per_rank": manifest["reshard"]["kv_heads_per_rank"],
        "rank_tensor_shape": manifest["reshard"]["rank_tensor_shape"],
        "raw_source_bytes": source_bytes,
        "raw_shard_bytes": total_shard_bytes,
        **validation,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
