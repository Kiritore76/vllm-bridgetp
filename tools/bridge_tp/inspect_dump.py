# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _validate_metadata(manifest: dict[str, Any], tokens: dict[str, Any]) -> None:
    computed = int(manifest["num_computed_tokens"])
    known = int(manifest["num_known_tokens"])
    pending = int(manifest["pending_known_tokens"])
    block_size = int(manifest["block_size"])
    block_ids = manifest["physical_block_ids"]

    assert manifest["tp_world_size"] == 1, "Phase 1-3 evidence must use TP1"
    assert known - computed == pending
    assert len(tokens["computed_token_ids"]) == computed
    assert len(tokens["known_not_computed_token_ids"]) == pending
    assert tokens["num_computed_tokens"] == computed
    assert tokens["num_known_tokens"] == known
    assert len(block_ids) == (computed + block_size - 1) // block_size
    assert manifest["num_physical_blocks"] == len(block_ids)
    assert manifest["num_layers"] == len(manifest["layers"])

    expected_last_valid = computed % block_size
    if computed and expected_last_valid == 0:
        expected_last_valid = block_size
    assert manifest["last_block_valid_tokens"] == expected_last_valid


def _validate_tensors(dump_dir: Path, manifest: dict[str, Any]) -> None:
    if not manifest["include_tensors"]:
        return

    import torch

    tensor_path = dump_dir / "kv_blocks.pt"
    assert tensor_path.is_file(), f"Missing tensor dump: {tensor_path}"
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == manifest["format_version"]
    assert payload["request_id"] == manifest["request_id"]
    assert payload["physical_block_ids"] == manifest["physical_block_ids"]

    tensors = payload["layers"]
    assert len(tensors) == manifest["num_layers"]
    num_blocks = manifest["num_physical_blocks"]
    block_axis = manifest["block_axis"]
    for layer in manifest["layers"]:
        tensor = tensors[layer["layer_name"]]
        assert list(tensor.shape) == layer["dump_shape"]
        assert tensor.shape[block_axis] == num_blocks
        assert str(tensor.dtype) == layer["dtype"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a BridgeTP D3 Phase 1-3 KV dump."
    )
    parser.add_argument(
        "dump_dir",
        type=Path,
        help="Directory containing manifest.json and tokens.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_dir = args.dump_dir.resolve()
    manifest = _load_json(dump_dir / "manifest.json")
    tokens = _load_json(dump_dir / "generated_tokens.json")

    _validate_metadata(manifest, tokens)
    _validate_tensors(dump_dir, manifest)

    summary = {
        "status": "PASS",
        "request_id": manifest["request_id"],
        "num_computed_tokens": manifest["num_computed_tokens"],
        "num_known_tokens": manifest["num_known_tokens"],
        "pending_known_tokens": manifest["pending_known_tokens"],
        "num_layers": manifest["num_layers"],
        "num_physical_blocks": manifest["num_physical_blocks"],
        "estimated_tensor_bytes": manifest["estimated_tensor_bytes"],
        "copy_ms": manifest["copy_ms"],
        "save_ms": manifest["save_ms"],
        "total_dump_ms": manifest["total_dump_ms"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
