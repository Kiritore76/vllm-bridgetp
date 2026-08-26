#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D-1: compare fixed-prefix TP1, native-TP4, and migrated-TP4 KV.

Every dump must contain the exact token IDs in ``--fixed-token-ids``. This
prevents native TP1 and TP4 from freely generating different histories before
the migration boundary K. The tool reports measurements only; it does not use
an uncalibrated threshold to declare migration blameless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp1-dump-dir", type=Path, required=True)
    parser.add_argument(
        "--native-tp4-dump-dirs",
        type=Path,
        nargs=4,
        required=True,
        metavar=("RANK0", "RANK1", "RANK2", "RANK3"),
    )
    parser.add_argument(
        "--migrated-tp4-dump-dirs",
        type=Path,
        nargs=4,
        default=None,
        metavar=("RANK0", "RANK1", "RANK2", "RANK3"),
    )
    parser.add_argument("--fixed-token-ids", type=Path, required=True)
    parser.add_argument("--boundary-k", type=int, required=True)
    parser.add_argument("--logical-request-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing evidence file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_fixed_tokens(path: Path, boundary_k: int) -> tuple[list[int], dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("fixed-token file must be a provenance object, not a list")
    metadata: dict[str, Any] = raw
    required = {
        "fixed_token_ids",
        "boundary_k",
        "prompt_token_count",
        "logical_request_id",
        "model",
    }
    missing = required - set(raw)
    if missing:
        raise SystemExit(f"fixed-token provenance is missing {sorted(missing)}")
    token_ids = raw.get("fixed_token_ids")
    recorded_k = int(raw["boundary_k"])
    if recorded_k != boundary_k:
        raise SystemExit(
            f"fixed-prefix K mismatch: file={recorded_k}, CLI={boundary_k}"
        )
    if not isinstance(token_ids, list) or not token_ids:
        raise SystemExit("fixed-token file must contain a nonempty token-id list")
    expected_count = int(raw["prompt_token_count"]) + boundary_k
    if len(token_ids) != expected_count:
        raise SystemExit(
            f"fixed-prefix length is {len(token_ids)}, expected "
            f"prompt+K={expected_count}"
        )
    return [int(value) for value in token_ids], metadata


@dataclass
class Dump:
    role: str
    rank: int
    directory: Path
    manifest: dict
    tokens: dict
    payload: dict


def load_dump(directory: Path, role: str, expected_rank: int, expected_tp: int) -> Dump:
    import torch

    manifest = load_json(directory / "manifest.json")
    tokens = load_json(directory / "generated_tokens.json")
    tensor_path = directory / "kv_blocks.pt"
    if not tensor_path.is_file():
        raise SystemExit(f"missing tensor dump: {tensor_path}")
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    for source_name, source in (("manifest", manifest), ("tensor", payload)):
        if int(source.get("tp_rank", -1)) != expected_rank:
            raise SystemExit(f"{directory}: {source_name} rank is not {expected_rank}")
        if int(source.get("tp_world_size", -1)) != expected_tp:
            raise SystemExit(f"{directory}: {source_name} TP size is not {expected_tp}")
    if payload.get("format_version") != 1:
        raise SystemExit(f"{tensor_path}: unsupported format")
    if not isinstance(payload.get("layers"), dict) or not payload["layers"]:
        raise SystemExit(f"{tensor_path}: no layer tensors")
    return Dump(role, expected_rank, directory, manifest, tokens, payload)


def validate_provenance(dumps: list[Dump], fixed_tokens: list[int]) -> dict:
    models = {str(dump.manifest.get("model")) for dump in dumps}
    dtypes = {str(dump.manifest.get("cache_dtype")) for dump in dumps}
    block_sizes = {int(dump.manifest.get("block_size", -1)) for dump in dumps}
    if len(models) != 1 or len(dtypes) != 1 or len(block_sizes) != 1:
        raise SystemExit(
            "model/dtype/block-size provenance mismatch: "
            f"models={models}, dtypes={dtypes}, block_sizes={block_sizes}"
        )
    request_ids: dict[str, set[str]] = {}
    expected_hash = hashlib.sha256(
        json.dumps(fixed_tokens, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for dump in dumps:
        observed = [int(value) for value in dump.tokens.get("computed_token_ids", [])]
        if observed != fixed_tokens:
            mismatch = next(
                (
                    index
                    for index, (left, right) in enumerate(zip(observed, fixed_tokens))
                    if left != right
                ),
                min(len(observed), len(fixed_tokens)),
            )
            raise SystemExit(
                f"{dump.directory}: fixed token IDs differ at index {mismatch}; "
                "do not compare freely generated histories"
            )
        if int(dump.manifest.get("num_computed_tokens", -1)) != len(fixed_tokens):
            raise SystemExit(f"{dump.directory}: computed-token count mismatch")
        ids = {
            str(dump.manifest.get("request_id")),
            str(dump.tokens.get("request_id")),
            str(dump.payload.get("request_id")),
        }
        if len(ids) != 1:
            raise SystemExit(f"{dump.directory}: request-id provenance mismatch {ids}")
        hashes = {
            str(dump.manifest.get("computed_token_ids_sha256")),
            str(dump.tokens.get("computed_token_ids_sha256")),
        }
        if hashes != {expected_hash}:
            raise SystemExit(f"{dump.directory}: computed-token hash mismatch {hashes}")
        request_ids.setdefault(dump.role, set()).add(
            str(dump.manifest.get("request_id"))
        )
    if any(len(values) != 1 for values in request_ids.values()):
        raise SystemExit(f"rank-local request IDs disagree: {request_ids}")
    return {
        "model": next(iter(models)),
        "cache_dtype": next(iter(dtypes)),
        "block_size": next(iter(block_sizes)),
        "num_computed_tokens": len(fixed_tokens),
        "request_ids_by_role": {
            role: next(iter(values)) for role, values in request_ids.items()
        },
    }


def logical_tokens(dump: Dump, tensor: Any) -> tuple[Any, int]:
    layout = dump.manifest.get("cache_layout") or dump.payload.get("cache_layout")
    if not isinstance(layout, dict):
        raise SystemExit(f"{dump.directory}: cache_layout is missing")
    block_axis = int(layout["block_axis"])
    token_axis = int(layout["token_axis"])
    head_axis = int(layout["head_axis"])
    permutation = [block_axis, token_axis]
    permutation.extend(
        axis for axis in range(tensor.ndim) if axis not in {block_axis, token_axis}
    )
    moved = tensor.permute(permutation).contiguous()
    flattened = moved.reshape(moved.shape[0] * moved.shape[1], *moved.shape[2:])
    count = int(dump.manifest["num_computed_tokens"])
    if flattened.shape[0] < count:
        raise SystemExit(f"{dump.directory}: tensor has too few logical token slots")
    # Collapsing block and in-block-token axes removes one leading dimension.
    logical_head_axis = permutation.index(head_axis) - 1
    return flattened[:count].contiguous(), logical_head_axis


def tensor_quantiles(values: Any) -> dict[str, float]:
    import torch

    flattened = values.reshape(-1).float()
    probabilities = torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float32)
    result = torch.quantile(flattened, probabilities).tolist()
    return dict(zip(("p50", "p90", "p99", "p999"), map(float, result)))


def ulp_distances(left: Any, right: Any, dtype: str) -> Any:
    import torch

    normalized = dtype.replace("torch.", "").lower()
    mantissa = {"bfloat16": 7, "float16": 10, "float32": 23}.get(normalized)
    if mantissa is None:
        raise SystemExit(f"unsupported ULP dtype: {dtype}")
    scale = torch.maximum(left.abs(), right.abs())
    exponent = torch.floor(torch.log2(scale.clamp_min(torch.finfo(torch.float32).tiny)))
    step = torch.pow(2.0, exponent - mantissa)
    return (left - right).abs() / step


def compare_tensors(left: Any, right: Any, label: str) -> dict:
    import torch

    if left.shape != right.shape:
        raise SystemExit(f"{label}: shape mismatch {left.shape} != {right.shape}")
    dtype = str(left.dtype).replace("torch.", "")
    if left.dtype != right.dtype:
        raise SystemExit(f"{label}: dtype mismatch {left.dtype} != {right.dtype}")
    left32 = left.float()
    right32 = right.float()
    difference = (left32 - right32).abs()
    scale = torch.maximum(left32.abs(), right32.abs())
    nonzero = scale[scale > 0]
    robust_floor = (
        float(torch.quantile(nonzero, 0.01).item()) if nonzero.numel() else 1.0
    )
    symmetric_relative = difference / scale.clamp_min(max(robust_floor, 1e-30))
    ulps = ulp_distances(left32, right32, dtype)
    elements = int(difference.numel())
    bins = {
        "exact": int((difference == 0).sum().item()),
        "le_1": int((ulps <= 1).sum().item()),
        "le_2": int((ulps <= 2).sum().item()),
        "le_4": int((ulps <= 4).sum().item()),
        "le_8": int((ulps <= 8).sum().item()),
    }
    return {
        "label": label,
        "shape": list(left.shape),
        "dtype": dtype,
        "elements": elements,
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
        "rms_diff": math.sqrt(float(torch.mean(difference.square()).item())),
        "reference_rms": math.sqrt(float(torch.mean(left32.square()).item())),
        "abs_diff_quantiles": tensor_quantiles(difference),
        "robust_relative_floor": robust_floor,
        "robust_relative_quantiles": tensor_quantiles(symmetric_relative),
        "ulp_distance_quantiles": tensor_quantiles(ulps),
        "ulp_histogram_cumulative": {
            key: {"count": count, "fraction": count / elements}
            for key, count in bins.items()
        },
    }


def compare_role(
    source: Dump,
    targets: list[Dump],
    *,
    comparison: str,
) -> list[dict]:
    source_layers = source.payload["layers"]
    target_names = [set(target.payload["layers"]) for target in targets]
    if any(names != set(source_layers) for names in target_names):
        raise SystemExit(f"{comparison}: layer-name sets differ")
    source_layout = source.manifest["cache_layout"]
    total_heads = int(source_layout["local_num_kv_heads"])
    if total_heads % len(targets):
        raise SystemExit("source KV heads do not divide across TP4 ranks")
    heads_per_rank = total_heads // len(targets)
    results: list[dict] = []
    for layer_name in sorted(source_layers):
        source_tokens, source_head_axis = logical_tokens(
            source, source_layers[layer_name]
        )
        for rank, target in enumerate(targets):
            target_tokens, target_head_axis = logical_tokens(
                target, target.payload["layers"][layer_name]
            )
            source_shard = source_tokens.narrow(
                source_head_axis, rank * heads_per_rank, heads_per_rank
            )
            if target_head_axis != source_head_axis:
                raise SystemExit(
                    f"{comparison}/{layer_name}/rank{rank}: logical head-axis mismatch"
                )
            results.append(
                compare_tensors(
                    source_shard,
                    target_tokens,
                    f"{comparison}/{layer_name}/rank{rank}",
                )
            )
    return results


def compare_rank_sets(
    left_ranks: list[Dump],
    right_ranks: list[Dump],
    *,
    comparison: str,
) -> list[dict]:
    results: list[dict] = []
    for rank, (left_dump, right_dump) in enumerate(zip(left_ranks, right_ranks)):
        left_layers = left_dump.payload["layers"]
        right_layers = right_dump.payload["layers"]
        if set(left_layers) != set(right_layers):
            raise SystemExit(f"{comparison}/rank{rank}: layer-name sets differ")
        for layer_name in sorted(left_layers):
            left, left_head_axis = logical_tokens(left_dump, left_layers[layer_name])
            right, right_head_axis = logical_tokens(
                right_dump, right_layers[layer_name]
            )
            if left_head_axis != right_head_axis:
                raise SystemExit(
                    f"{comparison}/{layer_name}/rank{rank}: head-axis mismatch"
                )
            results.append(
                compare_tensors(
                    left,
                    right,
                    f"{comparison}/{layer_name}/rank{rank}",
                )
            )
    return results


def summarize_comparison(records: list[dict]) -> dict:
    total = sum(record["elements"] for record in records)
    exact = sum(
        record["ulp_histogram_cumulative"]["exact"]["count"] for record in records
    )
    return {
        "layer_rank_records": len(records),
        "elements": total,
        "exact_fraction": exact / total,
        "worst_max_abs_diff": max(record["max_abs_diff"] for record in records),
        "per_layer_rank": records,
    }


def main() -> None:
    args = parse_args()
    if args.boundary_k < 0:
        raise SystemExit("--boundary-k must be nonnegative")
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise SystemExit("compare_kv_provenance.py requires torch") from error

    fixed_tokens, fixed_metadata = load_fixed_tokens(
        args.fixed_token_ids, args.boundary_k
    )
    if str(fixed_metadata["logical_request_id"]) != args.logical_request_id:
        raise SystemExit(
            "logical request mismatch between fixed-prefix file and CLI: "
            f"{fixed_metadata['logical_request_id']!r} != {args.logical_request_id!r}"
        )
    source = load_dump(args.tp1_dump_dir, "source_tp1", 0, 1)
    native = [
        load_dump(path, "native_tp4", rank, 4)
        for rank, path in enumerate(args.native_tp4_dump_dirs)
    ]
    migrated = (
        [
            load_dump(path, "migrated_tp4", rank, 4)
            for rank, path in enumerate(args.migrated_tp4_dump_dirs)
        ]
        if args.migrated_tp4_dump_dirs
        else None
    )
    all_dumps = [source, *native, *(migrated or [])]
    provenance = validate_provenance(all_dumps, fixed_tokens)
    expected_model = fixed_metadata.get("model")
    if expected_model is not None and str(expected_model) != provenance["model"]:
        raise SystemExit(
            f"model mismatch: fixed-prefix file={expected_model}, "
            f"dumps={provenance['model']}"
        )

    comparisons = {
        "source_reshard_vs_native_tp4": summarize_comparison(
            compare_role(
                source,
                native,
                comparison="source_reshard_vs_native_tp4",
            )
        )
    }
    if migrated is not None:
        comparisons["source_reshard_vs_migrated_tp4"] = summarize_comparison(
            compare_role(
                source,
                migrated,
                comparison="source_reshard_vs_migrated_tp4",
            )
        )
        comparisons["native_tp4_vs_migrated_tp4"] = summarize_comparison(
            compare_rank_sets(
                native,
                migrated,
                comparison="native_tp4_vs_migrated_tp4",
            )
        )

    payload = {
        "format_version": 1,
        "phase": "BridgeTP D3 Phase 9 D-1",
        "logical_request_id": args.logical_request_id,
        "boundary_k": args.boundary_k,
        "fixed_token_count": len(fixed_tokens),
        "fixed_token_ids_file": str(args.fixed_token_ids),
        "provenance": provenance,
        "comparisons": comparisons,
        "formal_causal_conclusion": None,
        "evidence_boundary": (
            "Per-layer/rank absolute, robust-relative, RMS, quantile, and ULP "
            "measurements are reported. No uncalibrated threshold is used to "
            "declare migration blameless; interpret these jointly with D-0 and "
            "paired D-3 evidence."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"fixed prefix: {len(fixed_tokens)} tokens; K={args.boundary_k}")
    for name, summary in comparisons.items():
        print(
            f"{name}: elements={summary['elements']:,}, "
            f"exact={summary['exact_fraction']:.6f}, "
            f"worst_abs={summary['worst_max_abs_diff']:.6e}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
