# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch

from vllm.bridge_tp.config import BridgeTPDumpConfig, get_bridge_tp_dump_config
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index

logger = init_logger(__name__)

_dumped_request_ids: set[str] = set()
_disabled_after_error = False
_warned_multi_request = False


def _safe_request_id(request_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_id)
    return safe[:160] or "request"


def _atomic_json_dump(data: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_path, path)


def _select_request_id(
    config: BridgeTPDumpConfig,
    input_batch: Any,
) -> str | None:
    global _warned_multi_request

    if config.target_request_id is not None:
        if config.target_request_id in input_batch.req_id_to_index:
            return config.target_request_id
        return None

    request_ids = input_batch.req_ids
    if len(request_ids) == 1:
        return request_ids[0]

    if not _warned_multi_request:
        logger.warning(
            "BridgeTP KV dump is waiting because %d requests are active. "
            "Run a single request or set BRIDGETP_DUMP_REQUEST_ID.",
            len(request_ids),
        )
        _warned_multi_request = True
    return None


def _get_block_axis(
    attn_groups: list[list[Any]],
    cache_dtype: str,
    block_size: int,
) -> int:
    if len(attn_groups) != 1 or len(attn_groups[0]) != 1:
        raise NotImplementedError(
            "BridgeTP Phase 1-3 supports one uniform attention/KV-cache group only"
        )

    group = attn_groups[0][0]
    spec = group.kv_cache_spec
    if not hasattr(spec, "num_kv_heads") or not hasattr(spec, "head_size"):
        raise NotImplementedError(
            "BridgeTP Phase 1-3 only supports standard attention KV caches"
        )

    block_axis = group.backend.get_kv_cache_block_dim(
        block_size,
        spec.num_kv_heads,
        spec.head_size,
        cache_dtype_str=cache_dtype,
    )
    if block_axis not in (0, 1):
        raise ValueError(f"Unexpected KV-cache block axis: {block_axis}")
    return block_axis


def _ordered_layer_names(kv_cache_config: Any) -> list[str]:
    groups = kv_cache_config.kv_cache_groups
    if len(groups) != 1:
        raise NotImplementedError("BridgeTP Phase 1-3 supports one KV-cache group only")
    return sorted(groups[0].layer_names, key=extract_layer_index)


def _get_request_block_ids(
    input_batch: Any,
    request_index: int,
    num_computed_tokens: int,
) -> tuple[list[int], int]:
    block_tables = input_batch.block_table.block_tables
    if len(block_tables) != 1:
        raise NotImplementedError(
            "BridgeTP Phase 1-3 supports one KV-cache block table only"
        )

    block_table = block_tables[0]
    block_size = int(block_table.block_size)
    num_required_blocks = (num_computed_tokens + block_size - 1) // block_size
    num_allocated_blocks = int(block_table.num_blocks_per_row[request_index])
    if num_required_blocks > num_allocated_blocks:
        raise RuntimeError(
            "Request block table is shorter than its computed-token state: "
            f"required={num_required_blocks}, allocated={num_allocated_blocks}"
        )

    row = block_table.get_numpy_array()[request_index, :num_required_blocks]
    return [int(block_id) for block_id in row], block_size


def _estimate_dump_bytes(
    kv_caches: list[torch.Tensor],
    block_axis: int,
    num_blocks: int,
) -> int:
    total = 0
    for cache in kv_caches:
        elements_per_block = cache.numel() // cache.shape[block_axis]
        total += elements_per_block * num_blocks * cache.element_size()
    return total


def _copy_request_blocks(
    kv_caches: list[torch.Tensor],
    layer_names: list[str],
    block_ids: list[int],
    block_axis: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], float]:
    if len(kv_caches) != len(layer_names):
        raise RuntimeError(
            "KV-cache tensor count does not match the configured layer count: "
            f"{len(kv_caches)} != {len(layer_names)}"
        )

    device = kv_caches[0].device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    block_index = torch.tensor(block_ids, dtype=torch.long, device=device)
    tensors: dict[str, torch.Tensor] = {}
    layers: list[dict[str, Any]] = []
    for cache_index, (layer_name, cache) in enumerate(zip(layer_names, kv_caches)):
        if cache.device != device:
            raise RuntimeError("All KV-cache tensors must be on the same device")
        if not block_ids:
            selected = cache.narrow(block_axis, 0, 0).detach().cpu()
        else:
            max_block_id = max(block_ids)
            if max_block_id >= cache.shape[block_axis]:
                raise IndexError(
                    f"Physical block {max_block_id} is outside layer "
                    f"{layer_name} shape {tuple(cache.shape)}"
                )
            selected = cache.index_select(block_axis, block_index).detach().cpu()

        tensors[layer_name] = selected
        layers.append(
            {
                "cache_index": cache_index,
                "layer_name": layer_name,
                "source_shape": list(cache.shape),
                "dump_shape": list(selected.shape),
                "dtype": str(cache.dtype),
                "block_axis": block_axis,
            }
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    copy_ms = (time.perf_counter() - started) * 1000
    return tensors, layers, copy_ms


def _dump_request_kv(
    *,
    config: BridgeTPDumpConfig,
    request_id: str,
    kv_caches: list[torch.Tensor],
    requests: dict[str, Any],
    input_batch: Any,
    scheduler_output: Any,
    kv_cache_config: Any,
    attn_groups: list[list[Any]],
    cache_dtype: str,
    model_name: str,
    tp_rank: int,
    tp_world_size: int,
) -> None:
    if tp_world_size != 1:
        raise NotImplementedError(
            "BridgeTP Phase 1-3 dump must first be validated with TP1"
        )
    if scheduler_output.scheduled_spec_decode_tokens:
        raise NotImplementedError(
            "BridgeTP Phase 1-3 does not support speculative decoding"
        )
    if not kv_caches:
        raise RuntimeError("The model runner has no initialized KV-cache tensors")

    request_index = input_batch.req_id_to_index[request_id]
    request = requests[request_id]
    num_scheduled_tokens = int(scheduler_output.num_scheduled_tokens.get(request_id, 0))
    num_computed_before = int(input_batch.num_computed_tokens_cpu[request_index])
    num_computed_tokens = num_computed_before + num_scheduled_tokens
    num_known_tokens = int(request.num_tokens)
    num_output_tokens = len(request.output_token_ids)

    if num_output_tokens < config.dump_after_output_tokens:
        return
    if num_computed_tokens > num_known_tokens:
        raise RuntimeError(
            "Computed-token count exceeds the known token history: "
            f"computed={num_computed_tokens}, known={num_known_tokens}"
        )

    block_ids, block_size = _get_request_block_ids(
        input_batch, request_index, num_computed_tokens
    )
    block_axis = _get_block_axis(attn_groups, cache_dtype, block_size)
    layer_names = _ordered_layer_names(kv_cache_config)
    if len(kv_caches) != len(layer_names):
        raise RuntimeError(
            "KV-cache tensor count does not match the configured layer count: "
            f"{len(kv_caches)} != {len(layer_names)}"
        )
    estimated_bytes = _estimate_dump_bytes(kv_caches, block_axis, len(block_ids))
    if config.include_tensors and estimated_bytes > config.max_bytes:
        raise RuntimeError(
            "BridgeTP KV dump exceeds BRIDGETP_DUMP_MAX_BYTES: "
            f"estimated={estimated_bytes}, limit={config.max_bytes}"
        )

    safe_request_id = _safe_request_id(request_id)
    output_dir = config.output_dir / safe_request_id / f"tp_rank_{tp_rank}"
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tensors: dict[str, torch.Tensor] = {}
    layers: list[dict[str, Any]] = []
    copy_ms = 0.0
    if config.include_tensors:
        tensors, layers, copy_ms = _copy_request_blocks(
            kv_caches, layer_names, block_ids, block_axis
        )
    else:
        layers = [
            {
                "cache_index": index,
                "layer_name": layer_name,
                "source_shape": list(cache.shape),
                "dtype": str(cache.dtype),
                "block_axis": block_axis,
            }
            for index, (layer_name, cache) in enumerate(zip(layer_names, kv_caches))
        ]

    tensor_path: Path | None = None
    save_ms = 0.0
    if config.include_tensors:
        tensor_path = output_dir / "kv_blocks.pt"
        temporary_path = tensor_path.with_suffix(".pt.tmp")
        save_started = time.perf_counter()
        torch.save(
            {
                "format_version": 1,
                "request_id": request_id,
                "tp_rank": tp_rank,
                "physical_block_ids": block_ids,
                "layers": tensors,
            },
            temporary_path,
        )
        os.replace(temporary_path, tensor_path)
        save_ms = (time.perf_counter() - save_started) * 1000

    computed_token_ids = [
        request.get_token_id(index) for index in range(num_computed_tokens)
    ]
    known_not_computed_token_ids = [
        request.get_token_id(index)
        for index in range(num_computed_tokens, num_known_tokens)
    ]
    tokens = {
        "request_id": request_id,
        "prompt_token_ids": request.prompt_token_ids,
        "output_token_ids": request.output_token_ids,
        "computed_token_ids": computed_token_ids,
        "known_not_computed_token_ids": known_not_computed_token_ids,
        "num_prompt_tokens": int(request.num_prompt_tokens),
        "num_output_tokens": num_output_tokens,
        "num_known_tokens": num_known_tokens,
        "num_computed_tokens": num_computed_tokens,
        "pending_known_tokens": num_known_tokens - num_computed_tokens,
    }
    _atomic_json_dump(tokens, output_dir / "generated_tokens.json")

    total_ms = (time.perf_counter() - started) * 1000
    last_block_valid_tokens = num_computed_tokens % block_size
    if num_computed_tokens and last_block_valid_tokens == 0:
        last_block_valid_tokens = block_size
    manifest = {
        "format_version": 1,
        "phase": "BridgeTP D3 Phase 1-3",
        "scope": "TP1 request-scoped KV-cache dump; no takeover",
        "model": model_name,
        "request_id": request_id,
        "tp_rank": tp_rank,
        "tp_world_size": tp_world_size,
        "cache_dtype": cache_dtype,
        "num_layers": len(kv_caches),
        "block_size": block_size,
        "block_axis": block_axis,
        "physical_block_ids": block_ids,
        "num_physical_blocks": len(block_ids),
        "last_block_valid_tokens": last_block_valid_tokens,
        "num_prompt_tokens": int(request.num_prompt_tokens),
        "num_output_tokens": num_output_tokens,
        "num_known_tokens": num_known_tokens,
        "num_computed_tokens_before_iteration": num_computed_before,
        "num_scheduled_tokens_this_iteration": num_scheduled_tokens,
        "num_computed_tokens": num_computed_tokens,
        "pending_known_tokens": num_known_tokens - num_computed_tokens,
        "estimated_tensor_bytes": estimated_bytes,
        "tensor_file_bytes": tensor_path.stat().st_size if tensor_path else 0,
        "include_tensors": config.include_tensors,
        "copy_ms": copy_ms,
        "save_ms": save_ms,
        "total_dump_ms": total_ms,
        "layers": layers,
    }
    _atomic_json_dump(manifest, output_dir / "manifest.json")
    _dumped_request_ids.add(request_id)

    logger.warning(
        "BridgeTP dumped request %s: computed=%d, known=%d, blocks=%d, "
        "copy_ms=%.3f, save_ms=%.3f, output=%s",
        request_id,
        num_computed_tokens,
        num_known_tokens,
        len(block_ids),
        copy_ms,
        save_ms,
        output_dir,
    )


def maybe_dump_kv_cache(
    *,
    kv_caches: list[torch.Tensor],
    requests: dict[str, Any],
    input_batch: Any,
    scheduler_output: Any,
    kv_cache_config: Any,
    attn_groups: list[list[Any]],
    cache_dtype: str,
    model_name: str,
    tp_rank: int,
    tp_world_size: int,
    async_scheduling: bool,
) -> None:
    """Dump one request's real TP1 KV blocks after sampling bookkeeping.

    This hook is a deliberately narrow diagnostic. It is disabled by default,
    only supports one standard-attention KV-cache group, and never changes KV
    ownership or scheduler state.
    """
    global _disabled_after_error

    config = get_bridge_tp_dump_config()
    if not config.enabled or _disabled_after_error:
        return

    try:
        if async_scheduling:
            raise NotImplementedError(
                "BridgeTP Phase 1-3 requires synchronous scheduling so token "
                "IDs and KV state can be captured at the same boundary"
            )

        request_id = _select_request_id(config, input_batch)
        if request_id is None or request_id in _dumped_request_ids:
            return

        _dump_request_kv(
            config=config,
            request_id=request_id,
            kv_caches=kv_caches,
            requests=requests,
            input_batch=input_batch,
            scheduler_output=scheduler_output,
            kv_cache_config=kv_cache_config,
            attn_groups=attn_groups,
            cache_dtype=cache_dtype,
            model_name=model_name,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
    except Exception:
        if config.strict:
            raise
        _disabled_after_error = True
        logger.exception(
            "BridgeTP KV dump failed and has been disabled for this process"
        )
