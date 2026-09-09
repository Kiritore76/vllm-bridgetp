# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure five-GPU BridgeTP KV transfer and split-KV attention.

Launch this program with ``torchrun --nproc-per-node=5``.  Global rank zero is
the TP1 coordinator and ranks one through four are the TP4 KV holders.  The
runner performs real NCCL tensor transfers and real CUDA attention arithmetic;
it is not a timing simulation and it does not yet patch vLLM's online decode
path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
protocol_path = REPO_ROOT / "vllm" / "bridge_tp" / "remote_attention_protocol.py"
protocol_spec = importlib.util.spec_from_file_location(
    "_bridgetp_remote_attention_protocol", protocol_path
)
if protocol_spec is None or protocol_spec.loader is None:
    raise RuntimeError(f"cannot load protocol module from {protocol_path}")
protocol_module = importlib.util.module_from_spec(protocol_spec)
sys.modules[protocol_spec.name] = protocol_module
protocol_spec.loader.exec_module(protocol_module)
RemoteAttentionGeometry = protocol_module.RemoteAttentionGeometry
make_partition = protocol_module.make_partition
one_layer_query_wire_bytes = protocol_module.one_layer_query_wire_bytes
one_layer_rank_transfer_bytes = protocol_module.one_layer_rank_transfer_bytes
one_layer_statistics_wire_bytes = protocol_module.one_layer_statistics_wire_bytes
projected_token_wire_bytes = protocol_module.projected_token_wire_bytes
validate_measurements = protocol_module.validate_measurements


@dataclass(frozen=True)
class ValidationCase:
    """One measured remote-attention condition."""

    context_tokens: int
    requested_remote_fraction: float
    target_load_repeats: int
    copy_bytes_per_rank_step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--expected-revision")
    parser.add_argument("--tensor-bundle", type=Path)
    parser.add_argument("--expected-tensor-sha256")
    parser.add_argument("--expected-world-size", type=int, default=5)
    parser.add_argument("--expected-gpu-name-substring", default="A100")
    parser.add_argument("--context-tokens", type=int, nargs="+", default=[128, 1024])
    parser.add_argument(
        "--remote-fractions", type=float, nargs="+", default=[0.25, 0.75]
    )
    parser.add_argument("--target-load-repeats", type=int, nargs="+", default=[0])
    parser.add_argument("--copy-bytes-per-rank-step", type=int, nargs="+", default=[0])
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument("--background-gemm-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-3)
    parser.add_argument("--mean-abs-tolerance", type=float, default=1e-4)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-query-heads", type=int, default=40)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--target-tp-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    return parser.parse_args()


def command_arguments_for_provenance(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return JSON-friendly arguments without mutating the live namespace."""
    arguments = dict(vars(args))
    for name, value in arguments.items():
        if isinstance(value, Path):
            arguments[name] = str(value)
    return arguments


def git_revision() -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPO_ROOT.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tensor_bundle(
    path: Path,
    geometry: RemoteAttentionGeometry,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one captured Qwen attention layer without executing pickle code."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("tensor bundle must be a dictionary")
    missing = {"q", "k", "v"} - set(raw)
    if missing:
        raise ValueError(f"tensor bundle is missing {sorted(missing)}")
    q, k, v = raw["q"], raw["k"], raw["v"]
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise ValueError("tensor bundle q/k/v values must be tensors")
    if tuple(q.shape) != (geometry.num_query_heads, geometry.head_dim):
        raise ValueError(f"captured Q has invalid shape {tuple(q.shape)}")
    if k.ndim != 3 or v.shape != k.shape:
        raise ValueError("captured K/V must have equal three-dimensional shapes")
    if k.shape[0] != geometry.num_kv_heads:
        raise ValueError(f"captured K has invalid head count {k.shape[0]}")
    if k.shape[2] != geometry.head_dim:
        raise ValueError(f"captured K has invalid head dimension {k.shape[2]}")
    if not torch.isfinite(q).all() or not torch.isfinite(k).all():
        raise ValueError("captured Q/K contains non-finite values")
    if not torch.isfinite(v).all():
        raise ValueError("captured V contains non-finite values")
    return q.contiguous(), k.contiguous(), v.contiguous()


def percentile(values: list[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("invalid percentile input")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype {name}")


def expand_gqa(kv: torch.Tensor, query_heads: int) -> torch.Tensor:
    """Expand KV heads to their grouped-query head assignments."""
    if kv.ndim != 3:
        raise ValueError("KV tensor must have [heads, tokens, head_dim] shape")
    kv_heads = kv.shape[0]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    return kv.repeat_interleave(query_heads // kv_heads, dim=0)


def attention_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stable per-head ``m``, ``l``, and unnormalised ``o``."""
    if q.ndim != 2 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("invalid Q/K/V shapes")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and KV head dimensions differ")
    if k.shape[1] == 0:
        heads = q.shape[0]
        return (
            torch.full((heads,), -torch.inf, device=q.device),
            torch.zeros((heads,), dtype=torch.float32, device=q.device),
            torch.zeros(q.shape, dtype=torch.float32, device=q.device),
        )
    expanded_k = expand_gqa(k, q.shape[0]).float()
    expanded_v = expand_gqa(v, q.shape[0]).float()
    scores = torch.einsum("hd,hsd->hs", q.float(), expanded_k)
    scores *= 1.0 / math.sqrt(q.shape[-1])
    maximum = scores.max(dim=-1).values
    exponentials = torch.exp(scores - maximum.unsqueeze(-1))
    denominator = exponentials.sum(dim=-1)
    numerator = torch.einsum("hs,hsd->hd", exponentials, expanded_v)
    return maximum, denominator, numerator


def merge_stats(
    local: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    remote: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Merge two token partitions using global online-softmax statistics."""
    local_m, local_l, local_o = local
    remote_m, remote_l, remote_o = remote
    maximum = torch.maximum(local_m, remote_m)
    local_weight = torch.exp(local_m - maximum)
    remote_weight = torch.exp(remote_m - maximum)
    denominator = local_weight * local_l + remote_weight * remote_l
    numerator = local_weight.unsqueeze(-1) * local_o
    numerator += remote_weight.unsqueeze(-1) * remote_o
    return numerator / denominator.unsqueeze(-1)


def full_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    maximum, denominator, numerator = attention_stats(q, k, v)
    del maximum
    return numerator / denominator.unsqueeze(-1)


def make_cases(args: argparse.Namespace) -> list[ValidationCase]:
    cases = []
    combinations = itertools.product(
        args.context_tokens,
        args.remote_fractions,
        args.target_load_repeats,
        args.copy_bytes_per_rank_step,
    )
    for context, fraction, load_repeats, copy_bytes in combinations:
        if context <= 0:
            raise ValueError("context lengths must be positive")
        if not 0 < fraction <= 1:
            raise ValueError("remote fractions must be in (0, 1]")
        if load_repeats < 0 or copy_bytes < 0:
            raise ValueError("load repeats and copy bytes cannot be negative")
        cases.append(
            ValidationCase(
                context_tokens=context,
                requested_remote_fraction=fraction,
                target_load_repeats=load_repeats,
                copy_bytes_per_rank_step=copy_bytes,
            )
        )
    if not cases:
        raise ValueError("no validation cases requested")
    return cases


def setup_output(path: Path, rank: int) -> None:
    error: list[str | None] = [None]
    if rank == 0:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except Exception as exc:  # pragma: no cover - exercised on server
            error[0] = f"cannot create output directory: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])


def gpu_inventory(rank: int, world_size: int) -> list[dict[str, Any]]:
    local = {
        "rank": rank,
        "device_index": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(),
        "total_memory_bytes": torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory,
    }
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return [dict(item) for item in gathered if item is not None]


def p2p_batch(operations: list[dist.P2POp]) -> None:
    if not operations:
        return
    requests = dist.batch_isend_irecv(operations)
    for request in requests:
        request.wait()


def worker_step(
    *,
    q_buffer: torch.Tensor,
    k_remote: torch.Tensor,
    v_remote: torch.Tensor,
    copy_buffer: torch.Tensor | None,
    load_left: torch.Tensor | None,
    load_right: torch.Tensor | None,
    load_repeats: int,
) -> None:
    receives = []
    if copy_buffer is not None:
        receives.append(dist.P2POp(dist.irecv, copy_buffer, 0))
    receives.append(dist.P2POp(dist.irecv, q_buffer, 0))
    p2p_batch(receives)
    load_result = None
    for _ in range(load_repeats):
        load_result = torch.mm(load_left, load_right)
    remote = attention_stats(q_buffer, k_remote, v_remote)
    sends = [
        dist.P2POp(dist.isend, remote[0], 0),
        dist.P2POp(dist.isend, remote[1], 0),
        dist.P2POp(dist.isend, remote[2], 0),
    ]
    p2p_batch(sends)
    del load_result


def coordinator_step(
    *,
    q: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    geometry: RemoteAttentionGeometry,
    copy_buffers: list[torch.Tensor] | None,
) -> torch.Tensor:
    sends = []
    q_shards = []
    for worker_rank in range(1, geometry.target_tp_size + 1):
        index = worker_rank - 1
        if copy_buffers is not None:
            sends.append(dist.P2POp(dist.isend, copy_buffers[index], worker_rank))
        q_start = index * geometry.query_heads_per_rank
        q_end = q_start + geometry.query_heads_per_rank
        q_shard = q[q_start:q_end].contiguous()
        q_shards.append(q_shard)
        sends.append(dist.P2POp(dist.isend, q_shard, worker_rank))
    requests = dist.batch_isend_irecv(sends)
    local = attention_stats(q, k_local, v_local)
    for request in requests:
        request.wait()

    remote_parts = []
    receives = []
    for worker_rank in range(1, geometry.target_tp_size + 1):
        maximum = torch.empty(
            geometry.query_heads_per_rank,
            dtype=torch.float32,
            device=q.device,
        )
        denominator = torch.empty_like(maximum)
        numerator = torch.empty(
            (geometry.query_heads_per_rank, geometry.head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        remote_parts.append((maximum, denominator, numerator))
        receives.extend(
            [
                dist.P2POp(dist.irecv, maximum, worker_rank),
                dist.P2POp(dist.irecv, denominator, worker_rank),
                dist.P2POp(dist.irecv, numerator, worker_rank),
            ]
        )
    p2p_batch(receives)
    remote = tuple(
        torch.cat([part[field] for part in remote_parts], dim=0) for field in range(3)
    )
    return merge_stats(local, remote)


def stage_remote_kv(
    *,
    rank: int,
    geometry: RemoteAttentionGeometry,
    k_full: torch.Tensor | None,
    v_full: torch.Tensor | None,
    k_remote: torch.Tensor | None,
    v_remote: torch.Tensor | None,
    local_tokens: int,
) -> tuple[float, int]:
    dist.barrier()
    torch.cuda.synchronize()
    started = time.perf_counter()
    if rank == 0:
        operations = []
        shards = []
        for worker_rank in range(1, geometry.target_tp_size + 1):
            index = worker_rank - 1
            kv_start = index * geometry.kv_heads_per_rank
            kv_end = kv_start + geometry.kv_heads_per_rank
            k_shard = k_full[kv_start:kv_end, local_tokens:].contiguous()
            v_shard = v_full[kv_start:kv_end, local_tokens:].contiguous()
            shards.extend([k_shard, v_shard])
            operations.extend(
                [
                    dist.P2POp(dist.isend, k_shard, worker_rank),
                    dist.P2POp(dist.isend, v_shard, worker_rank),
                ]
            )
        p2p_batch(operations)
    else:
        p2p_batch(
            [
                dist.P2POp(dist.irecv, k_remote, 0),
                dist.P2POp(dist.irecv, v_remote, 0),
            ]
        )
    dist.barrier()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000

    actual = 0
    if rank != 0:
        actual = (k_remote.numel() + v_remote.numel()) * k_remote.element_size()
    actual_tensor = torch.tensor(actual, dtype=torch.int64, device="cuda")
    dist.reduce(actual_tensor, dst=0, op=dist.ReduceOp.SUM)
    return elapsed_ms, int(actual_tensor.item()) if rank == 0 else actual


def run_case(
    *,
    args: argparse.Namespace,
    case: ValidationCase,
    case_index: int,
    geometry: RemoteAttentionGeometry,
    rank: int,
    input_dtype: torch.dtype,
    captured_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> dict[str, Any] | None:
    partition = make_partition(
        case.context_tokens,
        case.requested_remote_fraction,
        geometry.block_size,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    remote_shape = (
        geometry.kv_heads_per_rank,
        partition.remote_tokens,
        geometry.head_dim,
    )
    q = k_full = v_full = k_local = v_local = None
    k_remote = v_remote = q_buffer = None
    if rank == 0:
        if captured_tensors is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + case_index * 1009)
            q = torch.randn(
                (geometry.num_query_heads, geometry.head_dim),
                generator=generator,
                dtype=input_dtype,
                device=device,
            )
            k_full = torch.randn(
                (
                    geometry.num_kv_heads,
                    case.context_tokens,
                    geometry.head_dim,
                ),
                generator=generator,
                dtype=input_dtype,
                device=device,
            )
            v_full = torch.randn(
                k_full.shape,
                generator=generator,
                dtype=input_dtype,
                device=device,
            )
        else:
            captured_q, captured_k, captured_v = captured_tensors
            if captured_k.shape[1] < case.context_tokens:
                raise ValueError(
                    f"captured context {captured_k.shape[1]} is smaller than "
                    f"requested {case.context_tokens}"
                )
            q = captured_q.to(device=device, dtype=input_dtype)
            k_full = captured_k[:, : case.context_tokens].to(
                device=device, dtype=input_dtype
            )
            v_full = captured_v[:, : case.context_tokens].to(
                device=device, dtype=input_dtype
            )
        k_local = k_full[:, : partition.local_tokens].contiguous()
        v_local = v_full[:, : partition.local_tokens].contiguous()
    else:
        k_remote = torch.empty(remote_shape, dtype=input_dtype, device=device)
        v_remote = torch.empty_like(k_remote)
        q_buffer = torch.empty(
            (geometry.query_heads_per_rank, geometry.head_dim),
            dtype=input_dtype,
            device=device,
        )

    staging_ms, staged_bytes = stage_remote_kv(
        rank=rank,
        geometry=geometry,
        k_full=k_full,
        v_full=v_full,
        k_remote=k_remote,
        v_remote=v_remote,
        local_tokens=partition.local_tokens,
    )
    expected_staged = one_layer_rank_transfer_bytes(partition, geometry)
    expected_staged *= geometry.target_tp_size

    rounded_copy_bytes = (
        case.copy_bytes_per_rank_step // geometry.dtype_bytes
    ) * geometry.dtype_bytes
    copy_elements = rounded_copy_bytes // geometry.dtype_bytes
    copy_buffers = None
    copy_buffer = None
    if copy_elements:
        if rank == 0:
            copy_buffers = [
                torch.zeros(copy_elements, dtype=input_dtype, device=device)
                for _ in range(geometry.target_tp_size)
            ]
        else:
            copy_buffer = torch.empty(copy_elements, dtype=input_dtype, device=device)

    load_left = load_right = None
    if rank != 0 and case.target_load_repeats:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + rank)
        load_left = torch.randn(
            (args.background_gemm_size, args.background_gemm_size),
            generator=generator,
            dtype=input_dtype,
            device=device,
        )
        load_right = torch.randn_like(load_left)

    latencies_ms: list[float] = []
    last_output = None
    total_steps = args.warmup_steps + args.measured_steps
    for step in range(total_steps):
        dist.barrier()
        if rank == 0:
            torch.cuda.synchronize()
            started = time.perf_counter()
            last_output = coordinator_step(
                q=q,
                k_local=k_local,
                v_local=v_local,
                geometry=geometry,
                copy_buffers=copy_buffers,
            )
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            if step >= args.warmup_steps:
                latencies_ms.append(elapsed_ms)
        else:
            worker_step(
                q_buffer=q_buffer,
                k_remote=k_remote,
                v_remote=v_remote,
                copy_buffer=copy_buffer,
                load_left=load_left,
                load_right=load_right,
                load_repeats=case.target_load_repeats,
            )
    dist.barrier()

    if rank != 0:
        return None
    reference = full_attention(q, k_full, v_full)
    difference = (last_output - reference).abs()
    relative = difference / reference.abs().clamp_min(1e-8)
    finite = bool(torch.isfinite(last_output).all().item())
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    max_relative = float(relative.max().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            last_output.flatten().float(),
            reference.flatten().float(),
            dim=0,
        ).item()
    )
    staging_seconds = staging_ms / 1000
    staging_gib_s = staged_bytes / staging_seconds / 1024**3
    status = "PASS"
    if (
        not finite
        or max_abs > args.max_abs_tolerance
        or mean_abs > args.mean_abs_tolerance
        or staged_bytes != expected_staged
    ):
        status = "FAIL"
    return {
        "case_index": case_index,
        **asdict(case),
        **partition.to_dict(),
        "status": status,
        "finite": finite,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_relative_error": max_relative,
        "cosine_similarity": cosine,
        "staged_kv_bytes": staged_bytes,
        "expected_staged_kv_bytes": expected_staged,
        "staging_ms": staging_ms,
        "staging_gib_s": staging_gib_s,
        "step_min_ms": min(latencies_ms),
        "step_p50_ms": statistics.median(latencies_ms),
        "step_p95_ms": percentile(latencies_ms, 0.95),
        "step_max_ms": max(latencies_ms),
        "projected_all_layers_p50_ms": (
            statistics.median(latencies_ms) * geometry.num_layers
        ),
        "query_wire_bytes_per_layer": one_layer_query_wire_bytes(geometry),
        "statistics_wire_bytes_per_layer": (one_layer_statistics_wire_bytes(geometry)),
        "projected_wire_bytes_per_token": projected_token_wire_bytes(geometry),
        "actual_copy_bytes_per_rank_step": rounded_copy_bytes,
        "actual_copy_bytes_all_ranks_step": (
            rounded_copy_bytes * geometry.target_tp_size
        ),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    geometry = RemoteAttentionGeometry(
        num_layers=args.num_layers,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        target_tp_size=args.target_tp_size,
        block_size=args.block_size,
        dtype_bytes=2,
    )
    geometry.validate()
    cases = make_cases(args)
    if args.expected_world_size != geometry.target_tp_size + 1:
        raise ValueError("expected world size must be one TP1 plus TP4 ranks")
    if args.warmup_steps < 0 or args.measured_steps <= 0:
        raise ValueError("warmup must be non-negative and measured steps positive")
    revision = git_revision()
    if args.expected_revision and revision != args.expected_revision:
        raise RuntimeError(f"revision {revision} != expected {args.expected_revision}")
    preflight = {
        "format_version": 1,
        "status": "VALID",
        "revision": revision,
        "cases": len(cases),
        "geometry": geometry.to_dict(),
        "projected_wire_bytes_per_token": projected_token_wire_bytes(geometry),
        "evidence_class": (
            "GPU_CAPTURED_MODEL_TENSORS"
            if args.tensor_bundle
            else "GPU_SYNTHETIC_KV_TRANSFER"
        ),
    }
    tensor_sha = None
    if args.tensor_bundle:
        if not args.tensor_bundle.is_file():
            raise FileNotFoundError(args.tensor_bundle)
        tensor_sha = file_sha256(args.tensor_bundle)
        if args.expected_tensor_sha256 and tensor_sha != args.expected_tensor_sha256:
            raise RuntimeError(
                f"tensor SHA-256 {tensor_sha} != expected {args.expected_tensor_sha256}"
            )
        preflight["tensor_bundle"] = str(args.tensor_bundle.resolve())
        preflight["tensor_sha256"] = tensor_sha
    elif args.expected_tensor_sha256:
        raise ValueError("expected tensor SHA requires --tensor-bundle")
    if args.validate_only:
        print(json.dumps(preflight, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real transfer validation")
    if not dist.is_nccl_available():
        raise RuntimeError("NCCL is required for five-GPU validation")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError("launch with torchrun; LOCAL_RANK is missing")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"world size {world_size} != expected {args.expected_world_size}"
        )

    if world_size != geometry.target_tp_size + 1:
        raise RuntimeError("world size must be one TP1 plus all TP4 ranks")
    setup_output(args.out_dir, rank)
    inventory = gpu_inventory(rank, world_size)
    input_dtype = torch_dtype(args.dtype)
    captured_tensors = None
    if rank == 0 and args.tensor_bundle:
        captured_tensors = load_tensor_bundle(args.tensor_bundle, geometry)

    rows = []
    for case_index, case in enumerate(cases, start=1):
        row = run_case(
            args=args,
            case=case,
            case_index=case_index,
            geometry=geometry,
            rank=rank,
            input_dtype=input_dtype,
            captured_tensors=captured_tensors,
        )
        if rank == 0:
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    if rank == 0:
        acceptance = validate_measurements(
            rows,
            expected_cases=len(cases),
            max_abs_tolerance=args.max_abs_tolerance,
            mean_abs_tolerance=args.mean_abs_tolerance,
        )
        if args.expected_gpu_name_substring:
            mismatches = [
                item
                for item in inventory
                if args.expected_gpu_name_substring.lower()
                not in str(item["device_name"]).lower()
            ]
            if mismatches:
                acceptance["status"] = "FAIL"
                acceptance["errors"].append(
                    "GPU inventory does not match expected name substring"
                )
        runner_path = Path(__file__).resolve()
        protocol_path = (
            REPO_ROOT / "vllm" / "bridge_tp" / "remote_attention_protocol.py"
        )
        provenance = {
            "format_version": 1,
            "evidence_class": preflight["evidence_class"],
            "revision": revision,
            "command_arguments": command_arguments_for_provenance(args),
            "geometry": geometry.to_dict(),
            "world_size": world_size,
            "gpu_inventory": inventory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "source_sha256": {
                "runner": file_sha256(runner_path),
                "protocol": file_sha256(protocol_path),
                "tensor_bundle": tensor_sha,
            },
            "evidence_boundary": (
                "Real NCCL transfer and CUDA arithmetic. Captured model "
                "tensors are used only when evidence_class says so. This is "
                "a multi-GPU protocol microbenchmark, not an online vLLM "
                "Bridge result."
            ),
        }
        write_csv(rows, args.out_dir / "measurements.csv")
        (args.out_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / "acceptance.json").write_text(
            json.dumps(acceptance, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(acceptance, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and acceptance["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
