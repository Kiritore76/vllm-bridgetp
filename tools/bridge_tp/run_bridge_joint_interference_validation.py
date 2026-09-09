# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure joint Bridge remote-attention and KV-copy interference on TP4.

Global rank zero represents TP1 and ranks one through four represent TP4. Each
factorial cell measures ATTENTION_ONLY, COPY_ONLY, and JOINT. Every treatment
step is bracketed by identical TP4 GEMM controls without Bridge work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ra_path = REPO_ROOT / "tools" / "bridge_tp" / "run_remote_attention_validation.py"
protocol_path = (
    REPO_ROOT / "vllm" / "bridge_tp" / "bridge_joint_interference_protocol.py"
)
ra = load_module("_bridgetp_remote_attention_runner", ra_path)
protocol = load_module("_bridgetp_bridge_joint_interference_protocol", protocol_path)

MODES = protocol.MODES
build_cases = protocol.build_cases
factorial_summary = protocol.factorial_summary
validate_results = protocol.validate_results
RemoteAttentionGeometry = ra.RemoteAttentionGeometry


@dataclass
class TargetLoad:
    left: torch.Tensor | None
    right: torch.Tensor | None
    output: torch.Tensor | None
    stream: torch.cuda.Stream | None


@dataclass
class CaseTensors:
    q: torch.Tensor | None
    k_full: torch.Tensor | None
    v_full: torch.Tensor | None
    k_local: torch.Tensor | None
    v_local: torch.Tensor | None
    k_remote: torch.Tensor | None
    v_remote: torch.Tensor | None
    q_buffer: torch.Tensor | None
    copy_sources: list[torch.Tensor] | None
    copy_target: torch.Tensor | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--expected-revision")
    parser.add_argument("--expected-world-size", type=int, default=5)
    parser.add_argument("--expected-gpu-name-substring", default="A100")
    parser.add_argument(
        "--mode-order",
        nargs="+",
        choices=list(MODES),
        default=list(MODES),
    )
    parser.add_argument("--context-tokens", type=int, nargs="+", default=[1024])
    parser.add_argument(
        "--remote-fractions", type=float, nargs="+", default=[0.25, 0.75]
    )
    parser.add_argument("--target-load-repeats", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--copy-bytes-per-rank-step", type=int, nargs="+", default=[835584]
    )
    parser.add_argument("--attention-layers-per-step", type=int, default=48)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--measured-steps", type=int, default=64)
    parser.add_argument("--background-gemm-size", type=int, default=4096)
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


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def command_arguments(args: argparse.Namespace) -> dict[str, Any]:
    result = dict(vars(args))
    for name, value in result.items():
        if isinstance(value, Path):
            result[name] = str(value)
    return result


def validate_static(
    args: argparse.Namespace,
    cases: list[Any],
    revision: str,
) -> dict[str, Any]:
    if args.expected_revision and revision != args.expected_revision:
        raise RuntimeError(f"revision {revision} != expected {args.expected_revision}")
    if args.expected_world_size != args.target_tp_size + 1:
        raise ValueError("world size must be one TP1 plus every TP4 rank")
    if args.attention_layers_per_step <= 0:
        raise ValueError("attention layers per step must be positive")
    if args.warmup_steps <= 0 or args.measured_steps <= 0:
        raise ValueError("warmup and measured steps must be positive")
    if args.background_gemm_size <= 0:
        raise ValueError("background GEMM size must be positive")
    if args.num_layers != args.attention_layers_per_step:
        raise ValueError("G3-J evidence requires all model layers per step")
    if len(args.mode_order) != len(MODES) or set(args.mode_order) != set(MODES):
        raise ValueError("mode order must contain every G3-J mode exactly once")
    dtype_bytes = torch.empty((), dtype=ra.torch_dtype(args.dtype)).element_size()
    if any(case.copy_bytes_per_rank_step % dtype_bytes for case in cases):
        raise ValueError("copy byte counts must align to complete dtype elements")
    return {
        "format_version": 1,
        "status": "VALID",
        "revision": revision,
        "evidence_class": "GPU_SYNTHETIC_FULL_LAYER_BRIDGE_INTERFERENCE",
        "factorial_cells": len(cases),
        "treatment_modes": list(MODES),
        "mode_order": args.mode_order,
        "expected_rows": len(cases) * len(MODES),
        "expected_step_rows": len(cases) * len(MODES) * args.measured_steps,
        "paired_control_executions": (
            len(cases) * len(MODES) * args.measured_steps * 2
        ),
        "evidence_boundary": (
            "Real CUDA attention arithmetic, NCCL query/statistics traffic, "
            "full-Qwen-geometry KV copy, and concurrent CUDA GEMM controls. "
            "Q/K/V and target work are synthetic; this is not online vLLM TPOT, "
            "P99, or goodput evidence."
        ),
    }


def allocate_target_load(
    args: argparse.Namespace,
    rank: int,
    dtype: torch.dtype,
) -> TargetLoad:
    if rank == 0:
        return TargetLoad(None, None, None, None)
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + rank)
    size = args.background_gemm_size
    left = torch.randn((size, size), generator=generator, dtype=dtype, device=device)
    right = torch.randn_like(left)
    return TargetLoad(
        left=left,
        right=right,
        output=torch.empty_like(left),
        stream=torch.cuda.Stream(device=device),
    )


def launch_target_load(
    target: TargetLoad,
    repeats: int,
) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    if any(value is None for value in (target.left, target.right, target.output)):
        raise RuntimeError("target load tensors are missing")
    if target.stream is None:
        raise RuntimeError("target load stream is missing")
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(target.stream):
        started.record(target.stream)
        for _ in range(repeats):
            torch.mm(target.left, target.right, out=target.output)
        finished.record(target.stream)
    return started, finished


def reduce_target_load_ms(
    rank: int,
    started: torch.cuda.Event | None,
    finished: torch.cuda.Event | None,
) -> float | None:
    load_ms = 0.0
    if rank != 0:
        if started is None or finished is None:
            raise RuntimeError("target events are missing")
        finished.synchronize()
        load_ms = started.elapsed_time(finished)
    value = torch.tensor(load_ms, dtype=torch.float32, device="cuda")
    dist.reduce(value, dst=0, op=dist.ReduceOp.MAX)
    return float(value.item()) if rank == 0 else None


def run_control(
    *,
    rank: int,
    target_load: TargetLoad,
    load_repeats: int,
) -> float | None:
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    started = finished = None
    if rank != 0:
        started, finished = launch_target_load(target_load, load_repeats)
    return reduce_target_load_ms(rank, started, finished)


def allocate_case_tensors(
    *,
    args: argparse.Namespace,
    case: Any,
    cell_index: int,
    geometry: Any,
    rank: int,
    dtype: torch.dtype,
) -> tuple[CaseTensors, Any, int]:
    partition = ra.make_partition(
        case.context_tokens,
        case.remote_fraction,
        geometry.block_size,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    q = k_full = v_full = k_local = v_local = None
    k_remote = v_remote = q_buffer = None
    if rank == 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + cell_index * 1009)
        q = torch.randn(
            (geometry.num_query_heads, geometry.head_dim),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        k_full = torch.randn(
            (geometry.num_kv_heads, case.context_tokens, geometry.head_dim),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        v_full = torch.randn(
            k_full.shape,
            generator=generator,
            dtype=dtype,
            device=device,
        )
        k_local = k_full[:, : partition.local_tokens].contiguous()
        v_local = v_full[:, : partition.local_tokens].contiguous()
    else:
        remote_shape = (
            geometry.kv_heads_per_rank,
            partition.remote_tokens,
            geometry.head_dim,
        )
        k_remote = torch.empty(remote_shape, dtype=dtype, device=device)
        v_remote = torch.empty_like(k_remote)
        q_buffer = torch.empty(
            (geometry.query_heads_per_rank, geometry.head_dim),
            dtype=dtype,
            device=device,
        )

    _, staged_bytes = ra.stage_remote_kv(
        rank=rank,
        geometry=geometry,
        k_full=k_full,
        v_full=v_full,
        k_remote=k_remote,
        v_remote=v_remote,
        local_tokens=partition.local_tokens,
    )
    elements = case.copy_bytes_per_rank_step // torch.empty(
        (), dtype=dtype
    ).element_size()
    if rank == 0:
        copy_sources = [
            torch.full((elements,), 0.25, dtype=dtype, device=device)
            for _ in range(geometry.target_tp_size)
        ]
        copy_target = None
    else:
        copy_sources = None
        copy_target = torch.empty(elements, dtype=dtype, device=device)
    tensors = CaseTensors(
        q,
        k_full,
        v_full,
        k_local,
        v_local,
        k_remote,
        v_remote,
        q_buffer,
        copy_sources,
        copy_target,
    )
    return tensors, partition, staged_bytes


def run_copy(rank: int, tensors: CaseTensors, target_tp_size: int) -> None:
    if rank == 0:
        if tensors.copy_sources is None:
            raise RuntimeError("copy sources are missing")
        ra.p2p_batch(
            [
                dist.P2POp(dist.isend, tensors.copy_sources[index], worker_rank)
                for index, worker_rank in enumerate(range(1, target_tp_size + 1))
            ]
        )
    else:
        if tensors.copy_target is None:
            raise RuntimeError("copy target is missing")
        ra.p2p_batch([dist.P2POp(dist.irecv, tensors.copy_target, 0)])


def run_attention_layer(
    *,
    rank: int,
    tensors: CaseTensors,
    geometry: Any,
    include_copy: bool,
) -> torch.Tensor | None:
    if rank == 0:
        return ra.coordinator_step(
            q=tensors.q,
            k_local=tensors.k_local,
            v_local=tensors.v_local,
            geometry=geometry,
            copy_buffers=tensors.copy_sources if include_copy else None,
        )
    ra.worker_step(
        q_buffer=tensors.q_buffer,
        k_remote=tensors.k_remote,
        v_remote=tensors.v_remote,
        copy_buffer=tensors.copy_target if include_copy else None,
        load_left=None,
        load_right=None,
        load_repeats=0,
    )
    return None


def run_treatment(
    *,
    args: argparse.Namespace,
    mode: str,
    rank: int,
    tensors: CaseTensors,
    geometry: Any,
    target_load: TargetLoad,
    load_repeats: int,
) -> tuple[float, float, torch.Tensor | None] | None:
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    started = finished = None
    if rank != 0:
        started, finished = launch_target_load(target_load, load_repeats)
    wall_started = time.perf_counter()
    output = None
    if mode == "COPY_ONLY":
        run_copy(rank, tensors, geometry.target_tp_size)
    else:
        for layer in range(args.attention_layers_per_step):
            output = run_attention_layer(
                rank=rank,
                tensors=tensors,
                geometry=geometry,
                include_copy=(mode == "JOINT" and layer == 0),
            )
    torch.cuda.synchronize()
    path_ms = (time.perf_counter() - wall_started) * 1000
    target_ms = reduce_target_load_ms(rank, started, finished)
    if rank != 0:
        return None
    if target_ms is None:
        raise RuntimeError("rank zero did not receive target load timing")
    return path_ms, target_ms, output


def copy_verified(rank: int, tensors: CaseTensors, mode: str) -> bool:
    valid = True
    if rank != 0 and mode in ("COPY_ONLY", "JOINT"):
        if tensors.copy_target is None:
            raise RuntimeError("copy target is missing")
        valid = bool(torch.all(tensors.copy_target == 0.25).item())
    value = torch.tensor(int(valid), dtype=torch.int32, device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return bool(value.item())


def run_mode(
    *,
    args: argparse.Namespace,
    case: Any,
    cell_index: int,
    mode: str,
    rank: int,
    tensors: CaseTensors,
    partition: Any,
    staged_bytes: int,
    geometry: Any,
    target_load: TargetLoad,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    for _ in range(args.warmup_steps):
        run_treatment(
            args=args,
            mode=mode,
            rank=rank,
            tensors=tensors,
            geometry=geometry,
            target_load=target_load,
            load_repeats=case.target_load_repeats,
        )

    records = []
    last_output = None
    for step in range(args.measured_steps):
        before = run_control(
            rank=rank,
            target_load=target_load,
            load_repeats=case.target_load_repeats,
        )
        treatment = run_treatment(
            args=args,
            mode=mode,
            rank=rank,
            tensors=tensors,
            geometry=geometry,
            target_load=target_load,
            load_repeats=case.target_load_repeats,
        )
        after = run_control(
            rank=rank,
            target_load=target_load,
            load_repeats=case.target_load_repeats,
        )
        if rank == 0:
            if before is None or after is None or treatment is None:
                raise RuntimeError("rank zero did not receive bracketed measurement")
            path_ms, target_ms, last_output = treatment
            control_ms = (before + after) / 2
            delta_ms = target_ms - control_ms
            records.append(
                {
                    "cell_index": cell_index,
                    "mode": mode,
                    **case.to_dict(),
                    "step": step,
                    "target_control_before_ms": before,
                    "target_control_after_ms": after,
                    "target_control_ms": control_ms,
                    "target_treatment_ms": target_ms,
                    "target_delta_ms": delta_ms,
                    "target_harm_ms": max(0.0, delta_ms),
                    "target_slowdown_frac": delta_ms / control_ms,
                    "bridge_path_ms": path_ms,
                }
            )

    verified = copy_verified(rank, tensors, mode)
    if rank != 0:
        return None
    attention_finite = attention_accurate = True
    max_abs = mean_abs = 0.0
    if mode in ("ATTENTION_ONLY", "JOINT"):
        reference = ra.full_attention(tensors.q, tensors.k_full, tensors.v_full)
        difference = (last_output - reference).abs()
        max_abs = float(difference.max().item())
        mean_abs = float(difference.mean().item())
        attention_finite = bool(torch.isfinite(last_output).all().item())
        attention_accurate = (
            max_abs <= args.max_abs_tolerance
            and mean_abs <= args.mean_abs_tolerance
        )
    controls = [record["target_control_ms"] for record in records]
    treatments = [record["target_treatment_ms"] for record in records]
    deltas = [record["target_delta_ms"] for record in records]
    harms = [record["target_harm_ms"] for record in records]
    slowdowns = [record["target_slowdown_frac"] for record in records]
    paths = [record["bridge_path_ms"] for record in records]
    expected_staged_bytes = (
        ra.one_layer_rank_transfer_bytes(partition, geometry)
        * geometry.target_tp_size
    )
    status = (
        "PASS"
        if verified
        and attention_finite
        and attention_accurate
        and staged_bytes == expected_staged_bytes
        else "FAIL"
    )
    copy_bytes = (
        case.copy_bytes_per_rank_step * geometry.target_tp_size
        if mode in ("COPY_ONLY", "JOINT")
        else 0
    )
    summary = {
        "cell_index": cell_index,
        "mode": mode,
        **case.to_dict(),
        **partition.to_dict(),
        "status": status,
        "copy_verified": verified,
        "attention_finite": attention_finite,
        "attention_accurate": attention_accurate,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "staged_remote_kv_bytes": staged_bytes,
        "expected_staged_remote_kv_bytes": expected_staged_bytes,
        "attention_layers_per_step": args.attention_layers_per_step,
        "actual_copy_bytes_all_ranks_step": copy_bytes,
        "target_control_p50_ms": statistics.median(controls),
        "target_treatment_p50_ms": statistics.median(treatments),
        "target_signed_delta_ms": sum(deltas),
        "target_harm_ms": sum(harms),
        "target_signed_slowdown_frac": sum(deltas) / sum(controls),
        "target_step_slowdown_p50_frac": statistics.median(slowdowns),
        "target_step_slowdown_p95_frac": percentile(slowdowns, 0.95),
        "bridge_path_p50_ms": statistics.median(paths),
        "bridge_path_p95_ms": percentile(paths, 0.95),
        "query_wire_bytes_per_step": (
            ra.one_layer_query_wire_bytes(geometry)
            * args.attention_layers_per_step
            if mode in ("ATTENTION_ONLY", "JOINT")
            else 0
        ),
        "statistics_wire_bytes_per_step": (
            ra.one_layer_statistics_wire_bytes(geometry)
            * args.attention_layers_per_step
            if mode in ("ATTENTION_ONLY", "JOINT")
            else 0
        ),
        "evidence_boundary": "G3J_SYNTHETIC_FULL_LAYER_FACTORIAL_INTERFERENCE",
    }
    return summary, records


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cases = build_cases(
        context_tokens=args.context_tokens,
        remote_fractions=args.remote_fractions,
        target_load_repeats=args.target_load_repeats,
        copy_bytes_per_rank_step=args.copy_bytes_per_rank_step,
    )
    revision = git_revision()
    preflight = validate_static(args, cases, revision)
    if args.validate_only:
        print(json.dumps(preflight, indent=2))
        return
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("G3-J requires CUDA and NCCL")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError("launch G3-J with torchrun")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"world size {world_size} != expected {args.expected_world_size}"
        )
    ra.setup_output(args.out_dir, rank)
    inventory = ra.gpu_inventory(rank, world_size)
    dtype = ra.torch_dtype(args.dtype)
    geometry = RemoteAttentionGeometry(
        num_layers=args.num_layers,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        target_tp_size=args.target_tp_size,
        block_size=args.block_size,
        dtype_bytes=torch.empty((), dtype=dtype).element_size(),
    )
    geometry.validate()
    target_load = allocate_target_load(args, rank, dtype)
    rows = []
    step_rows = []
    for cell_index, case in enumerate(cases, start=1):
        tensors, partition, staged_bytes = allocate_case_tensors(
            args=args,
            case=case,
            cell_index=cell_index,
            geometry=geometry,
            rank=rank,
            dtype=dtype,
        )
        for mode in args.mode_order:
            result = run_mode(
                args=args,
                case=case,
                cell_index=cell_index,
                mode=mode,
                rank=rank,
                tensors=tensors,
                partition=partition,
                staged_bytes=staged_bytes,
                geometry=geometry,
                target_load=target_load,
            )
            if rank == 0 and result is not None:
                summary, steps = result
                rows.append(summary)
                step_rows.extend(steps)
                print(json.dumps(summary, sort_keys=True), flush=True)
        del tensors

    acceptance = None
    if rank == 0:
        acceptance = validate_results(
            rows,
            step_rows,
            expected_cells=len(cases),
            measured_steps=args.measured_steps,
        )
        mismatches = [
            item
            for item in inventory
            if args.expected_gpu_name_substring.lower()
            not in str(item["device_name"]).lower()
        ]
        if mismatches:
            acceptance["errors"].append("GPU inventory does not match expectation")
            acceptance["status"] = "FAIL"
        interactions = factorial_summary(rows)
        provenance = {
            **preflight,
            "command_arguments": command_arguments(args),
            "world_size": world_size,
            "gpu_inventory": inventory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "source_sha256": {
                "runner": file_sha256(Path(__file__).resolve()),
                "protocol": file_sha256(protocol_path),
                "remote_attention_runner": file_sha256(ra_path),
            },
        }
        write_csv(rows, args.out_dir / "measurements.csv")
        write_csv(step_rows, args.out_dir / "step_measurements.csv")
        write_csv(interactions, args.out_dir / "factorial_interactions.csv")
        (args.out_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / "acceptance.json").write_text(
            json.dumps(acceptance, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(acceptance, indent=2), flush=True)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.destroy_process_group()
    if rank == 0 and acceptance is not None and acceptance["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
