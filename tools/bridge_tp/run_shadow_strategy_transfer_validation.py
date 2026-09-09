# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure real five-GPU transfer schedules for the two Shadow strategies.

Global rank zero represents the TP1 source. Ranks one through four represent
the TP4 target. The benchmark sends the full-Qwen-geometry byte volume over
NCCL, verifies every payload before ACK, and measures the history backlog left
for Bridge. It is a G3 transfer microbenchmark, not online vLLM execution.
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
protocol_path = REPO_ROOT / "vllm" / "bridge_tp" / "shadow_transfer_protocol.py"
protocol_spec = importlib.util.spec_from_file_location(
    "_bridgetp_shadow_transfer_protocol", protocol_path
)
if protocol_spec is None or protocol_spec.loader is None:
    raise RuntimeError(f"cannot load protocol module from {protocol_path}")
protocol_module = importlib.util.module_from_spec(protocol_spec)
sys.modules[protocol_spec.name] = protocol_module
protocol_spec.loader.exec_module(protocol_module)
TransferUnit = protocol_module.TransferUnit
build_shadow_transfer_plan = protocol_module.build_shadow_transfer_plan
group_units_by_step = protocol_module.group_units_by_step


@dataclass(frozen=True)
class ExperimentCase:
    history_tokens: int
    shadow_steps: int
    target_load_repeats: int
    target_load_profile: str
    outcome: str


@dataclass
class PayloadBuffers:
    source_new: list[torch.Tensor] | None
    source_history: list[torch.Tensor] | None
    target_new: torch.Tensor | None
    target_history: torch.Tensor | None
    source_acks: list[torch.Tensor] | None
    target_ack: torch.Tensor | None
    source_done: list[torch.Tensor] | None
    target_done: torch.Tensor | None


@dataclass
class TargetLoad:
    left: torch.Tensor | None
    right: torch.Tensor | None
    output: torch.Tensor | None
    stream: torch.cuda.Stream | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--expected-revision")
    parser.add_argument("--expected-world-size", type=int, default=5)
    parser.add_argument("--expected-gpu-name-substring", default="A100")
    parser.add_argument(
        "--strategy-order",
        nargs="+",
        choices=["S_NEW", "S_NEW_OLD"],
        default=["S_NEW", "S_NEW_OLD"],
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        choices=["CANCEL", "COMMIT"],
        default=["CANCEL", "COMMIT"],
    )
    parser.add_argument("--history-tokens", type=int, nargs="+", default=[1024])
    parser.add_argument("--shadow-steps", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--target-load-repeats", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument(
        "--target-load-profiles",
        nargs="+",
        choices=[
            "CONSTANT",
            "STEP_UP_BRIDGE",
            "STEP_DOWN_BRIDGE",
            "PULSE",
            "OSCILLATE",
        ],
        default=["CONSTANT"],
        help="Phase-local TP4 load trajectories; repeats is the peak intensity.",
    )
    parser.add_argument("--history-blocks-per-shadow-step", type=int, default=1)
    parser.add_argument("--background-gemm-size", type=int, default=4096)
    parser.add_argument("--transport-warmup-steps", type=int, default=3)
    parser.add_argument("--strategy-warmup-steps", type=int, default=5)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--num-layers", type=int, default=48)
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


def command_arguments_for_provenance(
    args: argparse.Namespace,
) -> dict[str, Any]:
    arguments = dict(vars(args))
    for name, value in arguments.items():
        if isinstance(value, Path):
            arguments[name] = str(value)
    return arguments


def torch_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


def make_cases(args: argparse.Namespace) -> list[ExperimentCase]:
    cases = [
        ExperimentCase(*values)
        for values in itertools.product(
            args.history_tokens,
            args.shadow_steps,
            args.target_load_repeats,
            getattr(args, "target_load_profiles", ["CONSTANT"]),
            args.outcomes,
        )
    ]
    for case in cases:
        if case.history_tokens <= 0 or case.history_tokens % args.block_size:
            raise ValueError("history tokens must be positive and block aligned")
        if case.shadow_steps <= 0:
            raise ValueError("Shadow steps must be positive")
        if case.target_load_repeats < 0:
            raise ValueError("target load repeats cannot be negative")
        scheduled_load_repeats(
            case.target_load_profile,
            case.target_load_repeats,
            "SHADOW",
            0,
        )
    if len(args.strategy_order) != 2 or set(args.strategy_order) != {
        "S_NEW",
        "S_NEW_OLD",
    }:
        raise ValueError("strategy order must contain S_NEW and S_NEW_OLD exactly once")
    return cases


def scheduled_load_repeats(
    profile: str,
    peak_repeats: int,
    phase: str,
    step: int,
) -> int:
    """Return the causal TP4 load intensity for one phase-local step."""
    if peak_repeats < 0 or step < 0:
        raise ValueError("load repeats and step must be non-negative")
    if phase not in ("SHADOW", "BRIDGE"):
        raise ValueError(f"unknown transfer phase {phase!r}")
    if profile == "CONSTANT":
        return peak_repeats
    if profile == "STEP_UP_BRIDGE":
        return peak_repeats if phase == "BRIDGE" else 0
    if profile == "STEP_DOWN_BRIDGE":
        return peak_repeats if phase == "SHADOW" else 0
    if profile == "PULSE":
        return peak_repeats if step % 4 == 0 else 0
    if profile == "OSCILLATE":
        return peak_repeats if step % 2 == 0 else 0
    raise ValueError(f"unknown target load profile {profile!r}")


def elements_per_rank_per_token(args: argparse.Namespace) -> int:
    if args.num_kv_heads % args.target_tp_size:
        raise ValueError("KV heads must divide evenly over TP4 ranks")
    kv_heads_per_rank = args.num_kv_heads // args.target_tp_size
    return args.num_layers * 2 * kv_heads_per_rank * args.head_dim


def aggregate_bytes_for_tokens(args: argparse.Namespace, tokens: int) -> int:
    element_bytes = torch.empty((), dtype=torch_dtype(args.dtype)).element_size()
    return (
        elements_per_rank_per_token(args) * tokens * element_bytes * args.target_tp_size
    )


def projected_case_bytes(args: argparse.Namespace, plan: Any) -> int:
    units = plan.shadow_units + plan.bridge_units
    return sum(aggregate_bytes_for_tokens(args, unit.tokens) for unit in units)


def validate_static_inputs(
    args: argparse.Namespace,
    cases: list[ExperimentCase],
    revision: str,
) -> dict[str, Any]:
    if args.expected_revision and revision != args.expected_revision:
        raise RuntimeError(f"revision {revision} != expected {args.expected_revision}")
    if args.expected_world_size != args.target_tp_size + 1:
        raise ValueError("world size must be one TP1 source plus all TP4 ranks")
    if args.background_gemm_size <= 0 or args.transport_warmup_steps < 0:
        raise ValueError("GEMM size must be positive and warmup cannot be negative")
    if args.strategy_warmup_steps <= 0:
        raise ValueError("every measured strategy requires positive joint warmup")
    if args.history_blocks_per_shadow_step <= 0:
        raise ValueError("history blocks per Shadow step must be positive")

    plans = []
    total_bytes = 0
    measured_transfer_steps = 0
    for case in cases:
        for strategy in args.strategy_order:
            plan = build_shadow_transfer_plan(
                strategy=strategy,
                outcome=case.outcome,
                history_tokens=case.history_tokens,
                shadow_steps=case.shadow_steps,
                block_size=args.block_size,
                history_blocks_per_shadow_step=(args.history_blocks_per_shadow_step),
            )
            plans.append(plan)
            total_bytes += projected_case_bytes(args, plan)
            measured_transfer_steps += plan.shadow_steps + plan.bridge_steps
    return {
        "format_version": 1,
        "status": "VALID",
        "revision": revision,
        "evidence_class": "GPU_SYNTHETIC_FULL_GEOMETRY_KV_TRANSFER",
        "case_pairs": len(cases),
        "recorded_rows": len(plans),
        "strategy_order": args.strategy_order,
        "aggregate_bytes_per_new_token": aggregate_bytes_for_tokens(args, 1),
        "aggregate_bytes_per_history_block": aggregate_bytes_for_tokens(
            args, args.block_size
        ),
        "projected_total_transfer_bytes": total_bytes,
        "measured_transfer_steps": measured_transfer_steps,
        "paired_target_control_steps": measured_transfer_steps * 2,
        "evidence_boundary": (
            "Real NCCL payloads and bracketed CUDA GEMM controls with Qwen KV "
            "byte geometry. Payload values are synthetic and buffers are reused; "
            "this does not execute online vLLM, release real paged-KV blocks, "
            "measure native request SLOs, or run Bridge remote attention."
        ),
    }


def setup_output(path: Path, rank: int) -> None:
    error = [None]
    if rank == 0:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            error[0] = str(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"cannot create output directory: {error[0]}")


def gpu_barrier() -> None:
    dist.barrier(device_ids=[torch.cuda.current_device()])


def p2p_batch(operations: list[dist.P2POp]) -> None:
    requests = dist.batch_isend_irecv(operations)
    for request in requests:
        request.wait()


def allocate_payload_buffers(
    args: argparse.Namespace,
    rank: int,
    dtype: torch.dtype,
) -> PayloadBuffers:
    device = torch.device("cuda", torch.cuda.current_device())
    per_token = elements_per_rank_per_token(args)
    new_elements = per_token
    history_elements = per_token * args.block_size
    if rank == 0:
        return PayloadBuffers(
            source_new=[
                torch.empty(new_elements, dtype=dtype, device=device)
                for _ in range(args.target_tp_size)
            ],
            source_history=[
                torch.empty(history_elements, dtype=dtype, device=device)
                for _ in range(args.target_tp_size)
            ],
            target_new=None,
            target_history=None,
            source_acks=[
                torch.empty(1, dtype=torch.int32, device=device)
                for _ in range(args.target_tp_size)
            ],
            target_ack=None,
            source_done=[
                torch.empty(1, dtype=torch.int32, device=device)
                for _ in range(args.target_tp_size)
            ],
            target_done=None,
        )
    return PayloadBuffers(
        source_new=None,
        source_history=None,
        target_new=torch.empty(new_elements, dtype=dtype, device=device),
        target_history=torch.empty(history_elements, dtype=dtype, device=device),
        source_acks=None,
        target_ack=torch.empty(1, dtype=torch.int32, device=device),
        source_done=None,
        target_done=torch.ones(1, dtype=torch.int32, device=device),
    )


def allocate_target_load(
    args: argparse.Namespace,
    rank: int,
    dtype: torch.dtype,
) -> TargetLoad:
    if rank == 0:
        return TargetLoad(None, None, None, None)
    size = args.background_gemm_size
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device)
    generator.manual_seed(20260909 + rank)
    left = torch.randn((size, size), generator=generator, dtype=dtype, device=device)
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    return TargetLoad(left, right, output, torch.cuda.Stream(device=device))


def payload_value(unit: Any) -> float:
    base = (unit.token_start % 97 + 1) / 128
    return base if unit.kind == "NEW" else -base


def transfer_and_verify(
    *,
    args: argparse.Namespace,
    unit: Any,
    rank: int,
    buffers: PayloadBuffers,
) -> tuple[float, int, bool]:
    elements = elements_per_rank_per_token(args) * unit.tokens
    expected_value = payload_value(unit)
    started = time.perf_counter()
    if rank == 0:
        sources = buffers.source_new if unit.kind == "NEW" else buffers.source_history
        if sources is None or buffers.source_acks is None:
            raise RuntimeError("source payload buffers are missing")
        for source in sources:
            source[:elements].fill_(expected_value)
        torch.cuda.synchronize()
        started = time.perf_counter()
        p2p_batch(
            [
                dist.P2POp(dist.isend, sources[index][:elements], worker_rank)
                for index, worker_rank in enumerate(range(1, args.target_tp_size + 1))
            ]
        )
        p2p_batch(
            [
                dist.P2POp(dist.irecv, buffers.source_acks[index], worker_rank)
                for index, worker_rank in enumerate(range(1, args.target_tp_size + 1))
            ]
        )
        valid = all(int(ack.item()) == 1 for ack in buffers.source_acks)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return elapsed_ms, aggregate_bytes_for_tokens(args, unit.tokens), valid

    target = buffers.target_new if unit.kind == "NEW" else buffers.target_history
    if target is None or buffers.target_ack is None:
        raise RuntimeError("target payload buffers are missing")
    p2p_batch([dist.P2POp(dist.irecv, target[:elements], 0)])
    valid = bool(torch.all(target[:elements] == expected_value).item())
    buffers.target_ack.fill_(int(valid))
    p2p_batch([dist.P2POp(dist.isend, buffers.target_ack, 0)])
    return 0.0, aggregate_bytes_for_tokens(args, unit.tokens), valid


def launch_target_load(
    target_load: TargetLoad,
    repeats: int,
) -> tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
    if repeats == 0:
        return None, None
    if (
        target_load.stream is None
        or target_load.left is None
        or target_load.right is None
        or target_load.output is None
    ):
        raise RuntimeError("target load tensors are missing")
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(target_load.stream):
        started.record(target_load.stream)
        for _ in range(repeats):
            torch.mm(
                target_load.left,
                target_load.right,
                out=target_load.output,
            )
        finished.record(target_load.stream)
    return started, finished


def run_logical_step(
    *,
    args: argparse.Namespace,
    units: list[Any],
    rank: int,
    buffers: PayloadBuffers,
    target_load: TargetLoad,
    load_repeats: int,
) -> dict[str, Any] | None:
    torch.cuda.synchronize()
    gpu_barrier()
    step_started = time.perf_counter()
    load_started = load_finished = None
    if rank != 0:
        load_started, load_finished = launch_target_load(target_load, load_repeats)

    unit_latencies: dict[str, list[float]] = {"NEW": [], "HISTORY": []}
    actual_bytes = 0
    all_verified = True
    for unit in units:
        latency_ms, transferred_bytes, verified = transfer_and_verify(
            args=args,
            unit=unit,
            rank=rank,
            buffers=buffers,
        )
        if rank == 0:
            unit_latencies[unit.kind].append(latency_ms)
            actual_bytes += transferred_bytes
            all_verified = all_verified and verified

    load_ms = 0.0
    if rank != 0:
        if load_finished is not None and load_started is not None:
            load_finished.synchronize()
            load_ms = load_started.elapsed_time(load_finished)
        if buffers.target_done is None:
            raise RuntimeError("target completion buffer is missing")
        p2p_batch([dist.P2POp(dist.isend, buffers.target_done, 0)])
    else:
        if buffers.source_done is None:
            raise RuntimeError("source completion buffers are missing")
        p2p_batch(
            [
                dist.P2POp(dist.irecv, buffers.source_done[index], worker_rank)
                for index, worker_rank in enumerate(range(1, args.target_tp_size + 1))
            ]
        )
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - step_started) * 1000
    load_tensor = torch.tensor(load_ms, dtype=torch.float32, device="cuda")
    dist.reduce(load_tensor, dst=0, op=dist.ReduceOp.MAX)
    if rank != 0:
        return None
    return {
        "phase": units[0].phase,
        "step": units[0].step,
        "new_ack_ms": sum(unit_latencies["NEW"]),
        "history_ack_ms": sum(unit_latencies["HISTORY"]),
        "step_ms": step_ms,
        "target_load_ms": float(load_tensor.item()),
        "actual_bytes": actual_bytes,
        "expected_bytes": sum(
            aggregate_bytes_for_tokens(args, unit.tokens) for unit in units
        ),
        "all_verified": all_verified,
    }


def run_target_control_step(
    *,
    rank: int,
    buffers: PayloadBuffers,
    target_load: TargetLoad,
    load_repeats: int,
) -> float | None:
    """Measure the same TP4 work without any Shadow/Bridge transfer."""
    torch.cuda.synchronize()
    gpu_barrier()
    load_started = load_finished = None
    if rank != 0:
        load_started, load_finished = launch_target_load(target_load, load_repeats)

    load_ms = 0.0
    if rank != 0:
        if load_started is not None and load_finished is not None:
            load_finished.synchronize()
            load_ms = load_started.elapsed_time(load_finished)
        if buffers.target_done is None:
            raise RuntimeError("target completion buffer is missing")
        p2p_batch([dist.P2POp(dist.isend, buffers.target_done, 0)])
    else:
        if buffers.source_done is None:
            raise RuntimeError("source completion buffers are missing")
        p2p_batch(
            [
                dist.P2POp(dist.irecv, buffers.source_done[index], worker_rank)
                for index, worker_rank in enumerate(
                    range(1, len(buffers.source_done) + 1)
                )
            ]
        )
    torch.cuda.synchronize()
    load_tensor = torch.tensor(load_ms, dtype=torch.float32, device="cuda")
    dist.reduce(load_tensor, dst=0, op=dist.ReduceOp.MAX)
    if rank != 0:
        return None
    return float(load_tensor.item())


def run_phase(
    *,
    args: argparse.Namespace,
    units: tuple[Any, ...],
    rank: int,
    buffers: PayloadBuffers,
    target_load: TargetLoad,
    peak_load_repeats: int,
    load_profile: str,
) -> list[dict[str, Any]]:
    records = []
    for step_units in group_units_by_step(units):
        phase = step_units[0].phase
        step = step_units[0].step
        load_repeats = scheduled_load_repeats(
            load_profile, peak_load_repeats, phase, step
        )
        control_before = run_target_control_step(
            rank=rank,
            buffers=buffers,
            target_load=target_load,
            load_repeats=load_repeats,
        )
        record = run_logical_step(
            args=args,
            units=step_units,
            rank=rank,
            buffers=buffers,
            target_load=target_load,
            load_repeats=load_repeats,
        )
        control_after = run_target_control_step(
            rank=rank,
            buffers=buffers,
            target_load=target_load,
            load_repeats=load_repeats,
        )
        if rank == 0 and record is not None:
            if control_before is None or control_after is None:
                raise RuntimeError("rank zero did not receive TP4 control timing")
            control_ms = (control_before + control_after) / 2
            delta_ms = record["target_load_ms"] - control_ms
            record.update(
                {
                    "scheduled_load_repeats": load_repeats,
                    "target_control_before_ms": control_before,
                    "target_control_after_ms": control_after,
                    "target_control_ms": control_ms,
                    "target_interference_delta_ms": delta_ms,
                    "target_interference_harm_ms": max(0.0, delta_ms),
                    "target_interference_slowdown_frac": (
                        delta_ms / control_ms if control_ms > 0 else None
                    ),
                }
            )
            records.append(record)
    return records


def warm_up_transport(
    *,
    args: argparse.Namespace,
    rank: int,
    buffers: PayloadBuffers,
    target_load: TargetLoad,
    steps: int,
    load_repeats: int,
) -> None:
    for step in range(steps):
        units = [
            TransferUnit("SHADOW", step, "NEW", step, step + 1),
            TransferUnit(
                "SHADOW",
                step,
                "HISTORY",
                args.block_size,
                args.block_size * 2,
            ),
        ]
        run_logical_step(
            args=args,
            units=units,
            rank=rank,
            buffers=buffers,
            target_load=target_load,
            load_repeats=load_repeats,
        )


def metric(values: list[float], quantile: float = 0.5) -> float:
    if not values:
        return 0.0
    if quantile == 0.5:
        return statistics.median(values)
    return percentile(values, quantile)


def run_strategy(
    *,
    args: argparse.Namespace,
    case: ExperimentCase,
    strategy: str,
    case_index: int,
    rank: int,
    buffers: PayloadBuffers,
    target_load: TargetLoad,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    plan = build_shadow_transfer_plan(
        strategy=strategy,
        outcome=case.outcome,
        history_tokens=case.history_tokens,
        shadow_steps=case.shadow_steps,
        block_size=args.block_size,
        history_blocks_per_shadow_step=args.history_blocks_per_shadow_step,
    )
    shadow_records = run_phase(
        args=args,
        units=plan.shadow_units,
        rank=rank,
        buffers=buffers,
        target_load=target_load,
        peak_load_repeats=case.target_load_repeats,
        load_profile=case.target_load_profile,
    )
    bridge_records = run_phase(
        args=args,
        units=plan.bridge_units,
        rank=rank,
        buffers=buffers,
        target_load=target_load,
        peak_load_repeats=case.target_load_repeats,
        load_profile=case.target_load_profile,
    )
    gpu_barrier()
    if rank != 0:
        return None

    records = shadow_records + bridge_records
    actual_bytes = sum(record["actual_bytes"] for record in records)
    expected_bytes = projected_case_bytes(args, plan)
    all_verified = all(record["all_verified"] for record in records)
    byte_exact = actual_bytes == expected_bytes and all(
        record["actual_bytes"] == record["expected_bytes"] for record in records
    )
    shadow_new = [record["new_ack_ms"] for record in shadow_records]
    shadow_history = [
        record["history_ack_ms"]
        for record in shadow_records
        if record["history_ack_ms"] > 0
    ]
    shadow_load = [record["target_load_ms"] for record in shadow_records]
    bridge_load = [record["target_load_ms"] for record in bridge_records]
    shadow_control = [record["target_control_ms"] for record in shadow_records]
    bridge_control = [record["target_control_ms"] for record in bridge_records]
    shadow_harm = [
        record["target_interference_harm_ms"] for record in shadow_records
    ]
    bridge_harm = [
        record["target_interference_harm_ms"] for record in bridge_records
    ]
    shadow_delta = [
        record["target_interference_delta_ms"] for record in shadow_records
    ]
    bridge_delta = [
        record["target_interference_delta_ms"] for record in bridge_records
    ]
    shadow_slowdown = [
        record["target_interference_slowdown_frac"]
        for record in shadow_records
        if record["target_interference_slowdown_frac"] is not None
    ]
    bridge_slowdown = [
        record["target_interference_slowdown_frac"]
        for record in bridge_records
        if record["target_interference_slowdown_frac"] is not None
    ]
    takeover_ready = case.outcome == "COMMIT" and all_verified and byte_exact
    status = "PASS" if all_verified and byte_exact else "FAIL"
    aggregate_new_bytes = aggregate_bytes_for_tokens(args, 1)
    aggregate_block_bytes = aggregate_bytes_for_tokens(args, args.block_size)
    summary = {
        "case_index": case_index,
        **asdict(case),
        "strategy": strategy,
        "status": status,
        "all_payloads_verified": all_verified,
        "actual_transfer_bytes": actual_bytes,
        "expected_transfer_bytes": expected_bytes,
        "aggregate_new_kv_bytes": aggregate_new_bytes,
        "aggregate_history_block_bytes": aggregate_block_bytes,
        "shadow_new_tokens": case.shadow_steps,
        "shadow_history_tokens": plan.history_tokens_copied_in_shadow,
        "shadow_transfer_bytes": sum(
            record["actual_bytes"] for record in shadow_records
        ),
        "shadow_transfer_driver_ms": sum(
            record["step_ms"] for record in shadow_records
        ),
        "shadow_new_ack_p50_ms": metric(shadow_new),
        "shadow_new_ack_p95_ms": metric(shadow_new, 0.95),
        "shadow_history_ack_p50_ms": metric(shadow_history),
        "shadow_history_ack_p95_ms": metric(shadow_history, 0.95),
        "shadow_target_load_p50_ms": metric(shadow_load),
        "shadow_target_load_p95_ms": metric(shadow_load, 0.95),
        "shadow_target_control_p50_ms": metric(shadow_control),
        "shadow_target_control_p95_ms": metric(shadow_control, 0.95),
        "shadow_target_interference_harm_ms": sum(shadow_harm),
        "shadow_target_interference_signed_delta_ms": sum(shadow_delta),
        "shadow_target_interference_harm_p95_ms": metric(shadow_harm, 0.95),
        "shadow_target_interference_slowdown_p50_frac": metric(shadow_slowdown),
        "shadow_target_interference_slowdown_p95_frac": metric(
            shadow_slowdown, 0.95
        ),
        "shadow_target_interference_observed_steps": len(shadow_slowdown),
        "history_backlog_after_shadow_tokens": (
            plan.bridge_entry_history_backlog_tokens
        ),
        "history_backlog_after_shadow_bytes": aggregate_bytes_for_tokens(
            args, plan.bridge_entry_history_backlog_tokens
        ),
        "bridge_entry_history_backlog_tokens": (
            plan.bridge_entry_history_backlog_tokens
            if case.outcome == "COMMIT"
            else None
        ),
        "bridge_entry_history_backlog_bytes": (
            aggregate_bytes_for_tokens(args, plan.bridge_entry_history_backlog_tokens)
            if case.outcome == "COMMIT"
            else None
        ),
        "bridge_steps": plan.bridge_steps,
        "bridge_transfer_bytes": sum(
            record["actual_bytes"] for record in bridge_records
        ),
        "bridge_catchup_ms": sum(record["step_ms"] for record in bridge_records),
        "bridge_target_load_p50_ms": metric(bridge_load),
        "bridge_target_load_p95_ms": metric(bridge_load, 0.95),
        "bridge_target_control_p50_ms": metric(bridge_control),
        "bridge_target_control_p95_ms": metric(bridge_control, 0.95),
        "bridge_target_interference_harm_ms": sum(bridge_harm),
        "bridge_target_interference_signed_delta_ms": sum(bridge_delta),
        "bridge_target_interference_harm_p95_ms": metric(bridge_harm, 0.95),
        "bridge_target_interference_slowdown_p50_frac": metric(bridge_slowdown),
        "bridge_target_interference_slowdown_p95_frac": metric(
            bridge_slowdown, 0.95
        ),
        "bridge_target_interference_observed_steps": len(bridge_slowdown),
        "transfer_only_target_interference_harm_ms": sum(shadow_harm)
        + sum(bridge_harm),
        "transfer_only_target_interference_signed_delta_ms": sum(shadow_delta)
        + sum(bridge_delta),
        "cancel_wasted_history_bytes": (
            aggregate_bytes_for_tokens(args, plan.history_tokens_copied_in_shadow)
            if case.outcome == "CANCEL"
            else 0
        ),
        "cancel_wasted_total_bytes": (
            sum(record["actual_bytes"] for record in shadow_records)
            if case.outcome == "CANCEL"
            else 0
        ),
        "source_released_tokens_in_shadow": 0,
        "takeover_ready": takeover_ready,
        "evidence_boundary": "G3_TRANSFER_MICROBENCHMARK",
    }
    step_rows = [
        {
            "case_index": case_index,
            **asdict(case),
            "strategy": strategy,
            **record,
        }
        for record in records
    ]
    return summary, step_rows


def validate_rows(rows: list[dict[str, Any]], expected_rows: int) -> dict[str, Any]:
    errors = []
    if len(rows) != expected_rows:
        errors.append(f"recorded {len(rows)} rows, expected {expected_rows}")
    for row in rows:
        label = f"case {row.get('case_index')} {row.get('strategy')}"
        if row.get("status") != "PASS":
            errors.append(f"{label}: transfer validation failed")
        if not row.get("all_payloads_verified"):
            errors.append(f"{label}: a target rejected a payload")
        if row.get("actual_transfer_bytes") != row.get("expected_transfer_bytes"):
            errors.append(f"{label}: transfer byte accounting mismatch")
        if row.get("source_released_tokens_in_shadow") != 0:
            errors.append(f"{label}: Shadow released source KV")
        if row.get("strategy") == "S_NEW" and row.get("shadow_history_tokens") != 0:
            errors.append(f"{label}: S_NEW copied history during Shadow")
        if row.get("outcome") == "CANCEL":
            if row.get("bridge_steps") != 0 or row.get("takeover_ready"):
                errors.append(f"{label}: cancelled episode entered Bridge")
        elif not row.get("takeover_ready"):
            errors.append(f"{label}: committed episode did not drain its backlog")
    paired: dict[tuple[object, ...], set[object]] = {}
    for row in rows:
        key = (
            row.get("history_tokens"),
            row.get("shadow_steps"),
            row.get("target_load_repeats"),
            row.get("target_load_profile"),
            row.get("outcome"),
        )
        paired.setdefault(key, set()).add(row.get("strategy"))
    for key, strategies in paired.items():
        if strategies != {"S_NEW", "S_NEW_OLD"}:
            errors.append(f"cell {key}: missing paired Shadow strategy")
    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "expected_rows": expected_rows,
        "recorded_rows": len(rows),
        "errors": errors,
    }


def validate_step_rows(
    rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
) -> list[str]:
    """Fail closed if raw step telemetry cannot reproduce each summary."""
    errors = []
    expected_steps = sum(
        int(row["shadow_steps"]) + int(row["bridge_steps"]) for row in rows
    )
    if len(step_rows) != expected_steps:
        errors.append(f"recorded {len(step_rows)} step rows, expected {expected_steps}")
    for row in rows:
        case_steps = [
            step for step in step_rows if step["case_index"] == row["case_index"]
        ]
        if len(case_steps) != int(row["shadow_steps"]) + int(row["bridge_steps"]):
            errors.append(f"case {row['case_index']}: incomplete raw step telemetry")
        if sum(int(step["actual_bytes"]) for step in case_steps) != int(
            row["actual_transfer_bytes"]
        ):
            errors.append(f"case {row['case_index']}: raw step byte total mismatch")
        if not all(bool(step["all_verified"]) for step in case_steps):
            errors.append(f"case {row['case_index']}: raw step verification failed")
        shadow_harm = sum(
            float(step["target_interference_harm_ms"])
            for step in case_steps
            if step["phase"] == "SHADOW"
        )
        bridge_harm = sum(
            float(step["target_interference_harm_ms"])
            for step in case_steps
            if step["phase"] == "BRIDGE"
        )
        if not math.isclose(
            shadow_harm,
            float(row["shadow_target_interference_harm_ms"]),
            abs_tol=1e-9,
        ):
            errors.append(f"case {row['case_index']}: Shadow harm total mismatch")
        if not math.isclose(
            bridge_harm,
            float(row["bridge_target_interference_harm_ms"]),
            abs_tol=1e-9,
        ):
            errors.append(f"case {row['case_index']}: Bridge harm total mismatch")
        for step in case_steps:
            numeric = (
                "target_control_before_ms",
                "target_control_after_ms",
                "target_control_ms",
                "target_interference_delta_ms",
                "target_interference_harm_ms",
            )
            if not all(math.isfinite(float(step[name])) for name in numeric):
                errors.append(f"case {row['case_index']}: non-finite interference data")
                break
            if float(step["target_interference_harm_ms"]) < 0:
                errors.append(f"case {row['case_index']}: negative interference harm")
                break
            if int(step["scheduled_load_repeats"]) == 0 and (
                float(step["target_control_ms"]) != 0
                or float(step["target_load_ms"]) != 0
            ):
                errors.append(f"case {row['case_index']}: idle control was not idle")
                break
    return errors


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cases = make_cases(args)
    revision = git_revision()
    preflight = validate_static_inputs(args, cases, revision)
    if args.validate_only:
        print(json.dumps(preflight, indent=2))
        return
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("real five-GPU validation requires CUDA and NCCL")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError("launch with torchrun; LOCAL_RANK is missing")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"world size {world_size} != expected {args.expected_world_size}"
        )
    setup_output(args.out_dir, rank)
    inventory: list[dict[str, Any] | None] = [None] * world_size
    item = {
        "rank": rank,
        "device_index": local_rank,
        "device_name": torch.cuda.get_device_name(),
        "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
    }
    dist.all_gather_object(inventory, item)
    dtype = torch_dtype(args.dtype)
    buffers = allocate_payload_buffers(args, rank, dtype)
    target_load = allocate_target_load(args, rank, dtype)
    warm_up_transport(
        args=args,
        rank=rank,
        buffers=buffers,
        target_load=target_load,
        steps=args.transport_warmup_steps,
        load_repeats=0,
    )

    rows = []
    step_rows = []
    case_index = 0
    for case in cases:
        for strategy in args.strategy_order:
            # Condition both AB and BA arms immediately before measurement.
            # This prevents the first arm from uniquely paying cuBLAS/NCCL
            # initialization and GPU clock-ramp costs.
            warm_up_transport(
                args=args,
                rank=rank,
                buffers=buffers,
                target_load=target_load,
                steps=args.strategy_warmup_steps,
                load_repeats=case.target_load_repeats,
            )
            case_index += 1
            result = run_strategy(
                args=args,
                case=case,
                strategy=strategy,
                case_index=case_index,
                rank=rank,
                buffers=buffers,
                target_load=target_load,
            )
            if rank == 0 and result is not None:
                row, measured_steps = result
                rows.append(row)
                step_rows.extend(measured_steps)
                print(json.dumps(row, sort_keys=True), flush=True)

    acceptance = None
    if rank == 0:
        acceptance = validate_rows(rows, len(cases) * len(args.strategy_order))
        acceptance["recorded_step_rows"] = len(step_rows)
        acceptance["errors"].extend(validate_step_rows(rows, step_rows))
        mismatches = [
            entry
            for entry in inventory
            if entry is not None
            and args.expected_gpu_name_substring.lower()
            not in str(entry["device_name"]).lower()
        ]
        if mismatches:
            acceptance["errors"].append("GPU inventory does not match expectation")
        if acceptance["errors"]:
            acceptance["status"] = "FAIL"
        runner_path = Path(__file__).resolve()
        provenance = {
            **preflight,
            "command_arguments": command_arguments_for_provenance(args),
            "world_size": world_size,
            "gpu_inventory": inventory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "source_sha256": {
                "runner": file_sha256(runner_path),
                "protocol": file_sha256(protocol_path),
            },
        }
        write_csv(rows, args.out_dir / "measurements.csv")
        write_csv(step_rows, args.out_dir / "step_measurements.csv")
        (args.out_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / "acceptance.json").write_text(
            json.dumps(acceptance, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(acceptance, indent=2), flush=True)
    gpu_barrier()
    dist.destroy_process_group()
    if rank == 0 and acceptance is not None and acceptance["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
