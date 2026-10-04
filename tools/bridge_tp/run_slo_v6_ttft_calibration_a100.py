#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure isolated TP4 TTFT at seven frozen prompt lengths on A100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp.run_phase9_cap0_calibration import (  # noqa: E402
    server_command, start_process, stop_processes, wait_healthy,
)
LENGTHS = (128, 512, 1024, 2048, 4096, 6144, 7168)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take percentile of empty samples")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def schedule(samples_per_length: int, rounds: int, seed: int) -> list[list[int]]:
    if rounds < 2 or samples_per_length < 30 or samples_per_length % rounds:
        raise ValueError("require >=2 rounds and >=30 samples per length, divisible by rounds")
    per_round = samples_per_length // rounds
    result: list[list[int]] = []
    for round_index in range(rounds):
        order = [length for _ in range(per_round) for length in LENGTHS]
        random.Random(seed + round_index).shuffle(order)
        result.append(order)
    return result


def prompt_for_length(base: list[int], length: int) -> list[int]:
    if not base or any(isinstance(token, bool) or not isinstance(token, int)
                       or token < 0 for token in base):
        raise ValueError("base prompt must contain nonnegative token IDs")
    return (base * ((length + len(base) - 1) // len(base)))[:length]


def inspect_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[int]]:
    if os.name == "nt":
        raise RuntimeError("A100 Linux server required")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
    ).strip()
    if revision != args.expected_revision or branch != args.expected_branch or status:
        raise RuntimeError("HEAD, branch, or worktree differs; no GPU run")
    hostname = subprocess.check_output(["hostname"], text=True).strip()
    if hostname != args.expected_hostname:
        raise RuntimeError(f"hostname differs: {hostname}")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    names = [name.strip() for name in names]
    if names != [args.expected_gpu_name] * args.expected_gpu_count:
        raise RuntimeError(f"GPU inventory differs: {names}")
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    target_uuids = set(uuids[1:5])
    if any(line.split(",", 1)[0].strip() in target_uuids for line in active):
        raise RuntimeError("TP4 GPUs 1-4 already have compute processes; no run")
    inputs = {
        "model_config": (args.model_path / "config.json", args.expected_model_config_sha256),
        "base_manifest": (args.base_target_manifest, args.expected_base_sha256),
    }
    for label, (path, expected) in inputs.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"{label} path or SHA-256 differs: {path}")
    base = json.loads(args.base_target_manifest.read_text(encoding="utf-8"))
    jobs = base.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 4 or any(
        job.get("pool") != "target" for job in jobs
    ):
        raise ValueError("frozen A1D manifest must contain four target jobs")
    first = jobs[0]["request"]
    tokens = first["prompt"]
    if not isinstance(tokens, list) or len(tokens) != 2048:
        raise ValueError("frozen target prompt must be a 2048-token ID list")
    prompt_for_length(tokens, LENGTHS[-1])
    if first["model"] != "bridgetp-model":
        raise ValueError("frozen served model name differs")
    if args.max_model_len < LENGTHS[-1] + args.max_tokens:
        raise ValueError("longest prompt plus output exceeds max model length")
    return {
        "hostname": hostname, "revision": revision, "branch": branch,
        "gpu_names": names, "gpu_uuids": uuids,
        "model_path": str(args.model_path.resolve()),
        "base_manifest_path": str(args.base_target_manifest.resolve()),
        "model_config_sha256": args.expected_model_config_sha256,
        "base_manifest_sha256": args.expected_base_sha256,
        "base_prompt_sha256": hashlib.sha256(
            json.dumps(tokens, separators=(",", ":")).encode()
        ).hexdigest(),
        "samples_per_length": args.samples_per_length,
        "rounds": args.rounds, "seed": args.seed,
        "lengths": list(LENGTHS), "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "port": args.port,
        "server_command": server_command(args, 4, args.port),
    }, tokens


def one_request(
    *, base_url: str, base: list[int], length: int, max_tokens: int,
    request_id: str, timeout_s: float,
) -> dict[str, Any]:
    from vllm.bridge_tp.controller.online_io import post_streaming_completion
    from vllm.bridge_tp.controller.sampling_contract import freeze_strict_greedy_sampling

    token_times: list[float] = []
    payload = freeze_strict_greedy_sampling({
        "model": "bridgetp-model",
        "prompt": prompt_for_length(base, length),
        "max_tokens": max_tokens, "ignore_eos": True,
        "stream": True, "return_token_ids": True,
        "request_id": request_id,
    })
    result = post_streaming_completion(
        base_url, payload, timeout_s,
        lambda _index, _token_id, unix_s: token_times.append(unix_s),
    )
    if len(result["token_ids"]) != max_tokens or len(token_times) != max_tokens:
        raise RuntimeError(f"{request_id}: incomplete output")
    return {
        "request_id": request_id,
        "prompt_tokens": length,
        "request_started_unix_s": result["request_started_unix_s"],
        "first_token_unix_s": result["first_token_unix_s"],
        "completed_unix_s": result["completed_unix_s"],
        "ttft_ms": result["ttft_ms"],
        "e2e_ms": result["e2e_ms"],
        "token_times_unix_s": token_times,
        "output_tokens": len(result["token_ids"]),
        "finish_reason": result["finish_reason"],
    }


def summarize(rows: list[dict[str, Any]], rounds: int, samples_per_length: int) -> dict[str, Any]:
    anchors = []
    previous_p95 = 0.0
    for length in LENGTHS:
        selected = [row["ttft_ms"] for row in rows if row["prompt_tokens"] == length]
        if len(selected) != samples_per_length:
            raise RuntimeError(f"{length}: incomplete samples")
        raw_p95 = percentile(selected, 0.95)
        previous_p95 = max(previous_p95, raw_p95)
        anchors.append({
            "prompt_tokens": length,
            "samples": len(selected),
            "ttft_p50_ms": percentile(selected, 0.50),
            "ttft_p95_ms": raw_p95,
            "monotone_reference_p95_ms": previous_p95,
            "candidate_ttft_slo_ms": previous_p95 + 1000.0,
            "round_p95_ms": [
                percentile([
                    row["ttft_ms"] for row in rows
                    if row["round"] == round_index and row["prompt_tokens"] == length
                ], 0.95) for round_index in range(rounds)
            ],
        })
    return {
        "format_version": 1,
        "status": "CANDIDATE_UNREVIEWED_NOT_FROZEN",
        "reference": "native TP4 vLLM, prefix caching disabled, solo requests",
        "queue_allowance_ms": 1000.0,
        "interpolation": "piecewise linear after nondecreasing P95 envelope",
        "anchors": anchors,
    }


def run(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0 or args.request_timeout_s <= 0:
        raise ValueError("max tokens and request timeout must be positive")
    plan = schedule(args.samples_per_length, args.rounds, args.seed)
    provenance, base = inspect_inputs(args)
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as error:
            raise RuntimeError(f"target port {args.port} is unavailable") from error
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "preflight.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )
    (args.out_dir / "schedule.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8",
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
    env["OMP_NUM_THREADS"] = "1"
    rows: list[dict[str, Any]] = []
    with (args.out_dir / "samples.jsonl").open("w", encoding="utf-8") as output:
        for round_index, order in enumerate(plan):
            round_dir = args.out_dir / f"round_{round_index + 1:02d}"
            round_dir.mkdir()
            process = start_process(
                "target_tp4", server_command(args, 4, args.port), env,
                round_dir / "target_tp4.log",
            )
            try:
                url = f"http://127.0.0.1:{args.port}"
                wait_healthy(url, process, 900)
                for length in LENGTHS:
                    one_request(
                        base_url=url, base=base, length=length,
                        max_tokens=args.max_tokens,
                        request_id=f"warmup-r{round_index + 1}-p{length}",
                        timeout_s=args.request_timeout_s,
                    )
                for index, length in enumerate(order):
                    row = one_request(
                        base_url=url, base=base, length=length,
                        max_tokens=args.max_tokens,
                        request_id=f"r{round_index + 1:02d}-s{index + 1:03d}-p{length}",
                        timeout_s=args.request_timeout_s,
                    )
                    row["round"] = round_index
                    rows.append(row)
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")
                    output.flush()
                    print(
                        f"round {round_index + 1}/{args.rounds} "
                        f"sample {index + 1}/{len(order)} "
                        f"length={length} ttft_ms={row['ttft_ms']:.1f}",
                        flush=True,
                    )
            finally:
                stop_processes([process])
    summary = summarize(rows, args.rounds, args.samples_per_length)
    (args.out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-target-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--expected-model-config-sha256", required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    parser.add_argument("--samples-per-length", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
