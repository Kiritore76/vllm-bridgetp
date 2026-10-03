#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare solo and burst TP4 TTFT with the frozen A1D target workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp.run_phase9_cap0_calibration import (
    server_command, start_process, stop_processes, wait_healthy,
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preflight(args: argparse.Namespace) -> dict[str, object]:
    if os.name == "nt":
        raise RuntimeError("this diagnostic must run on the A100 Linux server")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True,
    ).strip()
    if revision != args.expected_revision or branch != args.expected_branch or status:
        raise RuntimeError("HEAD, branch, or worktree differs; no GPU run")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    names = [name.strip() for name in names]
    if names != [args.expected_gpu_name] * args.expected_gpu_count:
        raise RuntimeError(f"GPU inventory differs: {names}")
    inputs = {
        "model_config": (args.model_path / "config.json", args.expected_model_config_sha256),
        "base_manifest": (args.base_target_manifest, args.expected_base_sha256),
    }
    for label, (path, expected) in inputs.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"{label} path or SHA-256 differs: {path}")
    return {
        "revision": revision, "branch": branch,
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "gpu_names": names,
        "input_paths": {label: str(path.resolve()) for label, (path, _) in inputs.items()},
        "input_sha256": {label: sha for label, (_, sha) in inputs.items()},
    }


def run_probe(args: argparse.Namespace) -> None:
    provenance = preflight(args)
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as error:
            raise RuntimeError(f"target port {args.port} is unavailable") from error
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    base = json.loads(args.base_target_manifest.read_text(encoding="utf-8"))
    jobs = base.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 4 or any(
        job.get("pool") != "target" for job in jobs
    ):
        raise ValueError("expected exactly four frozen target jobs")
    for job in jobs:
        request = job["request"]
        if len(request["prompt"]) != 2048 or request["max_tokens"] != 1024:
            raise ValueError("target prompt or output length differs from the baseline")
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "preflight.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
    env["OMP_NUM_THREADS"] = "1"
    process = start_process(
        "target_tp4", server_command(args, 4, args.port), env,
        args.out_dir / "target_tp4.log",
    )
    try:
        wait_healthy(f"http://127.0.0.1:{args.port}", process, 900)
        summaries = {}
        for name, selected in (
            ("warmup", jobs[:1]), ("solo_before", jobs[:1]),
            ("burst", jobs), ("solo_after", jobs[:1]),
        ):
            manifest = {
                "format_version": 1,
                "scenario": f"TP4 TTFT diagnostic {name}",
                "jobs": [dict(job, job_id=f"{name}_{job['job_id']}",
                              start_after_s=(index * 0.02))
                         for index, job in enumerate(selected)],
            }
            path = args.out_dir / f"{name}_manifest.json"
            path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            out = args.out_dir / name
            command = [
                sys.executable, "tools/bridge_tp/run_phase9_capacity_background.py",
                "--manifest", str(path), "--out-dir", str(out),
                "--target-url", f"http://127.0.0.1:{args.port}",
                "--request-timeout-s", "180",
            ]
            subprocess.run(command, cwd=REPO, env=env, check=True, timeout=240)
            rows = json.loads((out / "background_summary.json").read_text(
                encoding="utf-8"
            ))["results"]
            if any(row["status"] != "COMPLETED" for row in rows):
                raise RuntimeError(f"{name}: a target job did not complete")
            summaries[name] = [
                {"job_id": row["job_id"], "ttft_ms": row["ttft_ms"],
                 "tpot_p50_ms": row["tpot_p50_ms"]} for row in rows
            ]
            print(json.dumps({name: summaries[name]}), flush=True)
        (args.out_dir / "ttft_comparison.json").write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8",
        )
    finally:
        stop_processes([process])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-target-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-model-config-sha256", required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    run_probe(parser.parse_args())


if __name__ == "__main__":
    main()
