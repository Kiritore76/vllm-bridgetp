#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect a workload-matched A100 TPOT pilot with managed API servers."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402

MODEL = Path(
    "/root/autodl-tmp/models/models/"
    "Qwen--Qwen2.5-14B-Instruct/snapshots/master"
)
BASE = Path(
    "/root/autodl-tmp/bridgetp/a1d_manifests/working/"
    "a1d-full-smoke-20260922T153543Z-output-1024.json"
)
SURVIVAL = Path(
    "/root/autodl-tmp/bridgetp/phase9_cap0_inputs/"
    "survival_table_m1_v1.json"
)
GUARD = Path(
    "/root/autodl-tmp/bridgetp/phase9_cap0_manifests/frozen/"
    "guard_free_kv_tokens.txt"
)
EXPECTED_INPUTS = {
    BASE: "47e4cf7f4d055eb82f32179fead20f3f1d9f9f061e51c16cbfd5755fae03b2ff",
    SURVIVAL: "031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a",
    GUARD: "0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument(
        "--result-root", type=Path,
        default=Path("/root/autodl-tmp/bridgetp/results/migration_manager_m3"),
    )
    parser.add_argument("--qps", nargs="+", type=float,
                        default=[0.1, 0.3, 0.7, 1.15])
    parser.add_argument("--reps", nargs="+", type=int, default=[1])
    args = parser.parse_args()
    if os.name == "nt":
        raise RuntimeError("A100 TPOT calibration must run on the Linux host")
    if any(value <= 0 for value in args.qps + args.reps):
        raise ValueError("QPS and repetitions must be positive")
    revision = common.git("rev-parse", "HEAD")
    if revision != args.expected_revision:
        raise RuntimeError(f"HEAD {revision} differs from expected revision")
    if common.git("branch", "--show-current") != "bridgetp/runtime-controller":
        raise RuntimeError("wrong branch for M3 calibration")
    if common.git("status", "--porcelain"):
        raise RuntimeError("worktree is not clean")
    gpu_list = subprocess.check_output(
        ["nvidia-smi", "-L"], text=True
    )
    if gpu_list.count("A100-PCIE-40GB") != 5:
        raise RuntimeError("expected five A100-PCIE-40GB GPUs")
    if not (MODEL / "config.json").is_file():
        raise FileNotFoundError(MODEL / "config.json")
    for path, expected in EXPECTED_INPUTS.items():
        if not path.is_file() or common.sha256(path) != expected:
            raise RuntimeError(f"original input path or SHA256 differs: {path}")
    if GUARD.read_text(encoding="utf-8").strip() != "8448":
        raise RuntimeError("guard is not the frozen A100 value")
    for port in (8001, 8200):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"port {port} is already in use") from error

    run = args.result_root / time.strftime("a100-m3-tpot-%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True, exist_ok=False)
    common.write_json(run / "preflight.json", {
        "revision": revision,
        "branch": "bridgetp/runtime-controller",
        "gpus": gpu_list.splitlines(),
        "model_path": str(MODEL),
        "model_config_sha256": common.sha256(MODEL / "config.json"),
        "original_inputs_sha256": {
            str(path): digest for path, digest in EXPECTED_INPUTS.items()
        },
        "workload": {"input_len": 2048, "output_len": 1024,
                     "qps": args.qps, "reps": args.reps,
                     "num_prompts": 100},
        "data_source": (
            "vllm bench random tokens; original manifest verified but unused"
        ),
    })
    server_args = SimpleNamespace(
        python_bin=Path(sys.executable), model_path=MODEL, dtype="bfloat16",
        max_model_len=8192, gpu_memory_utilization=0.88,
    )
    servers = []
    status = "FAILED"
    try:
        for name, tp, gpu_set, port in (
            ("tp4", 4, "1,2,3,4", 8200),
            ("tp1", 1, "0", 8001),
        ):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_set
            env["OMP_NUM_THREADS"] = "1"
            item = common.start_process(
                name, common.server_command(server_args, tp, port), env,
                run / f"{name}.log",
            )
            servers.append(item)
            common.wait_healthy(f"http://127.0.0.1:{port}", item, 900)
        command = [
            sys.executable, str(ROOT / "tools/bridge_tp/run_phase9_tpot_sweep.py"),
            "--out-root", str(run / "sweep"), "--model", str(MODEL),
            "--tp1-blocks", "1968", "--tp4-blocks", "35739",
            "--input-len", "2048", "--output-len", "1024",
            "--num-prompts", "100", "--num-warmups", "10",
            "--fit-load-model", "--qps", *(str(value) for value in args.qps),
            "--reps", *(str(value) for value in args.reps),
        ]
        with (run / "sweep.console.txt").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"TPOT sweep failed: {completed.returncode}")
        model = run / "sweep" / "tick_tpot_candidate.json"
        if not model.is_file():
            raise RuntimeError("TPOT sweep did not create a fitted model")
        status = "COMPLETE"
        print(f"model={model}")
        print(f"model_sha256={common.sha256(model)}")
    finally:
        common.stop_processes(servers)
        common.write_json(run / "status.json", {"status": status})
        archive = run.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as output:
            output.add(run, arcname=run.name)
        print(f"result_archive={archive}")


if __name__ == "__main__":
    main()
