#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the A3 stay-on-TP1 control with the migration workload manifest.

The TP1 and TP4 servers remain alive across sequential repetitions.  Target
background requests use the exact manifest passed to the MIGRATE arm, while
the anchor is served entirely by TP1.  The anchor starts at a measured offset
from the first target background token so both arms see comparable target load.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402
from tools.bridge_tp.run_phase9_cap0_noop import (  # noqa: E402
    wait_for_background_first_tokens,
)
from tools.bridge_tp.run_phase9_capacity_background import (  # noqa: E402
    load_manifest,
)
from vllm.bridge_tp.controller.online_io import (  # noqa: E402
    post_streaming_completion,
)
from vllm.bridge_tp.controller.sampling_contract import (  # noqa: E402
    freeze_strict_greedy_sampling,
)
from vllm.bridge_tp.online_shadow_strategy_protocol import (  # noqa: E402
    percentile,
    summarize_background_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--anchor-prompt-tokens", type=int, default=2048)
    parser.add_argument("--anchor-max-tokens", type=int, default=4096)
    parser.add_argument("--anchor-after-first-target-token-s", type=float,
                        default=7.9)
    parser.add_argument("--minimum-ready-target-jobs", type=int, default=1)
    parser.add_argument("--session-gap-s", type=float, default=1.0)
    parser.add_argument("--tp1-gpu", default="0")
    parser.add_argument("--tp4-gpus", default="1,2,3,4")
    parser.add_argument("--tp1-port", type=int, default=8180)
    parser.add_argument("--tp4-port", type=int, default=8380)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--server-start-timeout-s", type=float, default=900)
    parser.add_argument("--run-timeout-s", type=float, default=2400)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if os.name == "nt":
        raise RuntimeError("A3 STAY control requires Linux and five GPUs")
    if args.repetitions < (5 if args.phase == "formal" else 1):
        raise ValueError("formal requires at least five repetitions")
    if args.anchor_prompt_tokens <= 0 or args.anchor_max_tokens <= 128:
        raise ValueError("anchor must contain prompt and O128 progress")
    if args.anchor_prompt_tokens + args.anchor_max_tokens > args.max_model_len:
        raise ValueError("anchor exceeds max model length")
    if args.anchor_after_first_target_token_s < 0 or args.session_gap_s < 0:
        raise ValueError("timing delays cannot be negative")
    if args.minimum_ready_target_jobs < 1:
        raise ValueError("at least one target job must become ready")
    for path in (args.python_bin, args.model_path / "config.json", args.manifest,
                 common.SOURCE_REQUEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    revision = common.git("rev-parse", "HEAD")
    if revision != common.git("rev-parse", args.expected_revision):
        raise RuntimeError(f"HEAD {revision} differs from expected revision")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--"]
    ).returncode:
        raise RuntimeError("tracked working-tree changes are present")
    if common.sha256(args.manifest) != args.expected_manifest_sha256:
        raise RuntimeError("manifest SHA-256 differs from expected")
    manifest = load_manifest(args.manifest)
    jobs = manifest["jobs"]
    if len(jobs) < args.minimum_ready_target_jobs:
        raise ValueError("manifest has too few target jobs")
    for job in jobs:
        if job.get("pool") != "target":
            raise ValueError("A3 STAY requires a target-only manifest")
        request = job["request"]
        prompt = request.get("prompt")
        if not isinstance(prompt, list) or not all(isinstance(x, int) for x in prompt):
            raise ValueError("target prompts require exact token IDs")
        if len(prompt) + int(request["max_tokens"]) > args.max_model_len:
            raise ValueError("target request exceeds max model length")
    if args.out_root.exists():
        raise FileExistsError(f"refusing to reuse output root {args.out_root}")
    return revision, manifest


def first_target_token(event_path: Path) -> float:
    times = []
    for line in event_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "job_first_token" and row.get("pool") == "target":
                times.append(float(row["unix_s"]))
    if not times:
        raise RuntimeError("target workload has no first-token timestamp")
    return min(times)


def anchor_request(args: argparse.Namespace, repetition: int) -> dict[str, Any]:
    request = common.read_json(common.SOURCE_REQUEST)
    request["prompt"] = [100] * args.anchor_prompt_tokens
    request["max_tokens"] = args.anchor_max_tokens
    request = freeze_strict_greedy_sampling(request)
    request.update({
        "request_id": f"bridgetp-a3-stay-r{repetition:02d}",
        "stream": True,
        "return_token_ids": True,
    })
    request.setdefault("ignore_eos", True)
    return request


def run_one(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    repetition: int,
    env: dict[str, str],
) -> dict[str, Any]:
    root = args.out_root / f"r{repetition:02d}_stay_tp1"
    background_dir = root / "background"
    background_dir.mkdir(parents=True, exist_ok=False)
    background = common.start_process(
        "target background",
        [
            str(args.python_bin), str(common.BACKGROUND),
            "--manifest", str(args.manifest),
            "--source-url", f"http://127.0.0.1:{args.tp1_port}",
            "--target-url", f"http://127.0.0.1:{args.tp4_port}",
            "--out-dir", str(background_dir),
            "--request-timeout-s", str(args.run_timeout_s),
        ],
        env,
        background_dir / "background.log",
    )
    try:
        wait_for_background_first_tokens(
            background_dir / "background_events.jsonl",
            background,
            args.minimum_ready_target_jobs,
            min(args.run_timeout_s, args.server_start_timeout_s),
        )
        first_token = first_target_token(
            background_dir / "background_events.jsonl"
        )
        remaining = first_token + args.anchor_after_first_target_token_s - time.time()
        if remaining > 0:
            time.sleep(remaining)
        if background.process.poll() is not None:
            raise RuntimeError("background ended before anchor started")

        request = anchor_request(args, repetition)
        common.write_json(root / "anchor_request.json", request)
        token_rows: list[dict[str, Any]] = []

        def on_token(index: int, token_id: int, unix_s: float) -> None:
            token_rows.append({"index": index, "token_id": token_id,
                               "unix_s": unix_s})

        started = time.time()
        response = post_streaming_completion(
            f"http://127.0.0.1:{args.tp1_port}",
            request,
            args.run_timeout_s,
            on_token,
        )
        ended = time.time()
        background_rc = background.process.wait(timeout=args.run_timeout_s)
        if background_rc != 0:
            raise RuntimeError(f"background failed with code {background_rc}")
        bg = common.read_json(background_dir / "background_summary.json")
        token_times = [row["unix_s"] for row in token_rows]
        intervals = [
            (current - previous) * 1000
            for previous, current in zip(token_times, token_times[1:])
        ]
        windows = None
        if len(token_times) >= 128:
            windows = summarize_background_windows(
                bg["results"],
                shadow_start_unix_s=token_times[63],
                bridge_start_unix_s=token_times[95],
                committed_unix_s=token_times[127],
            )
        bg_tokens = sum(
            int(row.get("output_tokens", 0))
            for row in bg["results"] if row.get("status") == "COMPLETED"
        )
        bg_wall = max(0.0, float(bg["end_unix_s"]) - float(bg["start_unix_s"]))
        batch_end = max(ended, float(bg["end_unix_s"]))
        batch_wall = batch_end - float(bg["start_unix_s"])
        errors = []
        if len(token_rows) != args.anchor_max_tokens:
            errors.append("anchor output token count differs from request")
        if response.get("finish_reason") != "length":
            errors.append("anchor did not finish at requested output length")
        if bg.get("completed") != len(manifest["jobs"]) or bg.get("failed"):
            errors.append("target background workload did not complete")
        if windows is None:
            errors.append("anchor did not reach O128 windows")
        result = {
            "format_version": 1,
            "status": "PASS" if not errors else "FAIL",
            "mode": "STAY_ON_TP1",
            "repetition": repetition,
            "background_jobs": len(manifest["jobs"]),
            "first_target_token_unix_s": first_token,
            "anchor_started_unix_s": started,
            "anchor_start_after_first_target_token_s": started - first_token,
            "anchor_completed_unix_s": ended,
            "anchor_output_tokens": len(token_rows),
            "anchor_finish_reason": response.get("finish_reason"),
            "anchor_ttft_ms": (
                (token_times[0] - started) * 1000 if token_times else None
            ),
            "anchor_e2e_ms": (ended - started) * 1000,
            "anchor_tpot_p50_ms": percentile(intervals, 0.50),
            "anchor_tpot_p95_ms": percentile(intervals, 0.95),
            "anchor_tpot_p99_ms": percentile(intervals, 0.99),
            "target_tpot_windows": windows,
            "background_completed": bg["completed"],
            "background_failed": bg["failed"],
            "background_output_tokens": bg_tokens,
            "background_wall_time_s": bg_wall,
            "background_output_throughput_tokens_s": (
                bg_tokens / bg_wall if bg_wall else None
            ),
            "batch_wall_time_s": batch_wall,
            "batch_completed_output_tokens": bg_tokens + len(token_rows),
            "batch_output_throughput_tokens_s": (
                (bg_tokens + len(token_rows)) / batch_wall if batch_wall else None
            ),
            "errors": errors,
        }
        common.write_json(root / "anchor_tokens.json", token_rows)
        common.write_json(root / "result.json", result)
        if errors:
            raise RuntimeError("; ".join(errors))
        print(f"PASS: {root}", flush=True)
        return result
    finally:
        common.stop_processes([background])


def main() -> None:
    args = parse_args()
    revision, manifest = validate(args)
    contract = {
        "format_version": 1,
        "mode": "STAY_ON_TP1",
        "phase": args.phase,
        "revision": revision,
        "manifest_sha256": common.sha256(args.manifest),
        "manifest": str(args.manifest.resolve()),
        "background_jobs": len(manifest["jobs"]),
        "anchor_prompt_tokens": args.anchor_prompt_tokens,
        "anchor_max_tokens": args.anchor_max_tokens,
        "anchor_after_first_target_token_s": (
            args.anchor_after_first_target_token_s
        ),
        "repetitions": args.repetitions,
        "topology": {"tp1_gpu": args.tp1_gpu, "tp4_gpus": args.tp4_gpus},
    }
    if args.validate_only:
        print(json.dumps(contract, indent=2))
        return
    args.out_root.mkdir(parents=True, exist_ok=False)
    common.write_json(args.out_root / "contract.json", contract)
    env = {k: v for k, v in os.environ.items() if not k.startswith("BRIDGETP_")}
    env["OMP_NUM_THREADS"] = "1"
    services = []
    rows: list[dict[str, Any]] = []
    try:
        target = common.start_process(
            "target TP4",
            common.server_command(args, 4, args.tp4_port),
            env | {"CUDA_VISIBLE_DEVICES": args.tp4_gpus},
            args.out_root / "target_tp4.log",
        )
        services.append(target)
        common.wait_healthy(
            f"http://127.0.0.1:{args.tp4_port}", target,
            args.server_start_timeout_s,
        )
        source = common.start_process(
            "source TP1",
            common.server_command(args, 1, args.tp1_port),
            env | {"CUDA_VISIBLE_DEVICES": args.tp1_gpu},
            args.out_root / "source_tp1.log",
        )
        services.append(source)
        common.wait_healthy(
            f"http://127.0.0.1:{args.tp1_port}", source,
            args.server_start_timeout_s,
        )
        for repetition in range(1, args.repetitions + 1):
            row = run_one(args, manifest, repetition, env)
            rows.append(row)
            common.write_json(args.out_root / "batch_status.json", {
                "format_version": 1, "status": "RUNNING", "contract": contract,
                "runs": rows,
            })
            if repetition < args.repetitions and args.session_gap_s:
                time.sleep(args.session_gap_s)
        common.write_json(args.out_root / "acceptance.json", {
            "format_version": 1, "status": "PASS", "contract": contract,
            "runs": rows, "errors": [],
        })
        common.write_json(args.out_root / "batch_status.json", {
            "format_version": 1, "status": "COMPLETE", "contract": contract,
            "runs": rows,
        })
        print(f"A3_STAY_{args.phase.upper()}_COMPLETE: {args.out_root}")
    except Exception as error:
        common.write_json(args.out_root / "batch_status.json", {
            "format_version": 1, "status": "FAILED", "contract": contract,
            "runs": rows, "error": f"{type(error).__name__}: {error}",
        })
        raise
    finally:
        common.stop_processes(services)


if __name__ == "__main__":
    main()
