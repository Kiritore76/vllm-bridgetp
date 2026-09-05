#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate and run CAP-0 no-migration calibration repetitions.

Smoke and formal modes are deliberately separate.  Smoke runs once and emits a
machine-readable acceptance contract.  Formal mode requires that exact contract
before it can run r01-r03.  The orchestrator stays in the foreground, owns all
child services, fails closed on insufficient pressure or incomplete observation,
and never freezes the resulting guard candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
BACKGROUND = REPO / "tools" / "bridge_tp" / "run_phase9_capacity_background.py"
CONTROLLER = REPO / "tools" / "bridge_tp" / "run_phase9_controller.py"
STAGER = REPO / "tools" / "bridge_tp" / "phase8_stager.py"
CONFIG_TEMPLATE = (
    REPO / "experiments" / "phase9" / "configs" / "cap0_controller.template.json"
)
SOURCE_REQUEST = REPO / "experiments" / "phase9" / "configs" / "request_long.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--survival-table", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument(
        "--smoke-acceptance",
        type=Path,
        help="required in formal mode; PASS artifact from smoke mode",
    )
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-survival-sha256")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--tp1-blocks", type=int, required=True)
    parser.add_argument("--tp4-blocks", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--anchor-max-tokens", type=int, default=8000)
    parser.add_argument("--minimum-peak-kv-usage-frac", type=float, default=0.70)
    parser.add_argument(
        "--allow-censored",
        action="store_true",
        help="allow a run with no observed preemption (not recommended)",
    )
    parser.add_argument("--coverage-slack-s", type=float, default=1.0)
    parser.add_argument("--tp1-gpu", default="0")
    parser.add_argument("--tp4-gpus", default="1,2,3,4")
    parser.add_argument("--tp1-port", type=int, default=8001)
    parser.add_argument("--tp4-port", type=int, default=8200)
    parser.add_argument("--snapshot-port", type=int, default=29800)
    parser.add_argument("--delta-port", type=int, default=29900)
    parser.add_argument("--delivery-port", type=int, default=30000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--server-start-timeout-s", type=float, default=900)
    parser.add_argument("--run-timeout-s", type=float, default=1800)
    parser.add_argument("--stager-timeout-s", type=float, default=1800)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )
    return completed.stdout.strip()


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[bytes]
    handle: Any
    log_path: Path


def start_process(
    name: str,
    command: list[str],
    env: dict[str, str],
    log_path: Path,
) -> ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab", buffering=0)
    kwargs: dict[str, Any] = {
        "cwd": REPO,
        "env": env,
        "stdout": handle,
        "stderr": subprocess.STDOUT,
    }
    if os.name != "nt":
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    return ManagedProcess(name, process, handle, log_path)


def signal_process(item: ManagedProcess, sig: signal.Signals) -> None:
    if item.process.poll() is not None:
        return
    if os.name == "nt":
        if sig == signal.SIGKILL:
            item.process.kill()
        else:
            item.process.terminate()
    else:
        os.killpg(item.process.pid, sig)


def stop_processes(processes: list[ManagedProcess]) -> None:
    for item in reversed(processes):
        signal_process(item, signal.SIGINT)
    deadline = time.monotonic() + 45
    for item in reversed(processes):
        if item.process.poll() is None:
            try:
                item.process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                signal_process(item, signal.SIGTERM)
    deadline = time.monotonic() + 20
    for item in reversed(processes):
        if item.process.poll() is None:
            try:
                item.process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                signal_process(item, signal.SIGKILL)
                item.process.wait(timeout=10)
        item.handle.close()


def wait_healthy(url: str, process: ManagedProcess, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    health_url = url.rstrip("/") + "/health"
    while time.monotonic() < deadline:
        if process.process.poll() is not None:
            raise RuntimeError(
                f"{process.name} exited with code {process.process.returncode}; "
                f"see {process.log_path}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError(f"server did not become healthy: {health_url}")


def wait_for_file_or_exit(
    path: Path,
    process: ManagedProcess,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.process.poll() is not None:
            raise RuntimeError(
                f"{process.name} exited with code {process.process.returncode}; "
                f"see {process.log_path}"
            )
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {path}")


def wait_pair(
    first: ManagedProcess,
    second: ManagedProcess,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        first_rc = first.process.poll()
        second_rc = second.process.poll()
        if first_rc is not None and first_rc != 0:
            raise RuntimeError(
                f"{first.name} failed with code {first_rc}; see {first.log_path}"
            )
        if second_rc is not None and second_rc != 0:
            raise RuntimeError(
                f"{second.name} failed with code {second_rc}; see {second.log_path}"
            )
        if first_rc == 0 and second_rc == 0:
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {first.name} and {second.name}")


def server_command(args: argparse.Namespace, tp: int, port: int) -> list[str]:
    return [
        str(args.python_bin),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model_path),
        "--served-model-name",
        "bridgetp-model",
        "--tensor-parallel-size",
        str(tp),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--port",
        str(port),
        "--no-enable-prefix-caching",
        "--disable-hybrid-kv-cache-manager",
        "--no-async-scheduling",
    ]


def make_config(
    args: argparse.Namespace,
    controller_dir: Path,
    provenance_dir: Path,
) -> Path:
    config = read_json(CONFIG_TEMPLATE)
    config["run_dir"] = str(controller_dir)
    config["tp1_total_kv_blocks"] = args.tp1_blocks
    config["tp4_total_kv_blocks"] = args.tp4_blocks
    config["survival_table_path"] = str(args.survival_table.resolve())
    config["platform_note"] = (
        "CAP-0 ENGINEERING PILOT; automated 5x A100 PCIe calibration"
    )
    config["capacity_pilot"]["enabled"] = False
    config["capacity_pilot"]["guard_free_kv_tokens"] = 0
    path = provenance_dir / "controller_config.json"
    write_json(path, config)
    return path


def make_source_request(
    args: argparse.Namespace,
    provenance_dir: Path,
) -> Path:
    request = read_json(SOURCE_REQUEST)
    request["max_tokens"] = args.anchor_max_tokens
    path = provenance_dir / "anchor_request.json"
    write_json(path, request)
    return path


def target_connector(controller_dir: Path) -> str:
    return json.dumps(
        {
            "kv_connector": "BridgeTPStreamingConnector",
            "kv_connector_module_path": "vllm.bridge_tp.streaming_connector",
            "kv_role": "kv_consumer",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
                "bridgetp_stream_manifest": str(
                    controller_dir / "staging_manifest.json"
                ),
                "bridgetp_stream_receipt_dir": str(
                    controller_dir / "receiver_receipts"
                ),
                "bridgetp_stream_socket_timeout_s": 600,
                "bridgetp_stream_expected_phase": "BridgeTP D3 Phase 8",
                "bridgetp_takeover_control_path": str(
                    controller_dir / "takeover_state.json"
                ),
                "bridgetp_takeover_control_timeout_s": 600,
            },
        },
        separators=(",", ":"),
    )


def source_environment(
    args: argparse.Namespace,
    run_id: str,
    controller_dir: Path,
) -> dict[str, str]:
    # run_phase9_controller derives the external anchor request ID from
    # run_dir.name.  Keep the scheduler selector tied to that exact derivation;
    # the surrounding batch/run ID is not part of the request ID when the
    # orchestrator uses the nested <label>/controller layout.
    source_request_id_prefix = f"bridgetp-phase9-{controller_dir.name}"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.tp1_gpu,
            "OMP_NUM_THREADS": "1",
            "BRIDGETP_DUMP_ENABLED": "0",
            "BRIDGETP_STREAM_ENABLED": "1",
            "BRIDGETP_STREAM_MIGRATION_ID": run_id,
            "BRIDGETP_STREAM_RUN_DIR": str(controller_dir),
            "BRIDGETP_STREAM_HOST": "127.0.0.1",
            "BRIDGETP_STREAM_BASE_PORT": str(args.snapshot_port),
            "BRIDGETP_STREAM_TARGET_TP": "4",
            "BRIDGETP_STREAM_HEAD_AXIS": "3",
            "BRIDGETP_STREAM_EXPECTED_KV_HEADS": "8",
            "BRIDGETP_STREAM_AFTER_OUTPUT_TOKENS": "128",
            "BRIDGETP_STREAM_CHUNK_BYTES": "1048576",
            "BRIDGETP_STREAM_RATE_GIB_S": "0.50",
            "BRIDGETP_STREAM_SOCKET_TIMEOUT_S": "600",
            "BRIDGETP_STREAM_PIN_MEMORY": "1",
            "BRIDGETP_STREAM_STRICT": "1",
            "BRIDGETP_STREAM_SOURCE_REQUEST_ID_PREFIX": source_request_id_prefix,
            "BRIDGETP_PHASE8_ENABLED": "1",
            "BRIDGETP_PHASE8_CUTOVER_OUTPUT_TOKENS": "160",
            "BRIDGETP_PHASE8_DELTA_HOST": "127.0.0.1",
            "BRIDGETP_PHASE8_DELTA_BASE_PORT": str(args.delta_port),
            "BRIDGETP_TAKEOVER_ENABLED": "1",
            "BRIDGETP_TAKEOVER_MIGRATION_ID": run_id,
            "BRIDGETP_TAKEOVER_RUN_DIR": str(controller_dir),
        }
    )
    return env


def load_audit(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def accept_calibration(
    controller_dir: Path,
    background_dir: Path,
    expected_jobs: int,
    minimum_peak_kv_usage_frac: float,
    require_preemption: bool,
    coverage_slack_s: float,
) -> dict[str, Any]:
    background = read_json(background_dir / "background_summary.json")
    rows = load_audit(controller_dir / "phase9_audit.jsonl")
    telemetry = [row for row in rows if row.get("kind") == "telemetry"]
    decisions = [row for row in rows if row.get("kind") == "decision"]
    transitions = [row for row in rows if row.get("kind") == "transition"]
    ends = [row for row in rows if row.get("kind") == "run_end"]
    migration_transitions = [
        row
        for row in transitions
        if row.get("to") in {"SHADOW", "HANDOFF", "TAKEOVER"}
    ]
    telemetry_times = [
        float(row["tp1"]["sampled_unix_s"])
        for row in telemetry
        if isinstance(row.get("tp1", {}).get("sampled_unix_s"), (int, float))
    ]
    kv_usage = [float(row["tp1"]["kv_usage_frac"]) for row in telemetry]
    preemptions = [int(row["tp1"]["preemptions_total"]) for row in telemetry]
    peak_kv_usage = max(kv_usage) if kv_usage else None
    initial_preemptions = preemptions[0] if preemptions else None
    maximum_preemptions = max(preemptions) if preemptions else None
    preemption_delta = (
        maximum_preemptions - initial_preemptions
        if maximum_preemptions is not None and initial_preemptions is not None
        else None
    )
    background_start = background.get("start_unix_s")
    background_end = background.get("end_unix_s")
    audit_end = ends[0].get("unix_s") if len(ends) == 1 else None
    last_telemetry = max(telemetry_times) if telemetry_times else None
    errors: list[str] = []
    if background.get("jobs") != expected_jobs:
        errors.append(
            f"background jobs={background.get('jobs')!r}, expected {expected_jobs}"
        )
    if background.get("completed") != expected_jobs or background.get("failed") != 0:
        errors.append(
            f"background did not complete {expected_jobs}/{expected_jobs}: "
            f"completed={background.get('completed')!r}, "
            f"failed={background.get('failed')!r}"
        )
    if not telemetry:
        errors.append("no telemetry records")
    if not decisions:
        errors.append("no decision records")
    if len(ends) != 1 or ends[0].get("final_state") != "COMPLETED_ON_TP1":
        errors.append(f"unexpected run_end records: {ends!r}")
    if migration_transitions:
        errors.append(f"observed {len(migration_transitions)} migration transitions")
    if peak_kv_usage is None or peak_kv_usage < minimum_peak_kv_usage_frac:
        errors.append(
            f"peak TP1 KV usage {peak_kv_usage!r} is below required "
            f"{minimum_peak_kv_usage_frac:.3f}"
        )
    if require_preemption and not preemption_delta:
        errors.append("no TP1 preemption observed under calibration pressure")
    if not isinstance(background_start, (int, float)) or not isinstance(
        background_end, (int, float)
    ):
        errors.append("background timing is missing")
    else:
        if (
            audit_end is None
            or float(audit_end) + coverage_slack_s < float(background_end)
        ):
            errors.append("controller ended before the background workload")
        if (
            last_telemetry is None
            or last_telemetry + coverage_slack_s < float(background_end)
        ):
            errors.append("telemetry did not cover the background workload end")
    result = {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "background_completed": background.get("completed"),
        "background_failed": background.get("failed"),
        "telemetry_samples": len(telemetry),
        "decision_records": len(decisions),
        "migration_transitions": len(migration_transitions),
        "peak_tp1_kv_usage_frac": peak_kv_usage,
        "minimum_peak_kv_usage_frac": minimum_peak_kv_usage_frac,
        "preemption_delta": preemption_delta,
        "require_preemption": require_preemption,
        "background_start_unix_s": background_start,
        "background_end_unix_s": background_end,
        "last_telemetry_unix_s": last_telemetry,
        "controller_end_unix_s": audit_end,
        "coverage_slack_s": coverage_slack_s,
        "final_state": ends[0].get("final_state") if len(ends) == 1 else None,
        "errors": errors,
    }
    return result


def calibration_metrics(controller_dir: Path) -> dict[str, Any]:
    rows = load_audit(controller_dir / "phase9_audit.jsonl")
    telemetry = [row for row in rows if row.get("kind") == "telemetry"]
    initial = int(telemetry[0]["tp1"]["preemptions_total"])
    first_index = next(
        (
            index
            for index, row in enumerate(telemetry)
            if int(row["tp1"]["preemptions_total"]) > initial
        ),
        None,
    )
    first = telemetry[first_index] if first_index is not None else None
    before_first = (
        telemetry[:first_index] if first_index is not None else telemetry
    )

    # Admission, completion and recompute can allocate or release thousands of
    # KV tokens between two scrapes.  Those discrete scheduler events are not a
    # token-generation slope and must not be extrapolated for T_enter seconds.
    # Use unchanged scheduler-state intervals and a robust p95 instead.  The
    # maximum p95 across formal repetitions remains deliberately conservative.
    steady_declines: list[float] = []
    for previous, current in zip(before_first, before_first[1:]):
        previous_tp1 = previous["tp1"]
        current_tp1 = current["tp1"]
        scheduler_fields = (
            "num_running",
            "num_waiting",
            "preemptions_total",
        )
        if any(
            int(previous_tp1[field]) != int(current_tp1[field])
            for field in scheduler_fields
        ):
            continue
        previous_time = float(previous_tp1["sampled_unix_s"])
        current_time = float(current_tp1["sampled_unix_s"])
        elapsed = current_time - previous_time
        if elapsed <= 0 or elapsed > 2.0:
            continue
        previous_free = int(previous["capacity_signal"]["free_kv_tokens"])
        current_free = int(current["capacity_signal"]["free_kv_tokens"])
        consumed = previous_free - current_free
        if consumed > 0:
            steady_declines.append(consumed / elapsed)
    if not steady_declines:
        raise RuntimeError("no steady TP1 KV-decline samples before preemption")
    ordered_declines = sorted(steady_declines)
    p95_index = max(0, math.ceil(0.95 * len(ordered_declines)) - 1)
    steady_p95 = ordered_declines[p95_index]

    free = [int(row["capacity_signal"]["free_kv_tokens"]) for row in telemetry]
    decline = [
        float(row["capacity_signal"]["decline_rate_tokens_s"])
        for row in telemetry
    ]
    kv_usage = [float(row["tp1"]["kv_usage_frac"]) for row in telemetry]
    waiting = [int(row["tp1"]["num_waiting"]) for row in telemetry]
    preemptions = [int(row["tp1"]["preemptions_total"]) for row in telemetry]
    return {
        "samples": len(telemetry),
        "minimum_free_kv_tokens": min(free),
        "pre_preemption_free_kv_tokens": (
            int(telemetry[first_index - 1]["capacity_signal"]["free_kv_tokens"])
            if first_index is not None and first_index > 0
            else None
        ),
        "post_preemption_free_kv_tokens": (
            int(first["capacity_signal"]["free_kv_tokens"])
            if first is not None
            else None
        ),
        "censored": first is None,
        "steady_decline_sample_count": len(steady_declines),
        "steady_decline_p95_tokens_s": steady_p95,
        "maximum_observed_ewma_decline_tokens_s": max(decline),
        "peak_tp1_kv_usage_frac": max(kv_usage),
        "maximum_tp1_waiting": max(waiting),
        "preemption_delta": max(preemptions) - initial,
    }


def calculate_guard_candidate(
    metrics: list[dict[str, Any]],
    block_size: int,
    tp1_total_tokens: int,
    enter_seconds: float = 8.0,
) -> dict[str, Any]:
    if not metrics:
        raise ValueError("at least one calibration metric is required")
    f_values = [
        int(item["pre_preemption_free_kv_tokens"])
        if item["pre_preemption_free_kv_tokens"] is not None
        else int(item["minimum_free_kv_tokens"])
        for item in metrics
    ]
    maximum_decline = max(
        float(item["steady_decline_p95_tokens_s"]) for item in metrics
    )
    raw = max(f_values) + maximum_decline * enter_seconds
    rounded = math.ceil(raw / block_size) * block_size
    candidate = min(max(rounded, block_size), tp1_total_tokens - block_size)
    return {
        "format_version": 1,
        "status": "CANDIDATE_NOT_FROZEN",
        "f_values": f_values,
        "maximum_steady_decline_p95_tokens_s": maximum_decline,
        "enter_seconds": enter_seconds,
        "raw_guard_tokens": raw,
        "block_size": block_size,
        "tp1_total_tokens": tp1_total_tokens,
        "guard_candidate_tokens": candidate,
    }


def experiment_contract(args: argparse.Namespace, revision: str) -> dict[str, Any]:
    return {
        "format_version": 1,
        "revision": revision,
        "model_path": str(args.model_path.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "survival_table_sha256": sha256(args.survival_table),
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tp1_gpu": args.tp1_gpu,
        "tp4_gpus": args.tp4_gpus,
        "tp1_blocks": args.tp1_blocks,
        "tp4_blocks": args.tp4_blocks,
        "anchor_max_tokens": args.anchor_max_tokens,
        "minimum_peak_kv_usage_frac": args.minimum_peak_kv_usage_frac,
        "require_preemption": not args.allow_censored,
        "coverage_slack_s": args.coverage_slack_s,
    }


def validate_smoke_acceptance(
    args: argparse.Namespace,
    revision: str,
) -> dict[str, Any]:
    if args.smoke_acceptance is None:
        raise ValueError("--smoke-acceptance is required in formal mode")
    if not args.smoke_acceptance.is_file():
        raise FileNotFoundError(args.smoke_acceptance)
    smoke = read_json(args.smoke_acceptance)
    if smoke.get("status") != "PASS":
        raise RuntimeError("smoke acceptance status is not PASS")
    expected = experiment_contract(args, revision)
    if smoke.get("contract") != expected:
        raise RuntimeError(
            "smoke contract differs from the requested formal experiment"
        )
    run = smoke.get("run")
    metrics = run.get("metrics") if isinstance(run, dict) else None
    if not isinstance(metrics, dict) or run.get("status") != "PASS":
        raise RuntimeError("smoke acceptance has no passing run metrics")
    if float(metrics.get("peak_tp1_kv_usage_frac", 0.0)) < (
        args.minimum_peak_kv_usage_frac
    ):
        raise RuntimeError("smoke run metrics do not satisfy the KV pressure gate")
    if not args.allow_censored and int(metrics.get("preemption_delta", 0)) <= 0:
        raise RuntimeError("smoke run metrics do not contain a preemption")
    return smoke


def validate_manifest_pressure(
    manifest: Any,
    *,
    tp1_blocks: int,
    anchor_max_tokens: int,
    max_model_len: int,
) -> int:
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("calibration manifest requires format_version=1")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("calibration manifest requires a non-empty jobs list")
    identifiers: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"manifest job {index} must be an object")
        job_id = str(job.get("job_id", "")).strip()
        if not job_id or job_id in identifiers:
            raise ValueError(f"manifest job {index} has an invalid job_id")
        identifiers.add(job_id)
        if job.get("pool") not in {"source", "target"}:
            raise ValueError(f"manifest job {job_id} has an invalid pool")
        request = job.get("request")
        if not isinstance(request, dict):
            raise ValueError(f"manifest job {job_id} has no request")
        maximum = int(request.get("max_tokens", 0))
        if maximum <= 0 or maximum > max_model_len - 128:
            raise ValueError(
                f"manifest job {job_id} max_tokens must be in "
                f"[1, {max_model_len - 128}]"
            )
        if float(job.get("start_after_s", -1)) < 0:
            raise ValueError(f"manifest job {job_id} has an invalid start time")
    source_jobs = [job for job in jobs if job["pool"] == "source"]
    target_jobs = [job for job in jobs if job["pool"] == "target"]
    if len(source_jobs) < 4 or not target_jobs:
        raise ValueError(
            "calibration requires at least four source jobs and one target job"
        )
    planned_source_tokens = anchor_max_tokens + sum(
        int(job["request"]["max_tokens"]) for job in source_jobs
    )
    required_planned_tokens = math.ceil(tp1_blocks * 16 * 1.10)
    if planned_source_tokens < required_planned_tokens:
        raise ValueError(
            f"planned source tokens {planned_source_tokens} are below the "
            f"110% capacity floor {required_planned_tokens}"
        )
    if min(float(job["start_after_s"]) for job in source_jobs) > 5:
        raise ValueError("source pressure must begin within five seconds")
    return len(jobs)


def run_one(
    args: argparse.Namespace,
    batch_root: Path,
    label: str,
    repetition: int | None,
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"cap0-calibration-{label}-{stamp}"
    rep_root = batch_root / label
    controller_dir = rep_root / "controller"
    background_dir = rep_root / "background"
    provenance_dir = rep_root / "provenance"
    for path in (controller_dir, background_dir, provenance_dir):
        path.mkdir(parents=True, exist_ok=False)
    config_path = make_config(args, controller_dir, provenance_dir)
    source_request_path = make_source_request(args, provenance_dir)
    (provenance_dir / "git_revision.txt").write_text(
        git("rev-parse", "HEAD") + "\n", encoding="utf-8"
    )
    (provenance_dir / "git_status.txt").write_text(
        git("status", "--short", "--branch") + "\n", encoding="utf-8"
    )
    write_json(
        provenance_dir / "inputs.json",
        {
            "run_id": run_id,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "survival_table": str(args.survival_table.resolve()),
            "survival_table_sha256": sha256(args.survival_table),
            "model_path": str(args.model_path.resolve()),
            "tp1_blocks": args.tp1_blocks,
            "tp4_blocks": args.tp4_blocks,
            "anchor_max_tokens": args.anchor_max_tokens,
            "minimum_peak_kv_usage_frac": args.minimum_peak_kv_usage_frac,
            "require_preemption": not args.allow_censored,
            "coverage_slack_s": args.coverage_slack_s,
            "source_request_id_prefix": (
                f"bridgetp-phase9-{controller_dir.name}"
            ),
        },
    )

    base_env = os.environ.copy()
    base_env["OMP_NUM_THREADS"] = "1"
    processes: list[ManagedProcess] = []
    try:
        print(f"[{run_id}] starting target TP4", flush=True)
        target_env = base_env | {"CUDA_VISIBLE_DEVICES": args.tp4_gpus}
        target_command = server_command(args, 4, args.tp4_port)
        target_command += ["--kv-transfer-config", target_connector(controller_dir)]
        target = start_process(
            "target TP4", target_command, target_env, controller_dir / "target_tp4.log"
        )
        processes.append(target)
        wait_healthy(
            f"http://127.0.0.1:{args.tp4_port}",
            target,
            args.server_start_timeout_s,
        )
        print(f"[{run_id}] target TP4 healthy", flush=True)

        print(f"[{run_id}] starting source TP1", flush=True)
        source = start_process(
            "source TP1",
            server_command(args, 1, args.tp1_port),
            source_environment(args, run_id, controller_dir),
            controller_dir / "source_tp1.log",
        )
        processes.append(source)
        wait_healthy(
            f"http://127.0.0.1:{args.tp1_port}",
            source,
            args.server_start_timeout_s,
        )
        print(f"[{run_id}] source TP1 healthy", flush=True)

        print(f"[{run_id}] starting stager and controller", flush=True)
        stager = start_process(
            "stager",
            [
                str(args.python_bin),
                str(STAGER),
                "--run-dir",
                str(controller_dir),
                "--delta-host",
                "127.0.0.1",
                "--delta-base-port",
                str(args.delta_port),
                "--delivery-host",
                "127.0.0.1",
                "--delivery-base-port",
                str(args.delivery_port),
                "--timeout-s",
                str(args.stager_timeout_s),
            ],
            base_env,
            controller_dir / "stager.log",
        )
        processes.append(stager)
        time.sleep(1)
        if stager.process.poll() is not None:
            raise RuntimeError(
                f"stager exited with code {stager.process.returncode}; "
                f"see {stager.log_path}"
            )

        controller = start_process(
            "controller",
            [
                str(args.python_bin),
                str(CONTROLLER),
                "--config",
                str(config_path),
                "--run-dir",
                str(controller_dir),
                "--source-request",
                str(source_request_path),
                "--migration-id",
                run_id,
                "--preflight-timeout-s",
                "120",
                "--dry-run",
            ],
            base_env,
            provenance_dir / "controller_console.txt",
        )
        processes.append(controller)
        wait_for_file_or_exit(
            controller_dir / "source_progress.json", controller, timeout_s=150
        )
        print(
            f"[{run_id}] source anchor is live; starting frozen load",
            flush=True,
        )

        background = start_process(
            "background workload",
            [
                str(args.python_bin),
                str(BACKGROUND),
                "--manifest",
                str(args.manifest),
                "--source-url",
                f"http://127.0.0.1:{args.tp1_port}",
                "--target-url",
                f"http://127.0.0.1:{args.tp4_port}",
                "--out-dir",
                str(background_dir),
            ],
            base_env,
            background_dir / "background.log",
        )
        processes.append(background)
        wait_pair(controller, background, args.run_timeout_s)
        for service in (target, source, stager):
            if service.process.poll() is not None:
                raise RuntimeError(
                    f"{service.name} exited early with code "
                    f"{service.process.returncode}; see {service.log_path}"
                )

        manifest = read_json(args.manifest)
        acceptance = accept_calibration(
            controller_dir,
            background_dir,
            expected_jobs=len(manifest["jobs"]),
            minimum_peak_kv_usage_frac=args.minimum_peak_kv_usage_frac,
            require_preemption=not args.allow_censored,
            coverage_slack_s=args.coverage_slack_s,
        )
        write_json(provenance_dir / "calibration_acceptance.json", acceptance)
        if acceptance["status"] != "PASS":
            raise RuntimeError("; ".join(acceptance["errors"]))
        metrics = calibration_metrics(controller_dir)
        write_json(provenance_dir / "calibration_metrics.json", metrics)
        result = {
            "phase": args.mode,
            "repetition": repetition,
            "run_id": run_id,
            "status": "PASS",
            "root": str(rep_root),
            "metrics": metrics,
        }
        write_json(rep_root / "status.json", result)
        return result
    except Exception as error:
        result = {
            "phase": args.mode,
            "repetition": repetition,
            "run_id": run_id,
            "status": "FAIL",
            "root": str(rep_root),
            "error": f"{type(error).__name__}: {error}",
        }
        write_json(rep_root / "status.json", result)
        raise
    finally:
        stop_processes(processes)


def validate_inputs(args: argparse.Namespace) -> str:
    if os.name == "nt":
        raise RuntimeError("CAP-0 calibration runner requires Linux")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if args.anchor_max_tokens <= 0:
        raise ValueError("--anchor-max-tokens must be positive")
    if args.anchor_max_tokens > args.max_model_len - 128:
        raise ValueError(
            "--anchor-max-tokens must leave at least 128 tokens for the prompt"
        )
    if not 0 < args.minimum_peak_kv_usage_frac <= 1:
        raise ValueError("--minimum-peak-kv-usage-frac must be in (0, 1]")
    if args.coverage_slack_s < 0:
        raise ValueError("--coverage-slack-s cannot be negative")
    if args.tp1_blocks <= 0 or args.tp4_blocks <= 0:
        raise ValueError("KV block counts must be positive")
    for path in (
        args.python_bin,
        args.model_path,
        args.manifest,
        args.survival_table,
        CONFIG_TEMPLATE,
        SOURCE_REQUEST,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    revision = git("rev-parse", "HEAD")
    expected = git("rev-parse", args.expected_revision)
    if revision != expected:
        raise RuntimeError(f"HEAD {revision} differs from expected {expected}")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--"]
    ).returncode != 0:
        raise RuntimeError("tracked working-tree changes are present")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--cached", "--quiet", "HEAD", "--"]
    ).returncode != 0:
        raise RuntimeError("staged changes are present")
    if args.expected_manifest_sha256:
        actual = sha256(args.manifest)
        if actual != args.expected_manifest_sha256:
            raise RuntimeError(
                f"manifest SHA-256 {actual} differs from expected "
                f"{args.expected_manifest_sha256}"
            )
    if args.expected_survival_sha256:
        actual = sha256(args.survival_table)
        if actual != args.expected_survival_sha256:
            raise RuntimeError(
                f"survival SHA-256 {actual} differs from expected "
                f"{args.expected_survival_sha256}"
            )
    validate_manifest_pressure(
        read_json(args.manifest),
        tp1_blocks=args.tp1_blocks,
        anchor_max_tokens=args.anchor_max_tokens,
        max_model_len=args.max_model_len,
    )
    if args.mode == "formal":
        validate_smoke_acceptance(args, revision)
    elif args.smoke_acceptance is not None:
        raise ValueError("--smoke-acceptance is only valid in formal mode")
    return revision


def main() -> None:
    args = parse_args()
    revision = validate_inputs(args)
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=False)
    status_path = args.out_root / "batch_status.json"
    batch = {
        "format_version": 1,
        "status": "RUNNING",
        "revision": revision,
        "mode": args.mode,
        "started_unix_s": time.time(),
        "repetitions_expected": 1 if args.mode == "smoke" else args.repetitions,
        "runs": [],
    }
    if args.mode == "formal" and args.smoke_acceptance is not None:
        batch["smoke_acceptance"] = str(args.smoke_acceptance.resolve())
        batch["smoke_acceptance_sha256"] = sha256(args.smoke_acceptance)
    write_json(status_path, batch)
    try:
        if args.mode == "smoke":
            label = "smoke"
            print("[smoke] starting non-counted calibration gate", flush=True)
            result = run_one(args, args.out_root, label, repetition=None)
            batch["runs"].append(result)
            smoke_acceptance = {
                "format_version": 1,
                "status": "PASS",
                "contract": experiment_contract(args, revision),
                "run": result,
            }
            write_json(args.out_root / "smoke_acceptance.json", smoke_acceptance)
            batch["status"] = "SMOKE_COMPLETE"
            batch["ended_unix_s"] = time.time()
            write_json(status_path, batch)
            print(f"SMOKE_COMPLETE: {args.out_root}", flush=True)
            print("Review smoke_acceptance.json before formal mode.", flush=True)
            return

        for repetition in range(1, args.repetitions + 1):
            label = f"r{repetition:02d}"
            print(f"[{repetition}/{args.repetitions}] starting calibration", flush=True)
            result = run_one(args, args.out_root, label, repetition)
            batch["runs"].append(result)
            write_json(status_path, batch)
            print(
                f"[{repetition}/{args.repetitions}] PASS: {result['run_id']}",
                flush=True,
            )
        metrics = [item["metrics"] for item in batch["runs"]]
        guard = calculate_guard_candidate(
            metrics,
            block_size=16,
            tp1_total_tokens=args.tp1_blocks * 16,
        )
        write_json(args.out_root / "guard_candidate.json", guard)
        batch["status"] = "COMPLETE"
        batch["ended_unix_s"] = time.time()
        batch["guard_candidate"] = guard
        write_json(status_path, batch)
        print(f"COMPLETE: {args.out_root}", flush=True)
        print(
            f"guard candidate (not frozen): {guard['guard_candidate_tokens']}",
            flush=True,
        )
    except BaseException as error:
        failed_status = (
            args.out_root / label / "status.json" if "label" in locals() else None
        )
        if failed_status is not None and failed_status.is_file():
            failed_result = read_json(failed_status)
            if not batch["runs"] or batch["runs"][-1].get("run_id") != (
                failed_result.get("run_id")
            ):
                batch["runs"].append(failed_result)
        batch["status"] = "FAILED"
        batch["ended_unix_s"] = time.time()
        batch["error"] = f"{type(error).__name__}: {error}"
        write_json(status_path, batch)
        raise


if __name__ == "__main__":
    main()
