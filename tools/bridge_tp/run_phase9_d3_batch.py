#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the formal Phase 9 D-3 paired numerical-fidelity batch.

The batch has two GPU stages because the five-GPU machine cannot host the
migration stack and an independent pair of clean control servers at once:

* ``migrate`` restarts the TP1/TP4 migration stack for each prompt, performs a
  real BridgeTP cutover, and records the migrated stream plus a clean TP1 run.
* ``controls`` starts clean TP1 and TP4 servers once and collects A/B/C from
  the exact fixed prefix produced by ``migrate``.

Runs are resumable.  A request is attempted at most twice in ``migrate``;
failed attempt directories are retained and the request is then skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CONTROLLER = REPO / "tools" / "bridge_tp" / "run_phase9_controller.py"
STAGER = REPO / "tools" / "bridge_tp" / "phase8_stager.py"
CONTINUATION = REPO / "tools" / "bridge_tp" / "run_fixed_prefix_continuation.py"
MEASURE = REPO / "tools" / "bridge_tp" / "measure_agreement.py"
SUMMARIZE = REPO / "tools" / "bridge_tp" / "summarize_agreement.py"
GROUPS = "ABCD"
DEFAULT_CONFIG = REPO / "experiments" / "phase9" / "configs" / "e1_correctness.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    raw = read_json(path)
    if not isinstance(raw, dict) or raw.get("format_version") != 1:
        raise ValueError("manifest must be a format_version=1 JSON object")
    prompts = raw.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 50:
        raise ValueError("formal D3 manifest must contain exactly 50 prompts")
    smoke_prompts = raw.get("smoke_prompts")
    if not isinstance(smoke_prompts, list) or len(smoke_prompts) != 2:
        raise ValueError("D3 manifest must contain exactly two smoke prompts")
    ids = [str(item.get("request_id", "")) for item in prompts]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest request IDs must be nonempty and unique")
    for item in prompts:
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            raise ValueError(f"{item.get('request_id')}: prompt is empty")
    smoke_ids = [str(item.get("request_id", "")) for item in smoke_prompts]
    if any(not value for value in smoke_ids) or len(set(smoke_ids)) != 2:
        raise ValueError("smoke request IDs must be nonempty and unique")
    if set(ids) & set(smoke_ids):
        raise ValueError("formal and smoke request IDs must be disjoint")
    design = raw.get("design") or {}
    expected = {
        "trigger_output_tokens": 128,
        "cutover_output_tokens": 160,
        "target_local_budget_tokens": 256,
    }
    if any(int(design.get(key, -1)) != value for key, value in expected.items()):
        raise ValueError(f"manifest must freeze the validated design {expected}")
    if raw.get("status") != "FROZEN":
        raise ValueError("formal D3 manifest status must be FROZEN")
    return raw


def selected_prompts(
    manifest: dict[str, Any], mode: str, limit: int | None
) -> list[dict[str, Any]]:
    prompts = list(
        manifest["smoke_prompts"] if mode == "smoke" else manifest["prompts"]
    )
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def request_dir(root: Path, request_id: str) -> Path:
    return root / "requests" / request_id


def load_status(directory: Path) -> dict[str, Any]:
    path = directory / "status.json"
    if path.exists():
        return read_json(path)
    return {
        "format_version": 1,
        "request_id": directory.name,
        "migration_status": "PENDING",
        "control_status": "PENDING",
        "attempts": [],
    }


def save_status(directory: Path, status: dict[str, Any]) -> None:
    write_json(directory / "status.json", status)


def source_request(item: dict[str, Any], model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "prompt": item["prompt"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
        "n": 1,
        "use_beam_search": False,
        "stream": True,
        "bridgetp_group_id": None,
        "bridgetp_group_longest": False,
    }


def post_json(url: str, payload: dict, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.load(response)


def wait_healthy(url: str, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    health_url = url.rstrip("/") + "/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError(f"server did not become healthy: {health_url}")


def start_process(
    command: list[str], env: dict[str, str], log_path: Path
) -> tuple[subprocess.Popen, Any]:
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
    return process, handle


def stop_processes(processes: list[tuple[subprocess.Popen, Any]]) -> None:
    for process, _handle in reversed(processes):
        if process.poll() is None:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    for process, handle in reversed(processes):
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        handle.close()


def server_command(
    args: argparse.Namespace, tensor_parallel: int, port: int
) -> list[str]:
    return [
        str(args.python_bin),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model_path),
        "--served-model-name",
        args.served_model_name,
        "--tensor-parallel-size",
        str(tensor_parallel),
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


def target_connector(run: Path) -> str:
    return json.dumps(
        {
            "kv_connector": "BridgeTPStreamingConnector",
            "kv_connector_module_path": "vllm.bridge_tp.streaming_connector",
            "kv_role": "kv_consumer",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
                "bridgetp_stream_manifest": str(run / "staging_manifest.json"),
                "bridgetp_stream_receipt_dir": str(run / "receiver_receipts"),
                "bridgetp_stream_socket_timeout_s": 600,
                "bridgetp_stream_expected_phase": "BridgeTP D3 Phase 8",
                "bridgetp_takeover_control_path": str(run / "takeover_state.json"),
                "bridgetp_takeover_control_timeout_s": 600,
            },
        },
        separators=(",", ":"),
    )


def migration_env(
    args: argparse.Namespace, run: Path, migration_id: str
) -> tuple[dict[str, str], dict[str, str]]:
    common = os.environ.copy()
    target = common | {
        "CUDA_VISIBLE_DEVICES": args.tp4_gpus,
        "PHASE9_STAGING_MANIFEST": str(run / "staging_manifest.json"),
        "PHASE9_RECEIPTS": str(run / "receiver_receipts"),
        "PHASE9_CONTROL": str(run / "takeover_state.json"),
        "BRIDGETP_LOGIT_CAPTURE_ENABLED": "0",
        "BRIDGETP_DUMP_ENABLED": "0",
    }
    source = common | {
        "CUDA_VISIBLE_DEVICES": args.tp1_gpu,
        "BRIDGETP_LOGIT_CAPTURE_ENABLED": "0",
        "BRIDGETP_DUMP_ENABLED": "0",
        "BRIDGETP_STREAM_ENABLED": "1",
        "BRIDGETP_STREAM_MIGRATION_ID": migration_id,
        "BRIDGETP_STREAM_RUN_DIR": str(run),
        "BRIDGETP_STREAM_HOST": "127.0.0.1",
        "BRIDGETP_STREAM_BASE_PORT": str(args.stream_base_port),
        "BRIDGETP_STREAM_TARGET_TP": "4",
        "BRIDGETP_STREAM_HEAD_AXIS": "3",
        "BRIDGETP_STREAM_EXPECTED_KV_HEADS": "8",
        "BRIDGETP_STREAM_AFTER_OUTPUT_TOKENS": str(args.trigger_tokens),
        "BRIDGETP_STREAM_CHUNK_BYTES": "1048576",
        "BRIDGETP_STREAM_RATE_GIB_S": str(args.copy_rate_gib_s),
        "BRIDGETP_STREAM_SOCKET_TIMEOUT_S": "600",
        "BRIDGETP_STREAM_PIN_MEMORY": "1",
        "BRIDGETP_STREAM_STRICT": "1",
        "BRIDGETP_PHASE8_ENABLED": "1",
        "BRIDGETP_PHASE8_CUTOVER_OUTPUT_TOKENS": str(args.cutover_tokens),
        "BRIDGETP_PHASE8_DELTA_HOST": "127.0.0.1",
        "BRIDGETP_PHASE8_DELTA_BASE_PORT": str(args.delta_base_port),
        "BRIDGETP_TAKEOVER_ENABLED": "1",
        "BRIDGETP_TAKEOVER_MIGRATION_ID": migration_id,
        "BRIDGETP_TAKEOVER_RUN_DIR": str(run),
    }
    return source, target


def controller_config(args: argparse.Namespace, run: Path) -> Path:
    template = args.controller_config_template or DEFAULT_CONFIG
    config = read_json(template)
    diagnostic_note = "D3 fixed-boundary diagnostic only; value not used for action"
    config["tpot_tp1"] = {
        "base_s": 0.03,
        "per_running_s": 0.0,
        "calibration_source": diagnostic_note,
    }
    config["tpot_tp4"] = {
        "base_s": 0.02,
        "per_running_s": 0.0,
        "calibration_source": diagnostic_note,
    }
    config["interference"] = {
        "s_per_gib_at_ref": 0.0,
        "calibration_source": diagnostic_note,
        "model_kind": "legacy_power",
    }
    config["source_url"] = args.tp1_url
    config["target_url"] = args.tp4_url
    config["run_dir"] = str(run)
    config["tp1_total_kv_blocks"] = args.tp1_blocks
    config["tp4_total_kv_blocks"] = args.tp4_blocks
    survival_path = run / "diagnostic_survival_table.json"
    write_json(
        survival_path,
        {
            "format_version": 1,
            "source": diagnostic_note,
            "bucket_edges": [0],
            "remaining": [[args.max_tokens]],
            "max_observed_length": args.max_tokens,
        },
    )
    config["survival_table_path"] = str(survival_path)
    path = run / "controller_config.json"
    write_json(path, config)
    return path


def extract_migration_artifacts(
    args: argparse.Namespace, run: Path, destination: Path
) -> None:
    stats = read_json(run / "response_proxy_stats.json")
    staging = read_json(run / "staging_manifest.json")
    state = read_json(run / "takeover_state.json")
    migrated = [int(value) for value in stats.get("token_ids") or []]
    control = read_json(run / "control_tokens.json")
    cutover = int(stats.get("cutover_index", -1))
    prompt_count = int(staging["num_prompt_tokens"])
    known = [int(value) for value in staging["all_known_token_ids"]]
    if state.get("state") != "COMMITTED" or not stats.get("committed"):
        raise RuntimeError("migration did not reach COMMITTED")
    if cutover != args.cutover_tokens:
        raise RuntimeError(f"observed K={cutover}, expected {args.cutover_tokens}")
    if stats.get("source_origin_tokens") != cutover:
        raise RuntimeError("unified stream does not contain exactly K source tokens")
    if int(stats.get("target_origin_tokens", 0)) < args.budget:
        raise RuntimeError("migrated stream has fewer than budget target tokens")
    if len(control) < cutover + args.budget:
        raise RuntimeError("clean TP1 control is shorter than K + budget")
    if known[prompt_count:] != control[:cutover]:
        raise RuntimeError("staged fixed-prefix suffix differs from TP1 control")
    fixed = known[:prompt_count] + [int(value) for value in control[:cutover]]
    write_json(destination / "migrated_tokens.json", migrated)
    write_json(destination / "control_tokens.json", control)
    write_json(
        destination / "fixed_prefix.json",
        {"fixed_token_ids": fixed, "boundary_k": cutover},
    )
    (destination / "K.txt").write_text(f"{cutover}\n", encoding="utf-8")
    for name in (
        "response_proxy_stats.json",
        "staging_manifest.json",
        "takeover_state.json",
        "phase9_audit.jsonl",
    ):
        source = run / name
        if source.exists():
            target = destination / "migration_evidence" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())


def clean_control(args: argparse.Namespace, run: Path) -> None:
    payload = read_json(run / "source_request.json")
    payload["request_id"] = f"d3-control-{run.name}"
    payload["stream"] = False
    payload["return_token_ids"] = True
    result = post_json(args.tp1_url, payload, args.request_timeout_s)
    tokens = result.get("choices", [{}])[0].get("token_ids")
    if not isinstance(tokens, list):
        raise RuntimeError("clean TP1 response did not include token IDs")
    write_json(run / "control_request.json", payload)
    write_json(run / "control_response.json", result)
    write_json(run / "control_tokens.json", [int(value) for value in tokens])


def migrate_attempt(
    args: argparse.Namespace, item: dict[str, Any], attempt: int
) -> Path:
    directory = request_dir(args.out_root, item["request_id"])
    run = directory / "attempts" / f"attempt_{attempt}"
    run.mkdir(parents=True, exist_ok=False)
    migration_id = str(uuid.uuid4())
    write_json(
        run / "source_request_input.json",
        source_request(item, args.served_model_name, args.max_tokens),
    )
    source_env, target_env = migration_env(args, run, migration_id)
    processes: list[tuple[subprocess.Popen, Any]] = []
    try:
        target_command = server_command(args, 4, args.tp4_port)
        target_command += ["--kv-transfer-config", target_connector(run)]
        target = start_process(target_command, target_env, run / "target_tp4.log")
        processes.append(target)
        source = start_process(
            server_command(args, 1, args.tp1_port),
            source_env,
            run / "source_tp1.log",
        )
        processes.append(source)
        wait_healthy(args.tp4_url, target[0], args.server_start_timeout_s)
        wait_healthy(args.tp1_url, source[0], args.server_start_timeout_s)
        stager_command = [
            str(args.python_bin),
            str(STAGER),
            "--run-dir",
            str(run),
            "--delta-host",
            "127.0.0.1",
            "--delta-base-port",
            str(args.delta_base_port),
            "--delivery-host",
            "127.0.0.1",
            "--delivery-base-port",
            str(args.delivery_base_port),
            "--timeout-s",
            "600",
        ]
        processes.append(
            start_process(stager_command, os.environ.copy(), run / "stager.log")
        )
        command = [
            str(args.python_bin),
            str(CONTROLLER),
            "--config",
            str(controller_config(args, run)),
            "--run-dir",
            str(run),
            "--source-request",
            str(run / "source_request_input.json"),
            "--migration-id",
            migration_id,
            "--diagnostic-trigger-output-tokens",
            str(args.trigger_tokens),
            "--diagnostic-cutover-output-tokens",
            str(args.cutover_tokens),
            "--request-timeout-s",
            str(args.request_timeout_s),
        ]
        with (run / "controller.log").open("wb") as log:
            subprocess.run(
                command,
                cwd=REPO,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        clean_control(args, run)
        extract_migration_artifacts(args, run, directory)
        return run
    finally:
        stop_processes(processes)


def run_migrate(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> None:
    for index, item in enumerate(prompts, 1):
        directory = request_dir(args.out_root, item["request_id"])
        directory.mkdir(parents=True, exist_ok=True)
        status = load_status(directory)
        recovered = all(
            (directory / name).exists()
            for name in (
                "migrated_tokens.json",
                "control_tokens.json",
                "fixed_prefix.json",
                "K.txt",
            )
        )
        if recovered:
            status["migration_status"] = "COMPLETE"
            save_status(directory, status)
        known_attempts = {int(item["attempt"]) for item in status["attempts"]}
        attempts_root = directory / "attempts"
        if attempts_root.exists() and not recovered:
            for path in sorted(attempts_root.glob("attempt_*")):
                try:
                    number = int(path.name.removeprefix("attempt_"))
                except ValueError:
                    continue
                if number not in known_attempts:
                    status["attempts"].append(
                        {
                            "attempt": number,
                            "status": "FAILED",
                            "error": "interrupted attempt recovered on resume",
                            "run_dir": str(path),
                        }
                    )
            status["attempts"].sort(key=lambda item: int(item["attempt"]))
            if len(status["attempts"]) >= args.max_attempts:
                status["migration_status"] = "SKIPPED"
            save_status(directory, status)
        if status["migration_status"] == "COMPLETE":
            print(f"[{index}/{len(prompts)}] {item['request_id']}: already complete")
            continue
        attempts = len(status["attempts"])
        while attempts < args.max_attempts:
            used = {int(item["attempt"]) for item in status["attempts"]}
            attempt = next(
                number
                for number in range(1, args.max_attempts + 1)
                if number not in used
            )
            print(
                f"[{index}/{len(prompts)}] {item['request_id']}: "
                f"migration attempt {attempt}/{args.max_attempts}"
            )
            started = time.time()
            try:
                run = migrate_attempt(args, item, attempt)
                status["attempts"].append(
                    {"attempt": attempt, "status": "COMPLETE", "run_dir": str(run)}
                )
                status["migration_status"] = "COMPLETE"
                save_status(directory, status)
                break
            except Exception as error:
                status["attempts"].append(
                    {
                        "attempt": attempt,
                        "status": "FAILED",
                        "error": f"{type(error).__name__}: {error}",
                        "elapsed_s": time.time() - started,
                    }
                )
                attempts += 1
                status["migration_status"] = (
                    "SKIPPED" if attempts >= args.max_attempts else "PENDING"
                )
                save_status(directory, status)
                print(f"  failed: {error}")
        if status["migration_status"] == "SKIPPED":
            print("  failed twice; artifacts retained, continuing")


def start_clean_servers(args: argparse.Namespace) -> list[tuple[subprocess.Popen, Any]]:
    processes = []
    disabled = {
        "BRIDGETP_STREAM_ENABLED": "0",
        "BRIDGETP_PHASE8_ENABLED": "0",
        "BRIDGETP_TAKEOVER_ENABLED": "0",
        "BRIDGETP_LOGIT_CAPTURE_ENABLED": "0",
        "BRIDGETP_DUMP_ENABLED": "0",
    }
    tp4_env = os.environ.copy() | disabled | {
        "CUDA_VISIBLE_DEVICES": args.tp4_gpus
    }
    tp1_env = os.environ.copy() | disabled | {
        "CUDA_VISIBLE_DEVICES": args.tp1_gpu
    }
    tp4 = start_process(
        server_command(args, 4, args.tp4_port),
        tp4_env,
        args.out_root / "logs" / "clean_tp4.log",
    )
    processes.append(tp4)
    tp1 = start_process(
        server_command(args, 1, args.tp1_port),
        tp1_env,
        args.out_root / "logs" / "clean_tp1.log",
    )
    processes.append(tp1)
    wait_healthy(args.tp4_url, tp4[0], args.server_start_timeout_s)
    wait_healthy(args.tp1_url, tp1[0], args.server_start_timeout_s)
    return processes


def run_command(command: list[str], log: Path | None = None) -> None:
    if log is None:
        subprocess.run(command, cwd=REPO, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        subprocess.run(
            command,
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_continuation(
    args: argparse.Namespace,
    directory: Path,
    name: str,
    url: str,
    batch_size: int,
) -> Path:
    out = directory / "controls" / name
    command = [
        str(args.python_bin),
        str(CONTINUATION),
        "--base-url",
        url,
        "--model",
        args.served_model_name,
        "--fixed-prefix-token-ids",
        str(directory / "fixed_prefix.json"),
        "--request-id",
        f"{directory.name}-{name}",
        "--max-tokens",
        str(args.budget),
        "--batch-size",
        str(batch_size),
        "--timeout-s",
        str(args.request_timeout_s),
        "--out-dir",
        str(out),
    ]
    run_command(command, directory / "controls.log")
    return out / "primary_tokens.json"


def record_metadata(args: argparse.Namespace, directory: Path) -> Path:
    metadata = {
        "model": str(args.model_path),
        "gpu_platform": args.gpu_platform,
        "vllm_commit": args.vllm_commit,
        "cutover_rule": (
            f"diagnostic-fixed-trigger-{args.trigger_tokens}-"
            f"K-{args.cutover_tokens}"
        ),
        "manifest_sha256": sha256_file(args.manifest),
    }
    path = directory / "metadata.json"
    write_json(path, metadata)
    return path


def measure_groups(args: argparse.Namespace, directory: Path) -> None:
    k = int((directory / "K.txt").read_text(encoding="utf-8"))
    metadata = record_metadata(args, directory)
    tokens = {
        "control": directory / "control_tokens.json",
        "migrated": directory / "migrated_tokens.json",
        "tp1_a": directory / "controls" / "tp1_a" / "primary_tokens.json",
        "tp1_b": directory / "controls" / "tp1_b" / "primary_tokens.json",
        "tp4_b1": directory / "controls" / "tp4_b1" / "primary_tokens.json",
        "tp4_b8": directory / "controls" / "tp4_b8" / "primary_tokens.json",
    }
    comparisons = {
        "A": (tokens["tp1_a"], tokens["tp1_b"], 0, 0),
        "B": (tokens["tp1_a"], tokens["tp4_b1"], 0, 0),
        "C": (tokens["tp4_b1"], tokens["tp4_b8"], 0, 0),
        "D": (tokens["control"], tokens["migrated"], k, k),
    }
    for group, (left, right, left_offset, right_offset) in comparisons.items():
        command = [
            str(args.python_bin),
            str(MEASURE),
            "--group",
            group,
            "--request-id",
            directory.name,
            "--left",
            str(left),
            "--right",
            str(right),
            "--left-offset",
            str(left_offset),
            "--right-offset",
            str(right_offset),
            "--boundary-k",
            str(k),
            "--budget",
            str(args.budget),
            "--fixed-prefix-token-ids",
            str(directory / "fixed_prefix.json"),
            "--metadata",
            str(metadata),
            "--out",
            str(args.out_root / "agreement" / group / f"{directory.name}.json"),
        ]
        run_command(command, directory / "controls.log")


def run_controls(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> None:
    eligible = []
    for item in prompts:
        directory = request_dir(args.out_root, item["request_id"])
        status = load_status(directory)
        if status["migration_status"] == "COMPLETE":
            eligible.append((directory, status))
    if not eligible:
        raise RuntimeError("no completed migration requests are available")
    processes: list[tuple[subprocess.Popen, Any]] = []
    try:
        processes = start_clean_servers(args)
        for index, (directory, status) in enumerate(eligible, 1):
            if status["control_status"] == "COMPLETE":
                print(f"[{index}/{len(eligible)}] {directory.name}: controls complete")
                continue
            try:
                run_continuation(args, directory, "tp1_a", args.tp1_url, 1)
                run_continuation(args, directory, "tp1_b", args.tp1_url, 1)
                run_continuation(args, directory, "tp4_b1", args.tp4_url, 1)
                run_continuation(args, directory, "tp4_b8", args.tp4_url, 8)
                measure_groups(args, directory)
                status["control_status"] = "COMPLETE"
                save_status(directory, status)
                print(f"[{index}/{len(eligible)}] {directory.name}: controls complete")
            except Exception as error:
                status["control_status"] = "FAILED"
                status["control_error"] = f"{type(error).__name__}: {error}"
                save_status(directory, status)
                print(f"[{index}/{len(eligible)}] {directory.name}: {error}")
    finally:
        stop_processes(processes)


def write_progress(
    args: argparse.Namespace, prompts: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    for item in prompts:
        directory = request_dir(args.out_root, item["request_id"])
        status = load_status(directory)
        rows.append(status)
    counts = {
        key: sum(row["control_status"] == key for row in rows)
        for key in ("PENDING", "FAILED", "COMPLETE")
    }
    migration_counts = {
        key: sum(row["migration_status"] == key for row in rows)
        for key in ("PENDING", "SKIPPED", "COMPLETE")
    }
    payload = {
        "format_version": 1,
        "status": (
            "COMPLETE" if counts["COMPLETE"] == len(prompts) else "INCOMPLETE"
        ),
        "expected": len(prompts),
        "migration": migration_counts,
        "controls": counts,
        "requests": rows,
    }
    write_json(args.out_root / "batch_progress.json", payload)
    return payload


def run_summary(args: argparse.Namespace) -> None:
    if args.noninferiority_margin_tokens is None:
        raise ValueError("summarize requires a preregistered NI margin")
    command = [str(args.python_bin), str(SUMMARIZE)]
    for group in GROUPS:
        command += [
            f"--group-{group.lower()}",
            str(args.out_root / "agreement" / group),
        ]
    command += [
        "--noninferiority-margin-tokens",
        str(args.noninferiority_margin_tokens),
        "--out",
        str(args.out_root / "d3_summary.json"),
    ]
    run_command(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("validate", "migrate", "controls", "summarize", "status"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--served-model-name", default="bridgetp-model")
    parser.add_argument("--controller-config-template", type=Path)
    parser.add_argument("--tp1-blocks", type=int)
    parser.add_argument("--tp4-blocks", type=int)
    parser.add_argument("--tp1-gpu", default="0")
    parser.add_argument("--tp4-gpus", default="1,2,3,4")
    parser.add_argument("--tp1-port", type=int, default=8001)
    parser.add_argument("--tp4-port", type=int, default=8200)
    parser.add_argument("--stream-base-port", type=int, default=29800)
    parser.add_argument("--delta-base-port", type=int, default=29900)
    parser.add_argument("--delivery-base-port", type=int, default=30000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--trigger-tokens", type=int, default=128)
    parser.add_argument("--cutover-tokens", type=int, default=160)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=416)
    parser.add_argument("--copy-rate-gib-s", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--server-start-timeout-s", type=float, default=900)
    parser.add_argument("--request-timeout-s", type=float, default=1800)
    parser.add_argument("--gpu-platform", default="NVIDIA A100-PCIE-40GB x5")
    parser.add_argument("--vllm-commit")
    parser.add_argument("--noninferiority-margin-tokens", type=float)
    args = parser.parse_args()
    args.tp1_url = f"http://127.0.0.1:{args.tp1_port}"
    args.tp4_url = f"http://127.0.0.1:{args.tp4_port}"
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.mode == "formal" and args.limit is not None:
        parser.error("formal mode always runs the complete 50-prompt manifest")
    if args.max_attempts != 2:
        parser.error("formal D3 freezes --max-attempts at 2")
    if (args.trigger_tokens, args.cutover_tokens, args.budget, args.max_tokens) != (
        128,
        160,
        256,
        416,
    ):
        parser.error("formal design is frozen at trigger=128, K=160, budget=256")
    if args.stage in {"migrate", "controls"} and args.model_path is None:
        parser.error(f"{args.stage} requires --model-path")
    if args.stage == "migrate":
        required = {
            "--tp1-blocks": args.tp1_blocks,
            "--tp4-blocks": args.tp4_blocks,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("migrate requires " + ", ".join(missing))
    if args.stage == "controls" and not args.vllm_commit:
        parser.error("controls requires --vllm-commit")
    return args


def validate_stage_inputs(args: argparse.Namespace) -> None:
    paths = {
        "manifest": args.manifest,
    }
    if args.stage == "migrate":
        paths.update(
            {
                "model": args.model_path,
            }
        )
        if args.controller_config_template is not None:
            paths["controller config template"] = args.controller_config_template
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    validate_stage_inputs(args)
    manifest = load_manifest(args.manifest)
    prompts = selected_prompts(manifest, args.mode, args.limit)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "validate":
        print(f"VALID: {len(manifest['prompts'])} frozen prompts")
        print(f"sha256: {sha256_file(args.manifest)}")
        return
    if args.stage == "migrate":
        run_migrate(args, prompts)
    elif args.stage == "controls":
        run_controls(args, prompts)
    elif args.stage == "summarize":
        run_summary(args)
    progress = write_progress(args, prompts)
    print(json.dumps(progress, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
