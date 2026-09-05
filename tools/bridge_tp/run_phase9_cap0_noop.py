#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one non-counted CAP-0 No-op bring-up with owned child processes.

The source capacity signal is enabled, but a deliberately busy TP4 target must
fail the current target guard.  Passing therefore requires active capacity
pressure, explicit STAY decisions, a complete TP1 anchor response, and no
Shadow/Handoff/Takeover artifacts.  This runner only emits a bring-up artifact;
it never freezes a workload or starts reportable repetitions.
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
from tools.bridge_tp.run_phase9_capacity_background import (  # noqa: E402
    load_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--survival-table", type=Path, required=True)
    parser.add_argument("--guard-file", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-survival-sha256")
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--expected-guard", type=int, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--tp1-blocks", type=int, required=True)
    parser.add_argument("--tp4-blocks", type=int, required=True)
    parser.add_argument("--anchor-max-tokens", type=int, default=8000)
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
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def prompt_tokens(request: dict[str, Any]) -> int | None:
    prompt = request.get("prompt")
    if isinstance(prompt, list) and all(isinstance(item, int) for item in prompt):
        return len(prompt)
    return None


def validate_noop_pressure(
    manifest: dict[str, Any],
    *,
    anchor_max_tokens: int,
    tp1_total_tokens: int,
    tp4_total_tokens: int,
    max_model_len: int,
    max_target_waiting: int = 4,
) -> dict[str, Any]:
    if manifest.get("scenario") != "CAP-0 No-op bring-up":
        raise ValueError("manifest is not labeled CAP-0 No-op bring-up")
    jobs = manifest["jobs"]
    source = [job for job in jobs if job["pool"] == "source"]
    target = [job for job in jobs if job["pool"] == "target"]
    if not source or not target:
        raise ValueError("No-op manifest requires source and target jobs")

    source_output_demand = anchor_max_tokens + sum(
        int(job["request"]["max_tokens"]) for job in source
    )
    if source_output_demand < int(tp1_total_tokens * 1.10):
        raise ValueError("source demand is below the 110% TP1 pressure floor")
    if len(target) <= max_target_waiting + 1:
        raise ValueError("too few target jobs to exercise the waiting guard")
    target_starts = [float(job["start_after_s"]) for job in target]
    source_starts = [float(job["start_after_s"]) for job in source]
    if min(target_starts) > min(source_starts):
        raise ValueError("target pressure must start before source pressure")
    if max(target_starts) - min(target_starts) > 1.0:
        raise ValueError("target arrivals must form a burst within one second")

    target_context_demand = 0
    for job in target:
        request = job["request"]
        exact_prompt_tokens = prompt_tokens(request)
        if exact_prompt_tokens is None:
            raise ValueError(
                f"target job {job['job_id']} must use exact prompt token IDs"
            )
        context = exact_prompt_tokens + int(request["max_tokens"])
        if context > max_model_len:
            raise ValueError(f"target job {job['job_id']} exceeds max model length")
        target_context_demand += context
    if target_context_demand <= tp4_total_tokens:
        raise ValueError("target context demand must exceed TP4 KV capacity")
    return {
        "source_jobs": len(source),
        "target_jobs": len(target),
        "source_output_demand_tokens": source_output_demand,
        "target_context_demand_tokens": target_context_demand,
        "tp1_total_tokens": tp1_total_tokens,
        "tp4_total_tokens": tp4_total_tokens,
    }


def make_config(
    args: argparse.Namespace,
    controller_dir: Path,
    provenance_dir: Path,
    guard: int,
) -> Path:
    config = common.read_json(common.CONFIG_TEMPLATE)
    config["run_dir"] = str(controller_dir)
    config["tp1_total_kv_blocks"] = args.tp1_blocks
    config["tp4_total_kv_blocks"] = args.tp4_blocks
    config["survival_table_path"] = str(args.survival_table.resolve())
    config["platform_note"] = (
        "CAP-0 NO-OP BRING-UP; automated 5x A100 PCIe; not reportable"
    )
    config["capacity_pilot"]["enabled"] = True
    config["capacity_pilot"]["guard_free_kv_tokens"] = guard
    path = provenance_dir / "controller_config.json"
    common.write_json(path, config)
    return path


def load_audit(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def accept_noop(
    controller_dir: Path,
    background_dir: Path,
    *,
    expected_jobs: int,
    expected_anchor_tokens: int,
    max_target_kv_usage_frac: float = 0.85,
    max_target_waiting: int = 4,
) -> dict[str, Any]:
    background = common.read_json(background_dir / "background_summary.json")
    rows = load_audit(controller_dir / "phase9_audit.jsonl")
    ends = [row for row in rows if row.get("kind") == "run_end"]
    transitions = [row for row in rows if row.get("kind") == "transition"]
    capacity = [
        row for row in rows if row.get("kind") == "capacity_pilot_decision"
    ]
    active = [row for row in capacity if row.get("signal", {}).get("active")]
    blocked = [
        row
        for row in active
        if row.get("action") == "STAY"
        and (
            float(row.get("target_kv_usage_frac", 0.0))
            > max_target_kv_usage_frac
            or int(row.get("target_waiting", 0)) > max_target_waiting
        )
    ]
    start_shadow = [row for row in capacity if row.get("action") == "START_SHADOW"]
    migration_transitions = [
        row
        for row in transitions
        if row.get("to") in {"SHADOW", "HANDOFF", "TAKEOVER"}
    ]
    telemetry = [row for row in rows if row.get("kind") == "telemetry"]
    peak_source_usage = max(
        (float(row["tp1"]["kv_usage_frac"]) for row in telemetry),
        default=None,
    )
    peak_blocked_target_usage = max(
        (float(row.get("target_kv_usage_frac", 0.0)) for row in blocked),
        default=None,
    )
    peak_blocked_target_waiting = max(
        (int(row.get("target_waiting", 0)) for row in blocked),
        default=None,
    )
    proxy_path = controller_dir / "response_proxy_stats.json"
    proxy = common.read_json(proxy_path) if proxy_path.is_file() else {}
    forbidden_artifacts = [
        str(path.name)
        for path in (
            controller_dir / "staging_manifest.json",
            controller_dir / "takeover_state.json",
            controller_dir / "cleanup_request.json",
        )
        if path.exists()
    ]

    errors: list[str] = []
    if background.get("jobs") != expected_jobs:
        errors.append("background job count differs from manifest")
    if background.get("completed") != expected_jobs or background.get("failed") != 0:
        errors.append("background workload did not complete cleanly")
    if len(ends) != 1 or ends[0].get("final_state") != "COMPLETED_ON_TP1":
        errors.append(f"unexpected run_end records: {ends!r}")
    if not active:
        errors.append("source capacity signal never became active")
    if not blocked:
        errors.append("no active capacity decision was blocked by the target guard")
    if start_shadow:
        errors.append(f"observed {len(start_shadow)} START_SHADOW decisions")
    if migration_transitions:
        errors.append(f"observed {len(migration_transitions)} migration transitions")
    if forbidden_artifacts:
        errors.append(f"unexpected migration artifacts: {forbidden_artifacts!r}")
    if proxy.get("emitted_tokens") != expected_anchor_tokens:
        errors.append(
            f"anchor emitted {proxy.get('emitted_tokens')!r}, "
            f"expected {expected_anchor_tokens}"
        )
    if proxy.get("target_origin_tokens") != 0 or proxy.get("committed") is not False:
        errors.append("proxy reports target-origin or committed output")

    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "background_completed": background.get("completed"),
        "background_failed": background.get("failed"),
        "telemetry_samples": len(telemetry),
        "capacity_decisions": len(capacity),
        "active_capacity_decisions": len(active),
        "blocked_stay_decisions": len(blocked),
        "start_shadow_decisions": len(start_shadow),
        "migration_transitions": len(migration_transitions),
        "final_state": ends[0].get("final_state") if len(ends) == 1 else None,
        "peak_source_kv_usage_frac": peak_source_usage,
        "peak_blocked_target_kv_usage_frac": peak_blocked_target_usage,
        "peak_blocked_target_waiting": peak_blocked_target_waiting,
        "anchor_emitted_tokens": proxy.get("emitted_tokens"),
        "forbidden_artifacts": forbidden_artifacts,
        "errors": errors,
    }


def validate_inputs(args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    if os.name == "nt":
        raise RuntimeError("CAP-0 No-op runner requires Linux")
    if args.tp1_blocks <= 0 or args.tp4_blocks <= 0:
        raise ValueError("KV block counts must be positive")
    if args.anchor_max_tokens <= 0:
        raise ValueError("anchor max tokens must be positive")
    if args.anchor_max_tokens > args.max_model_len - 128:
        raise ValueError("anchor must leave 128 tokens for its prompt")
    for path in (
        args.python_bin,
        args.model_path,
        args.manifest,
        args.survival_table,
        args.guard_file,
        common.CONFIG_TEMPLATE,
        common.SOURCE_REQUEST,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    revision = common.git("rev-parse", "HEAD")
    expected = common.git("rev-parse", args.expected_revision)
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
        actual = common.sha256(args.manifest)
        if actual != args.expected_manifest_sha256:
            raise RuntimeError("manifest SHA-256 differs from expected value")
    if args.expected_survival_sha256:
        actual = common.sha256(args.survival_table)
        if actual != args.expected_survival_sha256:
            raise RuntimeError("survival SHA-256 differs from expected value")
    guard = int(args.guard_file.read_text(encoding="utf-8").strip())
    actual_guard_sha256 = common.sha256(args.guard_file)
    if actual_guard_sha256 != args.expected_guard_sha256:
        raise RuntimeError("frozen guard SHA-256 differs from expected value")
    if guard != args.expected_guard:
        raise RuntimeError(f"frozen guard {guard} differs from {args.expected_guard}")
    if not 0 < guard < args.tp1_blocks * 16:
        raise ValueError("frozen guard is outside TP1 capacity")
    manifest = load_manifest(args.manifest)
    pressure = validate_noop_pressure(
        manifest,
        anchor_max_tokens=args.anchor_max_tokens,
        tp1_total_tokens=args.tp1_blocks * 16,
        tp4_total_tokens=args.tp4_blocks * 16,
        max_model_len=args.max_model_len,
    )
    if args.out_root.exists():
        raise FileExistsError(f"refusing to reuse output root {args.out_root}")
    return revision, guard, pressure


def run(
    args: argparse.Namespace,
    revision: str,
    guard: int,
    pressure: dict[str, Any],
) -> None:
    out_root = args.out_root.resolve()
    controller_dir = out_root / "controller"
    background_dir = out_root / "background"
    provenance_dir = out_root / "provenance"
    for path in (controller_dir, background_dir, provenance_dir):
        path.mkdir(parents=True, exist_ok=False)
    run_id = out_root.name
    manifest = load_manifest(args.manifest)
    config_path = make_config(args, controller_dir, provenance_dir, guard)
    source_request = common.make_source_request(args, provenance_dir)
    common.write_json(
        provenance_dir / "inputs.json",
        {
            "format_version": 1,
            "scenario": "CAP-0 No-op bring-up",
            "status": "BRINGUP_NOT_REPORTABLE",
            "run_id": run_id,
            "revision": revision,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": common.sha256(args.manifest),
            "survival_table": str(args.survival_table.resolve()),
            "survival_table_sha256": common.sha256(args.survival_table),
            "guard_file": str(args.guard_file.resolve()),
            "guard_file_sha256": common.sha256(args.guard_file),
            "guard_free_kv_tokens": guard,
            "pressure_preflight": pressure,
        },
    )
    (provenance_dir / "git_revision.txt").write_text(
        revision + "\n", encoding="utf-8"
    )
    (provenance_dir / "git_status.txt").write_text(
        common.git("status", "--short", "--branch") + "\n",
        encoding="utf-8",
    )

    base_env = os.environ.copy()
    base_env["OMP_NUM_THREADS"] = "1"
    processes: list[common.ManagedProcess] = []
    status: dict[str, Any] = {
        "format_version": 1,
        "status": "RUNNING",
        "scenario": "noop",
        "run_id": run_id,
        "revision": revision,
        "started_unix_s": time.time(),
    }
    common.write_json(out_root / "status.json", status)
    try:
        print(f"[{run_id}] starting target TP4", flush=True)
        target = common.start_process(
            "target TP4",
            common.server_command(args, 4, args.tp4_port)
            + ["--kv-transfer-config", common.target_connector(controller_dir)],
            base_env | {"CUDA_VISIBLE_DEVICES": args.tp4_gpus},
            controller_dir / "target_tp4.log",
        )
        processes.append(target)
        common.wait_healthy(
            f"http://127.0.0.1:{args.tp4_port}",
            target,
            args.server_start_timeout_s,
        )
        print(f"[{run_id}] target TP4 healthy", flush=True)

        source = common.start_process(
            "source TP1",
            common.server_command(args, 1, args.tp1_port),
            common.source_environment(args, run_id, controller_dir),
            controller_dir / "source_tp1.log",
        )
        processes.append(source)
        common.wait_healthy(
            f"http://127.0.0.1:{args.tp1_port}",
            source,
            args.server_start_timeout_s,
        )
        print(f"[{run_id}] source TP1 healthy", flush=True)

        stager = common.start_process(
            "stager",
            [
                str(args.python_bin),
                str(common.STAGER),
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
            raise RuntimeError("stager exited before the controller")

        controller = common.start_process(
            "controller",
            [
                str(args.python_bin),
                str(common.CONTROLLER),
                "--config",
                str(config_path),
                "--run-dir",
                str(controller_dir),
                "--source-request",
                str(source_request),
                "--migration-id",
                run_id,
                "--preflight-timeout-s",
                "120",
            ],
            base_env,
            provenance_dir / "controller_console.txt",
        )
        processes.append(controller)
        common.wait_for_file_or_exit(
            controller_dir / "source_progress.json", controller, timeout_s=150
        )
        print(f"[{run_id}] source anchor live; starting No-op load", flush=True)

        background = common.start_process(
            "background workload",
            [
                str(args.python_bin),
                str(common.BACKGROUND),
                "--manifest",
                str(args.manifest),
                "--source-url",
                f"http://127.0.0.1:{args.tp1_port}",
                "--target-url",
                f"http://127.0.0.1:{args.tp4_port}",
                "--out-dir",
                str(background_dir),
                "--request-timeout-s",
                str(args.run_timeout_s),
            ],
            base_env,
            background_dir / "background.log",
        )
        processes.append(background)
        common.wait_pair(controller, background, args.run_timeout_s)
        for service in (target, source, stager):
            if service.process.poll() is not None:
                raise RuntimeError(
                    f"{service.name} exited early with code "
                    f"{service.process.returncode}; see {service.log_path}"
                )

        acceptance = accept_noop(
            controller_dir,
            background_dir,
            expected_jobs=len(manifest["jobs"]),
            expected_anchor_tokens=args.anchor_max_tokens,
        )
        common.write_json(provenance_dir / "noop_acceptance.json", acceptance)
        if acceptance["status"] != "PASS":
            raise RuntimeError("; ".join(acceptance["errors"]))
        status.update(
            {
                "status": "BRINGUP_COMPLETE",
                "ended_unix_s": time.time(),
                "acceptance": acceptance,
            }
        )
        common.write_json(out_root / "status.json", status)
        print(f"NOOP_BRINGUP_COMPLETE: {out_root}", flush=True)
    except Exception as error:
        status.update(
            {
                "status": "FAILED",
                "ended_unix_s": time.time(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        common.write_json(out_root / "status.json", status)
        raise
    finally:
        common.stop_processes(processes)


def main() -> None:
    args = parse_args()
    revision, guard, pressure = validate_inputs(args)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "scenario": "CAP-0 No-op bring-up",
                    "revision": revision,
                    "guard_free_kv_tokens": guard,
                    "manifest_sha256": common.sha256(args.manifest),
                    "pressure_preflight": pressure,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    run(args, revision, guard, pressure)


if __name__ == "__main__":
    main()
