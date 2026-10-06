#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one exploratory natural-EOS STAY/MIGRATE load matrix on five A100s.

The four cells share a frozen anchor and input selection. Each cell starts fresh
services for its two arms. Pressure cells are diagnostic: their padded prompts
must pass an observed near-guard gate before they can be interpreted as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ANCHOR_ID = "oasst1:0ffd5b9c-d93a-4c60-b66a-f8786fbea2a0"
EXPECTED_UUIDS = [
    "GPU-8d2a8196-d89d-ceeb-c8da-ce6a1293f58d",
    "GPU-cc103a2b-e56c-ce25-1bfe-4ae3151ddeb6",
    "GPU-b19db15f-27d6-c3c4-b99d-dd5a6c613141",
    "GPU-b11851d9-a463-7a18-8962-350780825656",
    "GPU-ef512882-db3a-d713-2051-ba54d4c92c02",
]
EXPECTED_SHAS = {
    "input": "75cae22e298548b54b6164b3df9adc9ebdc61ced3a85eecfd7df9e96ea7a1be3",
    "model_config": "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e",
    "base": "47e4cf7f4d055eb82f32179fead20f3f1d9f9f061e51c16cbfd5755fae03b2ff",
    "survival": "031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a",
    "guard": "0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b",
    "checkpoint": "7d55bec981884ce50aa986693e0e2f687b7f1f4e2ae6f609a833ec50228f506f",
    "reference": "c6d81aad4cc9f6c33e0fadbb5cb600f60acdd27924610b9c60b4d61aa9ff78f3",
}
SCENARIOS = (
    ("A_safe_light", False, False, ("stay", "migrate")),
    ("B_safe_busy", False, True, ("migrate", "stay")),
    ("C_guard_light", True, False, ("stay", "migrate")),
    ("D_guard_busy", True, True, ("migrate", "stay")),
)


def guard_contract(args: argparse.Namespace) -> tuple[int, str]:
    profile = getattr(args, "guard_profile", "legacy8448")
    if profile == "legacy8448":
        return 8448, EXPECTED_SHAS["guard"]
    if profile == "reduced2000" and getattr(args, "probability_pilot", False):
        return 2000, "1d8fa3c8ab49d50b30fccbbd901735d5896a5d7959a5ad7ccecb79c1c849cc66"
    raise ValueError("guard profile requires an explicit probability pilot contract")


def reference_contract(args: argparse.Namespace) -> str:
    profile = getattr(args, "slo_profile", "legacy1pct")
    if profile == "legacy1pct":
        return EXPECTED_SHAS["reference"]
    if profile == "slow2pct" and getattr(args, "probability_pilot", False):
        return "0bea93bd3af4e6d44b3164c14b6e3504666c8adf3ad8679c411f775936f4f6cb"
    raise ValueError("SLO profile requires an explicit probability pilot contract")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def expected_gpu_uuids(args: argparse.Namespace) -> list[str]:
    override = getattr(args, "expected_gpu_uuids", None)
    if override is None:
        return EXPECTED_UUIDS
    values = [value.strip() for value in override.split(",")]
    if len(values) != 5 or len(set(values)) != 5 or any(
        not value.startswith("GPU-") for value in values
    ):
        raise ValueError("expected GPU UUIDs must name five distinct GPUs")
    return values


def verify(args: argparse.Namespace) -> dict[str, Any]:
    guard_tokens, guard_sha = guard_contract(args)
    reference_sha = reference_contract(args)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != args.expected_revision:
        raise ValueError(f"HEAD differs: {revision}")
    portable_hardware = getattr(args, "portable_hardware", False)
    if not portable_hardware and socket.gethostname() != args.expected_host:
        raise ValueError(f"hostname differs: {socket.gethostname()}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).strip()
    if status:
        raise ValueError(f"tracked worktree changes: {status}")
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).strip().replace("\r", "").splitlines()
    expected_uuids = expected_gpu_uuids(args)
    if not portable_hardware and uuids != expected_uuids:
        raise ValueError(f"GPU UUIDs differ: {uuids}")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True).strip().replace("\r", "").splitlines()
    if names != ["NVIDIA A100-PCIE-40GB"] * 5:
        raise ValueError(f"GPU models differ: {names}")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader"], text=True).strip()
    if any(uuid in active for uuid in uuids):
        raise ValueError("an A100 already has a compute process")
    paths = {
        "input": args.input,
        "model_config": args.model / "config.json",
        "base": args.base,
        "survival": args.survival,
        "guard": args.guard,
        "checkpoint": args.checkpoint,
        "reference": args.reference,
    }
    measured = {name: sha256(path) for name, path in paths.items()}
    for name, digest in measured.items():
        expected = EXPECTED_SHAS[name]
        if name == "guard":
            expected = guard_sha
        if name == "reference":
            expected = reference_sha
        if name == "input" and getattr(args, "constructed_workload", False):
            if not getattr(args, "probability_pilot", False):
                raise ValueError(
                    "constructed input is restricted to probability collection")
            expected = getattr(args, "expected_input_sha256", None)
        if digest != expected:
            raise ValueError(f"{name} SHA differs: {paths[name]} {digest}")
    if args.guard.read_text(encoding="utf-8").strip() != str(guard_tokens):
        raise ValueError("guard value differs")
    return {
        "format_version": 1,
        "revision": revision,
        "hostname": socket.gethostname(),
        "gpu_uuids": uuids,
        "gpu_models": names,
        "portable_hardware": portable_hardware,
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "sha256": measured,
        "model_config_sha256": measured["model_config"],
        "input_path": str(args.input.resolve()),
        "input_sha256": measured["input"],
    }


def prepare(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from tools.bridge_tp.run_natural_eos_gap_probe_a100 import select_requests

    selected = select_requests(args.input, 60)
    matches = [row for row in selected if row["id"] == ANCHOR_ID]
    if len(matches) != 1 or matches[0]["workload_group"] != "long_form":
        raise ValueError("frozen anchor not found in test selection")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def token_ids(row: dict[str, Any]) -> list[int]:
        prompt = row.get("prompt")
        if prompt is None:
            prompt = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(prompt)

    anchor_tokens = token_ids(matches[0])
    if len(anchor_tokens) + 4096 > 8192:
        raise ValueError("anchor exceeds context")
    anchor = {
        "model": "bridgetp-model", "prompt": anchor_tokens,
        "max_tokens": 4096, "ignore_eos": False, "temperature": 0.0,
        "top_p": 1.0, "top_k": 0, "min_p": 0.0,
        "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "repetition_penalty": 1.0, "n": 1, "use_beam_search": False,
        "stream": True, "bridgetp_group_id": None,
        "bridgetp_group_longest": False,
    }
    anchor_path = run_dir / "inputs" / "natural_anchor.json"
    write_json(anchor_path, anchor)
    background = [row for row in selected if row["id"] != ANCHOR_ID]
    natural = [row for row in background if row["workload_group"] == "natural"]
    source = natural[:12]
    source_ids = {row["id"] for row in source}
    target = [row for row in background if row["id"] not in source_ids]
    if len(source) != 12 or len(target) != 47:
        raise ValueError("unexpected frozen source/target selection")

    manifests: dict[str, Any] = {}
    for name, near_guard, busy, order in SCENARIOS:
        jobs = []
        target_rows = target if busy else target[:2]
        for index, row in enumerate(target_rows):
            tokens = token_ids(row)
            if len(tokens) + 4096 > 8192:
                raise ValueError(f"target context exceeds limit: {row['id']}")
            jobs.append({
                "job_id": f"target_{index:03d}", "pool": "target",
                "start_after_s": 0.5 + index / 1.15,
                "request": {"model": "bridgetp-model", "prompt": tokens,
                            "max_tokens": 4096, "ignore_eos": False},
                "input_id": row["id"],
                "workload_group": row["workload_group"],
            })
        source_rows = source[:6] if near_guard else source
        for index, row in enumerate(source_rows):
            tokens = token_ids(row)
            if near_guard:
                # Frozen synthetic prefill pressure, scored separately from the
                # natural workload in any later formal GoodOutput comparison.
                padding = tokenizer.encode(
                    " Context for a detailed answer: the observations continue. ",
                    add_special_tokens=False)
                if not padding:
                    raise ValueError("pressure padding tokenized to an empty list")
                needed = max(0, 3584 - len(tokens))
                tokens = (tokens + padding * (
                    (needed + len(padding) - 1) // len(padding)))[:3584]
                if len(tokens) != 3584:
                    raise ValueError("source pressure prompt is too short")
            maximum = 512 if near_guard else 1024
            if len(tokens) + maximum > 8192:
                raise ValueError(f"source context exceeds limit: {row['id']}")
            job = {
                "job_id": f"source_{index:03d}", "pool": "source",
                "start_after_s": index * 0.1 if near_guard else 0.5 + index / 1.15,
                "request": {"model": "bridgetp-model", "prompt": tokens,
                            "max_tokens": maximum, "ignore_eos": False},
                "input_id": row["id"],
                "workload_group": (
                    "padded_prefill_pressure" if near_guard else row["workload_group"]),
            }
            if not near_guard:
                job["start_after_event"] = "ANCHOR_FIRST_OUTPUT"
            jobs.append(job)
        manifest = {
            "format_version": 1, "scenario": name,
            "status": "EXPLORATORY_PAIRED_MATRIX",
            "source_input": str(args.input.resolve()),
            "anchor_input_id": ANCHOR_ID,
            "near_guard_intended": near_guard,
            "target_busy_intended": busy,
            "synthetic_pressure_excluded_from_formal_benefit": near_guard,
            "arm_order": order,
            "jobs": jobs,
        }
        path = run_dir / "inputs" / f"{name}.json"
        write_json(path, manifest)
        manifests[name] = {
            "path": str(path), "sha256": sha256(path),
            "jobs": len(jobs), "source_jobs": len(source_rows),
            "target_jobs": len(target_rows), "arm_order": order,
        }
    return {
        "anchor_path": str(anchor_path),
        "anchor_sha256": sha256(anchor_path),
        "anchor_prompt_tokens": len(anchor_tokens),
        "manifests": manifests,
    }


def execute(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("running:", " ".join(command[:3]), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def online_command(args: argparse.Namespace, setup: dict[str, Any],
                   name: str, arm: str, root: Path) -> list[str]:
    guard_tokens, guard_sha = guard_contract(args)
    manifest = setup["manifests"][name]
    near_guard = name.startswith(("C_", "D_"))
    command = [
        sys.executable, "tools/bridge_tp/run_shadow_strategy_online_validation.py",
        "--phase", "smoke", "--repetitions", "1",
        "--model-path", str(args.model),
        "--manifest", manifest["path"],
        "--survival-table", str(args.survival),
        "--guard-file", str(args.guard),
        "--out-root", str(root),
        "--expected-revision", args.expected_revision,
        "--expected-manifest-sha256", manifest["sha256"],
        "--expected-survival-sha256", EXPECTED_SHAS["survival"],
        "--expected-guard-sha256", guard_sha,
        "--expected-guard", str(guard_tokens), "--tp1-blocks", "1968",
        "--tp4-blocks", "35739", "--shadow-only-only",
        "--gpu-resident-shadow", "--gpu-direct-history",
        "--gpu-direct-history-pacing", "--gpu-direct-delta",
        "--gpu-direct-delta-batch-tokens", "16",
        "--gpu-direct-delta-flush-ms", "25",
        "--ready-sync-mode", "STREAM_EVENT",
        "--ready-notification-mode", "UDP",
        "--persistent-channel", "--preconnect-persistent-channel",
        "--channel-generation", "1",
        "--manager-m0-shadow", "--manager-m1-auto-start",
        "--manager-m2-rate", "--m1-min-output-tokens", "96",
        "--m1-source-release-tail-s", "5.0",
        "--manager-m2-force-initial-high",
        "--manager-m2-expected-profile", "HIGH",
        "--manager-m2-min-history-byte-frac", "0.9",
        "--manager-m3-commit", "--manager-m4-cancel",
        "--manager-m5-predictor-shadow",
        "--predictor-checkpoint", str(args.checkpoint),
        "--predictor-checkpoint-sha256", EXPECTED_SHAS["checkpoint"],
        "--m2-low-gib-s", "0.5", "--m2-medium-gib-s", "2.4",
        "--m2-high-gib-s", "8.0",
        "--source-pressure", "--minimum-ready-source-jobs",
        "3" if near_guard else "0",
        "--trigger-output-tokens", "64", "--bridge-output-tokens", "96",
        "--commit-timing", "EARLIEST_READY", "--natural-eos-anchor",
        "--anchor-request-file", setup["anchor_path"],
        "--expected-anchor-request-sha256", setup["anchor_sha256"],
        "--anchor-prompt-tokens", str(setup["anchor_prompt_tokens"]),
        "--anchor-max-tokens", "4096", "--minimum-ready-target-jobs", "2",
        "--tp1-gpu", "0", "--tp4-gpus", "1,2,3,4",
        "--tp1-port", "8001", "--tp4-port", "8200",
        "--snapshot-port", "29800", "--delta-port", "29900",
        "--delivery-port", "30000", "--gpu-direct-base-port", "30400",
        "--ready-notification-port", "30500",
    ]
    if arm == "stay":
        command.append("--paired-stay")
    return command


def observed_pressure(run_root: Path) -> dict[str, Any]:
    audit = run_root / "controller" / "phase9_audit.jsonl"
    if not audit.is_file():
        return {"near_guard_verified": False, "reason": "no TP1 telemetry"}
    rows = []
    for line in audit.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    samples = [
        row for row in rows if row.get("kind") == "telemetry"
        and isinstance(row.get("tp1"), dict)
        and row["tp1"].get("free_kv_blocks") is not None
        and row["tp1"].get("block_size") is not None
    ]
    if not samples:
        return {"near_guard_verified": False, "reason": "no free KV samples"}
    anchor_samples = [
        row for row in samples if int(row.get("output_tokens") or 0) > 0
    ]
    if not anchor_samples:
        return {"near_guard_verified": False,
                "reason": "no anchor output telemetry"}
    decision_time = float(anchor_samples[0]["unix_s"])
    sample = min(samples, key=lambda row: abs(float(row["unix_s"])
                                          - decision_time))
    tp1 = sample["tp1"]
    free = int(tp1["free_kv_blocks"]) * int(tp1["block_size"])
    return {
        "free_kv_tokens_at_anchor_first_output": free,
        "sample_offset_s": abs(float(sample["unix_s"]) - decision_time),
        "near_guard_verified": (
            8448 <= free <= 10496
            and abs(float(sample["unix_s"]) - decision_time) <= 1.0),
        "peak_kv_usage_frac": max(
            (float(row["tp1"].get("kv_usage_frac", 0)) for row in samples),
            default=None),
    }


def collect(args: argparse.Namespace, run_dir: Path,
            setup: dict[str, Any], outcomes: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"format_version": 1,
                               "status": "EXPLORATORY_NOT_FITTING_DATA",
                               "scenarios": {}}
    for name, near_guard, busy, order in SCENARIOS:
        arms = outcomes[name]
        metrics = {}
        request_ids = {}
        completions = {}
        for arm in ("stay", "migrate"):
            path = run_dir / name / f"{arm}.slo_v6.json"
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("computable") and not value.get("errors"):
                    metrics[arm] = value["metrics"]
                    request_ids[arm] = {
                        row["request_id"] for row in value["request_rows"]
                    }
            background_path = (run_dir / name / arm / "r01_shadow_only"
                               / "background" / "background_summary.json")
            if background_path.is_file():
                background = json.loads(
                    background_path.read_text(encoding="utf-8"))
                completions[arm] = {
                    "all_completed": (
                        background["completed"]
                        == setup["manifests"][name]["jobs"]
                        and background["failed"] == 0),
                    "all_natural_stop": all(
                        result.get("finish_reason") == "stop"
                        for result in background["results"]),
                }
        pair_complete = (
            len(metrics) == 2 and len(completions) == 2
            and all(arms.get(f"{arm}_runner_rc") == 0
                    and arms.get(f"{arm}_audit_rc") == 0
                    and completions[arm]["all_completed"]
                    and (near_guard or completions[arm]["all_natural_stop"])
                    for arm in ("stay", "migrate"))
            and request_ids.get("stay") == request_ids.get("migrate")
        )
        row = {
            "near_guard_intended": near_guard,
            "target_busy_intended": busy,
            "arm_order": order,
            "manifest": setup["manifests"][name],
            "arms": arms,
            "v6_metrics": metrics,
            "background_completion": completions,
            "pair_complete": pair_complete,
            "delta_goodoutput_tokens_s": (
                metrics["migrate"]["goodoutput_tokens_s"]
                - metrics["stay"]["goodoutput_tokens_s"]
                if pair_complete else None),
            "formal_benefit_eligible": False,
            "limitation": (
                "synthetic padded source pressure; diagnostic only"
                if near_guard else "single natural-EOS pair; repeat before fitting"),
        }
        if near_guard:
            row["observed_pressure"] = {
                arm: observed_pressure(
                    run_dir / name / arm / "r01_shadow_only")
                for arm in ("stay", "migrate")
            }
            row["scenario_verified"] = all(
                value["near_guard_verified"]
                for value in row["observed_pressure"].values())
        summary["scenarios"][name] = row
    write_json(run_dir / "matrix_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--expected-gpu-uuids")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--survival", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight = verify(args)
    run_dir = args.out_dir.resolve()
    if run_dir.exists():
        raise ValueError(f"output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json(run_dir / "preflight.json", preflight)
    setup = prepare(args, run_dir)
    write_json(run_dir / "inputs" / "matrix_setup.json", setup)
    outcomes: dict[str, Any] = {}
    try:
        for name, near_guard, busy, order in SCENARIOS:
            print(f"=== {name}: TP1={'near guard' if near_guard else 'safe'}, "
                  f"TP4={'busy' if busy else 'light'} ===", flush=True)
            info = setup["manifests"][name]
            validation = [
                sys.executable, "tools/bridge_tp/run_phase9_capacity_background.py",
                "--manifest", info["path"], "--out-dir",
                str(run_dir / name / "manifest-validation"), "--validate-only",
            ]
            outcome = {"manifest_validation_rc": execute(
                validation, run_dir / name / "manifest-validation.log")}
            outcomes[name] = outcome
            if outcome["manifest_validation_rc"]:
                continue
            for arm in order:
                print(f"=== {name} {arm} ===", flush=True)
                arm_dir = run_dir / name / arm
                outcome[f"{arm}_runner_rc"] = execute(
                    online_command(args, setup, name, arm, arm_dir),
                    run_dir / name / f"{arm}.console.log")
                root = arm_dir / "r01_shadow_only"
                audit_cmd = [
                    sys.executable, "tools/bridge_tp/audit_slo_v6.py",
                    "--run-root", str(root), "--reference", str(args.reference),
                    "--preflight-json", str(run_dir / "preflight.json"),
                    "--require-reference-match", "--out-json",
                    str(run_dir / name / f"{arm}.slo_v6.json"),
                ]
                outcome[f"{arm}_audit_rc"] = execute(
                    audit_cmd, run_dir / name / f"{arm}.slo_v6.console.log")
            write_json(run_dir / name / "outcome.json", outcome)
    finally:
        for name, *_ in SCENARIOS:
            outcomes.setdefault(name, {"status": "NOT_STARTED"})
        summary = collect(args, run_dir, setup, outcomes)
        archive = run_dir.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(run_dir, arcname=run_dir.name)
        print(json.dumps({
            "archive_to_retrieve": str(archive),
            "matrix_summary": str(run_dir / "matrix_summary.json"),
            "complete_pairs": sum(
                len(row["v6_metrics"]) == 2
                for row in summary["scenarios"].values()),
        }), flush=True)


if __name__ == "__main__":
    main()
