#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate repeatability, guarded pressure, and late cutover in one A100 batch."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp.run_goodoutput_matrix_a100 import (
    EXPECTED_SHAS,
    execute,
    online_command,
    parse_args,
    prepare,
    sha256,
    verify,
    write_json,
)

GUARD_TOKENS = 8448
REPEATS = 3
PRESSURE_START_FREE_LIMIT = 14000


def build_natural_pressure(
    args: argparse.Namespace, run_dir: Path, setup: dict[str, Any]
) -> None:
    from transformers import AutoTokenizer

    rows = {
        row["id"]: row
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": "<CONTEXT>"}],
        tokenize=False, add_generation_prompt=True)
    if template.count("<CONTEXT>") != 1:
        raise ValueError("chat template did not retain the context marker")
    start, end = template.split("<CONTEXT>")
    prefix = tokenizer.encode(
        start + "Read the following context.\n", add_special_tokens=False)
    suffix = tokenizer.encode(
        "\nWrite a detailed synthesis of about 250 words. Explain the main "
        "points and end with a brief conclusion." + end,
        add_special_tokens=False)
    budget = 3584 - len(prefix) - len(suffix)
    if budget < 3000:
        raise ValueError("chat template leaves too little context room")

    for name, base_name in (
        ("C_guard_light", "A_safe_light"),
        ("D_guard_busy", "B_safe_busy"),
    ):
        base = json.loads(Path(
            setup["manifests"][base_name]["path"]).read_text(encoding="utf-8"))
        source_jobs = [job for job in base["jobs"] if job["pool"] == "source"]
        target_jobs = [copy.deepcopy(job) for job in base["jobs"]
                       if job["pool"] == "target"]
        pressure_jobs = []
        for index, old in enumerate(source_jobs[:5]):
            row = rows[old["input_id"]]
            content = "\n".join(
                str(message.get("content", "")) for message in row["messages"])
            body = tokenizer.encode(content + "\n", add_special_tokens=False)
            if not body:
                raise ValueError(f"empty source context: {row['id']}")
            repeat = (budget + len(body) - 1) // len(body)
            prompt = prefix + (body * repeat)[:budget] + suffix
            if len(prompt) != 3584:
                raise ValueError("pressure prompt length differs from 3584")
            pressure_jobs.append({
                "job_id": f"source_{index:03d}", "pool": "source",
                "start_after_s": index * 0.1,
                "start_after_event": "ANCHOR_FIRST_OUTPUT",
                "request": {"model": "bridgetp-model", "prompt": prompt,
                            "max_tokens": 768, "ignore_eos": False},
                "input_id": row["id"],
                "workload_group": "augmented_long_context_natural_eos",
            })
        manifest = {
            "format_version": 1, "scenario": name,
            "status": "GATE_VALIDATION_AUGMENTED_CONTEXT",
            "source_input": str(args.input.resolve()),
            "anchor_input_id": base["anchor_input_id"],
            "pressure_prompt_tokens": 3584,
            "pressure_request_max_tokens": 768,
            "natural_eos_required": True,
            "pressure_start_event": "ANCHOR_FIRST_OUTPUT",
            "diagnostic_m1_max_source_free_kv_tokens": (
                PRESSURE_START_FREE_LIMIT),
            "jobs": target_jobs + pressure_jobs,
        }
        path = run_dir / "inputs" / f"{name}.json"
        write_json(path, manifest)
        setup["manifests"][name] = {
            "path": str(path), "sha256": sha256(path),
            "jobs": len(manifest["jobs"]), "source_jobs": 5,
            "target_jobs": len(target_jobs),
        }


def replace_option(command: list[str], name: str, value: str) -> None:
    index = command.index(name)
    command[index + 1] = value


def configure_late_command(command: list[str]) -> None:
    replace_option(command, "--m1-min-output-tokens", "1024")
    replace_option(command, "--trigger-output-tokens", "1000")
    replace_option(command, "--bridge-output-tokens", "1024")
    # The controller still validates this configured window when the runtime
    # cutover is EARLIEST_READY. Its upper boundary must exceed the trigger.
    command += ["--cutover-output-tokens", "1120"]


def configure_pressure_command(command: list[str]) -> None:
    replace_option(command, "--minimum-ready-source-jobs", "0")
    replace_option(command, "--m1-min-output-tokens", "192")
    replace_option(command, "--cutover-output-tokens", "320")
    command += [
        "--background-lead-s", "0",
        "--diagnostic-m1-max-source-free-kv-tokens",
        str(PRESSURE_START_FREE_LIMIT),
    ]


def pressure_evidence(root: Path) -> dict[str, Any]:
    audit = root / "controller" / "phase9_audit.jsonl"
    if not audit.is_file():
        return {"valid": False, "reason": "missing controller audit"}
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
        return {"valid": False, "reason": "missing free KV telemetry"}
    free = [int(row["tp1"]["free_kv_blocks"])
            * int(row["tp1"]["block_size"]) for row in samples]
    first_output = next(
        (index for index, row in enumerate(samples)
         if int(row.get("output_tokens") or 0) > 0), None)
    if first_output is None:
        return {"valid": False, "reason": "missing anchor output telemetry"}
    preemptions = [int(row["tp1"].get("preemptions_total") or 0)
                   for row in samples]
    starts = [row for row in rows
              if row.get("kind") == "manager_m1_start_decision"
              and row.get("decision", {}).get("action") == "START_SHADOW"]
    start = starts[0] if starts else None
    start_free = (start.get("snapshot", {}).get("source_free_kv_tokens")
                  if start else None)
    start_output = (start.get("snapshot", {}).get("generated_tokens")
                    if start else None)
    start_unix_s = start.get("unix_s") if start else None
    background_path = root / "background" / "background_summary.json"
    if background_path.is_file() and start_unix_s is not None:
        background = json.loads(background_path.read_text(encoding="utf-8"))
        active_sources = sum(
            row.get("pool") == "source"
            and (row.get("first_token_unix_s") or float("inf"))
            <= start_unix_s < (row.get("request_ended_unix_s") or 0)
            for row in background.get("results", []))
    else:
        active_sources = 0
    return {
        "free_at_first_output": free[first_output],
        "minimum_free_kv_tokens": min(free),
        "samples_below_guard": sum(value < GUARD_TOKENS for value in free),
        "preemption_delta": max(preemptions) - preemptions[0],
        "m1_start_free_kv_tokens": start_free,
        "m1_start_output_tokens": start_output,
        "active_source_jobs_at_m1_start": active_sources,
        "valid": (min(free) >= GUARD_TOKENS
                  and max(preemptions) == preemptions[0]
                  and start_free is not None
                  and GUARD_TOKENS < start_free <= PRESSURE_START_FREE_LIMIT
                  and active_sources >= 3),
    }


def pair_report(run_dir: Path, key: str, arms: dict[str, Any],
                *, guard_pressure: bool, late: bool) -> dict[str, Any]:
    path = run_dir / key
    details = {}
    for arm in ("stay", "migrate"):
        root = path / arm / "r01_shadow_only"
        slo_path = path / f"{arm}.slo_v6.json"
        background_path = root / "background" / "background_summary.json"
        acceptance_path = root / "provenance" / "shadow_online_acceptance.json"
        if not (slo_path.is_file() and background_path.is_file()
                and acceptance_path.is_file()):
            details[arm] = {"valid": False, "reason": "missing arm evidence"}
            continue
        slo = json.loads(slo_path.read_text(encoding="utf-8"))
        background = json.loads(background_path.read_text(encoding="utf-8"))
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        natural = all(row.get("finish_reason") == "stop"
                      for row in background["results"])
        pressure = pressure_evidence(root) if guard_pressure else None
        arm_valid = (
            arms.get(f"{arm}_runner_rc") == 0
            and arms.get(f"{arm}_audit_rc") == 0
            and acceptance.get("status") == "PASS"
            and not acceptance.get("errors")
            and slo.get("computable")
            and slo.get("reference_applicability")
            == "VERIFIED_GPU_AND_MODEL_CONFIG"
            and not slo.get("errors")
            and background["failed"] == 0
            and background["completed"] == background["jobs"]
            and natural
            and slo["metrics"]["slo_attainment"] >= 0.95
            and (not guard_pressure or pressure["valid"])
            and (arm == "stay" or (
                acceptance.get("handoff_stall_ms") is not None
                and acceptance["handoff_stall_ms"] <= 1000))
        )
        details[arm] = {
            "valid": bool(arm_valid),
            "slo_metrics": slo.get("metrics"),
            "request_ids": sorted(row["request_id"]
                                  for row in slo.get("request_rows", [])),
            "all_background_natural_eos": natural,
            "background_finish_reasons": {
                reason: sum(row.get("finish_reason") == reason
                            for row in background["results"])
                for reason in {row.get("finish_reason")
                               for row in background["results"]}},
            "pressure": pressure,
            "anchor_cutover_output_tokens": acceptance.get(
                "earliest_ready_cutover_output_tokens"),
            "handoff_stall_ms": acceptance.get("handoff_stall_ms"),
        }
    pair_valid = (
        len(details) == 2 and all(details[arm]["valid"]
                                  for arm in ("stay", "migrate"))
        and details["stay"]["request_ids"]
        == details["migrate"]["request_ids"])
    if late:
        cutover = details.get("migrate", {}).get(
            "anchor_cutover_output_tokens")
        pair_valid = bool(pair_valid and cutover is not None and cutover >= 1000)
    return {
        "valid": pair_valid, "arms": arms, "details": details,
        "delta_goodoutput_tokens_s": (
            details["migrate"]["slo_metrics"]["goodoutput_tokens_s"]
            - details["stay"]["slo_metrics"]["goodoutput_tokens_s"]
            if pair_valid else None),
    }


def summarize(run_dir: Path, outcomes: dict[str, Any]) -> dict[str, Any]:
    pairs = {}
    for name, result in outcomes.items():
        pairs[name] = pair_report(
            run_dir, name, result,
            guard_pressure=name.startswith(("C_", "D_")),
            late=name.startswith("L_"))
    repeatability = {}
    for scenario in ("A_safe_light", "B_safe_busy"):
        rows = [value for key, value in pairs.items()
                if key.startswith(scenario + "/")]
        deltas = [row["delta_goodoutput_tokens_s"] for row in rows
                  if row["valid"]]
        same_sign = (all(x > 0 for x in deltas)
                     or all(x < 0 for x in deltas)) if len(deltas) == REPEATS else False
        repeatability[scenario] = {
            "complete_pairs": len(deltas),
            "required_pairs": REPEATS,
            "deltas": deltas,
            "mean_delta": statistics.mean(deltas) if deltas else None,
            "range": [min(deltas), max(deltas)] if deltas else None,
            "same_sign": same_sign,
            "gate_pass": len(deltas) == REPEATS and same_sign,
        }
    guard_pass = all(pairs.get(name, {}).get("valid", False) for name in (
        "C_guard_light/r01", "D_guard_busy/r01"))
    late_pass = pairs.get("L_late_light/r01", {}).get("valid", False)
    report = {
        "format_version": 1,
        "status": "EXPLORATORY_GATE_VALIDATION",
        "repeatability": repeatability,
        "guard_natural_eos_gate_pass": guard_pass,
        "late_cutover_gate_pass": late_pass,
        "ready_for_large_paired_capture": (
            all(row["gate_pass"] for row in repeatability.values())
            and guard_pass and late_pass),
        "notes": [
            "Guard prompts are augmented from frozen OASST1 inputs; natural EOS "
            "and guard safety must pass the reported gates; representativeness "
            "requires review.",
            "A/B repeats estimate run-to-run variation; no benefit function is "
            "fitted in this batch.",
        ],
        "pairs": pairs,
    }
    write_json(run_dir / "gate_summary.json", report)
    return report


def main() -> None:
    args = parse_args()
    scope = os.environ.get("BRIDGETP_GOODOUTPUT_GATE_SCOPE", "all")
    if scope not in {"all", "pressure"}:
        raise ValueError(f"unknown GoodOutput gate scope: {scope}")
    preflight = verify(args)
    run_dir = args.out_dir.resolve()
    if run_dir.exists():
        raise ValueError(f"output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json(run_dir / "preflight.json", preflight)
    setup = prepare(args, run_dir)
    build_natural_pressure(args, run_dir, setup)
    write_json(run_dir / "inputs" / "gate_setup.json", setup)
    plan = []
    if scope == "all":
        for repetition in range(1, REPEATS + 1):
            for name in ("A_safe_light", "B_safe_busy"):
                order = (("stay", "migrate") if
                         (repetition + (name.startswith("B_"))) % 2
                         else ("migrate", "stay"))
                plan.append((f"{name}/r{repetition:02d}", name, order, False))
    for name in ("C_guard_light", "D_guard_busy"):
        plan.append((f"{name}/r01", name, ("stay", "migrate"), False))
    plan.append(("L_late_light/r01", "A_safe_light",
                 ("migrate", "stay"), True))
    outcomes: dict[str, Any] = {}
    try:
        for key, name, order, late in plan:
            print(f"=== {key} order={order} ===", flush=True)
            info = setup["manifests"][name]
            pair_dir = run_dir / key
            validation = [
                sys.executable,
                "tools/bridge_tp/run_phase9_capacity_background.py",
                "--manifest", info["path"], "--out-dir",
                str(pair_dir / "manifest-validation"), "--validate-only",
            ]
            outcome = {"manifest_validation_rc": execute(
                validation, pair_dir / "manifest-validation.log")}
            outcomes[key] = outcome
            if outcome["manifest_validation_rc"]:
                continue
            for arm in order:
                root = pair_dir / arm
                command = online_command(args, setup, name, arm, root)
                if late:
                    configure_late_command(command)
                if name.startswith(("C_", "D_")):
                    configure_pressure_command(command)
                outcome[f"{arm}_runner_rc"] = execute(
                    command, pair_dir / f"{arm}.console.log")
                audit = [
                    sys.executable, "tools/bridge_tp/audit_slo_v6.py",
                    "--run-root", str(root / "r01_shadow_only"),
                    "--reference", str(args.reference),
                    "--preflight-json", str(run_dir / "preflight.json"),
                    "--require-reference-match", "--out-json",
                    str(pair_dir / f"{arm}.slo_v6.json"),
                ]
                outcome[f"{arm}_audit_rc"] = execute(
                    audit, pair_dir / f"{arm}.slo_v6.console.log")
            write_json(pair_dir / "outcome.json", outcome)
    finally:
        for key, *_ in plan:
            outcomes.setdefault(key, {"status": "NOT_STARTED"})
        report = summarize(run_dir, outcomes)
        report["scope"] = scope
        write_json(run_dir / "gate_summary.json", report)
        archive = run_dir.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(run_dir, arcname=run_dir.name)
        print(json.dumps({
            "archive_to_retrieve": str(archive),
            "gate_summary": str(run_dir / "gate_summary.json"),
            "ready_for_large_paired_capture": report[
                "ready_for_large_paired_capture"],
        }), flush=True)


if __name__ == "__main__":
    main()
