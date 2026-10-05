#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect a six-trace randomized natural-EOS migration timing pilot on A100.

The pilot checks the paired collection path and actual M1 start tokens. The
configured minimum output token is an eligibility gate, not an exact action.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp.run_goodoutput_matrix_a100 import (  # noqa: E402
    execute,
    online_command,
    sha256,
    verify,
    write_json,
)

SEED = 20261004
CASES = (
    ("p00_source1_target2", 1, 2, 1.2, 0.8),
    ("p01_source1_target16", 1, 16, 0.35, 0.8),
    ("p02_source3_target8", 3, 8, 0.65, 0.35),
    ("p03_source3_target24", 3, 24, 0.2, 0.35),
    ("p04_source5_target2", 5, 2, 1.2, 0.15),
    ("p05_source5_target24", 5, 24, 0.2, 0.15),
)
ACTIONS = ("stay", "early128", "late1024")
TIMING_ACTIONS = ("stay", "now", "wait")
RETIRED_P03_ANCHOR_ID = "oasst1:98d36c03-5335-4f5b-8dbd-1a6e5fd9b0b2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "input", "base", "survival", "guard",
                 "checkpoint", "reference", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-host")
    parser.add_argument("--expected-gpu-uuids")
    parser.add_argument("--portable-hardware", action="store_true",
                        help="Record actual AutoDL host/UUIDs; require five idle A100s")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--paired-only", action="store_true",
                        help="Collect STAY and early128 without the late EOS arm")
    parser.add_argument("--timing-pilot", action="store_true",
                        help="Compare STAY, NOW, and one M5-refresh WAIT")
    parser.add_argument("--actions", nargs="+", choices=TIMING_ACTIONS,
                        help="Collect only these timing actions")
    parser.add_argument("--cases", nargs="+", choices=[row[0] for row in CASES],
                        help="Run only these cases with the original seed")
    parser.add_argument("--anchor-context-limit", action="store_true",
                        help="Use the largest anchor output allowed by context")
    parser.add_argument(
        "--anchor-total-max-tokens", type=int,
        help="client-visible total output budget after TP1-to-TP4 migration",
    )
    parser.add_argument(
        "--p03-anchor-id",
        help="held-out long-form request ID replacing the p03 anchor",
    )
    parser.add_argument("--background-context-limit", action="store_true",
                        help="Use the largest background output allowed by context")
    parser.add_argument("--background-max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp4-max-model-len", type=int)
    parser.add_argument("--random-arrivals", action="store_true",
                        help="Use seeded exponential interarrival gaps")
    parser.add_argument("--evaluation-horizon-s", type=float, default=180.0)
    return parser.parse_args()


def select_inputs(path: Path, seed: int = SEED,
                  p03_anchor_id: str | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8")
            .splitlines() if line.strip()]
    tree_splits: dict[str, str] = {}
    for row in rows:
        tree_id = str(row.get("source_tree_id") or row["id"])
        split = str(row.get("split"))
        if tree_id in tree_splits and tree_splits[tree_id] != split:
            raise ValueError(f"request tree crosses dataset splits: {tree_id}")
        tree_splits[tree_id] = split
    test = [row for row in rows if row.get("split") == "test"
            and row.get("workload_group") in {"natural", "long_form"}
            and (p03_anchor_id is None
                 or row.get("id") != RETIRED_P03_ANCHOR_ID)]
    rng = random.Random(seed)
    long_form = [row for row in test
                 if row["workload_group"] == "long_form"]
    natural = [row for row in test
               if row["workload_group"] == "natural"]
    rng.shuffle(long_form)
    rng.shuffle(natural)
    if len(long_form) < 5 or len(natural) < 1:
        raise ValueError("pilot needs five held-out long-form anchors")
    anchors = long_form[:5] + natural[:1]
    if p03_anchor_id is not None:
        replacements = [row for row in long_form
                        if str(row["id"]) == p03_anchor_id]
        if len(replacements) != 1:
            raise ValueError("p03 replacement must be one held-out long-form row")
        replacement = replacements[0]
        existing = next((index for index, row in enumerate(anchors)
                         if row["id"] == p03_anchor_id), None)
        if existing is not None and existing != 3:
            anchors[existing], anchors[3] = anchors[3], anchors[existing]
        else:
            anchors[3] = replacement
    selected_ids = {str(row["id"]) for row in anchors}
    if len(selected_ids) != len(anchors):
        raise ValueError("duplicate anchor request IDs")
    others = [row for row in test if str(row["id"]) not in selected_ids]
    rng.shuffle(others)
    required = sum(source + target for _, source, target, *_ in CASES)
    if len(others) < required:
        raise ValueError(f"pilot needs {required} unique background requests")
    selected = anchors + others[:required]
    if len({str(row["id"]) for row in selected}) != len(selected):
        raise ValueError("pilot request IDs are not unique")
    return selected


def build_setup(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    seed = getattr(args, "seed", SEED)
    rows = select_inputs(args.input, seed, getattr(args, "p03_anchor_id", None))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def tokens(row: dict[str, Any]) -> list[int]:
        prompt = row.get("prompt")
        if prompt is None:
            prompt = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True)
        value = tokenizer.encode(prompt)
        if not value:
            raise ValueError(f"empty prompt for {row['id']}")
        return value

    setup: dict[str, Any] = {
        "format_version": 1, "seed": seed,
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "manifests": {}, "anchors": {}, "cases": [],
    }
    cursor = len(CASES)
    for index, (name, source_count, target_count, target_spacing,
                source_spacing) in enumerate(CASES):
        arrival_rng = random.Random(f"{seed}:{name}:arrivals")
        offsets: dict[str, list[float]] = {}
        for pool, count, spacing, initial in (
            ("target", target_count, target_spacing, 0.5),
            ("source", source_count, source_spacing, 0.2),
        ):
            values = []
            elapsed = initial
            for job_index in range(count):
                if job_index:
                    gap = (arrival_rng.expovariate(1 / spacing)
                           if getattr(args, "random_arrivals", False)
                           else spacing)
                    elapsed += gap
                values.append(round(elapsed, 3))
            offsets[pool] = values
        anchor = rows[index]
        anchor_tokens = tokens(anchor)
        source_max_model_len = getattr(args, "max_model_len", 8192)
        anchor_cap = (min(source_max_model_len - len(anchor_tokens),
                          source_max_model_len - 128)
                      if getattr(args, "anchor_context_limit", False)
                      else 4096)
        total_budget = getattr(args, "anchor_total_max_tokens", None)
        if total_budget is None:
            total_budget = anchor_cap
        if (anchor_cap <= 128
                or len(anchor_tokens) + anchor_cap > source_max_model_len):
            raise ValueError(f"anchor exceeds context limit: {anchor['id']}")
        target_max_model_len = (
            getattr(args, "tp4_max_model_len", None) or source_max_model_len
        )
        if (total_budget < anchor_cap
                or len(anchor_tokens) + total_budget > target_max_model_len):
            raise ValueError(f"anchor total budget exceeds TP4: {anchor['id']}")
        anchor_request = {
            "model": "bridgetp-model", "prompt": anchor_tokens,
            "max_tokens": anchor_cap, "ignore_eos": False,
            "temperature": 0.0, "top_p": 1.0, "top_k": 0,
            "min_p": 0.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "repetition_penalty": 1.0,
            "n": 1, "use_beam_search": False, "stream": True,
            "bridgetp_group_id": None,
            "bridgetp_group_longest": False,
        }
        anchor_path = root / "inputs" / f"{name}_anchor.json"
        write_json(anchor_path, anchor_request)
        setup["anchors"][name] = {
            "path": str(anchor_path), "sha256": sha256(anchor_path),
            "prompt_tokens": len(anchor_tokens), "input_id": anchor["id"],
            "max_tokens": anchor_cap,
            "total_max_tokens": total_budget,
            "workload_group": anchor["workload_group"],
        }
        jobs = []
        chosen = rows[cursor:cursor + source_count + target_count]
        cursor += len(chosen)
        for pool, subset in (("target", chosen[:target_count]),
                             ("source", chosen[target_count:])):
            for job_index, row in enumerate(subset):
                prompt_tokens = tokens(row)
                background_cap = (
                    min(8192 - len(prompt_tokens), 8192 - 128)
                    if getattr(args, "background_context_limit", False)
                    else getattr(args, "background_max_tokens", 2048)
                )
                if (background_cap <= 0
                        or len(prompt_tokens) + background_cap > 8192):
                    raise ValueError(f"background exceeds context: {row['id']}")
                job = {
                    "job_id": f"{pool}_{job_index:03d}", "pool": pool,
                    "start_after_s": offsets[pool][job_index],
                    "request": {
                        "model": "bridgetp-model", "prompt": prompt_tokens,
                        "max_tokens": background_cap,
                        "ignore_eos": False,
                        "temperature": 0.0,
                    },
                    "input_id": row["id"],
                    "workload_group": row["workload_group"],
                }
                if pool == "source":
                    job["start_after_event"] = "ANCHOR_FIRST_OUTPUT"
                jobs.append(job)
        manifest = {
            "format_version": 1, "scenario": name,
            "status": "RANDOMIZED_GOODOUTPUT_PILOT",
            "seed": seed, "source_input": str(args.input.resolve()),
            "source_input_sha256": sha256(args.input),
            "anchor_input_id": anchor["id"],
            "source_count": source_count, "target_count": target_count,
            "source_spacing_s": source_spacing,
            "target_spacing_s": target_spacing,
            "natural_eos_required": True,
            "arrival_process": ("seeded_exponential"
                                if getattr(args, "random_arrivals", False)
                                else "fixed_spacing"),
            "jobs": jobs,
        }
        manifest_path = root / "inputs" / f"{name}.json"
        write_json(manifest_path, manifest)
        setup["manifests"][name] = {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "jobs": len(jobs), "source_jobs": source_count,
            "target_jobs": target_count,
        }
        setup["cases"].append(name)
    if cursor != len(rows):
        raise ValueError("pilot selection was not consumed exactly once")
    return setup


def configure_action(command: list[str], action: str) -> None:
    if action not in ACTIONS + TIMING_ACTIONS:
        raise ValueError(f"unknown pilot action: {action}")
    for flag in ("--manager-m2-force-initial-high",):
        command.remove(flag)
    for flag in ("--manager-m2-expected-profile",
                 "--manager-m2-min-history-byte-frac"):
        index = command.index(flag)
        del command[index:index + 2]
    threshold = 1024 if action == "late1024" else 128
    replacements = {
        "--m1-min-output-tokens": str(threshold),
        "--trigger-output-tokens": "1000" if threshold == 1024 else "64",
        "--bridge-output-tokens": "1024" if threshold == 1024 else "96",
    }
    for flag, value in replacements.items():
        command[command.index(flag) + 1] = value
    # Natural EOS and earliest-ready can leave no BRIDGE-window tokens.
    # Keep the measured windows but do not require a fixed sample count.
    command += ["--minimum-window-samples", "0"]
    command += ["--cutover-output-tokens",
                "1120" if threshold == 1024 else "320"]
    if action in {"now", "wait"}:
        command += ["--experiment-m1-action", action.upper()]


def action_order(name: str, seed: int = SEED,
                 actions: tuple[str, ...] = ACTIONS) -> list[str]:
    actions = list(actions)
    random.Random(f"{seed}:{name}").shuffle(actions)
    return actions


def observed_action(run: Path, *, paired_stay: bool = False) -> dict[str, Any]:
    audit = run / "controller" / "phase9_audit.jsonl"
    if not audit.is_file():
        return {"audit_available": False}
    rows = []
    for line in audit.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    starts = [row for row in rows
              if row.get("kind") == "manager_m1_start_decision"
              and row.get("decision", {}).get("action") == "START_SHADOW"]
    start_tokens = [row.get("snapshot", {}).get("generated_tokens")
                    for row in starts]
    decisions = [row for row in rows
                 if row.get("kind") == "manager_m1_start_decision"]
    timing = [row for row in rows
              if row.get("kind") == "experiment_m1_timing"]
    latest_m5: dict[str, Any] | None = None
    first_candidate: dict[str, Any] | None = None
    for row in rows:
        if row.get("kind") == "manager_m5_predictor_shadow":
            latest_m5 = row
        if row.get("kind") != "manager_m1_start_decision":
            continue
        snapshot = row.get("snapshot") or {}
        output_tokens = snapshot.get("generated_tokens")
        if not isinstance(output_tokens, int) or output_tokens < 128:
            continue
        decision = row.get("decision") or {}
        matching_m5 = (latest_m5 if latest_m5 is not None
                       and latest_m5.get("output_tokens") == output_tokens
                       else None)
        first_candidate = {
            "output_tokens": output_tokens,
            "target_running": snapshot.get("target_running"),
            "target_waiting": snapshot.get("target_waiting"),
            "target_kv_usage_frac": snapshot.get("target_kv_usage_frac"),
            "source_free_kv_tokens": snapshot.get("source_free_kv_tokens"),
            "source_guard_free_kv_tokens": snapshot.get(
                "source_guard_free_kv_tokens"),
            "source_prefill_pending_kv_tokens": snapshot.get(
                "source_prefill_pending_kv_tokens"),
            "source_time_to_guard_s": decision.get("source_time_to_guard_s"),
            "estimated_preparation_s": decision.get("estimated_preparation_s"),
            "m1_action": decision.get("action"),
            "m1_reason": decision.get("reason"),
            "m5_status": matching_m5.get("status") if matching_m5 else None,
            "p_remaining_gt_512_lower": (
                (matching_m5.get(
                    "p_remaining_gt_long_window_runtime_bounds") or [None])[0]
                if matching_m5 else None),
        }
        break
    acceptance = run / "provenance" / "shadow_online_acceptance.json"
    accepted = (json.loads(acceptance.read_text(encoding="utf-8"))
                if acceptance.is_file() else {})
    return {
        "audit_available": True,
        "start_count": 0 if paired_stay else len(starts),
        "actual_m1_start_output_tokens": [] if paired_stay else start_tokens,
        "m1_recommendation_output_tokens": start_tokens,
        "m1_decision_count": len(decisions),
        "timing_gate_reasons": [row.get("gate_reason") for row in timing
                                if row.get("gate_reason")],
        "timing_first_natural_start_output_tokens": next(
            (row.get("output_tokens") for row in timing
             if (row.get("natural_decision") or {}).get("action")
             == "START_SHADOW"), None),
        "first_candidate_at_or_after_128": first_candidate,
        "last_m1_decision": decisions[-1].get("decision")
        if decisions else None,
        "actual_cutover_output_tokens": accepted.get(
            "earliest_ready_cutover_output_tokens"),
        "handoff_stall_ms": accepted.get("handoff_stall_ms"),
        "acceptance_status": accepted.get("status"),
        "acceptance_errors": accepted.get("errors"),
    }


def is_context_censored(*, runner_rc: int, audit_rc: int,
                        source: dict[str, Any], anchor_cap: int,
                        acceptance_errors: list[str],
                        total_cap: int | None = None,
                        target: dict[str, Any] | None = None,
                        proxy: dict[str, Any] | None = None) -> bool:
    """Recognize a clean run stopped only by an output length cap."""
    source_capped = (
        source.get("finish_reason") == "length"
        and len(source.get("token_ids") or []) == anchor_cap
        and acceptance_errors == ["source did not naturally finish before its cap"]
    )
    target_capped = (
        (target or {}).get("finish_reason") == "length"
        and (proxy or {}).get("emitted_tokens") == (total_cap or anchor_cap)
        and acceptance_errors == [
            "unified response did not naturally finish before cap"]
    )
    return runner_rc != 0 and audit_rc == 0 and (source_capped or target_capped)


def collect_arm(args: argparse.Namespace, root: Path,
                setup: dict[str, Any], name: str,
                action: str) -> dict[str, Any]:
    case_root = root / name
    arm_root = case_root / action
    command_setup = {
        "anchor_path": setup["anchors"][name]["path"],
        "anchor_sha256": setup["anchors"][name]["sha256"],
        "anchor_prompt_tokens": setup["anchors"][name]["prompt_tokens"],
        "manifests": setup["manifests"],
    }
    command = online_command(args, command_setup, name,
                             "stay" if action == "stay" else "migrate",
                             arm_root)
    command.extend(("--max-model-len", str(args.max_model_len)))
    if args.tp4_max_model_len is not None:
        command.extend(("--tp4-max-model-len",
                        str(args.tp4_max_model_len)))
    command[command.index("--anchor-max-tokens") + 1] = str(
        setup["anchors"][name]["max_tokens"])
    if (setup["anchors"][name]["total_max_tokens"]
            > setup["anchors"][name]["max_tokens"]):
        command.extend(("--anchor-total-max-tokens", str(
            setup["anchors"][name]["total_max_tokens"])))
    configure_action(command, action)
    write_json(case_root / f"{action}.command.json", command)
    runner_rc = execute(command, case_root / f"{action}.console.log")
    run = arm_root / "r01_shadow_only"
    audit_path = case_root / f"{action}.slo_v6.json"
    audit_command = [
        sys.executable, "tools/bridge_tp/audit_slo_v6.py",
        "--run-root", str(run), "--reference", str(args.reference),
        "--preflight-json", str(root / "preflight.json"),
        "--require-reference-match", "--out-json", str(audit_path),
    ]
    if args.portable_hardware:
        audit_command += ["--gpu-match-mode", "model"]
    audit_rc = execute(audit_command,
                       case_root / f"{action}.slo_v6.console.log")
    slo = (json.loads(audit_path.read_text(encoding="utf-8"))
           if audit_path.is_file() else {})
    if (args.max_model_len != 8192
            or (args.tp4_max_model_len or args.max_model_len) != 8192):
        slo["reference_applicability"] = "EXPLORATORY_CONTEXT_UNCALIBRATED"
        slo["comparison_scope"] = "diagnostic_only"
        write_json(audit_path, slo)
    background_path = run / "background" / "background_summary.json"
    background = (json.loads(background_path.read_text(encoding="utf-8"))
                  if background_path.is_file() else {})
    source_path = run / "controller" / "source_response.json"
    source = (json.loads(source_path.read_text(encoding="utf-8"))
              if source_path.is_file() else {})
    target_path = run / "controller" / "target_response.json"
    target = (json.loads(target_path.read_text(encoding="utf-8"))
              if target_path.is_file() else {})
    proxy_path = run / "controller" / "response_proxy_stats.json"
    proxy = (json.loads(proxy_path.read_text(encoding="utf-8"))
             if proxy_path.is_file() else {})
    finish_reasons = {
        reason: sum(row.get("finish_reason") == reason
                    for row in background.get("results", []))
        for reason in {row.get("finish_reason")
                       for row in background.get("results", [])}
    }
    observed = observed_action(run, paired_stay=action == "stay")
    acceptance_errors = observed.get("acceptance_errors") or []
    context_censored = is_context_censored(
        runner_rc=runner_rc, audit_rc=audit_rc, source=source,
        anchor_cap=setup["anchors"][name]["max_tokens"],
        total_cap=setup["anchors"][name]["total_max_tokens"],
        acceptance_errors=acceptance_errors, target=target, proxy=proxy,
    )
    natural_noop = (
        action != "stay" and audit_rc == 0
        and slo.get("computable") is True
        and observed.get("audit_available") is True
        and observed.get("start_count") == 0
        and source.get("finish_reason") == "stop"
    )
    result = {
        "assigned_action": action,
        "configured_eligibility_tokens": (
            1024 if action == "late1024" else 128),
        "runner_rc": runner_rc, "audit_rc": audit_rc,
        "observed_action": observed,
        "anchor_source_finish_reason": source.get("finish_reason"),
        "anchor_target_finish_reason": target.get("finish_reason"),
        "anchor_source_output_tokens": len(source.get("token_ids", [])),
        "natural_noop_needs_review": natural_noop,
        "context_censored": context_censored,
        "fatal_error": bool(audit_rc or (runner_rc and not natural_noop
                                          and not context_censored)),
        "slo_computable": slo.get("computable"),
        "slo_reference_applicability": slo.get("reference_applicability"),
        "slo_metrics": slo.get("metrics"),
        "slo_errors": slo.get("errors"),
        "background_jobs": background.get("jobs"),
        "background_completed": background.get("completed"),
        "background_finish_reasons": finish_reasons,
    }
    write_json(case_root / f"{action}.result.json", result)
    return result


def fixed_horizon_result(result: dict[str, Any], horizon_s: float) -> None:
    """Score a fully drained natural-EOS arm on one common horizon."""
    metrics = result.get("slo_metrics") or {}
    wall_s = metrics.get("wall_time_s")
    good_tokens = metrics.get("good_output_tokens")
    request_count = metrics.get("requests")
    background_count = result.get("background_jobs")
    completed = (isinstance(request_count, int) and request_count > 0
                 and request_count == metrics.get("completed_requests")
                 and isinstance(background_count, int)
                 and background_count == result.get("background_completed"))
    finish_reasons = result.get("background_finish_reasons")
    source_reason = result.get("anchor_source_finish_reason")
    target_reason = result.get("anchor_target_finish_reason")
    anchor_natural = (source_reason == "stop" if not target_reason
                      else source_reason == "abort" and target_reason == "stop")
    natural_eos = (anchor_natural
                   and isinstance(finish_reasons, dict)
                   and set(finish_reasons) <= {"stop"}
                   and sum(finish_reasons.values()) == background_count)
    eligible = (
        not result.get("fatal_error", True) and result.get("audit_rc") == 0
        and result.get("slo_reference_applicability")
        != "EXPLORATORY_CONTEXT_UNCALIBRATED"
        and completed and natural_eos and isinstance(wall_s, (int, float))
        and math.isfinite(wall_s) and wall_s <= horizon_s
        and isinstance(good_tokens, int) and good_tokens >= 0
    )
    result["fixed_horizon_eligible"] = eligible
    result["fixed_horizon_goodoutput_tokens_s"] = (
        good_tokens / horizon_s if eligible else None)
    result["fixed_horizon_exclusions"] = {
        "technical_or_slo_audit": bool(result.get("fatal_error", True)
                                       or result.get("audit_rc") != 0),
        "not_drained": not completed,
        "not_natural_eos": not natural_eos,
        "exceeds_horizon": not isinstance(wall_s, (int, float))
        or not math.isfinite(wall_s) or wall_s > horizon_s,
    }


def pair_results(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep descriptive paired deltas separate from fitted causal effects."""
    pairs = {}
    for name, arms in summary["cases"].items():
        stay = arms.get("stay", {})
        now = arms.get("early128", {})
        arm_eligible = bool(stay.get("fixed_horizon_eligible")
                            and now.get("fixed_horizon_eligible"))
        starts = now.get("observed_action", {}).get(
            "actual_m1_start_output_tokens", [])
        eligible = arm_eligible and bool(starts)
        pairs[name] = {
            "eligible": eligible,
            "assigned_now_actuated": bool(starts),
            "paired_arms_natural_eos": arm_eligible,
            "stay_predecision_state": stay.get("observed_action", {}).get(
                "first_candidate_at_or_after_128"),
            "now_predecision_state": now.get("observed_action", {}).get(
                "first_candidate_at_or_after_128"),
            "stay_goodoutput_tokens_s": stay.get(
                "fixed_horizon_goodoutput_tokens_s"),
            "now_goodoutput_tokens_s": now.get(
                "fixed_horizon_goodoutput_tokens_s"),
            "descriptive_delta_tokens_s": (
                now["fixed_horizon_goodoutput_tokens_s"]
                - stay["fixed_horizon_goodoutput_tokens_s"]
                if eligible else None),
            "stay_raw_output_tokens": (stay.get("slo_metrics") or {}).get(
                "raw_output_tokens"),
            "now_raw_output_tokens": (now.get("slo_metrics") or {}).get(
                "raw_output_tokens"),
        }
    return {
        "status": "EXPLORATORY_PAIRED_OUTCOMES_NOT_FITTED_EFFECTS",
        "seed": summary["seed"],
        "evaluation_horizon_s": summary["evaluation_horizon_s"],
        "pairs": pairs,
    }


def timing_results(summary: dict[str, Any]) -> dict[str, Any]:
    """Report observed timing and descriptive deltas, without fitting policy."""
    cases = {}
    for name, arms in summary["cases"].items():
        stay_rate = arms.get("stay", {}).get(
            "fixed_horizon_goodoutput_tokens_s")
        cases[name] = {}
        for action in TIMING_ACTIONS:
            arm = arms.get(action, {})
            observed = arm.get("observed_action") or {}
            rate = arm.get("fixed_horizon_goodoutput_tokens_s")
            cases[name][action] = {
                "eligible": arm.get("fixed_horizon_eligible", False),
                "context_censored": arm.get("context_censored", False),
                "slo_reference_applicability": arm.get(
                    "slo_reference_applicability"),
                "goodoutput_tokens_s": rate,
                "descriptive_delta_vs_stay_tokens_s": (
                    rate - stay_rate if rate is not None
                    and stay_rate is not None else None),
                "actual_start_output_tokens": observed.get(
                    "actual_m1_start_output_tokens", []),
                "first_natural_candidate_output_tokens": observed.get(
                    "timing_first_natural_start_output_tokens"),
                "timing_gate_reasons": observed.get("timing_gate_reasons", []),
                "handoff_stall_ms": observed.get("handoff_stall_ms"),
                "slo_attainment": (arm.get("slo_metrics") or {}).get(
                    "slo_attainment"),
                "exclusions": arm.get("fixed_horizon_exclusions"),
            }
    return {
        "status": "ENGINEERING_TIMING_PILOT_NOT_FITTED_BENEFIT",
        "seed": summary["seed"],
        "evaluation_horizon_s": summary["evaluation_horizon_s"],
        "cases": cases,
    }


def execute_pilot(args: argparse.Namespace) -> None:
    if not 128 < args.max_model_len <= 32768:
        raise ValueError("source max model length must be in (128, 32768]")
    if (args.tp4_max_model_len is not None
            and not args.max_model_len <= args.tp4_max_model_len <= 32768):
        raise ValueError("TP4 max model length must cover TP1 and be <= 32768")
    if not math.isfinite(args.evaluation_horizon_s) or args.evaluation_horizon_s <= 0:
        raise ValueError("evaluation horizon must be positive and finite")
    if args.paired_only and args.timing_pilot:
        raise ValueError("choose paired-only or timing-pilot, not both")
    if args.actions and not args.timing_pilot:
        raise ValueError("--actions requires --timing-pilot")
    if args.actions and len(args.actions) != len(set(args.actions)):
        raise ValueError("timing actions must be unique")
    if (not args.background_context_limit
            and not 1 <= args.background_max_tokens <= 4096):
        raise ValueError("background max tokens must be in [1, 4096]")
    if args.anchor_total_max_tokens is not None and args.anchor_total_max_tokens <= 0:
        raise ValueError("anchor total max tokens must be positive")
    selected_cases = list(args.cases or (row[0] for row in CASES))
    if len(selected_cases) != len(set(selected_cases)):
        raise ValueError("case names must be unique")
    actions = (tuple(args.actions) if args.actions else
               TIMING_ACTIONS if args.timing_pilot else
               ("stay", "early128") if args.paired_only else ACTIONS)
    preflight = verify(args)
    root = args.out_dir.resolve()
    setup_path = root / "pilot_setup.json"
    if args.resume:
        if not setup_path.is_file():
            raise ValueError("resume requires an existing pilot_setup.json")
        saved = json.loads((root / "preflight.json").read_text(
            encoding="utf-8"))
        if saved != preflight:
            raise ValueError("resume preflight differs from original run")
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
    else:
        if root.exists():
            raise ValueError(f"pilot output directory exists: {root}")
        root.mkdir(parents=True)
        write_json(root / "preflight.json", preflight)
        setup = build_setup(args, root)
        write_json(setup_path, setup)
    summary: dict[str, Any] = {
        "format_version": 1, "status": "PILOT_IN_PROGRESS",
        "seed": args.seed, "cases": {},
        "actions": actions,
        "evaluation_horizon_s": args.evaluation_horizon_s,
        "background_max_tokens": (None if args.background_context_limit
                                  else args.background_max_tokens),
        "background_context_limit": args.background_context_limit,
        "selected_cases": selected_cases,
        "anchor_context_limit": args.anchor_context_limit,
        "anchor_total_max_tokens": args.anchor_total_max_tokens,
        "p03_anchor_id": args.p03_anchor_id,
        "source_max_model_len": args.max_model_len,
        "tp4_max_model_len": args.tp4_max_model_len or args.max_model_len,
    }
    summary_path = root / "pilot_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("seed") != args.seed
                or tuple(summary.get("actions", ())) != tuple(actions)
                or summary.get("evaluation_horizon_s")
                != args.evaluation_horizon_s
                or summary.get("background_max_tokens")
                != (None if args.background_context_limit
                    else args.background_max_tokens)
                or summary.get("background_context_limit")
                != args.background_context_limit
                or summary.get("selected_cases") != selected_cases
                or summary.get("anchor_context_limit")
                != args.anchor_context_limit
                or summary.get("anchor_total_max_tokens")
                != args.anchor_total_max_tokens
                or summary.get("p03_anchor_id") != args.p03_anchor_id
                or summary.get("source_max_model_len") != args.max_model_len
                or summary.get("tp4_max_model_len")
                != (args.tp4_max_model_len or args.max_model_len)):
            raise ValueError("resume options differ from original pilot")
    failed = False
    try:
        for name in selected_cases:
            manifest = setup["manifests"][name]
            if sha256(Path(manifest["path"])) != manifest["sha256"]:
                raise ValueError(f"manifest changed: {name}")
            anchor = setup["anchors"][name]
            if sha256(Path(anchor["path"])) != anchor["sha256"]:
                raise ValueError(f"anchor request changed: {name}")
            validation = [
                sys.executable,
                "tools/bridge_tp/run_phase9_capacity_background.py",
                "--manifest", manifest["path"], "--out-dir",
                str(root / name / "manifest-validation"), "--validate-only",
            ]
            if execute(validation, root / name / "manifest-validation.log"):
                failed = True
                break
            outcomes = summary["cases"].setdefault(name, {})
            for action in action_order(name, args.seed, actions):
                if action in outcomes:
                    prior = outcomes[action]
                    if prior.get("fatal_error", True):
                        raise ValueError(
                            f"previous failed arm must be diagnosed: {name}/{action}")
                    continue
                if (root / name / action).exists():
                    raise ValueError(
                        f"unrecorded arm directory must be inspected: "
                        f"{name}/{action}")
                print(f"=== {name} {action} ===", flush=True)
                result = collect_arm(args, root, setup, name, action)
                outcomes[action] = result
                fixed_horizon_result(result, args.evaluation_horizon_s)
                write_json(summary_path, summary)
                if result["fatal_error"]:
                    failed = True
                    break
            if failed:
                break
    finally:
        summary["status"] = (
            "INCOMPLETE_DIAGNOSTIC" if failed
            or any(len(summary["cases"].get(name, {})) < len(actions)
                   for name in selected_cases)
            else "PILOT_COLLECTION_COMPLETE")
        write_json(summary_path, summary)
        if args.paired_only:
            write_json(root / "paired_outcomes.json", pair_results(summary))
        if args.timing_pilot:
            write_json(root / "timing_outcomes.json", timing_results(summary))
        archive = root.with_suffix(".tar.gz")
        print(f"packing={root}", flush=True)
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(root, arcname=root.name)
        print(f"archive_to_retrieve={archive}", flush=True)
        print(f"pilot_status={summary['status']}", flush=True)
    if summary["status"] != "PILOT_COLLECTION_COMPLETE":
        raise RuntimeError("pilot incomplete; retrieve the diagnostic archive")


def main() -> None:
    execute_pilot(parse_args())


if __name__ == "__main__":
    main()
