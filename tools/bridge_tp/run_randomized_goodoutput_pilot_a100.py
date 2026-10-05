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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "input", "base", "survival", "guard",
                 "checkpoint", "reference", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--paired-only", action="store_true",
                        help="Collect STAY and early128 without the late EOS arm")
    parser.add_argument("--random-arrivals", action="store_true",
                        help="Use seeded exponential interarrival gaps")
    parser.add_argument("--evaluation-horizon-s", type=float, default=180.0)
    return parser.parse_args()


def select_inputs(path: Path, seed: int = SEED) -> list[dict[str, Any]]:
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
            and row.get("workload_group") in {"natural", "long_form"}]
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
    rows = select_inputs(args.input, seed)
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
        if len(anchor_tokens) + 4096 > 8192:
            raise ValueError(f"anchor exceeds context limit: {anchor['id']}")
        anchor_request = {
            "model": "bridgetp-model", "prompt": anchor_tokens,
            "max_tokens": 4096, "ignore_eos": False,
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
            "workload_group": anchor["workload_group"],
        }
        jobs = []
        chosen = rows[cursor:cursor + source_count + target_count]
        cursor += len(chosen)
        for pool, subset in (("target", chosen[:target_count]),
                             ("source", chosen[target_count:])):
            for job_index, row in enumerate(subset):
                prompt_tokens = tokens(row)
                if len(prompt_tokens) + 2048 > 8192:
                    raise ValueError(f"background exceeds context: {row['id']}")
                job = {
                    "job_id": f"{pool}_{job_index:03d}", "pool": pool,
                    "start_after_s": offsets[pool][job_index],
                    "request": {
                        "model": "bridgetp-model", "prompt": prompt_tokens,
                        "max_tokens": 2048, "ignore_eos": False,
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
    if action not in ACTIONS:
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
        "first_candidate_at_or_after_128": first_candidate,
        "last_m1_decision": decisions[-1].get("decision")
        if decisions else None,
        "actual_cutover_output_tokens": accepted.get(
            "earliest_ready_cutover_output_tokens"),
        "handoff_stall_ms": accepted.get("handoff_stall_ms"),
        "acceptance_status": accepted.get("status"),
        "acceptance_errors": accepted.get("errors"),
    }


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
    audit_rc = execute(audit_command,
                       case_root / f"{action}.slo_v6.console.log")
    slo = (json.loads(audit_path.read_text(encoding="utf-8"))
           if audit_path.is_file() else {})
    background_path = run / "background" / "background_summary.json"
    background = (json.loads(background_path.read_text(encoding="utf-8"))
                  if background_path.is_file() else {})
    source_path = run / "controller" / "source_response.json"
    source = (json.loads(source_path.read_text(encoding="utf-8"))
              if source_path.is_file() else {})
    target_path = run / "controller" / "target_response.json"
    target = (json.loads(target_path.read_text(encoding="utf-8"))
              if target_path.is_file() else {})
    finish_reasons = {
        reason: sum(row.get("finish_reason") == reason
                    for row in background.get("results", []))
        for reason in {row.get("finish_reason")
                       for row in background.get("results", [])}
    }
    observed = observed_action(run, paired_stay=action == "stay")
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
        "fatal_error": bool(audit_rc or (runner_rc and not natural_noop)),
        "slo_computable": slo.get("computable"),
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


def execute_pilot(args: argparse.Namespace) -> None:
    if not math.isfinite(args.evaluation_horizon_s) or args.evaluation_horizon_s <= 0:
        raise ValueError("evaluation horizon must be positive and finite")
    actions = ("stay", "early128") if args.paired_only else ACTIONS
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
    }
    summary_path = root / "pilot_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("seed") != args.seed
                or tuple(summary.get("actions", ())) != tuple(actions)
                or summary.get("evaluation_horizon_s")
                != args.evaluation_horizon_s):
            raise ValueError("resume options differ from original pilot")
    failed = False
    try:
        for name in setup["cases"]:
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
                   for name in setup["cases"])
            else "PILOT_COLLECTION_COMPLETE")
        write_json(summary_path, summary)
        if args.paired_only:
            write_json(root / "paired_outcomes.json", pair_results(summary))
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
