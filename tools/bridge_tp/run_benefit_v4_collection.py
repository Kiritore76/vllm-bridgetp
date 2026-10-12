#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-register and execute six target-phase/source-urgency paired strata."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp.build_constructed_probability_workload import (  # noqa: E402
    make_rows,
)
from tools.bridge_tp.result_archive import pack  # noqa: E402

PROTOCOL = "RISK76_MLP4_TARGET_PHASE_V4"


def design():
    """Return an immutable 24/6/6 split; pilot is training replicate zero."""
    rows = []
    for role, repeats, base in (
        ("train", 4, 20261200),
        ("validation", 1, 20261300),
        ("test", 1, 20261400),
    ):
        for replicate in range(repeats):
            for source_index, source in enumerate(("slack", "urgent")):
                for target_index, target in enumerate(("idle", "decode", "prefill")):
                    seed = base + replicate * 6 + source_index * 3 + target_index
                    rows.append(
                        {
                            "pair_id": f"{role}-{source}-{target}-{seed}",
                            "role": role,
                            "seed": seed,
                            "replicate": replicate,
                            "source_state": source,
                            "target_state": target,
                            "source_count": 3 if source == "slack" else 5,
                            "source_prompt_tokens": 4096 if source == "slack" else 4992,
                            "target_count": {"idle": 0, "decode": 4, "prefill": 8}[
                                target
                            ],
                            "target_prompt_tokens": 6144
                            if target == "prefill"
                            else 2048,
                            "target_period_s": 10 if target == "prefill" else 40,
                            "target_max_tokens": 1024 if target == "prefill" else 4096,
                            "minimum_u": 0 if source == "slack" else 0.8,
                            "maximum_u": 0.5 if source == "slack" else None,
                            "threshold": (0.0, 0.0, 0.1, 0.3)[replicate]
                            if role == "train"
                            else 0.1,
                            "decode_pressure_min": 0.25,
                            "prefill_min_tokens": 1024,
                            "source_family_group": f"constructed-record-set-{seed}",
                        }
                    )
    return rows


def repair_design():
    """Two diagnostic pairs; keep later selection and test sets untouched."""
    rows = []
    for target, seed in (("decode", 20261251), ("prefill", 20261252)):
        base = next(row for row in design() if row["role"] == "train"
                    and row["replicate"] == 0 and row["source_state"] == "slack"
                    and row["target_state"] == target)
        row = {
            **base,
            "pair_id": f"repair-slack-{target}-{seed}",
            "role": "validation",
            "seed": seed,
            "source_family_group": f"constructed-repair-record-set-{seed}",
            "diagnostic_use_only": True,
            "target_count": 12 if target == "decode" else 4,
            "target_period_s": 40 if target == "decode" else 30,
            "target_max_tokens": 4096 if target == "decode" else 256,
            "target_burst_arrivals": target == "prefill",
        }
        rows.append(row)
    return rows


def dispatch_design():
    """Repeat one decode pair/seed to isolate collector changes."""
    return [
        {**row, "pair_id": f"dispatch-slack-decode-{row['seed']}"}
        for row in repair_design() if row["target_state"] == "decode"
    ]


def prefill_repeat_design():
    """New request records for diagnosing pre-action prefix divergence."""
    seed = 20261253
    return [{
        **repair_design()[1], "seed": seed,
        "pair_id": f"prefill-repeat-slack-{seed}",
        "source_family_group": f"constructed-prefill-repeat-record-set-{seed}",
    }]


def urgent_phase_design():
    """Reach high U with previously tested target-phase recipes, not high p gates."""
    rows = []
    for index, target in enumerate(("idle", "decode", "prefill")):
        seed = 20261261 + index
        base = next(row for row in design() if row["role"] == "train"
                    and row["replicate"] == 0 and row["source_state"] == "urgent"
                    and row["target_state"] == target)
        rows.append({
            **base, "role": "validation", "seed": seed,
            "pair_id": f"urgent-phase-{target}-{seed}",
            "source_family_group": f"constructed-urgent-phase-record-set-{seed}",
            "diagnostic_use_only": True,
            "target_count": {"idle": 0, "decode": 12, "prefill": 4}[target],
            "target_period_s": 30 if target == "prefill" else 40,
            "target_max_tokens": 256 if target == "prefill" else 4096,
            "target_burst_arrivals": target == "prefill",
        })
    return rows


def urgent_followup_design():
    """Repeat decode; align prefill to the source event using the same inputs."""
    rows = []
    for row in urgent_phase_design():
        if row["target_state"] == "idle":
            continue
        rows.append({
            **row, "pair_id": f"urgent-followup-{row['target_state']}-{row['seed']}",
            "target_start_after_anchor": row["target_state"] == "prefill",
            "target_burst_offset_s": 1.0 if row["target_state"] == "prefill" else 0.5,
            "purpose": ("repeat dispatch eligibility" if row["target_state"] == "decode"
                        else "earlier source-relative prefill phase"),
        })
    return rows


def urgent_prefill_recovery_design():
    """Only the unstarted prefill pair, with identical inputs and timing."""
    return [row for row in urgent_followup_design()
            if row["target_state"] == "prefill"]


def formal_expansion_design():
    """Fresh 12 train / 6 validation / 6 test pairs using verified recipes."""
    rows = []
    for role, repeats, seed_base in (
        ("train", 2, 20261500), ("validation", 1, 20261600),
        ("test", 1, 20261700),
    ):
        for replicate in range(repeats):
            for index, base in enumerate(design()[:6]):
                seed = seed_base + replicate * 6 + index
                target = base["target_state"]
                rows.append({
                    **base, "role": role, "replicate": replicate, "seed": seed,
                    "pair_id": f"formal-{role}-{base['source_state']}-{target}-{seed}",
                    "source_family_group": f"constructed-formal-record-set-{seed}",
                    "target_count": {"idle": 0, "decode": 12, "prefill": 4}[target],
                    "target_period_s": 30 if target == "prefill" else 40,
                    "target_max_tokens": 256 if target == "prefill" else 4096,
                    "target_burst_arrivals": target == "prefill",
                    "target_start_after_anchor": target == "prefill",
                    "target_burst_offset_s": 1.0,
                    "threshold": 0.0,
                })
    return rows


def recovery_validation_design():
    """Repeat only two arrival-ineligible training pairs, then blind validation."""
    rows = formal_expansion_design()
    return ([r for r in rows if r["seed"] in {20261504, 20261508}]
            + [r for r in rows if r["role"] == "validation"])


def prefix_diagnostic(pair_root):
    """Describe pre-action divergence without selecting on benefit outcomes."""
    summary = json.loads((pair_root / "pilot_summary.json").read_text())
    rows = []
    for case, arms in summary["cases"].items():
        for action, result in arms.items():
            candidate = (result.get("observed_action") or {}).get(
                "probability_candidate"
            ) or {}
            snapshot = candidate.get("snapshot") or {}
            run = pair_root / case / action / "r01_shadow_only/controller"
            source = json.loads((run / "source_response.json").read_text())
            rows.append({
                "action": action, "snapshot": snapshot,
                "token_ids": source.get("token_ids") or [],
            })
    if len(rows) != 2:
        raise ValueError("prefix diagnosis requires two completed arms")
    if any(row["snapshot"].get("generated_tokens") is None for row in rows):
        report = {
            "diagnostic_only": True, "status": "CANDIDATE_NOT_OBSERVED",
            "identical_pre_action_prefix": None, "selection_on_goodoutput": False,
            "candidate_snapshots": [
                {key: value for key, value in row.items() if key != "token_ids"}
                for row in rows
            ],
        }
        (pair_root / "pre_action_prefix_diagnostic.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print("[诊断] 某臂未观察到合格候选；保留无START结果，不伪造前缀匹配。",
              flush=True)
        return
    boundary = min(int(row["snapshot"]["generated_tokens"]) for row in rows)
    left, right = [row.pop("token_ids")[:boundary] for row in rows]
    available = min(len(left), len(right))
    first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
    report = {
        "diagnostic_only": True, "comparison_tokens": available,
        "common_pre_action_boundary": boundary,
        "complete_pre_action_prefix": available == boundary,
        "first_divergence_zero_based": first,
        "aligned_mismatch_tokens": sum(a != b for a, b in zip(left, right)),
        "identical_pre_action_prefix": available == boundary and first is None,
        "selection_on_goodoutput": False, "candidate_snapshots": rows,
    }
    (pair_root / "pre_action_prefix_diagnostic.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[诊断] 动作前共同前缀{available} token；首次分歧索引={first}；"
          f"一致={report['identical_pre_action_prefix']}", flush=True)


def command(args, row, input_path, out):
    """Build probability interventions; keep migration/SLO contracts frozen."""
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    cmd = [
        sys.executable,
        str(ROOT / "tools/bridge_tp/run_randomized_goodoutput_pilot_a100.py"),
    ]
    for key in ("model", "base", "survival", "guard", "checkpoint", "reference"):
        cmd += [f"--{key}", str(getattr(args, key))]
    cmd += [
        "--input",
        str(input_path),
        "--out-dir",
        str(out),
        "--expected-revision",
        args.expected_revision,
        "--portable-hardware",
        "--seed",
        str(row["seed"]),
        "--probability-pilot",
        "--no-archive",
        "--constructed-workload",
        "--expected-input-sha256",
        digest,
        "--predictor-profile",
        "risk76_mlp4",
        "--guard-profile",
        "reduced2000",
        "--slo-profile",
        "slow1pct_soft",
        "--window-token-goodoutput",
        "--evaluation-horizon-s",
        "300",
        "--arrival-window-s",
        "300",
        "--arrival-wave-period-s",
        "40",
        "--target-arrival-wave-period-s",
        str(row["target_period_s"]),
        "--constructed-source-count",
        str(row["source_count"]),
        "--constructed-target-count",
        str(row["target_count"]),
        "--probability-thresholds",
        str(row["threshold"]),
        "--probability-min-urgency",
        str(row["minimum_u"]),
        "--benefit-target-state",
        row["target_state"],
        "--target-decode-pressure-min",
        str(row["decode_pressure_min"]),
        "--target-prefill-min-tokens",
        str(row["prefill_min_tokens"]),
        "--save-predictor-features",
        "--risk-observation-shadow",
        "--rolling-cutover",
        "--rolling-reserve-tokens",
        "512",
        "--max-model-len",
        "8192",
        "--tp4-max-model-len",
        "16384",
        "--anchor-context-limit",
        "--pre-episode-warmup",
        "--minimum-initial-source-headroom-tokens",
        "1536",
        "--background-max-tokens",
        str(row["target_max_tokens"]),
        "--source-background-max-tokens",
        "2048",
    ]
    if row["maximum_u"] is not None:
        cmd += ["--probability-max-urgency", str(row["maximum_u"])]
    if row.get("target_burst_arrivals"):
        cmd += ["--target-burst-arrivals"]
    if row.get("target_start_after_anchor"):
        cmd += ["--target-start-after-anchor", "--target-burst-offset-s",
                str(row["target_burst_offset_s"])]
    return cmd


def verify_completed_pair(args, pair_root, inspect_prefix):
    import torch

    from tools.bridge_tp.verify_predictor_feature_journal import verify

    torch.set_num_threads(1)
    journals = list(
        (pair_root).glob(
            "*/*/r01_shadow_only/controller/predictor_features"
        )
    )
    if len(journals) != 2:
        raise RuntimeError("paired source feature journals missing")
    for directory in journals:
        replay = verify(
            directory,
            args.checkpoint,
            "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e",
            directory.parent / "phase9_audit.jsonl",
        )
        (directory / "replay_verification.json").write_text(
            json.dumps(replay, indent=2) + "\n", encoding="utf-8"
        )
    if inspect_prefix:
        prefix_diagnostic(pair_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("pilot", "repair-pilot", "dispatch-pilot", "prefill-repeat",
                 "urgent-phase",
                 "urgent-followup",
                 "urgent-prefill-recovery",
                 "formal-train1", "formal-train2", "formal-train-all",
                 "formal-validation", "formal-test",
                 "formal-recovery-validation",
                 "train-rest",
                 "validation", "test"),
        default="pilot",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-failed-pairs", type=int, default=1,
                        help="stop at this cumulative number of failed pairs")
    parser.add_argument("--no-archive", action="store_true",
                        help="outer launcher packs the entire batch once")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    for key in ("model", "base", "survival", "guard", "checkpoint", "reference"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    if args.max_failed_pairs < 1:
        parser.error("--max-failed-pairs must be positive")
    diagnostic = args.stage in {
        "repair-pilot", "dispatch-pilot", "prefill-repeat", "urgent-phase",
        "urgent-followup",
        "urgent-prefill-recovery",
    }
    all_rows = repair_design() if diagnostic else design()
    if args.stage == "dispatch-pilot":
        all_rows = dispatch_design()
    elif args.stage == "prefill-repeat":
        all_rows = prefill_repeat_design()
    elif args.stage == "urgent-phase":
        all_rows = urgent_phase_design()
    elif args.stage == "urgent-followup":
        all_rows = urgent_followup_design()
    elif args.stage == "urgent-prefill-recovery":
        all_rows = urgent_prefill_recovery_design()
    elif args.stage == "formal-recovery-validation":
        all_rows = recovery_validation_design()
    elif args.stage.startswith("formal-"):
        all_rows = formal_expansion_design()
    selected = [
        row
        for row in all_rows
        if (
            diagnostic
            or args.stage == "formal-recovery-validation"
            or args.stage == "formal-train-all" and row["role"] == "train"
            or args.stage in {"formal-train1", "formal-train2"}
            and row["role"] == "train"
            and row["replicate"] == int(args.stage[-1]) - 1
            or args.stage == f"formal-{row['role']}"
            or args.stage == "pilot"
            and row["role"] == "train"
            and row["replicate"] == 0
            or args.stage == "train-rest"
            and row["role"] == "train"
            and row["replicate"] > 0
            or args.stage == row["role"]
        )
    ]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frozen = {
        "protocol": PROTOCOL,
        "stage": args.stage,
        "diagnostic_only": diagnostic,
        "max_failed_pairs": args.max_failed_pairs,
        "failure_count_basis": "CUMULATIVE_FAILED_PAIRS_NOT_ARMS",
        "replacement_source_batch": (
            "benefit-v4-mlp4-formal-train-all-24arm-v1-20261012T144446Z-1232"
            if args.stage == "formal-recovery-validation" else None
        ),
        "replacement_pair_ids": (
            [r["pair_id"] for r in selected if r["role"] == "train"]
            if args.stage == "formal-recovery-validation" else []
        ),
        "collection_revision": args.expected_revision,
        "arrival_schedule_version": "FIXED_EVENT_EPOCH_DUE_TIME_DISPATCH_V2",
        "all_pairs": all_rows,
        "predictor_sha256": (
            "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7"
        ),
        "selected_pairs": [row["pair_id"] for row in selected],
        "generalization_scope": "new record roots, same task templates",
        "threshold_status": "pilot zero; review later thresholds before outcomes",
        "target_ranges_status": "pilot engineering assumptions, not calibrated",
    }
    (args.out_dir / "preregistration.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
    )
    inputs = args.out_dir / "inputs"
    inputs.mkdir()
    commands = []
    for row in selected:
        input_path = inputs / f"{row['pair_id']}.jsonl"
        recipes = make_rows(
            seed=row["seed"],
            split=f"engineering_{row['role']}",
            anchor_prompt_tokens=2048,
            background_prompt_tokens=row["source_prompt_tokens"],
            anchor_sections=24,
            background_sections=16,
            backgrounds=640,
            target_prompt_tokens=row["target_prompt_tokens"],
        )
        input_path.write_text(
            "".join(json.dumps(recipe) + "\n" for recipe in recipes), encoding="utf-8"
        )
        cmd = command(args, row, input_path, args.out_dir / row["pair_id"])
        commands.append({"pair": row, "argv": cmd})
    (args.out_dir / "commands.json").write_text(
        json.dumps(commands, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[进度] {args.stage}: {len(commands)}对/{2 * len(commands)} arm", flush=True)
    if args.plan_only:
        return
    results = []
    try:
        for index, item in enumerate(commands, 1):
            print(
                f"[进度] 配对 {index}/{len(commands)}: {item['pair']['pair_id']}",
                flush=True,
            )
            error = None
            try:
                rc = subprocess.call(item["argv"], cwd=ROOT)
            except OSError as failure:
                rc = 1
                error = f"{type(failure).__name__}: {failure}"
            if rc == 0:
                try:
                    verify_completed_pair(
                        args, args.out_dir / item["pair"]["pair_id"],
                        diagnostic or args.stage.startswith("formal-"),
                    )
                except Exception as failure:
                    rc = 1
                    error = f"{type(failure).__name__}: {failure}"
            results.append({"pair_id": item["pair"]["pair_id"], "returncode": rc,
                            "verification_error": error})
            (args.out_dir / "stage_results.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
            failed = sum(row["returncode"] != 0 for row in results)
            stopped = failed >= args.max_failed_pairs
            completed = index == len(commands)
            status = ("STOPPED_AT_FAILURE_LIMIT" if stopped else
                      "COMPLETE_WITH_FAILURES" if completed and failed else
                      "COLLECTION_COMPLETE" if completed else "RUNNING")
            (args.out_dir / "stage_status.json").write_text(json.dumps({
                "status": status, "planned_pairs": len(commands),
                "attempted_pairs": len(results), "failed_pairs": failed,
                "successful_pairs": len(results) - failed,
                "unattempted_pairs": len(commands) - len(results),
                "max_failed_pairs": args.max_failed_pairs,
            }, indent=2) + "\n", encoding="utf-8")
            if rc:
                next_step = ("达到上限，停止后续配对" if stopped
                             else "保留证据，继续下一对")
                print(f"[进度] 配对失败；累计{failed}/{args.max_failed_pairs}；"
                      + next_step,
                      flush=True)
            if stopped:
                raise RuntimeError("cumulative failed-pair limit reached")
        if any(row["returncode"] for row in results):
            raise RuntimeError(
                "collection attempted all pairs with failures; inspect archive"
            )
    finally:
        if not args.no_archive:
            archive = args.out_dir.with_suffix(".tar.gz")
            print(json.dumps(pack(args.out_dir, archive)), flush=True)
            print(f"请取回完整目录压缩包：{archive}", flush=True)


if __name__ == "__main__":
    main()
