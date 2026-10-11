#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-register and execute six target-phase/source-urgency paired strata."""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp.build_constructed_probability_workload import (  # noqa: E402
    make_rows,
)

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
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("pilot", "train-rest", "validation", "test"),
        default="pilot",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    for key in ("model", "base", "survival", "guard", "checkpoint", "reference"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    all_rows = design()
    selected = [
        row
        for row in all_rows
        if (
            args.stage == "pilot"
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
            rc = subprocess.call(item["argv"], cwd=ROOT)
            if rc == 0:
                import torch

                from tools.bridge_tp.verify_predictor_feature_journal import verify

                torch.set_num_threads(1)
                journals = list(
                    (args.out_dir / item["pair"]["pair_id"]).glob(
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
            results.append({"pair_id": item["pair"]["pair_id"], "returncode": rc})
            (args.out_dir / "stage_results.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
            if rc:
                raise RuntimeError(
                    "pair failed; inspect archived evidence before continuing"
                )
    finally:
        archive = args.out_dir.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(args.out_dir, arcname=args.out_dir.name)
        print(f"请取回完整目录压缩包：{archive}", flush=True)
        print(hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
