"""Reuse the audited decoder31 capture for a 3-width by 2-bin ablation.

Select width on 32-token bins, then compare bin widths at the selected width.
All six fits are retained to reveal interactions. Selection uses uncalibrated
validation Prefill Brier at common thresholds, never cross-bin categorical NLL
or test metrics. Temperature fitting retains its separate validation trees.
"""

import argparse
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
from prepare_oasst1_predictor_inputs import SOURCE_SHA256
from run_predictor_large_a100 import (
    INPUT_SHAS,
    MODEL_SHA,
    REPO,
    validate_shard,
)
from train_length_predictor import load_examples, sha256_file
from train_predictor_distribution import fit_distribution, validation_groups

CAPTURE_REVISION = "db2626d5b7af8c75fa8f3fe6a0424490b1d621bf"
BASELINE_REPORT_SHA = "a759790101695639160b17ef8e32570c908e71c69deab6916419607658c51163"
WIDTHS = (256, 128, 64)
BIN_STEPS = (32, 64)
HORIZONS = (128, 256, 512)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def summarize_fit(report: dict) -> dict:
    """Extract comparable risk metrics; keep categorical NLL diagnostic only."""
    summary = {
        "hidden_width": report["training_parameters"]["hidden_width"],
        "bin_step": report["training_parameters"]["bin_step"],
        "parameter_count": report["training_parameters"]["parameter_count"],
        "classes": len(report["category_upper_edges"]) + 1,
        "best_epoch": report["best_epoch"],
        "best_epoch_losses": report["history"][report["best_epoch"] - 1],
        "last_epoch_losses": report["history"][-1],
        "temperature": report["temperature"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "selection_request_ids": report["selection_validation_request_ids"],
        "calibration_request_ids": report["calibration_validation_request_ids"],
        "risk": {},
        "categorical_nll_not_comparable_across_bin_steps": {
            name: group["calibrated_nll"] for name, group in report["results"].items()
        },
    }
    groups = dict(report["validation_stratified"])
    groups["test"] = {
        phase: report["test_stratified"].get("phases/" + phase, {})
        for phase in ("PREFILL_COMPLETE", "DECODE")
    }
    for name, phases in groups.items():
        summary["risk"][name] = {}
        for phase, metrics in phases.items():
            if not metrics:
                continue
            summary["risk"][name][phase] = {}
            for model in ("raw_model", "calibrated_model"):
                rows = {str(h): metrics[str(h)][model] for h in HORIZONS}
                summary["risk"][name][phase][model] = {
                    "mean_brier": float(np.mean([r["brier"] for r in rows.values()])),
                    "mean_binary_log_loss": float(
                        np.mean([r["binary_log_loss"] for r in rows.values()])
                    ),
                    "risk_by_horizon_tokens": rows,
                }
    return summary


def selection_key(row: dict) -> tuple[float, float, int, int]:
    metrics = row["risk"]["selection_validation"]["PREFILL_COMPLETE"]["raw_model"]
    return (
        metrics["mean_brier"],
        metrics["mean_binary_log_loss"],
        row["hidden_width"],
        row["bin_step"],
    )


def phase_score(row: dict, group: str, phase: str, model: str) -> str:
    metrics = row["risk"][group].get(phase, {}).get(model)
    return f"{metrics['mean_brier']:.6f}" if metrics else "N/A"


def choose_stages(rows: list[dict]) -> dict:
    """Use identical selection trees and common thresholds for both stages."""
    expected = {(w, b) for w in WIDTHS for b in BIN_STEPS}
    if (
        len(rows) != len(expected)
        or {(r["hidden_width"], r["bin_step"]) for r in rows} != expected
    ):
        raise ValueError("six unique width/bin configurations are required")
    selected = rows[0]["selection_request_ids"]
    calibrated = rows[0]["calibration_request_ids"]
    if not selected or not calibrated or set(selected) & set(calibrated):
        raise ValueError("selection/calibration trees must be nonempty and disjoint")
    if any(
        r["selection_request_ids"] != selected
        or r["calibration_request_ids"] != calibrated
        for r in rows
    ):
        raise ValueError("validation partitions differ between configurations")
    width_ranking = sorted((r for r in rows if r["bin_step"] == 32), key=selection_key)
    width = width_ranking[0]["hidden_width"]
    bin_ranking = sorted(
        (r for r in rows if r["hidden_width"] == width), key=selection_key
    )
    return {
        "selection_rule": "raw selection Prefill mean Brier at 128/256/512; "
        "binary log loss, smaller width/bin break exact ties",
        "stage1_width_ranking_at_bin32": width_ranking,
        "stage2_bin_ranking_at_selected_width": bin_ranking,
        "selected_configuration": {
            "hidden_width": width,
            "bin_step": bin_ranking[0]["bin_step"],
        },
        "all_six_selection_rankings": sorted(rows, key=selection_key),
        "note": "one seed exploratory ablation; test is diagnostic only; "
        "categorical NLL cannot rank different bin widths; "
        "P(remaining>capacity) is not physical CUDA OOM probability",
    }


def validate_capture(args: argparse.Namespace) -> tuple[dict, dict]:
    """Pin the existing data rather than capturing new responses."""
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("raw OASST1 SHA256 differs")
    if sha256_file(args.model / "config.json") != MODEL_SHA:
        raise ValueError("model config SHA256 differs")
    meta = validate_shard(
        args.run_dir,
        args.run_dir / "input_requests.jsonl",
        CAPTURE_REVISION,
        "decoder:31",
    )
    if (
        meta["input_sha256"] != INPUT_SHAS[2000]
        or meta["model_path"] != str(args.model.resolve())
        or meta["requests"] != 2000
        or not meta.get("source_shards")
    ):
        raise ValueError("expected the merged decoder31 capture of 2000 requests")
    if sha256_file(args.baseline_report) != BASELINE_REPORT_SHA:
        raise ValueError("baseline report SHA256 differs from returned decoder31 run")
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    if (
        baseline["capture_preflight"] != meta
        or baseline["training_revision"] != CAPTURE_REVISION
    ):
        raise ValueError("baseline report belongs to another capture")
    data = load_examples(args.run_dir, include_censored=True)
    if data["audit"] != baseline["capture_audit"]:
        raise ValueError("capture audit differs from the existing baseline")
    if data["hidden"].shape[1] != 5120:
        raise ValueError("expected Qwen14B hidden width 5120")
    counts = {
        s: len(set(data["requests"][data["splits"] == s]))
        for s in ("train", "validation", "test")
    }
    if counts != {"train": 1600, "validation": 200, "test": 200}:
        raise ValueError(f"request split counts differ: {counts}")
    selection, calibration = validation_groups(data)
    if (
        sorted(set(data["requests"][selection]))
        != baseline["selection_validation_request_ids"]
        or sorted(set(data["requests"][calibration]))
        != baseline["calibration_validation_request_ids"]
    ):
        raise ValueError("baseline validation partitions changed")
    return meta, data


def run_ablation(args: argparse.Namespace, revision: str, names: list[str]) -> None:
    meta, data = validate_capture(args)
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    records = []
    capture_files = (
        "preflight.json",
        "summary.json",
        "input_requests.jsonl",
        "labels.jsonl",
        "sample_index.jsonl",
        "features/features.sqlite3",
    )
    hashes = {name: sha256_file(args.run_dir / name) for name in capture_files}
    preflight = {
        "hostname": socket.gethostname(),
        "training_revision": revision,
        "capture_dir": str(args.run_dir.resolve()),
        "capture_preflight": meta,
        "capture_audit": data["audit"],
        "capture_file_sha256": hashes,
        "baseline_report": str(args.baseline_report.resolve()),
        "baseline_report_sha256": sha256_file(args.baseline_report),
        "raw_source": str(args.source.resolve()),
        "raw_source_sha256": SOURCE_SHA256,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": MODEL_SHA,
        "gpu_names": names,
        "torch_version": str(torch.__version__),
        "seed": 42,
        "hidden_widths": WIDTHS,
        "bin_steps": BIN_STEPS,
        "epochs": 30,
        "patience": 6,
        "batch_size": 256,
        "learning_rate": 0.0003,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "capture_interval": 20,
    }
    write_json(args.out_dir / "ablation_preflight.json", preflight)
    for name in capture_files[:-1]:
        shutil.copyfile(args.run_dir / name, args.out_dir / Path(name).name)
    shutil.copyfile(
        args.baseline_report, args.out_dir / "previous_baseline_report.json"
    )
    completed = False
    try:
        for bin_step in BIN_STEPS:
            for width in WIDTHS:
                name = f"width{width}_bin{bin_step}"
                print(f"fit={len(records) + 1}/6 configuration={name}", flush=True)
                start = time.monotonic()
                report = fit_distribution(
                    data,
                    meta,
                    args.out_dir / name,
                    revision=revision,
                    epochs=30,
                    batch_size=256,
                    hidden_width=width,
                    learning_rate=0.0003,
                    patience=6,
                    seed=42,
                    bin_step=bin_step,
                )
                row = summarize_fit(report)
                row["elapsed_seconds"] = time.monotonic() - start
                row["output_dir"] = str((args.out_dir / name).resolve())
                records.append(row)
                write_json(args.out_dir / "progress.json", {"completed_fits": records})
                print(
                    json.dumps(
                        {
                            "completed": name,
                            "elapsed_seconds": row["elapsed_seconds"],
                            "selection_key": selection_key(row),
                        }
                    ),
                    flush=True,
                )
        if any(sha256_file(args.run_dir / n) != h for n, h in hashes.items()):
            raise ValueError("capture files changed during training")
        comparison = choose_stages(records)
        write_json(args.out_dir / "comparison.json", comparison)
        lines = [
            "# Decoder31 width/bin ablation",
            "",
            "Selection: raw Prefill Brier at 128/256/512 on selection trees only.",
            "Categorical NLL values from different bin widths are not comparable.",
            "",
            (
                "| Width | Bin | Classes | Parameters | Best epoch | "
                "Selection raw Prefill Brier | Selection raw Decode Brier | "
                "Test calibrated Prefill Brier | Test calibrated Decode Brier |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in records:
            scores = [
                phase_score(row, g, p, m)
                for g, m in (
                    ("selection_validation", "raw_model"),
                    ("test", "calibrated_model"),
                )
                for p in ("PREFILL_COMPLETE", "DECODE")
            ]
            lines.append(
                f"| {row['hidden_width']} | {row['bin_step']} | {row['classes']} | "
                f"{row['parameter_count']} | {row['best_epoch']} | "
                + " | ".join(scores)
                + " |"
            )
        lines += [
            "",
            "Selected exploratory configuration: "
            + json.dumps(comparison["selected_configuration"]),
            "",
            "One seed; test metrics are diagnostic, not a fresh held-out proof.",
        ]
        (args.out_dir / "comparison.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {"selected_configuration": comparison["selected_configuration"]}
            ),
            flush=True,
        )
        completed = True
    finally:
        write_json(
            args.out_dir / "status.json",
            {
                "completed": completed,
                "completed_fits": len(records),
                "required_fits": 6,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-hostname", required=True)
    args = parser.parse_args()
    if socket.gethostname() != args.expected_hostname:
        parser.error("hostname differs from expected")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if revision != args.expected_revision:
        parser.error("HEAD differs from expected revision")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip():
        parser.error("worktree has uncommitted files")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if [n.strip() for n in names] != ["NVIDIA A100-PCIE-40GB"]:
        parser.error(f"GPU inventory differs: {names}")
    if args.out_dir.exists():
        parser.error("out-dir must be new")
    print(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "revision": revision,
                "capture_dir": str(args.run_dir.resolve()),
                "gpu_names": names,
                "input_sha256": INPUT_SHAS[2000],
                "out_dir": str(args.out_dir.resolve()),
            }
        ),
        flush=True,
    )
    run_ablation(args, revision, names)


if __name__ == "__main__":
    main()
