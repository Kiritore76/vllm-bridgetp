"""Regularization, learning-rate and seed sweep on the frozen decoder31 capture.

Six seed-42 dropout/weight-decay fits are followed by two slower-learning-rate
fits. The original configuration and one challenger are then repeated at seeds
43 and 44. Only the selection-validation requests rank configurations.
"""

import argparse
import json
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from run_predictor_ablation_a100 import (
    HORIZONS,
    validate_capture,
    write_json,
)
from run_predictor_large_a100 import REPO
from train_length_predictor import sha256_file
from train_predictor_distribution import fit_distribution

DROPOUTS = (0.1, 0.2, 0.3)
WEIGHT_DECAYS = (0.01, 0.05)
INITIAL_LR = 0.0003
SLOW_LR = 0.0001
SEEDS = (42, 43, 44)
BASELINE = (0.1, 0.01, INITIAL_LR)


@dataclass(frozen=True)
class Configuration:
    dropout: float
    weight_decay: float
    learning_rate: float

    @property
    def name(self) -> str:
        return (
            f"drop{round(self.dropout * 100):02d}_"
            f"decay{round(self.weight_decay * 100):02d}_"
            f"lr{round(self.learning_rate * 1_000_000):03d}"
        )

    def as_dict(self) -> dict:
        return {
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate,
        }


def selection_metrics(report: dict) -> dict:
    """Summarize only pre-calibration model-selection requests."""
    metrics = {}
    phases = report["validation_stratified"]["selection_validation"]
    for phase in ("PREFILL_COMPLETE", "DECODE"):
        rows = {str(h): phases[phase][str(h)]["raw_model"] for h in HORIZONS}
        metrics[phase] = {
            "mean_brier": float(np.mean([row["brier"] for row in rows.values()])),
            "mean_binary_log_loss": float(
                np.mean([row["binary_log_loss"] for row in rows.values()])
            ),
            "risk_by_horizon_tokens": {
                h: {key: row[key] for key in ("brier", "binary_log_loss", "ece")}
                for h, row in rows.items()
            },
        }
    return metrics


def ranking_key(row: dict) -> tuple[float, float, float, str]:
    """Read the selection split only; never inspect calibration or test scores."""
    selected = row["selection"]
    return (
        selected["PREFILL_COMPLETE"]["mean_brier"],
        selected["DECODE"]["mean_brier"],
        selected["PREFILL_COMPLETE"]["mean_binary_log_loss"],
        row["configuration_name"],
    )


def choose_challenger(rows: list[dict]) -> dict:
    """Choose a non-baseline seed-42 configuration with validation only."""
    eligible = [
        row
        for row in rows
        if row["seed"] == SEEDS[0]
        and row["configuration"] != Configuration(*BASELINE).as_dict()
    ]
    if not eligible:
        raise ValueError("no non-baseline configuration")
    return min(eligible, key=ranking_key)


def summarize_run(
    report: dict,
    config: Configuration,
    seed: int,
    elapsed_seconds: float,
    out_dir: Path,
) -> dict:
    params = report["training_parameters"]
    expected = {**config.as_dict(), "hidden_width": 256, "bin_step": 32}
    if any(params[key] != value for key, value in expected.items()):
        raise ValueError(
            "training report parameters differ from requested configuration"
        )
    return {
        "configuration_name": config.name,
        "configuration": config.as_dict(),
        "seed": seed,
        "best_epoch": report["best_epoch"],
        "last_epoch": report["history"][-1]["epoch"],
        "best_epoch_train_nll": report["history"][report["best_epoch"] - 1][
            "train_nll"
        ],
        "best_epoch_selection_nll": report["history"][report["best_epoch"] - 1][
            "selection_validation_nll"
        ],
        "last_epoch_train_nll": report["history"][-1]["train_nll"],
        "last_epoch_selection_nll": report["history"][-1]["selection_validation_nll"],
        "temperature": report["temperature"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "selection_request_ids": report["selection_validation_request_ids"],
        "calibration_request_ids": report["calibration_validation_request_ids"],
        "selection": selection_metrics(report),
        "elapsed_seconds": elapsed_seconds,
        "output_dir": str(out_dir.resolve()),
    }


def summarize_finalists(
    rows: list[dict], baseline: Configuration, challenger: Configuration
) -> dict:
    """Compare three matched seeds without using test outcomes."""
    finalists = {}
    for config in (baseline, challenger):
        members = sorted(
            [row for row in rows if row["configuration"] == config.as_dict()],
            key=lambda row: row["seed"],
        )
        if [row["seed"] for row in members] != list(SEEDS):
            raise ValueError(f"three matched seeds required for {config.name}")
        finalists[config.name] = {
            "configuration": config.as_dict(),
            "seeds": list(SEEDS),
            "prefill_brier_by_seed": [
                x["selection"]["PREFILL_COMPLETE"]["mean_brier"] for x in members
            ],
            "decode_brier_by_seed": [
                x["selection"]["DECODE"]["mean_brier"] for x in members
            ],
            "best_epoch_by_seed": [x["best_epoch"] for x in members],
        }
        for phase in ("prefill", "decode"):
            finalists[config.name][f"mean_{phase}_brier"] = float(
                np.mean(finalists[config.name][f"{phase}_brier_by_seed"])
            )
    a, b = finalists[baseline.name], finalists[challenger.name]
    differences = {
        phase: [
            candidate - reference
            for reference, candidate in zip(
                a[f"{phase}_brier_by_seed"], b[f"{phase}_brier_by_seed"]
            )
        ]
        for phase in ("prefill", "decode")
    }
    return {
        "baseline": baseline.name,
        "challenger": challenger.name,
        "finalists": finalists,
        "challenger_minus_baseline_brier_by_seed": differences,
        "selection_rule": "seed-42 raw selection Prefill mean Brier at 128/256/512; "
        "Decode Brier and Prefill binary log loss break ties",
        "note": "Exploratory selection on 100 validation requests. No test outcome "
        "is used to choose configurations. Three seeds do not establish "
        "generalization or tail calibration; independent requests are still needed.",
    }


def run_regularization(
    args: argparse.Namespace, revision: str, names: list[str]
) -> None:
    meta, data = validate_capture(args)
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    source_files = (
        "preflight.json",
        "summary.json",
        "input_requests.jsonl",
        "labels.jsonl",
        "sample_index.jsonl",
        "features/features.sqlite3",
    )
    hashes = {name: sha256_file(args.run_dir / name) for name in source_files}
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
        "model_path": str(args.model.resolve()),
        "gpu_names": names,
        "torch_version": str(torch.__version__),
        "grid": {
            "dropouts": DROPOUTS,
            "weight_decays": WEIGHT_DECAYS,
            "initial_learning_rate": INITIAL_LR,
            "slow_learning_rate": SLOW_LR,
            "seeds": SEEDS,
            "hidden_width": 256,
            "bin_step": 32,
            "epochs": 40,
            "patience": 8,
            "batch_size": 256,
        },
    }
    write_json(args.out_dir / "regularization_preflight.json", preflight)
    for name in source_files[:-1]:
        shutil.copyfile(args.run_dir / name, args.out_dir / Path(name).name)
    shutil.copyfile(
        args.baseline_report, args.out_dir / "previous_baseline_report.json"
    )
    rows = []
    selected_ids = None
    calibrated_ids = None
    complete = False

    def fit(config: Configuration, seed: int) -> dict:
        nonlocal selected_ids, calibrated_ids
        name = f"{config.name}_seed{seed}"
        directory = args.out_dir / name
        print(f"fit={len(rows) + 1}/12 configuration={name}", flush=True)
        started = time.monotonic()
        report = fit_distribution(
            data,
            meta,
            directory,
            revision=revision,
            epochs=40,
            patience=8,
            batch_size=256,
            hidden_width=256,
            bin_step=32,
            learning_rate=config.learning_rate,
            dropout=config.dropout,
            weight_decay=config.weight_decay,
            seed=seed,
        )
        row = summarize_run(report, config, seed, time.monotonic() - started, directory)
        if selected_ids is None:
            selected_ids = row["selection_request_ids"]
            calibrated_ids = row["calibration_request_ids"]
            if (
                not selected_ids
                or not calibrated_ids
                or set(selected_ids) & set(calibrated_ids)
            ):
                raise ValueError("invalid validation partition")
        if (
            row["selection_request_ids"] != selected_ids
            or row["calibration_request_ids"] != calibrated_ids
        ):
            raise ValueError("validation partitions changed between fits")
        rows.append(row)
        write_json(args.out_dir / "progress.json", {"completed_fits": rows})
        print(
            json.dumps(
                {
                    "completed": name,
                    "seconds": row["elapsed_seconds"],
                    "selection_prefill_brier": ranking_key(row)[0],
                    "selection_decode_brier": ranking_key(row)[1],
                }
            ),
            flush=True,
        )
        return row

    try:
        initial = [
            fit(Configuration(dropout, decay, INITIAL_LR), 42)
            for dropout in DROPOUTS
            for decay in WEIGHT_DECAYS
        ]
        top_two = sorted(initial, key=ranking_key)[:2]
        print(
            "initial_top_two=" + json.dumps([r["configuration_name"] for r in top_two]),
            flush=True,
        )
        slow = [
            fit(
                Configuration(
                    row["configuration"]["dropout"],
                    row["configuration"]["weight_decay"],
                    SLOW_LR,
                ),
                42,
            )
            for row in top_two
        ]
        baseline = Configuration(*BASELINE)
        challenger_row = choose_challenger(initial + slow)
        challenger = Configuration(**challenger_row["configuration"])
        print("challenger=" + challenger.name, flush=True)
        for config in (baseline, challenger):
            for seed in SEEDS[1:]:
                fit(config, seed)
        if len(rows) != 12 or any(
            sha256_file(args.run_dir / name) != digest
            for name, digest in hashes.items()
        ):
            raise ValueError("fit count or capture file hashes changed")
        comparison = summarize_finalists(rows, baseline, challenger)
        comparison["initial_grid_ranking"] = [
            row["configuration_name"] for row in sorted(initial, key=ranking_key)
        ]
        comparison["slow_lr_candidates"] = [row["configuration_name"] for row in slow]
        comparison["all_seed42_ranking"] = [
            row["configuration_name"] for row in sorted(initial + slow, key=ranking_key)
        ]
        write_json(args.out_dir / "comparison.json", comparison)
        lines = [
            "# Decoder31 regularization study",
            "",
            "All rankings use selection-validation Prefill risk only.",
            "The previous 200-request test set is not used for selection.",
            "",
            (
                "| Candidate | Seed | Best epoch | Selection Prefill Brier | "
                "Selection Decode Brier |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['configuration_name']} | {row['seed']} | "
                f"{row['best_epoch']} | {ranking_key(row)[0]:.6f} | "
                f"{ranking_key(row)[1]:.6f} |"
            )
        lines += [
            "",
            "Finalists: " + baseline.name + ", " + challenger.name,
            "",
            "Challenger − baseline Prefill Brier by seed: "
            + json.dumps(
                comparison["challenger_minus_baseline_brier_by_seed"]["prefill"]
            ),
            "",
            "A new frozen dataset is needed before accepting a new model.",
        ]
        (args.out_dir / "comparison.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        complete = True
        print(
            "regularization_complete="
            + json.dumps(
                {
                    "baseline": baseline.name,
                    "challenger": challenger.name,
                    "prefill_differences_by_seed": comparison[
                        "challenger_minus_baseline_brier_by_seed"
                    ]["prefill"],
                }
            ),
            flush=True,
        )
    finally:
        write_json(
            args.out_dir / "status.json",
            {
                "completed": complete,
                "completed_fits": len(rows),
                "required_fits": 12,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-hostname", required=True)
    args = parser.parse_args()
    if socket.gethostname() != args.expected_hostname:
        parser.error("hostname differs from expected")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if revision != args.expected_revision:
        parser.error("HEAD differs from expected")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip():
        parser.error("worktree has uncommitted files")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    if [name.strip() for name in names] != ["NVIDIA A100-PCIE-40GB"]:
        parser.error(f"GPU inventory differs: {names}")
    if args.out_dir.exists():
        parser.error("out-dir must be new")
    print(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "revision": revision,
                "capture_dir": str(args.run_dir.resolve()),
                "source": str(args.source.resolve()),
                "model": str(args.model.resolve()),
                "gpu_names": names,
                "out_dir": str(args.out_dir.resolve()),
            }
        ),
        flush=True,
    )
    run_regularization(args, revision, names)


if __name__ == "__main__":
    main()
