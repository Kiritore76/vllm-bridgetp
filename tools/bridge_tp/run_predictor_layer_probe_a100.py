"""Screen four feature layers with 300 paired requests before a full run."""

import argparse
import json
import shutil
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
from predictor_distribution import binary_risk_metrics, probability_remaining_gt
from prepare_oasst1_predictor_inputs import SOURCE_SHA256
from run_predictor_large_a100 import (
    MODEL_SHA,
    REPO,
    archive,
    preserve_incomplete,
    run_logged,
    validate_shard,
)
from train_length_predictor import sha256_file

INPUT_SHA = "6903df497a1603e32ab67589ab5217d826f555ba9df93a8711aa304728e62c24"
LAYERS = ("decoder:17", "decoder:23", "decoder:31", "final")


def rank_layers(root: Path) -> list[dict]:
    """Rank using validation only; retain test results for later diagnosis."""
    ranking = []
    selection_ids = None
    for layer in LAYERS:
        trained = root / ("trained-" + layer.replace(":", ""))
        report = json.loads((trained / "report.json").read_text())
        current_ids = report["selection_validation_request_ids"]
        if selection_ids is not None and current_ids != selection_ids:
            raise ValueError("layer selection validation requests differ")
        selection_ids = current_ids
        with np.load(trained / "distribution_predictions.npz") as p:
            mask = (
                np.isin(p["request_ids"], current_ids)
                & (p["phases"] == "PREFILL_COMPLETE")
                & ~p["censored"]
            )
            if not mask.any():
                raise ValueError("no exact prefill validation examples")
            metrics = {}
            for h in (128, 256, 512):
                risk = probability_remaining_gt(
                    p["probabilities"][mask], p["category_upper_edges"], h
                )
                metrics[str(h)] = binary_risk_metrics(
                    risk, p["observed_remaining"][mask] > h, p["request_ids"][mask]
                )
            ranking.append(
                {
                    "feature_layer": layer,
                    "selection_prefill_requests": int(mask.sum()),
                    "mean_prefill_brier": float(
                        np.mean([x["brier"] for x in metrics.values()])
                    ),
                    "selection_calibrated_nll": report["results"][
                        "selection_validation"
                    ]["calibrated_nll"],
                    "best_epoch": report["best_epoch"],
                    "prefill_risk_by_horizon": metrics,
                }
            )
    ranking.sort(
        key=lambda row: (row["mean_prefill_brier"], row["selection_calibrated_nll"])
    )
    (root / "layer_ranking.json").write_text(
        json.dumps(
            {
                "ranking_rule": (
                    "validation prefill mean Brier at 128/256/512; NLL breaks ties"
                ),
                "note": (
                    "small pilot screening; neither global optimum "
                    "nor calibrated rare-OOM proof"
                ),
                "rankings": ranking,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ranking


def run_probe(args: argparse.Namespace, revision: str, names: list[str]) -> None:
    inputs = args.data_root / "inputs"
    if not inputs.exists():
        run_logged(
            [
                sys.executable,
                "tools/bridge_tp/prepare_oasst1_predictor_inputs.py",
                "--source",
                str(args.source),
                "--out-dir",
                str(inputs),
                "--exclude-legacy-train1000",
                "--layer-probe",
            ],
            args.data_root / "prepare.log",
        )
    input_path = inputs / "oasst1_pilot_requests.jsonl"
    if sha256_file(input_path) != INPUT_SHA:
        raise ValueError("probe input SHA differs")
    for name, limit in (("smoke", 8), ("capture", None)):
        out = args.data_root / name
        complete = False
        if out.exists():
            try:
                if limit is None:
                    for layer in LAYERS:
                        validate_shard(
                            out / "layers" / layer.replace(":", ""),
                            input_path,
                            revision,
                            layer,
                        )
                else:
                    summaries = json.loads((out / "summary.json").read_text())
                    if set(summaries) != set(LAYERS) or any(
                        x["requests"] != limit for x in summaries.values()
                    ):
                        raise ValueError("smoke summary differs")
                    if (
                        json.loads((out / "preflight.json").read_text())["revision"]
                        != revision
                    ):
                        raise ValueError("smoke revision differs")
                complete = True
            except (OSError, ValueError, KeyError, sqlite3.Error):
                preserve_incomplete(out)
        if not complete:
            command = [
                sys.executable,
                "tools/bridge_tp/run_predictor_capture.py",
                "--input",
                str(input_path),
                "--model",
                str(args.model),
                "--out-dir",
                str(out),
                "--expected-revision",
                revision,
                "--expected-input-sha256",
                INPUT_SHA,
                "--expected-gpu-name",
                names[0],
                "--expected-gpu-count",
                "1",
                "--feature-layers",
                ",".join(LAYERS),
                "--interval",
                "20",
                "--max-tokens",
                "4096",
                "--max-model-len",
                "6144",
            ]
            if limit:
                command += ["--limit", str(limit)]
            log = args.data_root / (name + ".runner.log")
            try:
                run_logged(command, log)
            finally:
                if out.exists():
                    shutil.copyfile(log, out / "runner.log")
        print(f"{name}: paired layer capture PASS", flush=True)
    for layer in LAYERS:
        layer_dir = args.data_root / "capture/layers" / layer.replace(":", "")
        trained = args.data_root / ("trained-" + layer.replace(":", ""))
        if trained.exists():
            # Each retry retains its previous training output.
            preserve_incomplete(trained)
        log = args.data_root / (trained.name + ".runner.log")
        command = [
            sys.executable,
            "tools/bridge_tp/train_predictor_distribution.py",
            "--run-dir",
            str(layer_dir),
            "--out-dir",
            str(trained),
            "--expected-revision",
            revision,
            "--expected-capture-revision",
            revision,
            "--expected-input-sha256",
            INPUT_SHA,
            "--expected-feature-layer",
            layer,
            "--expected-gpu-name",
            names[0],
            "--expected-gpu-count",
            "1",
            "--learning-rate",
            "0.0003",
        ]
        try:
            run_logged(command, log)
        finally:
            if trained.exists():
                shutil.copyfile(log, trained / "runner.log")
    ranking = rank_layers(args.data_root)
    print(
        json.dumps(
            {
                "validation_layer_ranking": [
                    {
                        "layer": x["feature_layer"],
                        "prefill_brier": x["mean_prefill_brier"],
                    }
                    for x in ranking
                ]
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/bridgetp/results/length_predictor/qwen14b-layer-probe300-v1"
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/root/autodl-tmp/bridgetp/length_predictor/inputs/2023-04-12_oasst_prompts.messages.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master"
        ),
    )
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if (
        revision != args.expected_revision
        or socket.gethostname() != args.expected_hostname
    ):
        parser.error("machine or HEAD differs from expected")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip():
        parser.error("uncommitted files; preserve them first")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if names != ["NVIDIA A100-PCIE-40GB"]:
        parser.error(f"GPU inventory differs: {names}")
    if (
        sha256_file(args.source) != SOURCE_SHA256
        or sha256_file(args.model / "config.json") != MODEL_SHA
    ):
        parser.error("raw source or model config SHA differs")
    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "qwen2" or config.get("num_hidden_layers") != 48:
        parser.error("this probe is pinned to the 48-layer Qwen2.5-14B model")
    args.data_root.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock_handle = (args.data_root / ".run.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another probe is using this data-root")
    print(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "revision": revision,
                "gpu_names": names,
                "source": str(args.source.resolve()),
                "model": str(args.model.resolve()),
                "layers": LAYERS,
                "data_root": str(args.data_root.resolve()),
            }
        ),
        flush=True,
    )
    try:
        run_probe(args, revision, names)
    finally:
        archive(args.data_root)


if __name__ == "__main__":
    main()
