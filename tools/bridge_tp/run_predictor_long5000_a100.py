"""Capture new natural/long-form requests and compare 1600/4000 training trees."""

import argparse
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from predictor_distribution import binary_risk_metrics, probability_remaining_gt
from prepare_oasst1_predictor_inputs import SOURCE_SHA256
from prepare_predictor_long5000 import prepare, ranked
from run_predictor_large_a100 import (
    MODEL_SHA,
    REPO,
    archive,
    merge_shards,
    preserve_incomplete,
    run_logged,
    validate_shard,
)
from train_length_predictor import load_examples, read_jsonl, sha256_file
from train_predictor_distribution import fit_distribution

INPUT_SHA = "75cae22e298548b54b6164b3df9adc9ebdc61ced3a85eecfd7df9e96ea7a1be3"
MAX_TOKENS = 8192
MAX_MODEL_LEN = 10240
HORIZONS = (128, 256, 512, 1024, 2048, 4096)


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def coverage(labels: list[dict], inputs: list[dict]) -> dict:
    by_id = {r["id"]: r for r in inputs}
    groups = {}
    for split in ("train", "validation", "test"):
        for group in ("natural", "long_form"):
            rows = [
                r
                for r in labels
                if r["split"] == split
                and by_id[r["input_id"]]["workload_group"] == group
            ]
            lengths = [r["output_tokens"] for r in rows]
            groups[f"{split}/{group}"] = {
                "requests": len(rows),
                "natural_finishes": sum(r["natural_finish"] for r in rows),
                "censored": sum(not r["natural_finish"] for r in rows),
                "mean_output_tokens": float(np.mean(lengths)) if rows else None,
                "quantiles": np.quantile(lengths, [0.5, 0.9, 0.99]).tolist()
                if rows
                else [],
                "observed_output_gt": {
                    str(h): sum(n > h for n in lengths) for h in HORIZONS
                },
            }
    return groups


def training_view(data: dict, inputs: list[dict], count: int) -> dict:
    """Nested training subsets; keep exactly the same held-out requests."""
    selected = []
    for group, quota in (("natural", count * 4 // 5), ("long_form", count // 5)):
        pool = ranked(
            [
                r
                for r in inputs
                if r["split"] == "train" and r["workload_group"] == group
            ],
            "learning-curve-long5000-v1:",
        )
        if len(pool) < quota:
            raise ValueError("insufficient training requests in stratum")
        selected += [r["id"] for r in pool[:quota]]
    if len(set(selected)) != count:
        raise ValueError("training request count differs")
    mask = (data["splits"] != "train") | np.isin(data["requests"], selected)
    view = {k: v[mask] if isinstance(v, np.ndarray) else v for k, v in data.items()}
    view["labels"] = [
        r
        for r in data["labels"]
        if r["split"] != "train" or r["input_id"] in set(selected)
    ]
    view["audit"] = {
        **data["audit"],
        "requests": len(view["labels"]),
        "naturally_finished_requests": sum(r["natural_finish"] for r in view["labels"]),
        "censored_requests": sum(not r["natural_finish"] for r in view["labels"]),
        "samples": int(mask.sum()),
        "samples_with_exact_remaining_length": int((~view["censored"]).sum()),
        "phases": dict(Counter(view["phases"].tolist())),
    }
    return view


def workload_metrics(directory: Path, inputs: list[dict], report: dict) -> dict:
    """Report natural and augmented tasks separately, excluding censored labels."""
    groups = {
        g: [r["id"] for r in inputs if r["workload_group"] == g]
        for g in ("natural", "long_form")
    }
    result = {}
    with np.load(directory / "distribution_predictions.npz") as f:
        for split in ("selection_validation", "calibration_validation", "test"):
            eligible = (
                (f["splits"] == "test")
                if split == "test"
                else np.isin(f["request_ids"], report[f"{split}_request_ids"])
            )
            p = f["probabilities"].astype(np.float64)
            if split == "selection_validation":
                p **= report["temperature"]
                p /= p.sum(axis=1, keepdims=True)
            for group, ids in groups.items():
                for phase in ("PREFILL_COMPLETE", "DECODE"):
                    mask = eligible & np.isin(f["request_ids"], ids)
                    mask &= (f["phases"] == phase) & ~f["censored"]
                    if not mask.any():
                        continue
                    metrics = {}
                    for h in HORIZONS:
                        predicted = probability_remaining_gt(
                            p[mask], f["category_upper_edges"], h
                        )
                        metrics[str(h)] = binary_risk_metrics(
                            predicted,
                            f["observed_remaining"][mask] > h,
                            f["request_ids"][mask],
                        )
                    result[f"{split}/{group}/{phase}"] = {
                        "probability_kind": "raw"
                        if split == "selection_validation"
                        else "calibrated",
                        "horizons": metrics,
                    }
    return result


def execute(args, revision, names):
    root = args.data_root
    root.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (root / ".run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    if shutil.disk_usage(root).free < 30 * 2**30:
        raise ValueError(
            "less than 30 GiB free; preserve room for captures and archives"
        )
    inputs = root / "inputs-5000"
    if not inputs.exists():
        prepare(args.source, inputs)
    full = inputs / "requests.jsonl"
    if sha256_file(full) != INPUT_SHA:
        raise ValueError("frozen 5000-request input SHA differs")
    manifest = json.loads((inputs / "manifest.json").read_text())
    if (
        len(manifest["shards"]) != 10
        or b"".join((inputs / s["filename"]).read_bytes() for s in manifest["shards"])
        != full.read_bytes()
    ):
        raise ValueError("shards do not cover the frozen input")
    write_json(
        root / "experiment_preflight.json",
        {
            "hostname": socket.gethostname(),
            "revision": revision,
            "gpu_names": names,
            "source": str(args.source.resolve()),
            "source_sha256": SOURCE_SHA256,
            "model": str(args.model.resolve()),
            "model_config_sha256": MODEL_SHA,
            "input_sha256": INPUT_SHA,
            "feature_layer": "decoder:31",
            "max_tokens": MAX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "train_requests": [1600, 4000],
            "seeds": [42, 43, 44],
            "disk_free_gib": shutil.disk_usage(root).free / 2**30,
        },
    )

    def validate(run, path):
        meta = validate_shard(
            run,
            path,
            revision,
            "decoder:31",
            max_tokens=MAX_TOKENS,
            max_model_len=MAX_MODEL_LEN,
        )
        if meta["model_path"] != str(args.model.resolve()):
            raise ValueError("model path changed")
        return meta

    runs = []
    for number, shard in enumerate(manifest["shards"]):
        path = inputs / shard["filename"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError("shard SHA differs")
        run = root / f"shard_{number:03d}"
        print(f"batch={number + 1}/10 directory={run}", flush=True)
        if run.exists():
            try:
                validate(run, path)
                runs.append(run)
                print("completed batch audit PASS; reusing", flush=True)
                continue
            except (OSError, ValueError, KeyError, sqlite3.Error):
                preserve_incomplete(run)
        log = root / f"shard_{number:03d}.runner.log"
        try:
            run_logged(
                [
                    sys.executable,
                    "tools/bridge_tp/run_predictor_capture.py",
                    "--input",
                    str(path),
                    "--model",
                    str(args.model),
                    "--out-dir",
                    str(run),
                    "--expected-revision",
                    revision,
                    "--expected-input-sha256",
                    shard["sha256"],
                    "--expected-gpu-name",
                    names[0],
                    "--expected-gpu-count",
                    "1",
                    "--feature-layer",
                    "decoder:31",
                    "--interval",
                    "20",
                    "--max-tokens",
                    str(MAX_TOKENS),
                    "--max-model-len",
                    str(MAX_MODEL_LEN),
                ],
                log,
            )
        finally:
            if run.exists():
                shutil.copyfile(path, run / "input_requests.jsonl")
                if log.exists():
                    shutil.copyfile(log, run / "runner.log")
        validate(run, path)
        runs.append(run)
        partial_labels = [
            label for r in runs for label in read_jsonl(r / "labels.jsonl")
        ]
        write_json(
            root / "coverage_progress.json", coverage(partial_labels, read_jsonl(full))
        )
    capture = root / "capture-5000"
    if capture.exists():
        validate(capture, full)
    else:
        merge_shards(runs, full, capture)
        shutil.copyfile(inputs / "manifest.json", capture / "input_manifest.json")
        validate(capture, full)
    input_rows = read_jsonl(full)
    labels = read_jsonl(capture / "labels.jsonl")
    write_json(capture / "length_coverage.json", coverage(labels, input_rows))
    archive(capture)
    data = load_examples(capture, include_censored=True)
    if data["audit"]["censored_requests"] > 1000:
        raise ValueError(
            "more than 20% capped; inspect length coverage before training"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trained = root / f"trained-5000-{stamp}"
    trained.mkdir(exist_ok=False)
    for name in (
        "preflight.json",
        "summary.json",
        "input_requests.jsonl",
        "input_manifest.json",
        "length_coverage.json",
    ):
        shutil.copyfile(capture / name, trained / name)
    shutil.copyfile(
        root / "experiment_preflight.json", trained / "experiment_preflight.json"
    )
    hashes = {
        name: sha256_file(capture / name)
        for name in (
            "input_requests.jsonl",
            "labels.jsonl",
            "sample_index.jsonl",
            "features/features.sqlite3",
        )
    }
    write_json(trained / "capture_sha256.json", hashes)
    fits = []
    complete = False
    try:
        meta = json.loads((capture / "preflight.json").read_text())
        for count in (1600, 4000):
            view = training_view(data, input_rows, count)
            selected = sorted(
                {r["input_id"] for r in view["labels"] if r["split"] == "train"}
            )
            write_json(
                trained / f"training_view_{count}.json",
                {
                    "source_capture_audit": data["audit"],
                    "view_audit": view["audit"],
                    "train_request_ids": selected,
                    "note": "training subset only; held-out requests unchanged",
                },
            )
            for seed in (42, 43, 44):
                name = f"train{count}_seed{seed}"
                print(f"fit={len(fits) + 1}/6 name={name}", flush=True)
                report = fit_distribution(
                    view,
                    meta,
                    trained / name,
                    revision=revision,
                    epochs=40,
                    patience=8,
                    batch_size=256,
                    hidden_width=256,
                    learning_rate=0.0003,
                    dropout=0.1,
                    weight_decay=0.01,
                    seed=seed,
                    bin_step=32,
                )
                write_json(
                    trained / name / "workload_metrics.json",
                    workload_metrics(trained / name, input_rows, report),
                )
                fits.append(
                    {
                        "name": name,
                        "seed": seed,
                        "train_requests": count,
                        "best_epoch": report["best_epoch"],
                        "checkpoint_sha256": report["checkpoint_sha256"],
                    }
                )
                write_json(trained / "progress.json", {"completed_fits": fits})
        for name, expected in hashes.items():
            if sha256_file(capture / name) != expected:
                raise ValueError("source capture changed during training")
        complete = True
    finally:
        write_json(
            trained / "status.json",
            {"completed": complete, "completed_fits": len(fits), "required_fits": 6},
        )
        archive(trained)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if (
        revision != args.expected_revision
        or socket.gethostname() != args.expected_hostname
    ):
        parser.error("machine or HEAD differs")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip():
        parser.error("uncommitted files; preserve before running")
    if names != ["NVIDIA A100-PCIE-40GB"]:
        parser.error(f"GPU differs: {names}")
    if (
        sha256_file(args.source) != SOURCE_SHA256
        or sha256_file(args.model / "config.json") != MODEL_SHA
    ):
        parser.error("raw dataset or model SHA differs")
    execute(args, revision, names)


if __name__ == "__main__":
    main()
