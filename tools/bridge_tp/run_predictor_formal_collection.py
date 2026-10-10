"""Run pinned single-GPU shards and preserve failures for diagnosis."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from train_length_predictor import load_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    parser.add_argument("--shard-stop", type=int, default=3)
    args = parser.parse_args()
    inputs = args.batch / "inputs"
    manifest = json.loads((inputs / "manifest.json").read_text())
    recipe_sha = hashlib.sha256((inputs / "recipe.jsonl").read_bytes()).hexdigest()
    if recipe_sha != args.expected_recipe_sha256:
        raise ValueError("recipe SHA differs")
    if manifest["recipe_sha256"] != recipe_sha:
        raise ValueError("manifest recipe SHA differs")
    if not 0 < args.shard_stop <= len(manifest["shards"]):
        raise ValueError("shard-stop outside manifest")
    labels = []
    completed = []
    for number, shard in enumerate(manifest["shards"][: args.shard_stop], 1):
        source = inputs / shard["input"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError(f"input SHA differs: {source}")
        out = args.batch / "captures" / shard["name"]
        if out.exists() and not (out / "summary.json").exists():
            preserved = out.with_name(out.name + f"-incomplete-{time.time_ns()}")
            out.rename(preserved)
            print(f"[进度] 保留未完成批次：{preserved}", flush=True)
        if not out.exists():
            print(
                f"[进度] 批次 {number}/{args.shard_stop}：{shard['name']}，"
                f"已完成{len(labels)}/600条",
                flush=True,
            )
            command = [
                sys.executable,
                "tools/bridge_tp/run_predictor_capture.py",
                "--input",
                str(source),
                "--model",
                args.model,
                "--out-dir",
                str(out),
                "--expected-revision",
                args.expected_revision,
                "--expected-input-sha256",
                shard["sha256"],
                "--expected-gpu-name",
                "NVIDIA A100-PCIE-40GB",
                "--expected-gpu-count",
                str(args.expected_gpu_count),
                "--feature-layer",
                "decoder:31",
                "--interval",
                "20",
                "--max-model-len",
                "16384",
                "--max-tokens",
                str(shard["max_tokens"]),
                "--gpu-memory-utilization",
                "0.85",
            ]
            subprocess.run(command, check=True)
        # Completed shards must pass a fresh label/index/SQLite audit.
        data = load_examples(out)
        preflight = json.loads((out / "preflight.json").read_text())
        if (
            preflight["revision"] != args.expected_revision
            or preflight["input_sha256"] != shard["sha256"]
            or len(data["labels"]) != shard["rows"]
        ):
            raise ValueError(f"completed shard identity differs: {out}")
        labels.extend(data["labels"])
        completed.append(shard["name"])
        print(
            f"[进度] 正式采集已完成{len(labels)}/600条，本批审计通过：{shard['name']}",
            flush=True,
        )
    histogram = Counter()
    for row in labels:
        if row["natural_finish"]:
            length = row["output_tokens"]
            bucket = next(
                (
                    str(edge)
                    for edge in (2048, 4096, 8192, 12288, 16384)
                    if length <= edge
                ),
                "overflow",
            )
            histogram[bucket] += 1
    summary = {
        "completed_shards": completed,
        "requests": len(labels),
        "natural_eos": sum(r["natural_finish"] for r in labels),
        "censored": sum(not r["natural_finish"] for r in labels),
        "actual_length_upper_edge_histogram": dict(histogram),
        "all_600_collected": len(labels) == 600,
        "training_started": False,
    }
    (args.batch / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[进度] 本轮采集完成：{len(labels)}/600条；尚未开始训练", flush=True)


if __name__ == "__main__":
    main()
