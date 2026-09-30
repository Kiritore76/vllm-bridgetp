"""Capture 2000 or 10000 new OASST1 requests in shared batches, then train."""

import argparse
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from prepare_oasst1_predictor_inputs import SOURCE_SHA256
from run_predictor_capture import audit_capture, load_requests
from train_length_predictor import read_jsonl, sha256_file

INPUT_SHAS = {
    2000: "8dc77e63c2b93d8448466f9ee2fbaa4f24489678c9049c52e8d1fd81c3c9424b",
    10000: "02ed1206ff67e59ee704c47492d894a9d178a35c07540516f2444710737972c3",
}
MODEL_SHA = "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e"
REPO = Path(__file__).resolve().parents[2]
COMMON_FIELDS = (
    "revision",
    "model_path",
    "model_config_sha256",
    "gpu_names",
    "interval",
    "max_tokens",
    "max_model_len",
    "temperature",
)


def validate_shard(run: Path, input_path: Path, revision: str) -> dict:
    """Reuse a batch only when inputs, protocol and independent audit agree."""
    preflight = json.loads((run / "preflight.json").read_text())
    if (
        preflight["revision"] != revision
        or preflight["input_sha256"] != sha256_file(input_path)
        or preflight["model_config_sha256"] != MODEL_SHA
        or preflight["max_tokens"] != 4096
        or preflight["max_model_len"] != 6144
        or preflight["interval"] != 20
        or preflight["temperature"] != 0
        or preflight["gpu_names"] != ["NVIDIA A100-PCIE-40GB"]
    ):
        raise ValueError(f"batch protocol/input differs: {run}")
    labels = read_jsonl(run / "labels.jsonl")
    requests = load_requests(input_path, None)
    inputs_by_id = {row["id"]: row for row in requests}
    if (
        len(labels) != len(requests)
        or {x["input_id"] for x in labels} != {x["id"] for x in requests}
        or len({x["request_id"] for x in labels}) != len(labels)
        or preflight["requests"] != len(requests)
    ):
        raise ValueError(f"batch response coverage differs: {run}")
    for label in labels:
        row = inputs_by_id[label["input_id"]]
        if any(
            label.get(key) != row.get(key)
            for key in ("split", "lang", "source_tree_id")
        ):
            raise ValueError(f"batch label metadata differs: {run}")
    audit = audit_capture(run / "features", {x["request_id"]: x for x in labels})
    if audit != json.loads((run / "summary.json").read_text()):
        raise ValueError(f"batch summary differs: {run}")
    if audit["phases"].get("PREFILL_COMPLETE") != len(requests):
        raise ValueError(f"batch missing prefill states: {run}")
    if sha256_file(run / "input_requests.jsonl") != preflight["input_sha256"]:
        raise ValueError(f"batch archived input differs: {run}")
    return preflight


def merge_shards(runs: list[Path], full_input: Path, out: Path) -> dict:
    """Prefix engine IDs and merge SQLite features without request collisions."""
    if out.exists():
        raise ValueError("merge destination must be new")
    out.mkdir(parents=True)
    (out / "features").mkdir()
    labels, sources = [], []
    seen_inputs = set()
    preflight = None
    with closing(sqlite3.connect(out / "features/features.sqlite3")) as db:
        db.execute("""CREATE TABLE samples (
            sample_id INTEGER PRIMARY KEY, request_id TEXT,
            generated_tokens INTEGER, phase TEXT, hidden_size INTEGER,
            hidden_fp16 BLOB, captured_unix_ns INTEGER)""")
        for number, run in enumerate(runs):
            meta = json.loads((run / "preflight.json").read_text())
            if preflight is None:
                preflight = meta.copy()
            if any(meta[key] != preflight[key] for key in COMMON_FIELDS):
                raise ValueError("capture protocol changed between batches")
            prefix = f"shard{number:03d}:"
            source_labels = read_jsonl(run / "labels.jsonl")
            audit = audit_capture(
                run / "features", {x["request_id"]: x for x in source_labels}
            )
            if audit != json.loads((run / "summary.json").read_text()):
                raise ValueError("source batch audit differs")
            for label in source_labels:
                if label["input_id"] in seen_inputs:
                    raise ValueError("duplicate input across batches")
                seen_inputs.add(label["input_id"])
                labels.append({**label, "request_id": prefix + label["request_id"]})
            with closing(sqlite3.connect(run / "features/features.sqlite3")) as source:
                for row in source.execute("""SELECT request_id, generated_tokens,
                    phase, hidden_size, hidden_fp16, captured_unix_ns
                    FROM samples ORDER BY sample_id"""):
                    db.execute(
                        "INSERT INTO samples VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                        (prefix + row[0], *row[1:]),
                    )
            sources.append(
                {
                    "path": str(run.resolve()),
                    "input_sha256": meta["input_sha256"],
                    "requests": len(source_labels),
                }
            )
            db.commit()
    expected = load_requests(full_input, None)
    if seen_inputs != {x["id"] for x in expected} or len(labels) != len(expected):
        raise ValueError("merged responses do not cover the full input")
    (out / "labels.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labels),
        encoding="utf-8",
    )
    shutil.copyfile(full_input, out / "input_requests.jsonl")
    preflight.update(
        input_path=str(full_input.resolve()),
        input_sha256=sha256_file(full_input),
        requests=len(labels),
        source_shards=sources,
    )
    (out / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    summary = audit_capture(
        out / "features",
        {x["request_id"]: x for x in labels},
        out / "sample_index.jsonl",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_logged(command: list[str], log: Path) -> None:
    """Show child progress while retaining its complete log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            rc = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if rc:
        raise RuntimeError(f"child exit code {rc}; log: {log}")


def archive(directory: Path) -> None:
    print(f"packing={directory}", flush=True)
    subprocess.run(
        [
            "tar",
            "-czf",
            str(directory) + ".tar.gz",
            "-C",
            str(directory.parent),
            directory.name,
        ],
        check=True,
    )
    print(f"archive_to_retrieve={directory}.tar.gz", flush=True)


def preserve_incomplete(directory: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = directory.with_name(directory.name + "-incomplete-" + stamp)
    directory.rename(backup)
    print(f"preserved_incomplete={backup}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--requests", type=int, choices=(2000, 10000), default=2000)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/bridgetp/results/length_predictor/oasst1-staged-v2"
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
    expected_input_sha = INPUT_SHAS[args.requests]
    batch_count = args.requests // 500
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
        parser.error("uncommitted files; preserve them before running")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if names != ["NVIDIA A100-PCIE-40GB"]:
        parser.error(f"GPU inventory differs: {names}")
    if (
        sha256_file(args.source) != SOURCE_SHA256
        or sha256_file(args.model / "config.json") != MODEL_SHA
    ):
        parser.error("raw dataset or model config SHA differs")
    args.data_root.mkdir(parents=True, exist_ok=True)
    # Hold this lock through capture and training; concurrent resumptions must stop.
    import fcntl

    lock_handle = (args.data_root / ".run.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another run is using this data-root; do not start a second job")
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(str(os.getpid()) + "\n")
    lock_handle.flush()
    print(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "revision": revision,
                "gpu_names": names,
                "source": str(args.source.resolve()),
                "model": str(args.model.resolve()),
                "data_root": str(args.data_root.resolve()),
                "disk_free_gib": shutil.disk_usage(args.data_root).free / 2**30,
            }
        ),
        flush=True,
    )
    inputs = args.data_root / f"inputs-{args.requests}"
    if not inputs.exists():
        run_logged(
            [
                sys.executable,
                "tools/bridge_tp/prepare_oasst1_predictor_inputs.py",
                "--source",
                str(args.source),
                "--out-dir",
                str(inputs),
                "--staged-requests",
                str(args.requests),
                "--exclude-legacy-train1000",
                "--shard-size",
                "500",
            ],
            args.data_root / "prepare.log",
        )
    full_input = inputs / "oasst1_pilot_requests.jsonl"
    if sha256_file(full_input) != expected_input_sha:
        parser.error("prepared full input SHA differs")
    manifest = json.loads((inputs / "manifest.json").read_text())
    concatenated = b"".join(
        (inputs / x["filename"]).read_bytes() for x in manifest["shards"]
    )
    if (
        concatenated != full_input.read_bytes()
        or len(manifest["shards"]) != batch_count
    ):
        parser.error("batch inputs do not match the pinned full input")
    runs = []
    for number, shard in enumerate(manifest["shards"]):
        input_path = inputs / shard["filename"]
        if sha256_file(input_path) != shard["sha256"]:
            parser.error("batch SHA differs")
        run = args.data_root / f"shard_{number:03d}"
        print(f"batch={number + 1}/{batch_count} directory={run}", flush=True)
        if run.exists():
            try:
                meta = validate_shard(run, input_path, revision)
                if meta["model_path"] != str(args.model.resolve()):
                    raise ValueError("model path changed")
                print("completed batch audit PASS; reusing", flush=True)
                runs.append(run)
                continue
            except (OSError, ValueError, KeyError, sqlite3.Error) as error:
                print(f"incomplete batch: {error}", flush=True)
                preserve_incomplete(run)
        log = args.data_root / f"shard_{number:03d}.runner.log"
        command = [
            sys.executable,
            "tools/bridge_tp/run_predictor_capture.py",
            "--input",
            str(input_path),
            "--model",
            str(args.model),
            "--out-dir",
            str(run),
            "--expected-revision",
            revision,
            "--expected-input-sha256",
            shard["sha256"],
            "--expected-gpu-name",
            "NVIDIA A100-PCIE-40GB",
            "--expected-gpu-count",
            "1",
            "--interval",
            "20",
            "--max-tokens",
            "4096",
            "--max-model-len",
            "6144",
        ]
        try:
            run_logged(command, log)
        finally:
            if run.exists():
                shutil.copyfile(log, run / "runner.log")
                shutil.copyfile(input_path, run / "input_requests.jsonl")
        validate_shard(run, input_path, revision)
        runs.append(run)
    merged = args.data_root / f"capture-{args.requests}"
    if merged.exists():
        try:
            meta = validate_shard(merged, full_input, revision)
            if meta.get("source_shards") is None:
                raise ValueError("merged provenance missing")
        except (OSError, ValueError, KeyError, sqlite3.Error):
            preserve_incomplete(merged)
    if not merged.exists():
        print("merging independently audited batches", flush=True)
        merge_shards(runs, full_input, merged)
        shutil.copyfile(inputs / "manifest.json", merged / "input_manifest.json")
    archive(merged)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    trained = args.data_root / (f"trained-{args.requests}-" + stamp)
    train_command = [
        sys.executable,
        "tools/bridge_tp/train_predictor_distribution.py",
        "--run-dir",
        str(merged),
        "--out-dir",
        str(trained),
        "--expected-revision",
        revision,
        "--expected-capture-revision",
        revision,
        "--expected-input-sha256",
        expected_input_sha,
        "--expected-gpu-name",
        names[0],
        "--expected-gpu-count",
        "1",
        "--learning-rate",
        "0.0003",
    ]
    previous_input = args.data_root / "inputs-2000/oasst1_pilot_requests.jsonl"
    if args.requests == 10000 and previous_input.exists():
        if sha256_file(previous_input) != INPUT_SHAS[2000]:
            raise ValueError("previous-stage input SHA differs")
        train_command += ["--known-test-input", str(previous_input)]
    log = args.data_root / (trained.name + ".runner.log")
    try:
        run_logged(train_command, log)
    finally:
        trained.mkdir(exist_ok=True)
        shutil.copyfile(log, trained / "runner.log")
        archive(trained)


if __name__ == "__main__":
    main()
