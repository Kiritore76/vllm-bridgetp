# SPDX-License-Identifier: Apache-2.0
"""Collect a frozen long-task supplement using four isolated TP1 engines."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from train_length_predictor import load_examples  # noqa: E402

from tools.bridge_tp.prepare_4090d_qwen3 import (  # noqa: E402
    PROFILE,
    read,
    runtime_inventory,
    sha,
)
from tools.bridge_tp.result_archive import pack  # noqa: E402

DATA = ROOT / "tools/bridge_tp/predictor_data/qwen3_long_supplement208_20261028.jsonl"
ASSET_SHA = read(ROOT / "experiments/platforms/qwen3_8b_verified_assets.json")["sha256"]


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def payload(data):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data).encode()


def prepare(tokenizer, output):
    """Freeze token-budget eligibility and disjoint GPU inputs before decoding."""
    source = read(DATA.with_suffix(".manifest.json"))
    if sha(DATA) != source["input_sha256"]:
        raise ValueError("candidate SHA differs")
    candidates = rows(DATA)
    accepted = []
    excluded = []
    families = {}
    ids = set()
    for row in candidates:
        if row["id"] in ids:
            raise ValueError("duplicate candidate ID")
        ids.add(row["id"])
        if families.setdefault(row["source_tree_id"], row["split"]) != row["split"]:
            raise ValueError("task family crosses splits")
        prompt = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        count = len(tokenizer.encode(prompt))
        if count > 6144:
            excluded.append(
                dict(
                    id=row["id"],
                    reason="prompt_exceeds_6144",
                    prompt_tokens=count,
                    split=row["split"],
                )
            )
            continue
        rendered = dict(row, prompt=prompt, planned_prompt_tokens=count)
        rendered.pop("messages")
        accepted.append(rendered)
    pools = [[] for _ in range(4)]
    group_counts = [Counter() for _ in range(4)]
    groups = defaultdict(list)
    for row in accepted:
        groups[row["task_type"], row["split"]].append(row)
    for group, data in sorted(groups.items()):
        for row in sorted(
            data, key=lambda item: hashlib.sha256(item["id"].encode()).hexdigest()
        ):
            gpu = min(
                range(4),
                key=lambda i: (
                    group_counts[i][group],
                    len(pools[i]),
                    sum(r["planned_prompt_tokens"] for r in pools[i]),
                ),
            )
            pools[gpu].append(row)
            group_counts[gpu][group] += 1
    if any(not pool for pool in pools):
        raise ValueError("one GPU has no eligible inputs")
    recipe = payload(accepted)
    expected = source.get("verified_rendered_recipe_sha256")
    if expected and hashlib.sha256(recipe).hexdigest() != expected:
        raise ValueError("rendered tokenizer/template differs from verified recipe")
    output.mkdir(parents=True, exist_ok=False)
    (output / "recipe.jsonl").write_bytes(recipe)
    shutil.copyfile(DATA, output / "candidate_requests.jsonl")
    shutil.copyfile(
        DATA.with_suffix(".manifest.json"), output / "candidate_sources.json"
    )
    shards = []
    for gpu, pool in enumerate(pools):
        name = f"shard{gpu:03d}_gpu{gpu}"
        raw = payload(pool)
        (output / (name + ".jsonl")).write_bytes(raw)
        shards.append(
            dict(
                name=name,
                input=name + ".jsonl",
                rows=len(pool),
                gpu_index=gpu,
                sha256=hashlib.sha256(raw).hexdigest(),
                max_tokens=10240,
            )
        )
    manifest = dict(
        profile="qwen3_long208_four_tp1",
        rows=len(accepted),
        source_rows=len(candidates),
        recipe_sha256=hashlib.sha256(recipe).hexdigest(),
        candidate_sha256=sha(DATA),
        thinking_mode="disabled",
        max_model_len=16384,
        shards=shards,
        split_rows=dict(Counter(r["split"] for r in accepted)),
        task_counts=dict(Counter(r["task_type"] for r in accepted)),
        excluded_before_generation=excluded,
        label_rule=(
            "actual natural EOS; short, capped, repetitive and "
            "out-of-domain outputs retained"
        ),
    )
    write(output / "manifest.json", manifest)
    return manifest


def worker_command(model, output, shard, revision):
    """Physical inventory is four; each child sees only one CUDA device."""
    command = [
        sys.executable,
        "-B",
        "tools/bridge_tp/run_predictor_capture.py",
        "--input",
        str(output / "inputs" / shard["input"]),
        "--model",
        str(model),
        "--out-dir",
        str(output / "captures" / shard["name"]),
        "--expected-revision",
        revision,
        "--expected-input-sha256",
        shard["sha256"],
        "--expected-gpu-name",
        "NVIDIA GeForce RTX 4090 D",
        "--expected-gpu-count",
        "4",
        "--expected-visible-gpu-count",
        "1",
        "--feature-layer",
        "decoder:31",
        "--interval",
        "20",
        "--max-model-len",
        "16384",
        "--max-tokens",
        "10240",
        "--max-num-batched-tokens",
        "2048",
        "--gpu-memory-utilization",
        "0.85",
        "--thinking-mode",
        "disabled",
        "--save-token-ids",
    ]
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("BRIDGETP_")
    }
    env.update(
        CUDA_VISIBLE_DEVICES=str(shard["gpu_index"]),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        VLLM_USE_V2_MODEL_RUNNER="0",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
    )
    return command, env


def audit_shard(output, shard, revision):
    directory = output / "captures" / shard["name"]
    if sha(output / "inputs" / shard["input"]) != shard["sha256"]:
        raise ValueError("shard input digest differs")
    pre = read(directory / "preflight.json")
    if (
        pre["revision"] != revision
        or pre["input_sha256"] != shard["sha256"]
        or pre.get("cuda_visible_devices") != str(shard["gpu_index"])
        or pre.get("cuda_visible_gpu_count") != 1
        or pre.get("tensor_parallel_size") != 1
        or pre["feature_layer"] != "decoder:31"
        or pre["max_model_len"] != 16384
        or pre["max_tokens"] != 10240
        or pre["model_config_sha256"] != ASSET_SHA["config.json"]
        or pre["model_runner"] != "V1"
        or pre["max_num_batched_tokens"] != 2048
        or pre["generation_contract"]
        != dict(thinking_mode="disabled", temperature=0.0, ignore_eos=False)
    ):
        raise ValueError("capture device/configuration identity differs")
    data = load_examples(directory, include_censored=True)
    if data["audit"]["hidden_size"] != 4096:
        raise ValueError("capture hidden width differs")
    inputs = {row["id"]: row for row in rows(output / "inputs" / shard["input"])}
    labels = data["labels"]
    if len(labels) != shard["rows"] or {r["input_id"] for r in labels} != set(inputs):
        raise ValueError("capture IDs/count differ")
    for label in labels:
        request = inputs[label["input_id"]]
        for key in ("split", "source_tree_id"):
            if label[key] != request[key]:
                raise ValueError("capture split/family differs")
        if (
            label["prompt_tokens"] != request["planned_prompt_tokens"]
            or label["prompt_sha256"]
            != hashlib.sha256(request["prompt"].encode()).hexdigest()
        ):
            raise ValueError("captured prompt differs")
    return labels


def summarize(output, manifest, revision, tokenizer):
    labels = []
    completed = []
    failed = []
    for shard in manifest["shards"]:
        directory = output / "captures" / shard["name"]
        if not (directory / "summary.json").is_file():
            failed.append(
                dict(name=shard["name"], reason="no completed capture summary")
            )
            continue
        try:
            labels.extend(audit_shard(output, shard, revision))
            completed.append(shard["name"])
        except Exception as error:
            failed.append(dict(name=shard["name"], reason=repr(error)))
    if len({r["input_id"] for r in labels}) != len(labels):
        raise ValueError("duplicate capture IDs")
    histogram = Counter()
    long_groups = Counter()
    responses = []
    by_id = {row["id"]: row for row in rows(output / "inputs/recipe.jsonl")}
    for label in labels:
        n = label["output_tokens"]
        natural = label["natural_finish"]
        request = by_id[label["input_id"]]
        histogram[
            str(
                next(
                    (edge for edge in (2048, 4096, 8192, 12288) if n <= edge),
                    "overflow",
                )
            )
            if natural
            else "censored"
        ] += 1
        if natural and 4096 < n <= 8192:
            long_groups[request["split"]] += 1
        ids = label["output_token_ids"]
        if len(ids) != n:
            raise ValueError("saved output IDs differ")
        grams = list(zip(ids, ids[1:], ids[2:]))
        responses.append(
            dict(
                **label,
                task_type=request["task_type"],
                decoded_text=tokenizer.decode(ids, skip_special_tokens=True),
                repeated_trigram_fraction=1 - len(set(grams)) / len(grams)
                if grams
                else 0,
                repetition_metric_is_not_a_quality_verdict=True,
            )
        )
    (output / "responses.jsonl").write_bytes(payload(responses))
    summary = dict(
        completed_shards=completed,
        failed_shards=failed,
        requests=len(labels),
        natural_eos=sum(r["natural_finish"] for r in labels),
        censored=sum(not r["natural_finish"] for r in labels),
        planned_requests=manifest["rows"],
        all_requested_collected=len(labels) == manifest["rows"],
        actual_length_upper_edge_histogram=dict(histogram),
        natural_4k_8k_by_split=dict(long_groups),
        training_started=False,
        collection_role="frozen_long_supplement_four_independent_tp1",
    )
    write(output / "collection_summary.json", summary)
    return summary


def interrupt(*_):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    archive = Path(
        str(args.output)
        + (".resume_" + str(time.time_ns()) if args.resume else "")
        + ".tar.gz"
    )
    if (
        args.output.is_relative_to(ROOT)
        or (args.output.exists() and not args.resume)
        or archive.exists()
    ):
        parser.error(
            "output must be outside code; "
            "use --resume only for this existing collection"
        )
    if args.resume and not (args.output / "inputs/manifest.json").is_file():
        parser.error("resume needs an existing prepared collection")
    args.output.mkdir(parents=True, exist_ok=True)
    owned = []
    code = 1
    result = {}
    manifest = None
    signal.signal(signal.SIGTERM, interrupt)
    from tools.bridge_tp import run_phase9_cap0_calibration as common

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        if (
            revision != args.expected_revision
            or subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ):
            raise ValueError("code revision differs or worktree is dirty")
        print("[进度] 正在核对原模型权重及 tokenizer 文件哈希", flush=True)
        for name, digest in ASSET_SHA.items():
            if sha(args.model / name) != digest:
                raise ValueError("model/tokenizer differs: " + name)
        write(
            args.output / "model_identity.json",
            dict(
                model_path=str(args.model.resolve()),
                sha256=ASSET_SHA,
                predictor_weights_loaded=False,
            ),
        )
        hardware = runtime_inventory(read(PROFILE))
        write(args.output / "hardware.json", hardware)
        if [r["index"] for r in hardware["gpus"]] != ["0", "1", "2", "3"]:
            raise ValueError("requires physical GPU indices 0,1,2,3")
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip():
            raise ValueError("a GPU is busy; leave existing work untouched")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if args.resume:
            manifest = read(args.output / "inputs/manifest.json")
            pre = read(args.output / "preflight.json")
            if (
                pre["revision"] != revision
                or manifest["candidate_sha256"] != sha(DATA)
                or sha(args.output / "inputs/recipe.jsonl") != manifest["recipe_sha256"]
            ):
                raise ValueError("resume identity differs")
        else:
            manifest = prepare(tokenizer, args.output / "inputs")
            write(
                args.output / "preflight.json",
                dict(
                    revision=revision,
                    model=str(args.model.resolve()),
                    candidate_sha256=sha(DATA),
                    model_asset_sha256=ASSET_SHA,
                    capture_max_model_len=16384,
                    service_max_model_len=dict(tp1=8192, tp2=16384),
                    workers=4,
                    tensor_parallel_size_per_worker=1,
                ),
            )
        for shard in manifest["shards"]:
            directory = args.output / "captures" / shard["name"]
            if (directory / "summary.json").is_file():
                audit_shard(args.output, shard, revision)
                print(f"[进度] 复用已审计 GPU{shard['gpu_index']} 完整批次", flush=True)
                continue
            if directory.exists():
                if not directory.resolve().is_relative_to(args.output):
                    raise ValueError("capture path escapes output")
                directory.rename(
                    directory.with_name(
                        directory.name + f".incomplete_{time.time_ns()}"
                    )
                )
            command, env = worker_command(args.model, args.output, shard, revision)
            write(
                args.output / f"gpu{shard['gpu_index']}_argv.json",
                dict(command=command, CUDA_VISIBLE_DEVICES=env["CUDA_VISIBLE_DEVICES"]),
            )
            owned.append(
                common.start_process(
                    f"gpu{shard['gpu_index']}",
                    command,
                    env,
                    args.output / f"gpu{shard['gpu_index']}.log",
                )
            )
            print(
                f"[进度] GPU{shard['gpu_index']} 启动："
                f"{shard['rows']}条，TP1，独立特征目录",
                flush=True,
            )
            time.sleep(5)
        while any(item.process.poll() is None for item in owned):
            for item in owned:
                text = (
                    item.log_path.read_text(encoding="utf-8", errors="replace")
                    if item.log_path.exists()
                    else ""
                )
                tail = [
                    line
                    for line in text.splitlines()
                    if "[进度]" in line or "Error" in line
                ][-1:]
                print(
                    f"[进度] {item.name} exit={item.process.poll()} {' '.join(tail)}",
                    flush=True,
                )
            time.sleep(30)
        common.stop_processes(owned)
        if any(common.process_group_alive(item) for item in owned):
            raise RuntimeError("owned worker group remains alive")
        summary = summarize(args.output, manifest, revision, tokenizer)
        code = 0 if summary["all_requested_collected"] else 1
        result = dict(
            status="COMPLETED" if code == 0 else "PARTIAL_CAPTURE",
            summary=summary,
            worker_returncodes={item.name: item.process.returncode for item in owned},
        )
        if any(item.process.returncode != 0 for item in owned):
            code = 1
            result["status"] = "PARTIAL_CAPTURE"
    except KeyboardInterrupt:
        code = 130
        result = dict(status="INTERRUPTED")
    except Exception as error:
        code = 1
        result = dict(status="TECHNICAL_FAILURE", error=repr(error))
        print(f"[错误] {error!r}", flush=True)
    finally:
        try:
            common.stop_processes(owned)
            if any(common.process_group_alive(item) for item in owned):
                raise RuntimeError("owned workers still alive")
        except Exception as error:
            code = 1
            result["status"] = "TECHNICAL_FAILURE"
            result["cleanup_error"] = repr(error)
        if (args.output / "result.json").exists():
            shutil.copyfile(
                args.output / "result.json",
                args.output / f"result_previous_{time.time_ns()}.json",
            )
        result.update(runner_code=code, training_started=False)
        write(args.output / "result.json", result)
        receipt = pack(args.output, archive)
        Path(str(archive) + ".sha256").write_text(
            receipt["sha256"] + "  " + archive.name + "\n"
        )
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        print(f"[交回文件] {archive}", flush=True)
        print("[结果] " + result["status"] + "；保留全部输出，不自动训练。", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
