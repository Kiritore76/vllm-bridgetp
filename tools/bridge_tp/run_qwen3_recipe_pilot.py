# SPDX-License-Identifier: Apache-2.0
"""Budget and capture one balanced 12-request Qwen3 single-GPU recipe pilot."""

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp.prepare_4090d_qwen3 import (  # noqa: E402
    PROFILE,
    read,
    runtime_inventory,
    sha,
    validate_config,
)
from tools.bridge_tp.result_archive import pack  # noqa: E402

DATA = ROOT / "tools/bridge_tp/predictor_data/qwen3_recipe_pilot48_20261027.jsonl"
MANIFEST = DATA.with_suffix(".manifest.json")
DIVERSE_DATA = (
    ROOT / "tools/bridge_tp/predictor_data/qwen3_diverse_long12_20261027.jsonl"
)


def write(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def balanced_batches(rows):
    """Partition all candidates once, keeping 6/3/2/1 sources in each batch."""
    if len(rows) != 48 or len({r["source_tree_id"] for r in rows}) != 48:
        raise ValueError("48 unique source families required")
    if len({r["id"] for r in rows}) != 48 or any(r["split"] != "pilot" for r in rows):
        raise ValueError("unique pilot-only identities required")
    pools = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        pools[row["candidate_group"], row["lang"]].append(row)
    batches = []
    for index in range(4):
        batch = []
        for group, en, zh in (
            ("oasst_natural", 3, 3),
            ("longwriter_raw", 2 if index % 2 == 0 else 1, 1 if index % 2 == 0 else 2),
            ("oasst_augmented", 1, 1),
            ("controlled", 0 if index % 2 == 0 else 1, 1 if index % 2 == 0 else 0),
        ):
            for lang, count in (("en", en), ("zh", zh)):
                pool = pools[group, lang]
                if len(pool) < count:
                    raise ValueError("candidate source/language quota differs")
                batch += pool[:count]
                del pool[:count]
        random.Random(202610270 + index).shuffle(batch)
        if len(batch) != 12:
            raise ValueError("pilot batch must contain 12 requests")
        batches.append(batch)
    if any(pools.values()):
        raise ValueError("unassigned candidates")
    return batches


def load_batch(profile, batch_index):
    """Keep historical inputs frozen while adding a separately identified pilot."""
    if profile not in ("recipe48", "diverse12"):
        raise ValueError("unknown candidate profile")
    source = DATA if profile == "recipe48" else DIVERSE_DATA
    manifest_path = source.with_suffix(".manifest.json")
    if sha(source) != read(manifest_path)["input_sha256"]:
        raise ValueError("candidate SHA differs")
    with source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if profile == "recipe48":
        if batch_index not in range(4):
            raise ValueError("recipe48 batch index outside 0..3")
        batch = balanced_batches(rows)[batch_index]
    else:
        if batch_index != 0:
            raise ValueError("diverse12 has one batch; batch-index must be 0")
        if (
            len(rows) != 12
            or len({r["id"] for r in rows}) != 12
            or len({r["source_tree_id"] for r in rows}) != 12
            or any(r["split"] != "pilot" for r in rows)
        ):
            raise ValueError("diverse12 requires 12 unique pilot families")
        if Counter(r["candidate_group"] for r in rows) != dict(
            coding_long=6, technical_tutorial=2, system_design=2, document_analysis=2
        ) or Counter(r["lang"] for r in rows) != {"en": 6, "zh": 6}:
            raise ValueError("diverse12 task/language quotas differ")
        batch = rows
    return batch, source, manifest_path


def render_budget(rows, tokenizer, max_model_len=16384):
    """Use the new tokenizer once and reserve one common valid output cap."""
    rendered = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = len(tokenizer.encode(text))
        if not 0 < prompt_tokens < max_model_len:
            raise ValueError(f"prompt outside context budget: {row['id']}")
        # Capture requires exactly one of messages/prompt per request.
        item = {k: v for k, v in row.items() if k != "messages"}
        item.update(
            prompt=text,
            thinking_mode="disabled",
            planned_prompt_tokens=prompt_tokens,
            tokenizer_budget_status="VALIDATED_QWEN3_NON_THINKING",
        )
        rendered.append(item)
    cap = min(15872, min(max_model_len - r["planned_prompt_tokens"] for r in rendered))
    if cap < 8192:
        raise ValueError("pilot output headroom below 8K; review long input separately")
    for row in rendered:
        row["planned_output_cap"] = cap
    return rendered, cap


def summarize(rows, labels, tokenizer):
    """Keep short, capped and repetitive results; do not fabricate exact tails."""
    by_id = {r["id"]: r for r in rows}
    if len(labels) != len(by_id) or {r["input_id"] for r in labels} != set(by_id):
        raise ValueError("capture label identities differ")
    responses, histogram, sources = [], Counter(), defaultdict(list)
    for label in labels:
        source = by_id[label["input_id"]]
        ids = label["output_token_ids"]
        if len(ids) != label["output_tokens"]:
            raise ValueError("saved token count differs")
        if label["prompt_tokens"] != source["planned_prompt_tokens"]:
            raise ValueError("capture prompt length differs from new tokenizer budget")
        natural = label["natural_finish"]
        length = len(ids)
        bucket = (
            next(
                (
                    str(edge)
                    for edge in (128, 512, 1024, 2048, 4096, 8192, 12288, 16384)
                    if length <= edge
                ),
                "overflow",
            )
            if natural
            else "censored"
        )
        histogram[bucket] += 1
        sources[source["candidate_group"]].append(
            dict(output_tokens=length, natural_finish=natural)
        )
        trigrams = [tuple(ids[i : i + 3]) for i in range(max(0, length - 2))]
        responses.append(
            dict(
                **label,
                candidate_group=source["candidate_group"],
                decoded_text=tokenizer.decode(ids, skip_special_tokens=False),
                repeated_trigram_fraction=(1 - len(set(trigrams)) / len(trigrams))
                if trigrams
                else 0.0,
                repetition_metric_is_not_a_quality_verdict=True,
            )
        )
    return dict(
        requests=len(labels),
        natural_eos=sum(r["natural_finish"] for r in labels),
        censored=sum(not r["natural_finish"] for r in labels),
        natural_length_upper_edge_histogram=dict(histogram),
        by_candidate_source=dict(sources),
        training_started=False,
        formal_holdout_eligible=False,
    ), responses


def log_tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 4096))
        return "\n".join(
            stream.read().decode("utf-8", errors="replace").splitlines()[-4:]
        )


def interrupted(*_):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--installation-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-index", type=int, choices=range(4), default=0)
    parser.add_argument(
        "--profile", choices=("recipe48", "diverse12"), default="recipe48"
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="capture two short probes before repeating the 12 requests",
    )
    args = parser.parse_args(argv)
    if args.probe_only and args.profile != "recipe48":
        parser.error("probe-only belongs to the historical recipe48 profile")
    args.output = args.output.resolve()
    archive_path = Path(str(args.output) + ".tar.gz")
    if (
        args.output.is_relative_to(ROOT)
        or args.output.exists()
        or archive_path.exists()
    ):
        parser.error("output must be new and outside the code directory")
    args.output.mkdir(parents=True)
    process, common, result, code = None, None, {}, 0
    signal.signal(signal.SIGTERM, interrupted)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip():
            raise ValueError("worktree must be clean")
        profile = read(PROFILE)
        validate_config(read(args.model / "config.json"), profile)
        profile["gpu_count"] = 1
        hardware = runtime_inventory(profile)
        write(args.output / "hardware.json", hardware)
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip():
            raise RuntimeError("GPU computation already exists; leave it untouched")
        installation = read(args.installation_record / "model_identity.json")
        if Path(installation["model_path"]).resolve() != args.model.resolve():
            raise ValueError("model path differs from verified installation")
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ):
            if sha(args.model / name) != installation["files"][name]["sha256"]:
                raise ValueError(f"model/tokenizer asset differs: {name}")
        shutil.copyfile(
            args.installation_record / "model_identity.json",
            args.output / "model_identity.json",
        )
        batch, candidate_path, candidate_manifest = load_batch(
            args.profile, args.batch_index
        )
        if args.probe_only:
            batch = [
                next(r for r in batch if r["candidate_group"] == "controlled"),
                next(
                    r
                    for r in batch
                    if r["candidate_group"] == "oasst_natural" and r["lang"] == "en"
                ),
            ]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        rows, cap = render_budget(batch, tokenizer)
        if args.probe_only:
            cap = 128
            for row in rows:
                row["planned_output_cap"] = cap
        input_path = args.output / "requests.jsonl"
        input_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        shutil.copyfile(candidate_manifest, args.output / "candidate_sources.json")
        write(
            args.output / "preflight.json",
            dict(
                revision=revision,
                batch_index=args.batch_index,
                candidate_profile=args.profile,
                requests=len(rows),
                probe_only=args.probe_only,
                model_runner="V1",
                groups=dict(Counter(r["candidate_group"] for r in rows)),
                languages=dict(Counter(r["lang"] for r in rows)),
                thinking_mode="disabled",
                max_model_len=16384,
                max_tokens=cap,
                prompt_tokens=[r["planned_prompt_tokens"] for r in rows],
                candidate_sha256=sha(candidate_path),
                input_sha256=sha(input_path),
                installation_record=str(args.installation_record),
                formal_training=False,
            ),
        )
        command = [
            sys.executable,
            "-B",
            "tools/bridge_tp/run_predictor_capture.py",
            "--input",
            str(input_path),
            "--model",
            str(args.model),
            "--out-dir",
            str(args.output / "capture"),
            "--expected-revision",
            revision,
            "--expected-input-sha256",
            sha(input_path),
            "--expected-gpu-name",
            hardware["gpus"][0]["name"],
            "--expected-gpu-count",
            "1",
            "--feature-layer",
            "decoder:31",
            "--interval",
            "20",
            "--max-tokens",
            str(cap),
            "--max-model-len",
            "16384",
            "--max-num-batched-tokens",
            "2048",
            "--gpu-memory-utilization",
            "0.85",
            "--thinking-mode",
            "disabled",
            "--save-token-ids",
        ]
        write(args.output / "capture_argv.json", command)
        env = {k: v for k, v in os.environ.items() if not k.startswith("BRIDGETP_")}
        env.update(
            CUDA_VISIBLE_DEVICES="0",
            VLLM_USE_V2_MODEL_RUNNER="0",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="1",
        )
        from tools.bridge_tp import run_phase9_cap0_calibration as common

        print(f"[进度] 单卡{len(rows)}条；总上下文16384，实际输出上限{cap}", flush=True)
        print(f"[进度] 采集日志：{args.output / 'capture.log'}", flush=True)
        process = common.start_process(
            "qwen3-recipe-pilot", command, env, args.output / "capture.log"
        )
        last = 0.0
        while process.process.poll() is None:
            if time.monotonic() - last > 30:
                print(log_tail(args.output / "capture.log"), flush=True)
                last = time.monotonic()
            time.sleep(1)
        print(log_tail(args.output / "capture.log"), flush=True)
        if process.process.returncode:
            raise RuntimeError(f"capture exited {process.process.returncode}")
        summary = read(args.output / "capture/summary.json")
        if (
            summary["hidden_size"] != 4096
            or summary["phases"].get("PREFILL_COMPLETE") != len(rows)
            or not summary["phases"].get("DECODE", 0)
        ):
            raise ValueError("Qwen3 feature width/prefill coverage differs")
        with (args.output / "capture/labels.jsonl").open(encoding="utf-8") as stream:
            labels = [json.loads(line) for line in stream if line.strip()]
        coverage, responses = summarize(rows, labels, tokenizer)
        write(args.output / "coverage.json", coverage)
        (args.output / "responses.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in responses),
            encoding="utf-8",
        )
        result = dict(status="COMPLETED", training_started=False, coverage=coverage)
    except KeyboardInterrupt:
        result, code = dict(status="INTERRUPTED"), 130
    except Exception as error:
        result, code = dict(status="TECHNICAL_FAILURE", error=repr(error)), 1
        print(f"[错误] {error!r}", flush=True)
    finally:
        if process is not None:
            try:
                common.stop_processes([process])
                if common.process_group_alive(process):
                    raise RuntimeError("owned process group still alive")
            except Exception as error:
                code, result["status"] = 1, "TECHNICAL_FAILURE"
                result["cleanup_error"] = repr(error)
        result["runner_code"] = code
        write(args.output / "result.json", result)
        receipt = pack(args.output, archive_path)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        print(f"[交回文件] {receipt['archive']}", flush=True)
        print(f"[SHA256] {receipt['sha256']}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
