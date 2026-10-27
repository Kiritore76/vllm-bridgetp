# SPDX-License-Identifier: Apache-2.0
"""Prepare a distinct Qwen3/4090D platform; never load old predictor/KV profiles."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "experiments/platforms/4090d_qwen3_8b_tp2.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_config(config, profile):
    if config.get("model_type") != profile["model_type"] or config.get(
        "architectures"
    ) != ["Qwen3ForCausalLM"]:
        raise ValueError("requires original dense Qwen3-8B architecture")
    for key, value in profile["model_geometry"].items():
        if config.get(key) != value:
            raise ValueError(f"Qwen3 geometry differs: {key}")
    if config.get("quantization_config"):
        raise ValueError("this BF16 baseline does not accept quantized weights")
    if config.get("use_sliding_window", False):
        raise ValueError("sliding-window KV is outside the migration contract")
    if config.get("max_position_embeddings", 0) < max(
        profile["service_max_model_len"].values()
    ) or config.get("rope_scaling"):
        raise ValueError("requires native context capacity without RoPE scaling")
    return (
        2
        * config["num_hidden_layers"]
        * config["num_key_value_heads"]
        * config["head_dim"]
        * 2
    )


def inspect_model(model, profile):
    model = Path(model).resolve()
    config = read(model / "config.json")
    bytes_per_token = validate_config(config, profile)
    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    }
    index = read(model / "model.safetensors.index.json")
    required.update(index["weight_map"].values())
    digests = {}
    for name in sorted(required):
        path = (model / name).resolve()
        if not path.is_relative_to(model) or not path.is_file():
            raise ValueError(f"model asset missing/escapes root: {name}")
        if path.stat().st_size == 0:
            raise ValueError(f"empty model asset: {name}")
        if name.endswith(".safetensors"):
            with path.open("rb") as stream:
                if stream.read(128).startswith(b"version https://git-lfs"):
                    raise ValueError("LFS pointer is not a model weight")
        digests[name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return dict(
        model_path=str(model),
        config=config,
        files=digests,
        kv_bytes_per_token_tp1=bytes_per_token,
        kv_bytes_per_token_per_rank_tp2=bytes_per_token // 2,
        predictor_status="NOT_TRAINED",
        performance_profiles="UNKNOWN",
    )


def validate_gpus(rows, profile):
    if len(rows) != profile["gpu_count"]:
        raise ValueError("requires four physical GPUs")
    for row in rows:
        family = re.sub(r"[^A-Z0-9]", "", row["name"].upper())
        if (
            not family.endswith("RTX4090D")
            or not profile["memory_mib_range"][0]
            <= row["memory_mib"]
            <= profile["memory_mib_range"][1]
        ):
            raise ValueError("requires RTX 4090 D 24GB, not A100/48GB variant")
    if len({r["uuid"] for r in rows}) != len(rows):
        raise ValueError("duplicate physical GPU UUID")


def runtime_inventory(profile):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, name, memory, uuid, driver = [v.strip() for v in line.split(",")]
        rows.append(
            dict(
                index=index, name=name, memory_mib=int(memory), uuid=uuid, driver=driver
            )
        )
    validate_gpus(rows, profile)
    return dict(
        gpus=rows,
        topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
    )


def download(model, provider, profile):
    model = Path(model)
    intent = model / "bridgetp_download_intent.json"
    if model.exists() and any(model.iterdir()) and not intent.is_file():
        raise ValueError("existing model directory: inspect it; do not overwrite")
    model.mkdir(parents=True, exist_ok=True)
    request = dict(
        provider=provider,
        model_id=profile["model_id"],
        revision=profile["huggingface_revision"]
        if provider == "huggingface"
        else "master",
        mutable_revision=provider != "huggingface",
    )
    if intent.exists() and read(intent) != request:
        raise ValueError("download source differs from existing intent")
    intent.write_text(json.dumps(request, indent=2) + "\n")
    if provider == "huggingface":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=request["model_id"],
            revision=request["revision"],
            local_dir=str(model),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    else:
        from modelscope.hub.snapshot_download import snapshot_download

        snapshot_download(
            request["model_id"],
            revision=request["revision"],
            local_dir=str(model),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    # ModelScope master is not claimed to be the HF commit. Actual bytes are
    # frozen below, including all weight shards. No silent provider fallback.
    return request


def prepare_inputs(model, output, profile):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    prompts = {}
    for name, content in (
        ("short_en", "Explain why a queue needs backpressure in five sentences."),
        ("short_zh", "用五句话解释为什么推理服务需要记录排队时间。"),
    ):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts[name] = dict(
            text=text,
            token_ids=tokenizer.encode(text, add_special_tokens=False),
            enable_thinking=False,
        )
    # Probe the declared context limit with real prefill, not just startup.
    marker = "BRIDGETP_CONTEXT_PLACEHOLDER"
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix, suffix = template.split(marker)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    body = tokenizer.encode(
        "A queue buffers work. Backpressure limits admission when resources are busy. "
        * 5000,
        add_special_tokens=False,
    )
    for tp, limit in profile["service_max_model_len"].items():
        size = limit - 32 - len(prefix_ids) - len(suffix_ids)
        if size <= 0 or len(body) < size:
            raise ValueError("cannot construct context-boundary probe")
        prompts[f"boundary_{tp}"] = dict(
            token_ids=prefix_ids + body[:size] + suffix_ids,
            enable_thinking=False,
            only_tp=int(tp.removeprefix("tp")),
            purpose="context_capacity_probe_not_quality_or_throughput",
        )
    (output / "smoke_prompts.json").write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    feature_rows = [
        dict(
            id=f"qwen3-feature-smoke-{i}",
            source_tree_id=f"qwen3-feature-smoke-tree-{i}",
            split="deployment_smoke",
            lang=lang,
            source="deployment_feature_smoke_not_training",
            messages=[dict(role="user", content=content)],
        )
        for i, (lang, content) in enumerate(
            (
                (
                    "en",
                    (
                        "Write twenty numbered examples of queues in computer systems, "
                        "with one sentence explaining each example."
                    ),
                ),
                ("zh", "列出二十个计算机系统中使用队列的例子，每个例子用一句话解释。"),
            )
        )
    ]
    (output / "feature_smoke.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in feature_rows),
        encoding="utf-8",
    )
    # Formal feature data must be generated separately from this deployment
    # smoke; old model token IDs/hidden states/labels are never copied.
    receipt = dict(
        profile_sha256=sha(PROFILE),
        model_config_sha256=sha(Path(model) / "config.json"),
        tokenizer_sha256=sha(Path(model) / "tokenizer.json"),
        thinking_mode=profile["thinking_mode"],
        training_inputs_prepared=False,
        prompts_sha256=sha(output / "smoke_prompts.json"),
    )
    (output / "inputs_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("download", "inspect", "inputs"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provider", choices=("modelscope", "huggingface"), default="modelscope"
    )
    parser.add_argument("--check-gpus", action="store_true")
    args = parser.parse_args(argv)
    profile = read(PROFILE)
    if args.output.exists():
        raise ValueError("receipt path must be new; preserve previous evidence")
    if args.phase == "inputs":
        result = prepare_inputs(args.model, args.output, profile)
    else:
        source = (
            download(args.model, args.provider, profile)
            if args.phase == "download"
            else None
        )
        result = inspect_model(args.model, profile)
        result.update(
            platform_id=profile["platform_id"],
            source=source,
            profile_sha256=sha(PROFILE),
        )
        if args.check_gpus:
            result["hardware"] = runtime_inventory(profile)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status="PREPARED",
                output=str(args.output),
                predictor="NOT_TRAINED",
                GPU_model_execution="NOT_TESTED",
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
