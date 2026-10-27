# SPDX-License-Identifier: Apache-2.0
"""Render frozen public instructions using the installed Qwen3 chat tokenizer."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "tools/bridge_tp/predictor_data/qwen3_formal16k2000_20261027.jsonl"
CONFIG_SHA = "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30"


def serialized(rows):
    return (
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    ).encode()


def materialize(row, tokenizer):
    """Preserve the complete prompt, with no padding or reference answers."""
    prompt = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    count = len(tokenizer.encode(prompt, add_special_tokens=True))
    for ceiling in (512, 2048, 4096, 8192):
        if count <= ceiling:
            result = dict(
                row,
                prompt=prompt,
                planned_prompt_tokens=count,
                max_tokens=16384 - ceiling,
                context_group=f"input_le_{ceiling}",
            )
            result.pop("messages")
            return result
    return None


def prepare(rows, tokenizer, output, source_sha):
    """Freeze eligibility and caps before generation, keeping source tree splits."""
    ids, trees, prompts = set(), {}, set()
    pools = defaultdict(list)
    excluded = []
    for row in rows:
        if row["id"] in ids or row["split"] not in ("train", "validation", "test"):
            raise ValueError("duplicate ID or invalid split")
        ids.add(row["id"])
        tree = row["source_tree_id"]
        if trees.setdefault(tree, row["split"]) != row["split"]:
            raise ValueError("source tree crosses splits")
        key = hashlib.sha256(serialized(row["messages"])).hexdigest()
        if key in prompts:
            raise ValueError("duplicate prompt")
        prompts.add(key)
        rendered = materialize(row, tokenizer)
        if rendered is None:
            excluded.append(
                dict(id=row["id"], split=row["split"], reason="prompt_exceeds_8192")
            )
        else:
            pools[rendered["context_group"]].append(rendered)
    accepted = [row for pool in pools.values() for row in pool]
    if not accepted:
        raise ValueError("no eligible requests")
    # Validate everything before creating the output directory.
    output.mkdir(parents=True, exist_ok=False)
    recipe = serialized(accepted)
    (output / "recipe.jsonl").write_bytes(recipe)
    shards = []
    for offset in range(0, max(map(len, pools.values())), 50):
        for context, pool in sorted(pools.items()):
            subset = pool[offset : offset + 50]
            if not subset:
                continue
            name = f"shard{len(shards):03d}_{context}"
            payload = serialized(subset)
            (output / (name + ".jsonl")).write_bytes(payload)
            shards.append(
                dict(
                    name=name,
                    input=name + ".jsonl",
                    rows=len(subset),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    max_tokens=subset[0]["max_tokens"],
                    context=context,
                )
            )
    manifest = dict(
        profile="qwen3_formal16k2000",
        thinking_mode="disabled",
        rows=len(accepted),
        frozen_source_rows=len(rows),
        frozen_source_sha256=source_sha,
        recipe_sha256=hashlib.sha256(recipe).hexdigest(),
        split_rows=dict(Counter(row["split"] for row in accepted)),
        source_counts=dict(Counter(row["source"] for row in accepted)),
        language_counts=dict(Counter(row["lang"] for row in accepted)),
        context_counts=dict(Counter(row["context_group"] for row in accepted)),
        excluded_before_generation=excluded,
        shards=shards,
        planned_training_buckets=dict(
            profile="nonuniform16k91",
            categories=91,
            overflow_category=False,
            maximum_tokens=16384,
        ),
        label_rule="Qwen3 natural EOS; capped requests diagnostic only",
        coverage_status="UNMEASURED_UNTIL_LOCAL_MODEL_GENERATION",
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--installation-record", type=Path, required=True)
    args = parser.parse_args()
    raw = RECIPE.read_bytes()
    frozen = json.loads(
        RECIPE.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    if hashlib.sha256(raw).hexdigest() != frozen["sha256"]:
        parser.error("frozen instruction SHA differs")
    if (
        hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest()
        != CONFIG_SHA
    ):
        parser.error("Qwen3 model config differs from installation receipt")
    receipt = json.loads(
        (args.installation_record / "model_identity.json").read_text(encoding="utf-8")
    )
    if Path(receipt["model_path"]).resolve() != args.model.resolve():
        parser.error("model path differs from verified installation")
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        digest = hashlib.sha256((args.model / name).read_bytes()).hexdigest()
        if digest != receipt["files"][name]["sha256"]:
            parser.error(f"model/tokenizer asset differs: {name}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # Physical JSONL lines: Unicode line separators may occur inside prompt strings.
    with RECIPE.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    manifest = prepare(rows, tokenizer, args.out_dir, frozen["sha256"])
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "shards"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[进度] 完成输入准备：{manifest['rows']}条、{len(manifest['shards'])}批")


if __name__ == "__main__":
    main()
