"""Prepare preregistered engineering requests, not fabricated length labels."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DOMAINS = (
    "distributed inference systems",
    "relational databases",
    "computer networks",
    "operating systems",
    "numerical computing",
    "software testing",
    "project planning",
    "technical writing",
    "environmental monitoring",
    "building design",
)
FOCUSES = (
    "capacity planning",
    "failure diagnosis",
    "measurement design",
    "resource allocation",
    "dependency management",
    "performance tradeoffs",
    "data integrity",
    "maintenance workflows",
    "comparison of alternatives",
    "incremental implementation",
)
GROUP_COUNTS = {
    "short": 120,
    "medium": 120,
    "upper_mid": 150,
    "tail8192": 120,
    "tail12288": 90,
}
UNITS = {"short": 12, "medium": 48, "upper_mid": 96, "tail8192": 144, "tail12288": 216}
CONTEXT = {"short": (0, 15872), "ctx2k": (2048, 14000), "ctx7k": (7168, 8704)}
STYLES = (
    "a case study collection, each case with a problem, analysis and solution",
    "a practical course, each lesson with explanation, example and exercise",
    "a comparative report, each section contrasting concrete alternatives",
    "a design review, each section with assumptions and failure scenarios",
    "a troubleshooting guide, each section with symptoms and diagnostic steps",
    "a worked example collection, showing reasoning and intermediate results",
)


def recipe():
    """Assign whole topic trees to fixed splits before generating outputs."""
    rng = random.Random(202610101)
    trees = list(range(100))
    rng.shuffle(trees)
    split = {
        t: ("train" if i < 70 else "validation" if i < 85 else "test")
        for i, t in enumerate(trees)
    }
    groups = [g for g, count in GROUP_COUNTS.items() for _ in range(count)]
    rng.shuffle(groups)
    rows = []
    for i, group in enumerate(groups):
        tree, variant = divmod(i, 6)
        topic = f"{DOMAINS[tree // 10]}: {FOCUSES[tree % 10]}"
        lang = "zh" if variant % 2 else "en"
        context = (
            "short"
            if group == "tail12288"
            else ("ctx2k" if i % 5 == 3 else "ctx7k" if i % 5 == 4 else "short")
        )
        if lang == "en":
            instruction = (
                f"Write {STYLES[variant]} about {topic}. "
                f"Provide {UNITS[group]} numbered units in English, each about "
                "75 words. Use varied concrete examples and explain tradeoffs. "
                "Complete the entire response now, without asking to continue. "
                "Do not replace the requested content with an outline."
            )
        else:
            instruction = (
                f"围绕{topic}，用中文撰写{STYLES[variant]}。"
                f"写出{UNITS[group]}个编号单元，每个单元约150个汉字。"
                "使用不同的具体例子，解释理由和取舍。一次写完全部内容，"
                "不要只列提纲，也不要询问是否继续。"
            )
        rows.append(
            {
                "id": f"predictor-formal-20261010-tree{tree:03d}-v{variant}",
                "source_tree_id": f"predictor-formal-20261010-tree{tree:03d}",
                "split": split[tree],
                "lang": lang,
                "source": "constructed_formal_predictor_not_natural_calibration",
                "topic": topic,
                "requested_group": group,
                "context_group": context,
                "instruction": instruction,
            }
        )
    return rows


def serialized(rows):
    return "".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows
    ).encode("utf-8")


def materialize(row, tokenizer):
    """Fit reference notes using the actual model tokenizer and chat template."""
    target, cap = CONTEXT[row["context_group"]]
    instruction = row["instruction"]

    def render(reference):
        content = (
            instruction
            if not reference
            else (
                "Reference notes / 参考记录:\n"
                + reference
                + "\nTask / 任务:\n"
                + instruction
            )
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt = render("")
    if target:
        notes = "\n".join(
            f"Record {n}: In {row['topic']}, scenario {n} has "
            f"{10 + n % 91} units of demand, {2 + n % 7} dependencies, "
            f"and a review interval of {1 + n % 13} days. "
            "Compare resource limits, measurement uncertainty, alternatives "
            "and recovery procedures before deciding on an implementation."
            for n in range(400)
        )
        ids = tokenizer.encode(notes, add_special_tokens=False)
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = render(tokenizer.decode(ids[:mid]))
            if len(tokenizer.encode(candidate)) <= target:
                lo = mid
            else:
                hi = mid - 1
        prompt = render(tokenizer.decode(ids[:lo]))
    count = len(tokenizer.encode(prompt))
    if count + cap > 16384 or (target and abs(count - target) > 16):
        raise ValueError(f"invalid prompt reservation: {row['id']} {count}+{cap}")
    result = {k: v for k, v in row.items() if k != "instruction"}
    result.update(prompt=prompt, planned_prompt_tokens=count, planned_output_cap=cap)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError("input directory must be new")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = recipe()
    raw = serialized(rows)
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "recipe.jsonl").write_bytes(raw)
    pools = defaultdict(list)
    for n, row in enumerate(rows, 1):
        pools[row["context_group"]].append(materialize(row, tokenizer))
        if n % 60 == 0:
            print(f"[进度] 输入准备 {n}/600", flush=True)
    shards = []
    # Interleave context groups so early collection covers all three contexts.
    for offset in range(0, max(map(len, pools.values())), 20):
        for context, pool in pools.items():
            subset = pool[offset : offset + 20]
            if not subset:
                continue
            name = f"shard{len(shards):02d}_{context}"
            payload = serialized(subset)
            path = args.out_dir / f"{name}.jsonl"
            path.write_bytes(payload)
            shards.append(
                {
                    "name": name,
                    "input": path.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "rows": len(subset),
                    "context": context,
                    "max_tokens": CONTEXT[context][1],
                    "split_counts": dict(Counter(r["split"] for r in subset)),
                }
            )
    manifest = {
        "rows": 600,
        "trees": 100,
        "recipe_sha256": hashlib.sha256(raw).hexdigest(),
        "split_rows": dict(Counter(r["split"] for r in rows)),
        "requested_group_counts": dict(Counter(r["requested_group"] for r in rows)),
        "context_counts": dict(Counter(r["context_group"] for r in rows)),
        "label_rule": "actual natural EOS; requested groups are not labels",
        "shards": shards,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("[进度] 600条输入及预注册划分准备完成", flush=True)


if __name__ == "__main__":
    main()
