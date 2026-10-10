"""Build diagnostic requests; requested length is not a training label."""

import argparse
import hashlib
import json
from pathlib import Path


def inputs():
    groups = [
        ("engineering", 24),
        ("medium", 72),
        ("tail8192", 144),
        ("tail12288", 216),
    ]
    topics = [
        "distributed inference queueing, KV memory management and observability",
        "database indexing, concurrent transactions and recovery",
        "operating systems, process scheduling and memory allocation",
    ]
    rows = []
    for group, sections in groups:
        for i, topic in enumerate(topics):
            identity = f"predictor-long-diagnostic-20261010-{group}-{i}"
            if group == "engineering":
                content = (
                    f"Write a complete practical guide on {topic}. "
                    "Use exactly 24 numbered sections. Each section must contain "
                    "a concrete example and explanation, about 80 words. "
                    "Write the whole guide now, without asking to continue."
                )
            else:
                content = (
                    f"Create a detailed training handbook on {topic}. "
                    f"Write all {sections} numbered lessons in this response. "
                    "Each lesson should contain an explanation, worked example, "
                    "and check question, about 75 words. Vary the examples. "
                    "Do not provide only an outline or abbreviate later lessons; "
                    "complete the entire handbook without asking to continue."
                )
            rows.append(
                {
                    "id": identity,
                    "source_tree_id": identity,
                    "split": "validation",
                    "source": "engineering_diagnostic_not_formal_holdout",
                    "lang": "en",
                    "requested_group": group,
                    "requested_sections": sections,
                    "messages": [{"role": "user", "content": content}],
                }
            )
    return rows


def write(path, rows):
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True)
    rows = inputs()
    paired = rows[:2]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True
        )
        count = len(tokenizer.encode(prompt))
        if count + 15872 > 16384:
            raise ValueError("pilot prompt exceeds reserved 512-token budget")
        tokens.append({"id": row["id"], "prompt_tokens": count})
    manifest = {
        "paired_sha256": write(args.out_dir / "paired.jsonl", paired),
        "long_sha256": write(args.out_dir / "long12.jsonl", rows),
        "rows": len(rows),
        "tokenizer_preview": tokens,
        "role": "diagnostic; no formal predictor or benefit training",
        "output_cap": 15872,
        "max_model_len": 16384,
        "new_edges": [12288, 16384],
        "label_rule": "actual output IDs and natural EOS; caps are censored",
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
