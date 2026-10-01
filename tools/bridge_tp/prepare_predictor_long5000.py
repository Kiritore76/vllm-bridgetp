"""Freeze independent OASST1 natural and explicitly long-form requests."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from prepare_oasst1_predictor_inputs import (
    SOURCE_SHA256,
    SPLITS,
    choose_pilot,
    choose_staged_pilot,
    load_roots,
    sha256_file,
    write_jsonl,
)

TASK_WORDS = (
    "explain",
    "write",
    "how",
    "code",
    "describe",
    "implement",
    "design",
    "tutorial",
    "guide",
    "develop",
    "compare",
    "analy",
    "story",
    "essay",
)
TARGET_WORDS = (1000, 1800, 2600)


def ranked(rows: list[dict], salt: str) -> list[dict]:
    return sorted(
        rows, key=lambda r: hashlib.sha256((salt + r["id"]).encode()).digest()
    )


def choose_long5000(roots: list[dict]) -> tuple[list[dict], set[str]]:
    """Exclude all legacy/probe trees before selecting new holdouts."""
    legacy = {r["source_tree_id"] for r in choose_pilot(roots, 800, 200)}
    staged = choose_staged_pilot(
        [r for r in roots if r["source_tree_id"] not in legacy], 2000
    )
    excluded = legacy | {r["source_tree_id"] for r in staged}
    pool = [r for r in roots if r["source_tree_id"] not in excluded]
    selected = []
    used = set()
    # 4000 untouched requests: 3900 English, 100 Chinese.
    for language, quotas in (("en", (3120, 390, 390)), ("zh", (80, 10, 10))):
        for split, count in zip(SPLITS, quotas):
            candidates = ranked(
                [r for r in pool if r["lang"] == language and r["split"] == split],
                "natural-long5000-v1:",
            )
            if len(candidates) < count:
                raise ValueError(f"insufficient new natural {language}/{split}")
            for row in candidates[:count]:
                selected.append({**row, "workload_group": "natural"})
                used.add(row["source_tree_id"])
    # Separate roots for 1000 long-form requests; no natural/augmented twins.
    for split, count in zip(SPLITS, (800, 100, 100)):
        candidates = ranked(
            [
                r
                for r in pool
                if r["lang"] == "en"
                and r["split"] == split
                and r["source_tree_id"] not in used
                and any(w in r["messages"][0]["content"].casefold() for w in TASK_WORDS)
            ],
            "augmented-long5000-v1:",
        )
        if len(candidates) < count:
            raise ValueError(f"insufficient long-form task roots for {split}")
        for index, row in enumerate(candidates[:count]):
            words = TARGET_WORDS[index % len(TARGET_WORDS)]
            original = row["messages"][0]["content"]
            instruction = (
                "\n\nPlease give a substantial, self-contained response of about "
                f"{words} words to the request above. For an explanation or how-to "
                "task, use organized sections, detailed reasoning, worked examples, "
                "and practical limitations. For a coding task, include a complete "
                "implementation, tests, and explanations. For a creative task, "
                "develop a complete long-form piece. Keep every section relevant; "
                "do not repeat passages or add padding just to reach the length. "
                "Finish the response naturally when it is complete."
            )
            selected.append(
                {
                    **row,
                    "messages": [{"role": "user", "content": original + instruction}],
                    "workload_group": "long_form",
                    "augmentation": "long-form-v1",
                    "target_words": words,
                    "original_prompt_sha256": hashlib.sha256(
                        original.encode()
                    ).hexdigest(),
                }
            )
            used.add(row["source_tree_id"])
    if len(selected) != 5000 or len(used) != 5000 or used & excluded:
        raise ValueError("request/tree overlap in the frozen long5000 input")
    return ranked(selected, "capture-order-long5000-v1:"), excluded


def prepare(source: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("input destination must be new or empty")
    rows, excluded = choose_long5000(load_roots(source, 20, 1500))
    out.mkdir(parents=True, exist_ok=True)
    full = out / "requests.jsonl"
    write_jsonl(full, rows)
    shards = []
    for number, start in enumerate(range(0, len(rows), 500)):
        path = out / f"shard_{number:03d}.jsonl"
        write_jsonl(path, rows[start : start + 500])
        shards.append(
            {"filename": path.name, "sha256": sha256_file(path), "count": 500}
        )
    manifest = {
        "format_version": 1,
        "dataset": "oasst1-long5000-v1",
        "source_sha256": SOURCE_SHA256,
        "excluded_legacy_and_staged_trees": len(excluded),
        "split_rule": "original oasst1-split-v1 tree hash; new trees only",
        "input_sha256": sha256_file(full),
        "requests": len(rows),
        "counts": dict(
            sorted(
                Counter(
                    f"{r['split']}/{r['workload_group']}/{r['lang']}" for r in rows
                ).items()
            )
        ),
        "long_form_target_words": list(TARGET_WORDS),
        "length_note": "Requested words are input metadata, never ground-truth labels; "
        "labels come from the local model's actual token/EOS output. "
        "Natural and augmented workloads must be reported separately.",
        "shards": shards,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
