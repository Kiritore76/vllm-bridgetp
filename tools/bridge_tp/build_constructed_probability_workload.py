# SPDX-License-Identifier: Apache-2.0
"""Build controlled, natural-EOS tasks for engineering benefit fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

FAMILIES = ("inventory", "workflow", "capacity")
SPLITS = ("engineering_train", "engineering_validation", "engineering_test")


def record_text(family: str, rng: random.Random, index: int) -> str:
    values = [rng.randint(10, 900) for _ in range(5)]
    a, b, c, d, e = values
    if family == "inventory":
        return (
            f"Item {index:04d}: demand {a} units/week; stock {b}; lead time "
            f"{c % 20 + 1} days; unit cost {d}; holding budget {e}. "
            "Assess reorder timing, uncertainty, and a workable alternative.\n"
        )
    if family == "workflow":
        return (
            f"Project {index:04d}: preparation {a % 30 + 1} hours; delivery "
            f"{b % 30 + 1} hours; review {c % 10 + 1} hours; deadline "
            f"{d % 90 + 30} hours; staff budget {e % 10 + 1}. "
            "Explain dependencies, bottlenecks, and a fallback schedule.\n"
        )
    return (
        f"Service {index:04d}: arrivals {a}/minute; processing {b} ms; "
        f"workers {c % 8 + 1}; queue budget {d}; burst multiplier "
        f"{e % 5 + 1}. Explain capacity, delays, and scaling alternatives.\n"
    )


def make_rows(
    *,
    seed: int,
    split: str,
    anchor_prompt_tokens: int,
    background_prompt_tokens: int,
    anchor_sections: int,
    background_sections: int,
    backgrounds: int,
    target_prompt_tokens: int | None = None,
) -> list[dict]:
    if split not in SPLITS or backgrounds < 94:
        raise ValueError("use an engineering split and at least 94 backgrounds")
    if min(anchor_prompt_tokens, background_prompt_tokens) < 1024:
        raise ValueError("constructed context targets must be at least 1024 tokens")
    if target_prompt_tokens is not None and target_prompt_tokens < 1024:
        raise ValueError("target context target must be at least 1024 tokens")
    if not (4 <= background_sections <= 16 and 8 <= anchor_sections <= 64):
        raise ValueError("background sections must be 4..16; anchor sections 8..64")
    rows = []
    for index in range(6 + backgrounds):
        role = "anchor" if index < 6 else "background"
        family = FAMILIES[index % len(FAMILIES)]
        row_seed = f"constructed-v1:{split}:{seed}:{index}"
        sections = anchor_sections if role == "anchor" else background_sections
        length = anchor_prompt_tokens if role == "anchor" else background_prompt_tokens
        recipe = {
            "version": 1,
            "family": family,
            "seed": row_seed,
            "requested_prompt_tokens": length,
            "response_sections": sections,
            "role": role,
            "output_length_is_not_guaranteed": True,
        }
        if role == "background" and target_prompt_tokens is not None:
            recipe["target_prompt_tokens"] = target_prompt_tokens
        rows.append(
            {
                "id": row_seed,
                "source_tree_id": row_seed,
                "split": split,
                "controller_split": split,
                "workload_origin": "constructed",
                "workload_group": f"constructed_{role}",
                "template_family": family,
                "construction": recipe,
                "augmentation": {
                    "kind": "structured_unique_records",
                    "recipe": recipe,
                    "natural_eos": True,
                    "ignore_eos": False,
                },
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Analyze {sections} {family} cases "
                            "from controlled records. "
                            f"Experiment record set: {row_seed}."
                        ),
                    }
                ],
            }
        )
    return rows


def constructed_prompt_tokens(
    tokenizer, row: dict, pool: str | None = None
) -> list[int]:
    """Fit unique structured records between intact chat and task boundaries."""
    recipe = row["construction"]
    family = recipe["family"]
    sections = recipe["response_sections"]
    length = recipe["requested_prompt_tokens"]
    if pool == "target":
        length = recipe.get("target_prompt_tokens", length)
    rng = random.Random(recipe["seed"])
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": "<CONSTRUCTED_RECORDS>"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if template.count("<CONSTRUCTED_RECORDS>") != 1:
        raise ValueError("chat template lost the constructed context marker")
    before, after = template.split("<CONSTRUCTED_RECORDS>")
    prefix = tokenizer.encode(
        before + "Controlled case records:\n", add_special_tokens=False
    )
    task = (
        f"\nAnalyze records 0001 through {sections:04d}, in that order. "
        "Give each its own numbered section with a recommendation, numerical "
        "reasoning, assumptions, and an alternative. Aim for 70-100 words per "
        "section. Finish with an overall comparison. Additional records are "
        "reference material; the last reference record may be incomplete. "
        "Conclude naturally when the requested analysis is complete."
    )
    suffix = tokenizer.encode(task + after, add_special_tokens=False)
    budget = length - len(prefix) - len(suffix)
    required_text = "".join(record_text(family, rng, i) for i in range(1, sections + 1))
    body = tokenizer.encode(required_text, add_special_tokens=False)
    if len(body) > budget:
        raise ValueError("context target truncates a record needed by the output task")
    index = sections + 1
    while len(body) < budget:
        text = "".join(record_text(family, rng, i) for i in range(index, index + 32))
        body += tokenizer.encode(text, add_special_tokens=False)
        index += 32
    result = prefix + body[:budget] + suffix
    if len(result) != length:
        raise AssertionError("constructed token budget mismatch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=SPLITS, default="engineering_train")
    parser.add_argument("--anchor-prompt-tokens", type=int, default=8192)
    parser.add_argument("--background-prompt-tokens", type=int, default=2048)
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        help="Independent target background context budget",
    )
    parser.add_argument("--anchor-sections", type=int, default=24)
    parser.add_argument("--background-sections", type=int, default=6)
    parser.add_argument("--backgrounds", type=int, default=640)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = make_rows(
        seed=args.seed,
        split=args.split,
        anchor_prompt_tokens=args.anchor_prompt_tokens,
        background_prompt_tokens=args.background_prompt_tokens,
        anchor_sections=args.anchor_sections,
        background_sections=args.background_sections,
        backgrounds=args.backgrounds,
        target_prompt_tokens=args.target_prompt_tokens,
    )
    data = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
    print(
        json.dumps(
            {
                "path": str(args.out.resolve()),
                "rows": len(rows),
                "split": args.split,
                "sha256": hashlib.sha256(data.encode()).hexdigest(),
                "tokenization": "actual model tokenizer at collector setup",
                "purpose": (
                    "engineering conditional benefit; not natural-risk calibration"
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
