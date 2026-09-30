"""Prepare deterministic OASST1 root prompts for length-predictor capture.

This reads the original OpenAssistant messages archive. Assistant replies are
never used as length labels: the capture runner obtains labels from the local
model's own continuations.
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCE_SHA256 = "621ccd86a6ef320ca4e24c137121bd4b39bcc7a0df839f0897fcc965ef2076ed"
SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for_tree(tree_id: str) -> str:
    """Keep every message from a conversation tree in the same split."""
    bucket = (
        int.from_bytes(
            hashlib.sha256(("oasst1-split-v1:" + tree_id).encode()).digest()[:8],
            "big",
        )
        % 10
    )
    return SPLITS[0] if bucket < 8 else SPLITS[1] if bucket == 8 else SPLITS[2]


def load_roots(source: Path, min_chars: int, max_chars: int) -> list[dict]:
    """Select standalone, reviewed root user turns from the pinned archive."""
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("OASST1 source SHA256 differs from the pinned release")
    roots = []
    seen_text = set()
    seen_id = set()
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if (
                row.get("role") != "prompter"
                or row.get("parent_id") is not None
                or row.get("deleted")
                or row.get("synthetic")
            ):
                continue
            language = row.get("lang")
            if language not in ("en", "zh"):
                continue
            message_id = row.get("message_id")
            tree_id = row.get("message_tree_id")
            prompt = row.get("text")
            if not all(isinstance(x, str) and x for x in (message_id, tree_id, prompt)):
                raise ValueError(f"invalid root message at line {line_number}")
            prompt = prompt.strip()
            if not min_chars <= len(prompt) <= max_chars:
                continue
            text_key = hashlib.sha256(prompt.casefold().encode()).hexdigest()
            if text_key in seen_text:
                continue
            if message_id in seen_id:
                raise ValueError(f"duplicate message_id: {message_id}")
            seen_text.add(text_key)
            seen_id.add(message_id)
            roots.append(
                {
                    "id": "oasst1:" + message_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "source": "OpenAssistant/oasst1",
                    "source_message_id": message_id,
                    "source_tree_id": tree_id,
                    "lang": language,
                    "split": split_for_tree(tree_id),
                }
            )
    return sorted(roots, key=lambda row: row["id"])


def choose_pilot(roots: list[dict], en_count: int, zh_count: int) -> list[dict]:
    """Select a stable language-stratified pilot across all three splits."""
    selected = []
    for language, count in (("en", en_count), ("zh", zh_count)):
        quotas = (count * 8 // 10, count // 10)
        quotas += (count - sum(quotas),)
        for split, quota in zip(SPLITS, quotas):
            ranked = sorted(
                (
                    row
                    for row in roots
                    if row["lang"] == language and row["split"] == split
                ),
                key=lambda row: hashlib.sha256(
                    ("oasst1-pilot-v1:" + row["id"]).encode()
                ).digest(),
            )
            if len(ranked) < quota:
                raise ValueError(
                    f"only {len(ranked)} eligible {language}/{split} prompts"
                )
            selected.extend(ranked[:quota])
    return sorted(selected, key=lambda row: row["id"])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_staged_pilot(roots: list[dict], total: int) -> list[dict]:
    """Keep the first 2000 requests/batches identical when expanding to 10000."""
    if total not in (2000, 10000):
        raise ValueError("staged total must be 2000 or 10000")
    initial = choose_pilot(roots, 1800, 200)
    if total == 2000:
        return initial
    initial_ids = {row["id"] for row in initial}
    full = choose_pilot(roots, 9800, 200)
    if not initial_ids <= {row["id"] for row in full}:
        raise ValueError("initial stage is not contained in the full input")
    return initial + [row for row in full if row["id"] not in initial_ids]


def choose_layer_probe(roots: list[dict]) -> list[dict]:
    """Use future training trees for an independent 300-request layer probe."""
    pool = [row for row in choose_staged_pilot(roots, 2000) if row["split"] == "train"]
    selected = []
    for language, quotas in (("en", (162, 72, 36)), ("zh", (18, 8, 4))):
        ranked = sorted(
            (row for row in pool if row["lang"] == language),
            key=lambda row: hashlib.sha256(
                ("layer-probe-v1:" + row["id"]).encode()
            ).digest(),
        )
        if len(ranked) < sum(quotas):
            raise ValueError("not enough stage-training roots for the layer probe")
        start = 0
        for split, quota in zip(SPLITS, quotas):
            selected += [
                {**row, "parent_split": "train", "split": split}
                for row in ranked[start : start + quota]
            ]
            start += quota
    return sorted(selected, key=lambda row: row["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--pilot-en", type=int, default=120)
    parser.add_argument("--pilot-zh", type=int, default=40)
    parser.add_argument("--exclude-legacy-train1000", action="store_true")
    parser.add_argument("--shard-size", type=int, default=0)
    parser.add_argument("--staged-requests", type=int, choices=(2000, 10000))
    parser.add_argument("--layer-probe", action="store_true")
    args = parser.parse_args()
    if (
        args.min_chars < 1
        or args.max_chars < args.min_chars
        or args.pilot_en < 0
        or args.pilot_zh < 0
        or args.shard_size < 0
    ):
        parser.error("invalid character range or pilot counts")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("out-dir must be new or empty")
    roots = load_roots(args.source, args.min_chars, args.max_chars)
    excluded_trees = set()
    if args.exclude_legacy_train1000:
        excluded_trees = {
            row["source_tree_id"] for row in choose_pilot(roots, 800, 200)
        }
    eligible = [row for row in roots if row["source_tree_id"] not in excluded_trees]
    pilot = (
        choose_staged_pilot(eligible, args.staged_requests)
        if args.staged_requests
        else choose_pilot(eligible, args.pilot_en, args.pilot_zh)
    )
    if args.layer_probe:
        pilot = choose_layer_probe(eligible)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.out_dir / "oasst1_root_requests.jsonl"
    pilot_path = args.out_dir / "oasst1_pilot_requests.jsonl"
    write_jsonl(full_path, roots)
    write_jsonl(pilot_path, pilot)
    summary = {
        "format_version": 1,
        "source_sha256": SOURCE_SHA256,
        "excluded_legacy_trees": len(excluded_trees),
        "staged_requests": args.staged_requests,
        "layer_probe": args.layer_probe,
        "filters": {
            "root_prompter": True,
            "deleted": False,
            "synthetic": False,
            "languages": ["en", "zh"],
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
        },
        "split_rule": "sha256(oasst1-split-v1:<tree_id>) mod 10: 0-7/8/9",
        "full": {
            "path": str(full_path.resolve()),
            "sha256": sha256_file(full_path),
            "count": len(roots),
            "by_lang_split": dict(
                sorted(
                    Counter(f"{row['lang']}/{row['split']}" for row in roots).items()
                )
            ),
        },
        "pilot": {
            "path": str(pilot_path.resolve()),
            "sha256": sha256_file(pilot_path),
            "count": len(pilot),
            "by_lang_split": dict(
                sorted(
                    Counter(f"{row['lang']}/{row['split']}" for row in pilot).items()
                )
            ),
        },
    }
    summary["shards"] = []
    if args.shard_size:
        for number, start in enumerate(range(0, len(pilot), args.shard_size)):
            shard_path = args.out_dir / f"shard_{number:03d}.jsonl"
            rows = pilot[start : start + args.shard_size]
            write_jsonl(shard_path, rows)
            summary["shards"].append(
                {
                    "filename": shard_path.name,
                    "count": len(rows),
                    "sha256": sha256_file(shard_path),
                }
            )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
