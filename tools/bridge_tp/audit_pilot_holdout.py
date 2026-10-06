#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove randomized pilot inputs do not overlap predictor fitting inputs.

The predictor's training and model-selection splits are both excluded. The
audit compares request IDs, source message IDs, conversation trees, and exact
message payloads. It does not claim that a shared corpus is domain-independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bridge_tp.run_randomized_goodoutput_pilot_a100 import select_inputs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def messages_digest(row: dict[str, Any]) -> str:
    payload = json.dumps(
        row["messages"], ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit(
    input_path: Path, expected_input_sha256: str, seeds: list[int],
    training_report: Path | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    actual_sha = sha256(input_path)
    if actual_sha != expected_input_sha256:
        raise ValueError(f"pilot input SHA differs: {actual_sha}")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide distinct pilot seeds")
    rows = [json.loads(line) for line in input_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    fitting = [row for row in rows
               if row.get("split") in {"train", "validation"}]
    heldout = [row for row in rows if row.get("split") == "test"]
    if not fitting or not heldout or len(fitting) + len(heldout) != len(rows):
        raise ValueError("incomplete or unknown train/validation/test split")
    keys = ("id", "source_message_id", "source_tree_id")
    fitting_values = {key: {str(row[key]) for row in fitting} for key in keys}
    fitting_messages = {messages_digest(row) for row in fitting}
    fitting_original = {row.get("original_prompt_sha256") for row in fitting}
    fitting_original.discard(None)
    selected_by_seed: dict[str, Any] = {}
    all_selected: set[str] = set()
    for seed in seeds:
        selected = select_inputs(input_path, seed)
        for row in selected:
            if row.get("split") != "test":
                raise ValueError(f"pilot selected a fitting row: {row['id']}")
            for key in keys:
                if str(row[key]) in fitting_values[key]:
                    raise ValueError(f"{key} overlaps fitting split: {row['id']}")
            if messages_digest(row) in fitting_messages:
                raise ValueError(f"message content overlaps fitting split: {row['id']}")
            original = row.get("original_prompt_sha256")
            if original and original in fitting_original:
                raise ValueError(f"original prompt overlaps fitting split: {row['id']}")
        identifiers = [str(row["id"]) for row in selected]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"duplicate selected request in seed {seed}")
        all_selected.update(identifiers)
        selected_by_seed[str(seed)] = {
            "requests": len(selected),
            "workload_groups": dict(Counter(row["workload_group"]
                                            for row in selected)),
            "request_ids_sha256": hashlib.sha256(
                "\n".join(identifiers).encode("utf-8")
            ).hexdigest(),
        }
    result: dict[str, Any] = {
        "format_version": 1, "status": "PASS",
        "input_path": str(input_path.resolve()),
        "input_sha256": actual_sha,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "fitting_splits_excluded": ["train", "validation"],
        "disjoint_keys": [*keys, "messages_sha256", "original_prompt_sha256"],
        "selected_by_seed": selected_by_seed,
        "unique_selected_requests": len(all_selected),
    }
    if training_report is not None:
        report = json.loads(training_report.read_text(encoding="utf-8"))
        reported_sha = report["capture_preflight"]["input_sha256"]
        if reported_sha != actual_sha:
            raise ValueError("training report used a different input SHA")
        result["training_report_sha256"] = sha256(training_report)
        result["training_report_input_sha256"] = reported_sha
        if checkpoint is not None:
            checkpoint_sha = sha256(checkpoint)
            if checkpoint_sha != report["checkpoint_sha256"]:
                raise ValueError("checkpoint differs from training report")
            result["checkpoint_sha256"] = checkpoint_sha
    elif checkpoint is not None:
        raise ValueError("checkpoint verification requires training report")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out-file", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.input, args.expected_input_sha256, args.seed,
        args.training_report, args.checkpoint,
    )
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
