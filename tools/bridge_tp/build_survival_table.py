#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the survival-conditioned remaining-length table from a request trace.

The controller never trains a model; it queries an empirical conditional CCDF
built here from the SAME training split that M1 uses. Keeping the split
identical matters: if the table saw the evaluation requests, every escalation
decision is contaminated.

Usage
-----
    python tools/bridge_tp/build_survival_table.py \
        --trace traces/qwen_requests.jsonl \
        --output-field output_tokens \
        --train-frac 0.7 \
        --out calibration/survival_table.json

Input may be JSONL (one object per line) or CSV with a header row.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.bridge_tp.controller.predictor import SurvivalTable  # noqa: E402

DEFAULT_EDGES = (0, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-field", default="output_tokens")
    parser.add_argument(
        "--time-field",
        default=None,
        help="sort records chronologically by this field before splitting",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="leading fraction of the time-ordered trace used for calibration",
    )
    parser.add_argument(
        "--bucket-edges",
        type=int,
        nargs="+",
        default=list(DEFAULT_EDGES),
        help="progress checkpoints in output tokens",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-total-rows", type=int)
    parser.add_argument("--expected-train-rows", type=int)
    return parser.parse_args()


def load_lengths(path: Path, field: str, time_field: str | None = None) -> list[int]:
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, object]]
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    elif path.suffix.lower() == ".json":
        rows = json.loads(text)
    else:
        reader = csv.DictReader(text.splitlines())
        rows = list(reader)
    if not rows:
        raise ValueError(f"no rows read from {path}")
    required = [field] + ([time_field] if time_field else [])
    for required_field in required:
        if required_field not in rows[0]:
            raise KeyError(f"field {required_field!r} missing from {path}")
    if time_field:
        try:
            rows.sort(key=lambda row: float(row[time_field]))
        except (TypeError, ValueError):
            rows.sort(key=lambda row: str(row[time_field]))
    lengths = [int(row[field]) for row in rows]
    if not lengths:
        raise ValueError(f"no rows read from {path}")
    return lengths


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_frac <= 1.0:
        raise SystemExit("--train-frac must be in (0,1]")

    lengths = load_lengths(args.trace, args.output_field, args.time_field)
    if (
        args.expected_total_rows is not None
        and len(lengths) != args.expected_total_rows
    ):
        raise SystemExit(
            f"expected {args.expected_total_rows} total rows, got {len(lengths)}"
        )
    cut = int(len(lengths) * args.train_frac)
    if args.expected_train_rows is not None and cut != args.expected_train_rows:
        raise SystemExit(f"expected {args.expected_train_rows} train rows, got {cut}")
    train = lengths[:cut]
    if len(train) < 100:
        raise SystemExit(
            f"only {len(train)} training requests; the conditional CCDF would be "
            "too noisy to drive escalation decisions"
        )

    table = SurvivalTable.from_output_lengths(
        train,
        bucket_edges=tuple(args.bucket_edges),
        source=(
            f"{args.trace.name} chronological first {args.train_frac:.0%} "
            f"({len(train)} requests)"
            if args.time_field
            else f"{args.trace.name} first {args.train_frac:.0%} "
            f"({len(train)} requests)"
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.save(args.out)

    print(f"wrote {args.out}")
    print(f"  training requests : {len(train)} of {len(lengths)}")
    print(f"  calibrated up to  : {table.max_observed_length} output tokens")
    print("  bucket   survivors   E[remaining]   P90[remaining]")
    for edge in table.bucket_edges:
        n = table.n_survivors(edge)
        if n == 0:
            continue
        print(
            f"  {edge:>6}   {n:>9}   {table.expected_remaining(edge):>12.1f}   "
            f"{table.quantile_remaining(edge, 0.90):>13.0f}"
        )


if __name__ == "__main__":
    main()
