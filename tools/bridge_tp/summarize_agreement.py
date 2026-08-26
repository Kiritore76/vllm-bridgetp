#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D-3: paired A/B/C/D target-local agreement statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.bridge_tp.controller.numerics import (  # noqa: E402
    paired_bootstrap_mean_difference,
    summarize_samples,
)

REQUIRED_METADATA = ("model", "gpu_platform", "vllm_commit", "cutover_rule")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for group in "abcd":
        parser.add_argument(
            f"--group-{group}",
            type=Path,
            nargs="+",
            required=True,
            help=f"group {group.upper()} record files or directories",
        )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--noninferiority-margin-tokens", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def expand_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.json")))
        else:
            expanded.append(path)
    return expanded


def load_records(paths: list[Path], expected_group: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in expand_paths(paths):
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            if item.get("group") != expected_group:
                raise SystemExit(
                    f"{path}: expected group {expected_group}, got {item.get('group')}"
                )
            request_id = str(item.get("request_id", ""))
            if not request_id or request_id in records:
                raise SystemExit(
                    f"{path}: missing or duplicate request_id {request_id!r}"
                )
            if not item.get("complete_budget"):
                raise SystemExit(f"{path}: incomplete comparison budget")
            metadata = item.get("metadata") or {}
            missing = [field for field in REQUIRED_METADATA if field not in metadata]
            if missing:
                raise SystemExit(f"{path}: metadata is missing {missing}")
            records[request_id] = item
    if not records:
        raise SystemExit(f"group {expected_group} has no records")
    return records


def validate_pairs(groups: dict[str, dict[str, dict]]) -> list[str]:
    id_sets = {name: set(records) for name, records in groups.items()}
    first = id_sets["A"]
    if any(values != first for values in id_sets.values()):
        raise SystemExit(
            "A/B/C/D must use the same request IDs: "
            + ", ".join(f"{name}={len(values)}" for name, values in id_sets.items())
        )
    request_ids = sorted(first)
    for request_id in request_ids:
        records = [groups[name][request_id] for name in "ABCD"]
        fields = {
            "boundary_k": {record["boundary_k"] for record in records},
            "budget": {record["budget"] for record in records},
            "fixed_prefix_sha256": {
                record["fixed_prefix_sha256"] for record in records
            },
        }
        metadata_values = {
            field: {record["metadata"][field] for record in records}
            for field in REQUIRED_METADATA
        }
        mismatches = {
            name: values
            for name, values in {**fields, **metadata_values}.items()
            if len(values) != 1
        }
        if mismatches:
            raise SystemExit(
                f"request {request_id}: paired provenance mismatch {mismatches}"
            )
    return request_ids


def main() -> None:
    args = parse_args()
    groups = {
        name: load_records(getattr(args, f"group_{name.lower()}"), name)
        for name in "ABCD"
    }
    request_ids = validate_pairs(groups)
    values = {
        name: [
            groups[name][request_id]["agreement_length"] for request_id in request_ids
        ]
        for name in "ABCD"
    }
    budgets = {groups["A"][request_id]["budget"] for request_id in request_ids}
    if len(budgets) != 1:
        raise SystemExit("all requests must use the same comparison budget")
    budget = next(iter(budgets))
    summaries = {
        name: summarize_samples(samples, budget=budget)
        for name, samples in values.items()
    }
    paired = paired_bootstrap_mean_difference(
        values["D"],
        values["B"],
        confidence=args.confidence,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    n = len(request_ids)
    baseline_reproducible = summaries["A"]["fully_agreeing_fraction"] == 1.0
    formal_ready = n >= 30 and baseline_reproducible
    noninferiority = None
    if args.noninferiority_margin_tokens is not None:
        if args.noninferiority_margin_tokens < 0:
            raise SystemExit("noninferiority margin must be nonnegative")
        noninferiority = {
            "margin_tokens": args.noninferiority_margin_tokens,
            "criterion": "paired bootstrap CI low >= -margin",
            "supported": (
                formal_ready and paired["ci_low"] >= -args.noninferiority_margin_tokens
            ),
        }
    warnings: list[str] = []
    if n < 30:
        warnings.append("fewer than 30 paired requests: no formal D-vs-B conclusion")
    elif n < 50:
        warnings.append("30-49 pairs are usable; 50 pairs remain preferred")
    if not baseline_reproducible:
        warnings.append(
            "group A is not fully reproducible; D-vs-B is not interpretable"
        )
    if noninferiority is None:
        warnings.append(
            "no preregistered noninferiority margin was supplied; CI is descriptive"
        )

    payload = {
        "format_version": 1,
        "phase": "BridgeTP D3 Phase 9 D-3",
        "comparison_space": "target_local_after_fixed_prefix_k",
        "paired_request_ids": request_ids,
        "n": n,
        "budget": budget,
        "groups": {
            "A": {"definition": "native TP1 vs native TP1 rerun", **summaries["A"]},
            "B": {"definition": "native TP1 vs native TP4", **summaries["B"]},
            "C": {
                "definition": "native TP4 bs=1 vs native TP4 bs=8",
                **summaries["C"],
            },
            "D": {"definition": "migrated vs native TP1 control", **summaries["D"]},
        },
        "paired_D_minus_B": paired,
        "baseline_reproducible": baseline_reproducible,
        "formal_ready": formal_ready,
        "noninferiority": noninferiority,
        "warnings": warnings,
        "evidence_boundary": (
            "D is the migration object and B is only the topology control. "
            "Statistics are paired by request, fixed-prefix hash, K, platform, "
            "model, vLLM commit, budget, and cutover rule."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"paired requests: {n}; target-local budget: {budget}")
    for name in "ABCD":
        summary = summaries[name]
        print(
            f"{name}: median={summary['median']:.1f}, "
            f"mean={summary['mean']:.1f}, "
            f"full={summary['fully_agreeing_fraction']:.1%}"
        )
    print(
        "paired D-B mean difference: "
        f"{paired['estimate']:.3f} tokens, "
        f"{args.confidence:.0%} CI [{paired['ci_low']:.3f}, "
        f"{paired['ci_high']:.3f}]"
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
