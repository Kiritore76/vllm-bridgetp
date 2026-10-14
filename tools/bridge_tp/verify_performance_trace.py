# SPDX-License-Identifier: Apache-2.0
"""Validate sampled batch identity, durable receipts and actual GPU timing."""

import argparse
import json
import math
from pathlib import Path

SCHEMA = "BRIDGETP_BATCH_PERFORMANCE_V1"


def read_journal(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if (
        len(rows) < 2
        or rows[0].get("kind") != "trace_header"
        or rows[0].get("schema") != SCHEMA
    ):
        raise ValueError(f"missing performance header: {path}")
    footer = rows[-1]
    if (
        footer.get("kind") != "evidence_footer"
        or footer.get("dropped") != 0
        or footer.get("writer_error") is not None
        or footer.get("saved_records") != len(rows) - 1
    ):
        raise ValueError(f"incomplete performance journal: {path}")
    return rows


def indexed(rows, kind):
    selected = [row for row in rows if row.get("kind") == kind]
    result = {row["step_id"]: row for row in selected}
    if len(result) != len(selected):
        raise ValueError("duplicate batch id")
    if not result:
        raise ValueError("no completed performance samples")
    return result


def verify_pool(directory, ranks):
    engine = indexed(read_journal(directory / "engine.jsonl"), "engine_batch")
    expected_names = {f"gpu_rank{rank}.jsonl" for rank in range(ranks)}
    if {p.name for p in directory.glob("gpu_rank*.jsonl")} != expected_names:
        raise ValueError("GPU rank evidence set differs")
    gpu_max = {}
    for step, row in engine.items():
        duration = row["completion_latency_ms"]
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid engine completion latency")
        if sum(r["query_tokens"] for r in row["requests"]) != row["scheduled_tokens"]:
            raise ValueError("batch composition count differs")
        if (
            row["prefill_tokens"] + row["recompute_tokens"] + row["decode_tokens"]
            != row["scheduled_tokens"]
        ):
            raise ValueError("scheduled work partition differs")
        if row["speculative"]:
            raise ValueError("pilot requires non-speculative decode")
        gpu_max[step] = 0.0
    for rank in range(ranks):
        rows = read_journal(directory / f"gpu_rank{rank}.jsonl")
        receipts = [row for row in rows if row.get("kind") == "gpu_trace_receipt"]
        if (
            len(receipts) != 1
            or receipts[0]["pending"]
            or receipts[0]["dropped_samples"]
            or receipts[0]["timing_error"]
        ):
            raise ValueError("missing or incomplete GPU event receipt")
        batches = indexed(rows, "gpu_batch")
        if set(batches) != set(engine):
            raise ValueError("engine/GPU batch ids differ")
        for step, row in batches.items():
            if (
                row["rank"] != rank
                or row["requests"] != engine[step]["requests"]
                or row["submitted_monotonic_s"] != engine[step]["submitted_monotonic_s"]
            ):
                raise ValueError("GPU batch identity/composition differs")
            duration = row["gpu_forward_logits_ms"]
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("invalid GPU event duration")
            gpu_max[step] = max(gpu_max[step], duration)
    return {
        "samples": len(engine),
        "ranks": ranks,
        "prefill_samples": sum(r["prefill_tokens"] > 0 for r in engine.values()),
        "decode_samples": sum(r["decode_tokens"] > 0 for r in engine.values()),
        "recompute_samples": sum(r["recompute_tokens"] > 0 for r in engine.values()),
        "gpu_rank_max_ms_min": min(gpu_max.values()),
        "gpu_rank_max_ms_max": max(gpu_max.values()),
        "completed_unix_s_min": min(r["completed_unix_s"] for r in engine.values()),
        "completed_unix_s_max": max(r["completed_unix_s"] for r in engine.values()),
    }


def verify(root, expected_arms):
    directories = sorted(root.glob("**/controller/performance_trace"))
    if len(directories) != expected_arms:
        raise ValueError(f"expected {expected_arms} arms, got {len(directories)}")
    arms = []
    for directory in directories:
        arm = {"path": str(directory.relative_to(root)), "pools": {}}
        for pool, ranks in (("source", 1), ("target", 4)):
            receipt = json.loads(
                (
                    directory.parent.parent
                    / "provenance"
                    / f"performance_trace_{pool}_finalization.json"
                ).read_text()
            )
            if (
                receipt.get("status") != "FINALIZED"
                or receipt.get("engine", {}).get("status") != "FINALIZED"
                or len(receipt.get("workers", [])) != ranks
                or {r.get("rank") for r in receipt["workers"]} != set(range(ranks))
                or any(r.get("status") != "FINALIZED" for r in receipt["workers"])
            ):
                raise ValueError(f"invalid finalization receipt: {pool}")
            arm["pools"][pool] = verify_pool(directory / pool, ranks)
        proxy = json.loads((directory.parent / "response_proxy_stats.json").read_text())
        epoch = float(proxy["emitted"][0]["unix_s"])
        for pool in ("source", "target"):
            batches = indexed(
                read_journal(directory / pool / "engine.jsonl"), "engine_batch"
            )
            measured = [
                r
                for r in batches.values()
                if epoch <= r["scheduled_unix_s"] < epoch + 300
            ]
            counts = {
                "samples": len(measured),
                "prefill_samples": sum(r["prefill_tokens"] > 0 for r in measured),
                "decode_samples": sum(r["decode_tokens"] > 0 for r in measured),
            }
            arm["pools"][pool]["after_anchor_first_token_300s"] = counts
            if pool == "source" and not counts["decode_samples"]:
                raise ValueError("no measured source decode evidence")
            if (
                pool == "target"
                and "performance-pilot-prefill-" in str(directory)
                and not counts["prefill_samples"]
            ):
                raise ValueError("high-prefill arm lacks measured prefill samples")
        arms.append(arm)
    return {
        "status": "PASS",
        "schema": SCHEMA,
        "arms": arms,
        "note": "sampling correctness only; overhead not yet qualified",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-arms", type=int, default=4)
    args = parser.parse_args()
    report = verify(args.root, args.expected_arms)
    output = args.root / "performance_trace_acceptance.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
