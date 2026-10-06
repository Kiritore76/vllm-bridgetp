#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reconcile per-request TP1 KV blocks against sampled free KV telemetry.

Client token timestamps bound, but do not reveal, server-side prefill allocation
and KV release times. Samples inside those intervals are marked uncertain rather
than forced to match. This is an observed-trajectory audit, not a risk predictor.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import tarfile
from pathlib import Path
from typing import Any


def read_json(archive: tarfile.TarFile, name: str) -> Any:
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f"missing archive member: {name}")
    return json.load(stream)


def read_jsonl(archive: tarfile.TarFile, name: str) -> list[dict[str, Any]]:
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f"missing archive member: {name}")
    return [json.loads(line) for line in stream if line.strip()]


def rounded_blocks(tokens: int, block_size: int) -> int:
    return (tokens + block_size - 1) // block_size


def background_state(
    jobs: list[dict[str, Any]], sample_s: float, block_size: int,
    release_uncertainty_s: float,
) -> tuple[int, list[dict[str, Any]], list[str]]:
    """Return known source blocks and client-observation uncertainty flags."""
    used_blocks = 0
    active: list[dict[str, Any]] = []
    uncertain: list[str] = []
    for job in jobs:
        started = job["request_started_unix_s"]
        first = job["first_token_unix_s"]
        last = job["last_token_unix_s"]
        ended = job["request_ended_unix_s"]
        if sample_s < started:
            continue
        if sample_s >= ended:
            if sample_s < ended + release_uncertainty_s:
                uncertain.append(f"{job['job_id']}:release_time_unknown")
            continue
        if first is None or sample_s < first:
            uncertain.append(f"{job['job_id']}:prefill_allocation_time_unknown")
            continue
        if last is not None and sample_s > last:
            uncertain.append(f"{job['job_id']}:release_time_unknown")
        generated = bisect.bisect_right(job["token_times_unix_s"], sample_s)
        blocks = rounded_blocks(job["prompt_tokens"] + generated, block_size)
        used_blocks += blocks
        active.append({
            "job_id": job["job_id"], "prompt_tokens": job["prompt_tokens"],
            "generated_tokens": generated, "used_blocks": blocks,
        })
    return used_blocks, active, uncertain


def replay_sample(
    sample: dict[str, Any], jobs: list[dict[str, Any]],
    anchor: dict[str, Any], tp1_blocks: int, guard_tokens: int,
    release_uncertainty_s: float,
) -> dict[str, Any]:
    tp1 = sample["tp1"]
    block_size = int(tp1["block_size"])
    sample_s = float(tp1["sampled_unix_s"])
    bg_blocks, active, uncertain = background_state(
        jobs, sample_s, block_size, release_uncertainty_s,
    )
    anchor_blocks = 0
    if anchor["started_s"] <= sample_s < anchor["released_s"]:
        if sample_s < anchor["first_token_s"]:
            uncertain.append("anchor:prefill_allocation_time_unknown")
        else:
            # The controller's token count is sampled with the same TP1
            # telemetry and is closer to scheduled KV than proxy output time.
            anchor_blocks = rounded_blocks(
                anchor["prompt_tokens"] + int(sample["output_tokens"]),
                block_size,
            )
    elif (not anchor["release_exact"]
          and anchor["released_s"] <= sample_s
          < anchor["released_s"] + release_uncertainty_s):
        uncertain.append("anchor:release_time_unknown")
    observed_free_blocks = int(tp1["free_kv_blocks"])
    replayed_free_blocks = tp1_blocks - bg_blocks - anchor_blocks
    residual_blocks = observed_free_blocks - replayed_free_blocks
    pending_prefill_tokens = int(tp1.get("prefill_pending_kv_tokens") or 0)
    return {
        "unix_s": sample_s, "tick": sample.get("tick"),
        "block_size": block_size, "tp1_total_blocks": tp1_blocks,
        "observed_free_tokens": observed_free_blocks * block_size,
        "replayed_free_tokens": replayed_free_blocks * block_size,
        "residual_tokens": residual_blocks * block_size,
        "pending_prefill_tokens": pending_prefill_tokens,
        "guard_tokens": guard_tokens,
        "observed_guard_hit": (
            observed_free_blocks * block_size - pending_prefill_tokens
            <= guard_tokens),
        "replayed_guard_hit": (
            replayed_free_blocks * block_size - pending_prefill_tokens
            <= guard_tokens),
        "anchor_used_blocks": anchor_blocks,
        "background_used_blocks": bg_blocks,
        "active_background": active,
        "uncertainty": uncertain,
        "verifiable": not uncertain,
        "tp1_preemptions_total": tp1.get("preemptions_total"),
    }


def episode_ledger(
    archive: tarfile.TarFile, prefix: str, tp1_blocks: int,
    tolerance_blocks: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(archive, prefix + "/background/background_manifest.json")
    background = read_json(archive, prefix + "/background/background_summary.json")
    request = read_json(archive, prefix + "/controller/source_request.json")
    response = read_json(archive, prefix + "/controller/source_response.json")
    members = set(archive.getnames())
    receipt_name = prefix + "/controller/source_kv_release_receipt.json"
    receipt = read_json(archive, receipt_name) if receipt_name in members else None
    audit = read_jsonl(archive, prefix + "/controller/phase9_audit.jsonl")
    samples = [row for row in audit if row.get("kind") == "telemetry"]
    if not samples:
        raise ValueError(f"no TP1 telemetry: {prefix}")
    times = sorted(float(row["tp1"]["sampled_unix_s"]) for row in samples)
    gaps = [right - left for left, right in zip(times, times[1:])
            if right > left]
    release_uncertainty_s = max(
        0.5, 2 * statistics.median(gaps) if gaps else 0.5,
    )
    guards = {row["capacity_signal"]["guard_free_kv_tokens"]
              for row in samples}
    blocks = {row["tp1"]["block_size"] for row in samples}
    if len(guards) != 1 or len(blocks) != 1:
        raise ValueError(f"guard or block size changed: {prefix}")
    guard = next(iter(guards))
    block_size = next(iter(blocks))
    if tp1_blocks <= 0 or tolerance_blocks < 0:
        raise ValueError("TP1 blocks must be positive and tolerance nonnegative")
    prompt_by_job = {
        job["job_id"]: len(job["request"]["prompt"])
        for job in manifest["jobs"] if job["pool"] == "source"
    }
    jobs = [{**job, "prompt_tokens": prompt_by_job[job["job_id"]]}
            for job in background["results"] if job["pool"] == "source"]
    if len(jobs) != len(prompt_by_job):
        raise ValueError(f"source job count mismatch: {prefix}")
    if any(len(job["token_times_unix_s"]) != job["output_tokens"]
           for job in jobs):
        raise ValueError(f"source token timestamps incomplete: {prefix}")
    if receipt and receipt.get("status") == "SOURCE_KV_RELEASED":
        released_s = receipt["released_unix_ns"] / 1e9
        release_evidence = "EXACT_SOURCE_KV_RELEASE_RECEIPT"
    elif response.get("finish_reason") == "stop":
        released_s = response["completed_unix_s"]
        release_evidence = "NATURAL_EOS_RESPONSE_PROXY"
    else:
        raise ValueError(f"source KV release time unavailable: {prefix}")
    anchor = {
        "prompt_tokens": len(request["prompt"]),
        "started_s": response["request_started_unix_s"],
        "first_token_s": response["first_token_unix_s"],
        "released_s": released_s,
        "release_exact": release_evidence == "EXACT_SOURCE_KV_RELEASE_RECEIPT",
    }
    rows = [replay_sample(row, jobs, anchor, tp1_blocks, guard,
                          release_uncertainty_s)
            for row in samples]
    verifiable = [row for row in rows if row["verifiable"]]
    if not verifiable:
        raise ValueError(f"no verifiable samples: {prefix}")
    bad = [row for row in verifiable
           if abs(row["residual_tokens"]) > tolerance_blocks * block_size]
    first_release = min(jobs, key=lambda job: job["request_ended_unix_s"])
    before = max((row for row in rows
                  if row["unix_s"] <= first_release["request_ended_unix_s"]),
                 key=lambda row: row["unix_s"], default=None)
    after = min((row for row in rows
                 if row["unix_s"] >= (
                     first_release["request_ended_unix_s"]
                     + release_uncertainty_s)),
                key=lambda row: row["unix_s"], default=None)
    released_between = [job for job in jobs
                        if before and after
                        and before["unix_s"] < job["request_ended_unix_s"]
                        <= after["unix_s"]]
    summary = {
        "episode": prefix, "format_version": 1,
        "tp1_total_blocks": tp1_blocks, "block_size": block_size,
        "guard_free_kv_tokens": guard,
        "source_jobs": len(jobs), "source_prompt_tokens_total": sum(
            job["prompt_tokens"] for job in jobs),
        "source_finish_reasons": {job["job_id"]: job["finish_reason"]
                                  for job in jobs},
        "anchor_release_evidence": release_evidence,
        "release_uncertainty_s": release_uncertainty_s,
        "telemetry_samples": len(rows), "verifiable_samples": len(verifiable),
        "uncertain_samples": len(rows) - len(verifiable),
        "samples_outside_tolerance": len(bad),
        "max_abs_residual_tokens": max(abs(row["residual_tokens"])
                                       for row in verifiable),
        "median_residual_tokens": statistics.median(
            row["residual_tokens"] for row in verifiable),
        "min_observed_free_tokens": min(row["observed_free_tokens"]
                                        for row in rows),
        "min_replayed_free_tokens_verifiable": min(
            row["replayed_free_tokens"] for row in verifiable),
        "observed_guard_hit_samples": sum(row["observed_guard_hit"]
                                          for row in rows),
        "verifiable_guard_classification_mismatches": sum(
            row["observed_guard_hit"] != row["replayed_guard_hit"]
            for row in verifiable),
        "preemptions_peak": max(int(row["tp1_preemptions_total"] or 0)
                                for row in rows),
        "first_source_eos": {
            "job_id": first_release["job_id"],
            "output_tokens": first_release["output_tokens"],
            "request_ended_unix_s": first_release["request_ended_unix_s"],
            "expected_released_blocks": rounded_blocks(
                first_release["prompt_tokens"]
                + first_release["output_tokens"], block_size),
            "observed_free_before_tokens": (
                before["observed_free_tokens"] if before else None),
            "observed_free_after_tokens": (
                after["observed_free_tokens"] if after else None),
            "sample_gap_s": (after["unix_s"] - before["unix_s"]
                             if before and after else None),
            "jobs_released_between_samples": [job["job_id"]
                                              for job in released_between],
            "expected_released_blocks_between_samples": sum(
                rounded_blocks(job["prompt_tokens"] + job["output_tokens"],
                               block_size) for job in released_between),
        },
        "reconciliation": ("PASS" if not bad else "MISMATCH"),
        "limitations": [
            "Client first-token time bounds but does not timestamp prefill allocation.",
            "Client completion time bounds but may lag server KV release.",
            "Telemetry is sampled; guard contact between samples is unobserved.",
        ],
    }
    return summary, rows


def audit_archives(
    paths: list[Path], tp1_blocks: int, tolerance_blocks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for path in paths:
        with tarfile.open(path, "r:gz") as archive:
            suffix = "/controller/phase9_audit.jsonl"
            prefixes = sorted({name[:-len(suffix)] for name in archive.getnames()
                               if name.endswith(suffix)})
            for prefix in prefixes:
                summary, rows = episode_ledger(
                    archive, prefix, tp1_blocks, tolerance_blocks,
                )
                summary["archive"] = str(path)
                summaries.append(summary)
                samples.extend({"episode": prefix, **row} for row in rows)
    if not summaries:
        raise ValueError("no episodes with TP1 telemetry found")
    return summaries, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tp1-blocks", type=int, required=True)
    parser.add_argument("--tolerance-blocks", type=int, default=2)
    args = parser.parse_args()
    summaries, rows = audit_archives(
        args.archive, args.tp1_blocks, args.tolerance_blocks,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "episode_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "sample_ledger.jsonl").open(
        "w", encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "episodes": len(summaries),
        "reconciled": sum(item["reconciliation"] == "PASS"
                          for item in summaries),
        "observed_guard_hit_episodes": sum(
            item["observed_guard_hit_samples"] > 0 for item in summaries),
        "outputs": str(args.out_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
