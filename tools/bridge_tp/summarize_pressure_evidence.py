#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize existing development traces for pressure policy, without fitting benefit.

Only STAY batches enter service lookup cells. START traces are retained as
migration observations. GPU execution, engine latency and visible ITL are
different quantities. No final-test archive or predictor training is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np


def quantiles(values):
    values = list(values)
    if not values:
        return None
    if any(not math.isfinite(v) for v in values):
        raise ValueError("nonfinite evidence")
    return dict(
        zip(
            ("min", "p50", "p95", "p99", "max"),
            map(float, np.percentile(values, [0, 50, 95, 99, 100])),
        )
    )


def phase(row):
    if row["recompute_tokens"]:
        return "recompute"
    tokens = row["prefill_tokens"]
    return (
        "decode"
        if tokens == 0
        else "prefill_lt256"
        if tokens < 256
        else "prefill_256_767"
        if tokens < 768
        else "prefill_768_1535"
        if tokens < 1536
        else "prefill_ge1536"
    )


def cell_key(row):
    nd = row["decode_requests"]
    context = row["decode_context_sum"] / nd if nd else 0
    return (
        row["pool"],
        phase(row),
        nd,
        int(context // 1024),
        math.ceil(row["prefill_tokens"] / 256),
    )


def lookup_cells(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[cell_key(row)].append(row)
    cells = []
    for key, group in sorted(groups.items()):
        by_run = defaultdict(list)
        for row in group:
            by_run[row["run"]].append(row["gpu_ms"])
        cells.append(
            dict(
                pool=key[0],
                phase=key[1],
                decode_requests=key[2],
                mean_context_bin_1024=key[3],
                prefill_bin_256=key[4],
                samples=len(group),
                runs=len(by_run),
                run_ids=sorted(by_run),
                gpu_ms=quantiles(r["gpu_ms"] for r in group),
                engine_ms=quantiles(r["engine_ms"] for r in group),
                run_median_gpu_ms=quantiles(np.median(v) for v in by_run.values()),
                context_sum_range=quantiles(r["decode_context_sum"] for r in group),
                prefill_tokens_range=quantiles(r["prefill_tokens"] for r in group),
                support="descriptive_only" if len(by_run) < 2 else "local_draft",
            )
        )
    return cells


def load_archive(entry):
    path = Path(entry["archive"])
    assessment = json.loads(Path(entry["assessment"]).read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != assessment["archive_sha256"]:
        raise ValueError("archive SHA differs from audited assessment")
    files = {}
    suffixes = (
        "pilot_summary.json",
        "phase9_audit.jsonl",
        "target_native_events.jsonl",
        "source_kv_release_receipt.json",
        "request_frozen_receipt.json",
        "background_summary.json",
        "response_proxy_stats.json",
    )
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if "/test/" in member.name:
                raise ValueError("final test evidence must remain sealed")
            if "/performance_trace/" in member.name or member.name.endswith(suffixes):
                files[member.name] = archive.extractfile(member).read()
    return assessment, files, digest


def lines(files, name):
    return [json.loads(v) for v in files.get(name, b"").splitlines() if v.strip()]


def rolling_pressure(intervals, origin, cutoff, window_s=3, tick_s=0.2):
    ordered = iter(sorted(intervals))
    future = next(ordered, None)
    recent = deque()
    ratios = []
    for tick in np.arange(origin, cutoff, tick_s):
        while future is not None and future[0] <= tick:
            recent.append(future)
            future = next(ordered, None)
        while recent and recent[0][0] < tick - window_s:
            recent.popleft()
        if len(recent) < 20:
            continue
        values = [v for _, v in recent]
        ratios.append(
            max(
                float(np.percentile(values, 95)) / 0.1,
                sum(v > 0.1 for v in values) / len(values) / 0.01,
            )
        )
    return dict(
        window_s=window_s,
        tick_s=tick_s,
        supported_ticks=len(ratios),
        ratio=quantiles(ratios),
        overload_ticks=sum(v >= 1.2 for v in ratios),
        scope="retrospective prefixes, not proof of live publication",
    )


def summarize(entries):
    rows, migrations, archives = [], [], []
    arrival_prefill, growth, evidence_gaps, releases = [], [], Counter(), []
    stream_pressure = []
    for entry in entries:
        print("reading", entry["archive"], flush=True)
        assessment, files, digest = load_archive(entry)
        archives.append(
            dict(
                path=entry["archive"],
                sha256=digest,
                assessment=entry["assessment"],
                role="development_only",
                provenance=assessment.get("launch_provenance"),
            )
        )
        for condition, record in assessment["conditions"].items():
            if record.get("handoff"):
                migrations.append(
                    dict(
                        archive=digest,
                        condition=condition,
                        handoff=record["handoff"],
                        exclusions=record.get("exclusions", []),
                    )
                )
        for summary_name in sorted(
            n for n in files if n.endswith("pilot_summary.json")
        ):
            summary = json.loads(files[summary_name])
            pair = summary_name.removesuffix("/pilot_summary.json")
            for case, arms in summary["cases"].items():
                for arm, detail in arms.items():
                    if arm not in {"prob_0_STAY", "prob_0_START"}:
                        continue
                    run = f"{pair}/{case}/{arm}/r01_shadow_only"
                    horizon = detail["horizon_score"]
                    origin, cutoff = horizon["origin_unix_s"], horizon["cutoff_unix_s"]
                    source_audit = lines(files, run + "/controller/phase9_audit.jsonl")
                    release_name = run + "/controller/source_kv_release_receipt.json"
                    freeze_name = run + "/controller/request_frozen_receipt.json"
                    starts = [
                        v
                        for v in source_audit
                        if v.get("kind") == "transition" and v.get("to") == "SHADOW"
                    ]

                    if starts and release_name in files:
                        release = json.loads(files[release_name])
                        start = starts[0]
                        if release["request_id"] != start["request_id"]:
                            raise ValueError("release receipt request mismatch")
                        released = release["released_unix_ns"] / 1e9
                        freeze = (
                            json.loads(files[freeze_name])
                            if freeze_name in files
                            else None
                        )
                        if freeze and freeze["request_id"] != release["request_id"]:
                            raise ValueError("freeze receipt request mismatch")
                        releases.append(
                            dict(
                                run=run,
                                request_id=release["request_id"],
                                start_to_release_s=released - start["unix_s"],
                                freeze_to_release_s=released
                                - freeze["frozen_unix_ns"] / 1e9
                                if freeze
                                else None,
                                evidence=release["evidence"],
                                context=release["num_computed_tokens"],
                            )
                        )
                    telemetry = [
                        v for v in source_audit if v.get("kind") == "telemetry"
                    ]
                    for a, b in zip(telemetry, telemetry[1:]):
                        dt = b["unix_s"] - a["unix_s"]
                        if not 0 < dt <= 2:
                            continue
                        for name in ("tp1", "tp4"):
                            if (
                                "free_kv_blocks" in a[name]
                                and "free_kv_blocks" in b[name]
                            ):
                                growth.append(
                                    dict(
                                        pool=name,
                                        blocks_s=(
                                            a[name]["free_kv_blocks"]
                                            - b[name]["free_kv_blocks"]
                                        )
                                        / dt,
                                    )
                                )
                    target_intervals = lines(
                        files, run + "/background/target_native_events.jsonl"
                    )
                    retrospective = {"source": [], "target": []}
                    background_name = run + "/background/background_summary.json"
                    if background_name in files:
                        background = json.loads(files[background_name])
                        results = background.get("results", [])
                        if isinstance(results, dict):
                            results = results.values()
                        for result in results:
                            times = result.get("token_times_unix_s", [])
                            for a, end in zip(times, times[1:]):
                                if origin <= end < cutoff and end >= a:
                                    retrospective[result["pool"]].append((end, end - a))
                    proxy_name = run + "/controller/response_proxy_stats.json"
                    if proxy_name in files:
                        emitted = json.loads(files[proxy_name]).get("emitted", [])
                        for a, end in zip(emitted, emitted[1:]):
                            if (
                                a["origin"] == end["origin"]
                                and origin <= end["unix_s"] < cutoff
                                and end["unix_s"] >= a["unix_s"]
                            ):
                                retrospective[end["origin"]].append(
                                    (end["unix_s"], end["unix_s"] - a["unix_s"])
                                )
                    for pool_name, intervals in retrospective.items():
                        stream_pressure.append(
                            dict(
                                run=run,
                                pool=pool_name,
                                arm=arm,
                                samples=len(intervals),
                                intervals_s=quantiles(v for _, v in intervals),
                                rolling=rolling_pressure(intervals, origin, cutoff),
                            )
                        )
                    if not target_intervals:
                        evidence_gaps["runs_without_target_live_itl"] += 1
                    evidence_gaps["runs_without_source_live_itl"] += 1
                    for pool, ranks in (("source", 1), ("target", 4)):
                        folder = run + f"/controller/performance_trace/{pool}"
                        gpu = defaultdict(list)
                        for rank in range(ranks):
                            for v in lines(files, folder + f"/gpu_rank{rank}.jsonl"):
                                if v.get("kind") == "gpu_batch":
                                    gpu[v["step_id"]].append(v["gpu_forward_logits_ms"])
                        engine = lines(files, folder + "/engine.jsonl")
                        prefill_spans = []
                        for v in engine:
                            if (
                                v.get("kind") != "engine_batch"
                                or not origin <= v["scheduled_unix_s"] < cutoff
                            ):
                                continue
                            values = gpu[v["step_id"]]
                            if len(values) != ranks or v["speculative"]:
                                raise ValueError(
                                    "incomplete ranks or speculative evidence"
                                )
                            if v["prefill_tokens"]:
                                prefill_spans.append(
                                    (v["scheduled_unix_s"], v["completed_unix_s"])
                                )
                            if arm == "prob_0_STAY":
                                rows.append(
                                    dict(
                                        pool=pool,
                                        run=pair,
                                        archive=digest,
                                        gpu_ms=max(values),
                                        engine_ms=v["completion_latency_ms"],
                                        **{
                                            k: v[k]
                                            for k in (
                                                "decode_requests",
                                                "decode_context_sum",
                                                "prefill_tokens",
                                                "recompute_tokens",
                                            )
                                        },
                                    )
                                )
                        if pool == "target":
                            groups = defaultdict(list)
                            for event in target_intervals:
                                if event.get("kind") != "native_itl":
                                    continue
                                t = event["unix_s"]
                                if not origin <= t < cutoff:
                                    continue
                                # Include an ITL if its interval overlaps a prefill
                                # engine span, not just if its ending timestamp does.
                                overlap = any(
                                    lo <= t and hi >= t - event["itl_s"]
                                    for lo, hi in prefill_spans
                                )
                                groups[
                                    "overlap_prefill" if overlap else "outside_prefill"
                                ].append(event["itl_s"])
                            arrival_prefill.append(
                                dict(
                                    run=run,
                                    arm=arm,
                                    timestamp_basis="retrospective_interval_overlap",
                                    **{
                                        key: dict(
                                            samples=len(v),
                                            itl_s=quantiles(v),
                                            slow_fraction=sum(x > 0.1 for x in v)
                                            / len(v),
                                        )
                                        for key, v in groups.items()
                                    },
                                )
                            )
        print("completed", Path(entry["archive"]).name, flush=True)
    return dict(
        schema="PRESSURE_EVIDENCE_V1",
        archives=archives,
        no_benefit_fit=True,
        no_predictor_training=True,
        no_final_test=True,
        service_lookup=lookup_cells(rows),
        service_totals={
            pool: dict(
                samples=len([v for v in rows if v["pool"] == pool]),
                runs=len({v["run"] for v in rows if v["pool"] == pool}),
            )
            for pool in ("source", "target")
        },
        prefill_itl=arrival_prefill,
        retrospective_stream_pressure=stream_pressure,
        migrations=migrations,
        source_releases=releases,
        source_release_summary={
            key: quantiles(v[key] for v in releases if v[key] is not None)
            for key in ("start_to_release_s", "freeze_to_release_s")
        },
        migration_budget_summary={
            key: quantiles(
                v["handoff"][key]
                for v in migrations
                if v["handoff"].get(key) is not None
            )
            for key in (
                "visible_handoff_ms",
                "freeze_to_final_delta_ack_ms_existing_acceptance_boundary",
                "scheduler_freeze_to_first_visible_token_ms",
            )
        },
        observed_signed_physical_growth={
            pool: quantiles(v["blocks_s"] for v in growth if v["pool"] == pool)
            for pool in ("tp1", "tp4")
        },
        evidence_gaps=dict(evidence_gaps)
        | {
            "unique_releasable_candidate_blocks": (
                "not present in old controller telemetry"
            ),
            "source_release_timing": (
                "matched free-return receipts; handoff/commit alone are insufficient"
            ),
            "lookup_limit": (
                "GPU span is not visible ITL; no service SLO threshold inferred"
            ),
            "counterfactual_limit": (
                "observed cells do not prove remove/add-one-request deltas"
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = summarize(entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "service_totals": result["service_totals"],
                "cells": len(result["service_lookup"]),
                "migration_budgets": result["migration_budget_summary"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
