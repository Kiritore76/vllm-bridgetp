# SPDX-License-Identifier: Apache-2.0
"""Summarize actual TP2 service traces; absent migration costs stay UNKNOWN."""

import argparse
import collections
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp.verify_performance_trace import (  # noqa: E402
    read_journal,
    verify_pool,
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(values, q):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def stats(values):
    return dict(
        min=min(values),
        median=statistics.median(values),
        p95=quantile(values, 0.95),
        max=max(values),
    )


def window_score(streams, left, right):
    """Frozen causal window quality, including the predecessor and open tail."""
    good = raw = 0
    for times in streams:
        inside = [t for t in times if left <= t < right]
        gaps = [b - a for a, b in zip(times, times[1:]) if left <= b < right]
        fraction = sum(g > 0.1 for g in gaps) / len(gaps) if gaps else 0
        maximum = max(
            gaps + [right - inside[-1] if inside and times[-1] >= right else 0]
        )
        weight = min(
            1, 0.01 / fraction if fraction else 1, 1 / maximum if maximum > 1 else 1
        )
        raw += len(inside)
        good += len(inside) * weight
    return dict(raw_s=raw / (right - left), good_s=good / (right - left))


def summarize(root):
    selection = read(root / "selection.json")
    if selection["target_tp"] != 2:
        raise ValueError("not a TP2 profile collection")
    groups = collections.defaultdict(list)
    windows = collections.defaultdict(list)
    failures = []
    provenance = {}
    for scene in sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "recipe.json").exists()
    ):
        if not (scene / "request_results.json").is_file():
            failures.append(dict(run=scene.name, reason="missing request results"))
            continue
        results = read(scene / "request_results.json")
        if results["status"] != "COMPLETED":
            failures.append(dict(run=scene.name, result=results))
            continue
        measure = read(scene / "measurement_window.json")
        origin, ended = measure["started_unix_s"], measure["ended_unix_s"]
        for role, ranks in (("source", 1), ("target", 2)):
            directory = scene / "controller/performance_trace" / role
            verify_pool(directory, ranks)
            batches = [
                r
                for r in read_journal(directory / "engine.jsonl")
                if r.get("kind") == "engine_batch"
                and origin <= r["scheduled_unix_s"] < ended
            ]
            tokens = []
            for request in results["requests"]:
                if request["pool"] != role:
                    continue
                path = scene / "controller" / (request["id"] + ".tokens.jsonl")
                events = [json.loads(line) for line in path.read_text().splitlines()]
                if [r["index"] for r in events] != list(range(request["tokens"])):
                    raise ValueError("noncontiguous or incomplete output evidence")
                times = [r["unix_s"] for r in events]
                if times != sorted(times):
                    raise ValueError("client output timestamps regressed")
                tokens.append(times)
            for row in batches:
                if row["recompute_tokens"] or row["speculative"]:
                    continue
                nd, kd, prefill = (
                    row[k]
                    for k in ("decode_requests", "decode_context_sum", "prefill_tokens")
                )
                key = (
                    role,
                    nd,
                    int(kd / nd // 1024) if nd else 0,
                    math.ceil(prefill / 256),
                )
                groups[key].append(dict(row, run=scene.name))
            if "-nd" not in scene.name:
                continue
            for index in range(int((ended - origin) // 10)):
                left, right = origin + 10 * index, origin + 10 * (index + 1)
                cohort = [r for r in batches if left <= r["scheduled_unix_s"] < right]
                if not cohort or any(
                    r["prefill_tokens"] or r["recompute_tokens"] or r["speculative"]
                    for r in cohort
                ):
                    continue
                stamps = [left, *sorted(r["scheduled_unix_s"] for r in cohort), right]
                if max(b - a for a, b in zip(stamps, stamps[1:])) > 2:
                    continue
                shapes = collections.Counter(
                    (
                        r["decode_requests"],
                        int(r["decode_context_sum"] / r["decode_requests"] // 2048),
                    )
                    for r in cohort
                    if r["decode_requests"] > 0
                )
                if not shapes:
                    continue
                (nd, cb), count = shapes.most_common(1)[0]
                if count / len(cohort) < 0.8:
                    continue
                windows[role, nd, cb].append(
                    dict(
                        run=scene.name,
                        left=left,
                        right=right,
                        **window_score(tokens, left, right),
                    )
                )
        for path in scene.rglob("*.jsonl"):
            provenance[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    service = []
    for (role, nd, cb, pb), rows in sorted(groups.items()):
        prefill = [r["prefill_tokens"] for r in rows]
        phase = (
            "decode"
            if pb == 0
            else "prefill_lt256"
            if max(prefill) < 256
            else "prefill_256_767"
            if max(prefill) < 768
            else "prefill_768_1535"
            if max(prefill) < 1536
            else "prefill_ge1536"
        )
        service.append(
            dict(
                pool=role,
                decode_requests=nd,
                mean_context_bin_1024=cb,
                prefill_bin_256=pb,
                phase=phase,
                samples=len(rows),
                runs=len({r["run"] for r in rows}),
                context_sum_range=stats([r["decode_context_sum"] for r in rows]),
                prefill_tokens_range=stats(prefill),
                engine_ms=stats([r["completion_latency_ms"] for r in rows]),
            )
        )
    cells = []
    for (role, nd, cb), rows in sorted(windows.items()):
        by_run = collections.defaultdict(list)
        for row in rows:
            by_run[row["run"]].append(row["good_s"])
        enough = len(by_run) >= 2 and all(len(v) >= 2 for v in by_run.values())
        cells.append(
            dict(
                pool=role,
                nd=nd,
                context_bin=cb,
                runs=len(by_run),
                health_status="REFERENCE" if enough else "DIAGNOSTIC_ONLY",
                health_reason="repeated empirical working point"
                if enough
                else "needs two complete windows in each of two runs",
                window_goodoutput=stats(
                    [statistics.median(v) for v in by_run.values()]
                ),
                windows=rows,
            )
        )
    supported = [c for c in cells if c["health_status"] == "REFERENCE"]
    workload = {}
    missing = []
    for role in ("source", "target"):
        capacities = []
        for cb in sorted({c["context_bin"] for c in supported if c["pool"] == role}):
            points = [
                c for c in supported if c["pool"] == role and c["context_bin"] == cb
            ]
            peak = max(c["window_goodoutput"]["median"] for c in points)
            capacities.append(
                [
                    cb,
                    min(
                        c["nd"]
                        for c in points
                        if c["window_goodoutput"]["median"] >= 0.9 * peak
                    ),
                ]
            )
        prefill = [
            c["engine_ms"]["p95"] / 1000 / c["prefill_tokens_range"]["min"]
            for c in service
            if c["pool"] == role
            and c["runs"] >= 2
            and c["samples"] >= 5
            and c["prefill_tokens_range"]["min"] >= 256
        ]
        workload[role] = dict(
            capacities=capacities or None,
            prefill_s_per_token=max(prefill) if prefill else None,
        )
        if not capacities or not prefill:
            missing.append(role + " workload/prefill support")
    return dict(
        schema="TP2_DEVELOPMENT_PROFILE_V1",
        target_tp=2,
        deployment_ready=False,
        failures=failures,
        workload=workload,
        service_lookup=service,
        cells=cells,
        input_sha256=provenance,
        migration_release_s=None,
        rebuild_ready_s=None,
        delta_g=None,
        C_pre=None,
        C_post=None,
        missing=missing
        + [
            "TP1-to-TP2 source-release profile",
            "bundle rebuild",
            "matched layout utility/transition costs",
        ],
        note="small empirical grid; no maximum-capacity or reliable P95 guarantee",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(output=str(args.output), missing=result["missing"])))
