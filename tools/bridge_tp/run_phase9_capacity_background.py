#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay background requests for the Phase 9 CAP-0 reachability pilot.

The schedule is intentionally isolated from the controller.  This process may
know future arrivals because it is the workload generator; no schedule field
is exported through controller telemetry or placed in the controller run dir.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.bridge_tp.controller.online_io import (  # noqa: E402
    atomic_json_dump,
    post_streaming_completion,
)
from vllm.bridge_tp.controller.sampling_contract import (  # noqa: E402
    freeze_strict_greedy_sampling,
)


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError("capacity background manifest requires format_version=1")
    jobs = value.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("capacity background manifest requires a non-empty jobs list")
    identifiers: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"job {index} must be an object")
        job_id = str(job.get("job_id", "")).strip()
        if not job_id or job_id in identifiers:
            raise ValueError(f"job {index} has an empty or duplicate job_id")
        identifiers.add(job_id)
        if job.get("pool") not in {"source", "target"}:
            raise ValueError(f"job {job_id} pool must be source or target")
        if float(job.get("start_after_s", -1)) < 0:
            raise ValueError(f"job {job_id} start_after_s must be non-negative")
        request = job.get("request")
        if not isinstance(request, dict):
            raise ValueError(f"job {job_id} request must be an object")
        for required in ("model", "prompt", "max_tokens"):
            if required not in request:
                raise ValueError(f"job {job_id} request is missing {required}")
        if int(request["max_tokens"]) <= 0:
            raise ValueError(f"job {job_id} max_tokens must be positive")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-url", default="http://127.0.0.1:8001")
    parser.add_argument("--target-url", default="http://127.0.0.1:8200")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.validate_only:
        print(f"valid CAP-0 background manifest: {len(manifest['jobs'])} jobs")
        return

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(manifest, out_dir / "background_manifest.json")
    event_path = out_dir / "background_events.jsonl"
    event_lock = threading.Lock()
    start_monotonic = time.monotonic()
    start_unix_s = time.time()

    def event(value: dict[str, Any]) -> None:
        record = {"unix_s": time.time(), **value}
        with event_lock, event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def run_job(job: dict[str, Any]) -> dict[str, Any]:
        job_id = str(job["job_id"])
        start_after_s = float(job["start_after_s"])
        remaining = start_monotonic + start_after_s - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        base_url = args.source_url if job["pool"] == "source" else args.target_url
        request = freeze_strict_greedy_sampling(dict(job["request"]))
        request.update(
            {
                "request_id": f"bridgetp-cap0-load-{job_id}",
                "stream": True,
                "return_token_ids": True,
            }
        )
        request.setdefault("ignore_eos", True)
        event({"kind": "job_start", "job_id": job_id, "pool": job["pool"]})
        try:
            result = post_streaming_completion(
                base_url,
                request,
                args.request_timeout_s,
                lambda _index, _token_id, _unix_s: None,
            )
            summary = {
                "job_id": job_id,
                "pool": job["pool"],
                "status": "COMPLETED",
                "response_id": result["response_id"],
                "finish_reason": result["finish_reason"],
                "output_tokens": len(result["token_ids"]),
            }
            event({"kind": "job_end", **summary})
            atomic_json_dump(summary, out_dir / f"{job_id}.json")
            return summary
        except Exception as error:
            summary = {
                "job_id": job_id,
                "pool": job["pool"],
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
            }
            event({"kind": "job_end", **summary})
            atomic_json_dump(summary, out_dir / f"{job_id}.json")
            return summary

    event(
        {
            "kind": "run_start",
            "start_unix_s": start_unix_s,
            "jobs": len(manifest["jobs"]),
        }
    )
    with ThreadPoolExecutor(max_workers=len(manifest["jobs"])) as executor:
        futures = [executor.submit(run_job, job) for job in manifest["jobs"]]
        results = [future.result() for future in as_completed(futures)]
    summary = {
        "format_version": 1,
        "jobs": len(results),
        "completed": sum(item["status"] == "COMPLETED" for item in results),
        "failed": sum(item["status"] == "FAILED" for item in results),
        "start_unix_s": start_unix_s,
        "end_unix_s": time.time(),
        "results": sorted(results, key=lambda item: item["job_id"]),
    }
    atomic_json_dump(summary, out_dir / "background_summary.json")
    event({"kind": "run_end", **summary})
    if summary["failed"]:
        raise SystemExit(f"{summary['failed']} background jobs failed")
    print(f"CAP-0 background complete: {summary['completed']} jobs")


if __name__ == "__main__":
    main()
