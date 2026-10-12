#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay background requests for the Phase 9 CAP-0 reachability pilot.

The schedule is intentionally isolated from the controller.  This process may
know future arrivals because it is the workload generator; no schedule field
is exported through controller telemetry or placed in the controller run dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bridge_tp.background_schedule import (  # noqa: E402
    SCHEDULE_VERSION,
    ControllerEpochs,
    event_matches,
    execute_schedule,
    output_threshold,
)
from tools.bridge_tp.collector_resources import ResourceSampler  # noqa: E402
from vllm.bridge_tp.async_journal import AsyncJournal as BackgroundJournal  # noqa: E402
from vllm.bridge_tp.controller.online_io import (  # noqa: E402
    atomic_json_dump,
    post_streaming_completion,
)
from vllm.bridge_tp.controller.sampling_contract import (  # noqa: E402
    freeze_strict_greedy_sampling,
)


def anchor_output_threshold(event: str) -> int | None:
    return output_threshold(event)


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
        start_event = job.get("start_after_event")
        if start_event is not None and not isinstance(start_event, str):
            raise ValueError(f"job {job_id} has an unknown start event")
        if start_event not in (
            None, "M2_INITIAL_RATE", "ANCHOR_FIRST_OUTPUT"
        ) and anchor_output_threshold(start_event) is None:
            raise ValueError(f"job {job_id} has an unknown start event")
        if start_event == "M2_INITIAL_RATE" and (
            job["pool"] != "source"
        ):
            raise ValueError(f"job {job_id} event start requires a source job")
        if (isinstance(start_event, str)
                and anchor_output_threshold(start_event) is not None
                and job["pool"] != "target"):
            raise ValueError(f"job {job_id} anchor-output start requires target")
        request = job.get("request")
        if not isinstance(request, dict):
            raise ValueError(f"job {job_id} request must be an object")
        for required in ("model", "prompt", "max_tokens"):
            if required not in request:
                raise ValueError(f"job {job_id} request is missing {required}")
        if int(request["max_tokens"]) <= 0:
            raise ValueError(f"job {job_id} max_tokens must be positive")
    return value


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-url", default="http://127.0.0.1:8001")
    parser.add_argument("--target-url", default="http://127.0.0.1:8200")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--controller-audit-path", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def wait_for_controller_event(path: Path, event: str, timeout_s: float) -> float:
    """Return the recorded event epoch, even when it is observed much later."""
    event_matches({}, event)  # Validate before waiting for a file.
    origin = time.monotonic()
    epochs = ControllerEpochs(path, {event}, origin, time.time())
    deadline = origin + timeout_s
    while time.monotonic() < deadline:
        epochs.poll()
        if event in epochs.epochs:
            return epochs.epochs[event]
        time.sleep(0.02)
    raise TimeoutError(f"{event} event was not observed")


def wait_for_m2_initial_rate(path: Path, timeout_s: float) -> float:
    """Retain the existing M2 event helper for callers of this module."""
    return wait_for_controller_event(path, "M2_INITIAL_RATE", timeout_s)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.validate_only:
        print(f"valid CAP-0 background manifest: {len(manifest['jobs'])} jobs")
        return
    if (
        any(job.get("start_after_event") for job in manifest["jobs"])
        and args.controller_audit_path is None
    ):
        raise ValueError("event-start jobs require --controller-audit-path")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(manifest, out_dir / "background_manifest.json")
    event_path = out_dir / "background_events.jsonl"
    events = BackgroundJournal(event_path)
    resources = BackgroundJournal(out_dir / "collector_resources.jsonl")
    sampler = ResourceSampler(resources)
    start_monotonic = time.monotonic()
    start_unix_s = time.time()
    live = None
    if os.environ.get("BRIDGETP_TARGET_OBSERVATION") == "1":
        from vllm.bridge_tp.async_journal import AsyncJournal

        live = AsyncJournal(out_dir / "target_native_events.jsonl")

    def event(value: dict[str, Any]) -> None:
        record = {"unix_s": time.time(), **value}
        events.submit(record)

    def run_job(planned) -> dict[str, Any]:
        worker_monotonic, worker_unix_s = time.monotonic(), time.time()
        job = planned.job
        job_id = str(job["job_id"])
        start_after_s = float(job["start_after_s"])
        schedule_lag_s = max(0.0, time.monotonic() - planned.scheduled_monotonic)
        schedule_fields = {
            "schedule_version": SCHEDULE_VERSION,
            "schedule_epoch_unix_s": planned.epoch_unix_s,
            "scheduled_unix_s": planned.scheduled_unix_s,
            "dispatch_unix_s": planned.dispatch_unix_s,
            "worker_started_unix_s": worker_unix_s,
            "due_to_dispatch_s": max(
                0.0, planned.dispatch_monotonic - planned.scheduled_monotonic
            ),
            "dispatch_to_worker_s": max(
                0.0, worker_monotonic - planned.dispatch_monotonic
            ),
        }
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
        event({"kind": "job_start", "job_id": job_id, "pool": job["pool"],
               "start_after_s": start_after_s,
               "schedule_lag_s": schedule_lag_s, **schedule_fields})
        request_started_unix_s = time.time()
        schedule_fields["worker_to_http_call_s"] = (
            time.monotonic() - worker_monotonic
        )
        schedule_fields["http_call_started_unix_s"] = request_started_unix_s
        schedule_fields["http_timestamp_basis"] = (
            "CLIENT_CALL_BOUNDARY_NOT_SOCKET_SEND_OR_SERVER_RECEIPT"
        )
        schedule_lag_s = max(0.0, time.monotonic() - planned.scheduled_monotonic)
        if live and job["pool"] == "target":
            live.submit({"kind": "native_arrival", "job_id": job_id,
                         "unix_s": request_started_unix_s,
                         "prompt_tokens": len(request["prompt"])})
        token_times: list[float] = []

        def record_token(_index: int, _token_id: int, unix_s: float) -> None:
            if not token_times:
                event(
                    {
                        "kind": "job_first_token",
                        "job_id": job_id,
                        "pool": job["pool"],
                    }
                )
            if live and job["pool"] == "target" and token_times:
                live.submit({"kind": "native_itl", "job_id": job_id,
                             "unix_s": unix_s, "itl_s": unix_s - token_times[-1],
                             "tpot_reference_s": 0.1})
            token_times.append(unix_s)
        try:
            result = post_streaming_completion(
                base_url,
                request,
                args.request_timeout_s,
                record_token,
            )
            request_ended_unix_s = time.time()
            token_intervals_ms = [
                (current - previous) * 1000
                for previous, current in zip(token_times, token_times[1:])
            ]
            summary = {
                "job_id": job_id,
                "pool": job["pool"],
                "start_after_s": start_after_s,
                "schedule_lag_s": schedule_lag_s,
                **schedule_fields,
                "status": "COMPLETED",
                "response_id": result["response_id"],
                "finish_reason": result["finish_reason"],
                "output_tokens": len(result["token_ids"]),
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(result["token_ids"], separators=(",", ":"))
                    .encode("ascii")
                ).hexdigest(),
                "token_ids": result["token_ids"] if live else None,
                "request_started_unix_s": request_started_unix_s,
                "first_token_unix_s": token_times[0] if token_times else None,
                "last_token_unix_s": token_times[-1] if token_times else None,
                "request_ended_unix_s": request_ended_unix_s,
                "ttft_ms": (
                    (token_times[0] - request_started_unix_s) * 1000
                    if token_times
                    else None
                ),
                "e2e_ms": (request_ended_unix_s - request_started_unix_s) * 1000,
                "tpot_p50_ms": percentile(token_intervals_ms, 0.50),
                "tpot_p95_ms": percentile(token_intervals_ms, 0.95),
                "tpot_p99_ms": percentile(token_intervals_ms, 0.99),
                # Preserved for paired pre/Shadow/Bridge/post window analysis.
                "token_times_unix_s": token_times,
            }
            event({"kind": "job_end", **summary})
            atomic_json_dump(summary, out_dir / f"{job_id}.json")
            return summary
        except Exception as error:
            summary = {
                "job_id": job_id,
                "pool": job["pool"],
                "start_after_s": start_after_s,
                "schedule_lag_s": schedule_lag_s,
                **schedule_fields,
                "status": "FAILED",
                "request_started_unix_s": request_started_unix_s,
                "request_ended_unix_s": time.time(),
                "output_tokens": len(token_times),
                "token_times_unix_s": token_times,
                "failure_kind": "SERVICE_REQUEST_FAILURE",
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
    try:
        results = execute_schedule(
            manifest["jobs"], run_job, args.controller_audit_path,
            start_monotonic, start_unix_s,
        )
    except BaseException:
        # Response workers have drained before execute_schedule raises.
        try:
            if live:
                live.close()
        finally:
            sampler.close()
            resources.close()
            events.close()
        raise
    sampler.close()
    resources.close()
    evidence_error = None
    if live:
        try:
            live.close()
        except Exception as error:
            evidence_error = error
    summary = {
        "format_version": 1,
        "schedule_version": SCHEDULE_VERSION,
        "native_evidence_integrity": {
            "enabled": live is not None,
            "complete": evidence_error is None,
            "dropped": live.dropped if live else 0,
            "writer_error": str(live.error) if live and live.error else None,
            "error": str(evidence_error) if evidence_error else None,
        },
        "jobs": len(results),
        "completed": sum(item["status"] == "COMPLETED" for item in results),
        "failed": sum(item["status"] == "FAILED" for item in results),
        "start_unix_s": start_unix_s,
        "end_unix_s": time.time(),
        "latest_planned_arrival_s": max(float(job["start_after_s"])
                                        for job in manifest["jobs"]),
        "max_schedule_lag_s": max(item.get("schedule_lag_s", 0.0)
                                  for item in results),
        "results": sorted(results, key=lambda item: item["job_id"]),
    }
    atomic_json_dump(summary, out_dir / "background_summary.json")
    event({"kind": "run_end", **summary})
    try:
        events.close()
    except Exception as error:
        summary["background_event_integrity"] = {
            "complete": False, "error": str(error), "dropped": events.dropped,
        }
        atomic_json_dump(summary, out_dir / "background_summary.json")
        raise
    summary["background_event_integrity"] = {
        "complete": True, "dropped": events.dropped,
        "saved_records": events.saved,
    }
    atomic_json_dump(summary, out_dir / "background_summary.json")
    if evidence_error is not None:
        raise RuntimeError("background native evidence incomplete") from evidence_error
    if summary["failed"]:
        raise SystemExit(f"{summary['failed']} background jobs failed")
    print(f"CAP-0 background complete: {summary['completed']} jobs")


if __name__ == "__main__":
    main()
