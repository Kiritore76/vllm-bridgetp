# SPDX-License-Identifier: Apache-2.0
"""Export completed request evidence with bounded per-request materialization."""

import json
import os
from pathlib import Path

from tools.bridge_tp.pressure_cpu import load


def write_request_results(path, dispatched, jobs, origin_unix_s):
    """Retain the old JSON payload; drain diagnostics between request records.

    Keep a partial file on failure and never publish an incomplete final array.
    This runs after the workload; online GC policy and scheduling are unchanged.
    """
    diagnostics = load("pause_diagnostics")
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    planned = {job["id"]: job["at_s"] for job in jobs}
    count = 0
    with diagnostics.span("shutdown.request_results"):
        with partial.open("w", encoding="utf-8") as output:
            output.write("[")
            for job_id, value in dispatched.items():
                row = dict(
                    job_id=job_id,
                    client_id=value.client_id,
                    server_id=value.server_id,
                    initial_pool=value.pool_id,
                    request=value.payload,
                    source_done=value.future.done(),
                    source_error=str(value.future.exception())
                    if value.future.done() and value.future.exception()
                    else None,
                    response=value.recorder.stats(),
                    source_transport={
                        key: value.future.result().get(key)
                        for key in (
                            "request_started_unix_s",
                            "completed_unix_s",
                            "ttft_ms",
                            "e2e_ms",
                            "finish_reason",
                        )
                    }
                    if value.future.done() and value.future.exception() is None
                    else None,
                    dispatch_timing=dict(
                        planned_unix_s=origin_unix_s + planned[job_id],
                        dispatch_entered_unix_s=value.dispatch_entered_at,
                        admission_finished_unix_s=value.admission_finished_at,
                        executor_submitted_unix_s=value.executor_submitted_at,
                        http_thread_started_unix_s=value.http_thread_started_at,
                        held_s=value.held_s,
                    ),
                )
                diagnostics.flush_gc()
                if count:
                    output.write(", ")
                output.write(json.dumps(row))
                del row
                diagnostics.flush_gc()
                count += 1
            output.write("]")
        diagnostics.flush_gc()
        os.replace(partial, path)
    return dict(
        status="WRITTEN",
        requests=count,
        materialization="one_request",
        gc_evidence="flushed_between_records",
    )
