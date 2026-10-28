# SPDX-License-Identifier: Apache-2.0
"""Small, explicit diagnostics for stager readiness and matched TP2 control batches."""

import json
import time

from tools.bridge_tp.qwen3_online_validation import background_completion


def wait_stager_ready(directory, process, timeout=60):
    deadline = time.monotonic() + timeout
    path = directory / "stager_process_ready.json"
    while time.monotonic() < deadline:
        if process.process.poll() is not None:
            raise RuntimeError("stager exited before process-ready gate")
        if path.exists():
            row = json.loads(path.read_text())
            if (
                row.get("status") != "STAGER_PROCESS_READY"
                or row.get("pid") != process.process.pid
            ):
                raise ValueError("stager process-ready identity differs")
            row["spawn_to_ready_s"] = row["unix_s"] - process.started_unix_s
            row["observed_unix_s"] = time.time()
            (directory / "stager_startup_gate.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
            return row
        time.sleep(0.05)
    raise TimeoutError("stager process-ready gate timed out")


def control_batch(io, url, inputs, directory, pool, count, phase):
    from tools.bridge_tp.run_tp2_migration_smoke import migration_request

    started = time.time()
    futures = [
        pool.submit(
            background_completion,
            io,
            url,
            migration_request(inputs, 1024, f"diagnostic-{phase}-{i}"),
            1200,
            directory,
            f"diagnostic-{phase}-{i}",
        )
        for i in range(count)
    ]
    results = [f.result(timeout=1200) for f in futures]
    if any(
        len(r["token_ids"]) != 1024 or r["finish_reason"] != "length" for r in results
    ):
        raise RuntimeError("ordinary target control batch incomplete")
    receipt = dict(
        phase=phase,
        requests=count,
        prompt_tokens=2048,
        output_tokens=1024,
        started_unix_s=started,
        ended_unix_s=time.time(),
        responses=results,
        scope="same-engine before/after control; contexts differ from migrated stream",
    )
    (directory / f"target_control_{phase}.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


def stream_quality(events):
    if len(events) < 2 or [r["index"] for r in events] != list(range(len(events))):
        raise ValueError("incomplete or noncontiguous output")
    gaps = [b["unix_s"] - a["unix_s"] for a, b in zip(events, events[1:])]
    if min(gaps) < 0:
        raise ValueError("output timestamps regressed")
    slow = sum(g > 0.1 for g in gaps) / len(gaps)
    return dict(
        mean_interval_ms=sum(gaps) / len(gaps) * 1000,
        slow_fraction=slow,
        maximum_gap_s=max(gaps),
        passed=slow <= 0.01 and max(gaps) <= 1,
    )


def performance_report(directory, result):
    controller = directory / "mechanism/controller"
    unified = json.loads((controller / "unified_tokens.json").read_text())
    background = []
    for i in range(result.get("target_background_requests", 0)):
        events = [
            json.loads(x)
            for x in (controller / f"background-{i}.tokens.jsonl")
            .read_text()
            .splitlines()
        ]
        background.append(stream_quality(events))
    stream = stream_quality(unified)
    handoff = result["visible_handoff_gap_s"]
    controls = [
        dict(
            name=path.name,
            **stream_quality([json.loads(x) for x in path.read_text().splitlines()]),
        )
        for path in sorted(controller.glob("background-diagnostic-*.tokens.jsonl"))
    ]
    return dict(
        handoff_s=handoff,
        handoff_limit_s=1.0,
        handoff_pass=handoff <= 1,
        unified=stream,
        background=background,
        controls=controls,
        passed=handoff <= 1
        and stream["passed"]
        and all(r["passed"] for r in background + controls),
    )
