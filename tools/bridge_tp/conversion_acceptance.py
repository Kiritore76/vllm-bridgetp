# SPDX-License-Identifier: Apache-2.0
"""S3b-4: acceptance and per-phase statistics of one live TP conversion.

Reads only the scene's own records: ``conversion_events.jsonl`` (backend
timeline), the controller's audit, drain observations and visible-token
journal, the instance manifest and, if present, per-request metrics.

Mechanism checks decide pass/fail. Service is split by when a request
arrived (before routing closed, while converting, after publication) and the
conversion window is reported on its own, so the drain-period cost is never
averaged into the whole run. Nothing here judges whether converting was
worthwhile.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

PHASES = ("before_quiesce", "converting", "after_publish")
# Output continuity contract used by every BridgeTP evaluation.
SLOW_ITL_S, MAX_GAP_S = 0.1, 1.0


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text("utf-8").splitlines()]


def _round(value, digits=3):
    return None if value is None else round(value, digits)


def assess(scene_dir, run_dir=None, events_path=None, metrics_path=None):
    """Acceptance record of the conversion that ran in ``scene_dir``."""
    scene_dir = Path(scene_dir)
    run_dir = Path(run_dir) if run_dir else scene_dir / "controller"
    events = _jsonl(events_path or scene_dir / "conversion_events.jsonl")
    audit = _jsonl(run_dir / "pressure_audit.jsonl")
    manifest = json.loads((scene_dir / "instances.json").read_text("utf-8"))
    metrics_path = Path(metrics_path or scene_dir / "comparison_metrics.json")
    metrics = (
        json.loads(metrics_path.read_text("utf-8")) if metrics_path.exists() else None
    )
    kinds = defaultdict(list)
    for event in events:
        kinds[event["kind"]].append(event)

    def at(kind):
        rows = kinds.get(kind)
        return rows[0]["unix_s"] if rows else None

    trigger = at("TRIGGER")
    sent = {e["revision"]: e for e in kinds["LAYOUT_CONTROL"]}
    applied = {
        r["revision"]: r["unix_s"]
        for r in audit
        if r["kind"] == "pressure_layout_applied" and r["changes"]
    }
    quiesce = applied.get(1)
    published = at("ROUTE_PUBLISHED")
    retired = list(sent[1]["changes"]) if 1 in sent else []
    new = kinds["ROUTE_PUBLISHED"][0]["instances"] if published else []
    pool_of = {
        r.get("legacy_pool") or r["instance_id"]: r["instance_id"]
        for r in manifest["instances"]
    }

    def instance(pool):
        return pool_of.get(pool, pool)

    def span(a, b):
        return _round(b - a) if a is not None and b is not None else None

    load = {e["instance"]: e["unix_s"] for e in kinds["LOAD_START"]}
    verified = {e["instance"]: e["unix_s"] for e in kinds["VERIFIED"]}
    retire = kinds["CHANNEL_RETIRE"][0] if kinds["CHANNEL_RETIRE"] else {}
    phases = dict(
        trigger_to_quiesce_applied=span(trigger, quiesce),
        drain=span(quiesce, at("FENCED")),
        fenced_to_controller_retired=span(at("FENCED"), applied.get(2)),
        channel_retire=span(at("CHANNEL_PLAN_WRITTEN"), at("CHANNEL_RETIRE")),
        stop_to_released=span(at("CHANNEL_RETIRE"), at("RESOURCES_RELEASED")),
        rebuild={i: span(load[i], verified.get(i)) for i in load},
        verified_to_published=span(max(verified.values(), default=None), published),
        quiesce_to_published=span(quiesce, published),
    )
    controller_apply_delay = {
        str(rev): _round(applied[rev] - e["unix_s"])
        for rev, e in sent.items()
        if rev in applied
    }

    # Admission by phase (the controller's own decision time).
    def phase(t):
        if quiesce is None or t < quiesce:
            return "before_quiesce"
        if published is None or t < published:
            return "converting"
        return "after_publish"

    dispatch = [r for r in audit if r["kind"] == "pressure_dispatch"]
    phase_of_job = {}
    routes = {p: Counter() for p in PHASES}
    for r in dispatch:
        p = phase(r["admission_finished_unix_s"])
        phase_of_job[r["job_id"]] = p
        routes[p][instance(r["pool"])] += 1
    late = sorted(
        r["job_id"]
        for r in dispatch
        if quiesce is not None
        and r["admission_finished_unix_s"] > quiesce
        and instance(r["pool"]) in retired
    )
    first_new = {}
    for r in sorted(dispatch, key=lambda r: r["admission_finished_unix_s"]):
        pool = instance(r["pool"])
        if pool in new and pool not in first_new:
            first_new[pool] = _round(r["admission_finished_unix_s"] - published)

    # Drain: peak streams still relayed per converted instance while draining.
    inflight = {}
    for check in _jsonl(run_dir / "conversion_drain_observations.jsonl"):
        for r in check["rows"]:
            inflight[r["instance_id"]] = max(
                inflight.get(r["instance_id"], 0), r["streams_pending"]
            )

    # Output continuity inside the conversion window, for every stream.
    tokens = defaultdict(list)
    for row in _jsonl(run_dir / "native_tokens.jsonl"):
        if row.get("kind") == "visible_token":
            tokens[row["job_id"]].append(row["unix_s"])
    window_gaps, window_slow, window_intervals = [], 0, 0
    if quiesce is not None and published is not None:
        for stamps in tokens.values():
            stamps.sort()
            gaps = [
                b - a for a, b in zip(stamps, stamps[1:]) if quiesce <= b <= published
            ]
            if gaps:
                window_gaps.append(max(gaps))
                window_slow += sum(g > SLOW_ITL_S for g in gaps)
                window_intervals += len(gaps)
    window = dict(
        streams=len(window_gaps),
        intervals=window_intervals,
        slow_intervals=window_slow,
        max_gap_s=_round(max(window_gaps, default=None)),
    )

    # Per-request service, grouped by admission phase (needs metrics).
    service = None
    if metrics is not None:
        service = {}
        for p in PHASES:
            rows = [
                r for r in metrics["requests"] if phase_of_job.get(r["job_id"]) == p
            ]
            ttft = sorted(r["ttft_s"] for r in rows if r.get("ttft_s") is not None)
            service[p] = dict(
                requests=len(rows),
                slo_pass=sum(r["slo_pass"] for r in rows),
                max_itl_s=_round(max((r["max_itl_s"] for r in rows), default=None)),
                ttft_p95_s=_round(ttft[int(0.95 * (len(ttft) - 1))] if ttft else None),
                failures=[r["job_id"] for r in rows if not r["slo_pass"]],
            )

    result = json.loads((scene_dir / "conversion_result.json").read_text("utf-8"))
    final = [r for r in audit if r["kind"] == "pressure_final"]
    checks = dict(
        plan_completed=result["state"] == "COMPLETED",
        controller_completed=bool(final) and final[-1].get("outcome") == "COMPLETED",
        quiesce_applied=quiesce is not None,
        no_admission_to_retired_after_quiesce=not late,
        channels_closed_before_stop=bool(retire)
        and at("CHANNEL_RETIRE") <= (at("STOP_RETURN") or float("inf")),
        channel_teardown_complete=retire.get("teardown_complete", True),
        new_instances_published=bool(new),
        # Routing resumed into the new layout (only meaningful when anything
        # arrived after publication); per-instance first times are reported.
        new_layout_serving=bool(first_new) or not routes["after_publish"],
        window_continuity=window["max_gap_s"] is None
        or window["max_gap_s"] <= MAX_GAP_S,
    )
    return dict(
        schema="BRIDGETP_CONVERSION_ACCEPTANCE_V1",
        passed=all(checks.values()),
        checks=checks,
        plan_state=result["state"],
        bundle=(kinds["TRIGGER"][0]["request"] if trigger else {}).get("bundle"),
        retired=retired,
        new_instances=new,
        drain_inflight_peak=inflight,
        phases_s=phases,
        controller_apply_delay_s=controller_apply_delay,
        channel_retire={
            k: retire.get(k)
            for k in ("closed", "teardown", "teardown_complete", "teardown_wait_s")
            if k in retire
        },
        routes_by_phase={p: dict(c) for p, c in routes.items()},
        admitted_to_retired_after_quiesce=late,
        first_request_after_publish_s=first_new,
        conversion_window=window,
        service_by_arrival_phase=service,
        scope="one conversion; mechanism checks decide pass; service is "
        "reported by phase, not judged",
    )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    value = assess(args.scene_dir)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, "utf-8")
    print(text, end="")
    return 0 if value["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
