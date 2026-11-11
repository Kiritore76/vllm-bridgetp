# SPDX-License-Identifier: Apache-2.0
"""Six-card engine-reuse smoke: A merge, split and merge again (sleep/wake).

L1AC start. 110 s: A merges to TP2; no sleeping TP2 exists yet, so it is cold
started and the two A TP1 engines sleep. 240 s: A splits; the same two TP1
engines wake and the A TP2 sleeps. 440 s: A merges again; the sleeping TP2
wakes. Two long bursts (90-100 s and 420-430 s) put long requests on A before
each merge, so drain recompute continues them on the new (then woken) TP2.

Exit 0: all three steps completed, both reverse steps woke the same engines
without any cold start or fallback, and every request delivered its full
output. Exit 2: valid, but no recompute or migration touched a woken engine
(coverage missing). Exit 1: technical failure or missing evidence. Timings
are a mechanism record, not a performance result.
"""

import argparse
import json
import sys
from pathlib import Path

from tools.bridge_tp import qwen3_long_burst_pair as burst
from tools.bridge_tp import qwen3_stay_migrate as matched

PROFILE = "six_sleep_reuse_smoke_v1"
HORIZON_S = 720.0
TRIGGER_S = 110.0
SEQUENCE = (("TP2", 110.0), ("TP1", 240.0), ("TP2", 440.0))


def recipe(inputs):
    jobs = []

    def add(at, prompt, cap, role, phase):
        jobs.append(
            dict(
                id=f"{PROFILE}-{len(jobs)}",
                at_s=at,
                route=None,
                request=burst.request(inputs, prompt, cap),
                workload_role=role,
                workload_phase=phase,
            )
        )

    for at in range(0, 600, 2):
        add(at, 512, 128, "short_probe", "background")
    for i in range(6):
        add(90 + 2 * i, 16384, 8192, "long_input_output", "burst_before_merge")
    for i in range(6):
        add(420 + 2 * i, 16384, 4096, "long_input_output", "burst_before_remerge")
    jobs.sort(key=lambda j: (j["at_s"], j["id"]))
    return dict(
        name=PROFILE,
        scenario=PROFILE,
        repetition=1,
        arm_order_offset=0,
        fixed_output=True,
        jobs=jobs,
        input_sha256=matched.digest(jobs),
        planned_output_tokens=sum(j["request"]["max_tokens"] for j in jobs),
        routing_scope="unified scheduler, unpinned; explicit A merge/split/merge "
        "with engine reuse; mechanism check, no performance comparison",
    )


def sequence():
    return [
        dict(
            bundle="A",
            target_mode=mode,
            trigger_s=at,
            deadline_s=900,
            method="DRAIN",
        )
        for mode, at in SEQUENCE
    ]


def jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def unique(root, name):
    found = list(root.rglob(name))
    if len(found) != 1:
        raise ValueError(f"expected one {name}, found {len(found)}")
    return found[0]


def step_record(step, events, origin):
    begin, end = step["triggered_unix_s"], step["completed_unix_s"]
    pid = step["plan_id"]
    inside = [e for e in events if begin - 1 <= e["unix_s"] <= end + 1]

    def at(kind, **match):
        rows = [
            e
            for e in inside
            if e["kind"] == kind and all(e.get(k) == v for k, v in match.items())
        ]
        return rows[0]["unix_s"] if rows else None

    def rel(value):
        return None if value is None else round(value - origin, 3)

    slept = [e for e in inside if e["kind"] == "SLEPT"]
    woke = [e for e in inside if e["kind"] == "WOKEN"]
    fenced = at("FENCED", plan_id=pid)
    released = at("RESOURCES_RELEASED", plan_id=pid)
    published = at("ROUTE_PUBLISHED", plan_id=pid)
    return dict(
        plan_id=pid,
        target_mode=step["plan"]["target_bundle_mode"],
        state=step["plan"]["state"],
        layout=step.get("layout"),
        trigger_s=rel(begin),
        fenced_s=rel(fenced),
        released_s=rel(released),
        route_published_s=rel(published),
        # Admission closed until routing reopens, and the part after drain.
        fence_to_route_s=None
        if None in (fenced, published)
        else round(published - fenced, 3),
        release_to_route_s=None
        if None in (released, published)
        else round(published - released, 3),
        slept=[e["instances"] for e in slept],
        sleep_s=[e.get("sleep_s") for e in slept],
        sleep_wait_s=[e.get("wait_s") for e in slept],
        residual_mib=[e.get("used_mib") for e in slept],
        cold_started=[e["instance"] for e in inside if e["kind"] == "LOAD_START"],
        woken=sorted(e["instance"] for e in inside if e["kind"] == "WAKE_START"),
        wake_s={e["instance"]: e.get("wake_s") for e in woke},
        fallbacks=[e for e in inside if e["kind"] == "SLEEP_FALLBACK"],
    )


def check(directory):
    root = Path(directory)
    value = dict(passed=False, reuse_exercised=False, errors=[])
    try:
        scene = unique(root, "conversion_sequence_result.json").parent
        result = json.loads((scene / "conversion_sequence_result.json").read_text())
        steps = result.get("steps", [])
        events = jsonl(scene / "conversion_events.jsonl")
        status = json.loads(unique(scene, "layout_status.json").read_text())
        origin = status.get("workload_origin_unix_s") or 0.0
        records = [step_record(s, events, origin) for s in steps]
        value["steps"] = records
        if [r["target_mode"] for r in records] != [m for m, _ in SEQUENCE] or any(
            r["state"] != "COMPLETED" for r in records
        ):
            raise ValueError("the three explicit A steps did not all complete")
        merge, split, remerge = records
        if any(r["fallbacks"] for r in records):
            raise ValueError("a sleep fell back to stop/cold start")
        tp1 = sorted(i for group in merge["slept"] for i in group)
        if len(tp1) != 2 or len(merge["cold_started"]) != 1:
            raise ValueError("first merge did not cold-start one TP2 and sleep A")
        tp2 = merge["cold_started"]
        if split["woken"] != tp1 or split["cold_started"]:
            raise ValueError("split did not wake the same two A TP1 engines")
        if sorted(i for g in split["slept"] for i in g) != tp2:
            raise ValueError("split did not sleep the A TP2")
        if remerge["woken"] != tp2 or remerge["cold_started"]:
            raise ValueError("re-merge did not wake the sleeping A TP2")
        if sorted(i for g in remerge["slept"] for i in g) != tp1:
            raise ValueError("re-merge did not sleep the A TP1 engines again")
        if any(
            value is None or value is False
            for r in (split, remerge)
            for value in r["wake_s"].values()
        ) or set(split["wake_s"]) != set(tp1) or set(remerge["wake_s"]) != set(tp2):
            raise ValueError("a wake has no WOKEN receipt")
        acceptance = json.loads(
            unique(root, "conversion_sequence_acceptance.json").read_text()
        )
        value["sequence_acceptance_passed"] = acceptance.get("mechanism_passed")
        if not acceptance.get("mechanism_passed"):
            raise ValueError("six-card sequence acceptance failed")
        rows = json.loads(unique(scene, "pressure_request_results.json").read_text())
        value["requests"] = len(rows)
        value["incomplete_requests"] = sum(
            len(r["response"]["emitted"]) != r["request"]["max_tokens"]
            or r["source_error"] is not None
            for r in rows
        )
        if value["incomplete_requests"]:
            raise ValueError("a request did not deliver its full output")
        woken_at = {i: split["route_published_s"] for i in tp1}
        woken_at.update({i: remerge["route_published_s"] for i in tp2})
        recomputed = [
            dict(
                job_id=r["job_id"],
                state=r["recompute"]["state"],
                source=r["recompute"]["source_pool"],
                target=r["recompute"]["target_pool"],
            )
            for r in rows
            if r.get("recompute")
        ]
        value["recompute_jobs"] = recomputed
        audit = jsonl(unique(scene, "pressure_audit.jsonl"))
        starts = [
            dict(
                t_s=round(e["unix_s"] - origin, 3),
                request_id=e.get("request_id"),
                target=e.get("target_instance"),
            )
            for e in audit
            if e.get("kind") == "pressure_actual_start" and "unix_s" in e
        ]
        value["migration_starts"] = starts
        # The audit names a migration's target only: a woken TP2 receiving
        # KV, or a recompute continued on it, exercises the woken receiver.
        touched = [
            j for j in recomputed if j["target"] in tp2 and j["state"] == "COMPLETED"
        ] + [
            s
            for s in starts
            if s["target"] in woken_at
            and woken_at[s["target"]] is not None
            and s["t_s"] >= woken_at[s["target"]]
        ]
        value["woken_engine_work"] = touched
        value["passed"] = True
        if not touched:
            value["errors"].append(
                "no recompute or migration touched a woken engine: coverage missing"
            )
            return value, 2
        value["reuse_exercised"] = True
        return value, 0
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        value["errors"].append(repr(error))
        return value, 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    value, code = check(args.directory)
    value["exit_code"] = code
    value["scope"] = "mechanism smoke; timings are not a performance result"
    output = args.directory.parent / "sleep_smoke_check.json"
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
