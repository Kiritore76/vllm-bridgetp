# SPDX-License-Identifier: Apache-2.0
"""Check a small explicit A merge with drain recompute (L1AC -> L2C).

Exit 0: the merge completed, recompute was requested and every recomputed
request delivered its full, gap-free stream. Exit 2: the run is valid but did
not exercise recompute (for example more than the cap stayed on A). Exit 1:
technical failure or missing evidence. Timings are reported, never judged as
performance.
"""

import argparse
import json
import sys
from pathlib import Path


def jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def unique(root, name):
    found = list(root.rglob(name))
    if len(found) != 1:
        raise ValueError(f"expected one {name}, found {len(found)}")
    return found[0]


def check(directory):
    root = Path(directory)
    value = dict(passed=False, recompute_exercised=False, errors=[])
    try:
        scene = unique(root, "conversion_sequence_result.json").parent
        sequence = json.loads((scene / "conversion_sequence_result.json").read_text())
        steps = sequence.get("steps", [])
        value["steps"] = [
            dict(bundle=s["plan"].get("bundle_id"), state=s["plan"]["state"])
            for s in steps
        ]
        if not steps or any(s["plan"]["state"] != "COMPLETED" for s in steps):
            raise ValueError("the explicit merge did not complete")
        events = jsonl(scene / "conversion_events.jsonl")
        status = json.loads(unique(scene, "layout_status.json").read_text())
        origin = status.get("workload_origin_unix_s") or 0.0

        def at(kind, **match):
            rows = [
                e
                for e in events
                if e["kind"] == kind
                and all(e.get(k) == v for k, v in match.items())
            ]
            return round(rows[0]["unix_s"] - origin, 3) if rows else None

        value["timeline_s"] = dict(
            trigger=at("TRIGGER"),
            draining=at("PLAN_STATE", state="DRAINING"),
            recompute_requested=at("RECOMPUTE_REQUESTED"),
            fenced=at("FENCED"),
            route_published=at("ROUTE_PUBLISHED"),
            completed=at("PLAN_STATE", state="COMPLETED"),
        )
        decisions = jsonl(unique(scene, "conversion_decisions.jsonl"))
        recompute = [d for d in decisions if d["kind"] == "RECOMPUTE"]
        value["recompute_reasons"] = [
            dict(t_s=round(d["unix_s"] - origin, 3), reason=d["reason"])
            for d in recompute
            if d["reason"]
        ]
        requested = [d["choice"] for d in recompute if d.get("choice")]
        value["recompute_requests"] = requested
        audit = jsonl(unique(scene, "pressure_audit.jsonl"))
        value["controller_requests"] = [
            dict(seq=r["seq"], state=r["state"], reason=r["reason"])
            for r in audit
            if r.get("kind") == "drain_recompute_request"
        ]
        rows = json.loads(unique(scene, "pressure_request_results.json").read_text())
        jobs = []
        for row in rows:
            job = row.get("recompute")
            if job is None:
                continue
            emitted = row["response"]["emitted"]
            cap = row["request"]["max_tokens"]
            origins = [t["origin"] for t in emitted]
            k = job["cutover"]
            times = [t["unix_s"] for t in emitted]
            gaps = [b - a for a, b in zip(times, times[1:])]
            jobs.append(
                dict(
                    job_id=row["job_id"],
                    state=job["state"],
                    source_pool=job["source_pool"],
                    target_pool=job["target_pool"],
                    cutover=k,
                    max_tokens=cap,
                    output_tokens=len(emitted),
                    complete=len(emitted) == cap
                    and [t["index"] for t in emitted] == list(range(len(emitted))),
                    seam=k is None
                    or origins == ["source"] * k + ["target"] * (len(emitted) - k),
                    source_error=row["source_error"],
                    finish_reason=(row.get("source_transport") or {}).get(
                        "finish_reason"
                    ),
                    stall_s=row["response"].get("handoff_stall_s"),
                    max_gap_s=max(gaps, default=0.0),
                    requested_s=job["requested_at"] - origin,
                    source_ended_s=None
                    if job["source_ended_at"] is None
                    else job["source_ended_at"] - origin,
                    parked_s=None
                    if job["parked_at"] is None
                    else job["parked_at"] - origin,
                    resumed_s=None
                    if job["resumed_at"] is None
                    else job["resumed_at"] - origin,
                )
            )
        value["jobs"] = jobs
        value["requests"] = len(rows)
        value["incomplete_requests"] = sum(
            len(r["response"]["emitted"]) != r["request"]["max_tokens"]
            or r["source_error"] is not None
            for r in rows
        )
        if value["incomplete_requests"]:
            raise ValueError("a request did not deliver its full output")
        if not requested:
            value["passed"] = True
            value["errors"].append("recompute not requested: coverage missing")
            return value, 2
        bad = [
            j
            for j in jobs
            if j["state"] not in {"COMPLETED", "NATURAL"}
            or not j["complete"]
            or not j["seam"]
            or j["source_error"] is not None
        ]
        if bad or not any(j["state"] == "COMPLETED" for j in jobs):
            raise ValueError("a recomputed request did not complete its seam")
        value.update(passed=True, recompute_exercised=True)
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
    output = args.directory.parent / "recompute_smoke_check.json"
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
