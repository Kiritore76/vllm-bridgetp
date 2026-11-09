# SPDX-License-Identifier: Apache-2.0
"""Short current-length smoke also requires real traffic on the joined TP2."""

import json
import sys
from pathlib import Path

from tools.bridge_tp import six_auto_smoke


def joined_routing_coverage(steps, events, rows):
    completed = {s["plan_id"] for s in steps if s["plan"]["state"] == "COMPLETED"}
    joined = {
        instance: e["unix_s"]
        for e in events
        if e["kind"] == "ROUTE_PUBLISHED" and e["plan_id"] in completed
        for instance in e["instances"]
    }
    counts = {
        instance: sum(
            r["initial_pool"] == instance
            and r["source_done"]
            and r["source_error"] is None
            and r["source_transport"]["request_started_unix_s"] >= published
            for r in rows
        )
        for instance, published in joined.items()
    }
    return dict(passed=bool(counts) and all(counts.values()), requests=counts)


def check(directory):
    root = Path(directory)
    status = six_auto_smoke.check(root)
    value = json.loads((root.parent / "smoke_check.json").read_text())
    if status == 0:
        try:
            scene = next(root.rglob("conversion_sequence_result.json")).parent
            steps = json.loads((scene / "conversion_sequence_result.json").read_text())[
                "steps"
            ]
            events = [
                json.loads(s)
                for s in (scene / "conversion_events.jsonl").read_text().splitlines()
            ]
            rows = json.loads(
                (scene / "controller/pressure_request_results.json").read_text()
            )
            value["joined_routing"] = joined_routing_coverage(steps, events, rows)
            proposals = [e for e in events if e["kind"] == "ASSIST_REQUESTED"]
            value["current_length_proposals_valid"] = all(
                e.get("current_length") is True and not e.get("final_length")
                for e in proposals
            )
            starts = []
            for line in (scene / "controller/pressure_audit.jsonl").open():
                e = json.loads(line)
                if e["kind"] == "pressure_assist_result" and e["state"] == "STARTED":
                    starts.append(e)
            value["current_length_assist_starts"] = len(starts)
            value["assist_start_coverage"] = bool(starts)
            value["current_length_starts_valid"] = all(
                e.get("length_basis") == "current" for e in starts
            )
            value["passed"] = bool(
                value["joined_routing"]["passed"]
                and value["current_length_proposals_valid"]
                and value["current_length_starts_valid"]
            )
            status = 0 if value["passed"] else 2
        except (OSError, ValueError, KeyError, StopIteration) as error:
            value.update(passed=False, status="TECHNICAL_FAILURE", error=repr(error))
            status = 1
    value["performance_validated"] = False
    (root.parent / "current_assist_smoke_check.json").write_text(
        json.dumps(value, indent=2) + "\n"
    )
    print(json.dumps(value, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(check(sys.argv[1]))
