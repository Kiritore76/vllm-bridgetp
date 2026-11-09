# SPDX-License-Identifier: Apache-2.0
"""Require actual decider-driven completion in the short six-card smoke run."""

import json
import sys
from pathlib import Path

from tools.bridge_tp.six_conversion import assess

SCENARIO = "auto_smoke_six_v1"
INPUT_SHA256 = "7c1519ba3700369d1607826f340bfa5ee397065619d4344abea97b179a4c57b2"


def completion_coverage(steps, events):
    completed = [
        s
        for s in steps
        if s["plan"]["state"] == "COMPLETED"
        and s.get("controller_acknowledged")
        and s.get("layout") == dict(A=2, B=2, C=1)
    ]
    return bool(completed and any(e["kind"] == "DECIDER_PROPOSAL" for e in events))


def check(directory):
    """Preserve an explicit technical failure if the runner produced no scene."""
    try:
        return check_complete(directory)
    except (OSError, ValueError, StopIteration, KeyError) as error:
        value = dict(
            passed=False,
            mechanism_passed=False,
            automatic_completion_coverage=False,
            performance_validated=False,
            status="TECHNICAL_FAILURE",
            error=f"{type(error).__name__}: {error}"
            if str(error)
            else "runner produced no conversion sequence result",
        )
        output = Path(directory).parent / "smoke_check.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value, indent=2) + "\n")
        print(json.dumps(value, indent=2))
        return 1


def check_complete(directory):
    root = Path(directory)
    scene = next(root.rglob("conversion_sequence_result.json")).parent
    # Frozen actual arrival list is the runner input, not a guessed count.
    arrivals = next(root.glob("*_arrivals.json"))
    jobs = json.loads(arrivals.read_text())
    from tools.bridge_tp.qwen3_stay_migrate import digest

    if digest(jobs) != INPUT_SHA256 or len(jobs) != 619:
        raise ValueError("smoke input differs from registered workload")
    value = assess(scene, len(jobs), require_conversion=False)
    sequence = json.loads((scene / "conversion_sequence_result.json").read_text())
    events = [
        json.loads(line)
        for line in (scene / "conversion_events.jsonl").read_text().splitlines()
    ]
    value["automatic_completion_coverage"] = completion_coverage(
        sequence["steps"], events
    )
    value["scenario"] = SCENARIO
    value["performance_validated"] = False
    value["request_count"] = len(jobs)
    value["planned_output_tokens"] = sum(v["request"]["max_tokens"] for v in jobs)
    value["assist_started"] = sum(e["kind"] == "ASSIST_REQUESTED" for e in events)
    # A request event is a proposal, not proof of migration START/COMMIT.
    value["assist_proposals"] = value.pop("assist_started")
    value["passed"] = bool(value["passed"] and value["automatic_completion_coverage"])
    shutdown = json.loads((scene / "owned_shutdown_receipt.json").read_text())
    value["cleanup_confirmed"] = shutdown["all_owned_groups_ended"]
    value["passed"] = value["passed"] and value["cleanup_confirmed"]
    (root.parent / "smoke_check.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))
    return 0 if value["passed"] else 2 if value.get("mechanism_passed") else 1


if __name__ == "__main__":
    raise SystemExit(check(sys.argv[1]))
