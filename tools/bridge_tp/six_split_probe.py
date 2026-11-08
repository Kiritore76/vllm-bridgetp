# SPDX-License-Identifier: Apache-2.0
"""Pinned-arrival split lifecycle probe; normal migration policy stays active."""

from tools.bridge_tp import qwen3_long_burst_pair as burst
from tools.bridge_tp import qwen3_stay_migrate as matched

PROFILE = "six_split_lifecycle_v1"
INITIAL_A = "A-g3-tp2"
NEW_A = ("A-g5-tp1", "A-g6-tp1")
TARGET = "B-g2-tp2"
TRIGGER_S = 110.0
HORIZON_S = 720.0


def recipe(inputs):
    jobs = []

    def add(at, prompt, cap, route, phase):
        jobs.append(
            dict(
                id=f"{PROFILE}-{len(jobs)}",
                at_s=at,
                route=route,
                request=burst.request(inputs, prompt, cap),
                workload_role="short_probe" if cap == 128 else "long_input_output",
                workload_phase=phase,
            )
        )

    for at in range(0, int(HORIZON_S), 2):
        add(at, 512, 128, None, "background")
    for at in (30, 32):
        add(at, 16384, 8192, INITIAL_A, "loaded_drain")
    # Stagger source pressure because migrations are globally serialized.
    for source, first in zip(NEW_A, (330, 450)):
        for at in (first, first + 2):
            add(at, 16384, 8192, source, "joined_source_pressure")
    jobs.sort(key=lambda row: (row["at_s"], row["id"]))
    return dict(
        name=PROFILE,
        scenario=PROFILE,
        repetition=1,
        arm_order_offset=0,
        fixed_output=True,
        jobs=jobs,
        input_sha256=matched.digest(jobs),
        planned_output_tokens=sum(j["request"]["max_tokens"] for j in jobs),
        routing_scope="diagnostic pinned long arrivals; background unified; "
        "normal migration triggers and safety gates; no performance comparison",
    )


def arrival_limits(manifest):
    """Declare future arrival identities, without registering serving instances."""
    rows = {r["instance_id"]: r for r in manifest["instances"]}
    expected = {INITIAL_A: 2, TARGET: 2, "C-g1-tp1": 1, "C-g4-tp1": 1}
    if set(rows) != set(expected) or any(
        rows[n]["tp_degree"] != tp or rows[n]["context_limit"] != 32768
        for n, tp in expected.items()
    ):
        raise ValueError("split probe needs the canonical initial L2C identities")
    if max(r["instance_generation"] for r in rows.values()) != 4:
        raise ValueError("split probe cannot predict replacement generations")
    return dict.fromkeys((INITIAL_A, *NEW_A), 32768)


def configure(plan):
    arrival_limits(plan["instances"])
    if not plan.get("dynamic_conversion") or plan.get("scheduler") != "unified":
        raise ValueError("split probe requires dynamic unified six-card conversion")
    plan["instances"]["diagnostic_arrival_profile"] = PROFILE


def coverage(sequence, events, rows, starts, sessions):
    """Report observed ownership transfers separately from conversion correctness.

    Session rows come from the runner's closed-loop verifier, which checks both
    READY ranks, source abort/release and IDLE receipts before writing them.
    Match those rows back to actual START and the final request's source ID.
    """
    by_request = {r["server_id"]: r for r in rows}
    by_migration = {r["migration_id"]: r for r in sessions}
    transfers = []
    for start in starts:
        session = by_migration.get(start["migration_id"], {})
        request = by_request.get(start["request_id"])
        release = session.get("source_release") or {}
        target = start.get("target_epoch") or start.get("target_instance")
        if (
            request is not None
            and session.get("state") == "COMMITTED"
            and session.get("settled") is True
            and release.get("status") == "SOURCE_KV_RELEASED"
            and release.get("request_id") == start["request_id"]
            and session.get("target_instance") == target
        ):
            transfers.append(
                dict(
                    migration_id=start["migration_id"],
                    request_id=start["request_id"],
                    source=request["initial_pool"],
                    target=target,
                    started_unix_s=start["unix_s"],
                )
            )
    steps = sequence.get("steps", [])
    split = steps[0] if len(steps) == 1 else {}
    group = [e for e in events if e.get("plan_id") == split.get("plan_id")]
    closed = next(
        (
            e["unix_s"]
            for e in group
            if e["kind"] == "LAYOUT_CONTROL"
            and e.get("changes", {}).get(INITIAL_A) == "QUIESCING"
        ),
        None,
    )
    fenced = next((e["unix_s"] for e in group if e["kind"] == "FENCED"), None)
    drained = []
    for r in rows:
        emitted = r["response"]["emitted"]
        if (
            r["initial_pool"] == INITIAL_A
            and len(r["request"]["prompt"]) >= 8192
            and r["request"]["max_tokens"] > 512
            and emitted
            and closed is not None
            and fenced is not None
            and emitted[0]["unix_s"] < closed < emitted[-1]["unix_s"] <= fenced
            and r["source_done"]
            and r["source_error"] is None
            and not r["response"].get("committed")
        ):
            drained.append(r["job_id"])
    joined = {
        t["source"]
        for t in transfers
        if t["source"] in NEW_A
        and t["target"] == TARGET
        and t["started_unix_s"] > split.get("completed_unix_s", float("inf"))
        and split.get("active_instances", {}).get(t["source"], {}).get("tp_degree") == 1
    }
    return dict(
        actual_committed_transfers=transfers,
        actual_migration_coverage=bool(transfers),
        loaded_drain_requests=drained,
        loaded_drain_coverage=bool(drained),
        joined_source_coverage={source: source in joined for source in NEW_A},
        split_probe_coverage=bool(drained) and joined == set(NEW_A),
    )
