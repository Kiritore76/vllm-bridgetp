# SPDX-License-Identifier: Apache-2.0
"""Read-only, bounded prefix search inside an explicit manual conversion.

Inputs are already bounded remaining-duration distributions, never future trace
events. An adapter must produce these from causal predictions and speed profiles.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RemainingRequest:
    request_id: str
    # Seconds FROM this snapshot, including queue/prefill; None means unknown.
    natural_s: float | None
    # Conservative discrete duration upper bounds and their probability masses.
    duration_pmf: tuple[tuple[float, float], ...] | None
    migratable: bool
    # START through source free and mandatory channel cleanup, not handoff.
    transfer_s: float | None
    transfer_bytes: int | None
    # Measured support for delaying START, including history growth and target
    # conditions. Zero supports immediate START only; not an infinite envelope.
    transfer_supported_delay_s: float = 0


@dataclass(frozen=True)
class ManualPlan:
    plan_id: str
    bundle_id: str
    target_tp: int
    objective_mode: str
    execution_mode: str
    deadline: float
    utility_end: float
    method: str = "SELECTIVE_ROLL"


def remaining_duration(remaining_upper, speed_lower, queue_prefill_s):
    values = (remaining_upper, speed_lower, queue_prefill_s)
    if any(v is None or not math.isfinite(v) for v in values):
        return None
    if remaining_upper < 0 or speed_lower <= 0 or queue_prefill_s < 0:
        return None
    return queue_prefill_s + remaining_upper / speed_lower


def overdue(request, deadline):
    if request.duration_pmf is None:
        return None
    pmf = request.duration_pmf
    if (
        not pmf
        or any(
            not math.isfinite(t) or not math.isfinite(p) or t < 0 or p < 0
            for t, p in pmf
        )
        or not math.isclose(sum(p for _, p in pmf), 1, abs_tol=1e-6)
    ):
        raise ValueError("invalid bounded duration PMF")
    return sum(p for t, p in pmf if t > deadline)


def timeline(requests, ordered_ids, rebuild_s, *, channel_busy_s=0, settlement_s=0):
    by_id = {r.request_id: r for r in requests}
    if len(by_id) != len(requests) or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("duplicate request identity")
    for row in requests:
        for value in (row.natural_s, row.transfer_s, row.transfer_bytes):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("invalid time/byte estimate")
    if any(v is None for v in (rebuild_s, channel_busy_s, settlement_s)):
        return dict(
            support_status="UNKNOWN",
            missing_inputs=["stage profiles"],
            estimated_clear_time=None,
            estimated_ready_time=None,
            timeline=[],
            transfer_bytes=None,
        )
    if any(
        not math.isfinite(v) or v < 0 for v in (rebuild_s, channel_busy_s, settlement_s)
    ):
        raise ValueError("invalid stage duration")
    waiting = [r for r in requests if r.request_id not in ordered_ids]
    unknown = [r.request_id for r in waiting if r.natural_s is None]
    cursor = channel_busy_s
    transfers = []
    total_bytes = 0
    for request_id in ordered_ids:
        r = by_id[request_id]
        if (
            not r.migratable
            or r.transfer_s is None
            or r.transfer_bytes is None
            or cursor > r.transfer_supported_delay_s
        ):
            unknown.append(request_id)
            continue
        # Adapter must include growth/support at this delayed start. This simple
        # first version accepts only supplied conservative duration envelopes.
        start = cursor
        cursor += r.transfer_s
        total_bytes += r.transfer_bytes
        transfers.append(
            dict(request_id=request_id, start_s=start, source_free_and_clean_s=cursor)
        )
    if any(v is None for v in (rebuild_s, channel_busy_s, settlement_s)):
        unknown.append("rebuild/channel/settlement profile")
    if unknown:
        return dict(
            support_status="UNKNOWN",
            missing_inputs=unknown,
            estimated_clear_time=None,
            estimated_ready_time=None,
            timeline=transfers,
            transfer_bytes=None,
        )
    clear = max(cursor, settlement_s, *(r.natural_s for r in waiting), 0)
    return dict(
        support_status="SUPPORTED",
        missing_inputs=[],
        estimated_clear_time=clear,
        estimated_ready_time=clear + rebuild_s,
        timeline=transfers,
        transfer_bytes=total_bytes,
    )


def candidates(requests, rebuild_s, *, channel_busy_s=0, settlement_s=0):
    eligible = [
        r
        for r in requests
        if r.migratable
        and r.transfer_s is not None
        and r.transfer_s > 0
        and r.natural_s is not None
        and r.duration_pmf is not None
        and r.transfer_bytes is not None
    ]
    long_first = sorted(eligible, key=lambda r: (-r.natural_s, r.request_id))
    efficiency = sorted(
        eligible, key=lambda r: (-r.natural_s / r.transfer_s, r.request_id)
    )
    points = {0.0, *(r.natural_s for r in requests if r.natural_s is not None)}
    for order in (long_first, efficiency):
        cursor = channel_busy_s
        for r in order:
            cursor += r.transfer_s
            points.add(cursor)
    points = sorted(points)
    if len(points) > 8:
        points = [points[round(i * (len(points) - 1) / 7)] for i in range(8)]
    plans = {}
    for deadline in points:
        orders = (
            sorted(
                eligible,
                key=lambda r: (-overdue(r, deadline), -r.natural_s, r.request_id),
            ),
            sorted(
                eligible,
                key=lambda r: (
                    -max(0, r.natural_s - deadline) / r.transfer_s,
                    r.request_id,
                ),
            ),
        )
        for order in orders:
            for count in range(len(order) + 1):
                ids = tuple(r.request_id for r in order[:count])
                if ids in plans:
                    continue
                estimate = timeline(
                    requests,
                    ids,
                    rebuild_s,
                    channel_busy_s=channel_busy_s,
                    settlement_s=settlement_s,
                )
                plans[ids] = dict(
                    **estimate,
                    ordered_migration_set=list(ids),
                    wait_set=[
                        r.request_id for r in requests if r.request_id not in ids
                    ],
                    deadline=deadline,
                )
    return points, list(plans.values())


def plan_next(
    plan,
    *,
    snapshot_id,
    now,
    requests,
    rebuild_s,
    profile_hashes,
    resources_safe,
    channel_busy_s=0,
    settlement_s=0,
    utilities=None,
    destroyed=False,
):
    if plan.target_tp not in (2, 4):
        raise ValueError("unsupported target TP")
    if plan.execution_mode not in {"SHADOW", "ACTIVE"}:
        raise ValueError("explicit SHADOW/ACTIVE required")
    if plan.objective_mode not in {"NET_UTILITY", "MECHANISM_ONLY"}:
        raise ValueError("explicit objective required")
    result = dict(
        plan_id=plan.plan_id,
        bundle_id=plan.bundle_id,
        snapshot_id=snapshot_id,
        objective_mode=plan.objective_mode,
        execution_mode=plan.execution_mode,
        profile_hashes=profile_hashes,
        support_status="UNKNOWN",
        decision="HOLD",
        wait_set=[],
        ordered_migration_set=[],
        candidate_deadlines=[],
        estimated_clear_time=None,
        estimated_ready_time=None,
        delta_g=None,
        C_pre=None,
        C_post=None,
        J=None,
        missing_inputs=[],
        target_resource_conditions=dict(safe=resources_safe),
    )
    if destroyed:
        return dict(
            result,
            decision="RECOVERY_REQUIRED",
            reason="destruction crossed; finish or recover",
        )
    if now >= plan.deadline:
        return dict(result, decision="PLAN_ABORT", reason="finite plan deadline")
    if not resources_safe:
        return dict(result, reason="fresh resource/protocol protection failed")
    if any(v is None for v in (rebuild_s, channel_busy_s, settlement_s)):
        return dict(
            result,
            reason="stage time unknown",
            missing_inputs=["rebuild/channel/settlement profiles"],
        )
    points, rows = candidates(
        requests, rebuild_s, channel_busy_s=channel_busy_s, settlement_s=settlement_s
    )
    result["candidate_deadlines"] = points
    supported = [r for r in rows if r["support_status"] == "SUPPORTED"]
    if plan.method == "DRAIN":
        supported = [r for r in supported if not r["ordered_migration_set"]]
    elif plan.method == "EVAC_ALL_ONCE":
        order = [
            r.request_id
            for r in sorted(
                (r for r in requests if r.migratable),
                key=lambda r: (-(r.natural_s or 0), r.request_id),
            )
        ]
        supported = [r for r in supported if r["ordered_migration_set"] == order]
    elif plan.method not in {"SELECTIVE_ONCE", "SELECTIVE_ROLL"}:
        raise ValueError("unknown manual method")
    if not supported:
        return dict(
            result,
            reason="required time profiles/inputs missing",
            missing_inputs=sorted({x for r in rows for x in r["missing_inputs"]}),
        )
    if plan.objective_mode == "NET_UTILITY":
        # Entries must be matched by the profile adapter to CURRENT state,
        # INCLUDING post-ready imbalance. There is no TP2/TP4 extrapolation.
        evaluated = []
        for row in supported:
            key = tuple(row["ordered_migration_set"])
            u = (utilities or {}).get(key)
            if not u or any(
                u.get(k) is None or not math.isfinite(u[k])
                for k in ("delta_g", "C_pre", "C_post")
            ):
                continue
            value = max(0, plan.utility_end - now - row["estimated_ready_time"])
            evaluated.append(
                dict(row, **u, J=value * u["delta_g"] - u["C_pre"] - u["C_post"])
            )
        if len(evaluated) != len(supported):
            return dict(
                result,
                reason="incomplete matched net utility profiles",
                missing_inputs=["delta_g/C_pre/C_post for candidates"],
            )
        chosen = min(
            evaluated,
            key=lambda r: (-r["J"], r["transfer_bytes"], r["ordered_migration_set"]),
        )
        if chosen["J"] <= 0:
            return dict(
                result,
                support_status="SUPPORTED",
                decision="NO_START",
                reason="nonpositive future utility",
                J=chosen["J"],
            )
    else:
        chosen = min(
            supported,
            key=lambda r: (
                r["estimated_ready_time"],
                r["transfer_bytes"],
                r["ordered_migration_set"],
            ),
        )
    result.update(chosen)
    result["decision"] = (
        "HOLD"
        if channel_busy_s > 0
        else "START_NEXT_MIGRATION"
        if chosen["ordered_migration_set"]
        else "HOLD"
        if requests or settlement_s > 0
        else "BEGIN_REBUILD"
    )
    result["reason"] = "bounded prefix search; estimates do not authorize release"
    return result
