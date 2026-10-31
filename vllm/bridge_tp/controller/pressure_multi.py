# SPDX-License-Identifier: Apache-2.0
"""Multi-instance routing and migration selection over the two-pool rules.

New requests: inside each TP class, instances are ranked by prompt fit, then
the unchanged projected pressure, then normalized in-flight admissions and
instance id; the best TP1 and best TP-k then go through the unchanged
two-pool ``admit``. A single-class layout takes the class leader directly.

Migration: every (TP1 source, TP-k target) pair is evaluated with the
unchanged ``decide``; each source keeps its best target and at most one pair
is executed globally. With one instance per class both reduce to the
existing two-pool behaviour.
"""

from __future__ import annotations

import math

from .pressure_policy import admission_count, projected_pressure, prompt_fit

FIT_RANK = {"FIT": 0, "UNKNOWN": 1, "NO_FIT": 2}


def class_ranking(
    readings,
    prompt_tokens,
    requested_output,
    ledger,
    cfg,
    now,
    routing_strategy="pressure",
):
    """Rank one TP class; rows are ordered best first.

    Key: context feasible, prompt fit (FIT > UNKNOWN > NO_FIT), projected
    pressure (only when the class has at least one valid score), normalized
    in-flight admissions, instance id.
    """
    output_cap = min(8192, requested_output)
    scored = cfg.workload_pressure and routing_strategy == "pressure"
    rows = []
    for raw in readings:
        reading = ledger.routing_reading(raw)
        projected = (
            projected_pressure(reading, prompt_tokens, cfg, ledger, now)
            if scored
            else dict(service=None, capacity=None, score=None)
        )
        rows.append(
            dict(
                pool_id=raw.pool_id,
                context_feasible=raw.context_limit > 0
                and prompt_tokens + output_cap <= raw.context_limit,
                fit=prompt_fit(raw, prompt_tokens, ledger, cfg, now)["status"],
                score=projected["score"],
                service=projected["service"],
                capacity=projected["capacity"],
                inflight=admission_count(ledger, raw.pool_id, routing_strategy),
            )
        )
    use_score = any(r["context_feasible"] and r["score"] is not None for r in rows)

    def key(row):
        score = row["score"]
        pressure = (score is None, score if score is not None else 0.0)
        return (
            not row["context_feasible"],
            FIT_RANK[row["fit"]],
            pressure if use_score else (False, 0.0),
            row["inflight"],
            row["pool_id"],
        )

    rows.sort(key=key)
    basis = (
        "fit, projected pressure, in-flight, id"
        if use_score
        else ("fit, in-flight, id (no valid projected pressure in class)")
    )
    for rank, row in enumerate(rows):
        row["rank"] = rank
        row["basis"] = basis
    return rows


def unified_choice(rankings):
    """Advisory only: the instance a single cross-class argmin would choose."""
    rows = [r for ranking in rankings for r in ranking if r["context_feasible"]]
    if not rows:
        return None
    use_score = any(r["score"] is not None for r in rows)

    def key(row):
        score = row["score"]
        return (
            FIT_RANK[row["fit"]],
            (score is None, score if score is not None else 0.0)
            if use_score
            else (False, 0.0),
            row["inflight"],
            row["pool_id"],
        )

    best = min(rows, key=key)
    return dict(
        pool_id=best["pool_id"],
        basis="unified fit, projected pressure, in-flight, id",
        executed=False,
    )


def target_headroom(reading, ledger):
    """Native free blocks after pending migration promises, or None."""
    if reading.free_blocks is None:
        return None
    return reading.free_blocks - ledger.pending(reading.pool_id)[0]


def pair_key(decision, headroom, source_id, target_id):
    """Order executable (source, target) decisions; smaller is better.

    KV capacity relief before service relief; inside a path, the unchanged
    selection key of the chosen request (joint KV benefit first when KV
    priority is on); then larger target headroom; then ids.
    """
    row = next(r for r in decision.candidates if r.get("selected"))
    return (
        0 if decision.capacity_path else 1,
        tuple(row["selection_key"]),
        -headroom if headroom is not None else math.inf,
        source_id,
        target_id,
    )
