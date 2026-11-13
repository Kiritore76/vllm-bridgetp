# SPDX-License-Identifier: Apache-2.0
"""Causal, cost-aware service-budget estimates for the six-card experiment.

This is a working SLO-time/KV-packing model, not a calibrated prediction of
strict goodoutput. Its unit is token-service opportunity over a fixed horizon.
Only the measured 4090D/32K/16-slot shape and observed fixed-output prototypes
are supported. Unknown inputs mean KEEP, never fabricated positive utility.
"""

from __future__ import annotations

import math
from collections import deque

# Single-run short-rate evidence; packing geometry from the matched six-card
# native receipts. Rebuild budgets cover the observed post-FENCED times
# (36.87s split, 50.54s merge), without assuming CPU prestart is enabled.
SHORT_CAPACITY = {"L0": 20.0, "L1AC": 18.0, "L2C": 16.0, "L3": 12.0}
BLOCKS = {1: 2325, 2: 11712}
# vLLM sleep mode (engine reuse, 11-12) allocates through its own memory pool
# and leaves slightly more KV: TP1 2343 and TP2 11747 blocks at util 0.88
# (+0.8%/+0.3%). Accept the measured shape up to 2% larger; packing keeps the
# profile blocks (conservative). Fewer blocks, or more than 2% more, is still
# an unknown shape.
BLOCKS_TOLERANCE = 0.02


def profile_shape(tp, total_blocks, block_size):
    """Whether a native KV shape matches the service-budget profile."""
    return (
        block_size == 16
        and type(total_blocks) is int
        and BLOCKS[tp] <= total_blocks <= BLOCKS[tp] * (1 + BLOCKS_TOLERANCE)
    )
REBUILD_S = {"TP1": 40.0, "TP2": 60.0}
REFERENCE_TPOT_S = 0.035
ARRIVAL_WINDOW_S = 240.0
HORIZON_S = 300.0
COST_MARGIN = 1.25


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class Arrivals:
    """Consume only jobs whose release time has passed, including held jobs."""

    def __init__(self):
        self.cursor = 0
        self.rows = deque()
        self.prototypes = {False: set(), True: set()}
        self.fixed = True

    def snapshot(self, jobs, elapsed, now):
        while self.cursor < len(jobs) and jobs[self.cursor]["at_s"] <= elapsed:
            job = jobs[self.cursor]
            body = job["request"]
            prompt, output = len(body["prompt"]), body["max_tokens"]
            long = prompt + output >= 8192
            self.fixed = self.fixed and body.get("ignore_eos") is True
            self.prototypes[long].add((prompt, output))
            self.rows.append((job["at_s"], long))
            self.cursor += 1
        while self.rows and self.rows[0][0] < elapsed - ARRIVAL_WINDOW_S:
            self.rows.popleft()
        span = min(elapsed, ARRIVAL_WINDOW_S)
        return dict(
            schema="SIX_CAUSAL_DEMAND_V1",
            sampled_at=now,
            coverage_s=span,
            window_s=ARRIVAL_WINDOW_S,
            fixed_output=self.fixed,
            short_count=sum(not long and at >= elapsed - 30 for at, long in self.rows),
            short_window_s=min(elapsed, 30.0),
            long_count=sum(long for _, long in self.rows),
            long_recent_count=sum(
                long and at >= elapsed - 60 for at, long in self.rows
            ),
            prototypes={
                "long" if long else "short": list(next(iter(values)))
                if len(values) == 1
                else None
                for long, values in self.prototypes.items()
            },
        )


def estimate(status, modes, members, edge, now, signals):
    """Compare staying against one adjacent move, charging the unavailable phase.

    Future arrivals use only past release counts; request completions release
    their long-slot demand. No future phase, future output, or output cap of a
    variable-EOS request is consulted.
    """
    unknown = lambda reason: dict(supported=False, worthwhile=False, reason=reason)
    if (
        set(modes) != set("ABC")
        or any(tp not in (1, 2) for tp in modes.values())
        or edge[0] not in modes
        or edge[1] not in ("TP1", "TP2")
    ):
        return unknown("unsupported layout or edge")
    stamp = status.get("unix_s")
    if not number(stamp) or not 0 <= now - stamp <= 2:
        return unknown("stale controller status")
    if status.get("service_shape") != dict(
        platform="qwen3",
        tp1_context=32768,
        tp2_context=32768,
        tp1_slots=16,
        tp2_slots=16,
    ):
        return unknown("service shape outside registered support")
    demand = (status.get("workload") or {}).get("conversion_demand") or {}
    if demand.get("schema") != "SIX_CAUSAL_DEMAND_V1":
        return unknown("missing causal demand")
    stamp = demand.get("sampled_at")
    if not number(stamp) or not 0 <= now - stamp <= 2:
        return unknown("stale causal demand")
    if demand.get("fixed_output") is not True:
        return unknown("variable output has no supported service-budget model")
    for key in (
        "coverage_s",
        "short_count",
        "short_window_s",
        "long_count",
        "long_recent_count",
    ):
        if not number(demand.get(key)):
            return unknown("invalid causal counts")
    if demand["coverage_s"] < 30 or demand["short_window_s"] < 30:
        return unknown("insufficient arrival history")
    if demand.get("window_s") != ARRIVAL_WINDOW_S:
        return unknown("arrival window differs from profile")
    proto = demand.get("prototypes") or {}
    short = proto.get("short")
    long = proto.get("long")
    if short != [512, 128] or (long is not None and long != [16384, 8192]):
        return unknown("request mix outside registered service-budget support")
    short_rate = demand["short_count"] / demand["short_window_s"]
    held_short, held_long = (
        demand.get(k, 0) for k in ("held_short_count", "held_long_count")
    )
    if not number(held_short) or not number(held_long):
        return unknown("invalid held demand")
    short_work_rate = short_rate + held_short / HORIZON_S
    # Both a sustained window and recent arrivals must support continuation.
    long_rate = min(
        demand["long_count"] / demand["coverage_s"], demand["long_recent_count"] / 60
    )
    after = dict(modes, **{edge[0]: int(edge[1][-1])})
    label = lambda values: (
        "L"
        + str(sum(v == 2 for v in values.values()))
        + (
            "".join(b for b in "ABC" if values[b] == 1)
            if sum(v == 2 for v in values.values()) in (1, 2)
            else ""
        )
    )
    before_name, after_name = label(modes), label(after)
    if before_name not in SHORT_CAPACITY or after_name not in SHORT_CAPACITY:
        return unknown("layout outside six-card path")
    instances = status["instances"]
    ids = [name for group in members.values() for name in group]
    active = {
        name for name, row in instances.items() if row.get("lifecycle") == "ACTIVE"
    }
    if (
        set(members) != set(modes)
        or len(ids) != len(set(ids))
        or set(ids) != active
        or any(len(members[b]) != (2 if tp == 1 else 1) for b, tp in modes.items())
    ):
        return unknown("incomplete or inconsistent active membership")
    requests = {}
    native_blocks = {}
    for b, ids in members.items():
        for name in ids:
            state = instances.get(name) or {}
            native = state.get("inventory") or {}
            if not profile_shape(
                modes[b], native.get("total_blocks"), native.get("block_size")
            ):
                return unknown("native shape differs from service-budget profile")
            native_blocks[modes[b]] = min(
                native["total_blocks"], native_blocks.get(modes[b], native["total_blocks"])
            )
            for q in state.get("requests", ()):
                if not all(
                    number(q.get(k))
                    for k in ("prompt_tokens", "generated", "max_output_tokens")
                ):
                    return unknown("incomplete request work")
                if q["generated"] > q["max_output_tokens"]:
                    return unknown("request progress exceeds fixed output")
                key = q["request_id"]
                if key not in requests or (
                    q["prompt_tokens"] + q["generated"]
                    > requests[key]["prompt_tokens"] + requests[key]["generated"]
                ):
                    requests[key] = q
    long_rows = [
        q
        for q in requests.values()
        if q["prompt_tokens"] + q["max_output_tokens"] >= 8192
    ]
    if long is None and (long_rows or demand["long_count"] or held_long):
        return unknown("missing observed long prototype")
    if long is not None and any(
        q["prompt_tokens"] + q["max_output_tokens"] != sum(long) for q in long_rows
    ):
        return unknown("active long request differs from observed prototype")
    # Fixed-output mode makes this remaining work known, not a cap-as-prediction.
    lifetimes = [
        max(0, q["max_output_tokens"] - q["generated"]) * REFERENCE_TPOT_S
        for q in long_rows
    ]
    lifetimes.extend([REFERENCE_TPOT_S * (long[1] if long else 0)] * int(held_long))
    source = [
        q for name in members[edge[0]] for q in instances[name].get("requests", ())
    ]
    drain_s = max(
        (
            max(0, q["max_output_tokens"] - q["generated"]) * REFERENCE_TPOT_S
            for q in source
        ),
        default=0,
    )
    ready_s = drain_s + REBUILD_S[edge[1]]
    long_blocks = math.ceil(sum(long) / 16) if long else 1536
    long_output = long[1] if long else 8192
    residence = long_output * REFERENCE_TPOT_S

    def slots(layout):
        # A per-instance packing limit, not a sum of fungible free KV blocks.
        count = sum(2 if tp == 1 else 1 for tp in layout.values())
        short_reserve = short_rate * 128 * REFERENCE_TPOT_S * 40 / max(count, 1)
        return sum(
            (2 if tp == 1 else 1)
            * max(0, math.floor((BLOCKS[tp] - 1 - 125 - short_reserve) / long_blocks))
            for tp in layout.values()
        )

    stay_slots, target_slots = slots(modes), slots(after)
    retained = {b: tp for b, tp in modes.items() if b != edge[0]}
    retained_slots = slots(retained)
    # Nominal per-bundle short capacity weights; total rates remain measured.
    weights = {b: (2 if tp == 1 else 1) for b, tp in modes.items()}
    retained_short = SHORT_CAPACITY[before_name] * (
        1 - weights[edge[0]] / sum(weights.values())
    )
    gain, cost, stay_deficit, steady_gain = 0.0, 0.0, 0.0, 0.0
    for t in range(0, int(HORIZON_S), 5):
        at = t + 2.5
        active = sum(life > at for life in lifetimes)
        active += long_rate * min(at, residence)
        desired_long = active / REFERENCE_TPOT_S
        stay_long = min(active, stay_slots) / REFERENCE_TPOT_S
        target_long = min(active, target_slots) / REFERENCE_TPOT_S
        stay_short = min(short_work_rate, SHORT_CAPACITY[before_name]) * 128
        target_short = min(short_work_rate, SHORT_CAPACITY[after_name]) * 128
        steady_gain += (target_long + target_short - stay_long - stay_short) * 5
        stay_deficit += (
            desired_long - stay_long + short_work_rate * 128 - stay_short
        ) * 5
        if at < ready_s:
            serving_long = min(active, retained_slots) / REFERENCE_TPOT_S
            serving_short = min(short_work_rate, retained_short) * 128
            cost += max(0, stay_long + stay_short - serving_long - serving_short) * 5
        else:
            gain += (target_long + target_short - stay_long - stay_short) * 5
    margin = COST_MARGIN * cost
    if not all(math.isfinite(v) for v in (gain, cost, steady_gain, stay_deficit)):
        return unknown("nonfinite service budget")
    tolerance = max(1e-6, 1e-9 * max(abs(gain), margin, stay_deficit))
    worthwhile = stay_deficit > tolerance and gain - margin > tolerance
    return dict(
        supported=True,
        worthwhile=worthwhile,
        reason="positive service budget" if worthwhile else "no demonstrated payback",
        unit="token_service_opportunity",
        model="fixed-output SLO-time/KV-packing estimate; not strict goodoutput",
        short_rate=short_rate,
        long_rate=long_rate,
        horizon_s=HORIZON_S,
        native_blocks={f"tp{tp}": n for tp, n in sorted(native_blocks.items())},
        profile_blocks={f"tp{tp}": n for tp, n in sorted(BLOCKS.items())},
        stay_long_slots=stay_slots,
        target_long_slots=target_slots,
        drain_s=drain_s,
        rebuild_s=REBUILD_S[edge[1]],
        ready_s=ready_s,
        gain_tokens=gain,
        cost_tokens=cost,
        required_gain_tokens=margin,
        net_tokens=gain - margin,
        stay_deficit_tokens=stay_deficit,
        steady_gain_tokens=steady_gain,
        demand_supports_move=stay_deficit > tolerance and steady_gain > tolerance,
    )


class Evidence:
    """A bounded, time-weighted window, unaffected by one threshold crossing."""

    def __init__(self):
        self.rows = {}

    def observe(self, key, now, supported, positive, hard, span):
        rows = self.rows.setdefault(key, deque())
        if not number(now):
            rows.clear()
            return dict(ready=False, coverage_s=0, positive_fraction=0)
        if rows and (now < rows[-1][0] or now - rows[-1][0] > 2):
            rows.clear()
        if rows and now == rows[-1][0]:
            return dict(ready=False, coverage_s=0, positive_fraction=0)
        rows.append((now, bool(supported and positive and hard), hard, supported))
        while len(rows) > 1 and rows[1][0] <= now - span:
            rows.popleft()
        covered, good = 0.0, 0.0
        for left, right in zip(rows, list(rows)[1:]):
            dt = max(0, right[0] - max(left[0], now - span))
            covered += dt
            if left[1]:
                good += dt
        ratio = good / covered if covered else 0
        complete = covered >= span and all(r[3] for r in rows)
        return dict(
            ready=complete and rows[-1][1] and ratio >= 0.8,
            coverage_s=covered,
            positive_fraction=ratio,
        )
