# SPDX-License-Identifier: Apache-2.0
"""Offline-only prefill reference and pressure ablation; no serving mutation."""

import math
from dataclasses import replace

from tools.bridge_tp.pressure_cpu import load

p = load("pressure_policy")
baseline_classify = p.classify


def quantile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def features(batch):
    if batch.get("error") or batch.get("recompute_tokens", 0):
        return None
    tokens = batch.get("prefill_tokens", 0)
    if tokens <= 0:
        return None
    prefix = (
        sum(row["computed_before"] * row["prefill_tokens"] for row in batch["requests"])
        / tokens
    )
    nd = batch["decode_requests"]
    context = batch["decode_context_sum"] / nd if nd else 0
    if not all(math.isfinite(v) and v >= 0 for v in (tokens, prefix, nd, context)):
        raise ValueError("invalid batch features")
    return dict(tokens=tokens, prefix=prefix, decode_context=context, nd=nd)


def key(pool, values):
    return (
        pool,
        values["nd"],
        values["tokens"],
        int(values["prefix"] // 1024),
        int(values["decode_context"] // 1024),
    )


class PrefillReference:
    """Exact shape keys, bounded context interpolation, empirical P95 cost.

    Sample count is not independent run count. Unseen shape/context is UNKNOWN,
    including 256-token shapes when only 512-token measurements are available.
    The P95 is descriptive, not a statistical upper bound.
    """

    def __init__(self, cells, minimum_events=2):
        self.cells = cells
        self.minimum_events = minimum_events
        self.index = {tuple(c["key"]): c for c in cells}

    @classmethod
    def fit(cls, records):
        groups = {}
        for row in records:
            values = features(row)
            if values is None:
                continue
            group = groups.setdefault(key(row["pool"], values), [])
            group.append((row, values))
        cells = []
        for shape, rows in sorted(groups.items()):
            times = [r["completion_latency_ms"] for r, _ in rows]
            if any(not math.isfinite(v) or v < 0 for v in times):
                raise ValueError("invalid engine latency")
            cells.append(
                dict(
                    key=list(shape),
                    samples=len(rows),
                    runs=sorted({r["run_id"] for r, _ in rows}),
                    events=len(
                        {
                            (r["run_id"], request["request_id"])
                            for r, _ in rows
                            for request in r["requests"]
                            if request["prefill_tokens"]
                        }
                    ),
                    prefix_range=[
                        min(v["prefix"] for _, v in rows),
                        max(v["prefix"] for _, v in rows),
                    ],
                    decode_context_range=[
                        min(v["decode_context"] for _, v in rows),
                        max(v["decode_context"] for _, v in rows),
                    ],
                    engine_ms=dict(
                        p50=quantile(times, 0.5),
                        p95=quantile(times, 0.95),
                        maximum=max(times),
                    ),
                )
            )
        return cls(cells)

    def estimate(self, pool, batch):
        values = features(batch)
        if values is None:
            return None
        cell = self.index.get(key(pool, values))
        if (
            cell is None
            or cell["events"] < self.minimum_events
            or cell["samples"] < self.minimum_events
        ):
            return None
        if not all(
            cell[name + "_range"][0] <= values[name] <= cell[name + "_range"][1]
            for name in ("prefix", "decode_context")
        ):
            return None
        return dict(
            engine_s=cell["engine_ms"]["p95"] / 1000,
            cell_key=cell["key"],
            samples=cell["samples"],
            events=cell["events"],
            runs=cell["runs"],
        )


def cost_pressure(reading, cfg, now, estimate, reference_s=0.1):
    """Replace only the step term in an offline classification ablation.

    Keep physical free, decode, backlog, waiting, age and missing-evidence rules.
    An unknown positive prefill shape never falls back to the token score.
    """
    no_prefill = reading.prefill_batch_tokens == 0 or (
        reading.prefill_batch_tokens is None and reading.pending_prompt_tokens == 0
    )
    known = estimate is not None or no_prefill or reading.idle
    base = baseline_classify(
        replace(reading, prefill_batch_tokens=0 if known else None), cfg, now
    )
    if base.prefill is None or "pool timestamp" in base.missing:
        return base
    ratio = estimate["engine_s"] / reference_s if estimate is not None else 0
    value = max(base.prefill, ratio)
    missing = base.missing
    if not known and "prefill step" not in missing:
        missing += ("prefill cost outside support",)
    values = (base.decode, value, base.capacity)
    if any(v is not None and v >= cfg.enter_ratio for v in values):
        zone = "OVERLOAD"
    elif missing:
        zone = "UNKNOWN"
    elif max(values) <= cfg.spare_ratio:
        zone = "SPARE"
    else:
        zone = "CRITICAL"
    return p.Pressure(base.decode, value, base.capacity, zone, missing)


class ReliefGate:
    """Pressure/persistence stage only, without inventing feasible candidates.

    This follows the ordinary policy's enter/leave latch. An emitted opportunity
    is not START: stream ownership, PMF, target commitments and release timing
    are deliberately not fabricated. Cooldown has no effect without a START.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.since = None
        self.last_tick = None
        self.latched = False
        self.reported = False

    def update(self, now, source, target):
        cfg = self.cfg
        if self.last_tick is not None and now < self.last_tick:
            raise ValueError("time regressed")
        if self.last_tick is not None and now - self.last_tick > cfg.max_tick_gap_s:
            self.since = None
            self.reported = False
        self.last_tick = now
        threshold = cfg.leave_ratio if self.latched else cfg.enter_ratio
        gap = cfg.imbalance_leave if self.latched else cfg.imbalance_enter
        source_service = [v for v in (source.decode, source.prefill) if v is not None]
        target_service = [v for v in (target.decode, target.prefill) if v is not None]
        needed = (
            not source.missing
            and target.zone == "SPARE"
            and source_service
            and target_service
            and max(source_service) >= threshold
            and max(source_service) - max(target_service) >= gap
        )
        if not needed:
            self.since = None
            self.latched = self.reported = False
            return False
        self.latched = True
        if self.since is None:
            self.since = now
        if now - self.since >= cfg.sustain_s and not self.reported:
            self.reported = True
            return True
        return False
