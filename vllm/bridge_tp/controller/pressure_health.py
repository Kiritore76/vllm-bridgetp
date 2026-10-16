# SPDX-License-Identifier: Apache-2.0
"""Past-only closed-window health; empirical working points are not ceilings."""

import bisect
import collections
import math
import statistics
import threading


def lookup(reference, pool, nd, context):
    if type(nd) is not int or nd <= 0 or not math.isfinite(context) or context < 0:
        return None
    key = (pool, nd, int(context // 2048))
    cells = [
        c
        for c in reference["cells"]
        if (c["pool"], c["nd"], c["context_bin"]) == key
        and c["health_status"] == "REFERENCE"
    ]
    return cells[0] if len(cells) == 1 else None


def efficiency(reference, pool, nd, context, goodoutput):
    cell = lookup(reference, pool, nd, context)
    if cell is None or not math.isfinite(goodoutput) or goodoutput < 0:
        return dict(status="UNKNOWN", reason="outside repeated health support")
    expected = cell["window_goodoutput"]["median"]
    peak = max(
        c["window_goodoutput"]["median"]
        for c in reference["cells"]
        if c["pool"] == pool
        and c["context_bin"] == cell["context_bin"]
        and c["health_status"] == "REFERENCE"
    )
    return dict(
        status="SUPPORTED",
        eta=goodoutput / expected,
        reference_load=expected / peak,
        reference_goodoutput_s=expected,
        measured_peak_goodoutput_s=peak,
        nd=nd,
        context_bin=cell["context_bin"],
        note="measured peak only; no upper-capacity claim",
    )


def supported_after(reference, nd, context):
    """Require a measured heavier target shape in the same context bin.

    This is an explicit monotonic-shape support assumption, not interpolation
    or a predicted goodoutput delta. Existing engine and physical budgets apply.
    """
    if nd <= 0 or not math.isfinite(context) or context < 0:
        return False
    return any(
        c["pool"] == "target"
        and c["health_status"] == "REFERENCE"
        and c["context_bin"] == int(context // 2048)
        and c["nd"] >= nd
        for c in reference["cells"]
    )


class TokenWindows:
    """Mirror real SSE callbacks, keeping window-boundary predecessors."""

    def __init__(self):
        self.rows = {}
        self.lock = threading.RLock()

    def arrive(self, request_id, pool, now):
        with self.lock:
            for key, row in list(self.rows.items()):
                if row["end"] is not None and row["end"] < now - 22:
                    del self.rows[key]
            self.rows[request_id] = dict(pool=pool, times=collections.deque(), end=None)

    def token(self, request_id, now):
        with self.lock:
            times = self.rows[request_id]["times"]
            if times and now < times[-1]:
                raise ValueError("health token time regressed")
            times.append(now)
            while len(times) > 1 and times[1] < now - 22:
                times.popleft()

    def finish(self, request_id):
        with self.lock:
            row = self.rows.get(request_id)
            if row is not None:
                row["end"] = row["times"][-1] if row["times"] else -math.inf

    def score(self, pool, left, right):
        with self.lock:
            good = raw = 0
            for key, row in list(self.rows.items()):
                if row["end"] is not None and row["end"] < left - 12:
                    del self.rows[key]
                    continue
                if row["pool"] != pool:
                    continue
                times = list(row["times"])
                a, b = bisect.bisect_left(times, left), bisect.bisect_left(times, right)
                gaps = [times[k] - times[k - 1] for k in range(max(1, a), b)]
                fraction = sum(g > 0.1 for g in gaps) / len(gaps) if gaps else 0
                old = min(1, 0.01 / fraction) if fraction else 1
                active = row["end"] is None or row["end"] >= right
                censored = right - times[b - 1] if active and b else 0
                maximum = max(gaps + [censored])
                weight = min(old, 1 / maximum if maximum > 1 else 1)
                raw += b - a
                good += (b - a) * weight
            return dict(goodoutput_s=good / (right - left), raw_s=raw / (right - left))


class HealthTracker:
    def __init__(self, reference, origin, tokens):
        if (
            reference.get("schema") != "PRESSURE_HEALTH_REFERENCE_V2"
            or reference.get("metric") != "CAUSAL_10S_ITL100MS_SLOW1PCT_MAXGAP1S_V1"
        ):
            raise ValueError("versioned causal health reference required")
        keys = [(c["pool"], c["nd"], c["context_bin"]) for c in reference["cells"]]
        if (
            len(set(keys)) != len(keys)
            or not keys
            or any(
                c["runs"] < 2
                or c["nd"] <= 0
                or c["window_goodoutput"]["median"] <= 0
                or not math.isfinite(c["window_goodoutput"]["median"])
                for c in reference["cells"]
            )
        ):
            raise ValueError("invalid repeated reference cells")
        self.reference, self.origin, self.tokens = reference, origin, tokens
        self.samples = {pool: collections.deque() for pool in ("tp1", "tp4")}

    def observe(self, raw, now):
        pool = raw["pool_id"]
        history = self.samples[pool]
        if history and history[-1][1] == raw["generation"]:
            return
        batch = raw.get("last_scheduled_batch")
        shape = None
        if (
            batch
            and not batch.get("error")
            and 0 <= now - raw["sampled_at"] <= 2
            and 0 <= now - batch["sampled_at"] <= 2
            and batch["decode_requests"] > 0
            and batch["recompute_tokens"] == 0
        ):
            shape = (
                batch["decode_requests"],
                batch["decode_context_sum"] / batch["decode_requests"],
            )
        mixed = bool(batch and batch.get("prefill_tokens"))
        history.append((now, raw["generation"], shape, mixed))
        while history and history[0][0] < now - 24:
            history.popleft()

    def snapshot(self, pool, now, idle=False):
        if idle:
            return dict(
                health_status="IDLE",
                health_eta=1.0,
                health_load=0.0,
                health_sampled_at=now,
                health_mixed_window=False,
            )
        right = self.origin + math.floor((now - self.origin) / 10) * 10
        unknown = dict(
            health_status="UNKNOWN",
            health_eta=None,
            health_load=None,
            health_sampled_at=right,
            health_mixed_window=False,
        )
        if right - self.origin < 10:
            return unknown
        left = right - 10
        rows = [r for r in self.samples[pool] if left <= r[0] < right]
        if (
            not rows
            or rows[0][0] > left + 2
            or rows[-1][0] < right - 2
            or any(b[0] - a[0] > 2 for a, b in zip(rows, rows[1:]))
        ):
            return unknown
        valid = [r for r in rows if r[2] is not None]
        if not valid:
            return unknown
        nd, nd_count = collections.Counter(r[2][0] for r in valid).most_common(1)[0]
        context_bin, bin_count = collections.Counter(
            int(r[2][1] // 2048) for r in valid
        ).most_common(1)[0]
        current = self.samples[pool][-1]
        if (
            nd_count / len(rows) < 0.8
            or bin_count / len(rows) < 0.8
            or current[2] is None
            or current[2][0] != nd
            or int(current[2][1] // 2048) != context_bin
            or now - current[0] > 2
        ):
            return unknown
        measured = self.tokens.score(pool, left, right)
        value = efficiency(
            self.reference,
            "source" if pool == "tp1" else "target",
            nd,
            statistics.median(
                r[2][1] for r in valid if int(r[2][1] // 2048) == context_bin
            ),
            measured["goodoutput_s"],
        )
        return dict(
            health_status=value["status"],
            health_eta=value.get("eta"),
            health_load=value.get("reference_load"),
            health_sampled_at=right,
            health_mixed_window=any(r[3] for r in rows),
        )
