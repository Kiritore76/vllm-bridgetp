# SPDX-License-Identifier: Apache-2.0
"""Stop-contract-aware summaries of the frozen categorical predictor.

Finite bins use a discrete uniform approximation for point estimates only.
Risk bounds allow arbitrary mass placement inside each original bin. The open
tail must be entirely clipped by the stop contract; it is never extrapolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundedLength:
    cap: int
    pmf: tuple[float, ...]
    # (original mass, smallest surviving R-lag, largest surviving R-lag,
    #  whole original bin survived). Upper None denotes an unresolved tail.
    bins: tuple[tuple[float, int, int | None, bool], ...]
    deterministic: bool = False

    @property
    def mean(self) -> float:
        return sum(i * p for i, p in enumerate(self.pmf))

    def quantile(self, q: float) -> int:
        if not 0 < q <= 1:
            raise ValueError("quantile must be in (0, 1]")
        cumulative = 0.0
        for i, mass in enumerate(self.pmf):
            cumulative += mass
            if cumulative >= q - 1e-12:
                return i
        return self.cap

    def probability_gt(self, horizon: int) -> float:
        return sum(self.pmf[max(0, horizon + 1) :])

    def probability_ge(self, horizon: int) -> float:
        return self.probability_gt(horizon - 1)

    def probability_gt_bounds(self, horizon: int) -> tuple[float, float]:
        if horizon < 0:
            return 1.0, 1.0
        if horizon >= self.cap:
            return 0.0, 0.0
        if self.deterministic:
            return 1.0, 1.0
        sure_a = possible_a = sure_b = possible_b = 0.0
        for mass, lo, hi, whole in self.bins:
            if hi is None or hi > horizon:
                possible_a += mass
                if whole and lo > horizon:
                    sure_a += mass
            if lo <= horizon:
                possible_b += mass
                if whole and hi is not None and hi <= horizon:
                    sure_b += mass
        # Nested numerator/survival events: no independent interval division.
        low_denom = sure_a + possible_b
        high_denom = possible_a + sure_b
        return (
            sure_a / low_denom if low_denom else 0.0,
            possible_a / high_denom if high_denom else 0.0,
        )

    def probability_ge_bounds(self, horizon: int) -> tuple[float, float]:
        return self.probability_gt_bounds(horizon - 1)

    def summary(self, horizon: int) -> dict:
        return {
            "remaining_cap": self.cap,
            "mean": self.mean,
            "q50": self.quantile(0.5),
            "q75": self.quantile(0.75),
            "survival_horizon": horizon,
            "p_gt_approx": self.probability_gt(horizon),
            "p_gt_bounds": self.probability_gt_bounds(horizon),
            "point_assumption": "finite integer bins uniform; tail clipped",
        }


def bounded_length(
    probabilities: tuple[float, ...],
    edges: tuple[int, ...],
    *,
    generated: int,
    predicted_at: int,
    max_output_tokens: int,
    ignore_eos: bool = False,
) -> BoundedLength:
    """Condition on R>=lag, shift, and clip to min(8192, stop)-generated.

    No actual final length is an input. Runtime max_output_tokens is the frozen
    TOTAL output limit, never the remaining budget of a target subrequest.
    """
    integers = (generated, predicted_at, max_output_tokens, *edges)
    if any(type(v) is not int or v < 0 for v in integers):
        raise ValueError("positions, cap, and edges must be nonnegative integers")
    if (
        predicted_at > generated
        or not edges
        or edges[0] != 0
        or any(a >= b for a, b in zip(edges, edges[1:]))
    ):
        raise ValueError("invalid alignment or bucket edges")
    if (
        len(probabilities) != len(edges) + 1
        or any(not math.isfinite(p) or p < 0 for p in probabilities)
        or abs(sum(probabilities) - 1) > 1e-4
    ):
        raise ValueError("invalid PMF")
    cap = max(0, min(8192, max_output_tokens) - generated)
    lag = generated - predicted_at
    pmf = [0.0] * (cap + 1)
    if ignore_eos or cap == 0:
        pmf[cap] = 1.0
        return BoundedLength(cap, tuple(pmf), (), True)
    bins = []
    for i, mass in enumerate(probabilities):
        lo = 0 if i == 0 else edges[i - 1] + 1
        hi = edges[i] if i < len(edges) else None
        if hi is not None and hi < lag:
            continue
        surviving_lo = max(lo, lag)
        bins.append(
            (mass, surviving_lo - lag, None if hi is None else hi - lag, lo >= lag)
        )
        if hi is None:
            if mass and surviving_lo - lag < cap:
                raise ValueError("unresolved tail is not clipped by stop contract")
            pmf[cap] += mass
            continue
        density = mass / (hi - lo + 1)
        # Integrate a crossing bin, rather than clipping its midpoint.
        stop = min(hi, lag + cap - 1)
        for r in range(surviving_lo, stop + 1):
            pmf[r - lag] += density
        pmf[cap] += max(0, hi - max(surviving_lo, lag + cap) + 1) * density
    surviving_mass = sum(pmf)
    if surviving_mass <= 0:
        raise ValueError("prediction contradicts observed survival")
    return BoundedLength(cap, tuple(p / surviving_mass for p in pmf), tuple(bins))


def from_advisory(advisory: dict, max_output_tokens: int) -> BoundedLength:
    """Consume M5 freshness/identity validation without relaxing it."""
    if advisory.get("status") != "AVAILABLE":
        raise ValueError("prediction missing or stale")
    return bounded_length(
        tuple(advisory["probabilities"]),
        tuple(advisory["category_upper_edges"]),
        generated=advisory["output_tokens"],
        predicted_at=advisory["prediction_output_tokens"],
        max_output_tokens=max_output_tokens,
        ignore_eos=advisory.get("ignore_eos", False),
    )
