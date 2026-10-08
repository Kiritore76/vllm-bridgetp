# SPDX-License-Identifier: Apache-2.0
"""F1 conditional bucket risk and separate release urgency (CPU only)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


def audit_finite(value: Any) -> Any:
    """Keep nonfinite input evidence unknown in standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: audit_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [audit_finite(v) for v in value]
    return value


def remaining_ge_bounds(prediction: dict, threshold: float) -> list[float] | None:
    """Bound P(min(R-lag, cap)>=ceil(threshold) | R>=lag).

    Bucket mass can be anywhere in its integer support. Conditioning on
    survival therefore changes both numerator and denominator; no uniform
    within-bin assumption is made. Observing lag emitted tokens establishes
    R>=lag; it cannot establish an additional future visible token. Fresh
    predictions (lag=0) retain their zero-remaining mass.
    """
    if prediction.get("status") != "AVAILABLE" or prediction.get("ignore_eos"):
        return None
    probs = prediction.get("probabilities")
    edges = prediction.get("category_upper_edges")
    lag = prediction.get("output_tokens", 0) - prediction.get(
        "prediction_output_tokens", 0
    )
    if not probs or not edges or lag < 0 or not math.isfinite(threshold):
        return None
    if (
        len(probs) != len(edges) + 1
        or edges[0] != 0
        or any(not isinstance(x, int) or x < 0 for x in edges)
        or any(a >= b for a, b in zip(edges, edges[1:]))
        or any(
            not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
            for x in probs
        )
        or abs(sum(probs) - 1) > 1e-4
    ):
        return None
    n = math.ceil(threshold)
    cap = prediction.get("max_remaining_output_tokens")
    if cap is not None and cap <= 0:
        return None
    if cap is not None and n > cap:
        return [0.0, 0.0]
    # Each bin intersects A=survives AND reaches threshold, B=survives
    # but below threshold, and C=already ended. Extremes allocate ambiguous
    # mass to B (lower) or A (upper), excluding optional C mass.
    a_min = b_max = a_max = b_min = 0.0
    for i, mass in enumerate(probs):
        lo = 0 if i == 0 else edges[i - 1] + 1
        hi = edges[i] if i < len(edges) else math.inf
        a_start = max(lag, lag + n)
        has_a = hi >= max(lo, a_start)
        has_b = max(lo, lag) <= min(hi, lag + n - 1)
        if has_a:
            a_max += mass
            if lo >= a_start:
                a_min += mass
        if has_b:
            b_max += mass
            if lo >= lag and hi < lag + n:
                b_min += mass
    if a_max + b_max <= 0:
        return None
    if n <= 0:
        return [1.0, 1.0]
    lower = a_min / (a_min + b_max) if a_min + b_max else 0.0
    upper = a_max / (a_max + b_min) if a_max + b_min else 0.0
    return [lower, upper]


@dataclass(frozen=True)
class RiskUrgencySnapshot:
    """Audit contract; time envelopes are assumptions, not calibrated CIs."""

    values: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)["values"]


class DecodeRateTracker:
    """EWMA of candidate output progress, distinct from pool growth."""

    def __init__(self) -> None:
        self.previous: tuple[float, int] | None = None
        self.rate: float | None = None

    def update(self, unix_s: float, generated: int) -> float | None:
        if self.previous:
            when, position = self.previous
            if unix_s > when and generated >= position:
                observed = (generated - position) / (unix_s - when)
                self.rate = (
                    observed if self.rate is None else 0.3 * observed + 0.7 * self.rate
                )
        self.previous = (unix_s, generated)
        return self.rate


def build_snapshot(
    *,
    snapshot: dict,
    prediction: dict | None,
    candidate_rate: float | None,
    initial_rate: dict,
    kv_bytes_per_token: int,
    release_tail_s: float | None,
    block_size: int = 16,
    safety_margin_s: float = 2.0,
    max_sample_age_s: float = 2.0,
    model_config_sha256: str | None = None,
) -> RiskUrgencySnapshot:
    """Use decode growth excluding isolated prefill, pending only once.

    Timing envelope assumes 0.5..1.5 current growth and 0.5..1.0 initial
    M2 effective bandwidth. Copy catches ongoing candidate decode. The
    configured release tail covers final sync/handoff/actual source release.
    These envelopes require pilot measurement; they are not guarantees.
    """
    snapshot = audit_finite(snapshot)
    p = audit_finite(prediction or {})
    candidate_rate = audit_finite(candidate_rate)
    initial_rate = audit_finite(initial_rate)
    row: dict[str, Any] = {
        "kind": "risk_urgency_snapshot",
        "format_version": 4,
        "capacity_model": "allocated_kv_plus_decode_growth",
        "prefill_capacity_policy": "OBSERVATION_ONLY",
        "source_guard_policy": "WARNING_NOT_START_DEADLINE",
        "unix_s": snapshot.get("unix_s"),
        "request_id": snapshot.get("request_id"),
        "migration_id": snapshot.get("migration_id"),
        "generated_tokens": snapshot.get("generated_tokens"),
        "prediction": p,
        "model_config_sha256": model_config_sha256,
        "candidate_decode_tokens_s": candidate_rate,
        "initial_rate": initial_rate,
        "block_size": block_size,
        "safety_margin_s": safety_margin_s,
        "probability_definition": "candidate_survives_projected_guard",
        "probability_is_pool_oom_risk": False,
        "time_bound_status": "ASSUMED_ENVELOPE_NOT_CALIBRATED_CI",
        "assumptions": [
            "current scheduled growth persists; pending prefill is observation only",
            "future arrivals/EOS change next snapshot",
            "growth 0.5..1.5x; effective M2 rate 0.5..1.0x",
            "release tail includes final sync/handoff/KV release",
        ],
        "H_tokens": None,
        "p_capacity_bounds": None,
        "p_guard_est_bounds": None,
        "r_guard_tokens": None,
        "T_guard_s": None,
        "T_guard_bounds_s": None,
        "T_release_s": None,
        "T_release_bounds_s": None,
        "U": None,
        "S_s": None,
        "U_bounds": None,
        "S_bounds_s": None,
        "physical_feasible": False,
        "physical_rejections": [],
        "guard_warnings": [],
        "guard_deadline_warning": False,
        "source_physical_headroom_tokens": None,
        "source_physical_capacity_exhausted": False,
        "source": snapshot,
    }
    reasons = row["physical_rejections"]
    now = snapshot.get("unix_s")
    for pool in ("source", "target"):
        sampled = snapshot.get(f"{pool}_sampled_unix_s")
        if (
            not isinstance(sampled, (int, float))
            or not math.isfinite(sampled)
            or not isinstance(now, (int, float))
            or not 0 <= now - sampled <= max_sample_age_s
        ):
            reasons.append(f"{pool}_telemetry_stale_or_missing")
    amounts = [
        snapshot.get(k)
        for k in (
            "source_free_kv_tokens",
            "source_guard_free_kv_tokens",
            "source_prefill_pending_kv_tokens",
        )
    ]
    if block_size <= 0 or any(not isinstance(x, int) or x < 0 for x in amounts):
        reasons.append("invalid_capacity")
        row["status"] = "INVALID_CAPACITY"
        return RiskUrgencySnapshot(row)
    free, guard, _pending = amounts
    # Account only for allocated KV; pending prefill remains in the snapshot.
    h = (
        (free // block_size) * block_size
        - math.ceil(guard / block_size) * block_size
    )
    row["H_tokens"] = h
    physical_headroom = (free // block_size) * block_size
    row["source_physical_headroom_tokens"] = physical_headroom
    row["source_physical_capacity_exhausted"] = physical_headroom <= 0
    if physical_headroom <= 0:
        reasons.append("source_physical_capacity_exhausted")
    row["p_capacity_bounds"] = remaining_ge_bounds(p, h)
    growth = snapshot.get("source_decode_growth_tokens_s")
    prefill_growth = snapshot.get("source_prefill_growth_tokens_s")
    row["prefill_growth_tokens_s"] = prefill_growth
    if prefill_growth is not None:
        if (not isinstance(prefill_growth, (int, float))
                or not math.isfinite(prefill_growth) or prefill_growth < 0):
            reasons.append("prefill_growth_invalid")
            growth = None
        elif isinstance(growth, (int, float)):
            growth += prefill_growth
        row["capacity_model"] = "allocated_kv_plus_scheduled_growth"
    row["growth_estimate_basis"] = (
        "SCHEDULED_TOKEN_EWMA_APPROXIMATION" if prefill_growth is not None
        else "LEGACY_DECODE_ONLY_PREFILL_RATE_UNAVAILABLE"
    )
    row["pool_growth_tokens_s"] = growth
    valid_growth = isinstance(growth, (int, float)) and math.isfinite(growth)
    valid_speed = (
        isinstance(candidate_rate, (int, float))
        and math.isfinite(candidate_rate)
        and candidate_rate > 0
    )
    if h <= 0:
        bounds = remaining_ge_bounds(p, 0)
        row.update(
            status="GUARD_REACHED" if bounds is not None else "PREDICTION_INVALID",
            T_guard_s=0.0,
            T_guard_bounds_s=[0.0, 0.0],
            r_guard_tokens=0,
            r_guard_integer_tokens=0,
            p_guard_est_bounds=bounds,
            guard_deadline_warning=True,
        )
        row["guard_warnings"].append("source_guard_reached")
    elif not valid_growth or growth <= 0:
        row["status"] = "NO_TRUSTED_POSITIVE_GROWTH"
        reasons.append("growth_evidence_unavailable")
    elif not valid_speed:
        tg = h / growth
        row.update(T_guard_s=tg, T_guard_bounds_s=[tg / 1.5, tg / 0.5])
        row["status"] = "NO_CANDIDATE_RATE"
        reasons.append("candidate_rate_unavailable")
    else:
        tg = h / growth
        row.update(
            T_guard_s=tg,
            T_guard_bounds_s=[tg / 1.5, tg / 0.5],
            r_guard_tokens=candidate_rate * tg,
        )
        row["p_guard_est_bounds"] = remaining_ge_bounds(p, candidate_rate * tg)
        row["r_guard_integer_tokens"] = math.ceil(candidate_rate * tg)
        row["status"] = (
            "VALID" if row["p_guard_est_bounds"] is not None else "PREDICTION_INVALID"
        )
    bandwidth = initial_rate.get("rate_bytes_s")
    context = snapshot.get("current_context_tokens")
    if (
        valid_speed
        and isinstance(bandwidth, (int, float))
        and math.isfinite(bandwidth)
        and bandwidth > 0
        and isinstance(context, int)
        and context > 0
        and kv_bytes_per_token > 0
        and release_tail_s is not None
        and math.isfinite(release_tail_s)
        and release_tail_s > 0
    ):
        produced_bytes_s = candidate_rate * kv_bytes_per_token
        if bandwidth * 0.5 > produced_bytes_s * 1.5:
            tr = (
                context * kv_bytes_per_token / (bandwidth - produced_bytes_s)
                + release_tail_s
            )
            upper = (
                context
                * kv_bytes_per_token
                / (bandwidth * 0.5 - produced_bytes_s * 1.5)
                + release_tail_s
            )
            row.update(T_release_s=tr, T_release_bounds_s=[tr, upper])
            history_s = context * kv_bytes_per_token / bandwidth
            row["release_components_point_s"] = {
                "history": history_s,
                "ongoing_decode_delta_catchup": tr - release_tail_s - history_s,
                "final_sync_handoff_source_release_allowance": release_tail_s,
            }
            if h <= 0:
                # A zero guard horizon has no finite urgency ratio. Keep U
                # unknown rather than fabricate a finite value or emit inf.
                row.update(
                    S_s=-tr - safety_margin_s,
                    S_bounds_s=[-upper - safety_margin_s, -tr - safety_margin_s],
                )
            elif row["T_guard_s"]:
                tg = row["T_guard_s"]
                low, high = row["T_guard_bounds_s"]
                row.update(
                    U=(tr + safety_margin_s) / tg,
                    S_s=tg - tr - safety_margin_s,
                    U_bounds=[
                        (tr + safety_margin_s) / high,
                        (upper + safety_margin_s) / low,
                    ],
                    S_bounds_s=[
                        low - upper - safety_margin_s,
                        high - tr - safety_margin_s,
                    ],
                )
        else:
            reasons.append("delta_cannot_catch_up")
    if row["T_release_bounds_s"] is None:
        reasons.append("release_time_unavailable")
    if row["S_bounds_s"] is not None and row["S_bounds_s"][0] <= 0:
        row["guard_deadline_warning"] = True
        row["guard_warnings"].append("source_release_may_miss_guard")
    if snapshot.get("state") != "LOCAL":
        reasons.append("request_not_local")
    if snapshot.get("channel_available") is not True:
        reasons.append("channel_unavailable")
    cap = p.get("max_remaining_output_tokens")
    target = snapshot.get("target_free_kv_tokens")
    target_pending = snapshot.get("target_prefill_pending_kv_tokens")
    if not isinstance(target_pending, int) or target_pending < 0:
        reasons.append("target_pending_reservation_unavailable")
    target_usage = snapshot.get("target_kv_usage_frac")
    target_guard = None
    if (
        isinstance(target, int)
        and isinstance(target_usage, (int, float))
        and math.isfinite(target_usage)
        and 0 <= target_usage < 1
    ):
        # Preserve the existing 85% target KV guard as a capacity reserve,
        # rather than a performance/waiting-count admission rule.
        target_guard = (
            math.ceil(target / (1 - target_usage) * 0.15 / block_size) * block_size
        )
    row["target_guard_tokens"] = target_guard
    row["target_guard_provenance"] = "existing_85pct_KV_guard_as_reserve"
    if (
        not isinstance(cap, int)
        or cap <= 1
        or not isinstance(context, int)
        or not isinstance(target, int)
        or target_guard is None
        or (
            isinstance(target_pending, int)
            and target - target_guard
            < math.ceil((context + cap) / block_size) * block_size
        )
    ):
        reasons.append("target_capped_kv_or_output_budget_unavailable")
    if row["p_guard_est_bounds"] is None:
        reasons.append("prediction_or_projected_risk_unavailable")
    if any("telemetry_stale_or_missing" in reason for reason in reasons):
        row["status"] = "STALE_TELEMETRY"
    if (
        model_config_sha256
        and p.get("model_config_sha256")
        and model_config_sha256 != p["model_config_sha256"]
    ):
        reasons.append("predictor_model_config_mismatch")
        row["status"] = "MODEL_CONFIG_MISMATCH"
    row["physical_feasible"] = not reasons
    return RiskUrgencySnapshot(row)
