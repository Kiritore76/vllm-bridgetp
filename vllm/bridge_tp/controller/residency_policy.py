# SPDX-License-Identifier: Apache-2.0
"""Advisory TP1 residency rule; never actuates a migration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResidencyConfig:
    long_probability_min: float = 0.8
    minimum_gain_s: float = 0.0
    start_margin_s: float = 2.0

    def validate(self) -> None:
        if not 0 < self.long_probability_min <= 1:
            raise ValueError("long probability threshold must be in (0, 1]")
        if self.minimum_gain_s < 0 or self.start_margin_s < 0:
            raise ValueError("residency margins cannot be negative")


@dataclass(frozen=True)
class ResidencyEvidence:
    prefill_complete: bool
    state: str
    source_fresh: bool
    source_free_kv_tokens: int | None
    source_guard_kv_tokens: int | None
    unallocated_prefill_tokens: int | None
    safe_pool_growth_tokens_s: float | None
    ready_best_s: float | None
    target_safe: bool | None
    channel_available: bool | None
    long_probability_lower: float | None = None
    gain_lower_s: float | None = None


@dataclass(frozen=True)
class ResidencyDecision:
    action: str
    reason: str
    headroom_tokens: int | None
    time_to_guard_s: float | None
    long_probability_lower: float | None
    gain_lower_s: float | None

    def to_json(self) -> dict:
        row = asdict(self)
        if row["time_to_guard_s"] == math.inf:
            row["time_to_guard_s"] = None
        return row


def decide_residency(
    evidence: ResidencyEvidence, config: ResidencyConfig = ResidencyConfig()
) -> ResidencyDecision:
    """Prefer TP1 until measured capacity or credible benefit justifies Shadow.

    ``CAPACITY_PROTECT`` is an advisory need for admission control or another
    safety path; it does not authorize unsafe early commit or source eviction.
    """
    config.validate()
    headroom = None
    time_to_guard = None

    def result(action: str, reason: str) -> ResidencyDecision:
        return ResidencyDecision(
            action, reason, headroom, time_to_guard,
            evidence.long_probability_lower, evidence.gain_lower_s,
        )

    if not evidence.source_fresh:
        return result("CAPACITY_UNKNOWN", "source telemetry is stale")
    if any(value is None for value in (
        evidence.source_free_kv_tokens, evidence.source_guard_kv_tokens,
        evidence.unallocated_prefill_tokens,
        evidence.safe_pool_growth_tokens_s,
    )):
        return result("CAPACITY_UNKNOWN", "source capacity evidence is incomplete")
    assert evidence.source_free_kv_tokens is not None
    assert evidence.source_guard_kv_tokens is not None
    assert evidence.unallocated_prefill_tokens is not None
    assert evidence.safe_pool_growth_tokens_s is not None
    if (
        evidence.source_free_kv_tokens < 0
        or evidence.source_guard_kv_tokens < 0
        or evidence.unallocated_prefill_tokens < 0
        or not math.isfinite(evidence.safe_pool_growth_tokens_s)
        or evidence.safe_pool_growth_tokens_s < 0
    ):
        return result("CAPACITY_UNKNOWN", "source capacity evidence is invalid")

    headroom = (
        evidence.source_free_kv_tokens - evidence.source_guard_kv_tokens
        - evidence.unallocated_prefill_tokens
    )
    growth = evidence.safe_pool_growth_tokens_s
    time_to_guard = (
        0.0 if headroom <= 0 else headroom / growth if growth > 0 else math.inf
    )

    if not evidence.prefill_complete or evidence.state != "LOCAL":
        return result("STAY_TP1", "request is not eligible")

    if headroom <= 0:
        if evidence.target_safe and evidence.channel_available:
            return result(
                "CAPACITY_PROTECT_AND_START_SHADOW", "source guard reached"
            )
        return result("CAPACITY_PROTECT", "source guard reached; target unavailable")

    ready = evidence.ready_best_s
    if ready is not None and (not math.isfinite(ready) or ready < 0):
        return result("CAPACITY_UNKNOWN", "preparation estimate is invalid")
    if ready is None:
        return result("CAPACITY_UNKNOWN", "preparation time is uncalibrated")

    if time_to_guard <= ready + config.start_margin_s:
        if evidence.target_safe and evidence.channel_available:
            if time_to_guard <= ready:
                return result(
                    "CAPACITY_PROTECT_AND_START_SHADOW",
                    "preparation may finish after the source guard",
                )
            return result("START_SHADOW_CAPACITY", "latest safe start approaching")
        return result("CAPACITY_PROTECT", "capacity due; target unavailable")

    if not evidence.target_safe or not evidence.channel_available:
        return result("STAY_TP1", "target or channel unavailable")
    probability = evidence.long_probability_lower
    gain = evidence.gain_lower_s
    if (
        probability is None or gain is None
        or not math.isfinite(probability) or not 0 <= probability <= 1
        or not math.isfinite(gain)
    ):
        return result("STAY_TP1", "long-request benefit is uncalibrated")
    if (
        probability >= config.long_probability_min
        and gain > config.minimum_gain_s
    ):
        return result("START_SHADOW_LONG", "credible long-request gain")
    return result("STAY_TP1", "TP1 residency preferred")
