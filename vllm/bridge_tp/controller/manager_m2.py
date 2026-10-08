# SPDX-License-Identifier: Apache-2.0
"""Three-profile migration rate decisions from observable runtime state.

The policy is transport-independent. The online controller applies its selected
rate before arming Shadow and on subsequent ticks; replay uses the same rules.
"""

from __future__ import annotations

import math
from copy import copy
from dataclasses import asdict, dataclass
from typing import Any

from tools.bridge_tp.guard_forecast import snapshot_guard_time

from .manager_m0 import RuntimeSnapshot


@dataclass(frozen=True)
class M2RateConfig:
    low_bytes_s: float
    medium_bytes_s: float
    high_bytes_s: float
    max_sample_age_s: float = 2.0
    target_busy_running: int = 4
    target_busy_waiting: int = 2
    target_busy_kv_frac: float = 0.65
    delta_high_tokens: int = 64
    source_safe_horizon_s: float = 30.0
    preparation_margin_s: float = 5.0
    cooldown_s: float = 2.0
    stable_ticks: int = 3

    def validate(self) -> None:
        rates = (self.low_bytes_s, self.medium_bytes_s, self.high_bytes_s)
        if not all(math.isfinite(value) and value > 0 for value in rates):
            raise ValueError("M2 rates must be finite and positive")
        if not self.low_bytes_s < self.medium_bytes_s < self.high_bytes_s:
            raise ValueError("M2 rates must satisfy LOW < MEDIUM < HIGH")
        if self.max_sample_age_s <= 0 or self.source_safe_horizon_s <= 0:
            raise ValueError("M2 sample age and source horizon must be positive")
        if not 0 < self.target_busy_kv_frac < 1:
            raise ValueError("M2 target KV threshold must be in (0, 1)")
        if (
            self.target_busy_running < 1
            or self.target_busy_waiting < 0
            or self.delta_high_tokens < 0
        ):
            raise ValueError("M2 load and delta thresholds must be non-negative")
        if self.preparation_margin_s < 0 or self.cooldown_s < 0:
            raise ValueError("M2 margins must be non-negative")
        if self.stable_ticks < 1:
            raise ValueError("M2 stable_ticks must be positive")

    def rate(self, profile: str) -> float:
        return {
            "LOW": self.low_bytes_s,
            "MEDIUM": self.medium_bytes_s,
            "HIGH": self.high_bytes_s,
        }[profile]


@dataclass(frozen=True)
class M2RateDecision:
    action: str
    profile: str
    rate_bytes_s: float
    reason: str
    missing: tuple[str, ...] = ()
    source_time_to_guard_s: float | None = None
    source_capacity_model: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class M2RateController:
    """Select LOW/MEDIUM/HIGH with immediate safety upgrades and slow downshifts."""

    _ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    def __init__(
        self, config: M2RateConfig, *, force_initial_high: bool = False
    ) -> None:
        config.validate()
        self.config = config
        self.force_initial_high = force_initial_high
        self.profile = "MEDIUM"
        self._last_change_s: float | None = None
        self._candidate: str | None = None
        self._candidate_ticks = 0

    def preview_initial(self, snapshot: RuntimeSnapshot) -> M2RateDecision:
        """Choose the start rate without changing M2 state before M1 admits it."""
        return copy(self).decide(snapshot, before_start=True)

    def decide(
        self, snapshot: RuntimeSnapshot, *, before_start: bool = False
    ) -> M2RateDecision:
        cfg = self.config
        current_rate = cfg.rate(self.profile)
        if snapshot.state not in {"SHADOW", "READY_NOT_COMMITTED"} and not (
            before_start and snapshot.state == "LOCAL"
        ):
            return M2RateDecision("HOLD", self.profile, current_rate,
                                  "no active Shadow")
        missing = snapshot.freshness_errors(cfg.max_sample_age_s)
        for name in (
            "source_free_kv_tokens",
            "source_guard_free_kv_tokens",
            "target_waiting",
            "target_running",
            "target_kv_usage_frac",
        ):
            if getattr(snapshot, name) is None:
                missing.append(name)
        if (
            (
                snapshot.source_decode_growth_tokens_s is None
                or snapshot.source_prefill_pending_kv_tokens is None
            )
            and snapshot.source_pool_growth_tokens_s is None
            and snapshot.source_pool_sustained_growth_tokens_s is None
        ):
            missing.append("source growth")
        if missing:
            self._candidate = None
            self._candidate_ticks = 0
            return M2RateDecision(
                "HOLD", self.profile, current_rate, "rate evidence incomplete",
                tuple(missing),
            )
        assert snapshot.source_free_kv_tokens is not None
        assert snapshot.source_guard_free_kv_tokens is not None
        assert snapshot.target_waiting is not None
        assert snapshot.target_running is not None
        assert snapshot.target_kv_usage_frac is not None
        separated = (
            snapshot.source_prefill_pending_kv_tokens is not None
            and snapshot.source_decode_growth_tokens_s is not None
        )
        pending_prefill = (
            snapshot.source_prefill_pending_kv_tokens if separated else 0
        )
        assert pending_prefill is not None
        # Free KV already excludes allocated blocks. Unfinished prefill is
        # observed demand, not an additional allocation.
        headroom = (
            snapshot.source_free_kv_tokens
            - snapshot.source_guard_free_kv_tokens
        )
        if separated:
            growth = snapshot.source_decode_growth_tokens_s
            capacity_model = "allocated_kv_plus_decode_growth"
            prefill_growth = snapshot.source_prefill_growth_tokens_s
            if prefill_growth is not None:
                if not math.isfinite(prefill_growth) or prefill_growth < 0:
                    return M2RateDecision(
                        "HOLD", self.profile, current_rate,
                        "prefill growth invalid", ("source prefill growth",),
                    )
                assert growth is not None
                growth += prefill_growth
                capacity_model = "allocated_kv_plus_scheduled_growth"
        else:
            # Older traces do not distinguish prompt allocation from decode.
            growth = (
                snapshot.source_pool_sustained_growth_tokens_s
                if snapshot.source_pool_sustained_growth_tokens_s is not None
                else snapshot.source_pool_growth_tokens_s
            )
            capacity_model = "net_kv_growth_fallback"
        assert growth is not None
        if not math.isfinite(growth) or growth < 0:
            return M2RateDecision(
                "HOLD", self.profile, current_rate, "source growth invalid",
                ("source growth",),
            )
        if pending_prefill < 0:
            return M2RateDecision(
                "HOLD", self.profile, current_rate, "prefill observation invalid",
                ("source_prefill_pending_kv_tokens",),
            )
        if separated:
            horizon = snapshot_guard_time(snapshot.to_json(), headroom)
            if horizon is None:
                return M2RateDecision(
                    "HOLD", self.profile, current_rate,
                    "prefill allocation evidence missing",
                )
            capacity_model = "FINITE_UNALLOCATED_PREFILL_PLUS_DECODE"
        else:
            horizon = math.inf if growth == 0 else max(0.0, headroom / growth)
        remaining_history = None
        if (
            snapshot.history_total_bytes is not None
            and snapshot.history_resident_bytes is not None
        ):
            remaining_history = max(
                0, snapshot.history_total_bytes - snapshot.history_resident_bytes
            )
        preparation_s = (
            cfg.preparation_margin_s
            + (remaining_history or 0) / cfg.high_bytes_s
        )
        if (
            headroom <= 0
            or horizon <= max(cfg.source_safe_horizon_s, preparation_s)
        ):
            desired, reason = "HIGH", "source guard horizon is short"
        elif (
            snapshot.delta_lag_tokens is not None
            and snapshot.delta_lag_tokens >= cfg.delta_high_tokens
        ):
            desired, reason = "HIGH", "delta backlog is high"
        elif (
            snapshot.target_waiting >= cfg.target_busy_waiting
            or snapshot.target_running >= cfg.target_busy_running
            or snapshot.target_kv_usage_frac >= cfg.target_busy_kv_frac
        ):
            desired, reason = "LOW", "target is busy and source is safe"
        else:
            desired, reason = "MEDIUM", "balanced source and target state"
        if before_start and desired == "LOW" and remaining_history is not None:
            low_preparation_s = (
                cfg.preparation_margin_s
                + remaining_history / cfg.low_bytes_s
            )
            if horizon <= low_preparation_s:
                desired, reason = "HIGH", "LOW cannot finish before source guard"
        if before_start and self.force_initial_high:
            desired, reason = "HIGH", "diagnostic HIGH transfer smoke"

        if desired == self.profile:
            self._candidate = None
            self._candidate_ticks = 0
            return M2RateDecision(
                "HOLD", self.profile, current_rate, reason,
                source_time_to_guard_s=None if math.isinf(horizon) else horizon,
                source_capacity_model=capacity_model,
            )
        if not before_start and self._ORDER[desired] < self._ORDER[self.profile]:
            if self._candidate == desired:
                self._candidate_ticks += 1
            else:
                self._candidate = desired
                self._candidate_ticks = 1
            elapsed = (
                math.inf if self._last_change_s is None
                else snapshot.unix_s - self._last_change_s
            )
            if self._candidate_ticks < cfg.stable_ticks or elapsed < cfg.cooldown_s:
                return M2RateDecision(
                    "HOLD", self.profile, current_rate,
                    f"waiting for stable downshift to {desired}",
                    source_time_to_guard_s=(
                        None if math.isinf(horizon) else horizon
                    ),
                    source_capacity_model=capacity_model,
                )
        self.profile = desired
        self._last_change_s = snapshot.unix_s
        self._candidate = None
        self._candidate_ticks = 0
        return M2RateDecision(
            "SET_RATE", desired, cfg.rate(desired), reason,
            source_time_to_guard_s=None if math.isinf(horizon) else horizon,
            source_capacity_model=capacity_model,
        )
