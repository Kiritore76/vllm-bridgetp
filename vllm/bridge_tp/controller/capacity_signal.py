# SPDX-License-Identifier: Apache-2.0
"""Auditable source-headroom signal for the Phase 9 CAP-0 pilot.

This is intentionally not named ``p_cap`` or ``p_oom``: it is a deterministic
reachability signal, not a calibrated probability.  It uses only telemetry
already observable at the current control tick and never reads the pilot's
future arrival manifest.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass


@dataclass
class CapacityPilotConfig:
    """Default-off configuration for the single-anchor capacity pilot."""

    enabled: bool = False
    # Keep the reachability experiment attributable to this signal.  Set
    # False only for a later interaction study with the performance policy.
    exclusive_trigger_path: bool = True
    guard_free_kv_tokens: int = 0
    trigger_time_to_guard_s: float = 8.0
    clear_time_to_guard_s: float = 20.0
    ewma_alpha: float = 0.35
    minimum_samples: int = 4
    minimum_decline_tokens_s: float = 1.0
    maximum_observation_gap_s: float = 2.0

    def validate(self) -> None:
        if self.enabled and self.guard_free_kv_tokens <= 0:
            raise ValueError(
                "capacity_pilot.guard_free_kv_tokens must be measured and positive"
            )
        if self.trigger_time_to_guard_s <= 0:
            raise ValueError("capacity_pilot trigger_time_to_guard_s must be positive")
        if self.clear_time_to_guard_s <= self.trigger_time_to_guard_s:
            raise ValueError(
                "capacity_pilot clear_time_to_guard_s must exceed the trigger"
            )
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("capacity_pilot ewma_alpha must be in (0, 1]")
        if self.minimum_samples < 2:
            raise ValueError("capacity_pilot minimum_samples must be at least 2")
        if self.minimum_decline_tokens_s < 0:
            raise ValueError(
                "capacity_pilot minimum_decline_tokens_s cannot be negative"
            )
        if self.maximum_observation_gap_s <= 0:
            raise ValueError(
                "capacity_pilot maximum_observation_gap_s must be positive"
            )


@dataclass(frozen=True)
class CapacitySignal:
    sampled_unix_s: float
    free_kv_tokens: int
    guard_free_kv_tokens: int
    decline_rate_tokens_s: float
    time_to_guard_s: float
    active: bool
    transition: str
    samples: int
    reason: str
    sustained_decline_rate_tokens_s: float | None = None
    prefill_pending_kv_tokens: int | None = None
    prefill_scheduled_tokens_total: int | None = None
    decode_scheduled_tokens_total: int | None = None
    decode_growth_tokens_s: float | None = None
    prefill_growth_tokens_s: float | None = None

    def to_json(self) -> dict:
        value = asdict(self)
        # JSON has no portable infinity.  Null means the measured free-token
        # trend is not currently declining toward the guard.
        if not math.isfinite(self.time_to_guard_s):
            value["time_to_guard_s"] = None
        return value


class CapacityHeadroomTracker:
    """EWMA decline estimator with enter/clear hysteresis."""

    def __init__(self, config: CapacityPilotConfig) -> None:
        config.validate()
        self.config = config
        self._previous_free: int | None = None
        self._previous_unix_s: float | None = None
        self._decline_rate = 0.0
        self._samples = 0
        self._active = False
        self._recent_declines: deque[float] = deque(maxlen=3)
        self._previous_decode_tokens: int | None = None
        self._decode_growth: float | None = None
        self._previous_prefill_tokens: int | None = None
        self._prefill_growth: float | None = None

    @property
    def active(self) -> bool:
        return self._active

    def update(
        self,
        free_kv_tokens: int,
        sampled_unix_s: float,
        *,
        prefill_pending_kv_tokens: int | None = None,
        prefill_scheduled_tokens_total: int | None = None,
        decode_scheduled_tokens_total: int | None = None,
    ) -> CapacitySignal:
        cfg = self.config
        free = max(0, int(free_kv_tokens))
        now = float(sampled_unix_s)
        transition = "DISABLED" if not cfg.enabled else "WARMUP"
        reason = "capacity pilot disabled" if not cfg.enabled else "warming up"
        dt = (
            now - self._previous_unix_s
            if self._previous_unix_s is not None else None
        )
        if (
            decode_scheduled_tokens_total is not None
            and self._previous_decode_tokens is not None
            and dt is not None
            and 0 < dt <= cfg.maximum_observation_gap_s
            and decode_scheduled_tokens_total >= self._previous_decode_tokens
        ):
            observed = (
                decode_scheduled_tokens_total - self._previous_decode_tokens
            ) / dt
            self._decode_growth = (
                observed if self._decode_growth is None
                else cfg.ewma_alpha * observed
                + (1.0 - cfg.ewma_alpha) * self._decode_growth
            )
        else:
            self._decode_growth = None
        self._previous_decode_tokens = decode_scheduled_tokens_total

        if (
            prefill_scheduled_tokens_total is not None
            and self._previous_prefill_tokens is not None
            and dt is not None
            and 0 < dt <= cfg.maximum_observation_gap_s
            and prefill_scheduled_tokens_total >= self._previous_prefill_tokens
        ):
            observed = (
                prefill_scheduled_tokens_total - self._previous_prefill_tokens
            ) / dt
            self._prefill_growth = (
                observed if self._prefill_growth is None
                else cfg.ewma_alpha * observed
                + (1.0 - cfg.ewma_alpha) * self._prefill_growth
            )
        else:
            self._prefill_growth = None
        self._previous_prefill_tokens = prefill_scheduled_tokens_total

        if self._previous_free is not None and self._previous_unix_s is not None:
            assert dt is not None
            if 0 < dt <= cfg.maximum_observation_gap_s:
                instantaneous = max(0.0, (self._previous_free - free) / dt)
                self._recent_declines.append(instantaneous)
                if self._samples <= 1:
                    self._decline_rate = instantaneous
                else:
                    alpha = cfg.ewma_alpha
                    self._decline_rate = (
                        alpha * instantaneous + (1.0 - alpha) * self._decline_rate
                    )
            else:
                # A stale or reordered sample must not manufacture urgency.
                self._decline_rate = 0.0
                self._recent_declines.clear()

        self._previous_free = free
        self._previous_unix_s = now
        self._samples += 1

        excess = max(0, free - cfg.guard_free_kv_tokens)
        if free <= cfg.guard_free_kv_tokens:
            time_to_guard = 0.0
        elif self._decline_rate >= cfg.minimum_decline_tokens_s:
            time_to_guard = excess / self._decline_rate
        else:
            time_to_guard = math.inf

        if cfg.enabled and self._samples >= cfg.minimum_samples:
            if not self._active and (
                free <= cfg.guard_free_kv_tokens
                or time_to_guard <= cfg.trigger_time_to_guard_s
            ):
                self._active = True
                transition = "ENTER"
                reason = "measured TP1 headroom is approaching the guard"
            elif self._active and (
                free > cfg.guard_free_kv_tokens
                and time_to_guard >= cfg.clear_time_to_guard_s
            ):
                self._active = False
                transition = "CLEAR"
                reason = "measured TP1 headroom recovered beyond the clear threshold"
            elif self._active:
                transition = "HOLD"
                reason = "capacity pressure remains inside the hysteresis band"
            else:
                transition = "NORMAL"
                reason = "measured TP1 headroom is outside the trigger threshold"

        return CapacitySignal(
            sampled_unix_s=now,
            free_kv_tokens=free,
            guard_free_kv_tokens=cfg.guard_free_kv_tokens,
            decline_rate_tokens_s=self._decline_rate,
            time_to_guard_s=time_to_guard,
            active=self._active if cfg.enabled else False,
            transition=transition,
            samples=self._samples,
            reason=reason,
            sustained_decline_rate_tokens_s=(
                statistics.median(self._recent_declines)
                if len(self._recent_declines) == self._recent_declines.maxlen
                else None
            ),
            prefill_pending_kv_tokens=prefill_pending_kv_tokens,
            prefill_scheduled_tokens_total=prefill_scheduled_tokens_total,
            decode_scheduled_tokens_total=decode_scheduled_tokens_total,
            decode_growth_tokens_s=self._decode_growth,
            prefill_growth_tokens_s=self._prefill_growth,
        )
