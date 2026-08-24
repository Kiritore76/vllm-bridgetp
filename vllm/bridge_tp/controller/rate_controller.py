# SPDX-License-Identifier: Apache-2.0
"""Interference-aware migration-rate control (paper Section 3.6).

The paper states the objective

    min_b  T_stall + lambda * I_native + mu * I_migrating

but deliberately does not solve it online. The measured rate-to-tail-inflation
response (P2-B / P2-D) is steep, load dependent, and platform specific, which
makes a closed-form optimum both fragile and unnecessary. We approximate the
objective with a closed loop that treats native P99 TPOT as the feedback
signal: multiplicative increase while there is SLO slack, multiplicative
decrease once the target is exceeded, with a deadline term that can override
the feedback path when the drain would otherwise miss the safety horizon.

Calibration: b_min / b_max are initialized from the measured smooth and
interference-heavy bands on the evaluation platform (0.4-0.7 GiB/s and
1.37-2.16 GiB/s on A100-PCIe). They are starting points, not constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1024.0**3


@dataclass(frozen=True)
class RateConfig:
    b_min_bytes_s: float = 0.4 * GIB
    b_max_bytes_s: float = 1.2 * GIB
    b_start_bytes_s: float = 0.5 * GIB
    # absolute ceiling the deadline override may reach
    b_hard_max_bytes_s: float = 2.0 * GIB

    control_period_s: float = 0.2
    slo_p99_tpot_s: float = 0.080
    slack_frac: float = 0.90  # increase while p99 < slo * slack_frac
    mult_increase: float = 1.25
    mult_decrease: float = 0.50

    # do not react to a single noisy sample
    min_samples_before_action: int = 2


@dataclass
class RateController:
    cfg: RateConfig = field(default_factory=RateConfig)
    rate_bytes_s: float = 0.0
    _samples: int = field(default=0, repr=False)
    last_reason: str = ""

    def __post_init__(self) -> None:
        if self.rate_bytes_s <= 0:
            self.rate_bytes_s = self.cfg.b_start_bytes_s

    def _clamp(self, value: float, hard: bool = False) -> float:
        top = self.cfg.b_hard_max_bytes_s if hard else self.cfg.b_max_bytes_s
        return max(self.cfg.b_min_bytes_s, min(top, value))

    def step(
        self,
        native_p99_tpot_s: float,
        remaining_bytes: int,
        seconds_to_deadline: float | None = None,
    ) -> float:
        """Advance one control period and return the new rate in bytes/s.

        ``seconds_to_deadline`` is the time until the request's OOM horizon or
        safety deadline. Pass ``None`` when no deadline is active.
        """
        self._samples += 1
        cfg = self.cfg

        if self._samples < cfg.min_samples_before_action:
            self.last_reason = "warmup"
        elif native_p99_tpot_s > cfg.slo_p99_tpot_s:
            self.rate_bytes_s = self._clamp(self.rate_bytes_s * cfg.mult_decrease)
            self.last_reason = (
                f"backoff: native p99 {native_p99_tpot_s * 1e3:.1f}ms > "
                f"slo {cfg.slo_p99_tpot_s * 1e3:.1f}ms"
            )
        elif native_p99_tpot_s < cfg.slo_p99_tpot_s * cfg.slack_frac:
            self.rate_bytes_s = self._clamp(self.rate_bytes_s * cfg.mult_increase)
            self.last_reason = (
                f"increase: native p99 {native_p99_tpot_s * 1e3:.1f}ms has slack"
            )
        else:
            self.last_reason = "hold: inside SLO deadband"

        # deadline override, applied after the feedback path
        if seconds_to_deadline is not None and seconds_to_deadline > 0:
            required = remaining_bytes / seconds_to_deadline
            if required > self.rate_bytes_s:
                self.rate_bytes_s = self._clamp(required, hard=True)
                self.last_reason = (
                    f"deadline override: need {required / GIB:.2f} GiB/s to drain "
                    f"{remaining_bytes / GIB:.2f} GiB in {seconds_to_deadline:.2f}s"
                )
        return self.rate_bytes_s

    @property
    def rate_gib_s(self) -> float:
        """Phase 6/8 transport expects GiB/s via BRIDGETP_STREAM_RATE_GIB_S."""
        return self.rate_bytes_s / GIB

    def projected_drain_s(self, remaining_bytes: int) -> float:
        return remaining_bytes / max(1.0, self.rate_bytes_s)
