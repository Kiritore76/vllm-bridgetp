# SPDX-License-Identifier: Apache-2.0
"""Router intervention epochs; only post-drain evidence can justify a move."""

import math

from . import pressure_workload as workload


class RoutingRelief:
    def __init__(self):
        self.epoch = 0
        self.active = False
        self.started_at = None
        self.acknowledged_at = None
        self.last_source_admission_at = None
        self.recovery_since = None
        self.prefill_generation = None
        self.confirming = False
        self.phase = "NORMAL"
        self.redirected_requests = 0
        self.redirected_prompt_tokens = 0
        self.blocked_redirects = 0
        self.last_observed_at = None

    def snapshot(self):
        return dict(vars(self))

    def acknowledge(self, now):
        if self.active and self.acknowledged_at is None:
            self.acknowledged_at = now
            self.phase = "DRAIN_PREFILL"

    def admitted(self, pool_id, prompt_tokens, now, redirected, source_pool_id="tp1"):
        if pool_id == source_pool_id:
            self.last_source_admission_at = now
            self.confirming = False
            self.recovery_since = None
            if self.active:
                self.phase = "DRAIN_PREFILL"
                self.blocked_redirects += 1
        elif self.active and redirected:
            self.redirected_requests += 1
            self.redirected_prompt_tokens += prompt_tokens

    def update(self, source, pressure, ledger, cfg, now, quality_bad):
        """Return whether the independent degradation window must restart."""
        was_confirming = self.confirming
        self.confirming = False
        if not cfg.routing_relief:
            return False
        gap = self.last_observed_at is not None and not (
            0 <= now - self.last_observed_at <= cfg.max_age_s
        )
        self.last_observed_at = max(now, self.last_observed_at or now)
        fresh = workload.valid(source, cfg, now) and not pressure.missing
        if (
            not self.active
            and fresh
            and (
                (quality_bad and source.workload["load"] >= cfg.leave_ratio)
                or (pressure.capacity is not None and pressure.capacity >= 1)
            )
        ):
            self.active = True
            self.epoch += 1
            self.started_at = now
            self.acknowledged_at = None
            self.recovery_since = None
            self.prefill_generation = None
            self.phase = "WAIT_ROUTER"
            self.redirected_requests = 0
            self.redirected_prompt_tokens = 0
            self.blocked_redirects = 0
        if not self.active:
            return False
        count = source.prefill_generation
        counter_known = type(count) is int and count >= 0
        changed = self.prefill_generation != count
        if counter_known:
            self.prefill_generation = count
        barrier = max(
            self.started_at,
            self.acknowledged_at or self.started_at,
            self.last_source_admission_at or self.started_at,
        )
        drained = (
            fresh
            and counter_known
            and self.acknowledged_at is not None
            and source.sampled_at > barrier
            and ledger.unobserved_ingress(source.pool_id)[0] == 0
            and source.waiting == source.pending_prompt_tokens == 0
            and source.prefill_batch_tokens == 0
            and source.workload["prefill_s"] == 0
        )
        recovered = (
            drained
            and source.workload["load"] <= cfg.spare_ratio
            and (
                source.idle
                or (
                    source.decode_ratio is not None
                    and math.isfinite(source.decode_ratio)
                    and source.decode_ratio <= cfg.leave_ratio
                    and source.decode_samples >= cfg.min_decode_samples
                    and source.decode_sampled_at is not None
                    and 0 <= now - source.decode_sampled_at <= cfg.max_age_s
                )
            )
            and pressure.decode is not None
            and pressure.decode <= cfg.leave_ratio
            and pressure.capacity is not None
            and pressure.capacity <= cfg.spare_ratio
            and source.growth_blocks_s is not None
            and math.isfinite(source.growth_blocks_s)
            and source.growth_blocks_s >= 0
            and type(source.unallocated_prompt_blocks) is int
            and source.unallocated_prompt_blocks >= 0
        )
        if not recovered or changed or gap:
            self.recovery_since = None
        elif self.recovery_since is None:
            self.recovery_since = now
        elif now - self.recovery_since >= cfg.sustain_s:
            self.active = False
            self.phase = "NORMAL"
            return True
        if recovered:
            self.phase = "RECOVERING"
        elif self.acknowledged_at is None:
            self.phase = "WAIT_ROUTER"
        elif not drained:
            self.phase = "DRAIN_PREFILL"
        else:
            self.phase = "CONFIRMING"
            self.confirming = True
        return gap or changed or was_confirming != self.confirming or not drained
