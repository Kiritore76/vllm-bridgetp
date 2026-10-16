# SPDX-License-Identifier: Apache-2.0
"""Causal, pre-freeze shadow abandonment; UNKNOWN never proves recovery."""

import math
from dataclasses import replace

from .pressure_policy import classify, health_valid


def without_shadow_demand(target, inventory, prefix):
    """Remove only this dormant replica's queue demand, never its allocated KV."""
    if inventory["sampled_at"] != target.sampled_at:
        return target
    rows = [
        row for row in inventory["requests"] if row["request_id"].startswith(prefix)
    ]
    if len(rows) != 1 or rows[0]["generated"] != 0:
        return target
    row = rows[0]
    return replace(
        target,
        waiting=max(0, target.waiting - int(not row["running"]))
        if target.waiting is not None
        else None,
        pending_prompt_tokens=max(
            0,
            target.pending_prompt_tokens
            - max(0, row["prompt_tokens"] - row["computed_tokens"]),
        )
        if target.pending_prompt_tokens is not None
        else None,
    )


class ShadowCancellation:
    def __init__(self):
        self.since = None
        self.reason = None
        self.last_tick = None

    def decide(self, now, source, target, candidate, cfg, *, reversible):
        reason = None
        if cfg.shadow_cancel and reversible:
            s, t = classify(source, cfg, now), classify(target, cfg, now)
            fresh_candidate = candidate is not None and all(
                math.isfinite(v) and 0 <= now - v <= cfg.max_age_s
                for v in (candidate.sampled_at, candidate.published_at)
            )
            cap = (
                max(0, min(8192, candidate.max_output_tokens) - candidate.generated)
                if fresh_candidate
                else None
            )
            # An upper bound proves short life; a low lower bound only means
            # uncertainty. Remaining-length predictions are already lag shifted.
            short = cap is not None and cap <= cfg.survival_horizon
            if fresh_candidate and candidate.prediction_status == "AVAILABLE":
                remaining = candidate.remaining
                if remaining is not None and remaining.cap == cap:
                    short |= (
                        remaining.probability_gt_bounds(cfg.survival_horizon)[1]
                        < cfg.min_survival_lower
                    )
            projected_kv_known = (
                cfg.capacity_lookahead_s > 0
                and source.growth_blocks_s is not None
                and math.isfinite(source.growth_blocks_s)
                and source.growth_blocks_s >= 0
                and type(source.unallocated_prompt_blocks) is int
                and source.unallocated_prompt_blocks >= 0
            )
            recovered = (
                projected_kv_known
                and s.capacity is not None
                and s.capacity <= cfg.spare_ratio
                and health_valid(source, cfg, now)
                and source.health_eta >= cfg.cancel_recover_eta
                and source.waiting == source.pending_prompt_tokens == 0
            )
            target_worse = (
                health_valid(target, cfg, now)
                and (
                    target.health_eta < cfg.health_min_efficiency
                    or target.health_load > cfg.health_target_load
                )
            ) or (
                t.prefill is not None
                and (target.waiting > 0 or target.pending_prompt_tokens > 0)
            )
            if short:
                reason = "remaining work too short"
            elif target_worse:
                reason = "target pressure increased"
            elif recovered:
                reason = "source service and projected KV recovered"
        if (
            reason is None
            or reason != self.reason
            or self.last_tick is None
            or not 0 <= now - self.last_tick <= cfg.max_age_s
        ):
            self.since = now if reason else None
        self.reason, self.last_tick = reason, now
        if reason and now - self.since >= cfg.cancel_sustain_s:
            return reason
        return None
