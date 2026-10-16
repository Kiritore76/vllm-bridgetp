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
        self.first_health_window = None

    def decide(self, now, source, target, candidate, cfg, *, reversible):
        reason = None
        if cfg.shadow_cancel and reversible:
            s = classify(source, cfg, now)
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
                and not source.health_mixed_window
                and s.decode is not None
                and s.decode <= cfg.leave_ratio
                and source.waiting == source.pending_prompt_tokens == 0
            )
            # A slower target or short prediction does not remove the need
            # for source capacity/service relief. Cancel only after BOTH have
            # recovered. Actual EOS/mechanism exhaustion has separate cleanup.
            if recovered:
                reason = "source service and projected KV recovered"
        if (
            reason is None
            or reason != self.reason
            or self.last_tick is None
            or not 0 <= now - self.last_tick <= cfg.max_age_s
        ):
            self.since = now if reason else None
            self.first_health_window = source.health_sampled_at if reason else None
        self.reason, self.last_tick = reason, now
        if (
            reason
            and now - self.since >= cfg.cancel_sustain_s
            and source.health_sampled_at > self.first_health_window
        ):
            return reason
        return None
