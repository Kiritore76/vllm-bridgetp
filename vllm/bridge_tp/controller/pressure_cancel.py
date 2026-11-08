# SPDX-License-Identifier: Apache-2.0
"""Causal, pre-freeze shadow abandonment; UNKNOWN never proves recovery."""

import math
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

from . import pressure_workload as workload
from .pressure_policy import classify, health_valid
from .unified_scheduler import source_recovery_ready


def assess_unified_cancellations(audit, max_age_s):
    """Check actual recovery cancellations, without demanding zero cancellations."""
    decisions = [e for e in audit if e.get("kind") == "pressure_cancel_decision"]
    requested = [
        e
        for e in audit
        if e.get("kind") == "pressure_cancel_requested"
        and e.get("reason") == "source service and projected KV recovered"
    ]
    errors = []
    for e in decisions:
        source = e.get("source") or {}
        if not (
            e.get("scheduler") == "unified"
            and all(k in source for k in ("pool_id", "sampled_at", "published_at"))
            and type(e.get("unix_s")) in (int, float)
            and source_recovery_ready(
                e.get("source_need"), SimpleNamespace(**source), e["unix_s"], max_age_s
            )
        ):
            errors.append(
                dict(
                    migration_id=e.get("migration_id"),
                    error="recovery lacks fresh CLEAR source-demand evidence",
                )
            )

    def identities(events):
        return Counter((e.get("migration_id"), e.get("request_id")) for e in events)

    if identities(decisions) != identities(requested):
        errors.append(dict(error="recovery decisions and cancellation requests differ"))
    sources = sorted(
        {
            e["source_need"]["pool_id"]
            for e in audit
            if e.get("kind") == "pressure_cancel_source_need"
            and isinstance(e.get("source_need"), dict)
            and e["source_need"].get("state") == "NEEDED"
            and isinstance(e["source_need"].get("pool_id"), str)
            and e["source_need"]["pool_id"] == e["source_need"].get("expected_pool_id")
        }
    )
    return dict(
        passed=not errors,
        recovery_decisions=len(decisions),
        recovery_requests=len(requested),
        observed_needed_sources=sources,
        errors=errors,
    )


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

    def decide(
        self,
        now,
        source,
        target,
        candidate,
        cfg,
        *,
        reversible,
        scheduler="pressure",
        source_need=None,
    ):
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
            if cfg.workload_pressure:
                recovered = (
                    projected_kv_known
                    and s.capacity is not None
                    and s.capacity <= cfg.spare_ratio
                    and workload.valid(source, cfg, now)
                    and source.workload["load"] <= cfg.spare_ratio
                    and source.workload["prefill_s"] == 0
                    and source.decode_ratio is not None
                    and math.isfinite(source.decode_ratio)
                    and source.decode_ratio <= cfg.leave_ratio
                    and source.decode_samples >= cfg.min_decode_samples
                    and source.decode_sampled_at is not None
                    and 0 <= now - source.decode_sampled_at <= cfg.max_age_s
                    and source.prefill_batch_tokens == 0
                    and source.waiting == source.pending_prompt_tokens == 0
                )
            # A slower target or short prediction does not remove the need
            # for source capacity/service relief. Cancel only after BOTH have
            # recovered. Actual EOS/mechanism exhaustion has separate cleanup.
            if scheduler == "unified":
                recovered = recovered and source_recovery_ready(
                    source_need, source, now, cfg.max_age_s
                )
            if recovered:
                reason = "source service and projected KV recovered"
        if (
            reason is None
            or reason != self.reason
            or self.last_tick is None
            or not 0 <= now - self.last_tick <= cfg.max_age_s
        ):
            self.since = now if reason else None
            self.first_health_window = (
                source.workload["sampled_at"]
                if cfg.workload_pressure and reason
                else source.health_sampled_at
                if reason
                else None
            )
        self.reason, self.last_tick = reason, now
        if (
            reason
            and now - self.since >= cfg.cancel_sustain_s
            and (
                source.workload["sampled_at"]
                if cfg.workload_pressure
                else source.health_sampled_at
            )
            > self.first_health_window
        ):
            return reason
        return None
