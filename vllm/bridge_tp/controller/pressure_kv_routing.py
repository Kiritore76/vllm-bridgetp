# SPDX-License-Identifier: Apache-2.0
"""Opt-in ingress KV protection, separate from ordinary service balancing."""

import math


def risk(pool, ledger, cfg, now, prompt_tokens=0):
    if (
        pool.block_size <= 0
        or pool.free_blocks is None
        or pool.free_blocks < 0
        or type(pool.unallocated_prompt_blocks) is not int
        or pool.unallocated_prompt_blocks < 0
        or pool.growth_blocks_s is None
        or not math.isfinite(pool.growth_blocks_s)
        or pool.growth_blocks_s < 0
        or not all(
            math.isfinite(t) and 0 <= now - t <= cfg.max_age_s
            for t in (pool.sampled_at, pool.published_at)
        )
    ):
        return None
    unpublished = sum(
        math.ceil(row.prefill_tokens / pool.block_size)
        for row in ledger.entries.values()
        if row.pool_id == pool.pool_id
        and row.kind == "admission"
        and row.state in {"ACTIVE", "COMMITTED"}
        and not row.prefill_observed
    )
    migration = ledger.pending(pool.pool_id)[0]
    new = math.ceil(prompt_tokens / pool.block_size)
    demand = pool.unallocated_prompt_blocks + unpublished + migration + new
    growth = math.ceil(
        pool.growth_blocks_s * (cfg.capacity_lookahead_s + now - pool.sampled_at)
    )
    return dict(
        sampled_at=pool.sampled_at,
        free_blocks=pool.free_blocks,
        native_unallocated_blocks=pool.unallocated_prompt_blocks,
        unpublished_prompt_blocks=unpublished,
        migration_pending_blocks=migration,
        new_prompt_blocks=new,
        projected_decode_growth_blocks=growth,
        guard_blocks=cfg.capacity_guard_blocks,
        projected_free_blocks=pool.free_blocks - demand - growth,
        pressure=(demand + growth + cfg.capacity_guard_blocks)
        / max(1, pool.free_blocks),
    )


class KVRouteProtection:
    def __init__(self):
        self.active = False
        self.recovery_since = None
        self.last_observed_at = None

    def observe(self, value, cfg, now):
        gap = self.last_observed_at is not None and not (
            0 <= now - self.last_observed_at <= cfg.max_age_s
        )
        self.last_observed_at = now
        if gap or value is None:
            self.recovery_since = None
        if value is None:
            return
        if value["pressure"] >= 1:
            self.active = True
            self.recovery_since = None
        elif self.active:
            if value["pressure"] > cfg.spare_ratio:
                self.recovery_since = None
            elif self.recovery_since is None:
                self.recovery_since = value["sampled_at"]
            elif value["sampled_at"] - self.recovery_since >= cfg.sustain_s:
                self.active = False
                self.recovery_since = None

    def snapshot(self):
        return dict(vars(self))


def preference(pools, ledger, cfg, now, prompt_tokens, fits, state):
    source, target = [risk(pool, ledger, cfg, now, prompt_tokens) for pool in pools]
    held = bool(state and state.get("active"))
    urgent = source is not None and source["pressure"] >= 1
    result = dict(
        source=source,
        target=target,
        hysteresis_active=held,
        redirect=False,
        mode="SERVICE_PRESSURE",
        reason="KV normal; original service routing",
    )
    if source is None or target is None:
        result["mode"] = "KV_EVIDENCE_UNAVAILABLE"
        result["reason"] = "KV evidence unavailable; original routing"
    elif urgent or held:
        result["mode"] = "KV_URGENT" if urgent else "KV_RECOVERY_HOLD"
        source_fit, target_fit = [fits[pool.pool_id]["status"] for pool in pools]
        queue_only = source_fit == target_fit == "NO_FIT"
        result["native_queue_only"] = queue_only
        if target["pressure"] >= source["pressure"]:
            result["reason"] = "target KV risk not lower; preserve original routing"
        elif target_fit != "FIT" and not queue_only:
            result["reason"] = "target prompt fit unavailable; preserve fit fallback"
        else:
            result.update(
                redirect=True,
                reason="lower projected KV risk; native queue"
                if queue_only
                else "lower projected KV risk and prompt fits",
            )
    return result
