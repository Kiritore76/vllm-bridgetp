# SPDX-License-Identifier: Apache-2.0
"""Conservative online admission for the first autonomous Shadow start."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .manager_m0 import RuntimeSnapshot
from .predictor import SurvivalTable


@dataclass(frozen=True)
class M1StartConfig:
    min_output_tokens: int = 32
    min_remaining_tokens: int = 128
    min_remaining_probability: float = 0.5
    min_survivors: int = 20
    min_target_owned_tokens: int = 64
    preparation_margin_s: float = 2.0
    source_release_tail_s: float | None = None
    max_target_kv_usage_frac: float = 0.85
    max_target_waiting: int = 4
    max_sample_age_s: float = 2.0

    def validate(self) -> None:
        if self.min_output_tokens < 0 or self.min_remaining_tokens <= 0:
            raise ValueError("M1 token thresholds must be positive")
        if not 0 < self.min_remaining_probability <= 1:
            raise ValueError("M1 remaining probability must be in (0, 1]")
        if self.min_survivors <= 0 or self.min_target_owned_tokens <= 0:
            raise ValueError("M1 survivor and target-token minima must be positive")
        if self.preparation_margin_s < 0 or self.max_sample_age_s <= 0:
            raise ValueError("M1 time margins must be non-negative")
        if self.source_release_tail_s is not None and (
            not math.isfinite(self.source_release_tail_s)
            or self.source_release_tail_s <= 0
        ):
            raise ValueError("M1 source release tail must be positive and finite")
        if not 0 < self.max_target_kv_usage_frac < 1:
            raise ValueError("M1 target KV limit must be in (0, 1)")
        if self.max_target_waiting < 0:
            raise ValueError("M1 target waiting limit must be non-negative")


@dataclass(frozen=True)
class M1StartDecision:
    action: str
    reason: str
    missing: tuple[str, ...] = ()
    expected_remaining_tokens: float | None = None
    remaining_probability: float | None = None
    survivors: int | None = None
    target_required_tokens: int | None = None
    estimated_preparation_s: float | None = None
    source_time_to_guard_s: float | None = None
    source_safe_headroom_tokens: int | None = None
    source_capacity_model: str | None = None
    source_release_tail_s: float | None = None

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        if value["source_time_to_guard_s"] == math.inf:
            value["source_time_to_guard_s"] = None
        return value


class M1StartController:
    """Start only with fresh capacity and supported remaining-length evidence."""

    def __init__(self, config: M1StartConfig) -> None:
        config.validate()
        self.config = config

    def decide(
        self,
        snapshot: RuntimeSnapshot,
        table: SurvivalTable,
        *,
        max_output_tokens: int,
        rate_bytes_s: float,
        kv_bytes_per_token: int,
    ) -> M1StartDecision:
        cfg = self.config
        if snapshot.state != "LOCAL":
            return M1StartDecision("STAY", "request is no longer local")
        missing = snapshot.freshness_errors(cfg.max_sample_age_s)
        for name in (
            "generated_tokens",
            "current_context_tokens",
            "source_free_kv_tokens",
            "source_guard_free_kv_tokens",
            "source_decode_growth_tokens_s",
            "source_prefill_pending_kv_tokens",
            "target_free_kv_tokens",
            "target_kv_usage_frac",
            "target_waiting",
            "channel_available",
        ):
            if getattr(snapshot, name) is None:
                missing.append(name)
        if missing:
            return M1StartDecision("STAY", "start evidence incomplete", tuple(missing))
        assert snapshot.generated_tokens is not None
        assert snapshot.current_context_tokens is not None
        assert snapshot.source_free_kv_tokens is not None
        assert snapshot.source_guard_free_kv_tokens is not None
        assert snapshot.source_decode_growth_tokens_s is not None
        assert snapshot.source_prefill_pending_kv_tokens is not None
        assert snapshot.target_free_kv_tokens is not None
        assert snapshot.target_kv_usage_frac is not None
        assert snapshot.target_waiting is not None
        if snapshot.generated_tokens < cfg.min_output_tokens:
            return M1StartDecision("STAY", "minimum output not reached")
        if max_output_tokens - snapshot.generated_tokens <= cfg.min_target_owned_tokens:
            return M1StartDecision("STAY", "insufficient target output budget")
        if not snapshot.channel_available:
            return M1StartDecision("STAY", "migration channel occupied")
        if (
            snapshot.target_kv_usage_frac > cfg.max_target_kv_usage_frac
            or snapshot.target_waiting > cfg.max_target_waiting
        ):
            return M1StartDecision("STAY", "target load exceeds admission guard")
        target_required = (
            snapshot.current_context_tokens
            + max_output_tokens - snapshot.generated_tokens
        )
        if snapshot.target_free_kv_tokens < target_required:
            return M1StartDecision(
                "STAY", "target KV cannot hold the capped request",
                target_required_tokens=target_required,
            )
        if not math.isfinite(rate_bytes_s) or rate_bytes_s <= 0:
            return M1StartDecision("STAY", "preparation rate unavailable")
        if kv_bytes_per_token <= 0:
            return M1StartDecision("STAY", "KV geometry unavailable")
        if cfg.source_release_tail_s is None:
            return M1StartDecision(
                "STAY", "source KV release time is uncalibrated",
                missing=("source_release_tail_s",),
            )
        growth = snapshot.source_decode_growth_tokens_s
        prefill_growth = snapshot.source_prefill_growth_tokens_s
        if prefill_growth is not None:
            if not math.isfinite(prefill_growth) or prefill_growth < 0:
                return M1StartDecision("STAY", "prefill growth invalid")
            growth += prefill_growth
        pending_prefill = snapshot.source_prefill_pending_kv_tokens
        if (
            not math.isfinite(growth) or growth < 0
            or pending_prefill < 0
        ):
            return M1StartDecision("STAY", "source capacity evidence invalid")
        history_transfer_s = (
            snapshot.current_context_tokens * kv_bytes_per_token / rate_bytes_s
        )
        prepare_s = history_transfer_s + max(
            cfg.preparation_margin_s, cfg.source_release_tail_s,
        )
        # Free KV already excludes allocated blocks. Unfinished prefill is
        # observed demand, not an additional allocation.
        headroom = (
            snapshot.source_free_kv_tokens - snapshot.source_guard_free_kv_tokens
        )
        time_to_guard = (
            0.0 if headroom <= 0 else
            math.inf if growth == 0 else headroom / growth
        )
        capacity_evidence = {
            "target_required_tokens": target_required,
            "estimated_preparation_s": prepare_s,
            "source_time_to_guard_s": time_to_guard,
            "source_safe_headroom_tokens": headroom,
            "source_capacity_model": (
                "allocated_kv_plus_scheduled_growth"
                if prefill_growth is not None
                else "allocated_kv_plus_decode_growth"
            ),
            "source_release_tail_s": cfg.source_release_tail_s,
        }
        if headroom <= 0 or time_to_guard <= prepare_s:
            return M1StartDecision(
                "STAY", "source guard may arrive before TP1 KV release",
                **capacity_evidence,
            )
        produced = snapshot.generated_tokens
        if not table.in_support(produced):
            return M1StartDecision("STAY", "survival table out of support")
        survivors = table.n_survivors(produced)
        if survivors < cfg.min_survivors:
            return M1StartDecision(
                "STAY", "too few survival-table observations",
                survivors=survivors,
            )
        expected = table.expected_remaining(produced)
        probability = table.p_remaining_gt(produced, cfg.min_remaining_tokens)
        evidence = {
            "expected_remaining_tokens": expected,
            "remaining_probability": probability,
            "survivors": survivors,
            **capacity_evidence,
        }
        if expected < cfg.min_remaining_tokens:
            return M1StartDecision(
                "STAY", "expected remaining work is short", **evidence
            )
        if probability < cfg.min_remaining_probability:
            return M1StartDecision(
                "STAY", "long-request probability is low", **evidence
            )
        return M1StartDecision(
            "START_SHADOW", "supported long request with safe preparation", **evidence
        )
