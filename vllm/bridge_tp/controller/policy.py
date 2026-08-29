# SPDX-License-Identifier: Apache-2.0
"""Fast per-request escalation policy (paper Section 3.5).

All terms are expressed in seconds. The escalation test is one-dimensional:
solve Benefit = Cost for the break-even remaining length N*, then read the
probability of exceeding it off the survival table.

    Ben(r,t)  = w_grp(r) * N * (tau1 - tau4)
    Cost(r,t) = T_stall + T_dup + T_intf(b, load4) + T_margin
    N*        = Cost / (w_grp * (tau1 - tau4))
    p_worth   = P(N_remain > N* | produced)

Escalate when p_worth >= theta_esc(t) and the target can accept the request,
or unconditionally when near-term source OOM risk exceeds the safety level.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field, replace

from .events import (
    Action,
    Decision,
    MigrationState,
    PoolTelemetry,
    SourceRequestView,
)
from .predictor import SurvivalTable


@dataclass(frozen=True)
class TpotModel:
    """Per-token decode time as a function of pool load.

    Calibrated offline from P1 / P1-B by ``tools/bridge_tp/calibrate_tpot_model.py``.
    A linear model in the number of running requests is sufficient in the
    tested region and keeps the audit record interpretable; replace with a
    lookup table if the measured response is not linear.
    """

    base_s: float
    per_running_s: float = 0.0
    # Free-text provenance, e.g. "P1-B, Qwen2.5-14B, A100-PCIe, 2026-09-12".
    calibration_source: str = ""
    num_running_min: int = 0
    num_running_max: int = 1_000_000_000
    model_kind: str = "running_linear"
    load_knots: tuple[float, ...] = ()
    tpot_knots_s: tuple[float, ...] = ()
    min_load_frac: float = 0.0
    max_load_frac: float = 1.0

    def in_support(
        self, num_running: int, kv_usage_frac: float | None = None
    ) -> bool:
        """Whether the runtime predictor lies inside calibration support."""
        if self.model_kind == "load_piecewise_monotone":
            return (
                kv_usage_frac is not None
                and len(self.load_knots) == len(self.tpot_knots_s)
                and len(self.load_knots) >= 2
                and self.min_load_frac <= kv_usage_frac <= self.max_load_frac
            )
        if self.model_kind != "running_linear":
            return False
        return self.num_running_min <= num_running <= self.num_running_max

    def tpot_s(self, num_running: int, kv_usage_frac: float | None = None) -> float:
        if self.model_kind == "load_piecewise_monotone":
            if kv_usage_frac is None or not self.load_knots:
                return math.inf
            x = float(kv_usage_frac)
            if x <= self.load_knots[0]:
                return max(1e-6, float(self.tpot_knots_s[0]))
            if x >= self.load_knots[-1]:
                return max(1e-6, float(self.tpot_knots_s[-1]))
            right = bisect_right(self.load_knots, x)
            left = right - 1
            x0, x1 = self.load_knots[left], self.load_knots[right]
            y0, y1 = self.tpot_knots_s[left], self.tpot_knots_s[right]
            fraction = (x - x0) / max(1e-12, x1 - x0)
            return max(1e-6, float(y0 + fraction * (y1 - y0)))
        return max(1e-6, self.base_s + self.per_running_s * max(0, num_running))


@dataclass(frozen=True)
class InterferenceModel:
    """Target-side penalty attributable to migration traffic.

    ``penalty_s`` returns the expected added foreground time over the whole
    drain, so it is directly comparable with the benefit term. Formal Phase 9
    uses a paired request-level TPOT response over measured load/rate support;
    ``legacy_power`` remains only for old engineering configurations.

    CALIBRATE THIS. The default below is an order-of-magnitude anchor derived
    from the M3 measurement, not a substitute for the P2-D sweep on your
    platform. Derivation, so the number is auditable rather than plausible:

        moving 1 GiB at the measured 2.164 GiB/s takes 0.462 s;
        during that window a native request at 40.07 ms/token emits ~11.5
        tokens, each inflated by (148.83 - 40.07) = 108.76 ms at the tail;
        => ~1.25 s of added tail time per GiB, per running native request.

    An under-calibrated value here is not a small error: it sets the break-even
    remaining length N*, and a too-small penalty makes every request look worth
    migrating. Run ``tools/bridge_tp/replay_policy.py`` after calibrating and
    check that the migrate / do-not-migrate boundary actually falls inside the
    range of remaining lengths your workload produces.
    """

    # seconds of added foreground cost per GiB moved, at reference load
    s_per_gib_at_ref: float = 1.25
    ref_load_frac: float = 0.5
    load_exponent: float = 2.0
    # Free-text provenance, e.g. "P2-D sweep, A100-PCIe, 2026-09-14".
    # ControllerConfig.validate() refuses to start a run while this is empty.
    calibration_source: str = ""

    # ``legacy_power`` preserves old engineering configs.  Formal Phase 9
    # configs use ``rate_aware_tpot`` and the paired request-level TPOT fit.
    model_kind: str = "legacy_power"
    tpot_rate_coef_s2_per_gib: float = 0.0
    tpot_rate_load_coef_s2_per_gib: float = 0.0
    min_load_frac: float = 0.0
    max_load_frac: float = 1.0
    min_rate_gib_s: float = 0.0
    # A large finite default keeps legacy configs strict-JSON serializable.
    max_rate_gib_s: float = 1.0e30

    def in_support(self, target_load_frac: float, copy_rate_gib_s: float) -> bool:
        """Whether a rate-aware prediction stays inside calibration support."""
        if self.model_kind == "legacy_power":
            return True
        return (
            self.min_load_frac <= target_load_frac <= self.max_load_frac
            and self.min_rate_gib_s <= copy_rate_gib_s <= self.max_rate_gib_s
        )

    def incremental_tpot_s(
        self, target_load_frac: float, copy_rate_gib_s: float
    ) -> float:
        """Predict request-level native TPOT increase during copy traffic."""
        if self.model_kind != "rate_aware_tpot":
            raise ValueError("incremental_tpot_s requires rate_aware_tpot")
        if not self.in_support(target_load_frac, copy_rate_gib_s):
            return math.inf
        coefficient = (
            self.tpot_rate_coef_s2_per_gib
            + self.tpot_rate_load_coef_s2_per_gib * target_load_frac
        )
        return max(0.0, copy_rate_gib_s * coefficient)

    def penalty_s(
        self,
        bytes_to_move: int,
        target_load_frac: float,
        copy_rate_bytes_s: float | None = None,
        native_tpot_s: float | None = None,
    ) -> float:
        gib = max(0.0, bytes_to_move) / (1024.0**3)
        load = max(0.0, min(1.0, target_load_frac))
        if self.model_kind == "rate_aware_tpot":
            if gib == 0:
                return 0.0
            if copy_rate_bytes_s is None or native_tpot_s is None:
                raise ValueError(
                    "rate_aware_tpot requires copy_rate_bytes_s and native_tpot_s"
                )
            rate_gib_s = copy_rate_bytes_s / (1024.0**3)
            delta_tpot_s = self.incremental_tpot_s(load, rate_gib_s)
            if math.isinf(delta_tpot_s) or copy_rate_bytes_s <= 0:
                return math.inf
            drain_s = bytes_to_move / copy_rate_bytes_s
            native_tokens_during_drain = drain_s / max(1e-6, native_tpot_s)
            return delta_tpot_s * native_tokens_during_drain
        if self.model_kind != "legacy_power":
            raise ValueError(f"unknown interference model_kind {self.model_kind!r}")
        scale = ((load + 1e-6) / max(1e-6, self.ref_load_frac)) ** self.load_exponent
        return self.s_per_gib_at_ref * gib * scale


@dataclass(frozen=True)
class PolicyConfig:
    # KV footprint of one token across all layers, both K and V, in bytes.
    # 192 KiB for Qwen2.5-14B-Instruct: 48 layers * 2 * 8 heads * 128 dim * 2 B.
    kv_bytes_per_token: int = 196_608

    # escalation threshold
    theta_0: float = 0.60
    theta_min: float = 0.15
    alpha: float = 0.45

    # safety
    p_oom_force: float = 0.85
    min_output_tokens_before_eligible: int = 32
    min_survivors_for_confidence: int = 20
    # Risk assumed for a request that has already outrun the calibrated range
    # of the survival table. Reporting 0.0 there would make the OOM guard blind
    # to exactly the longest requests, which are the ones this paper is about;
    # reporting 1.0 would force every such request to escalate. This value is
    # deliberately below ``p_oom_force`` so it is visible in the audit log
    # without silently triggering an escalation on its own.
    p_oom_unsupported: float = 0.50

    # cost terms measured or budgeted, in seconds
    t_stall_s: float = 0.15
    t_restore_commit_s: float = 0.25
    t_margin_s: float = 0.10

    # group straggler multiplier (Section 2.3 calibration)
    w_group_max: float = 1.6

    # target admission guard
    max_target_kv_usage_frac: float = 0.85
    max_target_waiting: int = 4
    max_concurrent_migrations: int = 1


@dataclass
class RiskTracker:
    """EWMA of normalized TP1 pressure. Shared with the slow controller.

    IMPORTANT: ``preemptions_total`` must come from a monotone counter. vLLM
    also exports ``*_created`` gauges that hold the counter creation timestamp;
    reading those as a risk count silently produces a constant enormous value.
    ``telemetry.py`` refuses to parse ``_created`` series for this reason.
    """

    alpha: float = 0.08
    value: float = 0.0
    _last_preemptions: int | None = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def update(self, pool: PoolTelemetry) -> float:
        if self._last_preemptions is None:
            delta = 0
        else:
            delta = max(0, pool.preemptions_total - self._last_preemptions)
        self._last_preemptions = pool.preemptions_total
        # normalize: KV occupancy plus a saturating preemption term
        preempt_term = 1.0 - math.exp(-delta / 4.0)
        instantaneous = min(1.0, 0.7 * pool.kv_usage_frac + 0.3 * preempt_term)
        if not self._initialized:
            self.value = instantaneous
            self._initialized = True
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * instantaneous
        return self.value


class FastPolicy:
    """Decides whether a request should leave TP1, and at what rate."""

    def __init__(
        self,
        config: PolicyConfig,
        table: SurvivalTable,
        tpot_tp1: TpotModel,
        tpot_tp4: TpotModel,
        interference: InterferenceModel,
    ) -> None:
        self.cfg = config
        self.table = table
        self.tpot_tp1 = tpot_tp1
        self.tpot_tp4 = tpot_tp4
        self.interference = interference

    # ---- components ---------------------------------------------------
    def w_group(self, req: SourceRequestView) -> float:
        if req.group_id is None:
            return 1.0
        return self.cfg.w_group_max if req.is_group_longest else 1.0

    def horizon_tokens(self, tp1: PoolTelemetry, others_expected: float) -> float:
        """TP1 generation horizon H(t), in tokens (paper Section 3.5)."""
        return max(0.0, tp1.free_kv_tokens - others_expected)

    def p_oom(
        self, req: SourceRequestView, tp1: PoolTelemetry, others_expected: float
    ) -> float:
        horizon = self.horizon_tokens(tp1, others_expected)
        if horizon <= 0:
            # No headroom left at all: continuing to decode overruns the pool
            # whatever the prediction says. This branch must not consult the
            # table, or an out-of-support request would report zero risk while
            # the source is already full.
            return 1.0
        if not self.table.in_support(req.output_tokens):
            return self.cfg.p_oom_unsupported
        return self.table.p_remaining_gt(req.output_tokens, horizon)

    def theta_esc(self, risk_tp1: float) -> float:
        return max(
            self.cfg.theta_min, self.cfg.theta_0 - self.cfg.alpha * max(0.0, risk_tp1)
        )

    def migration_bytes(self, req: SourceRequestView) -> int:
        return int(req.kv_tokens) * int(self.cfg.kv_bytes_per_token)

    def cost_breakdown(
        self,
        req: SourceRequestView,
        tp4: PoolTelemetry,
        rate_bytes_s: float,
        tp1: PoolTelemetry,
    ) -> dict[str, float]:
        move_bytes = self.migration_bytes(req)
        drain_s = move_bytes / max(1.0, rate_bytes_s)
        tau1 = self.tpot_tp1.tpot_s(tp1.num_running, tp1.kv_usage_frac)
        tau4 = self.tpot_tp4.tpot_s(tp4.num_running, tp4.kv_usage_frac)
        # tokens the source produces during handoff that will be discarded
        dup_tokens = self.cfg.t_restore_commit_s / tau1
        return {
            "t_stall_s": self.cfg.t_stall_s,
            "t_dup_s": dup_tokens * tau1,
            "t_interference_s": self.interference.penalty_s(
                move_bytes,
                tp4.kv_usage_frac,
                copy_rate_bytes_s=rate_bytes_s,
                native_tpot_s=tau4,
            ),
            "t_margin_s": self.cfg.t_margin_s,
            "_drain_s": drain_s,
        }

    def break_even_tokens(
        self,
        req: SourceRequestView,
        tp1: PoolTelemetry,
        tp4: PoolTelemetry,
        rate_bytes_s: float,
    ) -> tuple[float, float, dict[str, float]]:
        """Return (N*, cost_s, cost_breakdown)."""
        tau1 = self.tpot_tp1.tpot_s(tp1.num_running, tp1.kv_usage_frac)
        tau4 = self.tpot_tp4.tpot_s(tp4.num_running, tp4.kv_usage_frac)
        gain_per_token = tau1 - tau4
        breakdown = self.cost_breakdown(req, tp4, rate_bytes_s, tp1)
        cost = sum(v for k, v in breakdown.items() if not k.startswith("_"))
        if not self.tpot_tp1.in_support(
            tp1.num_running, tp1.kv_usage_frac
        ) or not self.tpot_tp4.in_support(tp4.num_running, tp4.kv_usage_frac):
            return math.inf, cost, breakdown
        if gain_per_token <= 0:
            return math.inf, cost, breakdown
        denom = self.w_group(req) * gain_per_token
        return cost / denom, cost, breakdown

    # ---- main entry ---------------------------------------------------
    def evaluate(
        self,
        req: SourceRequestView,
        state: MigrationState,
        tp1: PoolTelemetry,
        tp4: PoolTelemetry,
        risk_tp1: float,
        rate_bytes_s: float,
        others_expected_tokens: float = 0.0,
        active_migrations: int = 0,
        now_unix_s: float = 0.0,
    ) -> Decision:
        def mk(action: Action, to: MigrationState, reason: str, **kw) -> Decision:
            return Decision(
                request_id=req.request_id,
                unix_s=now_unix_s,
                action=action,
                from_state=state,
                to_state=to,
                output_tokens=req.output_tokens,
                risk_tp1=risk_tp1,
                theta_esc=self.theta_esc(risk_tp1),
                rate_bytes_s=rate_bytes_s,
                reason=reason,
                **kw,
            )

        if state is not MigrationState.LOCAL:
            return mk(Action.STAY, state, "not in LOCAL; handled by state machine")

        if req.output_tokens < self.cfg.min_output_tokens_before_eligible:
            return mk(
                Action.STAY,
                MigrationState.LOCAL,
                f"too early: {req.output_tokens} < "
                f"{self.cfg.min_output_tokens_before_eligible} output tokens",
            )

        n_star, cost, breakdown = self.break_even_tokens(req, tp1, tp4, rate_bytes_s)
        p_worth = self.table.p_remaining_gt(req.output_tokens, n_star)
        e_remain = self.table.expected_remaining(req.output_tokens)
        p_oom = self.p_oom(req, tp1, others_expected_tokens)
        theta = self.theta_esc(risk_tp1)
        tau1 = self.tpot_tp1.tpot_s(tp1.num_running, tp1.kv_usage_frac)
        tau4 = self.tpot_tp4.tpot_s(tp4.num_running, tp4.kv_usage_frac)
        benefit = self.w_group(req) * e_remain * max(0.0, tau1 - tau4)

        common = dict(
            expected_remaining_tokens=e_remain,
            p_oom=p_oom,
            p_worth=p_worth,
            break_even_tokens=n_star,
            benefit_s=benefit,
            cost_s=cost,
            cost_breakdown={
                key: value
                for key, value in breakdown.items()
                if not key.startswith("_")
            },
        )

        target_full = (
            tp4.kv_usage_frac > self.cfg.max_target_kv_usage_frac
            or tp4.num_waiting > self.cfg.max_target_waiting
        )
        at_capacity = active_migrations >= self.cfg.max_concurrent_migrations

        # Safety override: near-term source OOM beats the benefit test, but the
        # target must still be able to physically accept the request.
        if p_oom >= self.cfg.p_oom_force:
            if target_full or at_capacity:
                return mk(
                    Action.STAY,
                    MigrationState.LOCAL,
                    f"forced by p_oom={p_oom:.3f} but target unavailable",
                    **common,
                )
            return replace(
                mk(
                    Action.START_SHADOW,
                    MigrationState.SHADOW,
                    f"forced escalation: p_oom={p_oom:.3f} >= {self.cfg.p_oom_force}",
                    **common,
                ),
                forced=True,
            )

        if not self.table.in_support(req.output_tokens):
            return mk(
                Action.STAY,
                MigrationState.LOCAL,
                f"beyond the calibrated range of the survival table "
                f"({req.output_tokens} tokens produced); refusing to extrapolate",
                **common,
            )
        if (
            self.table.n_survivors(req.output_tokens)
            < self.cfg.min_survivors_for_confidence
        ):
            return mk(
                Action.STAY,
                MigrationState.LOCAL,
                "insufficient survival-table support at this progress bucket",
                **common,
            )

        if at_capacity:
            return mk(
                Action.STAY,
                MigrationState.LOCAL,
                f"migration slots busy ({active_migrations})",
                **common,
            )
        if target_full:
            return mk(
                Action.STAY,
                MigrationState.LOCAL,
                f"target unavailable: kv={tp4.kv_usage_frac:.2f} "
                f"waiting={tp4.num_waiting}",
                **common,
            )
        if not self.tpot_tp1.in_support(
            tp1.num_running, tp1.kv_usage_frac
        ) or not self.tpot_tp4.in_support(tp4.num_running, tp4.kv_usage_frac):
            return mk(
                Action.STAY,
                MigrationState.LOCAL,
                "TPOT model outside calibrated num_running support/runtime support: "
                f"tp1_running={tp1.num_running}, tp1_kv={tp1.kv_usage_frac:.3f}, "
                f"tp4_running={tp4.num_running}, tp4_kv={tp4.kv_usage_frac:.3f}",
                **common,
            )
        if math.isinf(n_star):
            return mk(
                Action.STAY,
                MigrationState.LOCAL,
                "no per-token gain at current loads (tau4 >= tau1)",
                **common,
            )
        if p_worth >= theta:
            return mk(
                Action.START_SHADOW,
                MigrationState.SHADOW,
                f"p_worth={p_worth:.3f} >= theta={theta:.3f} (N*={n_star:.0f})",
                **common,
            )
        return mk(
            Action.STAY,
            MigrationState.LOCAL,
            f"p_worth={p_worth:.3f} < theta={theta:.3f} (N*={n_star:.0f})",
            **common,
        )

    def rank_candidates(
        self,
        candidates: list[SourceRequestView],
        tp1: PoolTelemetry,
        tp4: PoolTelemetry,
        rate_bytes_s: float,
    ) -> list[tuple[SourceRequestView, float]]:
        """Order eligible requests by escalation value, best first.

        On this hardware the benefit test is rarely the binding constraint: a
        14B request carries only ~192 KiB of KV per token, so a request with a
        few hundred tokens left clears the break-even bar easily. What is
        genuinely scarce is TP4 itself -- ``max_concurrent_migrations`` is 1 on
        a five-GPU testbed. The interesting decision is therefore *which*
        request gets the slot, not whether any request deserves one.

        The index is net expected seconds saved per byte moved, so a long
        request with a small KV footprint outranks a slightly longer one that
        would monopolise the link.
        """
        scored: list[tuple[SourceRequestView, float]] = []
        for req in candidates:
            if not self.table.in_support(req.output_tokens):
                continue
            n_star, cost, _ = self.break_even_tokens(req, tp1, tp4, rate_bytes_s)
            if math.isinf(n_star):
                continue
            e_remain = self.table.expected_remaining(req.output_tokens)
            tau1 = self.tpot_tp1.tpot_s(tp1.num_running, tp1.kv_usage_frac)
            tau4 = self.tpot_tp4.tpot_s(tp4.num_running, tp4.kv_usage_frac)
            benefit = self.w_group(req) * e_remain * max(0.0, tau1 - tau4)
            net = benefit - cost
            move_bytes = max(1, self.migration_bytes(req))
            scored.append((req, net / move_bytes))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def should_abandon(
        self,
        req: SourceRequestView,
        tp1: PoolTelemetry,
        tp4: PoolTelemetry,
        rate_bytes_s: float,
        risk_tp1: float,
    ) -> tuple[bool, str]:
        """Re-evaluation during Shadow: has the reason to migrate evaporated?"""
        n_star, _, _ = self.break_even_tokens(req, tp1, tp4, rate_bytes_s)
        p_worth = self.table.p_remaining_gt(req.output_tokens, n_star)
        theta = self.theta_esc(risk_tp1)
        # hysteresis: abandon only when clearly below, to avoid flapping
        if p_worth < 0.6 * theta:
            return True, f"benefit gone: p_worth={p_worth:.3f} << theta={theta:.3f}"
        if tp4.kv_usage_frac > self.cfg.max_target_kv_usage_frac + 0.10:
            return True, f"target risk too high: kv={tp4.kv_usage_frac:.2f}"
        return False, ""
