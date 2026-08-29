# SPDX-License-Identifier: Apache-2.0
"""Phase 9: survival predictor and fast escalation policy.

python -m unittest tests.bridge_tp.test_phase9_predictor_policy
"""

from __future__ import annotations

import json
import math
import unittest

from tools.bridge_tp.build_survival_table import load_lengths
from vllm.bridge_tp.controller.events import (
    Action,
    MigrationState,
    PoolTelemetry,
    SourceRequestView,
)
from vllm.bridge_tp.controller.policy import (
    FastPolicy,
    InterferenceModel,
    PolicyConfig,
    RiskTracker,
    TpotModel,
)
from vllm.bridge_tp.controller.predictor import SurvivalTable

GIB = 1024.0**3

# A deliberately heavy-tailed synthetic population: most requests are short,
# a small fraction run very long. This is the shape M1 measures on the real
# trace, and it is what makes the conditional survival query non-trivial.
POPULATION = [16] * 400 + [64] * 300 + [256] * 200 + [1024] * 80 + [4096] * 20


def make_table() -> SurvivalTable:
    return SurvivalTable.from_output_lengths(POPULATION, source="unit-test")


def pool(
    running=1, waiting=0, kv=0.2, preempt=0, p99=0.05, free_blocks=1000, block=16
) -> PoolTelemetry:
    return PoolTelemetry(
        num_running=running,
        num_waiting=waiting,
        kv_usage_frac=kv,
        preemptions_total=preempt,
        p99_tpot_s=p99,
        mean_tpot_s=p99 * 0.6,
        free_kv_blocks=free_blocks,
        block_size=block,
    )


def request_view(output=200, computed=200, rid="r1", **kw) -> SourceRequestView:
    return SourceRequestView(
        request_id=rid,
        prompt_tokens=500,
        output_tokens=output,
        computed_tokens=computed,
        pending_tokens=1,
        arrival_unix_s=0.0,
        last_token_unix_s=10.0,
        **kw,
    )


class TestSurvivalTable(unittest.TestCase):
    def setUp(self) -> None:
        self.table = make_table()

    def test_conditioning_matches_manual_computation(self):
        # past 64 produced tokens, the 256/1024/4096 cohorts survive
        self.assertEqual(self.table.n_survivors(64), 300)
        # P(remaining > 192 | produced 64) = (80 + 20) / 300
        self.assertAlmostEqual(self.table.p_remaining_gt(64, 192), 100 / 300, places=9)
        self.assertEqual(self.table.p_remaining_gt(64, 0), 1.0)
        self.assertEqual(self.table.p_remaining_gt(64, math.inf), 0.0)

    def test_survival_is_monotone_in_x(self):
        previous = 1.0
        for x in (0, 10, 100, 1000, 5000):
            current = self.table.p_remaining_gt(128, x)
            self.assertLessEqual(current, previous)
            previous = current

    def test_surviving_population_is_longer_on_average(self):
        self.assertGreater(
            self.table.expected_remaining(512), self.table.expected_remaining(0)
        )

    def test_query_beyond_calibrated_range_reports_no_support(self):
        """The last bucket must not answer for a request that outran the trace.

        Without an explicit support bound, ``_bucket_index`` clamps to the top
        bucket, so a request at 100k tokens would silently receive the
        statistics of a 2048-token request -- confidence where there is none.
        """
        self.assertTrue(self.table.in_support(1000))
        self.assertFalse(self.table.in_support(100_000))
        self.assertEqual(self.table.n_survivors(100_000), 0)
        self.assertEqual(self.table.p_remaining_gt(100_000, 1), 0.0)
        self.assertEqual(self.table.expected_remaining(100_000), 0.0)

    def test_support_bound_survives_a_roundtrip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            self.table.save(path)
            back = SurvivalTable.load(path)
            self.assertEqual(back.max_observed_length, max(POPULATION))
            self.assertFalse(back.in_support(100_000))

    def test_hand_built_table_is_unbounded_by_default(self):
        table = SurvivalTable(bucket_edges=(0,), remaining=((1, 2, 3),))
        self.assertTrue(table.in_support(10**9))

    def test_quantile_and_uncertainty_are_well_defined(self):
        self.assertGreater(self.table.quantile_remaining(0, 0.9), 0)
        self.assertGreaterEqual(self.table.uncertainty(0), 0)

    def test_roundtrip_through_disk(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            self.table.save(path)
            back = SurvivalTable.load(path)
            self.assertEqual(back.bucket_edges, self.table.bucket_edges)
            self.assertEqual(
                back.p_remaining_gt(64, 192), self.table.p_remaining_gt(64, 192)
            )

    def test_trace_loader_sorts_before_time_split(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            path.write_text(
                "timestamp,output_len\n3,30\n1,10\n2,20\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_lengths(path, "output_len", "timestamp"),
                [10, 20, 30],
            )

    def test_rejects_unsorted_rows(self):
        with self.assertRaises(ValueError):
            SurvivalTable(bucket_edges=(0,), remaining=((5, 3, 9),))

    def test_rejects_non_increasing_edges(self):
        with self.assertRaises(ValueError):
            SurvivalTable(bucket_edges=(0, 0), remaining=((1,), (2,)))


class TestFastPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.table = make_table()
        self.policy = FastPolicy(
            config=PolicyConfig(),
            table=self.table,
            tpot_tp1=TpotModel(0.030, 0.0),
            tpot_tp4=TpotModel(0.020, 0.0),
            interference=InterferenceModel(s_per_gib_at_ref=0.35),
        )

    def test_break_even_is_cost_over_per_token_gain(self):
        n_star, cost, breakdown = self.policy.break_even_tokens(
            request_view(), pool(), pool(), 0.5 * GIB
        )
        self.assertAlmostEqual(n_star, cost / (0.030 - 0.020), places=6)
        for key in ("t_stall_s", "t_dup_s", "t_interference_s", "t_margin_s"):
            self.assertIn(key, breakdown)

    def test_no_per_token_gain_means_never_migrate(self):
        slow_target = FastPolicy(
            config=PolicyConfig(),
            table=self.table,
            tpot_tp1=TpotModel(0.020, 0.0),
            tpot_tp4=TpotModel(0.030, 0.0),
            interference=InterferenceModel(0.35),
        )
        n_star, _, _ = slow_target.break_even_tokens(
            request_view(), pool(), pool(), 0.5 * GIB
        )
        self.assertTrue(math.isinf(n_star))
        decision = slow_target.evaluate(
            request_view(), MigrationState.LOCAL, pool(), pool(), 0.1, 0.5 * GIB
        )
        self.assertIs(decision.action, Action.STAY)
        self.assertIn("no per-token gain", decision.reason)

    def test_break_even_rises_with_target_load(self):
        """A busier target must demand a longer remaining request.

        This is the behaviour a fixed token threshold cannot express, and it is
        the claim behind the N* figure in Section 3.5.
        """
        req = request_view()
        idle = self.policy.break_even_tokens(req, pool(), pool(kv=0.10), 0.5 * GIB)[0]
        busy = self.policy.break_even_tokens(req, pool(), pool(kv=0.70), 0.5 * GIB)[0]
        self.assertGreater(busy, idle)

    def test_rate_aware_interference_uses_request_tpot_fit(self):
        model = InterferenceModel(
            calibration_source="unit test",
            model_kind="rate_aware_tpot",
            tpot_rate_coef_s2_per_gib=0.016,
            tpot_rate_load_coef_s2_per_gib=0.004,
            min_load_frac=0.10,
            max_load_frac=0.65,
            min_rate_gib_s=0.40,
            max_rate_gib_s=1.20,
        )
        self.assertAlmostEqual(model.incremental_tpot_s(0.50, 0.70), 0.0126)
        low = model.penalty_s(
            1 * int(GIB),
            0.20,
            copy_rate_bytes_s=0.70 * GIB,
            native_tpot_s=0.10,
        )
        high = model.penalty_s(
            1 * int(GIB),
            0.60,
            copy_rate_bytes_s=0.70 * GIB,
            native_tpot_s=0.10,
        )
        self.assertGreater(high, low)

    def test_rate_aware_interference_fails_closed_outside_support(self):
        model = InterferenceModel(
            calibration_source="unit test",
            model_kind="rate_aware_tpot",
            tpot_rate_coef_s2_per_gib=0.016,
            min_load_frac=0.10,
            max_load_frac=0.65,
            min_rate_gib_s=0.40,
            max_rate_gib_s=1.20,
        )
        self.assertTrue(math.isinf(model.incremental_tpot_s(0.70, 0.70)))

    def test_tpot_model_fails_closed_outside_running_support(self):
        policy = FastPolicy(
            config=PolicyConfig(),
            table=self.table,
            tpot_tp1=TpotModel(
                0.030,
                0.0,
                num_running_min=1,
                num_running_max=8,
            ),
            tpot_tp4=TpotModel(
                0.020,
                0.0,
                num_running_min=1,
                num_running_max=4,
            ),
            interference=InterferenceModel(s_per_gib_at_ref=0.35),
        )
        decision = policy.evaluate(
            request_view(),
            MigrationState.LOCAL,
            pool(running=2),
            pool(running=5),
            0.1,
            0.5 * GIB,
        )
        self.assertIs(decision.action, Action.STAY)
        self.assertIn("outside calibrated num_running support", decision.reason)

    def test_load_piecewise_tpot_interpolates_and_fails_closed(self):
        model = TpotModel(
            0.020,
            model_kind="load_piecewise_monotone",
            load_knots=(0.10, 0.30, 0.50),
            tpot_knots_s=(0.020, 0.040, 0.080),
            min_load_frac=0.10,
            max_load_frac=0.50,
        )
        self.assertAlmostEqual(model.tpot_s(99, 0.20), 0.030)
        self.assertTrue(model.in_support(99, 0.20))
        self.assertFalse(model.in_support(99, 0.60))

    def test_too_early_requests_are_not_eligible(self):
        decision = self.policy.evaluate(
            request_view(output=8, computed=8),
            MigrationState.LOCAL,
            pool(),
            pool(),
            0.1,
            0.5 * GIB,
        )
        self.assertIs(decision.action, Action.STAY)
        self.assertIn("too early", decision.reason)

    def test_full_target_blocks_escalation(self):
        decision = self.policy.evaluate(
            request_view(output=1024, computed=1024),
            MigrationState.LOCAL,
            pool(),
            pool(kv=0.95, waiting=9),
            0.1,
            0.5 * GIB,
        )
        self.assertIs(decision.action, Action.STAY)
        self.assertIn("target unavailable", decision.reason)

    def test_concurrency_cap_blocks_escalation(self):
        decision = self.policy.evaluate(
            request_view(output=1024, computed=1024),
            MigrationState.LOCAL,
            pool(),
            pool(),
            0.1,
            0.5 * GIB,
            active_migrations=1,
        )
        self.assertIs(decision.action, Action.STAY)
        self.assertIn("slots busy", decision.reason)

    def test_oom_risk_forces_escalation_and_is_marked(self):
        starved = pool(free_blocks=0, kv=0.99)
        decision = self.policy.evaluate(
            request_view(output=64, computed=64),
            MigrationState.LOCAL,
            starved,
            pool(),
            0.9,
            0.5 * GIB,
        )
        self.assertIs(decision.action, Action.START_SHADOW)
        self.assertTrue(decision.forced)
        self.assertGreaterEqual(decision.p_oom, self.policy.cfg.p_oom_force)

    def test_forced_escalation_still_respects_a_full_target(self):
        """Safety may override the benefit test, never the physical guard."""
        decision = self.policy.evaluate(
            request_view(output=64, computed=64),
            MigrationState.LOCAL,
            pool(free_blocks=0, kv=0.99),
            pool(kv=0.99, waiting=20),
            0.9,
            0.5 * GIB,
        )
        self.assertIs(decision.action, Action.STAY)
        self.assertIn("target unavailable", decision.reason)

    def test_theta_falls_as_source_risk_rises(self):
        self.assertAlmostEqual(self.policy.theta_esc(0.0), self.policy.cfg.theta_0)
        self.assertLess(self.policy.theta_esc(1.0), self.policy.theta_esc(0.0))
        self.assertGreaterEqual(self.policy.theta_esc(10.0), self.policy.cfg.theta_min)

    def test_group_straggler_lowers_the_break_even_length(self):
        plain = request_view(output=256, computed=256)
        straggler = request_view(
            output=256, computed=256, group_id="g1", is_group_longest=True
        )
        n_plain = self.policy.break_even_tokens(plain, pool(), pool(), 0.5 * GIB)[0]
        n_strag = self.policy.break_even_tokens(straggler, pool(), pool(), 0.5 * GIB)[0]
        self.assertLess(n_strag, n_plain)

    def test_request_beyond_calibrated_range_refuses_to_decide(self):
        decision = self.policy.evaluate(
            request_view(output=5000, computed=5000),
            MigrationState.LOCAL,
            pool(),
            pool(),
            0.1,
            0.5 * GIB,
        )
        self.assertIs(decision.action, Action.STAY)
        self.assertIn("beyond the calibrated range", decision.reason)

    def test_out_of_support_reports_a_conservative_risk_not_zero(self):
        """Zero risk for the longest requests would blind the OOM guard."""
        p = self.policy.p_oom(
            request_view(output=5000, computed=5000), pool(), others_expected=0.0
        )
        self.assertAlmostEqual(p, self.policy.cfg.p_oom_unsupported, places=9)

    def test_exhausted_headroom_is_certain_risk_without_consulting_the_table(self):
        p = self.policy.p_oom(
            request_view(output=5000, computed=5000),
            pool(free_blocks=0),
            others_expected=0.0,
        )
        self.assertEqual(p, 1.0)

    def test_thin_survival_support_refuses_to_decide(self):
        thin = FastPolicy(
            config=PolicyConfig(min_survivors_for_confidence=50),
            table=SurvivalTable.from_output_lengths([100] * 900 + [4096] * 30),
            tpot_tp1=TpotModel(0.030, 0.0),
            tpot_tp4=TpotModel(0.020, 0.0),
            interference=InterferenceModel(0.35),
        )
        decision = thin.evaluate(
            request_view(output=1024, computed=1024),
            MigrationState.LOCAL,
            pool(),
            pool(),
            0.1,
            0.5 * GIB,
        )
        self.assertIs(decision.action, Action.STAY)
        self.assertIn("insufficient survival-table support", decision.reason)

    def test_decision_is_json_serializable(self):
        decision = self.policy.evaluate(
            request_view(), MigrationState.LOCAL, pool(), pool(), 0.1, 0.5 * GIB
        )
        json.dumps(decision.to_json())

    def test_abandon_when_target_risk_spikes(self):
        abandon, reason = self.policy.should_abandon(
            request_view(), pool(), pool(kv=0.99), 0.5 * GIB, 0.1
        )
        self.assertTrue(abandon)
        self.assertIn("target risk", reason)

    def test_ranking_prefers_value_per_byte_moved(self):
        """With one TP4 slot, ranking matters more than thresholding.

        Two requests at the same progress differ only in how much KV they
        would drag across the link; the cheaper one must win the slot.
        """
        light = request_view(output=256, computed=256, rid="light")
        heavy = request_view(output=256, computed=4096, rid="heavy")
        ranked = self.policy.rank_candidates([heavy, light], pool(), pool(), 0.5 * GIB)
        self.assertEqual([r.request_id for r, _ in ranked], ["light", "heavy"])
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_ranking_drops_out_of_support_candidates(self):
        ranked = self.policy.rank_candidates(
            [request_view(output=9999, computed=9999, rid="past-range")],
            pool(),
            pool(),
            0.5 * GIB,
        )
        self.assertEqual(ranked, [])

    def test_ranking_promotes_a_group_straggler_over_an_equal_peer(self):
        plain = request_view(output=256, computed=256, rid="plain")
        straggler = request_view(
            output=256,
            computed=256,
            rid="straggler",
            group_id="g1",
            is_group_longest=True,
        )
        ranked = self.policy.rank_candidates(
            [plain, straggler], pool(), pool(), 0.5 * GIB
        )
        self.assertEqual(ranked[0][0].request_id, "straggler")

    def test_migration_bytes_uses_measured_footprint(self):
        req = request_view(computed=1000)
        self.assertEqual(self.policy.migration_bytes(req), 1000 * 196_608)


class TestRiskTracker(unittest.TestCase):
    def test_ewma_and_preemption_delta(self):
        tracker = RiskTracker(alpha=0.5)
        first = tracker.update(pool(kv=0.2, preempt=100))
        self.assertAlmostEqual(first, 0.7 * 0.2, places=9)
        second = tracker.update(pool(kv=0.2, preempt=140))
        self.assertGreater(second, first)

    def test_counter_reset_is_not_read_as_a_burst(self):
        """A server restart resets the counter; that is not a risk event."""
        tracker = RiskTracker(alpha=0.5)
        tracker.update(pool(preempt=1000))
        before = tracker.value
        after = tracker.update(pool(preempt=5))
        self.assertLessEqual(after, before + 1e-9)


if __name__ == "__main__":
    unittest.main()
