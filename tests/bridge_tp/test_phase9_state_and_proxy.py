# SPDX-License-Identifier: Apache-2.0
"""Phase 9: rate controller, migration state machine, unified response proxy.

python -m unittest tests.bridge_tp.test_phase9_state_and_proxy
"""

from __future__ import annotations

import unittest

from vllm.bridge_tp.controller.events import MigrationState
from vllm.bridge_tp.controller.rate_controller import RateConfig, RateController
from vllm.bridge_tp.controller.response_proxy import (
    ProxyMode,
    ResponseProxy,
    StreamViolation,
)
from vllm.bridge_tp.controller.state_machine import (
    IllegalTransition,
    MigrationStateMachine,
)

GIB = 1024.0**3


class TestRateController(unittest.TestCase):
    def cfg(self, **kw) -> RateConfig:
        return RateConfig(min_samples_before_action=1, **kw)

    def test_increases_with_slo_slack(self):
        rc = RateController(self.cfg())
        start = rc.rate_bytes_s
        rc.step(native_p99_tpot_s=0.010, remaining_bytes=10**9)
        self.assertGreater(rc.rate_bytes_s, start)
        self.assertIn("increase", rc.last_reason)

    def test_backs_off_when_slo_is_violated(self):
        rc = RateController(self.cfg())
        rc.step(0.010, 10**9)
        high = rc.rate_bytes_s
        rc.step(0.500, 10**9)
        self.assertLess(rc.rate_bytes_s, high)
        self.assertIn("backoff", rc.last_reason)

    def test_stays_inside_clamps_under_sustained_pressure(self):
        cfg = self.cfg()
        rc = RateController(cfg)
        for _ in range(50):
            rc.step(0.001, 10**9)
        self.assertLessEqual(rc.rate_bytes_s, cfg.b_max_bytes_s)
        for _ in range(50):
            rc.step(1.000, 10**9)
        self.assertGreaterEqual(rc.rate_bytes_s, cfg.b_min_bytes_s)

    def test_deadline_override_beats_backoff(self):
        """Near the OOM horizon, interference is the lesser cost."""
        cfg = self.cfg()
        rc = RateController(cfg)
        rc.step(1.000, remaining_bytes=int(1.5 * GIB), seconds_to_deadline=0.5)
        self.assertGreater(rc.rate_bytes_s, cfg.b_max_bytes_s)
        self.assertLessEqual(rc.rate_bytes_s, cfg.b_hard_max_bytes_s)
        self.assertIn("deadline override", rc.last_reason)

    def test_deadband_holds_rate_steady(self):
        cfg = self.cfg()
        rc = RateController(cfg)
        steady = cfg.slo_p99_tpot_s * 0.95
        rc.step(steady, 10**9)
        held = rc.rate_bytes_s
        rc.step(steady, 10**9)
        self.assertEqual(rc.rate_bytes_s, held)
        self.assertIn("deadband", rc.last_reason)

    def test_warmup_does_not_act_on_a_single_sample(self):
        rc = RateController(RateConfig(min_samples_before_action=2))
        start = rc.rate_bytes_s
        rc.step(0.001, 10**9)
        self.assertEqual(rc.rate_bytes_s, start)
        self.assertEqual(rc.last_reason, "warmup")

    def test_gib_conversion_is_exact(self):
        rc = RateController(RateConfig())
        rc.rate_bytes_s = 0.5 * GIB
        self.assertAlmostEqual(rc.rate_gib_s, 0.5, places=12)


class TestMigrationStateMachine(unittest.TestCase):
    def happy(self) -> MigrationStateMachine:
        sm = MigrationStateMachine()
        sm.create("m1", "r1")
        return sm

    def test_happy_path_records_boundary_timestamps(self):
        sm = self.happy()
        sm.transition("m1", MigrationState.SHADOW, 1.0)
        sm.transition("m1", MigrationState.HANDOFF, 2.0)
        for rank in range(4):
            sm.mark_rank_ready("m1", rank)
        record = sm.transition("m1", MigrationState.TAKEOVER, 3.0)
        self.assertIs(record.state, MigrationState.TAKEOVER)
        self.assertEqual(
            (record.t_shadow_start, record.t_cutover, record.t_committed),
            (1.0, 2.0, 3.0),
        )

    def test_commit_requires_all_four_ranks(self):
        """Phase 9 must never be able to bypass the Phase 7 readback gate."""
        sm = self.happy()
        sm.transition("m1", MigrationState.SHADOW, 1.0)
        sm.transition("m1", MigrationState.HANDOFF, 2.0)
        for rank in (0, 1, 2):
            sm.mark_rank_ready("m1", rank)
        with self.assertRaisesRegex(IllegalTransition, "3/4 ranks ready"):
            sm.transition("m1", MigrationState.TAKEOVER, 3.0)

    def test_cannot_skip_handoff(self):
        sm = self.happy()
        sm.transition("m1", MigrationState.SHADOW, 1.0)
        for rank in range(4):
            sm.mark_rank_ready("m1", rank)
        with self.assertRaisesRegex(IllegalTransition, "not legal"):
            sm.transition("m1", MigrationState.TAKEOVER, 2.0)

    def test_transitions_are_idempotent(self):
        sm = self.happy()
        sm.transition("m1", MigrationState.SHADOW, 1.0)
        again = sm.transition("m1", MigrationState.SHADOW, 1.5)
        self.assertIs(again.state, MigrationState.SHADOW)
        self.assertEqual(len(again.history), 1)

    def test_terminal_states_are_absorbing(self):
        sm = self.happy()
        sm.transition("m1", MigrationState.SHADOW, 1.0)
        sm.transition("m1", MigrationState.CANCELLED, 2.0)
        with self.assertRaises(IllegalTransition):
            sm.transition("m1", MigrationState.HANDOFF, 3.0)
        self.assertEqual(sm.count_active(), 0)

    def test_rollback_from_handoff_is_legal(self):
        sm = self.happy()
        sm.transition("m1", MigrationState.SHADOW, 1.0)
        sm.transition("m1", MigrationState.HANDOFF, 2.0)
        record = sm.transition("m1", MigrationState.ROLLED_BACK, 3.0, "target risk")
        self.assertIs(record.state, MigrationState.ROLLED_BACK)

    def test_audit_sink_receives_every_transition(self):
        seen: list[dict] = []
        sm = MigrationStateMachine(audit_sink=seen.append)
        sm.create("m1", "r1")
        sm.transition("m1", MigrationState.SHADOW, 1.0, "risk")
        sm.transition("m1", MigrationState.CANCELLED, 2.0, "eos")
        self.assertEqual([r["to"] for r in seen], ["SHADOW", "CANCELLED"])
        self.assertEqual(seen[0]["reason"], "risk")

    def test_migration_id_cannot_be_rebound(self):
        sm = self.happy()
        with self.assertRaisesRegex(IllegalTransition, "already bound"):
            sm.create("m1", "r2")

    def test_unknown_migration_raises(self):
        with self.assertRaises(KeyError):
            MigrationStateMachine().transition("nope", MigrationState.SHADOW, 1.0)


def drive_source(proxy: ResponseProxy, start: int, count: int, t0: float = 0.0) -> None:
    for i in range(start, start + count):
        proxy.on_source_token(i, 1000 + i, t0 + i * 0.01)


class TestResponseProxy(unittest.TestCase):
    def test_hold_back_produces_contiguous_stream_with_no_duplicates(self):
        """Reproduces the Phase 8 dual-write run: cutover 160, source ran to 191."""
        proxy = ResponseProxy("ext-1", ProxyMode.HOLD_BACK)
        drive_source(proxy, 0, 160)
        proxy.set_cutover(160, 1.6)
        drive_source(proxy, 160, 31)
        self.assertEqual(len(proxy.emitted), 160)
        proxy.on_commit(2.0)
        for j in range(128):
            proxy.on_target_token(160 + j, 1000 + 160 + j, 2.1 + j * 0.01)
        proxy.assert_contiguous()
        stats = proxy.stats()
        self.assertEqual(stats["emitted_tokens"], 288)
        self.assertEqual(stats["source_origin_tokens"], 160)
        self.assertEqual(stats["target_origin_tokens"], 128)
        self.assertEqual(stats["discarded_source_tokens"], 31)
        self.assertEqual(proxy.token_ids(), [1000 + i for i in range(288)])

    def test_hold_back_stall_is_the_full_handoff_window(self):
        proxy = ResponseProxy("ext-1", ProxyMode.HOLD_BACK)
        drive_source(proxy, 0, 10, t0=0.0)
        proxy.set_cutover(10, 0.10)
        proxy.on_commit(0.90)
        proxy.on_target_token(10, 2000, 1.00)
        self.assertAlmostEqual(proxy.handoff_stall_s, 1.00 - 0.09, places=9)

    def test_rollback_flushes_held_tokens_in_order(self):
        proxy = ResponseProxy("ext-1", ProxyMode.HOLD_BACK)
        drive_source(proxy, 0, 10)
        proxy.set_cutover(10, 0.10)
        drive_source(proxy, 10, 5)
        self.assertEqual(len(proxy.emitted), 10)
        flushed = proxy.on_rollback(0.5, "benefit gone")
        self.assertEqual([t.index for t in flushed], [10, 11, 12, 13, 14])
        proxy.assert_contiguous()
        self.assertEqual(proxy.stats()["emitted_tokens"], 15)
        self.assertEqual(proxy.stats()["discarded_source_tokens"], 0)

    def test_greedy_fastpath_verifies_overlap_and_removes_the_stall(self):
        """Under greedy decoding the target reproduces the overlap exactly.

        The client keeps receiving source tokens through Handoff, so the
        visible stall collapses; the proxy still checks every overlap token.
        """
        proxy = ResponseProxy("ext-1", ProxyMode.GREEDY_FASTPATH)
        drive_source(proxy, 0, 160)
        proxy.set_cutover(160, 1.6)
        drive_source(proxy, 160, 31)
        self.assertEqual(len(proxy.emitted), 191)
        proxy.on_commit(2.0)
        for j in range(128):
            proxy.on_target_token(160 + j, 1000 + 160 + j, 2.1 + j * 0.01)
        proxy.assert_contiguous()
        stats = proxy.stats()
        self.assertEqual(stats["emitted_tokens"], 288)
        self.assertEqual(stats["verified_overlap_tokens"], 31)
        self.assertEqual(stats["source_origin_tokens"], 191)
        self.assertEqual(stats["target_origin_tokens"], 97)

    def test_greedy_fastpath_detects_divergence(self):
        proxy = ResponseProxy("ext-1", ProxyMode.GREEDY_FASTPATH)
        drive_source(proxy, 0, 12)
        proxy.set_cutover(10, 0.10)
        proxy.on_commit(0.2)
        with self.assertRaisesRegex(StreamViolation, "diverged at index 10"):
            proxy.on_target_token(10, 999_999, 0.3)

    def test_duplicate_index_is_rejected_in_hold_back_mode(self):
        proxy = ResponseProxy("ext-1", ProxyMode.HOLD_BACK)
        drive_source(proxy, 0, 12)
        proxy.set_cutover(12, 0.2)
        proxy.on_commit(0.3)
        with self.assertRaisesRegex(StreamViolation, "duplicate index"):
            proxy.on_target_token(5, 123, 0.4)

    def test_gap_in_stream_is_rejected(self):
        proxy = ResponseProxy("ext-1")
        proxy.on_source_token(0, 1, 0.0)
        with self.assertRaisesRegex(StreamViolation, "expected index 1"):
            proxy.on_source_token(5, 2, 0.1)

    def test_target_cannot_emit_before_commit(self):
        proxy = ResponseProxy("ext-1")
        drive_source(proxy, 0, 5)
        proxy.set_cutover(5, 0.1)
        with self.assertRaisesRegex(StreamViolation, "before commit"):
            proxy.on_target_token(5, 42, 0.2)

    def test_commit_without_cutover_is_rejected(self):
        with self.assertRaisesRegex(StreamViolation, "before a cutover boundary"):
            ResponseProxy("ext-1").on_commit(1.0)

    def test_cutover_cannot_move_backwards_past_emitted_tokens(self):
        proxy = ResponseProxy("ext-1", ProxyMode.HOLD_BACK)
        drive_source(proxy, 0, 20)
        with self.assertRaisesRegex(StreamViolation, "behind"):
            proxy.set_cutover(5, 0.3)

    def test_cutover_cannot_be_moved_once_set(self):
        proxy = ResponseProxy("ext-1", ProxyMode.HOLD_BACK)
        drive_source(proxy, 0, 5)
        proxy.set_cutover(10, 0.1)
        with self.assertRaisesRegex(StreamViolation, "cutover already set"):
            proxy.set_cutover(12, 0.2)

    def test_source_tokens_after_commit_are_discarded_not_emitted(self):
        proxy = ResponseProxy("ext-1", ProxyMode.HOLD_BACK)
        drive_source(proxy, 0, 10)
        proxy.set_cutover(10, 0.1)
        proxy.on_commit(0.2)
        self.assertEqual(proxy.on_source_token(10, 77, 0.3), [])
        self.assertEqual(proxy.stats()["discarded_source_tokens"], 1)

    def test_rollback_after_commit_is_rejected(self):
        proxy = ResponseProxy("ext-1")
        drive_source(proxy, 0, 5)
        proxy.set_cutover(5, 0.1)
        proxy.on_commit(0.2)
        with self.assertRaisesRegex(StreamViolation, "cannot roll back after commit"):
            proxy.on_rollback(0.3)


if __name__ == "__main__":
    unittest.main()
