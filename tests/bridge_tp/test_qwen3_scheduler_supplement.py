# SPDX-License-Identifier: Apache-2.0
"""Bounded additional measurement plan and failure gates, without GPU execution."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bridge_tp import run_qwen3_scheduler_supplement as supplement
from tools.bridge_tp.run_pressure_calibration import capacity_check


class SupplementTests(unittest.TestCase):
    def test_plan_is_eight_new_attempts_without_drain(self):
        stages = supplement.stages()
        self.assertEqual(len(stages), 8)
        self.assertEqual(len({s["name"] for s in stages}), 8)
        self.assertEqual([s["history"] for s in stages[:4]], [2048, 128, 2048, 128])
        self.assertTrue(all(s["kind"] == "service" for s in stages[4:]))
        self.assertEqual(
            [s["scene"]["target_count"] for s in stages[4:]], [8, 16, 16, 8]
        )

    def test_shapes_fit_observed_native_capacity(self):
        inventory = [dict(free_blocks=n, block_size=16) for n in (2341, 11712)]
        for stage in supplement.stages()[4:]:
            capacity_check(stage["scene"], inventory)

    def test_failure_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp))
            ledger = []
            with (
                patch.object(supplement, "gpu_idle", return_value=True),
                patch.object(
                    supplement, "execute_stage", side_effect=ValueError("bad")
                ) as execute,
                self.assertRaisesRegex(ValueError, "bad"),
            ):
                supplement.run_stages(args, {}, ledger)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(ledger[0]["status"], "FAILED")
            self.assertEqual(len(ledger), 1)

    def test_cleanup_blocks_next_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp))
            ledger = []
            with (
                patch.object(supplement, "gpu_idle", side_effect=[True, False]),
                patch.object(supplement, "execute_stage") as execute,
                self.assertRaisesRegex(RuntimeError, "cleanup"),
            ):
                supplement.run_stages(args, {}, ledger)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(ledger[0]["status"], "FAILED")

    def test_interrupt_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp))
            ledger = []
            with (
                patch.object(supplement, "gpu_idle", return_value=True),
                patch.object(
                    supplement, "execute_stage", side_effect=KeyboardInterrupt
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                supplement.run_stages(args, {}, ledger)
            self.assertEqual(ledger[0]["status"], "INTERRUPTED")

    def test_success_runs_exactly_eight_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp))
            ledger = []
            with (
                patch.object(supplement, "gpu_idle", return_value=True),
                patch.object(supplement, "execute_stage") as execute,
            ):
                supplement.run_stages(args, {}, ledger)
            self.assertEqual(execute.call_count, 8)
            self.assertTrue(all(r["status"] == "COMPLETED" for r in ledger))

    def test_migration_acceptance_rechecks_commit_release_and_idle(self):
        records = {
            "result.json": dict(
                status="COMPLETED",
                runner_code=0,
                all_owned_groups_ended=True,
                source_release_confirmed=True,
                source_release_to_channel_idle_checked=True,
                target_tp=2,
                mechanism_only=True,
                source_finish="abort",
                target_finish="length",
                full_speed=dict(
                    status="COMPLETED",
                    transfer_mode="FULL_SPEED",
                    actual_starts=1,
                    history_rank_receipts=2,
                    history_sender_receipts=1,
                ),
                start_to_release_s=8.0,
                visible_handoff_gap_s=3.5,
                start_to_source_release_and_idle_observed_s=8.1,
            ),
            "unified_tokens.json": [
                dict(index=i, origin="source" if i < 311 else "target")
                for i in range(6144)
            ],
            "takeover_state.json": dict(
                state="COMMITTED",
                ready_sender_ranks=[0, 1],
                ready_receiver_ranks=[0, 1],
                source_request_id="source-id",
            ),
            "source_kv_release_receipt.json": dict(
                status="SOURCE_KV_RELEASED", request_id="source-id"
            ),
        }
        with (
            patch.object(supplement, "read", side_effect=lambda p: records[p.name]),
            patch.object(supplement, "channel_idle", return_value=True),
        ):
            self.assertEqual(
                supplement.verify_migration(Path("run"))["status"], "COMPLETED"
            )
        with (
            patch.object(supplement, "read", side_effect=lambda p: records[p.name]),
            patch.object(supplement, "channel_idle", return_value=False),
            self.assertRaisesRegex(ValueError, "IDLE"),
        ):
            supplement.verify_migration(Path("run"))
