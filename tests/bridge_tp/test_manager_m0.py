# SPDX-License-Identifier: Apache-2.0
"""M0 manager decisions stay deterministic and fail closed on missing data."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.replay_manager_m0 import replay
from tools.bridge_tp.run_shadow_strategy_online_validation import parse_args
from vllm.bridge_tp.controller.manager_m0 import (
    ChannelRegistry,
    M0ExecutorAdapter,
    M0Proposal,
    MigrationManagerM0,
    RuntimeSnapshot,
    snapshot_from_telemetry,
)


def snapshot(state: str = "LOCAL") -> RuntimeSnapshot:
    return RuntimeSnapshot(
        unix_s=100.0,
        migration_id="migration-1",
        request_id="request-1",
        state=state,
        generated_tokens=64,
        source_sampled_unix_s=99.9,
        target_sampled_unix_s=99.9,
        source_free_kv_tokens=40_000,
        source_guard_free_kv_tokens=13_776,
        target_free_kv_tokens=50_000,
        target_p99_tpot_s=0.02,
        channel_available=True,
        current_rate_bytes_s=100.0,
    )


class TestManagerM0(unittest.TestCase):
    def test_outer_runner_accepts_shadow_flag(self) -> None:
        argv = [
            "runner",
            "--model-path", "model",
            "--manifest", "manifest.json",
            "--survival-table", "survival.json",
            "--guard-file", "guard.txt",
            "--out-root", "out",
            "--expected-revision", "revision",
            "--expected-manifest-sha256", "manifest-sha",
            "--expected-survival-sha256", "survival-sha",
            "--expected-guard-sha256", "guard-sha",
            "--expected-guard", "13776",
            "--tp1-blocks", "5534",
            "--tp4-blocks", "72378",
            "--manager-m0-shadow",
        ]
        with patch("sys.argv", argv):
            self.assertTrue(parse_args().manager_m0_shadow)

    def test_registry_and_executor_are_advisory(self) -> None:
        registry = ChannelRegistry()
        self.assertTrue(registry.available("tp4"))
        registry.observe_active("tp4", "migration-1")
        self.assertFalse(registry.available("tp4", "migration-2"))
        with self.assertRaises(ValueError):
            registry.observe_active("tp4", "migration-2")
        registry.observe_idle("tp4", "migration-2")
        self.assertFalse(registry.available("tp4", "migration-2"))
        registry.observe_idle("tp4", "migration-1")
        self.assertTrue(registry.available("tp4"))
        records = []
        M0ExecutorAdapter(records.append).publish(
            1,
            snapshot(),
            M0Proposal(start=True),
            MigrationManagerM0().decide(snapshot(), M0Proposal(start=True)),
        )
        self.assertEqual(records[0]["decision"]["action"], "WOULD_START")
        self.assertEqual(records[0]["kind"], "manager_m0_shadow")

    def test_start_requires_fresh_evidence_and_available_channel(self) -> None:
        manager = MigrationManagerM0()
        proposal = M0Proposal(start=True)
        self.assertEqual(manager.decide(snapshot(), proposal).action, "WOULD_START")
        stale = replace(snapshot(), target_sampled_unix_s=95.0)
        result = manager.decide(stale, proposal)
        self.assertEqual(result.action, "WOULD_WAIT")
        self.assertIn("target sample stale or in the future", result.missing)
        occupied = replace(snapshot(), channel_available=False)
        self.assertEqual(manager.decide(occupied, proposal).action, "WOULD_WAIT")

    def test_rate_and_commit_do_not_invent_missing_progress(self) -> None:
        manager = MigrationManagerM0()
        shadow = snapshot("SHADOW")
        result = manager.decide(shadow, M0Proposal(rate_bytes_s=200.0))
        self.assertEqual(result.action, "WOULD_SET_RATE")
        no_tpot = replace(shadow, target_p99_tpot_s=None)
        self.assertEqual(
            manager.decide(no_tpot, M0Proposal(rate_bytes_s=200.0)).action,
            "WOULD_WAIT",
        )
        result = manager.decide(shadow, M0Proposal(commit=True))
        self.assertEqual(result.action, "WOULD_WAIT")
        self.assertIn("all_ranks_history_resident", result.missing)
        safe = replace(
            shadow,
            all_ranks_history_resident=True,
            all_ranks_armed=True,
            delta_lag_tokens=16,
        )
        self.assertEqual(
            manager.decide(safe, M0Proposal(commit=True)).action, "WOULD_COMMIT"
        )
        lagged = replace(safe, delta_lag_tokens=17)
        self.assertEqual(
            manager.decide(lagged, M0Proposal(commit=True)).action,
            "WOULD_WAIT",
        )
        pressured = replace(shadow, capacity_pressure=True)
        self.assertEqual(
            manager.decide(pressured, M0Proposal(cancel=True)).action,
            "WOULD_WAIT",
        )

    def test_zero_interval_samples_hide_stale_tpot(self) -> None:
        telemetry = {
            "unix_s": 100.0,
            "state": "SHADOW",
            "tp1": {"sampled_unix_s": 99.9},
            "tp4": {
                "sampled_unix_s": 99.9,
                "p99_tpot_s": 0.02,
                "tpot_samples": 0,
            },
        }
        result = snapshot_from_telemetry(telemetry)
        self.assertEqual(result.target_tpot_samples, 0)
        self.assertIsNone(result.target_p99_tpot_s)
        telemetry["tp4"]["tpot_samples"] = 2
        self.assertEqual(
            snapshot_from_telemetry(telemetry).target_p99_tpot_s, 0.02
        )

    def test_replay_keeps_historical_evidence_gaps(self) -> None:
        rows = [
        {
            "kind": "run_metadata",
            "migration_id": "migration-1",
            "source_request_id": "request-1",
        },
        {
            "kind": "telemetry",
            "tick": 1,
            "unix_s": 100.0,
            "state": "LOCAL",
            "output_tokens": 64,
            "rate_bytes_s": 100.0,
            "tp1": {
                "sampled_unix_s": 99.9,
                "free_kv_blocks": 100,
                "block_size": 16,
            },
            "tp4": {
                "sampled_unix_s": 99.9,
                "free_kv_blocks": 100,
                "block_size": 16,
            },
        },
        {"kind": "decision", "action": "START_SHADOW"},
        {
            "kind": "telemetry",
            "tick": 2,
            "unix_s": 101.0,
            "state": "SHADOW",
            "output_tokens": 80,
            "rate_bytes_s": 100.0,
            "tp1": {"sampled_unix_s": 100.9},
            "tp4": {"sampled_unix_s": 100.9, "p99_tpot_s": 0.02},
        },
        {"kind": "rate", "rate_bytes_s": 200.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "phase9_audit.jsonl"
            audit.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            output = replay(audit)
        self.assertEqual(
            [row["decision"]["action"] for row in output],
            ["WOULD_START", "WOULD_SET_RATE"],
        )
        self.assertIsNone(output[1]["snapshot"]["history_resident_bytes"])
        self.assertIsNone(output[1]["snapshot"]["all_ranks_armed"])


if __name__ == "__main__":
    unittest.main()
