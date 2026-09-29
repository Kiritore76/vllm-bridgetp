# SPDX-License-Identifier: Apache-2.0
"""M3 candidate timing and provisional-ready safety behavior."""

from __future__ import annotations

import unittest
import json
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.bridge_tp.run_phase9_controller import (
    _m3_tpot_evidence, parse_args, step_shadow,
)
from vllm.bridge_tp.controller.events import (
    MigrationState, SourceRequestView, TriggerPath,
)
from vllm.bridge_tp.controller.online_io import ProxyRecorder
from vllm.bridge_tp.controller.response_proxy import ProxyMode
from vllm.bridge_tp.controller.manager_m3 import (
    M3CommitConfig,
    M3CommitController,
)
from vllm.bridge_tp.controller.state_machine import (
    IllegalTransition,
    MigrationStateMachine,
)


class TestM3Commit(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = M3CommitController(M3CommitConfig(
            handoff_s=0.5, gain_margin_s=0.5, defer_tokens=64,
        ))
        self.inputs = dict(
            output_tokens=100,
            base_candidate=164,
            max_output_tokens=1024,
            expected_remaining_tokens=400.0,
            source_tpot_s=0.030,
            target_tpot_s=0.020,
            target_waiting=0,
            source_time_to_guard_s=60.0,
            capacity_emergency=False,
        )

    def test_positive_gain_keeps_earliest_boundary(self) -> None:
        decision = self.controller.plan_candidate(**self.inputs)
        self.assertEqual(decision.action, "COMMIT_EARLIEST")
        self.assertEqual(decision.candidate_output_tokens, 164)
        self.assertGreater(decision.expected_gain_s, 0.5)

    def test_cli_requires_m2_and_explicit_calibration(self) -> None:
        base = [
            "run_phase9_controller.py", "--config", "config.json",
            "--run-dir", "run", "--source-request", "request.json",
            "--manager-m1-auto-start", "--manager-m0-shadow",
            "--manager-m2-rate", "--diagnostic-earliest-ready-cutover",
            "--handoff-mode", "shadow-only", "--gpu-resident-shadow",
            "--manager-m3-commit",
        ]
        with patch("sys.argv", base), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()
        with patch("sys.argv", base + [
            "--m3-handoff-s", "0.5", "--m3-gain-margin-s", "0.5",
        ]):
            self.assertTrue(parse_args().manager_m3_commit)

    def test_busy_target_defers_only_while_source_safe(self) -> None:
        busy = self.inputs | {"target_waiting": 4}
        decision = self.controller.plan_candidate(**busy)
        self.assertEqual((decision.action, decision.candidate_output_tokens),
                         ("DEFER", 228))
        urgent = self.controller.plan_candidate(
            **(busy | {"capacity_emergency": True})
        )
        self.assertEqual(urgent.candidate_output_tokens, 164)
        close_guard = self.controller.plan_candidate(
            **(busy | {"source_time_to_guard_s": 1.0})
        )
        self.assertEqual(close_guard.candidate_output_tokens, 164)

    def test_missing_tpot_does_not_invent_gain(self) -> None:
        decision = self.controller.plan_candidate(
            **(self.inputs | {"source_tpot_s": None})
        )
        self.assertEqual(decision.action, "COMMIT_EARLIEST")
        self.assertIsNone(decision.expected_gain_s)

    def test_no_remaining_output_for_deferral(self) -> None:
        decision = self.controller.plan_candidate(**(
            self.inputs | {
                "output_tokens": 800, "base_candidate": 900,
                "target_waiting": 4,
            }
        ))
        self.assertEqual(decision.candidate_output_tokens, 900)

    def test_provisional_ready_requires_four_rank_commit_gate(self) -> None:
        machine = MigrationStateMachine(allow_shadow_takeover=True)
        record = machine.create("m", "r")
        machine.transition("m", MigrationState.SHADOW, 1.0)
        machine.transition("m", MigrationState.READY_NOT_COMMITTED, 2.0)
        with self.assertRaises(IllegalTransition):
            machine.transition("m", MigrationState.TAKEOVER, 3.0)
        for rank in range(4):
            machine.mark_rank_ready("m", rank)
        machine.transition("m", MigrationState.TAKEOVER, 3.0)
        self.assertEqual(record.state, MigrationState.TAKEOVER)

    def test_uncalibrated_tpot_is_not_used_as_gain_evidence(self) -> None:
        pool = SimpleNamespace(
            p99_tpot_s=None, tpot_samples=0, num_running=4,
            kv_usage_frac=0.2,
        )
        model = SimpleNamespace(
            calibration_source="CAP-0 placeholder",
            in_support=lambda *_args: True,
            tpot_s=lambda *_args: 0.02,
        )
        self.assertEqual(_m3_tpot_evidence(pool, model),
                         (None, "unavailable"))
        model.calibration_source = "A100 measured target TPOT"
        self.assertEqual(_m3_tpot_evidence(pool, model),
                         (0.02, "calibrated_model"))

    def test_candidate_is_deferred_before_target_admission(self) -> None:
        class Adapter:
            def __init__(self, root: Path) -> None:
                self.run_dir = root

            def set_rate(self, *_args, **_kwargs) -> None:
                pass

            def poll_initial_history_gpu_buffered(self):
                return True, {0, 1, 2, 3}, "buffered"

        class Audit:
            def __init__(self) -> None:
                self.rows: list[dict] = []

            def write(self, row: dict) -> None:
                self.rows.append(row)

        class M2:
            def decide(self, *_args):
                return SimpleNamespace(
                    profile="LOW", rate_bytes_s=0.5 * 1024**3,
                    reason="target is busy and source is safe",
                    source_time_to_guard_s=60.0,
                    to_json=lambda: {"profile": "LOW"},
                )

        table = SimpleNamespace(
            in_support=lambda _tokens: True,
            expected_remaining=lambda _tokens: 400.0,
        )
        policy = SimpleNamespace(
            table=table,
            cfg=SimpleNamespace(max_target_kv_usage_frac=0.85),
            migration_bytes=lambda _request: 1024,
            tpot_tp1=SimpleNamespace(
                calibration_source="A100 source calibration",
                in_support=lambda *_args: True,
                tpot_s=lambda *_args: 0.030,
            ),
            tpot_tp4=SimpleNamespace(
                calibration_source="A100 target calibration",
                in_support=lambda *_args: True,
                tpot_s=lambda *_args: 0.020,
            ),
        )
        rate = SimpleNamespace(rate_bytes_s=0.5 * 1024**3,
                               rate_gib_s=0.5, last_reason="LOW")
        source_pool = SimpleNamespace(
            p99_tpot_s=0.030, tpot_samples=1, num_running=1,
            kv_usage_frac=0.2,
        )
        target_pool = SimpleNamespace(
            p99_tpot_s=0.020, tpot_samples=1, num_running=4,
            num_waiting=4, kv_usage_frac=0.2,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "session_manifest.json").write_text(
                json.dumps({"num_computed_tokens": 2100}), encoding="utf-8"
            )
            audit = Audit()
            machine = MigrationStateMachine(audit_sink=audit.write)
            record = machine.create("m", "r")
            record.trigger_output_tokens = 64
            record.trigger_path = TriggerPath.MANAGER_M1_START
            machine.transition("m", MigrationState.SHADOW, 1.0)
            step_shadow(
                policy, machine, Adapter(root), audit, record,
                SourceRequestView(
                    request_id="r", prompt_tokens=2048, output_tokens=100,
                    computed_tokens=2148, pending_tokens=1,
                    arrival_unix_s=0.0, last_token_unix_s=1.0,
                ),
                source_pool, target_pool, 0.0, rate, 2.0, False,
                ProxyRecorder("external", ProxyMode.HOLD_BACK),
                diagnostic_earliest_ready_cutover=True, max_tokens=1024,
                manager_m2=M2(),
                m2_snapshot=SimpleNamespace(to_json=lambda: {}),
                manager_m3=self.controller,
            )
            candidate = json.loads(
                (root / "earliest_ready_candidate.json").read_text()
            )
            # Base: output 100 + max(64, backlog 48 + 64) = 212.
            self.assertEqual(candidate["cutover_output_tokens"], 276)
            self.assertTrue(any(
                row.get("kind") == "manager_m3_candidate_decision"
                and row["decision"]["action"] == "DEFER"
                for row in audit.rows
            ))
