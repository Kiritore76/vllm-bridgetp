# SPDX-License-Identifier: Apache-2.0
"""M3 earliest-safe commit and provisional-ready safety behavior."""

from __future__ import annotations

import json
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from tools.bridge_tp.run_phase9_controller import parse_args, step_shadow
from vllm.bridge_tp.controller.events import (
    MigrationState,
    SourceRequestView,
    TriggerPath,
)
from vllm.bridge_tp.controller.manager_m3 import M3CommitController
from vllm.bridge_tp.controller.online_io import ProxyRecorder
from vllm.bridge_tp.controller.response_proxy import ProxyMode
from vllm.bridge_tp.controller.state_machine import (
    IllegalTransition,
    MigrationStateMachine,
)


class TestM3Commit(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = M3CommitController()

    def test_ready_candidate_is_never_deferred(self) -> None:
        decision = self.controller.plan_candidate(
            output_tokens=100, base_candidate=164, max_output_tokens=1024,
        )
        self.assertEqual(decision.action, "COMMIT_EARLIEST")
        self.assertEqual(decision.candidate_output_tokens, 164)

    def test_candidate_must_leave_output_for_target(self) -> None:
        for candidate in (100, 1024):
            with self.assertRaises(ValueError):
                self.controller.plan_candidate(
                    output_tokens=100, base_candidate=candidate,
                    max_output_tokens=1024,
                )

    def test_cli_requires_m2_and_earliest_ready(self) -> None:
        base = [
            "run_phase9_controller.py", "--config", "config.json",
            "--run-dir", "run", "--source-request", "request.json",
            "--manager-m1-auto-start", "--manager-m0-shadow",
            "--m1-source-release-tail-s", "5.0",
            "--manager-m2-rate", "--diagnostic-earliest-ready-cutover",
            "--handoff-mode", "shadow-only", "--gpu-resident-shadow",
            "--manager-m3-commit",
        ]
        with patch("sys.argv", base):
            self.assertTrue(parse_args().manager_m3_commit)
        with patch("sys.argv", base[:9] + base[10:]), redirect_stderr(
            StringIO()
        ):
            with self.assertRaises(SystemExit):
                parse_args()

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

    def test_candidate_equals_base_before_target_admission(self) -> None:
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

        policy = SimpleNamespace(
            cfg=SimpleNamespace(max_target_kv_usage_frac=0.85),
            migration_bytes=lambda _request: 1024,
        )
        rate = SimpleNamespace(
            rate_bytes_s=0.5 * 1024**3, rate_gib_s=0.5,
            last_reason="LOW",
        )
        source_pool = SimpleNamespace(num_running=1, kv_usage_frac=0.2)
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
            # Base: output 100 + max(256, backlog 48 + 64) = 356.
            self.assertEqual(candidate["cutover_output_tokens"], 356)
            self.assertTrue(any(
                row.get("kind") == "manager_m3_candidate_decision"
                and row["decision"]["action"] == "COMMIT_EARLIEST"
                and row["decision"]["candidate_output_tokens"] == 356
                and row["base_candidate_output_tokens"] == 356
                for row in audit.rows
            ))
