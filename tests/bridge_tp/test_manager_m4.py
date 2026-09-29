# SPDX-License-Identifier: Apache-2.0
"""M4 pre-freeze cancellation decisions and cleanup gates."""

from __future__ import annotations

import unittest
import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.bridge_tp.run_phase9_controller import parse_args, step_m4_cancel
from tools.bridge_tp.run_shadow_strategy_online_validation import accept_m4_cancel
from vllm.bridge_tp.controller.action_adapter import ActionError
from vllm.bridge_tp.controller.events import MigrationState, SourceRequestView
from vllm.bridge_tp.controller.manager_m4 import M4CancelController
from vllm.bridge_tp.controller.predictor import SurvivalTable
from vllm.bridge_tp.controller.state_machine import MigrationStateMachine


def request(output_tokens: int = 100) -> SourceRequestView:
    return SourceRequestView(
        request_id="r", prompt_tokens=2048, output_tokens=output_tokens,
        computed_tokens=2048 + output_tokens, pending_tokens=0,
        arrival_unix_s=1.0, last_token_unix_s=10.0,
    )


class TestM4Cancel(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = M4CancelController()
        self.short_table = SurvivalTable.from_output_lengths(
            [105] * 200, bucket_edges=(0, 32, 64, 96)
        )

    def decide(self, **overrides):
        args = {
            "max_output_tokens": 1024, "ignore_eos": False,
            "source_free_kv_tokens": 12000,
            "source_guard_free_kv_tokens": 8448,
            "source_capacity_pressure": False, "freeze_started": False,
        }
        args.update(overrides)
        return self.manager.decide(request(), self.short_table, **args)

    def test_likely_eos_cancels_before_freeze(self) -> None:
        decision = self.decide()
        self.assertEqual(decision.action, "CANCEL_SHADOW")
        self.assertEqual(decision.probability_beyond_window, 0.0)

    def test_cli_requires_m3(self) -> None:
        args = [
            "run_phase9_controller.py", "--config", "config.json",
            "--run-dir", "run", "--source-request", "request.json",
            "--manager-m4-cancel",
        ]
        with patch("sys.argv", args), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_ignored_eos_requires_output_cap_to_be_near(self) -> None:
        self.assertEqual(self.decide(ignore_eos=True).action, "KEEP_SHADOW")
        self.assertEqual(
            self.decide(ignore_eos=True, max_output_tokens=150).action,
            "CANCEL_SHADOW",
        )

    def test_capacity_pressure_or_freeze_blocks_cancel(self) -> None:
        for override in (
            {"source_capacity_pressure": True},
            {"source_free_kv_tokens": 8500},
            {"freeze_started": True},
        ):
            with self.subTest(override=override):
                self.assertEqual(self.decide(**override).action, "KEEP_SHADOW")

    def test_unsupported_or_long_request_stays_in_shadow(self) -> None:
        long_table = SurvivalTable.from_output_lengths([500] * 200)
        self.assertEqual(
            self.manager.decide(
                request(), long_table, max_output_tokens=1024,
                ignore_eos=False, source_free_kv_tokens=12000,
                source_guard_free_kv_tokens=8448,
                source_capacity_pressure=False, freeze_started=False,
            ).action,
            "KEEP_SHADOW",
        )

    def test_cancel_keeps_source_and_cleans_target(self) -> None:
        class Adapter:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir
                self.calls: list[tuple] = []

            def disarm(self, reason: str) -> None:
                self.calls.append(("disarm", reason))

            def refresh_binding(self):
                return object()

            def cancel(self, reason: str, *, abort_source: bool):
                self.calls.append(("cancel", abort_source))
                return {"state": "CANCELLED", "source_abort_dispatched": False}

            def cancel_shadow_target(self, reason: str):
                self.calls.append(("target", reason))
                return {"status": "CANCELLED"}

        class Audit:
            def __init__(self) -> None:
                self.rows: list[dict] = []

            def write(self, row: dict) -> None:
                self.rows.append(row)

        class Recorder:
            def on_rollback(self, now: float, reason: str) -> None:
                pass

        with TemporaryDirectory() as temporary:
            adapter = Adapter(Path(temporary))
            audit = Audit()
            machine = MigrationStateMachine(audit_sink=audit.write)
            record = machine.create("m", "r")
            machine.transition("m", MigrationState.SHADOW, 1.0)
            cancelled = step_m4_cancel(
                self.manager, self.short_table, machine, adapter, audit,
                record, request(), max_output_tokens=1024,
                ignore_eos=False, source_free_kv_tokens=12000,
                source_guard_free_kv_tokens=8448,
                source_capacity_pressure=False, now=10.0, dry_run=False,
                recorder=Recorder(),
            )
            self.assertTrue(cancelled)
            self.assertEqual(record.state, MigrationState.CANCELLED)
            self.assertEqual([call[0] for call in adapter.calls], [
                "disarm", "cancel", "target",
            ])
            self.assertEqual(adapter.calls[1], ("cancel", False))

    def test_freeze_receipt_prevents_cleanup(self) -> None:
        class Adapter:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir

            def disarm(self, _reason: str) -> None:
                raise AssertionError("frozen request must not be disarmed")

        class Audit:
            def write(self, _row: dict) -> None:
                pass

        class Recorder:
            def on_rollback(self, _now: float, _reason: str) -> None:
                raise AssertionError("frozen request must not roll back")

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "request_frozen_receipt.json").write_text("{}")
            machine = MigrationStateMachine()
            record = machine.create("m", "r")
            machine.transition("m", MigrationState.SHADOW, 1.0)
            self.assertFalse(step_m4_cancel(
                self.manager, self.short_table, machine, Adapter(root),
                Audit(), record, request(), max_output_tokens=1024,
                ignore_eos=False, source_free_kv_tokens=12000,
                source_guard_free_kv_tokens=8448,
                source_capacity_pressure=False, now=10.0, dry_run=False,
                recorder=Recorder(),
            ))
            self.assertEqual(record.state, MigrationState.SHADOW)

    def test_cancel_acceptance_requires_complete_source_response(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"kind": "manager_m4_cancel_decision", "decision": {
                    "action": "CANCEL_SHADOW", "output_tokens": 64,
                }},
                {"kind": "manager_m4_source_cleanup",
                 "source_abort_dispatched": False},
                {"kind": "manager_m4_cancelled"},
                {"kind": "run_end", "final_state": "CANCELLED"},
            ]
            (root / "phase9_audit.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            (root / "source_response.json").write_text(
                json.dumps({"token_ids": [1, 2, 3]}), encoding="utf-8"
            )
            (root / "takeover_state.json").write_text(json.dumps({
                "state": "CANCELLED", "source_continues_on_tp1": True,
            }), encoding="utf-8")
            (root / "unified_response.jsonl").write_text(
                "\n".join(json.dumps({
                    "token_id": token_id, "origin": "source",
                }) for token_id in (1, 2, 3)), encoding="utf-8"
            )
            self.assertEqual(
                accept_m4_cancel(root, root, 0, 3)["status"], "PASS"
            )
            (root / "request_frozen_receipt.json").write_text("{}")
            self.assertEqual(
                accept_m4_cancel(root, root, 0, 3)["status"], "FAIL"
            )

    def test_target_cleanup_retry_does_not_repeat_source_cleanup(self) -> None:
        class Adapter:
            def __init__(self, run_dir: Path) -> None:
                self.run_dir = run_dir
                self.source_cleanups = 0
                self.target_cleanups = 0

            def refresh_binding(self):
                return object()

            def disarm(self, _reason: str) -> None:
                pass

            def cancel(self, _reason: str, *, abort_source: bool):
                self.source_cleanups += 1
                return {"state": "CANCELLED", "source_abort_dispatched": False}

            def cancel_shadow_target(self, _reason: str):
                self.target_cleanups += 1
                if self.target_cleanups == 1:
                    raise ActionError("temporary target failure")
                return {"status": "CLEANED"}

        class Audit:
            def write(self, _row: dict) -> None:
                pass

        class Recorder:
            def on_rollback(self, _now: float, _reason: str) -> None:
                pass

        with TemporaryDirectory() as temporary:
            adapter = Adapter(Path(temporary))
            machine = MigrationStateMachine()
            record = machine.create("m", "r")
            machine.transition("m", MigrationState.SHADOW, 1.0)
            args = dict(
                max_output_tokens=1024, ignore_eos=False,
                source_free_kv_tokens=12000,
                source_guard_free_kv_tokens=8448,
                source_capacity_pressure=False, now=10.0,
                dry_run=False, recorder=Recorder(),
            )
            self.assertTrue(step_m4_cancel(
                self.manager, self.short_table, machine, adapter,
                Audit(), record, request(), **args,
            ))
            self.assertEqual(record.state, MigrationState.SHADOW)
            self.assertTrue(step_m4_cancel(
                self.manager, self.short_table, machine, adapter,
                Audit(), record, request(), **args,
            ))
            self.assertEqual(record.state, MigrationState.CANCELLED)
            self.assertEqual(adapter.source_cleanups, 1)
            self.assertEqual(adapter.target_cleanups, 2)


if __name__ == "__main__":
    unittest.main()
