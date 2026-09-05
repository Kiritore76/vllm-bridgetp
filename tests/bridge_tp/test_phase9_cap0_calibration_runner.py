# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "bridge_tp"
    / "run_phase9_cap0_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("phase9_cap0_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestGuardCandidate(unittest.TestCase):
    def test_uses_preemption_or_censored_minimum_and_rounds(self) -> None:
        result = MODULE.calculate_guard_candidate(
            [
                {
                    "first_preemption_free_kv_tokens": 100,
                    "minimum_free_kv_tokens": 80,
                    "maximum_ewma_decline_tokens_s": 10.0,
                },
                {
                    "first_preemption_free_kv_tokens": None,
                    "minimum_free_kv_tokens": 120,
                    "maximum_ewma_decline_tokens_s": 5.0,
                },
            ],
            block_size=16,
            tp1_total_tokens=1000,
        )
        self.assertEqual(result["f_values"], [100, 120])
        self.assertEqual(result["raw_guard_tokens"], 200.0)
        self.assertEqual(result["guard_candidate_tokens"], 208)
        self.assertEqual(result["status"], "CANDIDATE_NOT_FROZEN")

    def test_clamps_below_last_block(self) -> None:
        result = MODULE.calculate_guard_candidate(
            [
                {
                    "first_preemption_free_kv_tokens": 990,
                    "minimum_free_kv_tokens": 900,
                    "maximum_ewma_decline_tokens_s": 100.0,
                }
            ],
            block_size=16,
            tp1_total_tokens=1000,
        )
        self.assertEqual(result["guard_candidate_tokens"], 984)


class TestCalibrationAcceptance(unittest.TestCase):
    def test_accepts_three_jobs_completed_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "controller"
            background = root / "background"
            controller.mkdir()
            background.mkdir()
            (background / "background_summary.json").write_text(
                json.dumps({"jobs": 3, "completed": 3, "failed": 0}),
                encoding="utf-8",
            )
            records = [
                {
                    "kind": "telemetry",
                    "tp1": {"preemptions_total": 0},
                    "capacity_signal": {
                        "free_kv_tokens": 100,
                        "decline_rate_tokens_s": 2.0,
                    },
                },
                {"kind": "decision"},
                {"kind": "run_end", "final_state": "COMPLETED_ON_TP1"},
            ]
            (controller / "phase9_audit.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            result = MODULE.accept_calibration(controller, background)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["migration_transitions"], 0)

    def test_rejects_migration_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "controller"
            background = root / "background"
            controller.mkdir()
            background.mkdir()
            (background / "background_summary.json").write_text(
                json.dumps({"jobs": 3, "completed": 3, "failed": 0}),
                encoding="utf-8",
            )
            records = [
                {"kind": "telemetry"},
                {"kind": "decision"},
                {"kind": "transition", "to": "SHADOW"},
                {"kind": "run_end", "final_state": "COMPLETED_ON_TP1"},
            ]
            (controller / "phase9_audit.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            result = MODULE.accept_calibration(controller, background)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["migration_transitions"], 1)


if __name__ == "__main__":
    unittest.main()
