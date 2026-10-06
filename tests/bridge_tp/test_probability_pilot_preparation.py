# SPDX-License-Identifier: Apache-2.0
"""Warmup is outside measurement; failed cancellation never becomes EOS PASS."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.experiment_probability_gate import (
    ProbabilityGate,
    source_load_eligibility,
)
from tools.bridge_tp.run_phase9_cap0_noop import (
    initial_source_prompt_budget,
    warmup_prompt_shapes,
)
from tools.bridge_tp.run_shadow_strategy_online_validation import (
    accept_shadow_without_cutover,
)


class TestProbabilityPilotPreparation(unittest.TestCase):
    def test_candidate_waits_for_source_stratum_without_rewriting_risk(self):
        gate = ProbabilityGate(0.0, "seed", assigned_action="START")
        risk = {
            "status": "VALID",
            "physical_feasible": True,
            "p_guard_est_bounds": [0.2, 0.3],
        }
        for tick, load in enumerate(
            (
                {"source_running": 1, "source_prefill_pending_kv_tokens": 0},
                {"source_running": 4, "source_prefill_pending_kv_tokens": 4000},
            )
        ):
            row = gate.observe(risk, tick, source_load_eligibility(load, 4))
            self.assertIsNone(gate.candidate)
            self.assertEqual(row["reason"], "LOAD_STRATUM_NOT_READY")
            self.assertIs(row["snapshot"]["physical_feasible"], True)
        load = {"source_running": 4, "source_prefill_pending_kv_tokens": 0}
        row = gate.observe(risk, 2, source_load_eligibility(load, 4))
        self.assertIs(row["first_feasible_candidate"], True)
        self.assertEqual(row["requested_action"], "START_SHADOW")
        self.assertEqual(row["snapshot"]["p_guard_est_bounds"], [0.2, 0.3])

    def test_budget_rounds_blocks_and_ignores_target_and_later_waves(self):
        manifest = {
            "jobs": [
                {"pool": pool, "wave": wave, "request": {"prompt": [1] * 4097}}
                for pool, wave in (("source", 0), ("source", 1), ("target", 0))
            ]
        }
        result = initial_source_prompt_budget(
            manifest,
            {"prompt": [1] * 8192},
            total_tokens=31408,
            guard_tokens=8448,
            block_size=16,
            minimum_headroom_tokens=1536,
        )
        self.assertEqual(result["first_burst_prompt_occupied_tokens"], 12304)
        self.assertEqual(result["initial_headroom_tokens"], 10656)
        self.assertEqual(result["first_burst_source_requests_including_anchor"], 2)

    def test_budget_rejects_plan_that_starts_inside_guard(self):
        manifest = {
            "jobs": [
                {
                    "pool": "source",
                    "request": {"prompt": [1] * 4096},
                }
            ]
            * 3
        }
        with self.assertRaisesRegex(ValueError, "headroom insufficient"):
            initial_source_prompt_budget(
                manifest,
                {"prompt": [1] * 10240},
                total_tokens=31408,
                guard_tokens=8448,
                block_size=16,
                minimum_headroom_tokens=1536,
            )

    def test_cancelled_shadow_is_diagnostic_even_when_source_later_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"kind": "abandon", "reason": "delta lag too high"},
                {"kind": "run_end", "final_state": "CANCELLED"},
            ]
            (root / "phase9_audit.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (root / "source_response.json").write_text(
                json.dumps(
                    {
                        "finish_reason": "stop",
                        "token_ids": [1, 2, 3],
                    }
                )
            )
            result = accept_shadow_without_cutover(root, root, 0, 100)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["outcome"], "SHADOW_CANCELLED_BEFORE_CUTOVER")
            self.assertEqual(result["cancellation_reasons"], ["delta lag too high"])
            self.assertNotIn("source-EOS", " ".join(result["errors"]))

    def test_real_source_eos_routes_to_full_evidence_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "phase9_audit.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "run_end",
                        "final_state": "COMPLETED_ON_TP1",
                    }
                )
                + "\n"
            )
            with patch(
                "tools.bridge_tp.run_shadow_strategy_online_validation."
                "accept_source_eos_after_shadow",
                return_value={"checked": True},
            ) as accept:
                self.assertEqual(
                    accept_shadow_without_cutover(root, root, 0, 100), {"checked": True}
                )
                accept.assert_called_once_with(root, root, 0, 100)

    def test_warmup_strips_migration_and_records_shapes_outside_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "warmup.json"
            requests = [
                {
                    "model": "m",
                    "prompt": [1] * length,
                    "max_tokens": 4096,
                    "ignore_eos": True,
                    "kv_transfer_params": {"migration": "anchor"},
                }
                for length in (4, 4, 8)
            ]
            with patch(
                "vllm.bridge_tp.controller.online_io.post_streaming_completion",
                return_value={"finish_reason": "length", "token_ids": [3, 4]},
            ) as post:
                warmup_prompt_shapes("http://pool", requests, output, 30)
                self.assertEqual(post.call_count, 2)
                for call in post.call_args_list:
                    payload = call.args[1]
                    self.assertEqual(payload["max_tokens"], 2)
                    self.assertIs(payload["ignore_eos"], False)
                    self.assertNotIn("kv_transfer_params", payload)
                saved = json.loads(output.read_text())
                self.assertIs(saved["excluded_from_measured_episode"], True)
                self.assertEqual([r["prompt_tokens"] for r in saved["records"]], [4, 8])

    def test_warmup_failure_blocks_episode(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "vllm.bridge_tp.controller.online_io.post_streaming_completion",
                return_value={"finish_reason": "error"},
            ),
            self.assertRaisesRegex(RuntimeError, "did not complete"),
        ):
            warmup_prompt_shapes(
                "http://pool",
                [
                    {
                        "model": "m",
                        "prompt": [1],
                    }
                ],
                Path(directory) / "warmup.json",
                30,
            )
