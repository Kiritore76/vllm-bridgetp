# SPDX-License-Identifier: Apache-2.0
"""Native equal context limits and workload routing without length cutoffs."""

import contextlib
import copy
import dataclasses
import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.bridge_tp.test_pressure_workload import cfg, p, reading
from tools.bridge_tp import launch_pressure_balanced_pilot as launch
from tools.bridge_tp import qwen3_calibration_contract as contract
from tools.bridge_tp import run_qwen3_stay_migrate as runner


class ExpandedContextTests(unittest.TestCase):
    def test_workload_choices_do_not_use_fixed_prompt_thresholds(self):
        for prompt in (512, 2048, 6144):
            for load in (0.1, 0.9, 2.0):
                choices = []
                for boundary, very_long in ((1, 1), (12000, 16000)):
                    result = p.admit(
                        "r",
                        prompt,
                        512,
                        (reading(load=load), reading("tp4")),
                        dataclasses.replace(
                            cfg(),
                            prompt_boundary=boundary, very_long_prompt_tokens=very_long
                        ),
                        p.ReservationLedger(),
                        10,
                        "k",
                    )
                    choices.append(
                        (
                            result.pool_id,
                            result.routing_reason,
                            result.routing_pressure,
                            result.routing_capacity,
                        )
                    )
                self.assertEqual(*choices)

    def test_spec_and_launcher_require_explicit_native_profile(self):
        with patch.object(contract, "sha", return_value="MODEL_CONFIG_HASH"):
            spec = contract.migration_spec("model")
        before = copy.deepcopy(spec)
        profile = contract.read(contract.PROFILE)
        model = profile["model_geometry"] | dict(
            model_type="qwen3",
            architectures=["Qwen3ForCausalLM"],
            max_position_embeddings=32768,
        )
        contract.expanded_context_spec(spec, model)
        _, args = launch.arguments(spec)
        self.assertEqual((args.max_model_len, args.tp4_max_model_len), (32768, 32768))
        _, old = launch.arguments(before)
        self.assertEqual((old.max_model_len, old.tp4_max_model_len), (8192, 16384))
        spec.pop("context_profile")
        with self.assertRaises(ValueError):
            launch.arguments(spec)
        with self.assertRaises(ValueError):
            contract.expanded_context_spec(
                before, model | dict(max_position_embeddings=16384)
            )

    def test_cli_equal_limits_and_existing_policy_without_state_only_switch(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = runner.main(
                [
                    "--routine-four-mode",
                    "--current-machine",
                    "--expanded-context",
                    "--repetitions",
                    "1",
                    "--model",
                    "unused",
                    "--output",
                    "unused",
                    "--expected-revision",
                    "CPU",
                    "--validate-only",
                ]
            )
        result = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["runs"], 12)
        self.assertEqual(result["service_context_limits"], dict(tp1=32768, tp2=32768))
        self.assertTrue(result["policy_overrides"]["kv_routing_priority"])
        self.assertNotIn("routing_service_state_only", result["policy_overrides"])

    def test_native_check_rejects_too_small_memory_or_wrong_effective_limit(self):
        rows = [
            dict(
                context_limit=32768, free_blocks=2324, block_size=16, total_blocks=2325
            ),
            dict(
                context_limit=32768,
                free_blocks=11711,
                block_size=16,
                total_blocks=11712,
            ),
        ]
        plan = dict(
            run_dir=Path("unused"),
            expected_context_limits=(32768, 32768),
            config=dict(policy={}),
            config_path="unused",
        )
        from tools.bridge_tp import run_pressure_calibration as calibration

        with patch.object(calibration, "wait_empty", return_value=rows):
            runner.prepare_native(plan, Mock())
        for changes in (dict(free_blocks=2100), dict(context_limit=8192)):
            invalid = [rows[0] | changes, rows[1]]
            with (
                patch.object(calibration, "wait_empty", return_value=invalid),
                self.assertRaises(ValueError),
            ):
                runner.prepare_native(plan, Mock())

    def test_native_check_of_single_class_layout_reads_only_started_side(self):
        """S3b-2: L2 has no TP1 source; its template geometry stays."""
        from tools.bridge_tp import run_pressure_calibration as calibration

        target = dict(
            context_limit=32768, free_blocks=11711, block_size=16, total_blocks=11712
        )
        plan = dict(
            run_dir=Path("unused"),
            expected_context_limits=(32768, 32768),
            config=dict(policy={}, tp1_total_kv_blocks=2325),
            config_path="unused",
            instances=dict(legacy_pool_aliases={"tp4": "B-g2-tp2"}),
        )
        with patch.object(calibration, "wait_empty", return_value=[target]) as wait:
            runner.prepare_native(plan, Mock())
        self.assertEqual(wait.call_args.kwargs["pools"], ["target"])
        self.assertEqual(plan["config"]["tp1_total_kv_blocks"], 2325)
        self.assertEqual(plan["config"]["tp4_total_kv_blocks"], 11712)
        with (
            patch.object(
                calibration, "wait_empty", return_value=[target | dict(free_blocks=100)]
            ),
            self.assertRaises(ValueError),
        ):
            runner.prepare_native(plan, Mock())
