# SPDX-License-Identifier: Apache-2.0
"""Registered rotation and real runner planning/preflight; CPU only."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.bridge_tp.test_qwen3_six_card import build
from tools.bridge_tp import qwen3_phase_shift as phase
from tools.bridge_tp import qwen3_six_card as static_six
from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp import six_rotation as rotation

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "experiments/platforms/qwen3_pressure_observation"


class RotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = rotation.read(ASSETS / "inputs.json")
        cls.config = phase.config(cls.inputs, rotation.SCENARIO)

    def test_actual_token_input_hash_phase_counts_and_registered_budget(self):
        recipe = rotation.validate_recipe(self.config)
        self.assertEqual(len(recipe["jobs"]), 36371)
        self.assertEqual(recipe["planned_output_tokens"], 5574784)
        self.assertEqual(
            sum(j["workload_role"] != "short_probe" for j in recipe["jobs"]), 114
        )
        self.assertEqual(recipe["phases"][-1][1], 3900)
        self.assertEqual(recipe["phases"][2], [420, 900, 8, 0.02, "m_mixed"])
        self.assertEqual(recipe["phases"][4], [1800, 2040, 2, 0, "q_recovery"])
        self.assertLess(recipe["jobs"][-1]["at_s"], 3900)
        self.assertEqual(len({j["id"] for j in recipe["jobs"]}), 36371)

    def test_changed_input_is_rejected_before_launch(self):
        cfg = copy.copy(self.config)
        cfg["recipes"] = [copy.copy(self.config["recipes"][0])]
        cfg["recipes"][0]["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "contract changed"):
            rotation.validate_recipe(cfg)

    def test_real_runner_builds_and_preflights_static_arms_before_gpu(self):
        # Hardware, model geometry, predictor load and actual engine invocation
        # are CPU substitutes. Recipe, runner/build/configure and controller
        # preflight are real; the old validate-only early return is not used.
        for layout in getattr(self, "layouts", ("L1AC", "L2C")):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                spec, common, settings = build(root, layout, setup_only=True)
                spec["server_settings"] = {}
                spec["controller_config_template"] = str(
                    REPO
                    / "experiments/phase9/pressure_balanced_v2/controller_template.json"
                )

                def compact_write(path, value):
                    # Preserve exact logical inputs; avoid pretty-printing large
                    # fixture files when checking planning rather than layout.
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_text(json.dumps(value), encoding="utf-8")

                common.write_json = compact_write
                model = root / "model"
                model.mkdir()
                (model / "config.json").write_text("{}")
                output = root / "result"
                captured = []

                def preflight(plans, *, layout=layout, captured=captured):
                    self.assertEqual(len(plans), 1)
                    _, _, _, current = plans[0]
                    _, _, _, jobs, receipt = runner.controller_preflight(current)
                    self.assertEqual(len(jobs), len(self.config["recipes"][0]["jobs"]))
                    self.assertTrue(receipt["migration_execution_enabled"])
                    self.assertFalse(current["migration_coverage_required"])
                    self.assertEqual(current["expected_context_limits"], (32768, 32768))
                    if layout == "L2C":
                        self.assertEqual(current["receiver_stage"], 3)
                        self.assertEqual(
                            len(current["migration_plan"]["execution_targets"]), 2
                        )
                    captured.append(current)
                    raise RuntimeError("CPU_STOP_AFTER_REAL_PREFLIGHT")

                args = [
                    "--cards",
                    "6",
                    "--static-layout",
                    layout,
                    "--phase-shift",
                    "--phase-scenario",
                    self.config["recipes"][0]["scenario"],
                    "--scheduler",
                    "unified",
                    "--hold-queue",
                    "--kv-short-exempt",
                    "--overload-migrate",
                    "--migrate-current-length",
                    "--expanded-context",
                    "--current-machine",
                    "--repetitions",
                    "1",
                    "--model",
                    str(model),
                    "--output",
                    str(output),
                    "--expected-revision",
                    "cpu-fixture",
                ]
                if layout == "L2C":
                    args += ["--migration-targets", "all"]
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    patch.object(runner, "require_revision"),
                    patch.object(runner, "write", side_effect=compact_write),
                    patch.object(runner, "runtime_inventory", return_value={}),
                    patch.object(runner, "runtime_versions", return_value={}),
                    patch.object(runner, "verified_inputs", return_value={}),
                    patch.object(runner, "migration_spec", return_value=spec),
                    patch.object(runner, "expanded_context_spec"),
                    patch.object(runner, "configure_predictor"),
                    patch.object(
                        runner.pilot, "arguments", return_value=(common, settings)
                    ),
                    patch.object(runner, "serving_preflight", side_effect=preflight),
                    patch.object(runner, "pack", return_value={"sha256": "cpu-only"}),
                    patch.object(runner.pilot, "run_scene") as gpu,
                ):
                    self.assertEqual(runner.main(args), 1)
                self.assertEqual(
                    len(captured),
                    1,
                    (output / "failure_traceback.txt").read_text(),
                )
                self.assertIn(
                    "CPU_STOP_AFTER_REAL_PREFLIGHT",
                    rotation.read(output / "result.json")["error"],
                )
                gpu.assert_not_called()
                config = rotation.read(output / "comparison_recipe.json")
                self.assertEqual(config["horizon_s"], self.config["horizon_s"])
                self.assertEqual(
                    config["recipes"][0]["input_sha256"],
                    self.config["recipes"][0]["input_sha256"],
                )
                current = captured[0]
                argv = current["controller_argv"]
                self.assertEqual(argv[argv.index("--max-seconds") + 1], "5400")
                with self.assertRaisesRegex(ValueError, "budget/options"):
                    rotation.configure_plan(
                        dict(current, conversion=dict(max_conversions=4)),
                        recipe=self.config["recipes"][0],
                    )
                if layout == "L2C":
                    with self.assertRaisesRegex(ValueError, "both migration"):
                        rotation.configure_plan(
                            dict(current, receiver_stage=2),
                            recipe=self.config["recipes"][0],
                        )

    def test_cli_rejects_four_cards_and_legacy_scheduler(self):
        for cards, scheduler in ((4, "unified"), (6, "pressure")):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                runner.main(
                    [
                        "--cards",
                        str(cards),
                        "--static-layout",
                        "L1AC" if cards == 6 else "L1A",
                        "--phase-shift",
                        "--phase-scenario",
                        rotation.SCENARIO,
                        "--scheduler",
                        scheduler,
                        "--hold-queue",
                        "--expanded-context",
                        "--current-machine",
                        "--repetitions",
                        "1",
                        "--validate-only",
                        "--model",
                        "model",
                        "--output",
                        "unused",
                        "--expected-revision",
                        "cpu",
                    ]
                )

    def test_missing_gpu_evidence_never_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "trial"
            root.mkdir()
            value = rotation.assess(root)
            self.assertFalse(value["passed"])
            self.assertFalse(value["performance_validated"])
            self.assertTrue(value["errors"])

    def test_zero_migration_baseline_still_requires_complete_output(self):
        with tempfile.TemporaryDirectory() as folder:
            current = build(Path(folder), "L1AC")
            directory = current["run_dir"]
            directory.mkdir(parents=True)
            rotation.write(
                directory / "layout_status.json",
                dict(
                    instances={
                        row["instance_id"]: dict(lifecycle="ACTIVE", inventory={})
                        for row in current["instances"]["instances"]
                    }
                ),
            )
            requests = [
                dict(
                    job_id="one",
                    source_done=True,
                    source_error=None,
                    request=dict(max_tokens=1),
                    response=dict(emitted=[{}]),
                )
            ]
            rotation.write(directory / "pressure_request_results.json", requests)
            proof = dict(sessions=[])
            self.assertFalse(static_six.assess(current, proof, 1)["passed"])
            current["migration_coverage_required"] = False
            result = static_six.assess(current, proof, 1)
            self.assertTrue(result["passed"])
            self.assertFalse(result["committed_migration_observed"])
            self.assertFalse(result["performance_validated"])
            requests[0]["response"]["emitted"] = []
            rotation.write(directory / "pressure_request_results.json", requests)
            self.assertFalse(static_six.assess(current, proof, 1)["passed"])


if __name__ == "__main__":
    unittest.main()
