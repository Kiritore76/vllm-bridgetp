# SPDX-License-Identifier: Apache-2.0
"""Real build/configure/controller preflight and launch-boundary evidence; CPU."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp import test_six_edge_batch as batch_tests
from tests.bridge_tp.test_qwen3_six_card import build
from tools.bridge_tp import migration_plan
from tools.bridge_tp import qwen3_long_burst_pair as burst
from tools.bridge_tp import qwen3_six_card as static_six
from tools.bridge_tp import run_pressure_balanced_pilot as controller
from tools.bridge_tp import run_pressure_comparison as comparison
from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp import six_conversion as six
from tools.bridge_tp import six_edge_batch as batch

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "experiments/platforms/qwen3_pressure_observation"


def built_plan(root, layout):
    """Build the production deep-burst plan; only model geometry is a CPU fake."""
    spec, common, settings = build(root, layout, setup_only=True)
    spec.update(
        controller_config_template=str(
            REPO / "experiments/phase9/pressure_balanced_v2/controller_template.json"
        ),
        policy_config=str(ASSETS / "policy.json"),
        service_evidence=str(ASSETS / "service_evidence.json"),
        health_reference=str(ASSETS / "health_reference.json"),
        checkpoint_sha256=runner.CHECKPOINT_SHA,
        conversion=dict(
            profile="six_card",
            base_port=8400,
            prestart=False,
            sequence=[
                dict(
                    bundle="B" if layout in ("L0", "L1AC") else "C",
                    target_mode="TP2" if layout in ("L0", "L2C") else "TP1",
                    trigger_s=60,
                    deadline_s=900,
                    method="DRAIN",
                )
            ],
        ),
    )
    common.write_json = runner.write
    cfg = burst.config(json.loads((ASSETS / "inputs.json").read_text()))
    cfg["recipes"] = [r for r in cfg["recipes"] if r["scenario"] == "deep_long_burst"]
    cfg["arms"] = ["migrate"]
    cfg["recipes"][0]["arm_order_offset"] = 0
    cfg["policy_overrides"].update(
        kv_short_exempt_blocks=64,
        overload_migrates=True,
        overload_min_waiting=4,
        overload_kv_fraction=0.7,
        migration_current_length=True,
    )
    plans = comparison.build_plans(spec, cfg, {}, root, common, settings)
    plan = plans[0][3]
    if "--no-migration" in plan["controller_argv"]:
        plan["controller_argv"].remove("--no-migration")
    with patch.object(runner, "configure_predictor"):
        runner.adapt_plan(plan, "migrate", Path("checkpoint"), Path("model"))
    static_six.configure(plan)
    plan["scheduler"] = "unified"
    plan["controller_argv"] += ["--scheduler", "unified", "--hold-unstarted", "8"]
    migration_plan.install(plan)
    migration_plan.activate_instance_ids(plan)
    six.configure(plan)
    return plan


class PreflightTests(unittest.TestCase):
    def test_real_build_configure_and_full_controller_preflight_all_layouts(self):
        for layout in six.PATH:
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as folder:
                plan = built_plan(Path(folder), layout)
                _, _, _, jobs, receipt = runner.controller_preflight(plan)
                self.assertEqual(len(jobs), 156)
                self.assertTrue(receipt["migration_execution_enabled"])
                self.assertEqual(receipt["instance_manifest"], plan["instances"])
                self.assertNotIn("--deferred-migration", plan["controller_argv"])
                self.assertNotIn("--no-migration", plan["controller_argv"])

    def test_single_class_exception_requires_all_owned_dynamic_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = built_plan(Path(folder), "L0")
            runner.controller_preflight(plan)
            args = controller.parse_args(
                plan["controller_argv"][2:] + ["--validate-only"]
            )
            for key, value in [
                ("dynamic_conversion", False),
                ("target_scoped_receivers", False),
            ]:
                cfg = NS(**(plan["config"] | {key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    controller.instance_manifest(args, cfg)
            args.migration_targets = None
            with self.assertRaises(ValueError):
                controller.instance_manifest(args, NS(**plan["config"]))

    def test_static_single_class_rejects_enabled_migration_and_accepts_no_migration(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            plan = built_plan(Path(folder), "L3")
            runner.controller_preflight(plan)
            args = controller.parse_args(
                plan["controller_argv"][2:] + ["--validate-only"]
            )
            cfg = NS(**(plan["config"] | dict(dynamic_conversion=False)))
            with self.assertRaises(ValueError):
                controller.instance_manifest(args, cfg)
            args.no_migration = True
            self.assertEqual(controller.instance_manifest(args, cfg), plan["instances"])

    def test_wrong_canonical_identity_or_gpu_count_remains_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = built_plan(Path(folder), "L0")
            runner.controller_preflight(plan)
            args = controller.parse_args(
                plan["controller_argv"][2:] + ["--validate-only"]
            )
            original = copy.deepcopy(plan["instances"])
            for mutation in ("runtime", "gpus", "legacy"):
                manifest = copy.deepcopy(original)
                if mutation == "runtime":
                    manifest["runtime_pool_ids"] = dict(tp1="other")
                elif mutation == "gpus":
                    manifest["instances"][-1]["physical_gpu_ids"] = ["2"]
                else:
                    manifest.pop("runtime_pool_ids")
                args.instances.write_text(json.dumps(manifest))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    controller.instance_manifest(args, NS(**plan["config"]))

    def test_real_runner_prelaunch_failure_emits_safe_receipt_and_original_error(self):
        with tempfile.TemporaryDirectory() as folder:
            case = Path(folder) / "L0-L1AC"
            case.mkdir()
            output = case / "trial"
            with (
                patch.object(runner, "require_revision"),
                patch.object(
                    runner,
                    "runtime_inventory",
                    side_effect=ValueError("CPU startup failure"),
                ),
                patch.object(runner.pilot, "run_scene") as launch,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                status = runner.main(
                    [
                        "--cards",
                        "6",
                        "--static-layout",
                        "L0",
                        "--conversion",
                        "L0-L1AC",
                        "--long-burst-pair",
                        "--expanded-context",
                        "--current-machine",
                        "--scheduler",
                        "unified",
                        "--hold-queue",
                        "--kv-short-exempt",
                        "--overload-migrate",
                        "--migrate-current-length",
                        "--repetitions",
                        "1",
                        "--model",
                        "cpu",
                        "--output",
                        str(output),
                        "--expected-revision",
                        "CPU",
                    ]
                )
            self.assertEqual(status, 1)
            launch.assert_not_called()
            (case / "runner_exit_code.txt").write_text(str(status))
            value = batch.assess_case(case, "L0-L1AC", {})
            self.assertTrue(value["launch_not_started"])
            self.assertTrue(value["continuation_safe"])
            self.assertFalse(value["passed"])
            self.assertIn("CPU startup failure", value["runner_error"])

    def test_launch_attempt_marker_precedes_launcher_even_when_it_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner.write(
                root / "owned_launch_receipt.json",
                dict(schema="BRIDGETP_OWNED_LAUNCH_V1", state="NOT_STARTED"),
            )

            def launch(*args, **kwargs):
                value = json.loads((root / "owned_launch_receipt.json").read_text())
                self.assertEqual(value["state"], "LAUNCH_ATTEMPTED")
                raise RuntimeError("simulated launch failure")

            with (
                patch.object(runner.pilot, "run_scene", side_effect=launch),
                self.assertRaises(RuntimeError),
            ):
                runner.launch_scene(root, "unused")
            self.assertEqual(
                json.loads((root / "owned_launch_receipt.json").read_text())["state"],
                "LAUNCH_ATTEMPTED",
            )


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.case = batch_tests.BatchTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        (self.case.scene / "owned_shutdown_receipt.json").unlink()
        (self.case.directory / "runner_exit_code.txt").write_text("1")
        self.case.write(
            "result.json",
            dict(
                status="TECHNICAL_FAILURE",
                runs=[],
                total_profile_attempts=0,
                error="original preflight error",
            ),
            trial=True,
        )
        self.case.write(
            "owned_launch_receipt.json",
            dict(schema="BRIDGETP_OWNED_LAUNCH_V1", state="NOT_STARTED"),
            trial=True,
        )

    def test_explicit_not_started_keeps_failure_but_allows_continuation(self):
        result = self.case.assess()
        self.assertFalse(result["passed"])
        self.assertFalse(result["cleanup_confirmed"])
        self.assertTrue(result["continuation_safe"])
        self.assertEqual(result["runner_error"], "original preflight error")
        self.assertIn("original preflight error", result["errors"][0])

    def test_missing_or_attempted_marker_cannot_prove_no_launch(self):
        path = self.case.directory / "trial/owned_launch_receipt.json"
        path.unlink()
        self.assertFalse(self.case.assess()["continuation_safe"])
        for state in ("LAUNCH_ATTEMPTED", "UNKNOWN"):
            self.case.write(
                "owned_launch_receipt.json",
                dict(schema="BRIDGETP_OWNED_LAUNCH_V1", state=state),
                trial=True,
            )
            self.assertFalse(self.case.assess()["continuation_safe"])

    def test_started_attempts_or_process_evidence_override_not_started_claim(self):
        for name in ("owned_processes.json", "effective_argv.json"):
            path = self.case.directory / "trial" / name
            path.write_text("[]")
            self.assertFalse(self.case.assess()["continuation_safe"])
            path.unlink()
        self.case.write(
            "result.json",
            dict(status="TECHNICAL_FAILURE", runs=[{}], total_profile_attempts=1),
            trial=True,
        )
        self.assertFalse(self.case.assess()["continuation_safe"])

    def test_unconfirmed_shutdown_overrides_not_started_claim(self):
        self.case.write(
            "owned_shutdown_receipt.json", dict(all_owned_groups_ended=False)
        )
        self.assertFalse(self.case.assess()["continuation_safe"])


if __name__ == "__main__":
    unittest.main()
