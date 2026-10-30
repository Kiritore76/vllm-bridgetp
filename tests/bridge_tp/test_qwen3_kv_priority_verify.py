# SPDX-License-Identifier: Apache-2.0
"""Single fixed-work rerun and explicit mechanism versus performance acceptance."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp import qwen3_kv_priority_verify as verify
from tools.bridge_tp import qwen3_pressure_four_mode as pressure


def evidence():
    raw = [dict(server_id="s", response=dict(committed=True, handoff_stall_s=0.5))]
    audit = [
        dict(
            kind="pressure_decision",
            execution="RESERVED",
            unix_s=4.0,
            decision=dict(
                capacity_path=True,
                request_id="s",
                candidates=[
                    dict(
                        selected=True,
                        selection_eligible=True,
                        kv_joint_benefit=dict(
                            method="OBSERVED_KV_X_MEAN_EARLY_RELEASE_SECONDS_V1",
                            relief_block_seconds=200,
                        ),
                        kv_priority_override=True,
                        soft_warnings=["emergency target service budget unavailable"],
                    )
                ],
            ),
        ),
        dict(
            kind="pressure_actual_start", request_id="s", migration_id="m", unix_s=5.0
        ),
        dict(kind="pressure_source_release_confirmed", migration_id="m"),
    ]
    return raw, audit


class KVPriorityVerifyTests(unittest.TestCase):
    def test_acceptance_requires_actual_maximum_joint_score(self):
        raw, audit = evidence()
        candidates = audit[0]["decision"]["candidates"]
        candidates.append(dict(
            selected=False, selection_eligible=True,
            kv_joint_benefit=dict(relief_block_seconds=300),
        ))
        result = verify.assess(raw, audit)
        self.assertFalse(result["joint_scoring_pass"])
        self.assertFalse(result["verification_pass"])
        candidates[1]["selection_eligible"] = False
        self.assertTrue(verify.assess(raw, audit)["verification_pass"])

    def test_current_machine_runtime_reaches_server_preflight_without_old_identity(
        self,
    ):
        from tools.bridge_tp import run_qwen3_stay_migrate as runner

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            output_dir = Path(tmp) / "new-machine-run"
            for name, value in (
                ("require_revision", None),
                ("runtime_inventory", dict(gpus=[])),
                ("runtime_versions", {}),
                ("verified_inputs", {}),
                ("migration_spec", {}),
                ("pack", dict(sha256="CPU_ONLY")),
            ):
                stack.enter_context(patch.object(runner, name, return_value=value))
            stack.enter_context(
                patch.object(runner.pilot, "arguments", return_value=(None, None))
            )
            stack.enter_context(
                patch.object(runner.comparison, "build_plans", return_value=[])
            )
            stop = stack.enter_context(
                patch.object(
                    runner,
                    "serving_preflight",
                    side_effect=RuntimeError("CPU_STOP_BEFORE_GPU"),
                )
            )
            idle = stack.enter_context(patch.object(runner, "gpu_idle"))
            with contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(
                    [
                        "--kv-priority-verify",
                        "--current-machine",
                        "--repetitions",
                        "1",
                        "--model",
                        "unused",
                        "--output",
                        str(output_dir),
                        "--expected-revision",
                        "CPU_RUNTIME",
                    ]
                )
            self.assertEqual(code, 1)  # Deliberate stop before any GPU work.
            stop.assert_called_once()
            idle.assert_not_called()
            identity = json.loads((output_dir / "identity.json").read_text())
            self.assertFalse(identity["prior_environment_comparison"])
            baseline = json.loads((output_dir / "baseline_reference.json").read_text())
            self.assertIsNone(baseline["prior"])
            self.assertIn(
                "CPU_STOP_BEFORE_GPU",
                (output_dir / "failure_traceback.txt").read_text(),
            )

    def test_current_machine_plan_never_requires_a_previous_directory(self):
        inputs = dict(
            requests={str(n): dict(prompt=list(range(n))) for n in (2048, 6144)}
        )
        cfg, baseline = verify.config(inputs)
        self.assertEqual(len(cfg["recipes"][0]["jobs"]), 375)
        self.assertIsNone(baseline["prior"])
        self.assertNotIn("summary", baseline)
        from tools.bridge_tp import run_qwen3_stay_migrate as runner

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = runner.main(
                [
                    "--kv-priority-verify",
                    "--current-machine",
                    "--repetitions",
                    "1",
                    "--model",
                    "unused",
                    "--output",
                    "unused",
                    "--expected-revision",
                    "CPU_PLAN",
                    "--validate-only",
                ]
            )
        record = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(record["runs"], 1)
        self.assertEqual(
            record["environment_mode"], "CURRENT_MACHINE_NO_PRIOR_COMPARISON"
        )
        self.assertTrue(record["policy_overrides"]["kv_emergency_priority"])

    def test_only_one_migrate_scene_same_prefill_jobs_and_budget(self):
        inputs = dict(
            requests={str(n): dict(prompt=list(range(n))) for n in (2048, 6144)}
        )
        original = pressure.config(inputs)["recipes"][-1]
        with tempfile.TemporaryDirectory() as tmp:
            prior = Path(tmp)
            prior.joinpath("result.json").write_text(
                json.dumps(
                    dict(
                        runs=[
                            dict(
                                name="pressure-four-prefill_competition-r1-migrate",
                                status="COMPLETED",
                                input_sha256=original["input_sha256"],
                                summary=dict(migration_starts=0),
                            )
                        ]
                    )
                )
            )
            cfg, baseline = verify.config(inputs, prior)
        self.assertEqual(cfg["arms"], ["migrate"])
        self.assertEqual(len(cfg["recipes"]), 1)
        self.assertEqual(cfg["recipes"][0]["jobs"], original["jobs"])
        self.assertEqual(len(cfg["recipes"][0]["jobs"]), 375)
        self.assertEqual(
            cfg["policy_overrides"],
            dict(capacity_guard_blocks=125, kv_emergency_priority=True),
        )
        self.assertEqual(baseline["input_sha256"], original["input_sha256"])

    def test_actual_bypass_commit_free_and_one_second_handoff_are_required(self):
        raw, audit = evidence()
        self.assertTrue(verify.assess(raw, audit)["verification_pass"])
        self.assertFalse(verify.assess(raw, audit[:-1])["verification_pass"])
        bad = copy.deepcopy(raw)
        bad[0]["response"]["handoff_stall_s"] = 1.001
        self.assertTrue(verify.assess(bad, audit)["mechanism_pass"])
        self.assertFalse(verify.assess(bad, audit)["verification_pass"])
        no_bypass = copy.deepcopy(audit)
        no_bypass[0]["decision"]["candidates"][0]["soft_warnings"] = []
        self.assertFalse(verify.assess(raw, no_bypass)["mechanism_pass"])
        self.assertFalse(verify.assess(raw, [])["verification_pass"])

    def test_ordinary_or_legacy_start_cannot_count_as_fix_coverage(self):
        for key in ("capacity_path", "kv_priority_override"):
            raw, audit = evidence()
            if key == "capacity_path":
                audit[0]["decision"][key] = False
            else:
                audit[0]["decision"]["candidates"][0][key] = False
            with self.assertRaises(ValueError):
                verify.assess(raw, audit)
