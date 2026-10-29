# SPDX-License-Identifier: Apache-2.0
"""Qwen3 adapters must call the original policy and never bind migration actions."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from tools.bridge_tp import qwen3_pressure_observation as obs
from tools.bridge_tp import run_pressure_balanced_pilot as original
from tools.bridge_tp.run_qwen3_pressure_observation import child_arguments


class ObservationTests(unittest.TestCase):
    def test_summary_preserves_traffic_quality_and_rejects_executed_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            decision = obs.reactions()["cases"][0]["last_decision"]
            rows = [
                dict(kind="pressure_admission", admission=dict(pool_id="tp1")),
                dict(
                    kind="pressure_decision",
                    execution="ADVISORY",
                    decision=decision,
                    routing_relief=dict(phase="NORMAL"),
                    prediction_inputs=[],
                ),
                dict(
                    kind="pressure_final",
                    outcome="COMPLETED",
                    sessions=[dict(started=False)],
                ),
            ]
            audit = d / "pressure_audit.jsonl"
            audit.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            (d / "pressure_request_results.json").write_text(
                json.dumps([dict(source_done=True, source_error=None)])
            )
            (d / "native_tokens.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        dict(
                            kind="visible_token",
                            job_id="one",
                            index=i,
                            unix_s=10 + i * 0.02,
                        )
                    )
                    for i in range(3)
                )
                + "\n"
            )
            summary = obs.summarize(d)
            self.assertEqual(summary["output_tokens"], 3)
            self.assertEqual(summary["slo_pass"], 1)
            rows[1]["execution"] = "STARTED"
            audit.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaises(ValueError):
                obs.summarize(d)

    def test_original_policy_reaction_matrix(self):
        result = obs.reactions()
        self.assertEqual(len(result["cases"]), 14)
        self.assertTrue(all(r["passed"] for r in result["cases"]))
        self.assertFalse(result["migration_execution_enabled"])
        self.assertEqual(
            next(r for r in result["cases"] if r["name"] == "choose_longest_tier")[
                "selected_ids"
            ],
            ["long"],
        )

    def test_measured_tp2_workload_profiles(self):
        profiles = obs.profiles()
        self.assertEqual(profiles["tp1"]["capacities"], [[1, 4], [2, 4], [3, 4]])
        self.assertEqual(profiles["tp4"]["capacities"], [[1, 16], [2, 16], [3, 16]])
        scope = obs.read(obs.ASSETS / "scope.json")
        self.assertIsNone(scope["general_guard_blocks"])
        self.assertFalse(scope["migration_execution_enabled"])

    def test_stress_matrix_is_bounded_and_preserves_stop_scope(self):
        inputs = dict(requests={"2048": dict(prompt=list(range(2048)), model="m")})
        all_jobs = []
        for scene in obs.scenes():
            jobs = obs.jobs(scene, inputs)
            all_jobs += jobs
            self.assertTrue(
                all(
                    len(r["request"]["prompt"]) + r["request"]["max_tokens"] <= 8192
                    for r in jobs
                )
            )
            self.assertTrue(all(r["at_s"] <= 120 for r in jobs))
            self.assertLessEqual(len(jobs), 40)
            self.assertTrue(
                all(
                    r["request"]["ignore_eos"] is False
                    for r in jobs
                    if "natural_template" in r
                )
            )
        self.assertEqual(len({r["id"] for r in all_jobs}), len(all_jobs))

    def test_child_preflight_uses_existing_qwen3_observation_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            jobs_path = directory / "arrivals.json"
            jobs_path.write_text(
                json.dumps(
                    [
                        dict(
                            id="test",
                            at_s=0,
                            request=dict(prompt=[1] * 10, max_tokens=512),
                        )
                    ]
                )
            )
            controller = dict(
                source_url="http://127.0.0.1:8100",
                target_url="http://127.0.0.1:8200",
                block_size=16,
                tp1_total_kv_blocks=2326,
                tp4_total_kv_blocks=11713,
                transfer_mode="FULL_SPEED",
                target_tp_size=2,
                rate={},
                policy=dict(kv_bytes_per_token=147456),
            )
            (directory / "controller_config.json").write_text(json.dumps(controller))
            args = child_arguments(NS(), dict(run_dir=directory), directory, jobs_path)
            parsed = original.parse_args(args + ["--validate-only"])
            prepared = original.preflight(parsed)
            self.assertEqual(prepared[-1]["platform"], "qwen3")
            self.assertFalse(prepared[-1]["migration_execution_enabled"])
            args.remove("--no-migration")
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                original.parse_args(args)

    def test_a100_entry_still_requires_original_checkpoint(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            original.parse_args(["--platform", "a100", "--predictor-sha256", "wrong"])
