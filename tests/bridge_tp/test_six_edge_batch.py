# SPDX-License-Identifier: Apache-2.0
"""Batch acceptance boundaries and independent six-card CLI plans; CPU only."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp import six_edge_batch as batch


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / batch.EDGES[0]
        self.scene = self.directory / "trial/scene"
        self.scene.mkdir(parents=True)
        self.registered = dict(requests=2, output_tokens=256, input_sha256="sha")
        self.proof = dict(
            passed=True,
            mechanism_passed=True,
            late_dispatches=[],
            actual_migration_coverage=False,
            actual_committed_transfers=[],
            loaded_drain_coverage=False,
            phases=[],
        )
        self.write("owned_shutdown_receipt.json", dict(all_owned_groups_ended=True))
        self.write("conversion_sequence_acceptance.json", self.proof)
        self.write(
            "conversion_sequence_result.json",
            dict(
                steps=[
                    dict(
                        request=dict(bundle="B", target_mode="TP2"),
                        layout=batch.six.PATH["L1AC"],
                    )
                ]
            ),
        )
        self.write(
            "comparison_metrics.json",
            dict(
                summary=dict(
                    requests=2,
                    output_tokens=256,
                    fixed_arrivals_valid=True,
                    load_coverage_pass=True,
                )
            ),
        )
        self.write(
            "native_empty_receipt.json",
            dict(pool=dict(running=0, waiting=0, requests=[], writer_error=None)),
        )
        self.write(
            "controller/pressure_request_results.json",
            [
                dict(job_id=str(i), source_done=True, source_error=None)
                for i in range(2)
            ],
        )
        self.write("controller/pressure_audit.jsonl", "", raw=True)
        self.write("comparison_policy.json", dict(max_age_s=2), trial=True)
        self.write(
            "result.json",
            dict(
                status="COMPLETED_CONVERSION",
                edge=batch.EDGES[0],
                runs=[dict(input_sha256="sha")],
            ),
            trial=True,
        )
        (self.directory / "runner_exit_code.txt").write_text("0")
        self.addCleanup(patch.stopall)
        patch.object(batch.six, "assess", return_value=self.proof).start()

    def write(self, name, value, *, trial=False, raw=False):
        path = (self.directory / "trial" if trial else self.scene) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if raw else json.dumps(value))

    def assess(self):
        return batch.assess_case(self.directory, batch.EDGES[0], self.registered)

    def test_zero_migration_passes_mechanism_and_remains_false(self):
        value = self.assess()
        self.assertTrue(value["passed"], value)
        self.assertFalse(value["actual_migration_coverage"])
        self.assertFalse(value["migration_coverage_required"])

    def test_runner_failure_retained_but_cleanup_allows_next_case(self):
        (self.directory / "runner_exit_code.txt").write_text("1")
        value = self.assess()
        self.assertFalse(value["passed"])
        self.assertTrue(value["cleanup_confirmed"])

    def test_missing_cleanup_blocks_continuation(self):
        (self.scene / "owned_shutdown_receipt.json").unlink()
        value = self.assess()
        self.assertFalse(value["passed"])
        self.assertFalse(value["cleanup_confirmed"])

    def test_wrong_edge_or_workload_rejected(self):
        for mutation in [dict(edge="L1AC-L0"), dict(runs=[dict(input_sha256="other")])]:
            result = (
                dict(
                    status="COMPLETED_CONVERSION",
                    edge=batch.EDGES[0],
                    runs=[dict(input_sha256="sha")],
                )
                | mutation
            )
            self.write("result.json", result, trial=True)
            self.assertFalse(self.assess()["passed"])

    def test_wrong_bundle_or_final_layout_rejected(self):
        for request, layout in [
            (dict(bundle="A", target_mode="TP2"), batch.six.PATH["L1AC"]),
            (dict(bundle="B", target_mode="TP2"), batch.six.PATH["L3"]),
        ]:
            self.write(
                "conversion_sequence_result.json",
                dict(steps=[dict(request=request, layout=layout)]),
            )
            self.assertFalse(self.assess()["passed"])

    def test_missing_tokens_invalid_arrivals_and_incomplete_load_fail(self):
        for mutation in [
            dict(output_tokens=255),
            dict(fixed_arrivals_valid=False),
            dict(load_coverage_pass=False),
        ]:
            summary = (
                dict(
                    requests=2,
                    output_tokens=256,
                    fixed_arrivals_valid=True,
                    load_coverage_pass=True,
                )
                | mutation
            )
            self.write("comparison_metrics.json", dict(summary=summary))
            self.assertFalse(self.assess()["passed"])

    def test_duplicate_request_or_source_error_rejected(self):
        for rows in [
            [dict(job_id="same", source_done=True, source_error=None)] * 2,
            [
                dict(job_id=str(i), source_done=True, source_error="error")
                for i in range(2)
            ],
        ]:
            self.write("controller/pressure_request_results.json", rows)
            self.assertFalse(self.assess()["passed"])

    def test_native_pending_requests_rejected(self):
        self.write(
            "native_empty_receipt.json",
            dict(pool=dict(running=1, waiting=0, requests=["x"], writer_error=None)),
        )
        self.assertFalse(self.assess()["passed"])

    def test_recovery_cancel_without_clear_evidence_rejected(self):
        event = dict(
            kind="pressure_cancel_requested",
            migration_id="m",
            request_id="r",
            reason="source service and projected KV recovered",
        )
        self.write(
            "controller/pressure_audit.jsonl", json.dumps(event) + "\n", raw=True
        )
        self.assertFalse(self.assess()["passed"])

    def test_changed_offline_proof_rejected(self):
        changed = copy.deepcopy(self.proof) | dict(passed=False)
        self.write("conversion_sequence_acceptance.json", changed)
        self.assertFalse(self.assess()["passed"])

    def test_aggregate_needs_all_four_edges_and_preserves_failures(self):
        for edge in batch.EDGES:
            directory = self.root / edge
            directory.mkdir(exist_ok=True)
            (directory / "case_acceptance.json").write_text(
                json.dumps(dict(edge=edge, passed=True, cleanup_confirmed=True))
            )
        self.assertTrue(batch.aggregate(self.root)["passed"])
        path = self.root / batch.EDGES[-1] / "case_acceptance.json"
        path.unlink()
        self.assertFalse(batch.aggregate(self.root)["passed"])
        path.write_text(
            json.dumps(
                dict(
                    edge=batch.EDGES[-1],
                    passed=False,
                    cleanup_confirmed=True,
                    errors=["observed failure"],
                )
            )
        )
        self.assertEqual(
            batch.aggregate(self.root)["cases"][batch.EDGES[-1]]["errors"],
            ["observed failure"],
        )

    def test_registered_recipe_and_all_four_real_cli_plans(self):
        inputs = batch.read(
            "experiments/platforms/qwen3_pressure_observation/inputs.json"
        )
        registered = batch.plan(inputs)
        self.assertEqual(
            (registered["requests"], registered["output_tokens"]), (156, 68352)
        )
        self.assertEqual(
            set(registered["edges"]), {"L0-L1AC", "L1AC-L0", "L2C-L3", "L3-L2C"}
        )
        for edge in batch.EDGES:
            with self.subTest(edge=edge), contextlib.redirect_stdout(io.StringIO()):
                result = runner.main(
                    [
                        "--cards",
                        "6",
                        "--static-layout",
                        edge.split("-")[0],
                        "--conversion",
                        edge,
                        "--conversion-trigger-s",
                        "60",
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
                        "m",
                        "--output",
                        str(self.root / edge),
                        "--expected-revision",
                        "CPU",
                        "--validate-only",
                    ]
                )
            self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
