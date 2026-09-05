# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


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


class TestManifestPressure(unittest.TestCase):
    def test_v2_manifest_exceeds_capacity_floor(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "experiments"
            / "phase9"
            / "manifests"
            / "cap0_calibration_v2.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        count = MODULE.validate_manifest_pressure(
            manifest,
            tp1_blocks=1968,
            anchor_max_tokens=8000,
            max_model_len=8192,
        )
        self.assertEqual(count, 5)

    def test_rejects_old_light_manifest(self) -> None:
        jobs = [
            {
                "job_id": "target_000",
                "pool": "target",
                "start_after_s": 0.0,
                "request": {"max_tokens": 512},
            }
        ]
        jobs.extend(
            {
                "job_id": f"source_{index:03d}",
                "pool": "source",
                "start_after_s": 2.0 + index * 0.2,
                "request": {"max_tokens": 1024},
            }
            for index in range(4)
        )
        with self.assertRaisesRegex(ValueError, "110% capacity floor"):
            MODULE.validate_manifest_pressure(
                {"format_version": 1, "jobs": jobs},
                tp1_blocks=1968,
                anchor_max_tokens=8000,
                max_model_len=8192,
            )


class TestSmokeContract(unittest.TestCase):
    def test_formal_requires_exact_passing_smoke_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            manifest = root / "manifest.json"
            survival = root / "survival.json"
            manifest.write_text("{}", encoding="utf-8")
            survival.write_text("{}", encoding="utf-8")
            acceptance = root / "smoke_acceptance.json"
            args = SimpleNamespace(
                model_path=model,
                manifest=manifest,
                survival_table=survival,
                dtype="bfloat16",
                max_model_len=8192,
                gpu_memory_utilization=0.88,
                tp1_gpu="0",
                tp4_gpus="1,2,3,4",
                tp1_blocks=1968,
                tp4_blocks=35739,
                anchor_max_tokens=8000,
                minimum_peak_kv_usage_frac=0.90,
                allow_censored=False,
                coverage_slack_s=1.0,
                smoke_acceptance=acceptance,
            )
            contract = MODULE.experiment_contract(args, "revision")
            acceptance.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "contract": contract,
                        "run": {
                            "status": "PASS",
                            "metrics": {
                                "peak_tp1_kv_usage_frac": 0.95,
                                "preemption_delta": 1,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            MODULE.validate_smoke_acceptance(args, "revision")
            args.anchor_max_tokens = 7999
            with self.assertRaisesRegex(RuntimeError, "smoke contract differs"):
                MODULE.validate_smoke_acceptance(args, "revision")


class TestSourceSelectionContract(unittest.TestCase):
    def test_nested_controller_directory_drives_anchor_prefix(self) -> None:
        args = SimpleNamespace(
            tp1_gpu="0",
            snapshot_port=29800,
            delta_port=29900,
        )
        environment = MODULE.source_environment(
            args,
            "cap0-calibration-smoke-example",
            Path("batch") / "smoke" / "controller",
        )
        self.assertEqual(
            environment["BRIDGETP_STREAM_SOURCE_REQUEST_ID_PREFIX"],
            "bridgetp-phase9-controller",
        )


class TestCalibrationAcceptance(unittest.TestCase):
    @staticmethod
    def write_run(
        controller: Path,
        background: Path,
        *,
        peak_usage: float = 0.95,
        preemptions: int = 1,
        background_end: float = 20.0,
        telemetry_end: float = 20.0,
        transition: str | None = None,
    ) -> None:
        (background / "background_summary.json").write_text(
            json.dumps(
                {
                    "jobs": 5,
                    "completed": 5,
                    "failed": 0,
                    "start_unix_s": 10.0,
                    "end_unix_s": background_end,
                }
            ),
            encoding="utf-8",
        )
        records = [
            {
                "kind": "telemetry",
                "tp1": {
                    "preemptions_total": 0,
                    "kv_usage_frac": 0.1,
                    "sampled_unix_s": 9.9,
                },
                "capacity_signal": {
                    "free_kv_tokens": 100,
                    "decline_rate_tokens_s": 2.0,
                },
            },
            {"kind": "decision"},
            {
                "kind": "telemetry",
                "tp1": {
                    "preemptions_total": preemptions,
                    "kv_usage_frac": peak_usage,
                    "sampled_unix_s": telemetry_end,
                },
                "capacity_signal": {
                    "free_kv_tokens": 10,
                    "decline_rate_tokens_s": 20.0,
                },
            },
        ]
        if transition is not None:
            records.append({"kind": "transition", "to": transition})
        records.append(
            {
                "kind": "run_end",
                "final_state": "COMPLETED_ON_TP1",
                "unix_s": telemetry_end + 0.1,
            }
        )
        (controller / "phase9_audit.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records),
            encoding="utf-8",
        )

    def test_accepts_complete_pressured_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "controller"
            background = root / "background"
            controller.mkdir()
            background.mkdir()
            self.write_run(controller, background)
            result = MODULE.accept_calibration(
                controller,
                background,
                expected_jobs=5,
                minimum_peak_kv_usage_frac=0.90,
                require_preemption=True,
                coverage_slack_s=1.0,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["migration_transitions"], 0)

    def test_rejects_migration_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "controller"
            background = root / "background"
            controller.mkdir()
            background.mkdir()
            self.write_run(controller, background, transition="SHADOW")
            result = MODULE.accept_calibration(
                controller,
                background,
                expected_jobs=5,
                minimum_peak_kv_usage_frac=0.90,
                require_preemption=True,
                coverage_slack_s=1.0,
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["migration_transitions"], 1)

    def test_rejects_weak_pressure_and_short_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "controller"
            background = root / "background"
            controller.mkdir()
            background.mkdir()
            self.write_run(
                controller,
                background,
                peak_usage=0.0244,
                preemptions=0,
                background_end=30.0,
                telemetry_end=12.0,
            )
            result = MODULE.accept_calibration(
                controller,
                background,
                expected_jobs=5,
                minimum_peak_kv_usage_frac=0.90,
                require_preemption=True,
                coverage_slack_s=1.0,
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertIn(
                "no TP1 preemption observed under calibration pressure",
                result["errors"],
            )
            self.assertIn(
                "telemetry did not cover the background workload end",
                result["errors"],
            )


if __name__ == "__main__":
    unittest.main()
