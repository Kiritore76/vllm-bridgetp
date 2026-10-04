"""Checks for the A100 GoodOutput scale-up gates."""

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.bridge_tp.run_goodoutput_gates_a100 import (
    build_late_target_load,
    build_natural_pressure,
    configure_late_command,
    configure_pressure_command,
    pair_report,
    pressure_evidence,
    summarize,
)


class TestNaturalPressure(unittest.TestCase):
    def test_direct_script_entrypoint_resolves_tools_package(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        script = repo / "tools/bridge_tp/run_goodoutput_gates_a100.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=repo,
            capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--expected-revision", completed.stdout)

    def test_builds_natural_eos_five_peer_pressure(self) -> None:
        class Tokenizer:
            @staticmethod
            def apply_chat_template(*_args: object, **_kwargs: object) -> str:
                return "<user><CONTEXT></user><assistant>"

            @staticmethod
            def encode(value: str, **_kwargs: object) -> list[int]:
                return [index + 1 for index, _ in enumerate(value)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            rows = [
                {"id": f"source-{index}",
                 "messages": [{"role": "user", "content": "A real question."}]}
                for index in range(12)
            ]
            input_path.write_text("".join(json.dumps(row) + "\n"
                                          for row in rows))
            setup = {"manifests": {}}
            for name, target_count in (("A_safe_light", 2),
                                       ("B_safe_busy", 47)):
                path = root / f"{name}.json"
                jobs = [
                    {"job_id": f"source_{index:03d}", "pool": "source",
                     "input_id": f"source-{index}"}
                    for index in range(12)
                ] + [
                    {"job_id": f"target_{index:03d}", "pool": "target"}
                    for index in range(target_count)
                ]
                path.write_text(json.dumps({"anchor_input_id": "anchor",
                                            "jobs": jobs}))
                setup["manifests"][name] = {"path": str(path)}
            args = SimpleNamespace(input=input_path, model=root / "model")
            fake = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: Tokenizer()))
            with mock.patch.dict(sys.modules, {"transformers": fake}):
                build_natural_pressure(args, root, setup)
            for name, target_count in (("C_guard_light", 2),
                                       ("D_guard_busy", 47)):
                manifest = json.loads(Path(
                    setup["manifests"][name]["path"]).read_text())
                self.assertEqual(len(manifest["jobs"]), target_count + 5)
                source = [job for job in manifest["jobs"]
                          if job["pool"] == "source"]
                self.assertTrue(all(len(job["request"]["prompt"]) == 3584
                                    for job in source))
                self.assertTrue(all(job["request"]["ignore_eos"] is False
                                    for job in source))
                self.assertTrue(all("start_after_event" not in job
                                    for job in source))
                self.assertEqual(
                    [job["start_after_s"] for job in source],
                    [0.0, 0.6, 1.2, 1.8, 2.4],
                )
                self.assertEqual(manifest["requested_response_words"], 350)

    def test_late_targets_arrive_after_anchor_output_800(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.json"
            base_path.write_text(json.dumps({
                "format_version": 1,
                "jobs": [
                    {"job_id": f"target_{index:03d}", "pool": "target",
                     "start_after_s": 0.5 + index,
                     "request": {"prompt": [1, 2], "max_tokens": 4096},
                     "input_id": f"input-{index}"}
                    for index in range(2)
                ],
            }))
            busy_path = root / "busy.json"
            busy_path.write_text(json.dumps({"jobs": [
                {"job_id": job_id, "pool": "target",
                 "start_after_s": 10.0,
                 "request": {"prompt": [3, 4], "max_tokens": 4096},
                 "input_id": f"input-{job_id}"}
                for job_id in ("target_018", "target_026")
            ]}))
            setup = {"manifests": {
                "A_safe_light": {"path": str(base_path)},
                "B_safe_busy": {"path": str(busy_path)},
            }}
            build_late_target_load(root, setup)
            manifest = json.loads(Path(
                setup["manifests"]["L_late_light"]["path"]
            ).read_text())
            self.assertEqual(len(manifest["jobs"]), 4)
            self.assertEqual(
                [job["start_after_event"] for job in manifest["jobs"][2:]],
                ["ANCHOR_OUTPUT_800", "ANCHOR_OUTPUT_800"],
            )
            self.assertEqual(
                [job["job_id"] for job in manifest["jobs"]],
                [f"target_{index:03d}" for index in range(4)],
            )
            self.assertEqual(
                [job["input_id"] for job in manifest["jobs"][2:]],
                ["input-target_018", "input-target_026"],
            )
            self.assertFalse(manifest.get("late_target_reuses_warm_prompts"))

    def test_guard_requires_safe_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "controller" / "phase9_audit.jsonl"
            audit.parent.mkdir()
            rows = [
                {"kind": "telemetry", "output_tokens": 1,
                 "tp1": {"free_kv_blocks": 700, "block_size": 16,
                         "preemptions_total": 0}},
                {"kind": "telemetry", "output_tokens": 2,
                 "tp1": {"free_kv_blocks": 600, "block_size": 16,
                         "preemptions_total": 0}},
                {"kind": "manager_m1_start_decision", "unix_s": 10.0,
                 "decision": {"action": "START_SHADOW"},
                 "snapshot": {"source_free_kv_tokens": 9600,
                              "generated_tokens": 192}},
            ]
            background_path = root / "background" / "background_summary.json"
            background_path.parent.mkdir()
            background_path.write_text(json.dumps({
                "results": [{"pool": "source", "first_token_unix_s": 9.0,
                             "request_ended_unix_s": 11.0}] * 3}))
            audit.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertTrue(pressure_evidence(root)["valid"])
            self.assertEqual(
                pressure_evidence(root)["m1_start_output_tokens"], 192)
            rows[1]["tp1"]["free_kv_blocks"] = 500
            audit.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertFalse(pressure_evidence(root)["valid"])

    def test_late_cutover_has_positive_controller_window(self) -> None:
        command = ["--m1-min-output-tokens", "96",
                   "--trigger-output-tokens", "64",
                   "--bridge-output-tokens", "96"]
        configure_late_command(command)
        self.assertEqual(command[1::2], ["1024", "1000", "1024", "1120"])
        self.assertGreater(
            int(command[command.index("--cutover-output-tokens") + 1]),
            int(command[command.index("--trigger-output-tokens") + 1]),
        )

    def test_pressure_waits_for_five_prefills_and_near_guard(self) -> None:
        command = ["--minimum-ready-source-jobs", "3",
                   "--m1-min-output-tokens", "96"]
        configure_pressure_command(command)
        self.assertEqual(command[1::2], ["5", "96", "320", "0", "14000"])

    def test_late_repeatability_requires_three_valid_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcomes = {f"L_late_light/r{index:02d}": {}
                        for index in range(1, 4)}

            def fake_pair(_root: Path, key: str, _arms: object,
                          **_kwargs: object) -> dict[str, object]:
                return {"valid": True, "delta_goodoutput_tokens_s":
                        float(int(key[-2:]))}

            with mock.patch(
                "tools.bridge_tp.run_goodoutput_gates_a100.pair_report",
                side_effect=fake_pair,
            ):
                report = summarize(root, outcomes)
            self.assertTrue(report["late_cutover_gate_pass"])
            self.assertTrue(report["late_direction_gate_pass"])
            self.assertEqual(report["late_repeatability"]["deltas"],
                             [1.0, 2.0, 3.0])

            def mixed_pair(_root: Path, key: str, _arms: object,
                           **_kwargs: object) -> dict[str, object]:
                value = float(int(key[-2:]))
                return {"valid": True, "delta_goodoutput_tokens_s":
                        -value if key.endswith("03") else value}

            with mock.patch(
                "tools.bridge_tp.run_goodoutput_gates_a100.pair_report",
                side_effect=mixed_pair,
            ):
                mixed = summarize(root, outcomes)
            self.assertTrue(mixed["late_cutover_gate_pass"])
            self.assertFalse(mixed["late_direction_gate_pass"])

    def test_guard_repeats_keep_slo_failure_as_observed_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcomes = {
                f"{scenario}/r{index:02d}": {}
                for scenario in ("C_guard_light", "D_guard_busy")
                for index in range(1, 4)
            }

            def fake_pair(_root: Path, key: str, _arms: object,
                          **_kwargs: object) -> dict[str, object]:
                return {
                    "valid": False, "mechanism_valid": True,
                    "observed_delta_goodoutput_tokens_s":
                    float(int(key[-2:])),
                }

            with mock.patch(
                "tools.bridge_tp.run_goodoutput_gates_a100.pair_report",
                side_effect=fake_pair,
            ):
                report = summarize(root, outcomes, guard_required_repeats=3)
            self.assertTrue(report["guard_natural_eos_gate_pass"])
            self.assertFalse(report["guard_slo_gate_pass"])
            self.assertEqual(
                report["guard_repeatability"]["C_guard_light"]
                ["observed_deltas"], [1.0, 2.0, 3.0],
            )

    def test_pair_retains_goodoutput_when_slo_attainment_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = "C_guard_light/r01"
            arms = {}
            for arm, goodoutput in (("stay", 100.0), ("migrate", 120.0)):
                pair = root / key
                run = pair / arm / "r01_shadow_only"
                (run / "background").mkdir(parents=True)
                (run / "provenance").mkdir(parents=True)
                (pair / f"{arm}.slo_v6.json").write_text(json.dumps({
                    "computable": True,
                    "reference_applicability":
                    "VERIFIED_GPU_AND_MODEL_CONFIG",
                    "errors": [],
                    "metrics": {"slo_attainment": 0.9,
                                "goodoutput_tokens_s": goodoutput},
                    "request_rows": [{"request_id": "anchor"}],
                }))
                (run / "background" / "background_summary.json").write_text(
                    json.dumps({"jobs": 1, "completed": 1, "failed": 0,
                                "results": [{"finish_reason": "stop"}]}))
                (run / "provenance" /
                 "shadow_online_acceptance.json").write_text(json.dumps({
                     "status": "PASS", "errors": [],
                     "handoff_stall_ms": 200.0,
                 }))
                arms[f"{arm}_runner_rc"] = 0
                arms[f"{arm}_audit_rc"] = 0
            with mock.patch(
                "tools.bridge_tp.run_goodoutput_gates_a100.pressure_evidence",
                return_value={"valid": True},
            ):
                result = pair_report(root, key, arms,
                                     guard_pressure=True, late=False)
            self.assertTrue(result["mechanism_valid"])
            self.assertFalse(result["valid"])
            self.assertEqual(result["observed_delta_goodoutput_tokens_s"],
                             20.0)
            self.assertIsNone(result["delta_goodoutput_tokens_s"])


if __name__ == "__main__":
    unittest.main()
