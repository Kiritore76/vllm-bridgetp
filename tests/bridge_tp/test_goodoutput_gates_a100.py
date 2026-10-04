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
    build_natural_pressure,
    configure_late_command,
    configure_pressure_command,
    pressure_evidence,
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
                self.assertTrue(all(job["start_after_event"]
                                    == "ANCHOR_FIRST_OUTPUT" for job in source))

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

    def test_pressure_starts_only_after_anchor_and_near_guard(self) -> None:
        command = ["--minimum-ready-source-jobs", "3",
                   "--m1-min-output-tokens", "96",
                   "--cutover-output-tokens", "160"]
        configure_pressure_command(command)
        self.assertEqual(command[1::2], ["0", "192", "320", "0", "14000"])


if __name__ == "__main__":
    unittest.main()
