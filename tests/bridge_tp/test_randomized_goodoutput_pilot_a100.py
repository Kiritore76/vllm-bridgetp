"""Focused checks for the randomized GoodOutput pilot collector."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.bridge_tp.run_randomized_goodoutput_pilot_a100 import (
    ACTIONS,
    CASES,
    action_order,
    build_setup,
    configure_action,
    observed_action,
    select_inputs,
)


class TestRandomizedPilot(unittest.TestCase):
    def test_selects_unique_held_out_requests_and_builds_six_cases(self) -> None:
        class Tokenizer:
            @staticmethod
            def apply_chat_template(*_args: object,
                                    **_kwargs: object) -> str:
                return "<user>hello</user><assistant>"

            @staticmethod
            def encode(value: str) -> list[int]:
                return [index + 1 for index, _ in enumerate(value)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "requests.jsonl"
            rows = [
                {"id": f"input-{index:03d}", "split": "test",
                 "workload_group": "long_form" if index % 4 == 0
                 else "natural",
                 "messages": [{"role": "user", "content": "hello"}]}
                for index in range(160)
            ]
            input_path.write_text("".join(json.dumps(row) + "\n"
                                          for row in rows))
            selected = select_inputs(input_path)
            self.assertEqual(len(selected), 6 + sum(
                source + target for _, source, target, *_ in CASES))
            self.assertEqual(len({row["id"] for row in selected}),
                             len(selected))
            args = SimpleNamespace(input=input_path, model=root / "model")
            fake = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: Tokenizer()))
            with mock.patch.dict(sys.modules, {"transformers": fake}):
                setup = build_setup(args, root)
            self.assertEqual(len(setup["cases"]), 6)
            used_ids = [value["input_id"]
                        for value in setup["anchors"].values()]
            for name in setup["cases"]:
                manifest = json.loads(Path(
                    setup["manifests"][name]["path"]).read_text())
                self.assertTrue(manifest["natural_eos_required"])
                self.assertEqual(len(manifest["jobs"]),
                                 manifest["source_count"]
                                 + manifest["target_count"])
                for job in manifest["jobs"]:
                    used_ids.append(job["input_id"])
                    self.assertFalse(job["request"]["ignore_eos"])
                    if job["pool"] == "source":
                        self.assertEqual(job["start_after_event"],
                                         "ANCHOR_FIRST_OUTPUT")
            self.assertEqual(len(used_ids), len(set(used_ids)))

    def test_actions_keep_m2_policy_and_randomize_order(self) -> None:
        base = [
            "--manager-m2-force-initial-high",
            "--manager-m2-expected-profile", "HIGH",
            "--manager-m2-min-history-byte-frac", "0.9",
            "--m1-min-output-tokens", "96",
            "--trigger-output-tokens", "64",
            "--bridge-output-tokens", "96",
        ]
        for action in ACTIONS:
            command = base.copy()
            configure_action(command, action)
            self.assertNotIn("--manager-m2-force-initial-high", command)
            self.assertNotIn("--manager-m2-expected-profile", command)
            self.assertNotIn("--manager-m2-min-history-byte-frac", command)
            self.assertEqual(command[command.index(
                "--minimum-window-samples") + 1], "0")
            expected = "1024" if action == "late1024" else "128"
            self.assertEqual(command[command.index(
                "--m1-min-output-tokens") + 1], expected)
        self.assertEqual(action_order("p00_source1_target2"),
                         action_order("p00_source1_target2"))
        self.assertEqual(set(action_order("p00_source1_target2")),
                         set(ACTIONS))

    def test_reports_actual_start_separately_from_configured_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "controller" / "phase9_audit.jsonl"
            audit.parent.mkdir()
            audit.write_text(json.dumps({
                "kind": "manager_m1_start_decision",
                "decision": {"action": "START_SHADOW"},
                "snapshot": {"generated_tokens": 207},
            }) + "\n")
            acceptance = (root / "provenance" /
                          "shadow_online_acceptance.json")
            acceptance.parent.mkdir()
            acceptance.write_text(json.dumps({
                "status": "PASS", "errors": [],
                "earliest_ready_cutover_output_tokens": 302,
                "handoff_stall_ms": 240.0,
            }))
            observed = observed_action(root)
            self.assertEqual(observed["actual_m1_start_output_tokens"],
                             [207])
            self.assertEqual(observed["actual_cutover_output_tokens"], 302)


if __name__ == "__main__":
    unittest.main()
