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
    fixed_horizon_result,
    observed_action,
    pair_results,
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

    def test_rejects_predictor_train_tree_in_held_out_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in (
                {"id": "r-train", "split": "train", "source_tree_id": "tree"},
                {"id": "r-test", "split": "test", "source_tree_id": "tree"},
            )))
            with self.assertRaisesRegex(ValueError, "crosses dataset splits"):
                select_inputs(path)

    def test_reports_actual_start_separately_from_configured_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "controller" / "phase9_audit.jsonl"
            audit.parent.mkdir()
            audit.write_text(json.dumps({
                "kind": "manager_m5_predictor_shadow",
                "status": "AVAILABLE", "output_tokens": 207,
                "p_remaining_gt_long_window_runtime_bounds": [0.81, 0.88],
            }) + "\n" + json.dumps({
                "kind": "manager_m1_start_decision",
                "decision": {"action": "START_SHADOW",
                             "source_time_to_guard_s": 90.0,
                             "estimated_preparation_s": 10.0},
                "snapshot": {"generated_tokens": 207, "target_running": 8},
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
            candidate = observed["first_candidate_at_or_after_128"]
            self.assertEqual(candidate["target_running"], 8)
            self.assertEqual(candidate["p_remaining_gt_512_lower"], 0.81)

    def test_random_arrivals_are_seeded_and_shared_by_arms(self) -> None:
        class Tokenizer:
            @staticmethod
            def apply_chat_template(*_args: object, **_kwargs: object) -> str:
                return "prompt"

            @staticmethod
            def encode(value: str) -> list[int]:
                return list(range(len(value)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "requests.jsonl"
            input_path.write_text("".join(json.dumps({
                "id": f"r-{index}", "split": "test",
                "workload_group": "long_form" if index % 4 == 0
                else "natural", "prompt": "hello",
            }) + "\n" for index in range(160)))
            fake = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *_a, **_k: Tokenizer()))
            args = SimpleNamespace(input=input_path, model=root / "model",
                                   seed=713, random_arrivals=True)
            with mock.patch.dict(sys.modules, {"transformers": fake}):
                first = build_setup(args, root / "first")
                second = build_setup(args, root / "second")
            name = first["cases"][3]
            one = json.loads(Path(first["manifests"][name]["path"]).read_text())
            two = json.loads(Path(second["manifests"][name]["path"]).read_text())
            self.assertEqual(one["jobs"], two["jobs"])
            self.assertEqual(one["arrival_process"], "seeded_exponential")
            target = [job["start_after_s"] for job in one["jobs"]
                      if job["pool"] == "target"]
            self.assertEqual(target, sorted(target))
            self.assertNotEqual(round(target[2] - target[1], 3), 0.2)

    def test_fixed_horizon_excludes_censored_and_incomplete_arms(self) -> None:
        arm = {
            "fatal_error": False, "audit_rc": 0,
            "anchor_source_finish_reason": "stop", "background_jobs": 2,
            "background_completed": 2, "background_finish_reasons": {"stop": 2},
            "slo_metrics": {"requests": 3, "completed_requests": 3,
                            "good_output_tokens": 900, "wall_time_s": 75.0},
        }
        fixed_horizon_result(arm, 180.0)
        self.assertEqual(arm["fixed_horizon_goodoutput_tokens_s"], 5.0)
        arm["anchor_source_finish_reason"] = "abort"
        arm["anchor_target_finish_reason"] = "stop"
        fixed_horizon_result(arm, 180.0)
        self.assertTrue(arm["fixed_horizon_eligible"])
        arm["background_finish_reasons"] = {"stop": 1, "length": 1}
        fixed_horizon_result(arm, 180.0)
        self.assertFalse(arm["fixed_horizon_eligible"])
        summary = {"seed": 1, "evaluation_horizon_s": 180.0,
                   "cases": {"case": {"stay": arm}}}
        self.assertIsNone(pair_results(summary)["pairs"]["case"][
            "descriptive_delta_tokens_s"])


if __name__ == "__main__":
    unittest.main()
