"""Check public task provenance, real budgets, and partial-output diagnostics."""

import copy
import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
pilot = importlib.import_module("run_predictor_targeted4_pilot")
quality = importlib.import_module("predictor_targeted_quality")


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert tokenize is False
        return messages[0]["content"]

    def encode(self, prompt):
        return [1] * 12783

    def decode(self, ids, **kwargs):
        return "actual short local response"


class TestTargetedPilot(unittest.TestCase):
    def test_provenance_references_not_assistant_inputs(self):
        rows, refs = pilot.recipe()
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [r["max_output_tokens"] for r in rows], [11000] * 2 + [18432] * 2
        )
        self.assertEqual(len(refs), 4)
        for row in rows:
            self.assertFalse(row["prompt_adapted"])
            self.assertEqual([m["role"] for m in row["messages"]], ["user"])
            self.assertNotIn("reference_output", row)
            self.assertEqual(pilot.digest(refs[row["id"]]), row["quality_spec_sha256"])
        self.assertEqual(len(refs[rows[1]["id"]]["reference_output"].split("\n")), 479)

    def test_changed_prompt_rejected(self):
        rows, refs = pilot.recipe()
        corrupt = copy.deepcopy(rows)
        corrupt[0]["messages"][0]["content"] += "changed"
        with (
            patch.object(
                pilot.Path,
                "read_text",
                side_effect=[json.dumps(corrupt), json.dumps(refs)],
            ),
            self.assertRaisesRegex(ValueError, "provenance"),
        ):
            pilot.recipe()

    def test_string_encoding_budget_checks_all_before_write(self):
        class ActualTokenizer(Tokenizer):
            def encode(self, prompt):
                return [1] * (12783 if "[TASK]" in prompt else 805)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "pilot"
            manifest = pilot.prepare(
                directory, ActualTokenizer(), {"max_position_embeddings": 32768}
            )
            self.assertEqual(manifest["prompt_budgets"][0]["prompt_tokens"], 12783)
            self.assertEqual(manifest["prompt_budgets"][3]["prompt_tokens"], 805)
            self.assertFalse(manifest["ignore_eos"])
            self.assertEqual(manifest["min_tokens"], 0)
            for group in manifest["groups"]:
                data = (
                    directory / group["group"] / "input_requests.jsonl"
                ).read_bytes()
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), group["input_sha256"]
                )
                args = SimpleNamespace(
                    out_dir=directory,
                    model=Path("model"),
                    expected_revision="head",
                    expected_gpu_count=4,
                )
                command = pilot.command(args, group)
                self.assertEqual(
                    command[command.index("--max-tokens") + 1],
                    str(group["max_output_tokens"]),
                )
                self.assertNotIn("--ignore-eos", command)

    def test_overbudget_does_not_create_partial_inputs(self):
        # The encoding has 12783 token IDs; counting a two-key BatchEncoding
        # would falsely admit this long prompt with the 18432-token cap.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "pilot"
            with self.assertRaisesRegex(ValueError, "budget exceeded"):
                pilot.prepare(
                    directory, Tokenizer(), {"max_position_embeddings": 32768}
                )
            self.assertFalse(directory.exists())

    def test_partial_failure_preserves_response_and_censoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture = directory / "longproc/capture"
            capture.mkdir(parents=True)
            rows, _ = pilot.recipe()
            label = {
                "input_id": rows[0]["id"],
                "output_tokens": 9000,
                "output_token_ids": [7] * 9000,
                "natural_finish": False,
                "finish_reason": "length",
            }
            (capture / "labels.jsonl").write_bytes(pilot.serialized([label]))
            result = pilot.summarize(directory, Tokenizer(), {"longproc": 1})
            self.assertEqual(result["status"], "INCOMPLETE_DIAGNOSTIC")
            self.assertEqual(result["completed_requests"], 1)
            self.assertEqual(result["natural_gt8192"], 0)
            self.assertEqual(result["censored_requests"], 1)
            saved = json.loads((directory / "responses.jsonl").read_text())
            self.assertEqual(saved["output_text"], "actual short local response")
            self.assertEqual(saved["output_tokens"], 9000)

    def test_complete_two_groups_preserve_natural_and_cap_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows, _ = pilot.recipe()
            for index, (name, _) in enumerate(pilot.GROUPS):
                capture = directory / name / "capture"
                capture.mkdir(parents=True)
                labels = []
                for number, row in enumerate(rows[index * 2 : index * 2 + 2]):
                    count = 18432 if index == 1 and number == 1 else 9000
                    natural = not (index == 1 and number == 1)
                    labels.append(
                        {
                            "input_id": row["id"],
                            "output_tokens": count,
                            "output_token_ids": [7] * count,
                            "natural_finish": natural,
                            "finish_reason": "stop" if natural else "length",
                        }
                    )
                (capture / "labels.jsonl").write_bytes(pilot.serialized(labels))
                (capture / "summary.json").write_text("{}", encoding="utf-8")
                (capture / "sample_index.jsonl").write_text("{}\n", encoding="utf-8")
            result = pilot.summarize(
                directory, Tokenizer(), {"longproc": 0, "longgenbench": 0}
            )
            self.assertEqual(result["status"], "COLLECTION_COMPLETE")
            self.assertEqual(result["natural_gt8192"], 3)
            self.assertEqual(result["natural_gt12288"], 0)
            self.assertEqual(result["censored_requests"], 1)
            self.assertFalse(result["training_started"])


class TestCompletionDiagnostics(unittest.TestCase):
    def test_tsv_omissions_and_duplicates_are_not_hidden_by_count(self):
        spec = {"reference_output": "name\tage\nAlice\t10\nBob\t20\n"}
        result = quality.evaluate(
            "html_to_tsv", "```tsv\nname\tage\nAlice\t10\nAlice\t10\n```", spec, []
        )
        self.assertEqual(result["observed_rows"], 2)
        self.assertEqual(result["exact_row_recall"], 0.5)
        self.assertEqual(result["missing_first_cells"], ["Bob"])
        self.assertFalse(result["complete_exact_rows"])
        full = quality.evaluate("html_to_tsv", spec["reference_output"], spec, [])
        self.assertTrue(full["complete_exact_rows"])

    def test_route_wrong_order_and_missing_close(self):
        spec = {"reference_output": "step A\nstep B"}
        wrong = quality.evaluate(
            "path_traversal", "<Route>step B\nstep A</Route>", spec, []
        )
        self.assertEqual(wrong["exact_steps_matched_any_order"], 2)
        self.assertEqual(wrong["correct_prefix_steps"], 0)
        self.assertFalse(wrong["complete_exact_route"])
        missing = quality.evaluate("path_traversal", "<Route>step A\nstep B", spec, [])
        self.assertEqual(missing["correct_prefix_fraction"], 1.0)
        self.assertFalse(missing["complete_exact_route"])

    def spec(self):
        return {
            "type": "Week",
            "prefix": "#*# Week 1:",
            "number": 3,
            "minimum_words_per_entry": 2,
            "checks_once": {"2": "birthday"},
            "checks_range": {},
            "checks_periodic": {},
        }

    def test_prompt_heading_continuation_and_per_entry_words(self):
        text = (
            "First body.\n#*# Week 2:\nbirthday party\n"
            "#*# Week 3:\nThird body\n*** finished ***"
        )
        result = quality.evaluate("diary", text, self.spec(), [])
        self.assertTrue(result["first_heading_inferred_from_prompt"])
        self.assertTrue(result["complete_entry_number_sequence"])
        self.assertEqual(result["word_counts"], [2, 2, 2])
        self.assertEqual(result["entries_meeting_word_minimum"], 3)
        self.assertTrue(
            result["literal_constraint_hints_not_semantic_scores"][0][
                "literal_phrase_found"
            ]
        )

    def test_duplicate_week_and_range_summary_do_not_count_as_completion(self):
        text = (
            "#*# Week 1:\nSame body\n#*# Week 1:\nSame body\n"
            "#*# Week 2-3:\nSummary\n*** finished"
        )
        result = quality.evaluate("diary", text, self.spec(), [])
        self.assertEqual(result["missing_entries"], [3])
        self.assertEqual(result["duplicate_entries"], [1])
        self.assertEqual(result["identical_nonempty_entry_bodies"], 1)
        self.assertFalse(result["complete_entry_number_sequence"])

    def test_token_loop_diagnostic(self):
        result = quality.repetition([1, 2] * 64)
        self.assertEqual(result["full_32_token_blocks"], 4)
        self.assertEqual(result["maximum_32_token_block_occurrences"], 4)
        self.assertEqual(result["duplicate_32_token_block_fraction"], 0.75)


if __name__ == "__main__":
    unittest.main()
