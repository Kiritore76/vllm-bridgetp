"""Check revised workload identities, natural-length gates and profile isolation."""

import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
pilot = importlib.import_module("run_predictor_tail_pilot")


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(512))

    def decode(self, ids, **kwargs):
        return f"saved {len(ids)} tokens"


class TestTailWorkload(unittest.TestCase):
    def collect_summary(self, lengths, *, censored=(), code=0):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "pilot"
            pilot.prepare(
                directory,
                Tokenizer(),
                {"max_position_embeddings": 32768},
                recipe_profile="workload6",
            )
            capture = directory / "capture"
            capture.mkdir()
            labels = [
                {
                    "input_id": row["id"],
                    "output_tokens": length,
                    "output_token_ids": [7] * length,
                    "natural_finish": index not in censored,
                    "finish_reason": "length" if index in censored else "stop",
                }
                for index, (row, length) in enumerate(
                    zip(pilot.recipe("workload6"), lengths)
                )
            ]
            (capture / "labels.jsonl").write_bytes(pilot.serialized(labels))
            (capture / "summary.json").write_text("{}", encoding="utf-8")
            (capture / "sample_index.jsonl").write_text("{}\n", encoding="utf-8")
            result = pilot.summarize(
                directory,
                Tokenizer(),
                capture_returncode=code,
                recipe_profile="workload6",
            )
            responses = [
                json.loads(line)
                for line in (directory / "responses.jsonl").read_text().splitlines()
            ]
            return result, responses

    def test_old_recipe_bytes_are_unchanged(self):
        self.assertEqual(
            hashlib.sha256(pilot.serialized(pilot.recipe())).hexdigest(),
            "b18eff0ef44fe9046378f59a366a160214b249a2495b4a500611e2a7b0a39f6f",
        )

    def test_new_tasks_have_independent_sources_and_explicit_work(self):
        rows = pilot.recipe("workload6")
        old_ids = {row["source_tree_id"] for row in pilot.recipe()}
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["source_tree_id"] for row in rows}), 6)
        self.assertFalse(old_ids & {row["source_tree_id"] for row in rows})
        self.assertEqual(sum(row["lang"] == "zh" for row in rows), 3)
        self.assertEqual(
            sum(row["task_kind"] == "compiler_interpreter" for row in rows), 3
        )
        self.assertEqual(
            sum(row["target_output_range"] == "12289-16384" for row in rows), 3
        )
        for row in rows:
            prompt = row["messages"][-1]["content"]
            self.assertEqual(row["split"], "pilot")
            self.assertGreaterEqual(len(row["required_modules"]), 14)
            self.assertGreaterEqual(len(row["acceptance_case_groups"]), 4)
            for item in row["required_modules"] + row["acceptance_case_groups"]:
                self.assertIn(item, prompt)
            self.assertNotIn("550-700", prompt)
            self.assertNotIn("1200至1600", prompt)

    def test_manifest_uses_selected_profile_and_preserves_natural_eos(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = pilot.prepare(
                Path(temporary) / "pilot",
                Tokenizer(),
                {"max_position_embeddings": 32768},
                recipe_profile="workload6",
            )
            self.assertEqual(manifest["requests"], 6)
            self.assertEqual(manifest["recipe_profile"], "workload6")
            self.assertEqual(len(manifest["prompt_budgets"]), 6)
            self.assertEqual(
                manifest["input_sha256"],
                hashlib.sha256(pilot.serialized(pilot.recipe("workload6"))).hexdigest(),
            )
            self.assertFalse(manifest["ignore_eos"])
            self.assertEqual(manifest["min_tokens"], 0)
            self.assertFalse(manifest["formal_holdout_eligible"])

    def test_gate_requires_four_lower_and_two_upper_natural_outputs(self):
        result, responses = self.collect_summary(
            [12289, 13000, 9000, 10000, 5000, 7000]
        )
        self.assertTrue(result["recipe_promising_for_expansion"])
        self.assertEqual(result["planned_requests"], 6)
        self.assertEqual(len(responses), 6)
        result, _ = self.collect_summary([12288, 13000, 9000, 10000, 5000, 7000])
        self.assertFalse(result["recipe_promising_for_expansion"])
        result, _ = self.collect_summary([12289, 13000, 8192, 10000, 5000, 7000])
        self.assertFalse(result["recipe_promising_for_expansion"])

    def test_censored_lengths_do_not_pass_natural_gate(self):
        result, responses = self.collect_summary([18432] * 6, censored=range(6))
        self.assertEqual(result["status"], "COLLECTION_COMPLETE")
        self.assertEqual(result["natural_gt8192"], 0)
        self.assertFalse(result["recipe_promising_for_expansion"])
        self.assertEqual(len(responses), 6)
        self.assertEqual(
            result["censored_observed_lower_bounds_by_band"], {">16384": 6}
        )

    def test_partial_capture_preserves_answers_without_expansion(self):
        result, responses = self.collect_summary([13000, 14000], code=1)
        self.assertEqual(result["status"], "INCOMPLETE_DIAGNOSTIC")
        self.assertEqual(result["completed_requests"], 2)
        self.assertEqual(result["planned_requests"], 6)
        self.assertFalse(result["recipe_promising_for_expansion"])
        self.assertEqual(len(responses), 2)

    def test_unknown_profile_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "pilot"
            with self.assertRaises(ValueError):
                pilot.prepare(
                    directory,
                    Tokenizer(),
                    {"max_position_embeddings": 32768},
                    recipe_profile="typo",
                )
            self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()
