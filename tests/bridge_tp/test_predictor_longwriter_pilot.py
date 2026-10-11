"""Check source provenance and censored-length handling for public prompts."""

import copy
import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
pilot = importlib.import_module("run_predictor_tail_pilot")


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1] * 512

    def decode(self, ids, **kwargs):
        return f"actual local response, {len(ids)} tokens"


class TestLongWriterPilot(unittest.TestCase):
    def test_original_prompts_only_and_reference_lengths_are_metadata(self):
        rows = pilot.recipe("longwriter3")
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["lang"] for row in rows}, {"en", "zh"})
        for row in rows:
            self.assertEqual([m["role"] for m in row["messages"]], ["user"])
            self.assertEqual(row["split"], "pilot")
            self.assertEqual(
                hashlib.sha256(row["messages"][0]["content"].encode()).hexdigest(),
                row["source_prompt_sha256"],
            )
            self.assertNotIn("output_tokens", row)
            self.assertNotIn("output_token_ids", row)
            self.assertGreater(row["reference_output_tokens_qwen_tokenizer"], 8192)

    def test_changed_prompt_or_assistant_contamination_is_rejected(self):
        original = pilot.recipe("longwriter3")
        for corruption in ["prompt", "assistant", "duplicate_source"]:
            rows = copy.deepcopy(original)
            if corruption == "prompt":
                rows[0]["messages"][0]["content"] += "changed"
            elif corruption == "assistant":
                rows[0]["messages"].append(
                    {"role": "assistant", "content": "reference"}
                )
            else:
                rows[1]["source_tree_id"] = rows[0]["source_tree_id"]
            with (
                patch.object(pilot.Path, "read_text", return_value=json.dumps(rows)),
                self.assertRaises(ValueError),
            ):
                pilot.recipe("longwriter3")

    def collect(self, lengths, *, censored=(), code=0):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "pilot"
            manifest = pilot.prepare(
                directory,
                Tokenizer(),
                {"max_position_embeddings": 32768},
                recipe_profile="longwriter3",
            )
            self.assertEqual(manifest["requests"], 3)
            self.assertFalse(manifest["ignore_eos"])
            self.assertEqual(manifest["min_tokens"], 0)
            capture = directory / "capture"
            capture.mkdir()
            labels = [
                {
                    "input_id": row["id"],
                    "output_tokens": count,
                    "output_token_ids": [7] * count,
                    "natural_finish": number not in censored,
                    "finish_reason": "length" if number in censored else "stop",
                }
                for number, (row, count) in enumerate(
                    zip(pilot.recipe("longwriter3"), lengths)
                )
            ]
            (capture / "labels.jsonl").write_bytes(pilot.serialized(labels))
            (capture / "summary.json").write_text("{}", encoding="utf-8")
            (capture / "sample_index.jsonl").write_text("{}\n", encoding="utf-8")
            result = pilot.summarize(
                directory,
                Tokenizer(),
                capture_returncode=code,
                recipe_profile="longwriter3",
            )
            responses = [
                json.loads(line)
                for line in (directory / "responses.jsonl").read_text().splitlines()
            ]
            return result, responses

    def test_short_local_eos_is_not_replaced_by_long_public_reference(self):
        result, responses = self.collect([100, 200, 300])
        self.assertEqual(result["status"], "COLLECTION_COMPLETE")
        self.assertEqual([r["output_tokens"] for r in responses], [100, 200, 300])
        self.assertEqual(result["natural_gt8192"], 0)
        self.assertFalse(result["recipe_promising_for_expansion"])

    def test_three_request_gate_requires_two_lower_and_one_upper(self):
        result, _ = self.collect([9000, 13000, 4000])
        self.assertTrue(result["recipe_promising_for_expansion"])
        result, _ = self.collect([9000, 12288, 4000])
        self.assertFalse(result["recipe_promising_for_expansion"])

    def test_censored_upper_response_does_not_pass_natural_gate(self):
        result, _ = self.collect([9000, 18432, 4000], censored=[1])
        self.assertFalse(result["recipe_promising_for_expansion"])
        self.assertEqual(result["natural_gt12288"], 0)
        self.assertEqual(
            result["censored_observed_lower_bounds_by_band"], {">16384": 1}
        )

    def test_partial_failure_saves_actual_response(self):
        result, responses = self.collect([12300], code=1)
        self.assertEqual(result["status"], "INCOMPLETE_DIAGNOSTIC")
        self.assertEqual(result["planned_requests"], 3)
        self.assertEqual(result["completed_requests"], 1)
        self.assertEqual(len(responses), 1)
        self.assertFalse(result["recipe_promising_for_expansion"])


if __name__ == "__main__":
    unittest.main()
