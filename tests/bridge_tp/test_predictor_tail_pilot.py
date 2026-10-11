"""Check tail-pilot budgets, exact/censored labels, and diagnostic recovery."""

import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))

pilot = importlib.import_module("run_predictor_tail_pilot")


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(512))

    def decode(self, ids, **kwargs):
        return f"decoded response with {len(ids)} tokens"


class TestTailPilot(unittest.TestCase):
    def prepare(self, directory, tokenizer=None):
        return pilot.prepare(
            directory, tokenizer or Tokenizer(), {"max_position_embeddings": 32768}
        )

    def write_labels(self, directory, lengths, *, censored=(), complete=True):
        capture = directory / "capture"
        capture.mkdir()
        rows = []
        for number, (request, tokens) in enumerate(zip(pilot.recipe(), lengths)):
            rows.append(
                {
                    "input_id": request["id"],
                    "output_token_ids": [7] * tokens,
                    "output_tokens": tokens,
                    "natural_finish": number not in censored,
                    "finish_reason": "length" if number in censored else "stop",
                }
            )
        (capture / "labels.jsonl").write_bytes(pilot.serialized(rows))
        if complete:
            (capture / "summary.json").write_text("{}", encoding="utf-8")
            (capture / "sample_index.jsonl").write_text("{}\n", encoding="utf-8")
        return capture, rows

    def test_recipe_identity_language_and_targets(self):
        rows = pilot.recipe()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["source_tree_id"] for row in rows}), 12)
        self.assertEqual(sum(row["lang"] == "zh" for row in rows), 6)
        self.assertEqual(
            sum(row["target_output_range"] == "8193-12288" for row in rows), 6
        )
        self.assertTrue(all(row["split"] == "pilot" for row in rows))

    def test_actual_prompt_budget_and_input_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "new"
            manifest = self.prepare(directory)
            self.assertEqual(manifest["max_model_len"], 24576)
            self.assertEqual(manifest["max_output_tokens"], 18432)
            self.assertFalse(manifest["ignore_eos"])
            self.assertFalse(manifest["formal_holdout_eligible"])
            self.assertEqual(
                manifest["input_sha256"],
                hashlib.sha256(
                    (directory / "input_requests.jsonl").read_bytes()
                ).hexdigest(),
            )

    def test_budget_failure_preserves_no_partial_recipe(self):
        class TooLong(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return list(range(6144))

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "new"
            with self.assertRaisesRegex(ValueError, "budget exceeded"):
                self.prepare(directory, TooLong())
            self.assertFalse(directory.exists())

    def test_configuration_and_existing_output_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "new"
            with self.assertRaisesRegex(ValueError, "configuration"):
                pilot.prepare(
                    directory, Tokenizer(), {"max_position_embeddings": 16384}
                )
            directory.mkdir()
            saved = directory / "existing.txt"
            saved.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preserved"):
                self.prepare(directory)
            self.assertEqual(saved.read_text(), "preserve")

    def test_cap_hits_cannot_pass_natural_output_recipe_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_labels(directory, [18432] * 12, censored=range(12))
            result = pilot.summarize(directory, Tokenizer(), capture_returncode=0)
            self.assertEqual(result["status"], "COLLECTION_COMPLETE")
            self.assertEqual(result["natural_gt8192"], 0)
            self.assertFalse(result["recipe_promising_for_expansion"])
            self.assertEqual(
                result["censored_observed_lower_bounds_by_band"], {">16384": 12}
            )

    def test_natural_gate_and_text_preserved_for_every_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_labels(directory, [10000] * 6 + [14000] * 6)
            result = pilot.summarize(directory, Tokenizer(), capture_returncode=0)
            self.assertTrue(result["recipe_promising_for_expansion"])
            responses = [
                json.loads(line)
                for line in (directory / "responses.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(responses), 12)
            self.assertIn("decoded response", responses[0]["output_text"])
            self.assertEqual(len(responses[-1]["output_token_ids"]), 14000)

    def test_early_short_outputs_remain_in_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_labels(directory, [500] * 6 + [14000] * 6)
            result = pilot.summarize(directory, Tokenizer(), capture_returncode=0)
            self.assertEqual(result["completed_requests"], 12)
            self.assertEqual(result["natural_eos_by_output_band"]["<=4096"], 6)
            self.assertEqual(result["natural_target_hits"], {"12289-16384": 6})

    def test_failure_keeps_partial_responses_and_diagnostic_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_labels(directory, [9000, 200], complete=False)
            result = pilot.summarize(directory, Tokenizer(), capture_returncode=1)
            self.assertEqual(result["status"], "INCOMPLETE_DIAGNOSTIC")
            self.assertEqual(result["completed_requests"], 2)
            self.assertFalse(result["recipe_promising_for_expansion"])
            self.assertTrue((directory / "responses.jsonl").is_file())

    def test_missing_or_duplicate_tokens_fail_instead_of_fabricating_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture, rows = self.write_labels(directory, [100])
            rows[0]["output_token_ids"] = [7]
            (capture / "labels.jsonl").write_bytes(pilot.serialized(rows))
            with self.assertRaisesRegex(ValueError, "complete saved token"):
                pilot.summarize(directory, Tokenizer(), capture_returncode=1)
            rows[0]["output_token_ids"] = [7] * 100
            (capture / "labels.jsonl").write_bytes(pilot.serialized(rows * 2))
            with self.assertRaisesRegex(ValueError, "repeated response"):
                pilot.summarize(directory, Tokenizer(), capture_returncode=1)

    def test_capture_reuses_correct_layer_progress_and_single_sequence(self):
        args = SimpleNamespace(
            out_dir=Path("new"),
            model=Path("model"),
            expected_revision="revision",
            expected_gpu_count=4,
        )
        command = pilot.capture_command(args, {"input_sha256": "digest"})
        self.assertEqual(command[command.index("--feature-layer") + 1], "decoder:31")
        self.assertEqual(command[command.index("--interval") + 1], "20")
        self.assertEqual(command[command.index("--temperature") + 1], "0")
        self.assertIn("--save-token-ids", command)
        self.assertNotIn("--ignore-eos", command)


if __name__ == "__main__":
    unittest.main()
