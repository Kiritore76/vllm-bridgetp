# SPDX-License-Identifier: Apache-2.0
"""Frozen public corpus identities and real collector input compatibility."""

import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from prepare_qwen3_formal16k import RECIPE, materialize, prepare  # noqa: E402
from run_predictor_capture import load_requests  # noqa: E402
from train_length_predictor import read_jsonl  # noqa: E402


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return messages[0]["content"]

    def encode(self, prompt, **kwargs):
        return list(range(len(prompt)))


class Qwen3Formal16KTests(unittest.TestCase):
    def test_frozen_sources_splits_and_pilot_exclusion(self):
        rows = read_jsonl(RECIPE)
        manifest = json.loads(RECIPE.with_suffix(".manifest.json").read_text())
        self.assertEqual(
            hashlib.sha256(RECIPE.read_bytes()).hexdigest(), manifest["sha256"]
        )
        self.assertEqual(len(rows), 2000)
        self.assertEqual(
            Counter(row["split"] for row in rows),
            dict(train=1600, validation=200, test=200),
        )
        self.assertEqual(len({row["source_tree_id"] for row in rows}), 2000)
        self.assertTrue(all(row["local_output_length_label"] is None for row in rows))
        self.assertTrue(
            all(set(row["messages"][0]) == {"role", "content"} for row in rows)
        )
        formal = {row["source_tree_id"] for row in rows}
        for name in ("qwen3_recipe_pilot48_20261027", "qwen3_diverse_long12_20261027"):
            pilot = read_jsonl(RECIPE.with_name(name + ".jsonl"))
            self.assertFalse(formal & {row["source_tree_id"] for row in pilot})

    def row(self, count, index=0):
        return dict(
            id=f"r{index}",
            source_tree_id=f"t{index}",
            split="train",
            messages=[dict(role="user", content="a" * count)],
            source="test",
            lang="en",
            thinking_mode="disabled",
        )

    def test_budgets_preserve_prompt_without_padding(self):
        for count, cap in (
            (1, 15872),
            (512, 15872),
            (513, 14336),
            (2049, 12288),
            (4097, 8192),
            (8192, 8192),
        ):
            row = materialize(self.row(count), Tokenizer())
            self.assertEqual(row["max_tokens"], cap)
            self.assertEqual(row["prompt"], "a" * count)
            self.assertLessEqual(row["planned_prompt_tokens"] + cap, 16384)
            self.assertNotIn("messages", row)
        self.assertIsNone(materialize(self.row(8193), Tokenizer()))

    def test_manifest_shards_load_in_actual_capture_reader(self):
        rows = [self.row(100, 0), self.row(700, 1), self.row(8193, 2)]
        # U+2028 inside a string must not become a physical JSONL record break.
        rows[0]["messages"][0]["content"] += "\u2028text"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "inputs"
            result = prepare(rows, Tokenizer(), output, "frozen-sha")
            self.assertEqual(result["rows"], 2)
            self.assertEqual(len(result["excluded_before_generation"]), 1)
            for shard in result["shards"]:
                source = output / shard["input"]
                self.assertEqual(len(load_requests(source, None)), shard["rows"])
                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).hexdigest(), shard["sha256"]
                )


if __name__ == "__main__":
    unittest.main()
