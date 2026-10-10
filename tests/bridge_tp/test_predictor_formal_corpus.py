"""Validate split isolation and context reservations without a GPU."""

import hashlib
import unittest
from collections import Counter, defaultdict

from tools.bridge_tp.build_predictor_formal_corpus import (
    CONTEXT,
    GROUP_COUNTS,
    LONG_GROUP_COUNTS,
    LONG_TRAINING_EDGES,
    long300_recipe,
    materialize,
    recipe,
    serialized,
)


class CharacterTokenizer:
    """Use character tokens to exercise reservation logic deterministically."""

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids):
        return "".join(map(chr, ids))

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "</user><assistant>"


class FormalCorpusTests(unittest.TestCase):
    def test_merged_bucket_plan_preserves_special_and_range_boundaries(self):
        edges = LONG_TRAINING_EDGES
        self.assertEqual(len(edges) + 1, 84)
        self.assertEqual(edges[:4], [0, 8, 16, 32])
        self.assertEqual(edges[-1], 16384)
        self.assertTrue(all(a < b for a, b in zip(edges, edges[1:])))
        for lower, upper, width in (
            (32, 512, 32),
            (512, 2048, 64),
            (2048, 4096, 128),
            (4096, 8192, 256),
            (8192, 16384, 1024),
        ):
            self.assertEqual(
                [e for e in edges if lower < e <= upper],
                list(range(lower + width, upper + 1, width)),
            )

    def test_long300_independence_splits_and_tail_budget(self):
        rows = long300_recipe()
        self.assertEqual(len(rows), 300)
        self.assertEqual(serialized(rows), serialized(long300_recipe()))
        self.assertEqual(len({r["id"] for r in rows}), 300)
        old_trees = {r["source_tree_id"] for r in recipe()}
        self.assertFalse(old_trees & {r["source_tree_id"] for r in rows})
        self.assertFalse({r["topic"] for r in recipe()} & {r["topic"] for r in rows})
        self.assertEqual(Counter(r["requested_group"] for r in rows), LONG_GROUP_COUNTS)
        self.assertEqual(
            Counter(r["split"] for r in rows),
            {"train": 210, "validation": 45, "test": 45},
        )
        self.assertEqual(Counter(r["lang"] for r in rows), {"en": 150, "zh": 150})
        self.assertEqual(
            Counter(r["context_group"] for r in rows), {"short": 260, "ctx2k": 40}
        )
        splits = defaultdict(set)
        for row in rows:
            splits[row["source_tree_id"]].add(row["split"])
            if row["requested_group"] != "upper_mid":
                self.assertEqual(row["context_group"], "short")
        self.assertEqual(len(splits), 100)
        self.assertTrue(all(len(values) == 1 for values in splits.values()))

    def test_long300_materialization_and_legacy_recipe_stability(self):
        self.assertEqual(
            hashlib.sha256(serialized(recipe())).hexdigest(),
            "2fb6094e74028ca0ab81381fd44ea45aeb2c8a2db4fc52b344a03d4046a6ab28",
        )
        for context in ("short", "ctx2k"):
            row = next(r for r in long300_recipe() if r["context_group"] == context)
            actual = materialize(row, CharacterTokenizer())
            self.assertIn(row["instruction"], actual["prompt"])
            self.assertLessEqual(
                actual["planned_prompt_tokens"] + actual["planned_output_cap"], 16384
            )

    def test_preregistered_counts_and_whole_tree_splits(self):
        rows = recipe()
        self.assertEqual(serialized(rows), serialized(recipe()))
        self.assertEqual(len({r["id"] for r in rows}), 600)
        self.assertEqual(Counter(r["requested_group"] for r in rows), GROUP_COUNTS)
        splits = defaultdict(set)
        for row in rows:
            splits[row["source_tree_id"]].add(row["split"])
        self.assertEqual(len(splits), 100)
        self.assertTrue(all(len(values) == 1 for values in splits.values()))
        self.assertEqual(
            Counter(r["split"] for r in rows),
            {"train": 420, "validation": 90, "test": 90},
        )
        self.assertTrue(
            all(
                r["context_group"] == "short"
                for r in rows
                if r["requested_group"] == "tail12288"
            )
        )

    def test_context_reserves_output_and_keeps_instruction(self):
        for context in ("ctx2k", "ctx7k"):
            row = next(r for r in recipe() if r["context_group"] == context)
            actual = materialize(row, CharacterTokenizer())
            target, cap = CONTEXT[context]
            self.assertLessEqual(abs(actual["planned_prompt_tokens"] - target), 16)
            self.assertLessEqual(actual["planned_prompt_tokens"] + cap, 16384)
            self.assertIn(row["instruction"], actual["prompt"])


if __name__ == "__main__":
    unittest.main()
