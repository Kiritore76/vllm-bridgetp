"""Validate split isolation and context reservations without a GPU."""

import unittest
from collections import Counter, defaultdict

from tools.bridge_tp.build_predictor_formal_corpus import (
    CONTEXT,
    GROUP_COUNTS,
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
