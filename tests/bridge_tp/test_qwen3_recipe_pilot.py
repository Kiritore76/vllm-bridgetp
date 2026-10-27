# SPDX-License-Identifier: Apache-2.0
"""Validate balanced pilot selection, context reservation and exact/censored labels."""

import copy
import json
import unittest
from collections import Counter

from tools.bridge_tp.run_qwen3_recipe_pilot import (
    DATA,
    MANIFEST,
    balanced_batches,
    load_batch,
    read,
    render_budget,
    sha,
    summarize,
)


class Tokenizer:
    def __init__(self):
        self.thinking_arguments = []

    def apply_chat_template(self, messages, **kwargs):
        self.thinking_arguments.append(kwargs["enable_thinking"])
        return "<user>" + messages[0]["content"] + "</user><assistant>"

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids, **kwargs):
        return "decoded response"


class RecipePilotTests(unittest.TestCase):
    def setUp(self):
        with DATA.open(encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream]

    def test_all_four_batches_cover_exactly_the_frozen_candidates(self):
        self.assertEqual(sha(DATA), read(MANIFEST)["input_sha256"])
        batches = balanced_batches(self.rows)
        self.assertEqual(batches, balanced_batches(list(reversed(self.rows))))
        self.assertEqual(
            {r["id"] for b in batches for r in b}, {r["id"] for r in self.rows}
        )
        for batch in batches:
            self.assertEqual(Counter(r["lang"] for r in batch), {"en": 6, "zh": 6})
            self.assertEqual(
                Counter(r["candidate_group"] for r in batch),
                dict(
                    oasst_natural=6, longwriter_raw=3, oasst_augmented=2, controlled=1
                ),
            )
            self.assertTrue(all(r["split"] == "pilot" for r in batch))

    def test_diverse_tasks_are_separate_and_include_six_coding_projects(self):
        batch, source, manifest = load_batch("diverse12", 0)
        self.assertEqual(sha(source), read(manifest)["input_sha256"])
        self.assertEqual(len(batch), 12)
        self.assertEqual(Counter(r["lang"] for r in batch), {"en": 6, "zh": 6})
        self.assertEqual(
            Counter(r["candidate_group"] for r in batch),
            dict(
                coding_long=6,
                technical_tutorial=2,
                system_design=2,
                document_analysis=2,
            ),
        )
        self.assertFalse(
            {r["source_tree_id"] for r in batch}
            & {r["source_tree_id"] for r in self.rows}
        )
        self.assertTrue(all(r["split"] == "pilot" for r in batch))
        rendered, cap = render_budget(batch, Tokenizer())
        self.assertEqual(len(rendered), 12)
        self.assertGreaterEqual(cap, 8192)
        for index in range(4):
            old, _, _ = load_batch("recipe48", index)
            self.assertEqual(old, balanced_batches(self.rows)[index])

    def test_profile_indices_cannot_relabel_or_repeat_the_new_batch(self):
        for profile, index in (("diverse12", 1), ("recipe48", 4), ("unknown", 0)):
            with (
                self.subTest(profile=profile, index=index),
                self.assertRaises(ValueError),
            ):
                load_batch(profile, index)

    def test_duplicate_family_or_nonpilot_rows_rejected(self):
        for key, value in (
            ("source_tree_id", self.rows[0]["source_tree_id"]),
            ("id", self.rows[0]["id"]),
            ("split", "test"),
        ):
            rows = copy.deepcopy(self.rows)
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                balanced_batches(rows)

    def test_budget_reserves_output_with_new_non_thinking_template(self):
        tokenizer = Tokenizer()
        rows, cap = render_budget(balanced_batches(self.rows)[0], tokenizer)
        self.assertEqual(tokenizer.thinking_arguments, [False] * 12)
        self.assertGreaterEqual(cap, 8192)
        self.assertLessEqual(cap, 15872)
        for row in rows:
            self.assertNotIn("messages", row)
            self.assertEqual(row["thinking_mode"], "disabled")
            self.assertLessEqual(row["planned_prompt_tokens"] + cap, 16384)
            self.assertEqual(
                row["tokenizer_budget_status"], "VALIDATED_QWEN3_NON_THINKING"
            )

    def test_infeasible_long_prompt_is_not_truncated_or_silently_dropped(self):
        for count in (9000, 17000):
            rows = copy.deepcopy(self.rows[:1])
            rows[0]["messages"][0]["content"] = "x" * count
            with self.subTest(count=count), self.assertRaises(ValueError):
                render_budget(rows, Tokenizer())
            self.assertEqual(len(rows[0]["messages"][0]["content"]), count)

    def test_censored_and_repetitive_responses_preserved(self):
        rows, _ = render_budget(self.rows[:2], Tokenizer())
        labels = [
            dict(
                input_id=row["id"],
                prompt_tokens=row["planned_prompt_tokens"],
                output_token_ids=[7] * length,
                output_tokens=length,
                natural_finish=natural,
                finish_reason="stop" if natural else "length",
            )
            for row, length, natural in zip(rows, [40, 8192], [True, False])
        ]
        coverage, responses = summarize(rows, labels, Tokenizer())
        self.assertEqual(coverage["natural_eos"], 1)
        self.assertEqual(coverage["censored"], 1)
        self.assertEqual(
            coverage["natural_length_upper_edge_histogram"], {"128": 1, "censored": 1}
        )
        self.assertGreater(responses[0]["repeated_trigram_fraction"], 0.9)
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[1]["output_token_ids"], [7] * 8192)
        self.assertFalse(coverage["training_started"])

    def test_summary_rejects_missing_labels_and_wrong_prompt_budget(self):
        rows, _ = render_budget(self.rows[:1], Tokenizer())
        with self.assertRaises(ValueError):
            summarize(rows, [], Tokenizer())
        label = dict(
            input_id=rows[0]["id"],
            output_token_ids=[1],
            output_tokens=1,
            prompt_tokens=0,
            natural_finish=True,
        )
        with self.assertRaises(ValueError):
            summarize(rows, [label], Tokenizer())


if __name__ == "__main__":
    unittest.main()
