"""Check held-out isolation, long-form augmentation, and training-size fairness."""

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from prepare_predictor_long5000 import choose_long5000  # noqa: E402
from run_predictor_large_a100 import validate_shard  # noqa: E402
from run_predictor_long5000_a100 import (  # noqa: E402
    coverage,
    training_view,
    workload_metrics,
)

from tests.bridge_tp import test_predictor_large as large_fixtures  # noqa: E402


class TestPredictorLong5000(unittest.TestCase):
    def test_new_holdouts_exclude_old_trees_and_long_prompts_keep_provenance(self):
        roots = [
            {
                "id": f"{lang}-{split}-{i}",
                "source_tree_id": f"{lang}-{split}-{i}",
                "lang": lang,
                "split": split,
                "messages": [
                    {"role": "user", "content": f"Explain topic {i} in detail."}
                ],
            }
            for lang, counts in (("en", (10000, 1500, 1500)), ("zh", (500, 80, 80)))
            for split, count in zip(("train", "validation", "test"), counts)
            for i in range(count)
        ]
        selected, excluded = choose_long5000(roots)
        again, _ = choose_long5000(list(reversed(roots)))
        self.assertEqual(selected, again)
        self.assertEqual(len(selected), 5000)
        self.assertEqual(len({r["source_tree_id"] for r in selected}), 5000)
        self.assertFalse({r["source_tree_id"] for r in selected} & excluded)
        self.assertEqual(
            Counter(r["split"] for r in selected),
            {"train": 4000, "validation": 500, "test": 500},
        )
        self.assertEqual(
            Counter(r["workload_group"] for r in selected),
            {"natural": 4000, "long_form": 1000},
        )
        original = {r["id"]: r for r in roots}
        for row in selected:
            self.assertEqual(row["split"], original[row["id"]]["split"])
            if row["workload_group"] == "natural":
                self.assertEqual(row["messages"], original[row["id"]]["messages"])
            else:
                self.assertTrue(
                    row["messages"][0]["content"].startswith(
                        original[row["id"]]["messages"][0]["content"]
                    )
                )
                self.assertIn(row["target_words"], (1000, 1800, 2600))
                self.assertIn(
                    "Finish the response naturally", row["messages"][0]["content"]
                )

    def test_nested_training_view_keeps_all_heldouts_and_censored_bounds(self):
        inputs = [
            {"id": f"{split}-{group}-{i}", "split": split, "workload_group": group}
            for split in ("train", "validation", "test")
            for group, count in (("natural", 16), ("long_form", 4))
            for i in range(count)
        ]
        labels = [
            {
                "input_id": r["id"],
                "split": r["split"],
                "natural_finish": r["id"] != "train-long_form-0",
                "output_tokens": 8192 if r["id"] == "train-long_form-0" else 1500,
            }
            for r in inputs
        ]
        data = {
            "hidden": np.ones((60, 4), dtype=np.float16),
            "requests": np.array([r["id"] for r in inputs]),
            "splits": np.array([r["split"] for r in inputs]),
            "phases": np.array(["PREFILL_COMPLETE"] * 60),
            "censored": np.array([not r["natural_finish"] for r in labels]),
            "labels": labels,
            "audit": {"format_version": 1, "hidden_size": 4},
        }
        small, full = training_view(data, inputs, 10), training_view(data, inputs, 20)
        self.assertEqual(len(small["labels"]), 50)
        self.assertEqual(len(full["labels"]), 60)
        self.assertTrue(set(small["requests"]) <= set(full["requests"]))
        for split in ("validation", "test"):
            np.testing.assert_array_equal(
                small["requests"][small["splits"] == split],
                full["requests"][full["splits"] == split],
            )
        c = coverage(labels, inputs)
        self.assertEqual(c["train/long_form"]["censored"], 1)
        self.assertEqual(c["train/long_form"]["observed_output_gt"]["4096"], 1)
        self.assertEqual(c["test/natural"]["requests"], 16)

    def test_resume_rejects_old_output_limit_for_new_protocol(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            run, path, _ = large_fixtures.TestPredictorLarge().make_shard(
                Path(directory), 0, "train"
            )
            with self.assertRaisesRegex(ValueError, "protocol/input"):
                validate_shard(
                    run, path, "test-revision", max_tokens=8192, max_model_len=10240
                )
            meta = json.loads((run / "preflight.json").read_text())
            meta.update(max_tokens=8192, max_model_len=10240)
            (run / "preflight.json").write_text(json.dumps(meta))
            validate_shard(
                run, path, "test-revision", max_tokens=8192, max_model_len=10240
            )


class TestWorkloadMetrics(unittest.TestCase):
    def test_natural_and_long_request_weights_are_separate(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(
                root / "distribution_predictions.npz",
                request_ids=np.array(["natural", "natural", "long", "long"]),
                splits=np.array(["test"] * 4),
                phases=np.array(["PREFILL_COMPLETE", "DECODE"] * 2),
                censored=np.array([False] * 4),
                observed_remaining=np.array([100, 80, 3000, 2980]),
                category_upper_edges=np.array([0, 128, 512, 8192]),
                probabilities=np.array([[0, 1, 0, 0, 0]] * 2 + [[0, 0, 0, 1, 0]] * 2),
            )
            inputs = [
                {"id": "natural", "workload_group": "natural"},
                {"id": "long", "workload_group": "long_form"},
            ]
            report = {
                "temperature": 1.1,
                "selection_validation_request_ids": [],
                "calibration_validation_request_ids": [],
            }
            metrics = workload_metrics(root, inputs, report)
            self.assertEqual(len(metrics), 4)
            natural = metrics["test/natural/PREFILL_COMPLETE"]["horizons"]["512"]
            long = metrics["test/long_form/PREFILL_COMPLETE"]["horizons"]["512"]
            self.assertEqual(natural["requests"], 1)
            self.assertEqual(long["requests"], 1)
            self.assertEqual(natural["observed_exceedance_rate"], 0)
            self.assertEqual(long["observed_exceedance_rate"], 1)


if __name__ == "__main__":
    unittest.main()
