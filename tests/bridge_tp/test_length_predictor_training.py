"""CPU checks for request-level split hygiene in predictor training."""

import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from run_predictor_capture import audit_capture  # noqa: E402
from train_length_predictor import load_examples, request_balanced_mae  # noqa: E402


class TestLengthPredictorTraining(unittest.TestCase):
    def test_loads_exact_rows_and_rejects_tree_split_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            features = run_dir / "features"
            features.mkdir()
            with closing(sqlite3.connect(features / "features.sqlite3")) as db:
                db.execute("""CREATE TABLE samples (
                    sample_id INTEGER PRIMARY KEY, request_id TEXT,
                    generated_tokens INTEGER, phase TEXT, hidden_size INTEGER,
                    hidden_fp16 BLOB, captured_unix_ns INTEGER)""")
                for sample_id in (1, 2):
                    db.execute(
                        "INSERT INTO samples VALUES (?, ?, 0, ?, 4, ?, 1)",
                        (
                            sample_id,
                            f"{sample_id}-abcdef12",
                            "PREFILL_COMPLETE",
                            np.ones(4, dtype=np.float16).tobytes(),
                        ),
                    )
                db.commit()
            labels = [
                {
                    "request_id": str(i),
                    "input_id": f"prompt-{i}",
                    "source_tree_id": f"tree-{i}",
                    "split": split,
                    "output_tokens": 10,
                    "natural_finish": True,
                }
                for i, split in ((1, "train"), (2, "validation"))
            ]
            (run_dir / "labels.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in labels),
                encoding="utf-8",
            )
            index = [
                {
                    "request_id": f"{i}-abcdef12",
                    "feature_file": "features.sqlite3",
                    "feature_row": i,
                    "generated_tokens": 0,
                    "phase": "PREFILL_COMPLETE",
                    "remaining_tokens": 10,
                    "censored": False,
                    "split": split,
                }
                for i, split in ((1, "train"), (2, "validation"))
            ]
            (run_dir / "sample_index.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in index),
                encoding="utf-8",
            )
            summary = audit_capture(
                features, {row["request_id"]: row for row in labels}
            )
            (run_dir / "summary.json").write_text(json.dumps(summary))
            loaded = load_examples(run_dir)
            self.assertEqual(loaded["hidden"].shape, (2, 4))
            self.assertEqual(loaded["remaining"].tolist(), [10, 10])
            labels[1]["source_tree_id"] = labels[0]["source_tree_id"]
            (run_dir / "labels.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in labels),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "tree split leakage"):
                load_examples(run_dir)

    def test_request_balanced_mae_equalizes_response_lengths(self):
        actual = np.array([10, 10, 10, 10], dtype=np.float32)
        predicted = np.array([10, 10, 10, 0], dtype=np.float32)
        request_ids = np.array(["long", "long", "long", "short"])
        self.assertEqual(request_balanced_mae(actual, predicted, request_ids), 5)


if __name__ == "__main__":
    unittest.main()
