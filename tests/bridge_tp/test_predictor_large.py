"""Audit resumable captures and collision-free merging on CPU."""

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
from run_predictor_large_a100 import (  # noqa: E402
    MODEL_SHA,
    merge_shards,
    validate_shard,
)
from train_length_predictor import load_examples, sha256_file  # noqa: E402


class TestPredictorLarge(unittest.TestCase):
    def make_shard(self, root, number, split):
        run = root / f"shard_{number}"
        (run / "features").mkdir(parents=True)
        request = {
            "id": f"prompt-{number}",
            "prompt": "Example prompt",
            "split": split,
            "lang": "en",
            "source_tree_id": f"tree-{number}",
        }
        label = {
            "input_id": request["id"],
            "request_id": "0",
            "split": split,
            "lang": "en",
            "source_tree_id": request["source_tree_id"],
            "output_tokens": 4096 if number else 16,
            "natural_finish": not number,
        }
        input_path = root / f"input_{number}.jsonl"
        content = json.dumps(request) + "\n"
        input_path.write_text(content, encoding="utf-8")
        (run / "input_requests.jsonl").write_text(content, encoding="utf-8")
        (run / "labels.jsonl").write_text(json.dumps(label) + "\n")
        preflight = {
            "revision": "test-revision",
            "input_sha256": sha256_file(input_path),
            "model_path": "/same/model",
            "model_config_sha256": MODEL_SHA,
            "gpu_names": ["NVIDIA A100-PCIE-40GB"],
            "interval": 20,
            "max_tokens": 4096,
            "max_model_len": 6144,
            "temperature": 0,
            "requests": 1,
        }
        (run / "preflight.json").write_text(json.dumps(preflight))
        with closing(sqlite3.connect(run / "features/features.sqlite3")) as db:
            db.execute("""CREATE TABLE samples (sample_id INTEGER PRIMARY KEY,
                request_id TEXT, generated_tokens INTEGER, phase TEXT,
                hidden_size INTEGER, hidden_fp16 BLOB, captured_unix_ns INTEGER)""")
            db.execute(
                "INSERT INTO samples VALUES (1, ?, 0, ?, 4, ?, 1)",
                (
                    "0-abcdef12",
                    "PREFILL_COMPLETE",
                    np.ones(4, dtype=np.float16).tobytes(),
                ),
            )
            db.commit()
        audit = audit_capture(run / "features", {"0": label})
        (run / "summary.json").write_text(json.dumps(audit))
        return run, input_path, content

    def test_merge_preserves_censoring_and_avoids_engine_id_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, ia, ca = self.make_shard(root, 0, "train")
            b, ib, cb = self.make_shard(root, 1, "test")
            validate_shard(a, ia, "test-revision")
            validate_shard(b, ib, "test-revision")
            full = root / "full.jsonl"
            full.write_text(ca + cb, encoding="utf-8")
            summary = merge_shards([a, b], full, root / "merged")
            self.assertEqual(summary["requests"], 2)
            self.assertEqual(summary["censored_requests"], 1)
            validate_shard(root / "merged", full, "test-revision")
            examples = load_examples(root / "merged", include_censored=True)
            self.assertEqual(examples["remaining"].tolist(), [16, 4096])
            self.assertEqual(examples["censored"].tolist(), [False, True])
            self.assertEqual(len(load_examples(root / "merged")["requests"]), 1)
            with self.assertRaisesRegex(ValueError, "duplicate input"):
                merge_shards([a, a], full, root / "duplicated")

    def test_resume_rejects_wrong_revision_incomplete_coverage_and_bad_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, path, _ = self.make_shard(root, 0, "train")
            with self.assertRaisesRegex(ValueError, "protocol/input"):
                validate_shard(run, path, "other-revision")
            summary = json.loads((run / "summary.json").read_text())
            summary["samples"] += 1
            (run / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "summary differs"):
                validate_shard(run, path, "test-revision")
            (run / "labels.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "coverage"):
                validate_shard(run, path, "test-revision")


if __name__ == "__main__":
    unittest.main()
