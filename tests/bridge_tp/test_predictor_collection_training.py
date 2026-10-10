"""Protect shard identity and permit same-tree, same-split observations."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from predictor_collection_training import load_collection_examples  # noqa: E402


def collection_fixture(batch, splits=("train", "train")):
    """Construct two shards from the same source tree with audited stand-ins."""
    inputs = batch / "inputs"
    inputs.mkdir()
    recipe = [
        {"id": f"r{i}", "source_tree_id": "tree", "split": splits[i]} for i in range(2)
    ]
    raw = "".join(json.dumps(row) + "\n" for row in recipe).encode()
    (inputs / "recipe.jsonl").write_bytes(raw)
    recipe_sha = hashlib.sha256(raw).hexdigest()
    shards, parts = [], {}
    for i, row in enumerate(recipe):
        name = f"shard{i}"
        request = {**row, "prompt": f"prompt{i}", "planned_prompt_tokens": 2}
        payload = (json.dumps(request) + "\n").encode()
        source = inputs / f"{name}.jsonl"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        out = batch / "captures" / name
        out.mkdir(parents=True)
        meta = {
            "revision": "capture",
            "input_sha256": sha,
            "feature_layer": "decoder:31",
            "feature_semantics": "hidden+residual",
            "model_config_sha256": "base",
            "max_tokens": 16384,
        }
        (out / "preflight.json").write_text(json.dumps(meta))
        shards.append({"name": name, "input": source.name, "sha256": sha, "rows": 1})
        label = {
            "input_id": row["id"],
            "source_tree_id": "tree",
            "split": splits[i],
            "natural_finish": True,
            "prompt_tokens": 2,
            "prompt_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
        }
        parts[name] = {
            "hidden": np.zeros((1, 2), dtype=np.float16),
            "remaining": np.array([32]),
            "generated": np.array([0]),
            "splits": np.array([splits[i]]),
            "requests": np.array([row["id"]]),
            "phases": np.array(["DECODE"]),
            "languages": np.array(["en"]),
            "censored": np.array([False]),
            "labels": [label],
            "audit": {"requests": 1},
        }
    (inputs / "manifest.json").write_text(
        json.dumps({"rows": 2, "shards": shards, "recipe_sha256": recipe_sha})
    )
    (batch / "collection_summary.json").write_text(
        json.dumps(
            {
                "completed_shards": [s["name"] for s in shards],
                "requests": 2,
                "natural_eos": 2,
                "censored": 0,
            }
        )
    )
    return recipe_sha, parts


class CollectionTrainingTests(unittest.TestCase):
    def load(self, batch, sha, parts, **kwargs):
        with patch(
            "predictor_collection_training.load_examples",
            side_effect=lambda path, **unused: parts[path.name],
        ):
            return load_collection_examples(
                batch,
                expected_revision="capture",
                expected_recipe_sha256=sha,
                expected_feature_layer="decoder:31",
                **kwargs,
            )

    def test_same_tree_across_shards_is_valid_and_input_fallback_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            sha, parts = collection_fixture(batch)
            data, meta = self.load(batch, sha, parts)
            self.assertEqual(data["requests"].tolist(), ["r0", "r1"])
            self.assertEqual(meta["input_sha256"], sha)
            self.assertEqual(data["audit"]["requests"], 2)

    def test_cross_shard_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            sha, parts = collection_fixture(batch, splits=("train", "test"))
            with self.assertRaisesRegex(ValueError, "source tree"):
                self.load(batch, sha, parts)

    def test_digest_corruption_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            sha, parts = collection_fixture(batch)
            with (batch / "inputs/shard0.jsonl").open("a") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "input SHA"):
                self.load(batch, sha, parts)

    def test_partial_collection_is_only_allowed_as_explicit_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            sha, parts = collection_fixture(batch)
            (batch / "collection_summary.json").write_text(
                json.dumps(
                    {
                        "completed_shards": ["shard0"],
                        "requests": 1,
                        "natural_eos": 1,
                        "censored": 0,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "not complete"):
                self.load(batch, sha, parts)
            data, _ = self.load(batch, sha, parts, require_complete=False)
            self.assertEqual(data["audit"]["requests"], 1)


if __name__ == "__main__":
    unittest.main()
