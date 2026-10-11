"""Check direct 76-output training, open-tail semantics and aligned comparisons."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from predictor_distribution import (  # noqa: E402
    aggregate_probabilities,
    category_targets,
    nonuniform84_upper_edges,
    probability_gt_bounds,
    risk76_upper_edges,
)
from predictor_network import build_mlp  # noqa: E402
from predictor_warmstart import rebuild_output_head  # noqa: E402
from train_predictor_distribution import fit_distribution  # noqa: E402

from tests.bridge_tp.test_predictor_nonuniform84 import parent_fixture  # noqa: E402


class Risk76Tests(unittest.TestCase):
    def test_inclusive_boundary_and_single_open_tail(self):
        old = nonuniform84_upper_edges()
        new = risk76_upper_edges()
        self.assertEqual(len(old) + 1, 84)
        self.assertEqual(len(new) + 1, 76)
        np.testing.assert_array_equal(new, old[:75])
        self.assertEqual(new[-1], 8192)
        np.testing.assert_array_equal(
            category_targets(np.array([8191, 8192, 8193, 12000, 16384, 99999]), new),
            [74, 74, 75, 75, 75, 75],
        )

    def test_aggregation_preserves_shared_risk_without_inventing_tail_detail(self):
        old, new = nonuniform84_upper_edges(), risk76_upper_edges()
        p = np.random.default_rng(1).dirichlet(np.ones(84), size=3)
        merged = aggregate_probabilities(p, old, new)
        self.assertEqual(merged.shape, (3, 76))
        np.testing.assert_allclose(merged[:, :75], p[:, :75])
        np.testing.assert_allclose(merged[:, -1], p[:, 75:].sum(axis=1))
        for horizon in [0, 32, 512, 2048, 4096, 8192]:
            before = probability_gt_bounds(p, old, horizon)
            after = probability_gt_bounds(merged, new, horizon)
            np.testing.assert_allclose(after, before)
        lower, upper = probability_gt_bounds(merged, new, 12000)
        np.testing.assert_array_equal(lower, np.zeros(3))
        np.testing.assert_allclose(upper, merged[:, -1])

    def test_both_depths_train_direct_heads_and_preserve_reference(self):
        reference = rebuild_output_head(parent_fixture(), nonuniform84_upper_edges(), 9)
        ids = np.array([f"r{i}" for i in range(10)])
        splits = np.array(["train"] * 4 + ["validation"] * 4 + ["test"] * 2)
        lengths = np.array([64, 8192, 9000, 18000, 128, 9000, 16384, 2048, 8192, 20000])
        censored = np.array([False] * 3 + [True] + [False] * 6)
        data = {
            "hidden": np.random.default_rng(1).normal(size=(10, 2)).astype(np.float16),
            "generated": np.zeros(10, dtype=np.float32),
            "remaining": lengths,
            "requests": ids,
            "splits": splits,
            "censored": censored,
            "phases": np.array(["PREFILL_COMPLETE"] * 10),
            "languages": np.array(["en"] * 10),
            "audit": {"requests": 10},
            "labels": [
                {
                    "input_id": ids[i],
                    "source_tree_id": ids[i],
                    "split": splits[i],
                    "natural_finish": not bool(censored[i]),
                    "output_tokens": int(lengths[i]),
                }
                for i in range(10)
            ],
        }
        meta = {
            key: reference[key]
            for key in ("feature_layer", "feature_semantics", "model_config_sha256")
        }
        meta.update(max_tokens=24576, revision="capture", input_sha256="input")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.pt"
            torch.save(reference, path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            for depth in (2, 4):
                directory = Path(temporary) / f"depth{depth}"
                report = fit_distribution(
                    data,
                    meta,
                    directory,
                    revision="risk76-test",
                    device_name="cpu",
                    epochs=2,
                    hidden_width=4,
                    learning_rate=3e-4,
                    bucket_profile="risk76",
                    network_depth=depth,
                    reference_checkpoint=path,
                    reference_sha256=sha,
                    balanced_validation_ids={"old": {"r4", "r5"}, "new": {"r6", "r7"}},
                )
                saved = torch.load(
                    directory / "predictor_distribution.pt", weights_only=True
                )
                self.assertEqual(saved["category_upper_edges"][-1], 8192)
                self.assertEqual(saved["category_upper_edges"].numel(), 75)
                self.assertIsNone(saved["warmstart_provenance"]["parent_sha256"])
                self.assertEqual(
                    saved["warmstart_provenance"]["initialization"], "from scratch"
                )
                self.assertEqual(
                    saved["warmstart_provenance"]["bucket_profile"], "risk76"
                )
                np.testing.assert_allclose(
                    saved["feature_mean"], data["hidden"][:4].astype(np.float32).mean(0)
                )
                self.assertEqual(report["common_bucket_comparison"]["categories"], 76)
                self.assertEqual(report["open_tail_lower_inclusive"], 8193)
                self.assertEqual(report["censored_training_requests"], 1)
                model = build_mlp(3, 4, 76, saved["dropout"], depth)
                model.load_state_dict(saved["state_dict"], strict=True)
                model.eval()
                features = torch.column_stack(
                    (
                        (
                            torch.tensor(data["hidden"], dtype=torch.float32)
                            - saved["feature_mean"]
                        )
                        / saved["feature_std"],
                        torch.zeros(10),
                    )
                )
                with torch.no_grad():
                    probabilities = torch.softmax(
                        model(features) / saved["temperature"], 1
                    ).numpy()
                with np.load(directory / "distribution_predictions.npz") as predictions:
                    self.assertEqual(predictions["probabilities"].shape, (10, 76))
                    np.testing.assert_allclose(
                        probabilities, predictions["probabilities"], atol=1e-6
                    )
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sha)


if __name__ == "__main__":
    unittest.main()
