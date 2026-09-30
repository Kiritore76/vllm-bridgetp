"""Probability boundaries, censoring, split hygiene and checkpoint smoke."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from predictor_distribution import (  # noqa: E402
    binary_risk_metrics,
    category_targets,
    distribution_nll,
    km_conditional_survival,
    probability_gt_bounds,
    probability_remaining_gt,
    softmax,
)
from train_predictor_distribution import (  # noqa: E402
    fit_distribution,
    validation_groups,
)


class TestPredictorDistribution(unittest.TestCase):
    def test_boundaries_monotonicity_and_open_tail(self):
        edges = np.array([0, 8, 16])
        p = np.array([[0.1, 0.2, 0.3, 0.4]])
        self.assertEqual(
            category_targets(np.array([0, 8, 9, 17]), edges).tolist(), [0, 1, 2, 3]
        )
        lower, upper = probability_gt_bounds(p, edges, 8)
        np.testing.assert_allclose(lower, [0.7])
        np.testing.assert_allclose(upper, lower)
        lower, upper = probability_gt_bounds(p, edges, 12)
        np.testing.assert_allclose(lower, [0.4])
        np.testing.assert_allclose(upper, [0.7])
        lower, upper = probability_gt_bounds(p, edges, 100)
        np.testing.assert_allclose(lower, [0])
        np.testing.assert_allclose(upper, [0.4])
        risks = [
            probability_remaining_gt(p, edges, h)[0] for h in (-1, 0, 1, 8, 12, 16, 100)
        ]
        self.assertTrue(np.all(np.diff(risks) <= 0))
        with self.assertRaises(ValueError):
            probability_remaining_gt(p * 2, edges, 8)

    def test_censored_likelihood_is_tail_not_exact_length(self):
        p = np.array([[0.1, 0.2, 0.3, 0.4]])
        args = (p, np.array([2]))
        exact = distribution_nll(*args, np.array([False]), np.array(["a"]))
        capped = distribution_nll(*args, np.array([True]), np.array(["a"]))
        self.assertAlmostEqual(exact, -np.log(0.3))
        self.assertAlmostEqual(capped, -np.log(0.7))
        with self.assertRaises(ValueError):
            softmax(np.zeros((1, 2)), 0)

    def test_reliability_bins_cover_every_sample_once(self):
        p = np.arange(11) / 10
        result = binary_risk_metrics(p, p > 0.5, np.arange(11))
        self.assertEqual(sum(x["samples"] for x in result["reliability"]), 11)
        self.assertAlmostEqual(sum(x["weight"] for x in result["reliability"]), 1)

    def test_km_does_not_extrapolate_unknown_tail(self):
        result = km_conditional_survival(
            np.array([10, 20]), np.array([False, True]), np.array([0, 0, 20]), 15
        )
        np.testing.assert_allclose(result[:2], [0.5, 0.5])
        self.assertTrue(np.isnan(result[2]))
        result = km_conditional_survival(
            np.array([10, 20]), np.array([False, True]), np.array([0]), 21
        )
        self.assertTrue(np.isnan(result[0]))

    def test_validation_never_separates_messages_from_same_tree(self):
        data = {
            "labels": [
                {
                    "input_id": key,
                    "source_tree_id": tree,
                    "split": "validation",
                    "natural_finish": True,
                }
                for key, tree in (("a", "shared"), ("b", "shared"), ("c", "other"))
            ],
            "requests": np.array(["a", "b", "c"]),
        }
        selection, calibration = validation_groups(data)
        self.assertEqual(selection[0], selection[1])
        self.assertTrue(np.all(selection ^ calibration))

    def test_cpu_training_saves_calibrated_distribution(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch required for checkpoint smoke")
        rng = np.random.default_rng(42)
        splits = np.array(["train"] * 4 + ["validation"] * 4 + ["test"] * 2)
        ids = np.array([f"request-{i}" for i in range(10)])
        lengths = np.array([8, 16, 32, 64, 16, 32, 8, 64, 16, 32])
        capped = np.zeros(10, dtype=bool)
        capped[3] = True
        labels = [
            {
                "input_id": ids[i],
                "source_tree_id": f"tree-{i}",
                "split": splits[i],
                "natural_finish": not capped[i],
                "output_tokens": int(lengths[i]),
            }
            for i in range(10)
        ]
        data = {
            "hidden": rng.normal(size=(10, 4)).astype(np.float16),
            "generated": np.zeros(10),
            "remaining": lengths,
            "requests": ids,
            "splits": splits,
            "censored": capped,
            "phases": np.array(["PREFILL_COMPLETE"] * 10),
            "languages": np.array(["en"] * 10),
            "labels": labels,
            "audit": {"requests": 10},
        }
        preflight = {
            "max_tokens": 64,
            "revision": "capture-revision",
            "input_sha256": "input-sha",
            "model_config_sha256": "model-sha",
        }
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            report = fit_distribution(
                data,
                preflight,
                out,
                revision="test-revision",
                device_name="cpu",
                epochs=2,
                hidden_width=8,
                batch_size=4,
            )
            checkpoint = torch.load(
                out / "predictor_distribution.pt", weights_only=True
            )
            self.assertEqual(checkpoint["model_type"], "remaining_length_categorical")
            self.assertEqual(report["censored_training_requests"], 1)
            self.assertEqual(report["results"]["test"]["requests"], 2)
            self.assertFalse(
                set(report["selection_validation_request_ids"])
                & set(report["calibration_validation_request_ids"])
            )
            with np.load(out / "distribution_predictions.npz") as predictions:
                np.testing.assert_allclose(
                    predictions["probabilities"].sum(axis=1), 1, atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
