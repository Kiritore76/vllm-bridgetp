# SPDX-License-Identifier: Apache-2.0
"""A100 stage-weight reuse, finite filtering and weighted-midpoint decoding."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from predictor_distribution import finite_bucket_midpoint_mean  # noqa: E402
from predictor_training_mix import mixed_training_weights  # noqa: E402
from train_predictor_distribution import fit_distribution  # noqa: E402


class Qwen3RetrainingTests(unittest.TestCase):
    def sample(self):
        return dict(
            requests=np.array(["a"] * 4 + ["b"] * 4 + ["held"]),
            splits=np.array(["train"] * 8 + ["test"]),
            generated=np.array([0, 10, 50, 90] * 2 + [0], dtype=float),
            remaining=np.array([100, 90, 50, 10] * 2 + [10], dtype=float),
            censored=np.zeros(9, dtype=bool),
            phases=np.array(
                ["PREFILL_COMPLETE", "DECODE", "DECODE", "DECODE"] * 2
                + ["PREFILL_COMPLETE"]
            ),
        )

    def test_single_cohort_reuses_a100_weighting_exactly(self):
        data = self.sample()
        original, _ = mixed_training_weights(
            data, {"a"}, legacy_mass=0.5, prefill_mass=0.2
        )
        weights, report = mixed_training_weights(data, None, prefill_mass=0.2)
        np.testing.assert_allclose(weights, original)
        np.testing.assert_allclose(weights[:4] * 2, [0.2, 0.24, 0.32, 0.24])
        self.assertEqual(weights[-1], 0)
        self.assertEqual(set(report["cohorts"]), {"all_requests"})
        self.assertAlmostEqual(weights.sum(), 1)

    def test_prefill_only_and_missing_stages_preserve_request_mass(self):
        data = self.sample()
        # Drop the early decode state from a and every decode state from b.
        keep = np.array([0, 2, 3, 4, 8])
        data = {key: value[keep] for key, value in data.items()}
        weights, report = mixed_training_weights(data, None, prefill_mass=0.2)
        np.testing.assert_allclose(weights[:3] * 2, [0.2, 0.8 * 4 / 7, 0.8 * 3 / 7])
        self.assertAlmostEqual(weights[3], 0.5)
        self.assertEqual(weights[-1], 0)
        self.assertEqual(report["cohorts"]["all_requests"]["prefill_only_requests"], 1)

    def test_mean_uses_every_probability_not_the_median_bin(self):
        edges = np.array([0, 8, 16])
        p = np.array([[0.1, 0.2, 0.7]])
        self.assertAlmostEqual(finite_bucket_midpoint_mean(p, edges).item(), 9.65)
        with self.assertRaises(ValueError):
            finite_bucket_midpoint_mean(p * 2, edges)

    def test_filtered_finite_fit_records_real_stage_weights_and_mean(self):
        from tests.bridge_tp import test_predictor_nonuniform8k as previous

        data, meta = previous.Nonuniform8KTests().fixture()
        lengths = data["remaining"].copy()
        index = np.repeat(np.arange(len(lengths)), 4)
        expanded = {
            key: value[index] if isinstance(value, np.ndarray) else value
            for key, value in data.items()
        }
        generated = np.concatenate(
            [np.array([0, int(n * 0.1), int(n * 0.5), int(n * 0.9)]) for n in lengths]
        )
        expanded["generated"] = generated.astype(np.float32)
        expanded["remaining"] = lengths[index] - generated
        expanded["phases"] = np.where(generated == 0, "PREFILL_COMPLETE", "DECODE")
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            report = fit_distribution(
                expanded,
                meta,
                out,
                revision="retrain-test",
                device_name="cpu",
                epochs=2,
                hidden_width=4,
                network_depth=4,
                bucket_profile="nonuniform8k62",
                learning_rate=0.0003,
                request_weighting="prefill20_stage304030",
            )
            self.assertEqual(report["support_selection"]["retained_requests"], 8)
            self.assertEqual(
                report["training_parameters"]["request_weighting"],
                "prefill20_stage304030",
            )
            self.assertEqual(
                report["training_weighting"]["stage_mass"], [0.3, 0.4, 0.3]
            )
            cohort = report["training_weighting"]["cohorts"]["all_requests"]
            self.assertEqual(cohort["training_requests"], 3)
            self.assertAlmostEqual(cohort["loss_mass"], 1)
            self.assertAlmostEqual(cohort["prefill_loss_mass"], 0.2)
            checkpoint = torch.load(
                out / "predictor_distribution.pt", weights_only=True
            )
            self.assertEqual(
                checkpoint["training_weighting"], report["training_weighting"]
            )
            self.assertEqual(
                checkpoint["point_estimator"],
                "probability_weighted_integer_bucket_midpoint_mean",
            )
            with np.load(out / "distribution_predictions.npz") as z:
                self.assertAlmostEqual(z["training_loss_weights"].sum(), 1)
                self.assertTrue(
                    (z["training_loss_weights"][z["splits"] != "train"] == 0).all()
                )
                for identity in set(z["request_ids"][z["splits"] == "train"]):
                    mask = z["request_ids"] == identity
                    mass = z["training_loss_weights"][mask]
                    phases = z["phases"][mask]
                    self.assertAlmostEqual(mass.sum(), 1 / 3)
                    self.assertAlmostEqual(
                        mass[phases == "PREFILL_COMPLETE"].sum() / mass.sum(), 0.2
                    )
                expected = finite_bucket_midpoint_mean(
                    z["probabilities"], z["category_upper_edges"]
                )
                np.testing.assert_allclose(
                    z["bucket_mean_remaining_tokens"], expected, atol=1e-3
                )
                self.assertEqual(len(expected), 32)
            self.assertIn("mae", report["results"]["test"]["bucket_mean_point_metrics"])


if __name__ == "__main__":
    unittest.main()
