# SPDX-License-Identifier: Apache-2.0
"""Scalar targets, censoring, split isolation, point estimates and real fitting."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from predictor_length_regression import (  # noqa: E402
    accuracy,
    categorical_points,
    fit,
    normalization,
    regression_loss,
    selection_masks,
    targets,
)
from predictor_network import build_mlp  # noqa: E402
from predictor_training_mix import mixed_training_weights  # noqa: E402
from run_predictor_length_regression import comparisons  # noqa: E402


def fixture():
    ids = [f"r{i}" for i in range(10)]
    splits = ["train", "train", "validation", "validation", "test"] * 2
    requests = np.repeat(ids, 3)
    generated = np.tile([0, 5, 19], 10).astype(np.float32)
    data = dict(
        hidden=np.random.default_rng(1).normal(size=(30, 4)).astype(np.float16),
        remaining=(np.repeat(np.arange(20, 30), 3) - generated).astype(np.float32),
        generated=generated,
        requests=requests,
        splits=np.repeat(splits, 3),
        phases=np.tile(["PREFILL_COMPLETE", "DECODE", "DECODE"], 10),
        censored=np.zeros(30, dtype=bool),
        labels=[
            dict(
                input_id=key,
                source_tree_id=key,
                split=split,
                natural_finish=True,
                output_tokens=20 + i,
            )
            for i, (key, split) in enumerate(zip(ids, splits))
        ],
    )
    cohorts = dict(old=set(ids[:5]), new=set(ids[5:]))
    weights, _ = mixed_training_weights(
        data, cohorts["old"], legacy_mass=0.5, prefill_mass=0.2
    )
    meta = dict(
        max_tokens=16384,
        revision="capture",
        input_sha256="input",
        model_config_sha256="model",
        feature_layer="decoder:31",
        feature_semantics="hidden+residual",
    )
    return data, meta, weights, cohorts


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_remaining_label_is_not_subtracted_twice(self):
        bounded, exact = targets(
            dict(
                remaining=np.array([500, 9000, 100]),
                censored=np.array([False, True, True]),
            )
        )
        np.testing.assert_equal(bounded, [500, 8192, 100])
        np.testing.assert_equal(exact, [True, True, False])

    def test_censoring_has_no_penalty_above_lower_bound(self):
        prediction = torch.tensor([3.0, 1.0, 3.0], requires_grad=True)
        loss = regression_loss(
            prediction,
            torch.tensor([2.0, 2.0, 2.0]),
            torch.tensor([False, False, True]),
        )
        loss.sum().backward()
        self.assertEqual(prediction.grad[0], 0)
        self.assertLess(prediction.grad[1], 0)
        self.assertGreater(prediction.grad[2], 0)

    def test_normalization_never_uses_validation_or_test(self):
        data, _, _, _ = fixture()
        expected = normalization(data)
        data["hidden"][data["splits"] != "train"] = 10000
        for a, b in zip(expected, normalization(data)):
            np.testing.assert_array_equal(a, b)

    def test_selection_reserves_original_calibration_half_and_no_test(self):
        data, _, _, cohorts = fixture()
        _, selected, calibration = selection_masks(data, cohorts)
        self.assertFalse((selected & calibration).any())
        self.assertTrue((data["splits"][selected | calibration] == "validation").all())
        self.assertEqual(len(set(data["requests"][selected])), 2)
        self.assertEqual(len(set(data["requests"][calibration])), 2)

    def test_bucket_tail_mass_is_preserved_for_both_estimators(self):
        mean, median = categorical_points(
            np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]), np.array([0, 8192])
        )
        self.assertEqual(mean[0], 8192)
        self.assertEqual(median[0], 8192)
        self.assertAlmostEqual(mean[1], 2048.25)
        self.assertEqual(median[1], 0)

    def test_accuracy_is_request_equal(self):
        result = accuracy(
            np.zeros(4), np.array([0, 100, 100, 100]), np.array(["a", "b", "b", "b"])
        )
        self.assertAlmostEqual(result["request_balanced_mae_tokens"], 50)

    def test_four_affine_layers_end_in_one_output(self):
        model = build_mlp(5121, 256, 1, 0.1, 4)
        layers = [layer for layer in model if isinstance(layer, nn.Linear)]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in layers],
            [(5121, 256), (256, 512), (512, 256), (256, 1)],
        )

    def test_train_reload_and_test_label_isolation(self):
        data, meta, weights, cohorts = fixture()
        checkpoints = []
        with tempfile.TemporaryDirectory() as temp:
            for number in (0, 1):
                variant = copy.deepcopy(data)
                if number:
                    variant["remaining"][variant["splits"] == "test"] += 5000
                out = Path(temp) / str(number)
                predicted, report = fit(
                    variant,
                    meta,
                    out,
                    weights,
                    cohorts,
                    revision="test",
                    device="cpu",
                    epochs=2,
                    width=4,
                )
                self.assertTrue(np.isfinite(predicted).all())
                self.assertEqual(predicted.shape, (30,))
                self.assertIn(report["best_epoch"], (1, 2))
                checkpoints.append(
                    torch.load(out / "predictor_regression.pt", weights_only=True)
                )
            for key in checkpoints[0]["state_dict"]:
                self.assertTrue(
                    torch.equal(
                        checkpoints[0]["state_dict"][key],
                        checkpoints[1]["state_dict"][key],
                    )
                )

    def test_evaluation_uses_same_exact_states_for_all_models(self):
        data, _, _, cohorts = fixture()
        data["censored"][data["requests"] == "r4"] = True
        truth, _ = targets(data)
        result = comparisons(
            data, cohorts, dict(regression=truth, classifier_mean=truth)
        )
        self.assertNotIn("old/test/all_bounded/all", result)
        self.assertEqual(
            result["new/test/all_bounded/all"]["regression"],
            result["new/test/all_bounded/all"]["classifier_mean"],
        )


if __name__ == "__main__":
    unittest.main()
