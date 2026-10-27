# SPDX-License-Identifier: Apache-2.0
"""Finite 16K training, exclusion audit and real runtime parity."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))

from predictor_distribution import (  # noqa: E402
    nonuniform16k91_upper_edges,
    probability_gt_bounds,
)
from train_predictor_distribution import (  # noqa: E402
    finite_support_examples,
    fit_distribution,
)


class Nonuniform16KTests(unittest.TestCase):
    def fixture(self):
        from tests.bridge_tp import test_predictor_nonuniform24k as previous

        data, meta = previous.Nonuniform24KTests().fixture()
        data["remaining"][[2, 6, 9]] = [16384, 16385, 16384]
        for index in (2, 6, 9):
            data["labels"][index]["output_tokens"] = int(data["remaining"][index])
        return data, meta

    def test_frozen_geometry_and_closed_boundary(self):
        edges = nonuniform16k91_upper_edges()
        frozen = json.loads(
            (ROOT / "experiments/platforms/qwen3_nonuniform16k91.json").read_text()
        )
        self.assertEqual(len(edges), 91)
        self.assertEqual(edges.tolist(), frozen["finite_upper_edges"])
        self.assertFalse(frozen["overflow_category"])
        self.assertTrue(np.all(np.diff(np.diff(edges)) >= 0))
        p = np.zeros((1, 91))
        p[0, -1] = 1
        for horizon, expected in (
            (15872, (1, 1)),
            (16000, (0, 1)),
            (16384, (0, 0)),
            (24576, (0, 0)),
        ):
            low, high = probability_gt_bounds(p, edges, horizon)
            self.assertEqual((low.item(), high.item()), expected)

    def test_exclusion_removes_whole_requests_without_mutating_capture(self):
        data, _ = self.fixture()
        finite, audit = finite_support_examples(data)
        self.assertEqual(audit["excluded_samples"], 2)
        self.assertEqual(len(finite["labels"]), 8)
        self.assertEqual(len(data["labels"]), 10)
        self.assertFalse(finite["censored"].any())
        self.assertTrue((finite["remaining"] <= 16384).all())
        self.assertIn("output_exceeds_support", audit["excluded_requests"].values())
        self.assertIn("censored", audit["excluded_requests"].values())

    def test_91_class_training_runtime_and_support_metadata(self):
        from tests.bridge_tp.test_distribution_predictor_runtime import runtime

        data, meta = self.fixture()
        finite, _ = finite_support_examples(data)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            report = fit_distribution(
                data,
                meta,
                out,
                revision="test-16k",
                device_name="cpu",
                epochs=2,
                hidden_width=4,
                network_depth=4,
                bucket_profile="nonuniform16k91",
            )
            path = out / "predictor_distribution.pt"
            saved = torch.load(path, weights_only=True)
            self.assertFalse(saved["overflow_category"])
            self.assertEqual(saved["category_upper_edges"][-1].item(), 16384)
            self.assertEqual(report["censored_training_requests"], 0)
            self.assertIsNone(report["open_tail_lower_inclusive"])
            predictor = runtime.DistributionPredictor(
                path,
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                model_config_sha256=meta["model_config_sha256"],
            )
            actual = predictor.probabilities(
                torch.tensor(finite["hidden"]), torch.tensor(finite["generated"])
            ).numpy()
            self.assertEqual(actual.shape, (8, 91))
            with np.load(out / "distribution_predictions.npz") as predictions:
                np.testing.assert_allclose(
                    actual, predictions["probabilities"], atol=1e-6
                )
            horizons = torch.tensor(
                [-1, 0, 8000, 8192, 16000, 16384, 18000, float("inf")]
            )
            low, high = predictor.probability_gt_bounds(torch.tensor(actual), horizons)
            for index, horizon in enumerate(horizons.tolist()):
                expected = probability_gt_bounds(
                    actual[index], nonuniform16k91_upper_edges(), horizon
                )
                np.testing.assert_allclose(
                    [low[index], high[index]], expected, atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
