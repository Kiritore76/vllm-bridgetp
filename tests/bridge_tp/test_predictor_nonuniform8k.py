# SPDX-License-Identifier: Apache-2.0
"""Finite 8K scratch training, sparse-tail geometry and runtime parity."""

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
    nonuniform8k62_upper_edges,
    probability_gt_bounds,
)
from train_predictor_distribution import (  # noqa: E402
    finite_support_examples,
    fit_distribution,
)


class Nonuniform8KTests(unittest.TestCase):
    def fixture(self):
        from tests.bridge_tp import test_predictor_nonuniform24k as previous

        data, meta = previous.Nonuniform24KTests().fixture()
        data["remaining"][:] = [64, 6144, 8192, 4096, 128, 2048, 4096, 6144, 8192, 8193]
        for i, row in enumerate(data["labels"]):
            row["output_tokens"] = int(data["remaining"][i])
        return data, meta

    def test_geometry_has_progressively_wider_finite_tail(self):
        edges = nonuniform8k62_upper_edges()
        frozen = json.loads(
            (ROOT / "experiments/platforms/qwen3_nonuniform8k62.json").read_text()
        )
        self.assertEqual(len(edges), 62)
        self.assertEqual(edges.tolist(), frozen["finite_upper_edges"])
        self.assertEqual(edges[-4:].tolist(), [4096, 5120, 6144, 8192])
        self.assertTrue(np.all(np.diff(np.diff(edges)) >= 0))
        self.assertFalse(frozen["overflow_category"])
        p = np.zeros((1, 62))
        p[0, -1] = 1
        for horizon, expected in (
            (6144, (1, 1)),
            (7000, (0, 1)),
            (8192, (0, 0)),
            (9000, (0, 0)),
        ):
            low, high = probability_gt_bounds(p, edges, horizon)
            self.assertEqual((low.item(), high.item()), expected)

    def test_filters_whole_requests_instead_of_clamping_labels(self):
        data, _ = self.fixture()
        finite, audit = finite_support_examples(data, 8192)
        self.assertEqual(audit["excluded_samples"], 2)
        self.assertEqual(len(finite["labels"]), 8)
        self.assertIn(8192, finite["remaining"])
        self.assertFalse((finite["remaining"] > 8192).any())
        self.assertEqual(
            audit["excluded_requests"][data["requests"][9]], "output_exceeds_support"
        )
        self.assertEqual(audit["excluded_requests"][data["requests"][3]], "censored")

    def test_scratch_checkpoint_loads_with_62_outputs(self):
        from tests.bridge_tp.test_distribution_predictor_runtime import runtime

        data, meta = self.fixture()
        finite, _ = finite_support_examples(data, 8192)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            report = fit_distribution(
                data,
                meta,
                out,
                revision="8k-test",
                device_name="cpu",
                epochs=2,
                hidden_width=4,
                network_depth=4,
                bucket_profile="nonuniform8k62",
            )
            path = out / "predictor_distribution.pt"
            saved = torch.load(path, weights_only=True)
            self.assertFalse(saved["overflow_category"])
            self.assertEqual(saved["category_upper_edges"][-1].item(), 8192)
            self.assertEqual(report["support_selection"]["retained_requests"], 8)
            self.assertEqual(report["bucket_support"]["train"]["state_counts"][-1], 1)
            self.assertEqual(
                report["bucket_support"]["test"]["independent_request_counts"][-1], 1
            )
            predictor = runtime.DistributionPredictor(
                path,
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                model_config_sha256=meta["model_config_sha256"],
            )
            actual = predictor.probabilities(
                torch.tensor(finite["hidden"]), torch.tensor(finite["generated"])
            ).numpy()
            self.assertEqual(actual.shape, (8, 62))
            with np.load(out / "distribution_predictions.npz") as predictions:
                np.testing.assert_allclose(
                    actual, predictions["probabilities"], atol=1e-6
                )
            low, high = predictor.probability_gt_bounds(torch.tensor(actual), 8192)
            np.testing.assert_allclose(low, 0)
            np.testing.assert_allclose(high, 0)


if __name__ == "__main__":
    unittest.main()
