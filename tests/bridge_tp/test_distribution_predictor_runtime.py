"""Frozen distribution predictor inference and risk-bound checks."""

import hashlib
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

MODULE = (
    Path(__file__).resolve().parents[2]
    / "vllm/bridge_tp/controller/distribution_predictor.py"
)
spec = importlib.util.spec_from_file_location("distribution_predictor", MODULE)
assert spec is not None and spec.loader is not None
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
DistributionPredictor = runtime.DistributionPredictor


class TestDistributionPredictor(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "predictor.pt"
        model = nn.Sequential(
            nn.Linear(3, 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(4, 4)
        )
        torch.manual_seed(17)
        for parameter in model.parameters():
            nn.init.uniform_(parameter, -0.25, 0.25)
        self.model = model.eval()
        torch.save(
            {
                "format_version": 2,
                "model_type": "remaining_length_categorical",
                "state_dict": model.state_dict(),
                "input_width": 3,
                "hidden_width": 4,
                "activation": "GELU",
                "dropout": 0.1,
                "feature_mean": torch.tensor([1.0, 2.0]),
                "feature_std": torch.tensor([2.0, 4.0]),
                "position_log_scale": math.log1p(32),
                "category_upper_edges": torch.tensor([0, 8, 16]),
                "overflow_category": True,
                "temperature": 1.15,
                "feature_layer": "decoder:31",
                "model_config_sha256": "base-model",
                "capture_input_sha256": "input-sha",
            },
            self.path,
        )
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def load(self, **overrides):
        kwargs = {
            "checkpoint_sha256": self.sha,
            "model_config_sha256": "base-model",
            "feature_layer": "decoder:31",
        }
        kwargs.update(overrides)
        return DistributionPredictor(self.path, **kwargs)

    def test_inference_reproduces_standardization_and_temperature(self):
        predictor = self.load()
        hidden = torch.tensor([[3.0, 6.0], [1.0, 2.0]])
        generated = torch.tensor([8, 0])
        actual = predictor.probabilities(hidden, generated)
        features = torch.tensor(
            [[1.0, 1.0, math.log1p(8) / math.log1p(32)], [0.0, 0.0, 0.0]]
        )
        expected = torch.softmax(self.model(features) / 1.15, dim=-1)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
        self.assertEqual(predictor.capture_input_sha256, "input-sha")

    def test_exact_interior_tail_and_negative_horizons(self):
        predictor = self.load()
        probability = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).repeat(6, 1)
        horizon = torch.tensor([-1.0, 0.0, 4.0, 8.0, 20.0, math.inf])
        low, high = predictor.probability_gt_bounds(probability, horizon)
        torch.testing.assert_close(
            low, torch.tensor([1.0, 0.9, 0.7, 0.7, 0.0, 0.0])
        )
        torch.testing.assert_close(
            high, torch.tensor([1.0, 0.9, 0.9, 0.7, 0.4, 0.0])
        )

    def test_identity_and_shape_mismatch_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.load(checkpoint_sha256="wrong")
        with self.assertRaisesRegex(ValueError, "feature layer"):
            self.load(feature_layer="final")
        with self.assertRaisesRegex(ValueError, "base model"):
            self.load(model_config_sha256="another-model")
        predictor = self.load()
        with self.assertRaisesRegex(ValueError, "width"):
            predictor.probabilities(torch.zeros(1, 3), torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "one finite or infinite"):
            predictor.probability_gt_bounds(
                torch.full((1, 4), 0.25), torch.tensor([math.nan])
            )


if __name__ == "__main__":
    unittest.main()
