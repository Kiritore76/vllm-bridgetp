"""Exercise scratch training, source balancing and saved four-layer inference."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from predictor_distribution import nonuniform84_upper_edges  # noqa: E402
from predictor_network import build_mlp  # noqa: E402
from predictor_training_mix import mixed_training_weights  # noqa: E402
from predictor_warmstart import rebuild_output_head  # noqa: E402
from train_predictor_distribution import fit_distribution  # noqa: E402

from tests.bridge_tp.test_predictor_nonuniform84 import parent_fixture  # noqa: E402


class MLPScratchTests(unittest.TestCase):
    def test_mass_is_configurable_without_moving_held_out_rows(self):
        data = {
            "requests": np.array(["old"] * 4 + ["new"] * 4 + ["capped"] * 2 + ["val"]),
            "splits": np.array(["train"] * 10 + ["validation"]),
            "generated": np.array([0, 10, 50, 90] * 2 + [0, 50, 0]),
            "remaining": np.array([100, 90, 50, 10] * 2 + [100, 50, 1]),
            "censored": np.array([False] * 8 + [True, True, False]),
        }
        w, report = mixed_training_weights(data, {"old"}, legacy_mass=0.5)
        self.assertAlmostEqual(w[:4].sum(), 0.5)
        self.assertAlmostEqual(w[4:10].sum(), 0.5)
        self.assertEqual(w[10], 0)
        self.assertAlmostEqual(report["cohorts"]["legacy5000"]["loss_mass"], 0.5)
        for mass in (0, 1, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                mixed_training_weights(data, {"old"}, legacy_mass=mass)

    def test_scratch_fit_does_not_inherit_reference_and_balances_validation(self):
        parent = rebuild_output_head(parent_fixture(), nonuniform84_upper_edges(), 9)
        ids = np.array([f"r{i}" for i in range(10)])
        splits = np.array(["train"] * 4 + ["validation"] * 4 + ["test"] * 2)
        lengths = np.array([64, 128, 4096, 9000, 128, 4096, 9000, 2048, 64, 10000])
        data = {
            "hidden": np.random.default_rng(1).normal(size=(10, 2)).astype(np.float16),
            "generated": np.zeros(10, dtype=np.float32),
            "remaining": lengths,
            "requests": ids,
            "splits": splits,
            "censored": np.zeros(10, dtype=bool),
            "phases": np.array(["DECODE"] * 10),
            "languages": np.array(["en"] * 10),
            "audit": {"requests": 10},
            "labels": [
                {
                    "input_id": ids[i],
                    "source_tree_id": ids[i],
                    "split": splits[i],
                    "natural_finish": True,
                    "output_tokens": int(lengths[i]),
                }
                for i in range(10)
            ],
        }
        meta = {
            key: parent[key]
            for key in ("feature_layer", "feature_semantics", "model_config_sha256")
        }
        meta.update(max_tokens=16384, revision="capture", input_sha256="input")
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "reference.pt"
            torch.save(parent, reference)
            sha = hashlib.sha256(reference.read_bytes()).hexdigest()
            for depth in (2, 4):
                out = Path(tmp) / f"mlp{depth}"
                report = fit_distribution(
                    data,
                    meta,
                    out,
                    revision="scratch-test",
                    device_name="cpu",
                    epochs=2,
                    hidden_width=4,
                    learning_rate=3e-4,
                    bucket_profile="nonuniform84",
                    network_depth=depth,
                    reference_checkpoint=reference,
                    reference_sha256=sha,
                    balanced_validation_ids={"old": {"r4", "r5"}, "new": {"r6", "r7"}},
                )
                saved = torch.load(out / "predictor_distribution.pt", weights_only=True)
                self.assertIsNone(saved["warmstart_provenance"]["parent_sha256"])
                self.assertEqual(
                    saved["warmstart_provenance"]["initialization"], "from scratch"
                )
                np.testing.assert_allclose(
                    saved["feature_mean"].numpy(),
                    data["hidden"][:4].astype(np.float32).mean(0),
                )
                self.assertFalse(
                    torch.equal(saved["feature_mean"], parent["feature_mean"])
                )
                self.assertFalse(
                    torch.equal(
                        saved["state_dict"]["0.weight"],
                        parent["state_dict"]["0.weight"],
                    )
                )
                self.assertEqual(report["common_bucket_comparison"]["categories"], 84)
                self.assertEqual(report["reference_provenance"]["sha256"], sha)
                self.assertEqual(len(report["selection_validation_request_ids"]), 2)
                self.assertEqual(len(report["calibration_validation_request_ids"]), 2)
                self.assertEqual(set(report["validation_cohorts"]), {"old", "new"})
                self.assertIn(
                    "balanced_calibrated_nll", report["results"]["selection_validation"]
                )
                model = build_mlp(3, 4, 84, saved["dropout"], depth)
                self.assertEqual(sum(isinstance(m, nn.Linear) for m in model), depth)
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
                    p = torch.softmax(model(features) / saved["temperature"], 1).numpy()
                with np.load(out / "distribution_predictions.npz") as predictions:
                    np.testing.assert_allclose(
                        p, predictions["probabilities"], atol=1e-6
                    )
                if depth == 4:
                    from tests.bridge_tp.test_distribution_predictor_runtime import (
                        runtime,
                    )

                    predictor = runtime.DistributionPredictor(
                        out / "predictor_distribution.pt",
                        checkpoint_sha256=report["checkpoint_sha256"],
                        model_config_sha256="base",
                    )
                    self.assertEqual(predictor.network_depth, 4)
                    np.testing.assert_allclose(
                        predictor.probabilities(
                            torch.tensor(data["hidden"]), torch.zeros(10)
                        ).numpy(),
                        p,
                        atol=1e-6,
                    )
                self.assertEqual(
                    hashlib.sha256(reference.read_bytes()).hexdigest(), sha
                )


if __name__ == "__main__":
    unittest.main()
