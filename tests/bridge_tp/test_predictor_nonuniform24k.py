# SPDX-License-Identifier: Apache-2.0
"""24K head geometry, exact-boundary risk and scratch-training/runtime parity."""

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
    category_targets,
    nonuniform24k100_upper_edges,
    probability_gt_bounds,
    risk76_upper_edges,
)
from train_predictor_distribution import fit_distribution  # noqa: E402


class Nonuniform24KTests(unittest.TestCase):
    def fixture(self):
        ids = np.array([f"qwen3-24k-r{i}" for i in range(10)])
        splits = np.array(["train"] * 4 + ["validation"] * 4 + ["test"] * 2)
        lengths = np.array(
            [64, 16100, 22000, 24064, 128, 16000, 22000, 2048, 8192, 23000]
        )
        censored = np.array([False] * 3 + [True] + [False] * 6)
        data = dict(
            hidden=np.random.default_rng(7).normal(size=(10, 2)).astype(np.float16),
            generated=np.zeros(10, dtype=np.float32),
            remaining=lengths,
            requests=ids,
            splits=splits,
            censored=censored,
            phases=np.array(["PREFILL_COMPLETE"] * 10),
            languages=np.array(["en"] * 10),
            audit=dict(requests=10),
            labels=[
                dict(
                    input_id=ids[i],
                    source_tree_id=ids[i],
                    split=splits[i],
                    natural_finish=not bool(censored[i]),
                    output_tokens=int(lengths[i]),
                )
                for i in range(10)
            ],
        )
        meta = dict(
            max_tokens=24064,
            max_model_len=24576,
            revision="capture-test",
            input_sha256="test-input",
            model_config_sha256="test-qwen3-base",
            feature_layer="decoder:31",
            generation_contract=dict(thinking_mode="disabled"),
        )
        return data, meta

    def test_monotone_widths_and_24k_open_tail(self):
        edges = nonuniform24k100_upper_edges()
        self.assertEqual(len(edges) + 1, 100)
        frozen = json.loads(
            (ROOT / "experiments/platforms/qwen3_nonuniform24k100.json").read_text()
        )
        self.assertEqual(edges.tolist(), frozen["finite_upper_edges"])
        self.assertTrue(np.all(np.diff(edges) > 0))
        self.assertTrue(np.all(np.diff(np.diff(edges)) >= 0))
        np.testing.assert_array_equal(edges[:75], risk76_upper_edges())
        for lower, upper, width in (
            (8192, 16384, 512),
            (16384, 24576, 1024),
        ):
            segment = edges[(edges >= lower) & (edges <= upper)]
            np.testing.assert_array_equal(
                np.diff(segment), np.full(len(segment) - 1, width)
            )
        self.assertEqual(
            category_targets(np.array([24576, 24577]), edges).tolist(), [98, 99]
        )
        for value in [0, 8, 32, 512, 2048, 4096, 8192, 12288, 16384, 18432, 24576]:
            self.assertEqual(
                edges[category_targets(np.array([value]), edges)[0]], value
            )

    def test_probability_bounds_do_not_invent_within_bucket_precision(self):
        edges = nonuniform24k100_upper_edges()
        p = np.zeros((1, 100))
        p[0, np.searchsorted(edges, 16384)] = 1
        low, high = probability_gt_bounds(p, edges, 16300)
        self.assertEqual((low.item(), high.item()), (0, 1))
        low, high = probability_gt_bounds(p, edges, 16384)
        self.assertEqual((low.item(), high.item()), (0, 0))
        p[:] = 0
        p[0, -1] = 1
        low, high = probability_gt_bounds(p, edges, 24576)
        self.assertEqual((low.item(), high.item()), (1, 1))

    def test_new_head_cannot_inherit_old_model_weights(self):
        data, meta = self.fixture()
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(ValueError, "must train from scratch"),
        ):
            fit_distribution(
                data,
                meta,
                Path(temporary),
                revision="test",
                device_name="cpu",
                bucket_profile="nonuniform24k100",
                warmstart_checkpoint=Path(temporary) / "old.pt",
                warmstart_sha256="not-used",
            )

    def test_scratch_100_output_checkpoint_loads_in_real_runtime(self):
        from tests.bridge_tp.test_distribution_predictor_runtime import runtime

        data, meta = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            report = fit_distribution(
                data,
                meta,
                out,
                revision="test-24k",
                device_name="cpu",
                epochs=2,
                hidden_width=4,
                network_depth=4,
                bucket_profile="nonuniform24k100",
            )
            path = out / "predictor_distribution.pt"
            saved = torch.load(path, weights_only=True)
            self.assertEqual(len(saved["category_upper_edges"]) + 1, 100)
            self.assertEqual(saved["category_upper_edges"][-1].item(), 24576)
            self.assertEqual(
                saved["warmstart_provenance"]["initialization"], "from scratch"
            )
            self.assertEqual(
                saved["warmstart_provenance"]["bucket_profile"], "nonuniform24k100"
            )
            self.assertAlmostEqual(saved["position_log_scale"], float(np.log1p(24064)))
            self.assertEqual(report["open_tail_lower_inclusive"], 24577)
            self.assertIn("20480", report["results"]["test"]["risk_by_horizon_tokens"])
            predictor = runtime.DistributionPredictor(
                path,
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                model_config_sha256=meta["model_config_sha256"],
            )
            actual = predictor.probabilities(
                torch.tensor(data["hidden"]), torch.zeros(10)
            ).numpy()
            with np.load(out / "distribution_predictions.npz") as predictions:
                self.assertEqual(actual.shape, (10, 100))
                np.testing.assert_allclose(
                    actual, predictions["probabilities"], atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
