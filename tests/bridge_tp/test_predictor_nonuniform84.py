"""Verify inherited features, staged training and comparable coarse metrics."""

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
from predictor_distribution import (  # noqa: E402
    aggregate_probabilities,
    default_upper_edges,
    nonuniform84_upper_edges,
    probability_remaining_gt,
)
from predictor_warmstart import (  # noqa: E402
    configure_training_phase,
    rebuild_output_head,
)
from train_predictor_distribution import fit_distribution  # noqa: E402

from tests.bridge_tp.test_predictor_warmstart import fixture  # noqa: E402


def parent_fixture():
    """Use the real parent's bucket geometry with a tiny feature backbone."""
    parent = fixture()
    edges = default_upper_edges(8192)
    head = nn.Linear(4, len(edges) + 1)
    parent["category_upper_edges"] = torch.from_numpy(edges)
    parent["state_dict"]["3.weight"] = head.weight.detach().clone()
    parent["state_dict"]["3.bias"] = head.bias.detach().clone()
    return parent


class Nonuniform84Tests(unittest.TestCase):
    def test_rebuild_preserves_backbone_and_parent_without_rng_side_effects(self):
        parent = parent_fixture()
        edges = nonuniform84_upper_edges()
        before_rng = torch.get_rng_state().clone()
        new = rebuild_output_head(parent, edges, seed=88)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(new["state_dict"]["3.weight"].shape, (84, 4))
        self.assertEqual(parent["state_dict"]["3.weight"].shape, (260, 4))
        for key in ("0.weight", "0.bias"):
            self.assertTrue(
                torch.equal(new["state_dict"][key], parent["state_dict"][key])
            )
        for key in ("feature_mean", "feature_std"):
            self.assertTrue(torch.equal(new[key], parent[key]))
        self.assertEqual(new["position_log_scale"], parent["position_log_scale"])
        self.assertEqual(new["temperature"], 1.0)
        again = rebuild_output_head(parent, edges, seed=88)
        torch.testing.assert_close(
            new["state_dict"]["3.weight"],
            again["state_dict"]["3.weight"],
            rtol=0,
            atol=0,
        )

    def test_frozen_backbone_then_joint_updates(self):
        model = nn.Sequential(
            nn.Linear(3, 4), nn.GELU(), nn.Dropout(0), nn.Linear(4, 84)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        x = torch.randn(8, 3)
        targets = torch.arange(8)
        first = model[0].weight.detach().clone()
        last = model[3].weight.detach().clone()

        def step():
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(x), targets).backward()
            optimizer.step()

        configure_training_phase(
            model,
            optimizer,
            head_only=True,
            head_learning_rate=3e-4,
            learning_rate=5e-5,
        )
        step()
        self.assertTrue(torch.equal(first, model[0].weight))
        self.assertFalse(torch.equal(last, model[3].weight))
        configure_training_phase(
            model,
            optimizer,
            head_only=False,
            head_learning_rate=3e-4,
            learning_rate=5e-5,
        )
        step()
        self.assertFalse(torch.equal(first, model[0].weight))
        self.assertEqual(optimizer.param_groups[0]["lr"], 5e-5)

    def test_exact_aggregation_preserves_shared_thresholds(self):
        old_edges = default_upper_edges(8192)
        common = nonuniform84_upper_edges()
        common = common[common <= 8192]
        p = np.random.default_rng(1).dirichlet(np.ones(260), size=3)
        merged = aggregate_probabilities(p, old_edges, common)
        self.assertEqual(merged.shape, (3, 76))
        np.testing.assert_allclose(merged.sum(1), 1)
        for horizon in (32, 512, 2048, 8192):
            np.testing.assert_allclose(
                probability_remaining_gt(p, old_edges, horizon),
                probability_remaining_gt(merged, common, horizon),
            )
        with self.assertRaises(ValueError):
            aggregate_probabilities(p, old_edges, nonuniform84_upper_edges())

    def test_cpu_two_phase_training_and_runtime84(self):
        parent = parent_fixture()
        ids = np.array([f"r{i}" for i in range(10)])
        splits = np.array(["train"] * 4 + ["validation"] * 4 + ["test"] * 2)
        lengths = np.array([32, 64, 9000, 13000, 32, 96, 16000, 10000, 2000, 14000])
        data = {
            "hidden": np.random.default_rng(1).normal(size=(10, 2)).astype(np.float16),
            "generated": np.zeros(10),
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
            k: parent[k]
            for k in ("model_config_sha256", "feature_layer", "feature_semantics")
        }
        meta.update(max_tokens=16384, revision="capture", input_sha256="new")
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "seed.pt"
            torch.save(parent, cp)
            sha = hashlib.sha256(cp.read_bytes()).hexdigest()
            out = Path(tmp) / "new"
            report = fit_distribution(
                data,
                meta,
                out,
                revision="test",
                device_name="cpu",
                epochs=2,
                hidden_width=4,
                learning_rate=5e-5,
                warmstart_checkpoint=cp,
                warmstart_sha256=sha,
                bucket_profile="nonuniform84",
                head_warmup_epochs=1,
            )
            self.assertEqual(
                [r["phase"] for r in report["history"]], ["head_only", "joint"]
            )
            self.assertEqual(report["common_bucket_comparison"]["categories"], 76)
            saved = torch.load(out / "predictor_distribution.pt", weights_only=True)
            self.assertEqual(saved["state_dict"]["3.weight"].shape, (84, 4))
            for key in ("feature_mean", "feature_std"):
                torch.testing.assert_close(saved[key], parent[key], rtol=0, atol=0)
            self.assertEqual(hashlib.sha256(cp.read_bytes()).hexdigest(), sha)
            from tests.bridge_tp.test_distribution_predictor_runtime import runtime

            predictor = runtime.DistributionPredictor(
                out / "predictor_distribution.pt",
                checkpoint_sha256=report["checkpoint_sha256"],
                model_config_sha256="base",
            )
            p = predictor.probabilities(torch.tensor(data["hidden"]), torch.zeros(10))
            self.assertEqual(p.shape, (10, 84))
            torch.testing.assert_close(p.sum(1), torch.ones(10))


if __name__ == "__main__":
    unittest.main()
