"""Protect pretrained probabilities and split boundaries during extension."""

import copy
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from predictor_warmstart import expand_tail, load_warmstart, merge_captures


def fixture():
    model = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(4, 5))
    return {
        "format_version": 2,
        "model_type": "remaining_length_categorical",
        "state_dict": model.state_dict(),
        "category_upper_edges": torch.tensor([0, 8, 16, 8192]),
        "temperature": 1.15,
        "feature_mean": torch.tensor([1.0, 2.0]),
        "feature_std": torch.tensor([2.0, 3.0]),
        "position_log_scale": math.log1p(8192),
        "input_width": 3,
        "hidden_width": 4,
        "overflow_category": True,
        "feature_layer": "decoder:31",
        "model_config_sha256": "base",
        "feature_semantics": "hidden+residual",
        "activation": "GELU",
        "dropout": 0.1,
        "capture_input_sha256": "old",
    }


class WarmstartTests(unittest.TestCase):
    def test_tail_preserves_old_calibrated_mass(self):
        old = fixture()
        new = expand_tail(old, [12288, 16384])
        self.assertEqual(
            new["category_upper_edges"].tolist(), [0, 8, 16, 8192, 12288, 16384]
        )
        features = torch.randn(20, 4)
        a = torch.softmax(
            (features @ old["state_dict"]["3.weight"].T + old["state_dict"]["3.bias"])
            / old["temperature"],
            1,
        )
        b = torch.softmax(
            (features @ new["state_dict"]["3.weight"].T + new["state_dict"]["3.bias"])
            / new["temperature"],
            1,
        )
        torch.testing.assert_close(a[:, :-1], b[:, :4])
        torch.testing.assert_close(a[:, -1], b[:, 4:].sum(1))
        self.assertEqual(old["state_dict"]["3.weight"].shape[0], 5)
        for key in ["feature_mean", "feature_std"]:
            torch.testing.assert_close(new[key], old[key])
        self.assertEqual(new["position_log_scale"], old["position_log_scale"])

    def test_invalid_tail_and_parent_rejected(self):
        old = fixture()
        for edges in [[8192], [16384, 12288], [12288, 12288]]:
            with self.assertRaises(ValueError):
                expand_tail(old, edges)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "parent.pt"
            torch.save(old, p)
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            meta = {
                k: old[k]
                for k in ["feature_layer", "feature_semantics", "model_config_sha256"]
            }
            self.assertEqual(load_warmstart(p, sha, meta, 2)["hidden_width"], 4)
            with self.assertRaises(ValueError):
                load_warmstart(p, "bad", meta, 2)
            meta["feature_layer"] = "final"
            with self.assertRaises(ValueError):
                load_warmstart(p, sha, meta, 2)

    def test_cpu_continued_training_preserves_feature_contract(self):
        from train_predictor_distribution import fit_distribution

        parent = fixture()
        parent.update(dropout=0.1, capture_input_sha256="old")
        ids = np.array([f"r{i}" for i in range(10)])
        split = np.array(["train"] * 4 + ["validation"] * 4 + ["test"] * 2)
        lengths = np.array([32, 64, 9000, 13000, 32, 96, 16000, 10000, 2000, 14000])
        data = {
            "hidden": np.random.default_rng(1).normal(size=(10, 2)).astype(np.float16),
            "generated": np.zeros(10),
            "remaining": lengths,
            "requests": ids,
            "splits": split,
            "censored": np.zeros(10, dtype=bool),
            "phases": np.array(["DECODE"] * 10),
            "languages": np.array(["en"] * 10),
            "audit": {"requests": 10},
            "labels": [
                {
                    "input_id": ids[i],
                    "source_tree_id": ids[i],
                    "split": split[i],
                    "natural_finish": True,
                    "output_tokens": int(lengths[i]),
                }
                for i in range(10)
            ],
        }
        meta = {
            k: parent[k]
            for k in ["model_config_sha256", "feature_layer", "feature_semantics"]
        }
        meta.update(max_tokens=16384, revision="capture", input_sha256="new")
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "seed.pt"
            torch.save(parent, cp)
            sha = hashlib.sha256(cp.read_bytes()).hexdigest()
            out = Path(tmp) / "new"
            fit_distribution(
                data,
                meta,
                out,
                revision="test",
                device_name="cpu",
                epochs=1,
                hidden_width=4,
                learning_rate=5e-5,
                warmstart_checkpoint=cp,
                warmstart_sha256=sha,
                tail_edges=[12288, 16384],
            )
            saved = torch.load(out / "predictor_distribution.pt", weights_only=True)
            for key in ["feature_mean", "feature_std"]:
                torch.testing.assert_close(saved[key], parent[key])
            self.assertEqual(saved["position_log_scale"], parent["position_log_scale"])
            self.assertEqual(
                saved["category_upper_edges"].tolist(), [0, 8, 16, 8192, 12288, 16384]
            )
            self.assertEqual(saved["warmstart_provenance"]["parent_sha256"], sha)
            self.assertEqual(hashlib.sha256(cp.read_bytes()).hexdigest(), sha)

    def test_capture_overlap_rejected(self):
        a = {
            "labels": [{"source_tree_id": "one", "split": "train", "input_id": "a"}],
            "audit": {},
        }
        with self.assertRaises(ValueError):
            merge_captures(a, copy.deepcopy(a))

    def test_replay_preserves_test_split(self):
        def data(prefix):
            return {
                **{
                    k: np.arange(2)
                    for k in [
                        "hidden",
                        "remaining",
                        "generated",
                        "requests",
                        "phases",
                        "languages",
                        "censored",
                    ]
                },
                "splits": np.array(["train", "test"]),
                "labels": [
                    {
                        "source_tree_id": prefix + str(i),
                        "input_id": prefix + str(i),
                        "split": split,
                        "natural_finish": True,
                    }
                    for i, split in enumerate(["train", "test"])
                ],
                "audit": {},
            }

        merged = merge_captures(data("new"), data("old"))
        self.assertEqual(merged["splits"].tolist(), ["train", "test"] * 2)
        self.assertEqual(merged["audit"]["requests"], 4)

    def test_probe_replay_rejects_corrupt_probabilities(self):
        from unittest.mock import patch
        from tests.bridge_tp.test_distribution_predictor_runtime import runtime
        from verify_live_predictor_probe import verify

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            cp = path / "parent.pt"
            torch.save(expand_tail(fixture(), [12288, 16384]), cp)
            sha = hashlib.sha256(cp.read_bytes()).hexdigest()
            predictor = runtime.DistributionPredictor(
                cp, checkpoint_sha256=sha, model_config_sha256="base"
            )
            hidden = torch.randn(3, 2).half()
            generated = torch.tensor([0, 128, 9000])
            probe = {
                "hidden_fp16": hidden,
                "generated_tokens": generated,
                "probabilities": predictor.probabilities(hidden, generated),
                "checkpoint_sha256": sha,
            }
            file = path / "probe-1.pt"
            torch.save(probe, file)
            with patch.dict(
                sys.modules,
                {"vllm.bridge_tp.controller.distribution_predictor": runtime},
            ):
                self.assertEqual(verify(path, cp, sha)["samples"], 3)
                probe["probabilities"][0, 0] = float("nan")
                torch.save(probe, file)
                with self.assertRaises(ValueError):
                    verify(path, cp, sha)


if __name__ == "__main__":
    unittest.main()
