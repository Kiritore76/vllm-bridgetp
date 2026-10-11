"""Verify independent prefill mass, decode balance and held-out isolation."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
import run_predictor_nonuniform84_training as runner  # noqa: E402
from predictor_distribution import nonuniform84_upper_edges  # noqa: E402
from predictor_training_mix import mixed_training_weights  # noqa: E402
from predictor_warmstart import rebuild_output_head  # noqa: E402
from train_predictor_distribution import fit_distribution  # noqa: E402

from tests.bridge_tp.test_predictor_nonuniform84 import parent_fixture  # noqa: E402


class PrefillWeightTests(unittest.TestCase):
    def fixture(self):
        return {
            "requests": np.array(["old"] * 4 + ["new"] * 4 + ["capped"] * 3 + ["val"]),
            "splits": np.array(["train"] * 11 + ["validation"]),
            "generated": np.array([0, 10, 50, 90] * 2 + [0, 50, 99, 0]),
            "remaining": np.array([100, 90, 50, 10] * 2 + [100, 50, 1, 100]),
            "censored": np.array([False] * 8 + [True] * 3 + [False]),
            "phases": np.array(
                ["PREFILL_COMPLETE", "DECODE", "DECODE", "DECODE"] * 2
                + ["PREFILL_COMPLETE", "DECODE", "DECODE", "PREFILL_COMPLETE"]
            ),
        }

    def test_prefill_has_reserved_mass_in_each_source_request(self):
        for mass in (0.1, 0.2):
            with self.subTest(mass=mass):
                data = self.fixture()
                weights, report = mixed_training_weights(
                    data, {"old"}, legacy_mass=0.5, prefill_mass=mass
                )
                np.testing.assert_allclose(
                    weights[:4] / 0.5,
                    [mass, (1 - mass) * 0.3, (1 - mass) * 0.4, (1 - mass) * 0.3],
                )
                self.assertAlmostEqual(weights[4:8].sum(), 0.25)
                self.assertAlmostEqual(weights[8:11].sum(), 0.25)
                self.assertAlmostEqual(weights[4] / 0.25, mass)
                self.assertEqual(weights[11], 0)
                self.assertAlmostEqual(weights.sum(), 1)
                self.assertAlmostEqual(
                    report["cohorts"]["new300_and_first60"]["prefill_loss_mass"],
                    0.5 * mass,
                )

    def test_more_decode_states_do_not_dilute_prefill_or_request(self):
        data = self.fixture()
        indices = np.r_[np.arange(12), np.full(40, 1), np.full(20, 5)]
        many = {key: value[indices] for key, value in data.items()}
        weights, _ = mixed_training_weights(
            many, {"old"}, legacy_mass=0.5, prefill_mass=0.2
        )
        self.assertAlmostEqual(weights[0], 0.1)
        self.assertAlmostEqual(weights[4], 0.05)
        self.assertAlmostEqual(weights[many["requests"] == "old"].sum(), 0.5)
        self.assertAlmostEqual(weights[many["requests"] == "new"].sum(), 0.25)
        early_old = (many["requests"] == "old") & (many["generated"] == 10)
        self.assertAlmostEqual(weights[early_old].sum(), 0.5 * 0.24)

    def test_censored_decode_remains_flat_despite_apparent_progress(self):
        weights, _ = mixed_training_weights(
            self.fixture(), {"old"}, legacy_mass=0.5, prefill_mass=0.2
        )
        np.testing.assert_allclose(weights[8:11], [0.05, 0.1, 0.1])

    def test_missing_decode_stages_redistribute_only_decode_mass(self):
        data = self.fixture()
        data["generated"][1:4] = [1, 2, 3]
        data["remaining"][1:4] = [99, 98, 97]
        weights, _ = mixed_training_weights(
            data, {"old"}, legacy_mass=0.5, prefill_mass=0.2
        )
        np.testing.assert_allclose(weights[:4], [0.1, 0.4 / 3, 0.4 / 3, 0.4 / 3])

    def test_prefill_only_request_keeps_full_mass(self):
        data = self.fixture()
        indices = np.r_[0, np.arange(4, 12)]
        only = {key: value[indices] for key, value in data.items()}
        weights, report = mixed_training_weights(
            only, {"old"}, legacy_mass=0.5, prefill_mass=0.2
        )
        self.assertAlmostEqual(weights[0], 0.5)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertEqual(report["cohorts"]["legacy5000"]["prefill_only_requests"], 1)

    def test_zero_mass_preserves_previous_weights_without_phase_metadata(self):
        data = self.fixture()
        expected, _ = mixed_training_weights(data, {"old"}, legacy_mass=0.5)
        del data["phases"]
        actual, _ = mixed_training_weights(
            data, {"old"}, legacy_mass=0.5, prefill_mass=0
        )
        np.testing.assert_array_equal(actual, expected)

    def test_missing_duplicate_or_misplaced_prefill_is_rejected(self):
        for kind in ("missing", "duplicate", "misplaced", "zero_decode"):
            with self.subTest(kind=kind):
                data = self.fixture()
                if kind == "missing":
                    data["phases"][0] = "DECODE"
                elif kind == "duplicate":
                    data["phases"][1] = "PREFILL_COMPLETE"
                elif kind == "misplaced":
                    data["generated"][0] = 1
                else:
                    data["generated"][1] = 0
                with self.assertRaises(ValueError):
                    mixed_training_weights(data, {"old"}, prefill_mass=0.2)

    def test_phase_metadata_and_finite_mass_are_required(self):
        data = self.fixture()
        for mass in (-0.1, 1, float("nan"), float("inf")):
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                mixed_training_weights(data, {"old"}, prefill_mass=mass)
        del data["phases"]
        with self.assertRaisesRegex(ValueError, "phase per sample"):
            mixed_training_weights(data, {"old"}, prefill_mass=0.2)

    def test_comparison_runner_saves_both_scratch_fits_and_prefill_diagnostics(self):
        import torch

        parent = rebuild_output_head(parent_fixture(), nonuniform84_upper_edges(), 42)
        meta = {
            key: parent[key]
            for key in ("feature_layer", "feature_semantics", "model_config_sha256")
        }
        meta.update(max_tokens=16384, revision="capture", input_sha256="input")
        parts = []
        for cohort in ("new", "replay", "old"):
            names = [f"{cohort}{i}" for i in range(5)]
            splits = ["train", "train", "validation", "validation", "test"]
            parts.append(
                {
                    "hidden": np.random.default_rng(42).normal(size=(20, 2)),
                    "generated": np.tile([0, 10, 50, 90], 5),
                    "remaining": np.tile([100, 90, 50, 10], 5),
                    "requests": np.repeat(names, 4),
                    "splits": np.repeat(splits, 4),
                    "phases": np.tile(
                        ["PREFILL_COMPLETE", "DECODE", "DECODE", "DECODE"], 5
                    ),
                    "censored": np.zeros(20, dtype=bool),
                    "languages": np.full(20, "en"),
                    "audit": {"requests": 5},
                    "labels": [
                        {
                            "input_id": key,
                            "source_tree_id": key,
                            "split": split,
                            "natural_finish": True,
                            "output_tokens": 100,
                        }
                        for key, split in zip(names, splits)
                    ],
                }
            )

        def cpu_fit(*args, **kwargs):
            return fit_distribution(*args, device_name="cpu", **kwargs)

        def command(args, **kwargs):
            if args == ["git", "rev-parse", "HEAD"]:
                return "prefill-runner-test"
            if args == ["git", "status", "--porcelain"]:
                return ""
            if args[1] == "--query-gpu=name":
                return "NVIDIA A100-PCIE-40GB\n"
            return "index,name,uuid,memory.total\n0,NVIDIA A100-PCIE-40GB,record,40960"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            baseline.mkdir()
            checkpoint = baseline / "predictor_distribution.pt"
            torch.save(parent, checkpoint)
            (baseline / "report.json").write_text("{}")
            sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("config")
            config_sha = hashlib.sha256(b"config").hexdigest()
            legacy = root / "legacy"
            legacy.mkdir()
            (legacy / "preflight.json").write_text(json.dumps(meta))
            argv = [
                "training",
                "--results",
                str(root),
                "--out-dir",
                str(root / "output"),
                "--model",
                str(model),
                "--expected-revision",
                "prefill-runner-test",
                "--expected-gpu-count",
                "1",
                "--from-scratch",
                "--reference-checkpoint",
                str(checkpoint),
                "--reference-sha256",
                sha,
                "--network-depth",
                "4",
                "--prefill-comparison-trial",
                "--legacy-loss-mass",
                "0.5",
                "--balanced-validation",
                "--epochs",
                "2",
                "--patience",
                "1",
            ]
            # Exercise the real runner, weighting, fitting, calibration and reports.
            # Replace only hardware/repository checks and large audited data loading.
            with (
                patch.object(sys, "argv", argv),
                patch.object(runner.subprocess, "check_output", side_effect=command),
                patch.object(runner, "CONFIG_SHA", config_sha),
                patch.object(runner, "resolve_collection", return_value=root),
                patch.object(runner, "resolve_legacy", return_value=legacy),
                patch.object(
                    runner,
                    "load_collection_examples",
                    side_effect=[
                        (parts[0], {**meta, "collection_provenance": {}}),
                        (parts[1], {**meta, "collection_provenance": {}}),
                    ],
                ),
                patch.object(runner, "read_jsonl", return_value=[]),
                patch.object(runner, "load_examples", return_value=parts[2]),
                patch.object(runner, "validate_legacy"),
                patch.object(runner, "fit_distribution", side_effect=cpu_fit),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "device_count", return_value=1),
                patch.object(torch.cuda, "empty_cache"),
            ):
                runner.main()
            status = json.loads((root / "output/status.json").read_text())
            self.assertTrue(status["completed"])
            self.assertFalse(status["deployed"])
            self.assertEqual(set(status["models"]), {"prefill10", "prefill20"})
            for percent in (10, 20):
                path = root / f"output/train-prefill{percent}pct/report.json"
                report = json.loads(path.read_text())
                self.assertIsNone(report["warmstart_provenance"]["parent_sha256"])
                self.assertEqual(
                    report["training_weighting"]["prefill_mass_per_request"],
                    percent / 100,
                )
                self.assertIn("new300_test_prefill_complete", report["results"])
                self.assertIn("new300_validation_early_decode", report["results"])
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), sha)


if __name__ == "__main__":
    unittest.main()
