"""Check source weighting, true-progress weighting and held-out isolation."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from predictor_training_mix import (  # noqa: E402
    capture_coverage,
    length_metrics,
    mixed_training_weights,
)
from run_predictor_nonuniform84_training import (  # noqa: E402
    resolve_collection,
    resolve_legacy,
)
from train_predictor_distribution import validation_groups  # noqa: E402


class TrainingMixTests(unittest.TestCase):
    def fixture(self):
        ids = np.array(["old"] * 4 + ["new"] * 4 + ["capped"] * 2 + ["val"])
        return {
            "requests": ids,
            "splits": np.array(["train"] * 10 + ["validation"]),
            "generated": np.array([0, 10, 50, 90] * 2 + [0, 50, 0]),
            "remaining": np.array([100, 90, 50, 10] * 2 + [100, 50, 1]),
            "censored": np.array([False] * 8 + [True, True, False]),
        }

    def test_source_mass_and_stage_mass_ignore_state_count(self):
        data = self.fixture()
        w, report = mixed_training_weights(data, {"old"})
        self.assertAlmostEqual(w[:4].sum(), 0.25)
        self.assertAlmostEqual(w[4:8].sum(), 0.375)
        self.assertAlmostEqual(w[8:10].sum(), 0.375)
        np.testing.assert_allclose(w[:4] / 0.25, [0.15, 0.15, 0.4, 0.3])
        self.assertEqual(w[10], 0)
        self.assertEqual(report["cohorts"]["legacy5000"]["training_requests"], 1)
        self.assertAlmostEqual(w.sum(), 1)

    def test_censored_progress_not_treated_as_true_final_progress(self):
        data = self.fixture()
        w, _ = mixed_training_weights(data, {"old"})
        np.testing.assert_allclose(w[8:10], [0.1875, 0.1875])
        data["censored"][8] = False
        with self.assertRaisesRegex(ValueError, "mixes exact"):
            mixed_training_weights(data, {"old"})

    def test_missing_stages_renormalize_without_dropping_request(self):
        data = self.fixture()
        data["generated"][:4] = 0
        w, _ = mixed_training_weights(data, {"old"})
        np.testing.assert_allclose(w[:4], np.full(4, 0.25 / 4))

    def test_old_validation_never_enters_model_selection_or_calibration(self):
        data = {
            "requests": np.array(["v0", "v1", "oldval"]),
            "labels": [
                {
                    "input_id": name,
                    "source_tree_id": name,
                    "split": "validation",
                    "natural_finish": True,
                }
                for name in ("v0", "v1", "oldval")
            ],
        }
        selection, calibration = validation_groups(data, {"v0", "v1"})
        self.assertFalse(selection[2] or calibration[2])
        self.assertEqual(int(selection.sum()), 1)
        self.assertEqual(int(calibration.sum()), 1)
        self.assertFalse((selection & calibration).any())

    def test_interval_open_tail_is_reported_instead_of_clipped(self):
        result = length_metrics(
            np.array([[0.01, 0.03, 0.96]]),
            np.array([0, 32]),
            np.array([100]),
            np.array(["r"]),
        )
        self.assertEqual(result["interval90_open_tail_weight"], 1)
        self.assertIsNone(result["interval90_mean_finite_width_tokens"])
        self.assertIsNone(result["median_bucket_midpoint_mae_finite_tokens"])
        self.assertEqual(result["interval90_coverage"], 1)
        self.assertAlmostEqual(result["true_bucket_mean_probability"], 0.96)

    def test_coverage_does_not_use_capped_length_as_exact_tail(self):
        data = {
            "labels": [
                {"split": "train", "natural_finish": False, "output_tokens": 15000},
                {"split": "train", "natural_finish": True, "output_tokens": 9000},
            ]
        }
        self.assertEqual(
            capture_coverage(data)["train"]["exact_output_length_counts"],
            {"8193-12288": 1},
        )

    def test_ambiguous_batch_requires_explicit_path(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("predictor-a", "predictor-b"):
                path = root / name
                (path / "inputs").mkdir(parents=True)
                (path / "inputs/manifest.json").write_text(
                    json.dumps({"recipe_sha256": "pin"})
                )
                (path / "collection_summary.json").write_text(
                    json.dumps({"requests": 300})
                )
            with self.assertRaisesRegex(ValueError, "expected one"):
                resolve_collection(root, None, "pin", 300)
            self.assertEqual(
                resolve_collection(root, str(root / "predictor-a"), "pin", 300),
                root / "predictor-a",
            )

    def test_legacy_archive_rejects_escape_and_symlink(self):
        import io
        import tarfile

        for name, kind in (
            ("capture-5000/../../escaped", tarfile.REGTYPE),
            ("capture-5000/labels.jsonl", tarfile.SYMTYPE),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "legacy.tar.gz"
                with tarfile.open(source, "w:gz") as archive:
                    info = tarfile.TarInfo(name)
                    info.type = kind
                    info.size = 1 if kind == tarfile.REGTYPE else 0
                    archive.addfile(info, io.BytesIO(b"x"))
                with self.assertRaisesRegex(ValueError, "unexpected legacy"):
                    resolve_legacy(root, root, str(source))
                self.assertFalse((root / "escaped").exists())


if __name__ == "__main__":
    unittest.main()
