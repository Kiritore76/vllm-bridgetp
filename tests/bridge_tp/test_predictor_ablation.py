"""Fair width/bin comparisons, validation isolation and overflow preservation."""

import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from predictor_distribution import (  # noqa: E402
    category_targets,
    default_upper_edges,
    probability_remaining_gt,
)
from run_predictor_ablation_a100 import (  # noqa: E402
    BIN_STEPS,
    WIDTHS,
    choose_stages,
    run_ablation,
    summarize_fit,
)
from train_predictor_distribution import fit_distribution  # noqa: E402


def configuration_rows():
    rows = []
    for width in WIDTHS:
        for step in BIN_STEPS:
            score = {
                (256, 32): 0.14,
                (128, 32): 0.10,
                (64, 32): 0.12,
                (256, 64): 0.08,
                (128, 64): 0.09,
                (64, 64): 0.13,
            }[(width, step)]
            rows.append(
                {
                    "hidden_width": width,
                    "bin_step": step,
                    "selection_request_ids": ["selection"],
                    "calibration_request_ids": ["calibration"],
                    "risk": {
                        "selection_validation": {
                            "PREFILL_COMPLETE": {
                                "raw_model": {
                                    "mean_brier": score,
                                    "mean_binary_log_loss": 1,
                                },
                                "calibrated_model": {"mean_brier": 1 - score},
                            }
                        },
                        "test": {"mean_brier": 100 - score},
                    },
                    "categorical_nll_not_comparable_across_bin_steps": {"test": score},
                }
            )
    return rows


class TestPredictorAblation(unittest.TestCase):
    def test_failed_fit_is_recorded_as_incomplete(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch required for runner failure test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            (capture / "features").mkdir(parents=True)
            for name in (
                "preflight.json",
                "summary.json",
                "input_requests.jsonl",
                "labels.jsonl",
                "sample_index.jsonl",
                "features/features.sqlite3",
            ):
                (capture / name).write_text("fixture")
            baseline = root / "baseline.json"
            baseline.write_text("fixture")
            args = argparse.Namespace(
                run_dir=capture,
                baseline_report=baseline,
                source=root / "raw",
                model=root / "model",
                out_dir=root / "out",
            )
            with (
                patch(
                    "run_predictor_ablation_a100.validate_capture",
                    return_value=({}, {"audit": {}}),
                ),
                patch("torch.cuda.is_available", return_value=True),
                patch(
                    "run_predictor_ablation_a100.fit_distribution",
                    side_effect=RuntimeError("failed fit"),
                ),
                self.assertRaisesRegex(RuntimeError, "failed fit"),
            ):
                run_ablation(args, "revision", ["GPU"])
            status = json.loads((args.out_dir / "status.json").read_text())
            self.assertFalse(status["completed"])
            self.assertEqual(status["completed_fits"], 0)
            self.assertFalse((args.out_dir / "comparison.json").exists())

    def test_stage_order_and_no_test_or_cross_bin_nll_selection(self):
        rows = configuration_rows()
        result = choose_stages(rows)
        self.assertEqual(
            result["selected_configuration"], {"hidden_width": 128, "bin_step": 64}
        )
        # Globally lowest Brier has width256/bin64; the requested staged rule
        # must first select width128 using only the bin32 fits.
        self.assertEqual(result["all_six_selection_rankings"][0]["hidden_width"], 256)
        changed = copy.deepcopy(rows)
        for row in changed:
            row["risk"]["test"] = {"mean_brier": 0}
            row["categorical_nll_not_comparable_across_bin_steps"] = {"test": -999}
        self.assertEqual(
            choose_stages(changed)["selected_configuration"],
            result["selected_configuration"],
        )

    def test_changed_validation_partitions_or_duplicate_fits_fail(self):
        rows = configuration_rows()
        rows[0]["selection_request_ids"] = ["another"]
        with self.assertRaisesRegex(ValueError, "partitions differ"):
            choose_stages(rows)
        rows = configuration_rows()
        rows[-1] = rows[0]
        with self.assertRaisesRegex(ValueError, "six unique"):
            choose_stages(rows)
        rows = configuration_rows()
        for row in rows:
            row["calibration_request_ids"] = ["selection"]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            choose_stages(rows)

    def test_bins_keep_short_boundaries_and_open_tail(self):
        for step, classes in ((32, 132), (64, 68)):
            edges = default_upper_edges(4096, step)
            self.assertEqual(len(edges) + 1, classes)
            self.assertEqual(edges[:3].tolist(), [0, 8, 16])
            self.assertEqual(edges[-1], 4096)
            targets = category_targets(np.array([4096, 4097, 10000]), edges)
            self.assertEqual(targets.tolist(), [len(edges) - 1, len(edges), len(edges)])
            p = np.zeros((1, classes))
            p[0, -1] = 1
            self.assertEqual(probability_remaining_gt(p, edges, 8192)[0], 1)

    def test_cpu_training_reports_phase_risk_and_configuration(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch required for checkpoint smoke")
        splits = np.array(["train"] * 4 + ["validation"] * 4 + ["test"] * 2)
        ids = np.array([f"r{i}" for i in range(10)])
        remaining = np.array([8, 16, 32, 64, 16, 32, 8, 64, 16, 32])
        labels = [
            {
                "input_id": ids[i],
                "source_tree_id": f"tree{i}",
                "split": splits[i],
                "natural_finish": i != 3,
                "output_tokens": int(remaining[i]),
            }
            for i in range(10)
        ]
        data = {
            "hidden": np.random.default_rng(42).normal(size=(10, 4)).astype(np.float16),
            "generated": np.zeros(10),
            "remaining": remaining,
            "requests": ids,
            "splits": splits,
            "censored": np.arange(10) == 3,
            "phases": np.array(["PREFILL_COMPLETE"] * 10),
            "languages": np.array(["en"] * 10),
            "labels": labels,
            "audit": {"requests": 10},
        }
        meta = {
            "max_tokens": 512,
            "revision": "capture",
            "input_sha256": "input",
            "model_config_sha256": "model",
            "feature_layer": "decoder:31",
        }
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for step in BIN_STEPS:
                out = Path(directory) / str(step)
                report = fit_distribution(
                    data,
                    meta,
                    out,
                    revision="train",
                    device_name="cpu",
                    epochs=1,
                    batch_size=4,
                    hidden_width=8,
                    bin_step=step,
                )
                summary = summarize_fit(report)
                self.assertEqual(summary["hidden_width"], 8)
                self.assertEqual(summary["bin_step"], step)
                self.assertTrue(
                    np.isfinite(
                        summary["risk"]["selection_validation"]["PREFILL_COMPLETE"][
                            "raw_model"
                        ]["mean_brier"]
                    )
                )
                checkpoint = torch.load(
                    out / "predictor_distribution.pt", weights_only=True
                )
                self.assertTrue(checkpoint["overflow_category"])
                self.assertEqual(checkpoint["feature_layer"], "decoder:31")
                self.assertEqual(
                    len(checkpoint["category_upper_edges"]) + 1, summary["classes"]
                )
                self.assertEqual(report["censored_training_requests"], 1)
                outputs.append(summary)
            self.assertEqual(
                outputs[0]["selection_request_ids"], outputs[1]["selection_request_ids"]
            )
            self.assertEqual(
                outputs[0]["calibration_request_ids"],
                outputs[1]["calibration_request_ids"],
            )


if __name__ == "__main__":
    unittest.main()
