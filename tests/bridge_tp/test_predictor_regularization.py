"""Validate regularization search uses only the fixed model-selection split."""

import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from run_predictor_regularization_a100 import (  # noqa: E402
    BASELINE,
    INITIAL_LR,
    Configuration,
    choose_challenger,
    ranking_key,
    run_regularization,
    summarize_finalists,
)


def row(config: Configuration, seed: int, prefill: float, decode: float) -> dict:
    return {
        "configuration_name": config.name,
        "configuration": config.as_dict(),
        "seed": seed,
        "best_epoch": 2,
        "selection": {
            "PREFILL_COMPLETE": {
                "mean_brier": prefill,
                "mean_binary_log_loss": 0.5,
            },
            "DECODE": {"mean_brier": decode},
        },
        "test": {"mean_brier": 1 - prefill},
    }


class TestPredictorRegularization(unittest.TestCase):
    def test_full_schedule_preserves_capture_and_matches_three_seeds(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch required for runner preflight")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            (capture / "features").mkdir(parents=True)
            source_names = (
                "preflight.json",
                "summary.json",
                "input_requests.jsonl",
                "labels.jsonl",
                "sample_index.jsonl",
                "features/features.sqlite3",
            )
            for name in source_names:
                (capture / name).write_text("immutable fixture")
            baseline_report = root / "baseline.json"
            baseline_report.write_text("baseline fixture")
            args = argparse.Namespace(
                run_dir=capture,
                baseline_report=baseline_report,
                source=root / "raw",
                model=root / "model",
                out_dir=root / "result",
            )
            calls = []

            def fake_fit(
                data,
                meta,
                out_dir,
                *,
                dropout,
                weight_decay,
                learning_rate,
                seed,
                hidden_width,
                bin_step,
                **unused,
            ):
                self.assertEqual((hidden_width, bin_step), (256, 32))
                out_dir.mkdir()
                calls.append((dropout, weight_decay, learning_rate, seed))
                score = 0.08 - dropout * 0.02 + weight_decay * 0.01
                score += (seed - 42) * 0.0001
                if learning_rate < INITIAL_LR:
                    score += 0.0005
                risk = {
                    str(h): {
                        "raw_model": {
                            "brier": score,
                            "binary_log_loss": 0.5,
                            "ece": 0.1,
                        }
                    }
                    for h in (128, 256, 512)
                }
                return {
                    "training_parameters": {
                        "dropout": dropout,
                        "weight_decay": weight_decay,
                        "learning_rate": learning_rate,
                        "hidden_width": hidden_width,
                        "bin_step": bin_step,
                    },
                    "validation_stratified": {
                        "selection_validation": {
                            "PREFILL_COMPLETE": risk,
                            "DECODE": risk,
                        },
                    },
                    "selection_validation_request_ids": ["selection"],
                    "calibration_validation_request_ids": ["calibration"],
                    "history": [
                        {"epoch": 1, "train_nll": 2.0, "selection_validation_nll": 2.1}
                    ],
                    "best_epoch": 1,
                    "temperature": 1.0,
                    "checkpoint_sha256": "fake-checkpoint-sha",
                }

            with (
                patch(
                    "run_predictor_regularization_a100.validate_capture",
                    return_value=({"input_sha256": "fixture"}, {"audit": {}}),
                ),
                patch(
                    "run_predictor_regularization_a100.fit_distribution",
                    side_effect=fake_fit,
                ),
                patch("torch.cuda.is_available", return_value=True),
            ):
                run_regularization(args, "revision", ["GPU"])
            status = json.loads((args.out_dir / "status.json").read_text())
            comparison = json.loads((args.out_dir / "comparison.json").read_text())
            self.assertTrue(status["completed"])
            self.assertEqual((status["completed_fits"], len(calls)), (12, 12))
            self.assertEqual(calls[0], (*BASELINE, 42))
            self.assertEqual([call[2] for call in calls[:6]], [INITIAL_LR] * 6)
            self.assertEqual([call[2] for call in calls[6:8]], [0.0001] * 2)
            self.assertEqual([call[3] for call in calls[8:]], [43, 44, 43, 44])
            self.assertEqual(
                comparison["finalists"][comparison["baseline"]]["seeds"], [42, 43, 44]
            )
            self.assertEqual(
                (capture / "labels.jsonl").read_text(), "immutable fixture"
            )

    def test_challenger_selection_ignores_test_and_calibration(self):
        baseline = Configuration(*BASELINE)
        a = Configuration(0.2, 0.01, 0.0003)
        b = Configuration(0.3, 0.05, 0.0001)
        rows = [
            row(baseline, 42, 0.07, 0.08),
            row(a, 42, 0.06, 0.09),
            row(b, 42, 0.065, 0.07),
        ]
        self.assertEqual(choose_challenger(rows)["configuration"], a.as_dict())
        changed = copy.deepcopy(rows)
        for item in changed:
            item["test"] = {"mean_brier": -100}
            item["calibration"] = {"mean_brier": -100}
        self.assertEqual(choose_challenger(changed)["configuration"], a.as_dict())
        self.assertLess(ranking_key(rows[1]), ranking_key(rows[2]))

    def test_three_seed_comparison_and_missing_seed_guard(self):
        baseline = Configuration(*BASELINE)
        candidate = Configuration(0.2, 0.05, 0.0001)
        rows = [row(baseline, seed, 0.08 + seed / 10000, 0.09) for seed in (42, 43, 44)]
        rows += [
            row(candidate, seed, 0.07 + seed / 10000, 0.085) for seed in (42, 43, 44)
        ]
        summary = summarize_finalists(rows, baseline, candidate)
        for difference in summary["challenger_minus_baseline_brier_by_seed"]["prefill"]:
            self.assertAlmostEqual(difference, -0.01)
        self.assertEqual(summary["baseline"], baseline.name)
        with self.assertRaisesRegex(ValueError, "three matched seeds"):
            summarize_finalists(rows[:-1], baseline, candidate)


if __name__ == "__main__":
    unittest.main()
