"""Validate regularization search uses only the fixed model-selection split."""

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
from run_predictor_regularization_a100 import (  # noqa: E402
    BASELINE,
    Configuration,
    choose_challenger,
    ranking_key,
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
