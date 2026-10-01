"""Layer hooks, paired captures, and isolation from future test requests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))

from prepare_oasst1_predictor_inputs import (  # noqa: E402
    choose_layer_probe,
    choose_staged_pilot,
)
from run_predictor_layer_probe_a100 import (  # noqa: E402
    LAYERS,
    parse_probe_layers,
    rank_layers,
)

from tests.bridge_tp.test_predictor_capture import capture, runner  # noqa: E402


class TestPredictorLayerProbe(unittest.TestCase):
    def test_mlp_representation_tags_and_metadata_are_distinct(self):
        self.assertEqual(
            parse_probe_layers("decoder:31,mlp:31"), ("decoder:31", "mlp:31")
        )
        self.assertEqual(capture.decoder_layer_index("mlp:31"), 31)
        self.assertIn("MLP branch", runner.feature_semantics("mlp:31"))
        self.assertIn("hidden+residual", runner.feature_semantics("decoder:31"))
        for value in ("mlp:48,final", "mlp:-1,final", "mlp:31,mlp:31", "mlp:31"):
            with self.assertRaises(ValueError):
                parse_probe_layers(value)

    def test_mlp_hook_preserves_branch_before_subsequent_storage_mutation(self):
        import torch

        class Qwen2DecoderLayer(torch.nn.Module):
            def forward(self, x):
                return x * 2, torch.ones_like(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Qwen2DecoderLayer()])

            def forward(self, x):
                hidden, residual = self.layers[0](x)
                hidden.add_(100)
                residual.add_(1000)
                return hidden + residual

        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        with tempfile.TemporaryDirectory() as directory:
            observer = capture.PredictorFeatureCapture(Path(directory), 20, "mlp:0")
            model = Model()
            baseline = model(x)
            observer.attach_model(model)
            observer.begin_forward()
            final = model(x)
            torch.testing.assert_close(final, baseline)
            indices = torch.tensor([2, 0])
            observed = observer.sample_states(final[indices], indices)
            torch.testing.assert_close(observed, (x * 2)[indices])
            observer.database.close()

    def test_hook_copies_residual_stream_without_changing_output(self):
        import torch

        class Qwen2DecoderLayer(torch.nn.Module):
            def forward(self, x):
                return x * 2, torch.ones_like(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Qwen2DecoderLayer()])

            def forward(self, x):
                hidden, residual = self.layers[0](x)
                residual.add_(100)
                return hidden + residual

        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        with tempfile.TemporaryDirectory() as directory:
            observer = capture.PredictorFeatureCapture(Path(directory), 20, "decoder:0")
            model = Model()
            baseline = model(x)
            observer.attach_model(model)
            observer.begin_forward()
            final = model(x)
            torch.testing.assert_close(final, baseline)
            indices = torch.tensor([2, 0])
            observed = observer.sample_states(final[indices], indices)
            torch.testing.assert_close(observed, (x * 2 + 1)[indices])
            with self.assertRaisesRegex(RuntimeError, "did not run"):
                observer.sample_states(final[indices], indices)
            observer.database.close()

    def test_multi_layer_capture_pairs_request_and_token_positions(self):
        import torch

        class Qwen2DecoderLayer(torch.nn.Module):
            def forward(self, x):
                return x * 2, torch.ones_like(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Qwen2DecoderLayer()])

            def forward(self, x):
                a, b = self.layers[0](x)
                return (a + b) * 3

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observer = capture.MultiLayerFeatureCapture(
                root, 20, ["decoder:0", "final", "mlp:0"]
            )
            model = Model()
            observer.attach_model(model)
            observer.begin_forward()
            states = model(torch.ones(3, 4))
            indices = torch.tensor([2, 0])
            final = observer.sample_states(states[indices], indices)
            observer.capture(
                final, ["prefill", "decode"], [0, 20], [10, 30], [10, 1], [20, 10]
            )
            rows = []
            for writer in observer.writers:
                records = writer.database.execute(
                    "SELECT request_id, generated_tokens, hidden_fp16 FROM samples"
                ).fetchall()
                rows.append(records)
                writer.database.close()
            self.assertEqual(
                [(x[0], x[1]) for x in rows[0]], [("prefill", 0), ("decode", 20)]
            )
            self.assertEqual(
                [(x[0], x[1]) for x in rows[0]], [(x[0], x[1]) for x in rows[1]]
            )
            self.assertEqual(
                [(x[0], x[1]) for x in rows[0]], [(x[0], x[1]) for x in rows[2]]
            )
            np.testing.assert_array_equal(
                np.frombuffer(rows[0][0][2], dtype=np.float16), np.full(4, 3)
            )
            np.testing.assert_array_equal(
                np.frombuffer(rows[1][0][2], dtype=np.float16), np.full(4, 9)
            )
            np.testing.assert_array_equal(
                np.frombuffer(rows[2][0][2], dtype=np.float16), np.full(4, 2)
            )

    def test_probe_is_disjoint_from_stage_validation_and_test(self):
        roots = [
            {
                "id": f"{language}-{split}-{i}",
                "lang": language,
                "split": split,
                "source_tree_id": f"{language}-{split}-{i}",
            }
            for language, counts in (("en", (2000, 300, 300)), ("zh", (200, 30, 30)))
            for split, count in zip(("train", "validation", "test"), counts)
            for i in range(count)
        ]
        probe = choose_layer_probe(roots)
        stage = choose_staged_pilot(roots, 2000)
        self.assertEqual(len(probe), 300)
        self.assertEqual(
            [
                sum(x["split"] == s for x in probe)
                for s in ("train", "validation", "test")
            ],
            [180, 80, 40],
        )
        self.assertTrue(
            {x["id"] for x in probe}
            <= {x["id"] for x in stage if x["split"] == "train"}
        )

    def test_layer_ranking_uses_validation_not_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, layer in enumerate(LAYERS):
                out = root / ("trained-" + layer.replace(":", ""))
                out.mkdir()
                report = {
                    "selection_validation_request_ids": ["validation"],
                    "results": {"selection_validation": {"calibrated_nll": 1.0}},
                    "best_epoch": 1,
                }
                (out / "report.json").write_text(json.dumps(report))
                # Every validation length is zero. Give first layer perfect
                # predictions there but the worst possible held-out test result.
                pmf = np.array([[1 - i / 4, i / 4], [0, 1]])
                np.savez(
                    out / "distribution_predictions.npz",
                    request_ids=np.array(["validation", "test"]),
                    phases=np.array(["PREFILL_COMPLETE"] * 2),
                    censored=np.zeros(2, bool),
                    observed_remaining=np.zeros(2),
                    probabilities=pmf,
                    category_upper_edges=np.array([0]),
                )
            ranking = rank_layers(root)
            self.assertEqual(ranking[0]["feature_layer"], LAYERS[0])
            self.assertEqual(ranking[0]["mean_prefill_brier"], 0)

    def test_custom_branch_probe_reports_decode_validation(self):
        layers = ("decoder:31", "mlp:31")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for layer in layers:
                out = root / ("trained-" + layer.replace(":", ""))
                out.mkdir()
                (out / "report.json").write_text(
                    json.dumps(
                        {
                            "selection_validation_request_ids": ["validation"],
                            "results": {"selection_validation": {"calibrated_nll": 1}},
                            "best_epoch": 1,
                        }
                    )
                )
                np.savez(
                    out / "distribution_predictions.npz",
                    request_ids=np.array(["validation", "validation", "test"]),
                    phases=np.array(["PREFILL_COMPLETE", "DECODE", "DECODE"]),
                    censored=np.zeros(3, bool),
                    observed_remaining=np.zeros(3),
                    probabilities=np.array([[1, 0], [0.5, 0.5], [0, 1]]),
                    category_upper_edges=np.array([0]),
                )
            ranking = rank_layers(root, layers)
            self.assertEqual({r["feature_layer"] for r in ranking}, set(layers))
            self.assertEqual(ranking[0]["mean_decode_brier"], 0.25)


if __name__ == "__main__":
    unittest.main()
