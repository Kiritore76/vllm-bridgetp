"""CPU checks for the read-only M5 event contract and risk bounds."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m5 = load("manager_m5_for_test", "vllm/bridge_tp/controller/manager_m5.py")
capture = load("capture_m5_for_test", "vllm/bridge_tp/predictor_capture.py")
SHA = "a" * 64
EDGES = [0, 8, 16]


class TestM5(unittest.TestCase):
    def test_bounds_include_unresolved_bucket(self):
        probabilities = (0.1, 0.2, 0.3, 0.4)
        edges = tuple(EDGES)
        self.assertEqual(m5.probability_gt_bounds(probabilities, edges, 0), (0.9, 0.9))
        self.assertAlmostEqual(m5.probability_gt_bounds(probabilities, edges, 4)[0], 0.7)
        self.assertAlmostEqual(m5.probability_gt_bounds(probabilities, edges, 4)[1], 0.9)
        self.assertAlmostEqual(m5.probability_gt_bounds(probabilities, edges, 20)[1], 0.4)

    def test_reader_incremental_partial_stale_and_wrong_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            header = {
                "kind": "predictor_header", "format_version": 1,
                "checkpoint_sha256": SHA, "feature_layer": "decoder:31",
                "interval": 20, "category_upper_edges": EDGES,
            }
            event = {
                "kind": "predictor_prediction", "request_id": "r", "generated_tokens": 20,
                "checkpoint_sha256": SHA,
                "captured_unix_ns": time.time_ns(),
                "probabilities": [0.1, 0.2, 0.3, 0.4],
            }
            with path.open("wb") as sink:
                sink.write((json.dumps(header) + "\n").encode())
                sink.write(json.dumps(event).encode())
            reader = m5.PredictorEventReader(path, SHA)
            self.assertEqual(reader.advisory("r", 20, 4)["status"], "UNAVAILABLE")
            with path.open("ab") as sink:
                sink.write(b"\n")
            row = reader.advisory("r", 20, 4)
            self.assertEqual(row["status"], "AVAILABLE")
            self.assertAlmostEqual(row["p_remaining_gt_headroom_bounds"][1], 0.9)
            capped = reader.advisory(
                "r", 20, 40, max_output_tokens=32, ignore_eos=False
            )
            self.assertEqual(capped["p_remaining_gt_headroom_runtime_bounds"], (0.0, 0.0))
            forced = reader.advisory(
                "r", 20, 4, max_output_tokens=32, ignore_eos=True
            )
            self.assertEqual(forced["p_remaining_gt_headroom_runtime_bounds"], (1.0, 1.0))
            self.assertFalse(forced["model_applicable_to_runtime_stop_rule"])
            self.assertEqual(reader.advisory("r", 61, 4)["status"], "STALE")
            with self.assertRaisesRegex(ValueError, "header differs"):
                m5.PredictorEventReader(path, "b" * 64).poll()

    def test_live_observer_emits_cpu_event_for_prefill_and_decode(self):
        class FakePredictor:
            def __init__(self, *args, **kwargs):
                self.capture_input_sha256 = "input"
                self.checkpoint_sha256 = SHA
                self.category_upper_edges = torch.tensor(EDGES)

            def probabilities(self, hidden, counts, **kwargs):
                self_counts.append(tuple(counts.tolist()))
                return torch.tensor([[0.1, 0.2, 0.3, 0.4]] * len(counts))

        self_counts = []
        fake_module = types.SimpleNamespace(DistributionPredictor=FakePredictor)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            events = root / "events.jsonl"
            with patch.dict(sys.modules, {
                "vllm.bridge_tp.controller.distribution_predictor": fake_module
            }):
                observer = capture.PredictorLiveObserver(
                    root / "checkpoint.pt", SHA, events, model, torch.device("cpu")
                )
            hidden = torch.zeros((1, 5120))
            observer.capture(hidden, ["r"], [0], [10], [10], [20])
            observer.capture(hidden, ["r"], [20], [20], [1], [20])
            observer.close()
            lines = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual([line["kind"] for line in lines], [
                "predictor_header", "predictor_prediction", "predictor_prediction"
            ])
            self.assertEqual([line["phase"] for line in lines[1:]], [
                "PREFILL_COMPLETE", "DECODE"
            ])
            self.assertEqual(self_counts, [(0.0,), (20.0,)])
            self.assertEqual(
                lines[0]["model_config_sha256"],
                hashlib.sha256(b"{}").hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
