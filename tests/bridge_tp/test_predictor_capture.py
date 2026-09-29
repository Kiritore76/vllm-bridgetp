"""CPU-only checks for predictor capture selection and label integrity."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


capture = load_module(
    "predictor_capture_for_test", ROOT / "vllm/bridge_tp/predictor_capture.py"
)
runner = load_module(
    "run_predictor_capture_for_test", ROOT / "tools/bridge_tp/run_predictor_capture.py"
)


class TestPredictorCapture(unittest.TestCase):
    def test_selects_prefill_and_periodic_decode_after_prompt_complete(self):
        rows = capture.select_capture_rows(
            ["partial", "prefill", "decode20", "decode21"],
            [0, 0, 20, 21],
            [4, 8, 10, 10],
            [2, 2, 1, 1],
            [8, 10, 10, 10],
            20,
        )
        self.assertEqual(
            [(r.request_id, r.generated_tokens, r.phase) for r in rows],
            [
                ("prefill", 0, "PREFILL_COMPLETE"),
                ("decode20", 20, "DECODE"),
            ],
        )

    def test_audit_counts_censored_labels_without_treating_them_as_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.savez(
                directory / "features.npz",
                hidden_states=np.ones((2, 4), dtype=np.float16),
                request_ids=np.array(["natural", "capped"]),
                generated_tokens=np.array([0, 20]),
                phase=np.array(["PREFILL_COMPLETE", "DECODE"]),
            )
            summary = runner.audit_capture(
                directory,
                {
                    "natural": {"output_tokens": 40, "natural_finish": True},
                    "capped": {"output_tokens": 32, "natural_finish": False},
                },
                directory / "index.jsonl",
            )
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["samples_with_exact_remaining_length"], 1)
            self.assertEqual(summary["censored_requests"], 1)
            rows = [json.loads(line) for line in
                    (directory / "index.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["remaining_tokens"], 40)
            self.assertIsNone(rows[1]["remaining_tokens"])
            self.assertEqual(rows[1]["observed_remaining_lower_bound"], 12)

    def test_prompt_loader_rejects_duplicate_request_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            path.write_text(
                json.dumps({"id": "same", "prompt": "a"}) + "\n"
                + json.dumps({"id": "same", "prompt": "b"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                runner.load_requests(path, None)


if __name__ == "__main__":
    unittest.main()
