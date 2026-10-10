"""Test paired feature diagnostics and actual-length coverage accounting."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/bridge_tp"))
from build_predictor_long_pilot import inputs
from compare_predictor_feature_paths import compare
from summarize_predictor_long_pilot import summarize


class ProbeTests(unittest.TestCase):
    def test_paired_features_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live, offline = root / "live", root / "offline"
            (live / "probes").mkdir(parents=True)
            (offline / "features").mkdir(parents=True)
            label = {
                "input_id": "a",
                "request_id": "0",
                "prompt_sha256": "p",
                "output_token_ids": [1, 2, 3],
            }
            for directory in [live, offline]:
                (directory / "labels.jsonl").write_text(json.dumps(label) + "\n")
            torch.save(
                {
                    "request_ids": ["0-abcdef12"],
                    "generated_tokens": torch.tensor([0]),
                    "hidden_fp16": torch.tensor([[1.0, 2.0]]).half(),
                },
                live / "probes/probe-1.pt",
            )
            feature = offline / "features/0.npz"

            def save(state):
                np.savez(
                    feature,
                    hidden_states=np.array([state], dtype=np.float16),
                    request_ids=np.array(["0-abcdef13"]),
                    generated_tokens=np.array([0]),
                    phase=np.array(["PREFILL_COMPLETE"]),
                )

            save([1, 2])
            self.assertEqual(compare(live, offline)["status"], "PASS")
            save([1, 4])
            self.assertEqual(compare(live, offline)["status"], "FEATURE_PATH_MISMATCH")
            label["output_token_ids"] = [1, 3, 3]
            (offline / "labels.jsonl").write_text(json.dumps(label) + "\n")
            with self.assertRaises(ValueError):
                compare(live, offline)

    def test_coverage_excludes_cap_from_exact_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "input_id": str(i),
                            "prompt_tokens": 120,
                            "output_tokens": length,
                            "finish_reason": "stop" if eos else "length",
                            "natural_finish": eos,
                        }
                    )
                    + "\n"
                    for i, (length, eos) in enumerate(
                        [(2000, True), (10000, True), (15000, True), (15872, False)]
                    )
                )
            )
            result = summarize(path)
            self.assertEqual(result["natural_length_histogram"]["12289..16384"], 1)
            self.assertEqual(result["censored_requests"], 1)

    def test_inputs_are_diagnostic_and_disjoint(self):
        rows = inputs()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["source_tree_id"] for r in rows}), 12)
        self.assertTrue(all(r["split"] == "validation" for r in rows))


if __name__ == "__main__":
    unittest.main()
