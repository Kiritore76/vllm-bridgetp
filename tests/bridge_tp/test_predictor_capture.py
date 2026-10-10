"""CPU-only checks for predictor capture selection and label integrity."""

import gzip
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))


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
oasst = load_module(
    "prepare_oasst1_for_test",
    ROOT / "tools/bridge_tp/prepare_oasst1_predictor_inputs.py",
)


class TestPredictorCapture(unittest.TestCase):
    def test_capture_writes_sqlite_rows_auditable_by_runner(self):
        class FakeStates:
            ndim = 2
            shape = (1, 4)

            def __getitem__(self, indices):
                assert indices == [0]
                return self

            def detach(self):
                return self

            def to(self, **kwargs):
                return self

            def numpy(self):
                return np.ones((1, 4), dtype=np.float16)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            writer = capture.PredictorFeatureCapture(directory, 20)
            original_torch = sys.modules.get("torch")
            sys.modules["torch"] = SimpleNamespace(float16=object())
            try:
                writer.capture(FakeStates(), ["0-abcdef12"], [0], [5], [5], [10])
            finally:
                if original_torch is None:
                    del sys.modules["torch"]
                else:
                    sys.modules["torch"] = original_torch
                writer.database.close()
            summary = runner.audit_capture(
                directory, {"0": {"output_tokens": 12, "natural_finish": True}}
            )
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["hidden_size"], 4)

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
            rows = [
                json.loads(line)
                for line in (directory / "index.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[0]["remaining_tokens"], 40)
            self.assertIsNone(rows[1]["remaining_tokens"])
            self.assertEqual(rows[1]["observed_remaining_lower_bound"], 12)

    def test_randomized_worker_id_joins_external_response_id(self):
        labels = {"0": {"input_id": "prompt-0"}}
        self.assertIs(
            runner.label_for_engine_request("0-ac718888", labels), labels["0"]
        )
        self.assertIsNone(runner.label_for_engine_request("0-invalid", labels))
        self.assertIsNone(runner.label_for_engine_request("1-ac718888", labels))

    def test_prompt_loader_rejects_duplicate_request_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            path.write_text(
                json.dumps({"id": "same", "prompt": "a"})
                + "\n"
                + json.dumps({"id": "same", "prompt": "b"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                runner.load_requests(path, None)

    def test_sqlite_feature_audit_preserves_split_and_exact_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with closing(sqlite3.connect(directory / "features.sqlite3")) as database:
                database.execute("""CREATE TABLE samples (
                    sample_id INTEGER PRIMARY KEY, request_id TEXT,
                    generated_tokens INTEGER, phase TEXT, hidden_size INTEGER,
                    hidden_fp16 BLOB, captured_unix_ns INTEGER)""")
                database.execute(
                    "INSERT INTO samples VALUES (1, ?, ?, ?, ?, ?, ?)",
                    (
                        "0-abcdef12",
                        0,
                        "PREFILL_COMPLETE",
                        4,
                        np.ones(4, dtype=np.float16).tobytes(),
                        1,
                    ),
                )
                database.commit()
            summary = runner.audit_capture(
                directory,
                {
                    "0": {
                        "input_id": "oasst1:a",
                        "split": "validation",
                        "lang": "zh",
                        "output_tokens": 12,
                        "natural_finish": True,
                    }
                },
                directory / "index.jsonl",
            )
            self.assertEqual(summary["samples"], 1)
            index = json.loads((directory / "index.jsonl").read_text())
            self.assertEqual(index["feature_file"], "features.sqlite3")
            self.assertEqual(index["feature_row"], 1)
            self.assertEqual(index["split"], "validation")
            self.assertEqual(index["remaining_tokens"], 12)

    def test_oasst_root_filter_and_tree_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl.gz"
            base = {
                "role": "prompter",
                "parent_id": None,
                "deleted": False,
                "synthetic": False,
                "lang": "en",
                "message_tree_id": "tree",
            }
            rows = [
                {**base, "message_id": "root", "text": "A full root question?"},
                {
                    **base,
                    "message_id": "reply",
                    "parent_id": "root",
                    "text": "A follow-up question?",
                },
                {
                    **base,
                    "message_id": "deleted",
                    "deleted": True,
                    "text": "A deleted question?",
                },
            ]
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            original_sha = oasst.SOURCE_SHA256
            try:
                oasst.SOURCE_SHA256 = oasst.sha256_file(source)
                roots = oasst.load_roots(source, 5, 100)
            finally:
                oasst.SOURCE_SHA256 = original_sha
            self.assertEqual(len(roots), 1)
            self.assertEqual(roots[0]["id"], "oasst1:root")
            self.assertEqual(roots[0]["split"], oasst.split_for_tree("tree"))


if __name__ == "__main__":
    unittest.main()
