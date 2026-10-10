"""Test paired feature diagnostics and actual-length coverage accounting."""

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/bridge_tp"))
from build_predictor_long_pilot import inputs
from compare_predictor_feature_paths import compare
from summarize_predictor_long_pilot import summarize
from run_predictor_live_feature_probe import finish_probe


class ProbeTests(unittest.TestCase):
    def test_named_rpc_routes_to_drain_without_function_serialization(self):
        from tests.bridge_tp.test_predictor_capture import capture

        # Isolate the GPU worker method for CPU testing; run its actual body.
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "vllm/v1/worker/gpu_worker.py").read_text())
        worker = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Worker"
        )
        method = next(
            n
            for n in worker.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "bridge_tp_drain_predictor_diagnostics"
        )
        namespace = {}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(root), "exec"),
            namespace,
        )
        observer = capture.PredictorLiveObserver.__new__(capture.PredictorLiveObserver)
        observer._diagnostic_dir = Path("diagnostic")
        observer._diagnostic_count = 40
        observer.close = Mock()
        fake_worker = SimpleNamespace(
            model_runner=SimpleNamespace(predictor_feature_capture=observer)
        )
        drained = namespace[method.name]

        def rpc(name, timeout):
            # JSON has the same primitive-only property needed by default RPC.
            name, timeout = json.loads(json.dumps([name, timeout]))
            self.assertEqual(name, method.name)
            return [drained(fake_worker)]

        shutdown = Mock()
        llm = SimpleNamespace(
            collective_rpc=rpc,
            llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown)),
        )
        with patch.dict(sys.modules, {"vllm.bridge_tp.predictor_capture": capture}):
            finish_probe(llm)
            observer.close.assert_called_once()
            shutdown.assert_called_once()
            observer._diagnostic_dir = None
            with self.assertRaises(RuntimeError):
                drained(fake_worker)

    def test_rpc_failure_still_shuts_down_engine(self):
        shutdown = Mock()
        llm = SimpleNamespace(
            collective_rpc=Mock(side_effect=RuntimeError("worker failed")),
            llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown)),
        )
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            finish_probe(llm)
        shutdown.assert_called_once()

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
