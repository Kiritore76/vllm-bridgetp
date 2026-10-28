# SPDX-License-Identifier: Apache-2.0
"""GPU isolation, frozen splits, budgets and partial-run evidence contracts."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bridge_tp"))
import run_qwen3_long_four_gpu as runner  # noqa: E402
from run_predictor_capture import load_requests  # noqa: E402


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return messages[0]["content"]

    def encode(self, prompt):
        return list(range(len(prompt)))

    def decode(self, ids, **kwargs):
        return "decoded text"


class FourGPUCaptureTests(unittest.TestCase):
    def test_frozen_families_do_not_cross_old_or_new_splits(self):
        data = runner.rows(runner.DATA)
        manifest = runner.read(runner.DATA.with_suffix(".manifest.json"))
        self.assertEqual(len(data), 208)
        self.assertEqual(runner.sha(runner.DATA), manifest["input_sha256"])
        trees = {}
        for row in data:
            self.assertEqual(
                trees.setdefault(row["source_tree_id"], row["split"]), row["split"]
            )
            self.assertIsNone(row["local_output_length_label"])
        self.assertEqual(len(trees), 144)
        for name in (
            "qwen3_formal16k2000_20261027",
            "qwen3_recipe_pilot48_20261027",
            "qwen3_diverse_long12_20261027",
        ):
            previous = runner.rows(runner.DATA.with_name(name + ".jsonl"))
            self.assertFalse(set(trees) & {row["source_tree_id"] for row in previous})

    def test_worker_physical_inventory_and_cuda_visibility_are_distinct(self):
        with patch.dict(
            os.environ, {"BRIDGETP_PREDICTOR_CAPTURE_DIR": "wrong-shared-dir"}
        ):
            outputs = []
            for gpu in range(4):
                shard = dict(
                    name=f"shard{gpu}",
                    input=f"shard{gpu}.jsonl",
                    sha256="frozen",
                    gpu_index=gpu,
                )
                command, env = runner.worker_command(
                    Path("/model"), Path("/run"), shard, "revision"
                )
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], str(gpu))
                self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
                self.assertNotIn("BRIDGETP_PREDICTOR_CAPTURE_DIR", env)
                self.assertEqual(
                    command[command.index("--expected-gpu-count") + 1], "4"
                )
                self.assertEqual(
                    command[command.index("--expected-visible-gpu-count") + 1], "1"
                )
                outputs.append(command[command.index("--out-dir") + 1])
            self.assertEqual(len(set(outputs)), 4)

    def test_budget_exclusion_happens_before_generation_and_shards_are_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidates.jsonl"
            data = [
                dict(
                    id=f"r{i}",
                    source_tree_id=f"t{i}",
                    split="train",
                    lang="en",
                    task_type="code",
                    messages=[
                        dict(role="user", content="x" * (7000 if i == 8 else 100))
                    ],
                )
                for i in range(9)
            ]
            source.write_bytes(runner.payload(data))
            runner.write(
                source.with_suffix(".manifest.json"),
                dict(input_sha256=runner.sha(source)),
            )
            with patch.object(runner, "DATA", source):
                manifest = runner.prepare(Tokenizer(), root / "inputs")
            self.assertEqual(manifest["rows"], 8)
            self.assertEqual(len(manifest["excluded_before_generation"]), 1)
            identities = []
            for shard in manifest["shards"]:
                path = root / "inputs" / shard["input"]
                identities.extend(row["id"] for row in load_requests(path, None))
                self.assertEqual(runner.sha(path), shard["sha256"])
                self.assertEqual(shard["rows"], 2)
            self.assertEqual(len(set(identities)), 8)

    def test_partial_worker_failure_keeps_audited_successes_and_censor_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "inputs").mkdir()
            shards = [dict(name=f"s{i}", rows=1) for i in range(4)]
            requests = [
                dict(id=f"r{i}", task_type="code", split="train") for i in range(4)
            ]
            (root / "inputs/recipe.jsonl").write_bytes(runner.payload(requests))
            labels = []
            for i, n in enumerate((4097, 120, 10240)):
                directory = root / "captures" / f"s{i}"
                directory.mkdir(parents=True)
                runner.write(directory / "summary.json", {})
                labels.append(
                    dict(
                        input_id=f"r{i}",
                        output_tokens=n,
                        output_token_ids=[1] * n,
                        natural_finish=i != 2,
                        split="train",
                    )
                )

            def audit(_, shard, revision):
                return [labels[int(shard["name"][1:])]]

            with patch.object(runner, "audit_shard", side_effect=audit):
                summary = runner.summarize(
                    root, dict(shards=shards, rows=4), "rev", Tokenizer()
                )
            self.assertFalse(summary["all_requested_collected"])
            self.assertEqual(summary["requests"], 3)
            self.assertEqual(summary["natural_eos"], 2)
            self.assertEqual(summary["censored"], 1)
            self.assertEqual(summary["natural_4k_8k_by_split"], {"train": 1})
            self.assertEqual(summary["completed_shards"], ["s0", "s1", "s2"])

    def test_supervisor_starts_all_four_and_archives_partial_status(self):
        from tools.bridge_tp import run_phase9_cap0_calibration as common

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            shards = [
                dict(
                    name=f"s{i}",
                    input=f"s{i}.jsonl",
                    rows=1,
                    gpu_index=i,
                    sha256="test",
                )
                for i in range(4)
            ]

            def prepare(_, output):
                output.mkdir(parents=True)
                return dict(shards=shards, rows=4)

            def start(name, command, env, log_path):
                code = 1 if name == "gpu1" else 0
                return SimpleNamespace(
                    name=name,
                    process=SimpleNamespace(poll=lambda: code, returncode=code),
                    log_path=log_path,
                )

            def command_output(args, **kwargs):
                return "testrev" if args[:2] == ["git", "rev-parse"] else ""

            transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(
                    from_pretrained=lambda *a, **k: Tokenizer()
                )
            )
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "capture",
                        "--model",
                        "/model",
                        "--output",
                        str(root),
                        "--expected-revision",
                        "testrev",
                    ],
                ),
                patch.object(runner, "ASSET_SHA", {}),
                patch.object(runner, "prepare", side_effect=prepare),
                patch.object(
                    runner.subprocess, "check_output", side_effect=command_output
                ),
                patch.object(
                    runner,
                    "runtime_inventory",
                    return_value=dict(gpus=[dict(index=str(i)) for i in range(4)]),
                ),
                patch.object(
                    runner,
                    "summarize",
                    return_value=dict(all_requested_collected=False),
                ),
                patch.object(runner.time, "sleep"),
                patch.object(runner.signal, "signal"),
                patch.dict(sys.modules, {"transformers": transformers}),
                patch.object(common, "start_process", side_effect=start) as started,
                patch.object(common, "stop_processes"),
                patch.object(common, "process_group_alive", return_value=False),
            ):
                code = runner.main()
            self.assertEqual(code, 1)
            self.assertEqual(started.call_count, 4)
            self.assertEqual(
                runner.read(root / "result.json")["status"], "PARTIAL_CAPTURE"
            )
            self.assertTrue(Path(str(root) + ".tar.gz").is_file())
            self.assertTrue(Path(str(root) + ".tar.gz.sha256").is_file())


if __name__ == "__main__":
    unittest.main()
