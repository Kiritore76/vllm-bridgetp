# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the new platform's measured service contract."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bridge_tp import run_pressure_calibration as collect
from tools.bridge_tp import run_qwen3_tp2_calibration as runner


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "<user>" + messages[0]["content"] + "</user><assistant><think></think>"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class ServiceCalibrationTests(unittest.TestCase):
    def test_new_input_lengths_and_role_context_limits(self):
        inputs = runner.prepare_inputs(Tokenizer())
        runner.validate_inputs(inputs)
        self.assertEqual(len(runner.matrix()) * 2, 10)
        self.assertEqual(len(inputs["requests"]["6144"]["prompt"]), 6144)
        target = collect.payload(inputs, 6144, 8192, "target", context_limit=16384)
        self.assertTrue(target["ignore_eos"])
        self.assertEqual(target["max_tokens"], 8192)
        with self.assertRaises(ValueError):
            collect.payload(inputs, 6144, 8192, "source")

    def test_service_plan_has_no_old_predictor_or_transport(self):
        settings = SimpleNamespace(
            tp1_gpu="2",
            target_tp_size=2,
            tp4_gpus="0,1",
            tp1_port=8610,
            tp4_port=8611,
        )
        common = SimpleNamespace(
            server_command=lambda a, tp, port: [str(tp), str(port)]
        )
        with patch.dict(os.environ, {"BRIDGETP_PREDICTOR_LIVE_CHECKPOINT": "old"}):
            plan = collect.service_only_plan(
                {}, dict(name="case"), Path("/results"), common, settings
            )
        for role, devices in (("source", "2"), ("target", "0,1")):
            env = plan[role + "_env"]
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], devices)
            self.assertEqual(env["VLLM_USE_V2_MODEL_RUNNER"], "0")
            self.assertFalse(any("PREDICTOR" in k or "STREAM_" in k for k in env))
            self.assertNotIn("--kv-transfer-config", plan[role + "_argv"])
            self.assertIn("--enforce-eager", plan[role + "_argv"])

    def test_workload_passes_target_limit_without_loosening_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            workload = collect.Workload(
                runner.prepare_inputs(Tokenizer()), Path(tmp), {}, 1
            )
            try:
                with patch.object(workload.executor, "submit"):
                    workload.submit("target", 6144, 8192, "target-test")
                    with self.assertRaises(ValueError):
                        workload.submit("source", 6144, 8192, "source-test")
            finally:
                workload.executor.shutdown()

    def test_reuse_rejects_partial_or_unclosed_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "request_results.json").write_text(
                json.dumps(
                    dict(
                        status="COMPLETED",
                        requests=[
                            dict(
                                error=None,
                                done=True,
                                tokens=7,
                                max_tokens=8,
                            )
                        ],
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, "completion"):
                runner.accepted_scene(root)

    def test_sparse_matrix_does_not_enable_migration_or_old_guard(self):
        for scene in runner.matrix():
            self.assertLessEqual(scene["source_count"], 4)
            self.assertLessEqual(scene["target_count"], 8)
            self.assertEqual(scene["headroom_blocks"], 125)
        self.assertEqual(
            runner.main(
                [
                    "--model",
                    "/absent",
                    "--output",
                    "/absent",
                    "--expected-revision",
                    "not-a-real-revision",
                    "--validate-only",
                ]
            ),
            0,
        )

    def test_native_facts_match_roles_despite_target_starting_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "case"
            scene.mkdir()
            (scene / "native_initial_receipt.json").write_text(
                json.dumps(
                    [
                        dict(
                            total_blocks=100,
                            free_blocks=99,
                            block_size=16,
                            context_limit=8192,
                        ),
                        dict(
                            total_blocks=200,
                            free_blocks=199,
                            block_size=16,
                            context_limit=16384,
                        ),
                    ]
                )
            )
            (scene / "lifecycle_events.json").write_text(
                json.dumps(
                    [
                        dict(
                            pool="target",
                            load_started_unix_s=0,
                            healthy_unix_s=9,
                            warm_ready_unix_s=10,
                        ),
                        dict(
                            pool="source",
                            load_started_unix_s=20,
                            healthy_unix_s=24,
                            warm_ready_unix_s=25,
                        ),
                    ]
                )
            )
            facts = runner.service_facts(root)[0]["pools"]
            self.assertEqual(facts["source"]["load_to_health_s"], 4)
            self.assertEqual(facts["target"]["load_to_health_s"], 9)

    def test_pilot_then_resume_reuses_closed_runs(self):
        from tools.bridge_tp import run_phase9_cap0_calibration as common

        def command(argv, **kwargs):
            return "frozen\n" if argv[:3] == ["git", "rev-parse", "HEAD"] else ""

        def scene(spec, inputs, shape, rep, directory, *args):
            directory.mkdir()
            common.write_json(directory / "recipe.json", shape)
            return dict(status="COMPLETED")

        expected = runner.read(
            runner.ROOT / "experiments/platforms/qwen3_8b_verified_assets.json"
        )["sha256"]
        hardware = dict(gpus=[dict(index=str(i), uuid=str(i)) for i in range(4)])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner.subprocess, "check_output", side_effect=command),
            patch.object(runner, "runtime_inventory", return_value=hardware),
            patch.object(
                runner,
                "inspect_model",
                return_value=dict(
                    files={k: dict(sha256=v) for k, v in expected.items()}
                ),
            ),
            patch("torch.cuda.nccl.version", return_value=(2, 27, 3)),
            patch.dict(
                sys.modules,
                {
                    "transformers": SimpleNamespace(
                        AutoTokenizer=SimpleNamespace(
                            from_pretrained=lambda *a, **k: Tokenizer()
                        )
                    )
                },
            ),
            patch.object(runner.collect, "run_scene", side_effect=scene) as measured,
            patch.object(runner, "accepted_scene"),
            patch.object(runner, "summarize", return_value={}),
            patch.object(runner, "service_facts", return_value=[]),
            patch.object(runner, "pack", return_value=dict(sha256="digest")) as packed,
        ):
            output = Path(tmp) / "batch"
            args = [
                "--model",
                str(Path(tmp) / "model"),
                "--output",
                str(output),
                "--expected-revision",
                "frozen",
            ]
            self.assertEqual(runner.main(args), 0)
            self.assertEqual(measured.call_count, 2)
            self.assertEqual(runner.main(args + ["--phase", "full", "--resume"]), 0)
            self.assertEqual(measured.call_count, 10)
            self.assertEqual(packed.call_count, 2)
            self.assertNotEqual(
                packed.call_args_list[0].args[1], packed.call_args_list[1].args[1]
            )
            result = runner.read(output / "result.json")
            self.assertEqual(sum(r["reused"] for r in result["runs"]), 2)
            self.assertFalse(result["deployment_ready"])


if __name__ == "__main__":
    unittest.main()
