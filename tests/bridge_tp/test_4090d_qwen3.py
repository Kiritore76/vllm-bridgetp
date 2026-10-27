# SPDX-License-Identifier: Apache-2.0
"""New-platform isolation, asset checks and real Qwen3 residual-hook semantics."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bridge_tp.prepare_4090d_qwen3 import (
    PROFILE,
    download,
    inspect_model,
    prepare_inputs,
    read,
    validate_config,
    validate_gpus,
)


class Qwen3PlatformTests(unittest.TestCase):
    def setUp(self):
        self.profile = read(PROFILE)
        self.config = dict(
            model_type="qwen3",
            max_position_embeddings=40960,
            architectures=["Qwen3ForCausalLM"],
            **self.profile["model_geometry"],
        )

    def test_geometry_is_qwen3_not_old_model(self):
        self.assertEqual(validate_config(self.config, self.profile), 147456)
        for changes in (
            dict(model_type="qwen2"),
            dict(hidden_size=5120),
            dict(num_hidden_layers=32),
            dict(num_key_value_heads=4),
            dict(quantization_config={"quant_method": "awq"}),
            dict(use_sliding_window=True),
            dict(max_position_embeddings=8192),
            dict(rope_scaling={"rope_type": "yarn", "factor": 4}),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**self.config, **changes}, self.profile)
        self.assertTrue(
            all(v is None for v in self.profile["measured_profiles"].values())
        )
        self.assertIsNone(self.profile["predictor"]["checkpoint"])

    def test_four_real_24gb_4090d_required(self):
        rows = [
            dict(name="NVIDIA GeForce RTX 4090 D", memory_mib=24564, uuid=f"gpu-{i}")
            for i in range(4)
        ]
        validate_gpus(rows, self.profile)
        invalid = [
            rows[:3],
            rows + [rows[0]],
            [{**r, "memory_mib": 49140} for r in rows],
            [{**r, "name": "NVIDIA A100-PCIE-40GB"} for r in rows],
            [{**r, "uuid": "same"} for r in rows],
        ]
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_gpus(candidate, self.profile)

    def test_model_asset_identity_and_pointer_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
            ):
                (model / name).write_text("{}")
            (model / "config.json").write_text(json.dumps(self.config))
            index = model / "model.safetensors.index.json"
            index.write_text(
                json.dumps(dict(weight_map={"weight": "part.safetensors"}))
            )
            weight = model / "part.safetensors"
            weight.write_bytes(b"synthetic weight for identity test")
            receipt = inspect_model(model, self.profile)
            self.assertEqual(receipt["kv_bytes_per_token_per_rank_tp2"], 73728)
            self.assertEqual(receipt["predictor_status"], "NOT_TRAINED")
            before = receipt["files"]["part.safetensors"]["sha256"]
            weight.write_bytes(b"changed identity")
            self.assertNotEqual(
                before,
                inspect_model(model, self.profile)["files"]["part.safetensors"][
                    "sha256"
                ],
            )
            weight.write_bytes(b"version https://git-lfs.github.com/spec/v1")
            with self.assertRaisesRegex(ValueError, "LFS pointer"):
                inspect_model(model, self.profile)
            index.write_text(
                json.dumps(dict(weight_map={"weight": "../outside.safetensors"}))
            )
            with self.assertRaisesRegex(ValueError, "escapes root"):
                inspect_model(model, self.profile)

    def test_download_never_overwrites_unknown_existing_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.json").write_text("preserve me")
            with self.assertRaisesRegex(ValueError, "do not overwrite"):
                download(model, "modelscope", self.profile)
            self.assertEqual((model / "config.json").read_text(), "preserve me")

    def test_non_thinking_template_is_used_and_recorded(self):
        from tests.bridge_tp.test_predictor_formal_corpus import CharacterTokenizer
        from tools.bridge_tp.build_predictor_formal_corpus import materialize, recipe

        tokenizer = CharacterTokenizer()
        with patch.object(
            tokenizer, "apply_chat_template", wraps=tokenizer.apply_chat_template
        ) as render:
            result = materialize(copy.deepcopy(recipe()[0]), tokenizer, "disabled")
        self.assertEqual(result["thinking_mode"], "disabled")
        self.assertTrue(render.call_args_list)
        self.assertTrue(
            all(
                call.kwargs["enable_thinking"] is False
                for call in render.call_args_list
            )
        )

    def test_qwen3_hook_sums_residual_without_mutation(self):
        import torch

        from tests.bridge_tp.test_predictor_capture import capture

        class Qwen3DecoderLayer(torch.nn.Module):
            def forward(self, value):
                return value * 2, torch.ones_like(value)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Qwen3DecoderLayer()])

        with tempfile.TemporaryDirectory() as temporary:
            writer = capture.PredictorFeatureCapture(Path(temporary), 20, "decoder:0")
            try:
                model = Model()
                writer.attach_model(model)
                hidden, residual = model.layers[0](torch.tensor([[3.0, 5.0]]))
                residual.zero_()
                selected = writer.sample_states(torch.zeros((1, 2)), [0])
                self.assertEqual(selected.tolist(), [[7.0, 11.0]])
                self.assertEqual(hidden.tolist(), [[6.0, 10.0]])
            finally:
                writer.database.close()

    def test_context_probes_use_full_tp_specific_input_output_budget(self):
        from tests.bridge_tp.test_predictor_formal_corpus import CharacterTokenizer

        tokenizer = CharacterTokenizer()
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(self.config))
            (root / "tokenizer.json").write_text("{}")
            with patch.dict(sys.modules, transformers=transformers):
                prepare_inputs(root, root / "inputs", self.profile)
            prompts = read(root / "inputs/smoke_prompts.json")
            for tp, limit in ((1, 8192), (2, 16384)):
                row = prompts[f"boundary_tp{tp}"]
                self.assertEqual(len(row["token_ids"]) + 32, limit)
                self.assertEqual(row["only_tp"], tp)
                self.assertIs(row["enable_thinking"], False)


if __name__ == "__main__":
    unittest.main()
