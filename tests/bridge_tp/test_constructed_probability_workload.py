# SPDX-License-Identifier: Apache-2.0
"""Engineering construction is explicit, reproducible, and keeps natural EOS."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bridge_tp.build_constructed_probability_workload import (
    constructed_prompt_tokens,
    make_rows,
)
from tools.bridge_tp.run_randomized_goodoutput_pilot_a100 import (
    build_setup,
    select_inputs,
)


def encode(text, **kwargs):
    return [
        int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) for word in text.split()
    ]


TOKENIZER = SimpleNamespace(
    encode=encode,
    apply_chat_template=lambda messages, **kw: (
        "CHAT_USER " + messages[0]["content"] + " CHAT_ASSISTANT"
    ),
)


def recipes(seed=1, split="engineering_train", backgrounds=120):
    return make_rows(
        seed=seed,
        split=split,
        anchor_prompt_tokens=4096,
        background_prompt_tokens=2048,
        anchor_sections=24,
        background_sections=6,
        backgrounds=backgrounds,
    )


class TestConstructedProbabilityWorkload(unittest.TestCase):
    def test_constructed_sha_override_retains_frozen_model_and_checkpoint_checks(self):
        from tools.bridge_tp import run_goodoutput_matrix_a100 as matrix

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = root / "guard"
            guard.write_text("8448")
            args = SimpleNamespace(
                expected_revision="rev",
                expected_host=None,
                portable_hardware=True,
                constructed_workload=True,
                probability_pilot=True,
                expected_input_sha256="a" * 64,
                input=root / "input",
                model=root,
                base=root / "base",
                survival=root / "survival",
                guard=guard,
                checkpoint=root / "checkpoint",
                reference=root / "reference",
            )
            hashes = {args.input: "a" * 64}
            hashes.update(
                {
                    args.model / "config.json": matrix.EXPECTED_SHAS["model_config"],
                    **{
                        getattr(args, k): matrix.EXPECTED_SHAS[k]
                        for k in (
                            "base",
                            "survival",
                            "guard",
                            "checkpoint",
                            "reference",
                        )
                    },
                }
            )

            def check():
                outputs = iter(
                    (
                        "rev",
                        "",
                        "GPU-a\nGPU-b\nGPU-c\nGPU-d\nGPU-e",
                        "\n".join(["NVIDIA A100-PCIE-40GB"] * 5),
                        "",
                    )
                )
                with (
                    patch.object(
                        matrix.subprocess,
                        "check_output",
                        side_effect=lambda *a, **kw: next(outputs),
                    ),
                    patch.object(matrix, "sha256", side_effect=hashes.get),
                ):
                    return matrix.verify(args)

            self.assertEqual(check()["input_sha256"], "a" * 64)
            args.expected_input_sha256 = "b" * 64
            with self.assertRaisesRegex(ValueError, "input SHA differs"):
                check()
            args.expected_input_sha256 = "a" * 64
            hashes[args.checkpoint] = "c" * 64
            with self.assertRaisesRegex(ValueError, "checkpoint SHA differs"):
                check()

    def test_unique_recipes_reproduce_and_splits_are_explicit(self):
        rows = recipes()
        self.assertEqual(rows, recipes())
        self.assertEqual(len(rows), len({r["id"] for r in rows}))
        self.assertTrue(all(r["workload_origin"] == "constructed" for r in rows))
        self.assertTrue(all(r["split"] != "test" for r in rows))
        self.assertTrue(all(r["augmentation"]["ignore_eos"] is False for r in rows))
        validation = recipes(split="engineering_validation")
        self.assertFalse(
            {r["source_tree_id"] for r in rows}
            & {r["source_tree_id"] for r in validation}
        )
        self.assertNotEqual(rows, recipes(seed=2))

    def test_token_budget_keeps_task_suffix_and_distinct_record_content(self):
        rows = recipes()
        prompts = [constructed_prompt_tokens(TOKENIZER, row) for row in rows[:6]]
        self.assertTrue(all(len(p) == 4096 for p in prompts))
        self.assertTrue(all(p[0] == encode("CHAT_USER")[0] for p in prompts))
        self.assertTrue(all(p[-1] == encode("CHAT_ASSISTANT")[0] for p in prompts))
        self.assertEqual(len(prompts), len({tuple(p) for p in prompts}))
        self.assertEqual(prompts[0], constructed_prompt_tokens(TOKENIZER, rows[0]))
        rows[0]["construction"]["requested_prompt_tokens"] = 100
        with self.assertRaisesRegex(ValueError, "truncates"):
            constructed_prompt_tokens(TOKENIZER, rows[0])

    def test_collector_refuses_implicit_or_mixed_engineering_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            rows = recipes()
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaises(ValueError):
                select_inputs(path, all_held_out=True)
            selected = select_inputs(path, all_held_out=True, constructed_workload=True)
            self.assertTrue(
                all(r["construction"]["role"] == "anchor" for r in selected[:6])
            )
            rows[0]["split"] = "test"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "engineering split"):
                select_inputs(path, all_held_out=True, constructed_workload=True)

    def test_setup_rotates_recipes_and_preserves_natural_eos_and_provenance(self):
        fake = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: TOKENIZER)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.jsonl"
            path.write_text(
                "".join(json.dumps(r) + "\n" for r in recipes(backgrounds=350))
            )
            args = SimpleNamespace(
                input=path,
                model=root,
                constructed_workload=True,
                probability_pilot=True,
                cases=None,
                constructed_source_count=1,
                constructed_target_count=0,
                max_model_len=16384,
                tp4_max_model_len=32768,
                anchor_context_limit=True,
                background_context_limit=False,
                background_max_tokens=2048,
                arrival_window_s=30,
                arrival_wave_period_s=10,
            )
            with patch.dict("sys.modules", {"transformers": fake}):
                setup = build_setup(args, root)
            name = "constructed_source1_target0"
            anchor = json.loads(Path(setup["anchors"][name]["path"]).read_text())
            manifest = json.loads(Path(setup["manifests"][name]["path"]).read_text())
            self.assertEqual(len(anchor["prompt"]), 4096)
            self.assertIs(anchor["ignore_eos"], False)
            self.assertEqual(manifest["target_count"], 0)
            self.assertEqual(manifest["controller_split"], "engineering_train")
            jobs = manifest["jobs"]
            self.assertEqual(len(jobs), len({j["input_id"] for j in jobs}))
            self.assertTrue(all(len(j["request"]["prompt"]) == 2048 for j in jobs))
            self.assertTrue(all(j["request"]["ignore_eos"] is False for j in jobs))
            self.assertTrue(all(j["workload_origin"] == "constructed" for j in jobs))
            self.assertTrue(all(j["construction"] for j in jobs))

    def test_target_load_can_change_without_changing_source_requests(self):
        fake = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: TOKENIZER)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in recipes()))
            sources, anchors = [], []
            for target_count in (0, 8, 24):
                args = SimpleNamespace(
                    input=path,
                    model=root,
                    constructed_workload=True,
                    probability_pilot=True,
                    cases=None,
                    constructed_source_count=3,
                    constructed_target_count=target_count,
                    max_model_len=16384,
                    tp4_max_model_len=32768,
                    anchor_context_limit=True,
                    background_context_limit=False,
                    background_max_tokens=2048,
                )
                with patch.dict("sys.modules", {"transformers": fake}):
                    setup = build_setup(args, root / str(target_count))
                name = f"constructed_source3_target{target_count}"
                anchors.append(setup["anchors"][name]["input_id"])
                manifest = json.loads(
                    Path(setup["manifests"][name]["path"]).read_text()
                )
                sources.append(
                    [j["input_id"] for j in manifest["jobs"] if j["pool"] == "source"]
                )
                self.assertEqual(manifest["target_count"], target_count)
                self.assertEqual(manifest["source_count"], 3)
            self.assertEqual(len(set(anchors)), 1)
            self.assertEqual(sources[0], sources[1])
            self.assertEqual(sources[1], sources[2])


if __name__ == "__main__":
    unittest.main()
