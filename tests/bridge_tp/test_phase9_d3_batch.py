# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

from vllm.bridge_tp.controller.config import ControllerConfig

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "bridge_tp" / "run_phase9_d3_batch.py"
MANIFEST = ROOT / "experiments" / "phase9" / "manifests" / "d3_prompts_50.json"

spec = importlib.util.spec_from_file_location("phase9_d3_batch", SCRIPT)
batch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(batch)


class TestPhase9D3Batch(unittest.TestCase):
    def test_frozen_manifest_is_valid_and_unique(self) -> None:
        manifest = batch.load_manifest(MANIFEST)
        self.assertEqual(len(manifest["prompts"]), 50)
        self.assertEqual(
            len({item["request_id"] for item in manifest["prompts"]}), 50
        )
        self.assertEqual(
            [item["request_id"] for item in batch.selected_prompts(
                manifest, "smoke", None
            )],
            ["d3-formal-001", "d3-formal-002"],
        )

    def test_extracts_fixed_prefix_and_migration_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "attempt"
            destination = root / "request"
            run.mkdir()
            prompt = [11, 12, 13]
            source_suffix = list(range(1000, 1160))
            target_suffix = list(range(2000, 2256))
            control = source_suffix + list(range(3000, 3256))
            values = {
                "response_proxy_stats.json": {
                    "token_ids": source_suffix + target_suffix,
                    "cutover_index": 160,
                    "committed": True,
                    "source_origin_tokens": 160,
                    "target_origin_tokens": 256,
                },
                "staging_manifest.json": {
                    "num_prompt_tokens": len(prompt),
                    "all_known_token_ids": prompt + source_suffix,
                },
                "takeover_state.json": {"state": "COMMITTED"},
                "control_tokens.json": control,
            }
            for name, value in values.items():
                (run / name).write_text(json.dumps(value), encoding="utf-8")
            args = types.SimpleNamespace(cutover_tokens=160, budget=256)
            batch.extract_migration_artifacts(args, run, destination)
            fixed = json.loads(
                (destination / "fixed_prefix.json").read_text(encoding="utf-8")
            )
            self.assertEqual(fixed["fixed_token_ids"], prompt + source_suffix)
            self.assertEqual(fixed["boundary_k"], 160)
            self.assertEqual(
                len(json.loads(
                    (destination / "migrated_tokens.json").read_text()
                )),
                416,
            )

    def test_rejects_non_frozen_boundary(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["design"]["cutover_output_tokens"] = 159
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validated design"):
                batch.load_manifest(path)

    def test_controller_config_uses_calibrated_c_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tpot_path = root / "tpot.json"
            interference_path = root / "interference.json"
            tpot_path.write_text(
                json.dumps(
                    {
                        "tpot_tp1": {
                            "base_s": 0.03,
                            "per_running_s": 0.001,
                            "conditions": 21,
                        },
                        "tpot_tp4": {
                            "base_s": 0.02,
                            "per_running_s": 0.001,
                            "conditions": 30,
                        },
                    }
                ),
                encoding="utf-8",
            )
            interference_path.write_text(
                json.dumps(
                    {
                        "controller_model": {
                            "model_kind": "rate_aware_tpot",
                            "calibration_source": "frozen C2",
                            "tpot_rate_coef_s2_per_gib": 0.01,
                            "tpot_rate_load_coef_s2_per_gib": 0.01,
                            "min_load_frac": 0.01,
                            "max_load_frac": 0.65,
                            "min_rate_gib_s": 0.4,
                            "max_rate_gib_s": 1.2,
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = types.SimpleNamespace(
                controller_config_template=None,
                tpot_model=tpot_path,
                interference_model=interference_path,
                tp1_url="http://127.0.0.1:8001",
                tp4_url="http://127.0.0.1:8200",
                tp1_blocks=100,
                tp4_blocks=200,
                survival_table=root / "survival.json",
            )
            path = batch.controller_config(args, root / "run")
            config = ControllerConfig.load(path)
            self.assertEqual(config.tpot_tp1.calibration_source[:8], "Phase 9 ")
            self.assertEqual(config.interference.model_kind, "rate_aware_tpot")
            self.assertEqual(config.tp4_total_kv_blocks, 200)


if __name__ == "__main__":
    unittest.main()
