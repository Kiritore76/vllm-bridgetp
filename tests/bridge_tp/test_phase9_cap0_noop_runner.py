# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = load_script(
    "phase9_cap0_noop_builder",
    ROOT / "tools" / "bridge_tp" / "build_phase9_cap0_noop_manifest.py",
)
RUNNER = load_script(
    "phase9_cap0_noop_runner",
    ROOT / "tools" / "bridge_tp" / "run_phase9_cap0_noop.py",
)


class TestNoopManifest(unittest.TestCase):
    def test_default_manifest_exceeds_both_pool_pressure_floors(self) -> None:
        manifest = BUILDER.build_manifest()
        pressure = RUNNER.validate_noop_pressure(
            manifest,
            anchor_max_tokens=8000,
            tp1_total_tokens=1968 * 16,
            tp4_total_tokens=35739 * 16,
            max_model_len=8192,
        )
        self.assertEqual(len(manifest["jobs"]), 76)
        self.assertEqual(pressure["source_jobs"], 4)
        self.assertEqual(pressure["target_jobs"], 72)
        self.assertGreater(
            pressure["source_output_demand_tokens"],
            pressure["tp1_total_tokens"],
        )
        self.assertGreater(
            pressure["target_context_demand_tokens"],
            pressure["tp4_total_tokens"],
        )
        self.assertEqual(
            len(manifest["jobs"][0]["request"]["prompt"]),
            7000,
        )

    def test_rejects_target_demand_that_fits(self) -> None:
        manifest = BUILDER.build_manifest(
            target_copies=5,
            target_prompt_tokens=100,
            target_output_tokens=100,
        )
        with self.assertRaisesRegex(ValueError, "too few target jobs"):
            RUNNER.validate_noop_pressure(
                manifest,
                anchor_max_tokens=8000,
                tp1_total_tokens=1968 * 16,
                tp4_total_tokens=35739 * 16,
                max_model_len=8192,
            )

    def test_builder_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "noop.json"
            BUILDER.write_json(path, BUILDER.build_manifest(target_copies=6))
            loaded = RUNNER.load_manifest(path)
            self.assertEqual(len(loaded["jobs"]), 10)
            with self.assertRaises(FileExistsError):
                BUILDER.write_json(path, {})


class TestNoopAcceptance(unittest.TestCase):
    @staticmethod
    def write_case(
        controller: Path,
        background: Path,
        *,
        start_shadow: bool = False,
    ) -> None:
        (background / "background_summary.json").write_text(
            json.dumps(
                {
                    "jobs": 6,
                    "completed": 6,
                    "failed": 0,
                    "start_unix_s": 1.0,
                    "end_unix_s": 10.0,
                }
            ),
            encoding="utf-8",
        )
        records = [
            {
                "kind": "telemetry",
                "tp1": {"kv_usage_frac": 0.75},
            },
            {
                "kind": "capacity_pilot_decision",
                "action": "STAY",
                "signal": {"active": True},
                "target_kv_usage_frac": 0.90,
                "target_waiting": 5,
            },
        ]
        if start_shadow:
            records.extend(
                [
                    {
                        "kind": "capacity_pilot_decision",
                        "action": "START_SHADOW",
                        "signal": {"active": True},
                        "target_kv_usage_frac": 0.10,
                        "target_waiting": 0,
                    },
                    {"kind": "transition", "to": "SHADOW"},
                ]
            )
        records.append(
            {"kind": "run_end", "final_state": "COMPLETED_ON_TP1"}
        )
        (controller / "phase9_audit.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        (controller / "response_proxy_stats.json").write_text(
            json.dumps(
                {
                    "emitted_tokens": 8000,
                    "target_origin_tokens": 0,
                    "committed": False,
                }
            ),
            encoding="utf-8",
        )

    def test_accepts_active_signal_blocked_by_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "controller"
            background = root / "background"
            controller.mkdir()
            background.mkdir()
            self.write_case(controller, background)
            result = RUNNER.accept_noop(
                controller,
                background,
                expected_jobs=6,
                expected_anchor_tokens=8000,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["blocked_stay_decisions"], 1)
            self.assertEqual(result["start_shadow_decisions"], 0)

    def test_rejects_any_shadow_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "controller"
            background = root / "background"
            controller.mkdir()
            background.mkdir()
            self.write_case(controller, background, start_shadow=True)
            result = RUNNER.accept_noop(
                controller,
                background,
                expected_jobs=6,
                expected_anchor_tokens=8000,
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["start_shadow_decisions"], 1)
            self.assertEqual(result["migration_transitions"], 1)


if __name__ == "__main__":
    unittest.main()
