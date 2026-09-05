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
    "phase9_cap0_abandon_builder",
    ROOT / "tools" / "bridge_tp" / "build_phase9_cap0_abandon_manifest.py",
)
RUNNER = load_script(
    "phase9_cap0_abandon_runner",
    ROOT / "tools" / "bridge_tp" / "run_phase9_cap0_abandon.py",
)


class TestAbandonManifest(unittest.TestCase):
    def test_default_manifest_creates_finite_recovery_window(self) -> None:
        manifest = BUILDER.build_manifest()
        pressure = RUNNER.validate_abandon_pressure(
            manifest,
            anchor_max_tokens=8000,
            tp1_total_tokens=1968 * 16,
            tp4_total_tokens=35739 * 16,
            max_model_len=8192,
        )
        self.assertEqual(len(manifest["jobs"]), 8)
        self.assertEqual(pressure["source_jobs"], 4)
        self.assertEqual(pressure["target_jobs"], 4)
        self.assertGreaterEqual(pressure["source_to_capacity_frac"], 0.75)
        self.assertLessEqual(pressure["source_to_capacity_frac"], 0.90)
        self.assertLess(pressure["target_to_capacity_frac"], 0.10)

    def test_rejects_source_pressure_that_cannot_recover(self) -> None:
        manifest = BUILDER.build_manifest(source_output_tokens=7000)
        with self.assertRaisesRegex(ValueError, "recovery window"):
            RUNNER.validate_abandon_pressure(
                manifest,
                anchor_max_tokens=8000,
                tp1_total_tokens=1968 * 16,
                tp4_total_tokens=35739 * 16,
                max_model_len=8192,
            )

    def test_cli_writes_working_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "abandon.json"
            original = sys.argv
            try:
                sys.argv = [
                    "build_phase9_cap0_abandon_manifest.py",
                    "--out",
                    str(output),
                ]
                BUILDER.main()
            finally:
                sys.argv = original
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["scenario"], "CAP-0 Safe abandon bring-up")
            self.assertEqual(len(manifest["jobs"]), 8)


class TestAbandonAcceptance(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def write_case(self, root: Path, *, abort_source: bool = False) -> None:
        controller = root / "controller"
        background = root / "background"
        controller.mkdir()
        background.mkdir()
        source_id = "source-anchor"
        self.write_json(
            background / "background_summary.json",
            {
                "jobs": 8,
                "completed": 8,
                "failed": 0,
                "results": [
                    {"response_id": f"background-{index}"}
                    for index in range(8)
                ],
            },
        )
        reason = "CAP-0 source headroom recovered before cutover"
        audit = [
            {
                "kind": "capacity_pilot_decision",
                "action": "START_SHADOW",
                "signal": {"active": True},
                "target_kv_usage_frac": 0.05,
                "target_waiting": 0,
            },
            {"kind": "transition", "to": "SHADOW"},
            {"kind": "abandon", "reason": reason},
            {"kind": "transition", "to": "CANCELLED"},
            {
                "kind": "run_end",
                "final_state": "CANCELLED",
                "trigger_path": "CAPACITY_PILOT",
                "ranks_ready": [],
            },
        ]
        (controller / "phase9_audit.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in audit),
            encoding="utf-8",
        )
        emitted = [
            {"index": index, "token_id": 100 + index, "origin": "source"}
            for index in range(4)
        ]
        self.write_json(
            controller / "response_proxy_stats.json",
            {
                "committed": False,
                "emitted_tokens": 4,
                "source_origin_tokens": 4,
                "target_origin_tokens": 0,
                "token_ids": [100, 101, 102, 103],
                "emitted": emitted,
            },
        )
        (controller / "unified_response.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in emitted),
            encoding="utf-8",
        )
        self.write_json(
            controller / "session_manifest.json",
            {"migration_id": "migration-1", "source_request_id": source_id},
        )
        self.write_json(
            controller / "cleanup_request.json",
            {"abort_source": abort_source, "reason": reason},
        )
        self.write_json(
            controller / "takeover_state.json",
            {
                "state": "CANCELLED",
                "source_abort_dispatched": abort_source,
                "source_continues_on_tp1": not abort_source,
            },
        )
        self.write_json(
            controller / "source_cleanup_receipt.json",
            {"status": "CLEANED"},
        )
        self.write_json(
            controller / "stager_cleanup_receipt.json",
            {"status": "CLEANED"},
        )

    def test_accepts_safe_pre_cutover_abandon(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_case(root)
            result = RUNNER.accept_abandon(
                root / "controller",
                root / "background",
                expected_jobs=8,
                expected_anchor_tokens=4,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["transition_states"], [
                "SHADOW",
                "CANCELLED",
            ])
            self.assertTrue(result["source_continues_on_tp1"])

    def test_rejects_source_abort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_case(root, abort_source=True)
            result = RUNNER.accept_abandon(
                root / "controller",
                root / "background",
                expected_jobs=8,
                expected_anchor_tokens=4,
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any("source abort" in error for error in result["errors"])
            )


if __name__ == "__main__":
    unittest.main()
