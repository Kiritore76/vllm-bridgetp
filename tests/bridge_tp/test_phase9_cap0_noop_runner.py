# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


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
FREEZER = load_script(
    "phase9_cap0_noop_freezer",
    ROOT / "tools" / "bridge_tp" / "freeze_phase9_cap0_noop.py",
)
FORMAL = load_script(
    "phase9_cap0_noop_formal",
    ROOT / "tools" / "bridge_tp" / "run_phase9_cap0_noop_formal.py",
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

    def test_rejects_active_decision_not_blocked_by_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "controller"
            background = root / "background"
            controller.mkdir()
            background.mkdir()
            self.write_case(controller, background)
            audit = controller / "phase9_audit.jsonl"
            records = [
                json.loads(line)
                for line in audit.read_text(encoding="utf-8").splitlines()
            ]
            records.insert(
                -1,
                {
                    "kind": "capacity_pilot_decision",
                    "action": "STAY",
                    "signal": {"active": True},
                    "target_kv_usage_frac": 0.1,
                    "target_waiting": 0,
                },
            )
            audit.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = RUNNER.accept_noop(
                controller,
                background,
                expected_jobs=6,
                expected_anchor_tokens=8000,
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any(
                    "not every active capacity decision" in error
                    for error in result["errors"]
                )
            )


class TestNoopFreezeAndFormal(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def make_bringup(self, root: Path) -> tuple[Path, dict[str, object]]:
        manifest = BUILDER.build_manifest()
        working = root / "working.json"
        self.write_json(working, manifest)
        manifest_sha = FREEZER.common.sha256(working)
        acceptance = {
            "status": "PASS",
            "errors": [],
            "active_capacity_decisions": 10,
            "blocked_stay_decisions": 10,
            "start_shadow_decisions": 0,
            "migration_transitions": 0,
            "final_state": "COMPLETED_ON_TP1",
            "anchor_emitted_tokens": 8000,
            "forbidden_artifacts": [],
        }
        status = {
            "status": "BRINGUP_COMPLETE",
            "run_id": "bringup-1",
            "revision": "bringup-revision",
        }
        inputs = {
            "revision": "bringup-revision",
            "manifest_sha256": manifest_sha,
            "survival_table_sha256": "survival-sha",
            "guard_file_sha256": "guard-sha",
            "guard_free_kv_tokens": 8448,
        }
        self.write_json(root / "status.json", status)
        self.write_json(root / "provenance" / "noop_acceptance.json", acceptance)
        self.write_json(root / "provenance" / "inputs.json", inputs)
        self.write_json(
            root / "background" / "background_manifest.json",
            manifest,
        )
        return working, manifest

    def test_freezes_exact_jobs_and_formal_contract_accepts_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bringup = root / "bringup"
            working, manifest = self.make_bringup(bringup)
            freeze_args = Namespace(
                bringup_root=bringup,
                working_manifest=working,
                expected_bringup_revision="bringup-revision",
                expected_working_sha256=FREEZER.common.sha256(working),
                expected_survival_sha256="survival-sha",
                expected_guard_sha256="guard-sha",
                expected_guard=8448,
                tp1_blocks=1968,
                tp4_blocks=35739,
                anchor_max_tokens=8000,
                max_model_len=8192,
            )
            with mock.patch.object(
                FREEZER.common,
                "git",
                return_value="freeze-revision",
            ):
                frozen, evidence = FREEZER.build_frozen_manifest(freeze_args)
            self.assertEqual(frozen["status"], "FROZEN")
            self.assertEqual(frozen["scenario"], "CAP-0 No-op formal")
            self.assertEqual(frozen["jobs"], manifest["jobs"])

            frozen_path = root / "noop_v1.json"
            provenance_path = root / "noop_v1.provenance.json"
            self.write_json(frozen_path, frozen)
            evidence["frozen_manifest_sha256"] = FREEZER.common.sha256(frozen_path)
            evidence["origin_manifest_sha256"] = FREEZER.common.sha256(working)
            self.write_json(provenance_path, evidence)
            formal_args = Namespace(
                bringup_root=bringup,
                manifest=frozen_path,
                manifest_provenance=provenance_path,
                expected_survival_sha256="survival-sha",
                expected_guard_sha256="guard-sha",
                expected_guard=8448,
            )
            contract = FORMAL.validate_bringup_contract(formal_args)
            self.assertEqual(contract["bringup_run_id"], "bringup-1")

    def test_rejects_fewer_than_three_formal_repetitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            FORMAL.validate_inputs(Namespace(repetitions=2))


if __name__ == "__main__":
    unittest.main()
