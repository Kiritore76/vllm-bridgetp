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
    "phase9_cap0_abandon_formal_builder",
    ROOT / "tools" / "bridge_tp" / "build_phase9_cap0_abandon_manifest.py",
)
FREEZER = load_script(
    "phase9_cap0_abandon_freezer",
    ROOT / "tools" / "bridge_tp" / "freeze_phase9_cap0_abandon.py",
)
FORMAL = load_script(
    "phase9_cap0_abandon_formal",
    ROOT / "tools" / "bridge_tp" / "run_phase9_cap0_abandon_formal.py",
)


class TestAbandonFormalContract(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def make_bringup(self, root: Path) -> tuple[Path, dict]:
        manifest = BUILDER.build_manifest()
        working = root / "working.json"
        self.write_json(working, manifest)
        manifest_sha = FREEZER.common.sha256(working)
        acceptance = {
            "status": "PASS",
            "errors": [],
            "capacity_decisions": 1,
            "start_shadow_decisions": 1,
            "shadow_capacity_clear_samples": 1,
            "cleanup_complete_events": 1,
            "trigger_path": "CAPACITY_PILOT",
            "transition_states": ["SHADOW", "CANCELLED"],
            "final_state": "CANCELLED",
            "takeover_state": "CANCELLED",
            "source_abort_dispatched": False,
            "source_continues_on_tp1": True,
            "source_cleanup_status": "CLEANED",
            "stager_cleanup_status": "CLEANED",
            "forbidden_artifacts": [],
            "receiver_receipt_count": 0,
            "source_origin_tokens": 8000,
            "target_origin_tokens": 0,
            "anchor_emitted_tokens": 8000,
        }
        status = {
            "status": "BRINGUP_COMPLETE",
            "run_id": "abandon-bringup-1",
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
        self.write_json(
            root / "provenance" / "abandon_acceptance.json",
            acceptance,
        )
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
            self.assertEqual(frozen["scenario"], "CAP-0 Safe abandon formal")
            self.assertEqual(frozen["jobs"], manifest["jobs"])

            frozen_path = root / "abandon_v1.json"
            provenance_path = root / "abandon_v1.provenance.json"
            self.write_json(frozen_path, frozen)
            evidence["frozen_manifest_sha256"] = FREEZER.common.sha256(
                frozen_path
            )
            evidence["origin_manifest_sha256"] = FREEZER.common.sha256(working)
            self.write_json(provenance_path, evidence)
            formal_args = Namespace(
                bringup_root=bringup,
                manifest=frozen_path,
                manifest_provenance=provenance_path,
                expected_survival_sha256="survival-sha",
                expected_guard_sha256="guard-sha",
                expected_guard=8448,
                anchor_max_tokens=8000,
            )
            contract = FORMAL.validate_bringup_contract(formal_args)
            self.assertEqual(contract["bringup_run_id"], "abandon-bringup-1")

    def test_rejects_fewer_than_three_formal_repetitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            FORMAL.validate_inputs(Namespace(repetitions=2))


if __name__ == "__main__":
    unittest.main()
