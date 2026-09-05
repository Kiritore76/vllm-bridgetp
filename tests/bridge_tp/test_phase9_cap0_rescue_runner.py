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
    "phase9_cap0_rescue_builder",
    ROOT / "tools" / "bridge_tp" / "build_phase9_cap0_rescue_manifest.py",
)
RUNNER = load_script(
    "phase9_cap0_rescue_runner",
    ROOT / "tools" / "bridge_tp" / "run_phase9_cap0_rescue.py",
)
FREEZER = load_script(
    "phase9_cap0_rescue_freezer",
    ROOT / "tools" / "bridge_tp" / "freeze_phase9_cap0_rescue.py",
)
FORMAL = load_script(
    "phase9_cap0_rescue_formal",
    ROOT / "tools" / "bridge_tp" / "run_phase9_cap0_rescue_formal.py",
)


class TestRescueManifest(unittest.TestCase):
    def test_cli_writes_manifest_without_forwarding_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rescue.json"
            with mock.patch.object(
                sys,
                "argv",
                ["build_phase9_cap0_rescue_manifest.py", "--out", str(output)],
            ):
                BUILDER.main()
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["scenario"],
                "CAP-0 Rescue reachability bring-up",
            )
            self.assertEqual(len(manifest["jobs"]), 52)

    def test_default_manifest_creates_finite_rescue_window(self) -> None:
        manifest = BUILDER.build_manifest()
        pressure = RUNNER.validate_rescue_pressure(
            manifest,
            anchor_max_tokens=8000,
            tp1_total_tokens=1968 * 16,
            tp4_total_tokens=35739 * 16,
            max_model_len=8192,
        )
        self.assertEqual(len(manifest["jobs"]), 52)
        self.assertEqual(pressure["source_jobs"], 4)
        self.assertEqual(pressure["target_jobs"], 48)
        self.assertGreater(
            pressure["source_output_demand_tokens"],
            pressure["tp1_total_tokens"],
        )
        self.assertLess(
            pressure["target_context_demand_tokens"],
            pressure["tp4_total_tokens"],
        )
        self.assertGreater(pressure["target_to_capacity_frac"], 0.5)

    def test_rejects_target_burst_that_cannot_drain(self) -> None:
        manifest = BUILDER.build_manifest(target_copies=72)
        with self.assertRaisesRegex(ValueError, "must fit"):
            RUNNER.validate_rescue_pressure(
                manifest,
                anchor_max_tokens=8000,
                tp1_total_tokens=1968 * 16,
                tp4_total_tokens=35739 * 16,
                max_model_len=8192,
            )


class TestRescueAcceptance(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def write_case(self, root: Path, *, exact_readback: bool = True) -> None:
        controller = root / "controller"
        background = root / "background"
        controller.mkdir()
        background.mkdir()
        source_id = "source-anchor"
        target_id = "target-anchor"
        migration_id = "migration-1"
        self.write_json(
            background / "background_summary.json",
            {
                "jobs": 6,
                "completed": 6,
                "failed": 0,
                "results": [
                    {"response_id": f"background-{index}"}
                    for index in range(6)
                ],
            },
        )
        audit = [
            {
                "kind": "capacity_pilot_decision",
                "action": "STAY",
                "signal": {"active": True},
                "target_kv_usage_frac": 0.6,
                "target_waiting": 8,
            },
            {
                "kind": "capacity_pilot_decision",
                "action": "START_SHADOW",
                "signal": {"active": True},
                "target_kv_usage_frac": 0.6,
                "target_waiting": 4,
            },
            {"kind": "transition", "to": "SHADOW"},
            {"kind": "transition", "to": "HANDOFF"},
            {"kind": "commit"},
            {"kind": "transition", "to": "TAKEOVER"},
            {
                "kind": "run_end",
                "final_state": "TAKEOVER",
                "trigger_path": "CAPACITY_PILOT",
                "ranks_ready": [0, 1, 2, 3],
            },
        ]
        (controller / "phase9_audit.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in audit),
            encoding="utf-8",
        )
        emitted = [
            {
                "index": index,
                "token_id": 100 + index,
                "origin": "source" if index < 2 else "target",
            }
            for index in range(4)
        ]
        self.write_json(
            controller / "response_proxy_stats.json",
            {
                "committed": True,
                "emitted_tokens": 4,
                "source_origin_tokens": 2,
                "target_origin_tokens": 2,
                "cutover_index": 2,
                "token_ids": [100, 101, 102, 103],
                "emitted": emitted,
                "handoff_stall_s": 0.25,
            },
        )
        (controller / "unified_response.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in emitted),
            encoding="utf-8",
        )
        session = {
            "migration_id": migration_id,
            "source_request_id": source_id,
        }
        self.write_json(controller / "session_manifest.json", session)
        self.write_json(controller / "staging_manifest.json", session)
        self.write_json(
            controller / "takeover_state.json",
            {
                **session,
                "target_request_id": target_id,
                "state": "COMMITTED",
                "source_abort_dispatched": True,
            },
        )
        for rank in range(4):
            receipt = {
                **session,
                "target_request_id": target_id,
                "payload_sha256": f"digest-{rank}",
                "payload_bytes": 1000 + rank,
            }
            self.write_json(
                controller
                / "stage_delivery_receipts"
                / f"tp_rank_{rank}.json",
                {**receipt, "status": "READY"},
            )
            self.write_json(
                controller
                / "receiver_receipts"
                / target_id
                / f"tp_rank_{rank}.json",
                {
                    **receipt,
                    "status": "OWNERSHIP_COMMITTED",
                    "exact_readback": exact_readback,
                },
            )

    def test_accepts_complete_capacity_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_case(root)
            result = RUNNER.accept_rescue(
                root / "controller",
                root / "background",
                expected_jobs=6,
                expected_anchor_tokens=4,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["receiver_ranks"], [0, 1, 2, 3])
            self.assertEqual(result["transition_states"], [
                "SHADOW",
                "HANDOFF",
                "TAKEOVER",
            ])

    def test_rejects_failed_rank_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_case(root, exact_readback=False)
            result = RUNNER.accept_rescue(
                root / "controller",
                root / "background",
                expected_jobs=6,
                expected_anchor_tokens=4,
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any("exact readback failed" in error for error in result["errors"])
            )


class TestRescueRunnerWiring(unittest.TestCase):
    def test_rescue_allows_only_a_clean_stager_exit(self) -> None:
        args = Namespace(validate_only=False)
        with mock.patch.object(
            RUNNER, "parse_args", return_value=args
        ), mock.patch.object(
            RUNNER,
            "validate_inputs",
            return_value=("revision", 8448, {"target_jobs": 48}),
        ), mock.patch.object(RUNNER.scenario_runner, "run") as run:
            RUNNER.main()
        self.assertTrue(run.call_args.kwargs["allow_clean_stager_exit"])


class TestRescueFreezeAndFormal(unittest.TestCase):
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
            "capacity_decisions": 171,
            "blocked_stay_decisions_before_rescue": 170,
            "start_shadow_decisions": 1,
            "trigger_path": "CAPACITY_PILOT",
            "transition_states": ["SHADOW", "HANDOFF", "TAKEOVER"],
            "final_state": "TAKEOVER",
            "takeover_state": "COMMITTED",
            "source_abort_dispatched": True,
            "receiver_ranks": [0, 1, 2, 3],
            "exact_readback": [True, True, True, True],
            "source_origin_tokens": 5800,
            "target_origin_tokens": 2200,
            "anchor_emitted_tokens": 8000,
        }
        status = {
            "status": "BRINGUP_COMPLETE",
            "run_id": "rescue-bringup-1",
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
            root / "provenance" / "rescue_acceptance.json",
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
            self.assertEqual(frozen["scenario"], "CAP-0 Rescue formal")
            self.assertEqual(frozen["jobs"], manifest["jobs"])

            frozen_path = root / "rescue_v1.json"
            provenance_path = root / "rescue_v1.provenance.json"
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
            self.assertEqual(contract["bringup_run_id"], "rescue-bringup-1")

    def test_rejects_fewer_than_three_formal_repetitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            FORMAL.validate_inputs(Namespace(repetitions=2))


if __name__ == "__main__":
    unittest.main()
