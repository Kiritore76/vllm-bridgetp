# SPDX-License-Identifier: Apache-2.0
"""Verify generated argv using actual pure helper/parser bodies on CPU."""

import argparse
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tools.bridge_tp import launch_pressure_balanced_pilot as launch

ROOT = Path(__file__).resolve().parents[2]


def functions(path, names, context):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), context
    )
    return context


class LauncherTests(unittest.TestCase):
    def test_git_bundle_records_runtime_head_and_resolves_repo_paths(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder).resolve()
            path = root / "pilot_spec.json"
            spec = dict(
                identity_mode="git-tracked-bundle-v1",
                policy_config="policy.json",
                service_evidence="evidence.json",
                controller_config_template="controller.json",
                scenarios=[dict(name="both-light", arrivals="arrivals.json")],
            )
            path.write_text(json.dumps(spec), encoding="utf-8")
            with (
                patch.object(launch, "ROOT", root),
                patch.object(launch.subprocess, "run") as tracked,
                patch.object(
                    launch.subprocess, "check_output", return_value="abc123\n"
                ),
            ):
                resolved = launch.load_spec(path)
            self.assertEqual(resolved["expected_head"], "abc123")
            self.assertEqual(resolved["bundle_spec_sha256"], launch.sha(path))
            self.assertEqual(resolved["policy_config"], str(root / "policy.json"))
            self.assertEqual(
                resolved["scenarios"][0]["arrivals"], str(root / "arrivals.json")
            )
            self.assertIn("--error-unmatch", tracked.call_args.args[0])
            self.assertNotIn("expected_head", json.loads(path.read_text()))

    def test_git_bundle_paths_cannot_escape_checkout(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder).resolve()
            path = root / "pilot_spec.json"
            path.write_text(
                json.dumps(
                    dict(
                        identity_mode="git-tracked-bundle-v1",
                        policy_config="../outside.json",
                    )
                )
            )
            with (
                patch.object(launch, "ROOT", root),
                patch.object(launch.subprocess, "run"),
                patch.object(
                    launch.subprocess, "check_output", return_value="abc123\n"
                ),
                self.assertRaises(ValueError),
            ):
                launch.load_spec(path)

    def test_legacy_frozen_external_spec_keeps_exact_head_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pilot_spec.json"
            path.write_text(
                json.dumps(dict(expected_head="fixed", policy_config="/old/path"))
            )
            self.assertEqual(launch.load_spec(path)["expected_head"], "fixed")

    def test_actual_transport_helpers_and_stager_parser_accept_full_argv(self):
        context = dict(
            argparse=argparse,
            Path=Path,
            os=os,
            json=json,
            model_kv_geometry=lambda *args: (8, 196608),
        )
        functions(
            "tools/bridge_tp/run_phase9_cap0_calibration.py",
            {"server_command", "source_environment", "target_connector"},
            context,
        )
        common = NS(
            **{
                key: context[key]
                for key in ("server_command", "source_environment", "target_connector")
            },
            STAGER=Path("phase8_stager.py"),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            template = root / "template.json"
            template.write_text("{}", encoding="utf-8")
            args = NS(
                python_bin=Path("python"),
                model_path=Path("qwen"),
                dtype="bfloat16",
                max_model_len=8192,
                tp4_max_model_len=16384,
                gpu_memory_utilization=0.88,
                tp1_gpu="0",
                tp4_gpus="1,2,3,4",
                snapshot_port=29800,
                delta_port=29900,
                delivery_port=30000,
                tp1_port=8001,
                tp4_port=8200,
                gpu_direct_base_port=30400,
                persistent_channel=True,
                channel_generation=1,
                preconnect_persistent_channel=True,
                gpu_direct_history=True,
                gpu_direct_delta=True,
                gpu_direct_delta_batch_tokens=16,
                gpu_direct_delta_flush_ms=25,
                ready_notification_port=30300,
            )
            spec = dict(
                controller_config_template=str(template),
                checkpoint="checkpoint.pt",
                checkpoint_sha256=(
                    "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7"
                ),
                policy_config="policy.json",
                service_evidence="evidence.json",
                run_timeout_s=900,
            )
            scene = dict(name="both-light", arrivals="arrivals.json")
            plan = launch.build_scene(spec, scene, root, common, args)
            target = plan["target_argv"]
            self.assertEqual(target[target.index("--max-model-len") + 1], "16384")
            connector = json.loads(target[-1])["kv_connector_extra_config"]
            self.assertEqual(connector["bridgetp_ready_sync_mode"], "STREAM_EVENT")
            self.assertEqual(connector["bridgetp_ready_notification_mode"], "UDP")
            self.assertTrue(connector["bridgetp_persistent_channel"])
            self.assertEqual(plan["source_env"]["BRIDGETP_PRESSURE_POOL"], "tp1")
            self.assertNotIn("BRIDGETP_PERSISTENT_CHANNEL", plan["target_env"])
            parser = functions(
                "tools/bridge_tp/phase8_stager.py",
                {"parse_args"},
                dict(argparse=argparse, Path=Path),
            )["parse_args"]
            parsed = parser(plan["stager_argv"][2:])
            self.assertTrue(parsed.gpu_resident_shadow)
            self.assertEqual(parsed.delta_base_port, 29900)
            controller_args = launch.controller.parse_args(plan["controller_argv"][2:])
            self.assertEqual(controller_args.policy, "pressure-balanced-v1")
            self.assertEqual(controller_args.source_context_limit, args.max_model_len)
            self.assertEqual(
                controller_args.target_context_limit, args.tp4_max_model_len
            )

    def test_normal_eos_with_retained_promise_requires_owner_shutdown_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "controller"
            run.mkdir()
            row = dict(
                kind="pressure_final",
                outcome="COMPLETED_PENDING_OWNER_CLEANUP",
                shutdown_cleanup_required=True,
                reservations=[dict(kind="migration", state="ACTIVE")],
            )
            (run / "pressure_audit.jsonl").write_text(json.dumps(row) + "\n")
            (root / "owned_shutdown_receipt.json").write_text(
                json.dumps(dict(all_owned_groups_ended=True))
            )
            result = launch.scene_result(run, "source-decode")
            self.assertEqual(result["status"], "COVERAGE_GAP")
            (root / "owned_shutdown_receipt.json").write_text(
                json.dumps(dict(all_owned_groups_ended=False))
            )
            with self.assertRaises(RuntimeError):
                launch.scene_result(run, "source-decode")

    def test_normal_finish_rejects_accounted_but_unretired_request(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "controller"
            run.mkdir()
            (run / "pressure_audit.jsonl").write_text(
                json.dumps(
                    dict(
                        kind="pressure_final",
                        outcome="COMPLETED",
                        reservations=[dict(kind="admission", state="ACCOUNTED")],
                    )
                )
                + "\n"
            )
            (root / "owned_shutdown_receipt.json").write_text(
                json.dumps(dict(all_owned_groups_ended=True))
            )
            with self.assertRaisesRegex(RuntimeError, "outstanding reservation"):
                launch.scene_result(run, "both-light")


if __name__ == "__main__":
    unittest.main()
