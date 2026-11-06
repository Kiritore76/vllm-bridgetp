# SPDX-License-Identifier: Apache-2.0
"""Six-card static planning and evidence checks on CPU, without CUDA."""

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp.test_pressure_launcher import functions
from tools.bridge_tp import launch_pressure_balanced_pilot as launch
from tools.bridge_tp import qwen3_six_card as six
from tools.bridge_tp import run_qwen3_stay_migrate as runner


def build(root, name):
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
            k: context[k]
            for k in ("server_command", "source_environment", "target_connector")
        },
        STAGER=Path("phase8_stager.py"),
    )
    template = root / "template.json"
    template.write_text("{}")
    args = NS(
        python_bin=Path("python"),
        model_path=Path("qwen"),
        dtype="bfloat16",
        max_model_len=32768,
        tp4_max_model_len=32768,
        gpu_memory_utilization=0.88,
        tp1_gpu="4" if name == "L2C" else "2",
        tp4_gpus="0,1",
        target_tp_size=2,
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
        checkpoint_sha256="a" * 64,
        policy_config="policy.json",
        service_evidence="evidence.json",
        run_timeout_s=900,
        layout=dict(
            name=name,
            bundle_gpus=list(six.GPUS),
            base_port=8300,
            migration_sources=name in six.MIXED,
        ),
    )
    plan = launch.build_scene(spec, dict(name=name, arrivals="a"), root, common, args)
    for side in ("source", "target"):
        plan[side + "_env"].update(
            BRIDGETP_PERFORMANCE_TRACE_DIR=str(
                plan["run_dir"] / "performance_trace" / side
            ),
            BRIDGETP_PERFORMANCE_FINALIZE_TOKEN=side + "-token",
        )
    six.configure(plan)
    return plan


class SixCardTests(unittest.TestCase):
    def test_all_layouts_cover_six_gpus_once_and_ports_are_unique(self):
        for name, degrees in (
            ("L0", [1] * 6),
            ("L1AC", [1] * 4 + [2]),
            ("L2C", [1, 1, 2, 2]),
            ("L3", [2] * 3),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                plan = build(Path(folder), name)
                rows = plan["instances"]["instances"]
                self.assertEqual(sorted(r["tp_degree"] for r in rows), degrees)
                devices = [
                    g
                    for r in plan["registry"].instances.values()
                    for g in r.physical_gpu_ids
                ]
                self.assertEqual(sorted(devices), [str(i) for i in range(6)])
                self.assertEqual(len({r["endpoint"] for r in rows}), len(rows))
                self.assertEqual(len(six.instances(plan)), len(rows))
                if name in six.MIXED:
                    self.assertNotIn("--no-migration", plan["controller_argv"])
                    self.assertEqual(
                        len(plan["runtime_target_files"]), degrees.count(1)
                    )
                    paths = [r["inventory_path"] for r in six.instances(plan)]
                    self.assertEqual(len(set(paths)), len(rows))
                else:
                    self.assertIn("--no-migration", plan["controller_argv"])

    def test_each_source_has_unique_warmup_identity_and_only_B_target(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "L1AC")
            self.assertEqual(len(plan["channel_sources"]), 4)
            files = plan["runtime_target_files"]
            self.assertEqual(len({v[0]["generation"] for v in files.values()}), 4)
            self.assertEqual({v[0]["target"] for v in files.values()}, {"B-g2-tp2"})
            self.assertNotIn("BRIDGETP_GPU_DIRECT_TARGETS", plan["source_env"])
            for extra in plan["extra_engines"]:
                _, environment = launch.extra_engine(plan, extra)
                self.assertEqual(
                    environment["BRIDGETP_GPU_DIRECT_TARGETS_FILE"],
                    extra["targets_file"],
                )
                self.assertNotIn("BRIDGETP_GPU_DIRECT_TARGETS", environment)
            rows = six.instances(plan)
            self.assertEqual(
                [r["finalize_token"] for r in rows].count("source-token"), 4
            )

    def test_single_class_trace_tokens_follow_actual_launch_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "L0")
            for row in six.instances(plan):
                self.assertEqual(
                    row["finalize_token"],
                    "source-token" if row["trace_name"] == "source" else "target-token",
                )

    def test_cli_six_card_modes_and_four_card_default_remain_distinct(self):
        common = [
            "--current-machine",
            "--repetitions",
            "1",
            "--model",
            "m",
            "--output",
            "unused",
            "--expected-revision",
            "CPU",
            "--validate-only",
        ]
        for name in six.LAYOUTS:
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                code = runner.main(
                    [
                        "--cards",
                        "6",
                        "--static-layout",
                        name,
                        "--long-burst-pair",
                        "--expanded-context",
                        *common,
                    ]
                )
            value = json.loads(stream.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(value["cards"], 6)
            self.assertEqual(value["runs"], 1)
            self.assertEqual(
                value["arms"], ["migrate" if name in six.MIXED else "stay"]
            )
        for args in (
            ["--static-layout", "L1AC", "--long-burst-pair"],
            ["--cards", "6", "--static-layout", "L1A", "--long-burst-pair"],
            [
                "--cards",
                "6",
                "--static-layout",
                "L1AC",
                "--long-burst-pair",
                "--expanded-context",
                "--conversion-decider",
                "execute",
            ],
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                runner.main([*args, *common])

    def test_acceptance_requires_every_instance_and_a_real_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "L1AC")
            root = plan["run_dir"]
            root.mkdir(parents=True)
            active = {
                r["instance_id"]: dict(lifecycle="ACTIVE", inventory={})
                for r in plan["instances"]["instances"]
            }
            path = root / "layout_status.json"
            path.write_text(json.dumps(dict(instances=active)))
            (root / "pressure_request_results.json").write_text(
                json.dumps(
                    [
                        dict(
                            job_id="r",
                            source_done=True,
                            source_error=None,
                            request=dict(max_tokens=1),
                            response=dict(emitted=[{}]),
                        )
                    ]
                )
            )
            proof = dict(sessions=[])
            self.assertFalse(six.assess(plan, proof, 1)["passed"])
            proof["sessions"] = [dict(state="COMMITTED", source_release={"ok": True})]
            self.assertTrue(six.assess(plan, proof, 1)["passed"])
            active.pop(next(iter(active)))
            path.write_text(json.dumps(dict(instances=active)))
            self.assertFalse(six.assess(plan, proof, 1)["passed"])

    def test_prepare_checks_all_sources_and_rank_generations(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "L1AC")
            receipts = {}
            for channel in plan["channel_sources"]:
                source = channel["source"]
                pool = "tp1" if source == "A-g1-tp1" else source
                for name in [f"source-{pool}-B-g2-tp2.json"] + [
                    f"tp_rank_{rank}-{source}.json" for rank in range(2)
                ]:
                    receipts[name] = dict(
                        status="WARMUP_COMPLETE",
                        channel_generation=channel["generation"],
                    )
            inventories = {
                "i": dict(context_limit=32768, free_blocks=2400, block_size=16)
            }
            with (
                patch.object(runner, "prepare_native"),
                patch.object(six, "wait_empty", return_value=inventories),
                patch.object(six.pc, "preconnect_receipts", return_value=receipts),
            ):
                six.prepare(plan, NS(write_json=lambda *args: None))
                receipts[next(iter(receipts))]["channel_generation"] = -1
                with (
                    patch.object(six.time, "monotonic", side_effect=[0, 31]),
                    self.assertRaisesRegex(TimeoutError, "warm channel"),
                ):
                    six.prepare(plan, NS(write_json=lambda *args: None))
