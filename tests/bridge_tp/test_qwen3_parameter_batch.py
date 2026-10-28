# SPDX-License-Identifier: Apache-2.0
"""CPU orchestration and identity checks; real GPU mechanisms remain untested."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

from tools.bridge_tp import launch_pressure_balanced_pilot as launcher
from tools.bridge_tp import run_qwen3_parameter_batch as batch
from tools.bridge_tp import run_tp2_calibration as old
from tools.bridge_tp import run_tp_bundle_drain as drain
from tools.bridge_tp.freeze_pressure_baseline import cloud_identity
from tools.bridge_tp.qwen3_calibration_contract import adjust_migration_plan
from tools.bridge_tp.summarize_tp2_calibration import workload_from_cells


class BatchTests(unittest.TestCase):
    def test_transport_plan_keeps_full_speed_without_predictor(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    dict(
                        num_key_value_heads=8,
                        num_attention_heads=32,
                        num_hidden_layers=36,
                        hidden_size=4096,
                        head_dim=128,
                    )
                )
            )
            spec = old.prepare(model)
            spec["checkpoint"] = ""
            spec["checkpoint_sha256"] = ""
            common, args = launcher.arguments(spec)
            plan = launcher.build_scene(
                spec,
                dict(name="mechanism", arrivals=spec["scenarios"][0]["arrivals"]),
                Path(tmp),
                common,
                args,
            )
            adjust_migration_plan(plan)
            self.assertEqual(plan["config"]["transfer_mode"], "FULL_SPEED")
            self.assertEqual(plan["config"]["policy"]["kv_bytes_per_token"], 147456)
            for role in ("source", "target"):
                env = plan[role + "_env"]
                self.assertEqual(env["BRIDGETP_STREAM_TARGET_TP"], "2")
                self.assertFalse(any(k.startswith("BRIDGETP_PREDICTOR") for k in env))
                self.assertIn("--enforce-eager", plan[role + "_argv"])
            self.assertEqual(plan["source_env"]["BRIDGETP_STREAM_RATE_GIB_S"], "0")
            self.assertIn("--kv-transfer-config", plan["target_argv"])

    def test_disabled_predictor_identity_does_not_load_any_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, patch("torch.load") as load:
            result = cloud_identity(Path(tmp))
        load.assert_not_called()
        self.assertIsNone(result["predictor"])

    def test_combined_peak_uses_repeated_cells_only(self):
        cells = [
            dict(
                pool=role,
                context_bin=1,
                nd=n,
                window_goodoutput=dict(median=speed),
                health_status=status,
            )
            for role, n, speed, status in (
                ("source", 1, 50, "REFERENCE"),
                ("source", 2, 90, "REFERENCE"),
                ("target", 2, 120, "REFERENCE"),
                ("target", 4, 200, "REFERENCE"),
                ("target", 8, 1000, "DIAGNOSTIC_ONLY"),
            )
        ]
        workload, missing = workload_from_cells(cells, [])
        self.assertEqual(workload["source"]["capacities"], [[1, 2]])
        self.assertEqual(workload["target"]["capacities"], [[1, 4]])
        self.assertIn("target workload/prefill support", missing)

    def test_qwen3_drain_starts_tp2_with_16k_and_v1_eager(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                out_root=Path(tmp),
                base_port=8700,
                model_path=Path(tmp) / "model",
                platform="qwen3",
                startup_timeout_s=1,
            )
            common = Mock()
            common.server_command.return_value = ["server"]
            common.write_json.side_effect = lambda p, r: p.write_text(json.dumps(r))
            backend = drain.NativeDrainBackend(
                args,
                drain.pool.Registry(),
                common,
                {"requests": {"2048": {"prompt": [1] * 2048}}},
            )
            try:
                with patch.object(
                    drain.NativeDrainBackend,
                    "io",
                    new_callable=PropertyMock,
                    return_value=Mock(),
                ):
                    record = backend.start("A", 2, ["0", "1"])
                self.assertEqual(record.context_limit, 16384)
                settings = common.server_command.call_args.args[0]
                self.assertEqual(settings.tp4_max_model_len, 16384)
                call = common.start_process.call_args
                self.assertIn("--enforce-eager", call.args[1])
                self.assertEqual(call.args[2]["VLLM_USE_V2_MODEL_RUNNER"], "0")
                self.assertNotIn("BRIDGETP_TAKEOVER_ENABLED", call.args[2])
            finally:
                backend.executor.shutdown()

    def test_summary_keeps_single_costs_scoped_and_general_bounds_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            root.mkdir()
            pilot = Path(tmp) / "pilot"
            pilot.mkdir()
            batch.write(
                pilot / "service_profile.json", dict(cells=[], service_lookup=[])
            )
            d = root / "empty-L0-L1A"
            d.mkdir()
            batch.write(d / "result.json", dict(status="COMPLETED", runner_code=0))
            batch.write(
                d / "owned_shutdown_receipt.json", dict(all_owned_groups_ended=True)
            )
            batch.write(d / "plan.json", dict(state="COMPLETED"))
            (d / "lifecycle.jsonl").write_text(
                "\n".join(
                    json.dumps(e)
                    for e in [
                        dict(kind="RESOURCES_RELEASED", unix_s=10),
                        dict(kind="ROUTE_PUBLISHED", unix_s=15),
                    ]
                )
                + "\n"
            )
            m = root / "migration-128"
            m.mkdir()
            batch.write(
                m / "result.json",
                dict(
                    status="COMPLETED",
                    runner_code=0,
                    source_release_confirmed=True,
                    all_owned_groups_ended=True,
                    observed_generated_at_start=128,
                    start_to_release_s=1.0,
                    start_to_source_release_and_idle_observed_s=1.1,
                    visible_handoff_gap_s=0.2,
                ),
            )
            batch.write(m / "native_initial_receipt.json", [])
            result = batch.summary(root, pilot)
            self.assertEqual(
                result["empty_rebuild_observations"][0]["release_to_route_ready_s"], 5
            )
            self.assertEqual(
                result["migration_observations"][0][
                    "start_to_release_and_idle_observed_s"
                ],
                1.1,
            )
            self.assertIsNone(result["general_migration_release_s"])
            self.assertIsNone(result["capacity_guard_blocks"])
            self.assertFalse(result["deployment_ready"])

    def exercise(self, *, small_failure=False, cleanup_failure=False):
        hardware = dict(gpus=[dict(index=str(i), uuid=str(i)) for i in range(4)])
        assets = batch.read(
            batch.ROOT / "experiments/platforms/qwen3_8b_verified_assets.json"
        )["sha256"]

        def run(argv):
            option = "--output" if "--output" in argv else "--out-root"
            root = Path(argv[argv.index(option) + 1])
            root.mkdir()
            success = not (small_failure and root.name == "migration-128")
            (root / "result.json").write_text(
                json.dumps(
                    dict(
                        status="COMPLETED" if success else "TECHNICAL_FAILURE",
                        runner_code=0 if success else 1,
                    )
                )
            )
            if root.name == "service":
                (root / "attempt_ledger.json").write_text(
                    json.dumps([dict(status="COMPLETED")] * 7)
                )
            self.assertIn("--no-archive", argv)
            return 0 if success else 1

        def complete(path, kind):
            return batch.read(path / "result.json")["status"] == "COMPLETED"

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(batch, "require_revision"),
            patch.object(batch, "runtime_inventory", return_value=hardware),
            patch.object(
                batch,
                "gpu_idle",
                side_effect=[True, True, False] if cleanup_failure else None,
                return_value=True,
            ),
            patch.object(
                batch, "runtime_versions", return_value=dict(runtime="frozen")
            ),
            patch.object(batch, "accepted_scene"),
            patch.object(batch, "verified_inputs"),
            patch.object(
                batch,
                "inspect_model",
                return_value=dict(files={k: dict(sha256=v) for k, v in assets.items()}),
            ),
            patch.object(batch, "service_main", side_effect=run) as service,
            patch.object(batch, "drain_main", side_effect=run) as drain,
            patch.object(batch, "migration_main", side_effect=run) as migration,
            patch.object(batch, "complete", side_effect=complete),
            patch.object(batch, "summary", return_value=dict(deployment_ready=False)),
            patch.object(batch, "pack", return_value=dict(sha256="archive")),
        ):
            pilot = Path(tmp) / "pilot"
            pilot.mkdir()
            batch.write(
                pilot / "preflight.json",
                dict(
                    revision="f90c48b721dc3ab37ecb7db6b09b62ecf088f95b",
                    input_sha256=batch.INPUT_SHA,
                    versions=dict(runtime="frozen"),
                ),
            )
            batch.write(pilot / "hardware.json", hardware)
            batch.write(pilot / "service_profile.json", {})
            output = Path(tmp) / "batch"
            argv = [
                "--model",
                str(Path(tmp) / "model"),
                "--pilot-batch",
                str(pilot),
                "--output",
                str(output),
                "--expected-revision",
                "frozen",
            ]
            rc = batch.main(argv)
            result = batch.read(output / "result.json")
            if cleanup_failure:
                self.assertEqual(rc, 1)
                self.assertEqual(drain.call_count, 0)
                self.assertEqual(migration.call_count, 0)
                self.assertIn("cleanup", result["error"])
            elif small_failure:
                self.assertEqual(rc, 1)
                self.assertEqual(migration.call_count, 1)
                self.assertEqual(drain.call_count, 4)
                self.assertEqual(result["stages"][-1]["status"], "SKIPPED_DEPENDENCY")
                self.assertEqual(result["budget"]["migration_attempts"], 1)
            else:
                self.assertEqual(rc, 0)
                self.assertEqual(service.call_count, 1)
                self.assertEqual(drain.call_count, 4)
                self.assertEqual(migration.call_count, 2)
                self.assertEqual(result["budget"]["service_attempts"], 7)
                # Resume verifies completed steps, without another physical run.
                self.assertEqual(batch.main(argv + ["--resume"]), 0)
                self.assertEqual(service.call_count, 1)
                self.assertEqual(migration.call_count, 2)

    def test_batch_runs_both_costs_and_four_edges_then_resume_skips(self):
        self.exercise()

    def test_small_migration_failure_blocks_large_case(self):
        self.exercise(small_failure=True)

    def test_incomplete_cleanup_blocks_all_followup_actions(self):
        self.exercise(cleanup_failure=True)


if __name__ == "__main__":
    unittest.main()
