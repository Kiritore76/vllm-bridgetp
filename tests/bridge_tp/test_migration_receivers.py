# SPDX-License-Identifier: Apache-2.0
"""Scoped receiver stage: CPU contracts, without CUDA or NCCL."""

import json
import os
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp.test_gpu_direct_targets import Config, entry
from tests.bridge_tp.test_pressure_launcher import functions
from tests.bridge_tp.test_runtime_instance_ids import canonical_plan
from tools.bridge_tp import launch_pressure_balanced_pilot as launch
from tools.bridge_tp import migration_receivers as receivers
from tools.bridge_tp import qwen3_six_card as six
from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp.pressure_sessions import publish_active
from vllm.bridge_tp.gpu_direct_targets import apply_target, parse_targets, select_target
from vllm.bridge_tp.request_freeze import RequestFreezeGate
from vllm.bridge_tp.runtime_control import RuntimeControl


def build(root):
    plan = canonical_plan(root, "L2C")
    plan["controller_argv"] += ["--maximum-migrations", "0"]
    plan["stager_argv"][1] = "tools/bridge_tp/pressure_sessions.py"
    receivers.configure(plan)
    return plan


@dataclass(frozen=True)
class ScopedConfig(Config):
    run_dir: Path | None = None
    gpu_direct_mechanism_root: Path | None = None


@dataclass(frozen=True)
class StreamConfig(ScopedConfig):
    enabled: bool = True
    armed: bool = False
    migration_id: str = "before"
    source_request_id_prefix: str = "q"
    after_output_tokens: int = 1
    phase8_cutover_output_tokens: int = 2
    aggregate_rate_gib_s: float = 0

    def validate_boundaries(self):
        pass


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class ReceiverTests(unittest.TestCase):
    def initialize(self, plan):
        try:
            return receivers.initialize(plan)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows symlink privilege unavailable; Linux required")
            raise

    def test_source_stream_control_and_honored_use_the_same_B_session(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder))
            plan["run_dir"].mkdir(parents=True)
            # Directory selection/control discovery can run on Windows too;
            # the separate entry/rotation tests require actual symlinks.
            for target in plan["migration_targets"]:
                Path(target["mechanism_dir"]).mkdir(parents=True)
            session = Path(plan["migration_targets"][0]["mechanism_dir"])
            link = session
            targets = parse_targets(
                json.dumps(next(iter(plan["runtime_target_files"].values()))), 8
            )
            cfg = StreamConfig(
                run_dir=link,
                gpu_direct_targets=targets,
                gpu_direct_mechanism_root=plan["run_dir"] / "migration",
            )
            RuntimeControl(
                armed=True, migration_id="B-active", target_instance="B-g2-tp2"
            ).write(session)
            honored = []
            context = dict(
                os=NS(getenv=lambda key: None),
                replace=replace,
                get_bridge_tp_stream_env_config=lambda: cfg,
                GPU_DIRECT_TARGETS_FILE_ENV="unused",
                _control_cache=lambda path: NS(get=lambda: RuntimeControl.load(path)),
                select_target=select_target,
                BridgeTPStreamConfig=StreamConfig,
                mark_control_honored=lambda path, generation: honored.append(path),
            )
            functions(
                "vllm/bridge_tp/kv_stream.py", {"get_bridge_tp_stream_config"}, context
            )
            selected = context["get_bridge_tp_stream_config"]()
            self.assertEqual(selected.run_dir, session.resolve())
            self.assertEqual(selected.migration_id, "B-active")
            self.assertEqual(Path(honored[0]).resolve(), selected.run_dir)
            self.assertEqual(selected.target_instance, "B-g2-tp2")

    def test_source_API_operation_remains_pinned_after_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder))
            plan["run_dir"].mkdir(parents=True)
            first = self.initialize(plan)
            link = next(iter(plan["source_links"].values()))
            write(first / "session_manifest.json", dict(migration_id="first"))
            context = dict(
                Path=Path, os=os, _load_json=lambda p: json.loads(p.read_text())
            )
            functions(
                "vllm/bridge_tp/takeover_api.py", {"_configured_session"}, context
            )
            with patch.dict(
                os.environ,
                BRIDGETP_TAKEOVER_RUN_DIR=link,
                BRIDGETP_TAKEOVER_PERSISTENT_CHANNEL="1",
            ):
                pinned, identity = context["_configured_session"]()
                second = first.parent / "migration-0002"
                second.mkdir()
                write(second / "session_manifest.json", dict(migration_id="second"))
                publish_active(first.parent.parent, second)
                self.assertEqual((pinned, identity), (first.resolve(), "first"))
                self.assertEqual(
                    context["_configured_session"](), (second.resolve(), "second")
                )

    def test_four_pairs_have_separate_directories_and_ports(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder))
            targets = plan["migration_targets"]
            self.assertEqual([t["bundle"] for t in targets], ["B", "A"])
            self.assertEqual([t["execution_allowed"] for t in targets], [True, False])
            self.assertEqual(len({t["mechanism_root"] for t in targets}), 2)
            self.assertEqual(len({t["ready_notification_port"] for t in targets}), 2)
            ports = [c["base_port"] for t in targets for c in t["connector_sources"]]
            self.assertEqual(ports, [30400, 30410, 30500, 30510])
            self.assertEqual(len(plan["source_links"]), 2)
            for rows in plan["runtime_target_files"].values():
                self.assertEqual(
                    {r["target"] for r in rows}, {t["instance_id"] for t in targets}
                )
                self.assertEqual(len(parse_targets(json.dumps(rows), 8)), 2)
            self.assertEqual(plan["controller_argv"][-2:], ["--migration-targets", "B"])

    def test_extra_receiver_keeps_connector_and_extra_source_gets_own_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder))
            for extra in plan["extra_engines"]:
                argv, env = launch.extra_engine(plan, extra)
                if extra["tp"] == 2:
                    self.assertIn("--kv-transfer-config", argv)
                    connector = json.loads(argv[argv.index("--kv-transfer-config") + 1])
                    settings = connector["kv_connector_extra_config"]
                    self.assertEqual(len(settings["bridgetp_gpu_direct_sources"]), 2)
                    self.assertFalse(
                        settings["bridgetp_release_persistent_payload_buffer"]
                    )
                    self.assertIn(
                        extra["instance_id"], env["BRIDGETP_TAKEOVER_RUN_DIR"]
                    )
                else:
                    self.assertEqual(
                        env["BRIDGETP_STREAM_RUN_DIR"],
                        plan["source_links"][extra["instance_id"]],
                    )
                    self.assertEqual(
                        env["BRIDGETP_TAKEOVER_RUN_DIR"], env["BRIDGETP_STREAM_RUN_DIR"]
                    )
                    self.assertEqual(
                        env["BRIDGETP_GPU_DIRECT_MECHANISM_ROOT"],
                        str(plan["run_dir"] / "migration"),
                    )

    def test_A_B_match_only_after_allowing_owned_paths_and_ports(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder))
            b, a = plan["migration_targets"]
            extra = next(e for e in plan["extra_engines"] if e["tp"] == 2)
            argv, _ = launch.extra_engine(plan, extra)
            normalized = receivers.normalized_receiver_command
            self.assertEqual(normalized(argv, a), normalized(plan["target_argv"], b))
            changed = list(argv)
            changed[changed.index("--max-model-len") + 1] = "16384"
            self.assertNotEqual(
                normalized(changed, a), normalized(plan["target_argv"], b)
            )

    def test_wrong_layout_or_transport_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = canonical_plan(Path(folder), "L1AC")
            plan["controller_argv"] += ["--maximum-migrations", "0"]
            with self.assertRaisesRegex(ValueError, "L2C"):
                receivers.configure(plan)
        with tempfile.TemporaryDirectory() as folder:
            plan = canonical_plan(Path(folder), "L2C")
            plan["controller_argv"] += ["--maximum-migrations", "0"]
            plan["source_env"]["BRIDGETP_GPU_DIRECT_DELTA"] = "0"
            with self.assertRaisesRegex(ValueError, "GPU-direct"):
                receivers.configure(plan)

    def test_all_twelve_receipts_are_required_in_their_own_target(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder))
            rows = {}
            for target in plan["migration_targets"]:
                receipts = {}
                for source in target["connector_sources"]:
                    names = [f"source-{source['source']}-{target['instance_id']}.json"]
                    names += [f"tp_rank_{r}-{source['source']}.json" for r in range(2)]
                    for name in names:
                        receipts[name] = dict(
                            status="WARMUP_COMPLETE",
                            channel_generation=source["generation"],
                        )
                rows[Path(target["mechanism_root"])] = receipts
            self.assertEqual(len(receivers.warm_channels(plan, rows.__getitem__)), 12)
            a = Path(plan["migration_targets"][1]["mechanism_root"])
            removed = rows[a].pop(next(iter(rows[a])))
            rows[Path(plan["migration_targets"][0]["mechanism_root"])]["wrong.json"] = (
                removed
            )
            with self.assertRaises(TimeoutError):
                receivers.warm_channels(plan, rows.__getitem__, timeout_s=0)

    def test_source_entries_follow_B_rotation_and_cleanup_preserves_sessions(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder))
            plan["run_dir"].mkdir(parents=True)
            try:
                first = receivers.initialize(plan)
            except OSError as error:
                if getattr(error, "winerror", None) == 1314:
                    self.skipTest(
                        "Windows symlink privilege unavailable; Linux required"
                    )
                raise
            links = [Path(p) for p in plan["source_links"].values()]
            self.assertTrue(all(p.resolve() == first for p in links))
            pinned = links[0].resolve(strict=True)
            write(
                first / "request_freeze_control.json",
                dict(action="FREEZE", request_id="old"),
            )
            gate = RequestFreezeGate(links[0])
            self.assertTrue(gate.is_frozen("old"))
            second = first.parent / "migration-0002"
            second.mkdir()
            publish_active(first.parent.parent, second)
            self.assertTrue(all(p.resolve() == second for p in links))
            self.assertFalse(gate.is_frozen("old"))
            self.assertEqual(pinned, first)
            receivers.cleanup_links(plan, write)
            self.assertTrue(first.is_dir() and second.is_dir())
            self.assertTrue(all(not p.is_symlink() for p in links))
            self.assertTrue(
                all(
                    not Path(t["mechanism_dir"]).is_symlink()
                    for t in plan["migration_targets"]
                )
            )

    def test_directory_selection_is_owned_and_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            owned = root / "migration"
            selected = owned / "B-g2-tp2" / "sessions" / "migration-0001"
            selected.mkdir(parents=True)
            target = parse_targets(
                json.dumps(
                    [dict(entry("B-g2-tp2", 30400), mechanism_dir=str(selected))]
                ),
                8,
            )[0]
            cfg = ScopedConfig(run_dir=root, gpu_direct_mechanism_root=owned)
            self.assertEqual(apply_target(cfg, target).run_dir, selected.resolve())
            with self.assertRaisesRegex(ValueError, "owned mechanism root"):
                apply_target(replace(cfg, gpu_direct_mechanism_root=None), target)
            with self.assertRaisesRegex(ValueError, "escaped"):
                apply_target(cfg, replace(target, mechanism_dir=str(root)))
            self.assertEqual(
                apply_target(cfg, replace(target, mechanism_dir=None)).run_dir, root
            )

    def test_full_speed_checks_both_target_roots(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / "pressure_audit.jsonl", dict(kind="no_migration"))
            roots = [root / "migration" / name for name in ("A", "B")]
            for target in roots:
                session = target / "sessions" / "migration-0001"
                session.mkdir(parents=True)
                write(session / "runtime_control.json", dict(rate_gib_s=0))
            self.assertEqual(launch.verify_full_speed(root, roots)["actual_starts"], 0)
            write(
                roots[0] / "sessions/migration-0001/runtime_control.json",
                dict(rate_gib_s=1),
            )
            with self.assertRaisesRegex(RuntimeError, "rate limit"):
                launch.verify_full_speed(root, roots)

    def test_proof_rejects_even_cancelled_migration_to_A(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder))
            plan["run_dir"].mkdir(parents=True)
            target = plan["migration_targets"][1]
            session = Path(target["mechanism_root"]) / "sessions" / "migration-0001"
            session.mkdir(parents=True)
            write(session / "takeover_state.json", dict(state="CANCELLED"))
            audit = [
                dict(kind="pressure_actual_start"),
                dict(
                    kind="pressure_final",
                    outcome="COMPLETED",
                    sessions=[
                        dict(
                            started=True,
                            run_dir=str(session),
                            migration_id="a",
                            settled=True,
                        )
                    ],
                ),
            ]
            (plan["run_dir"] / "pressure_audit.jsonl").write_text(
                "\n".join(json.dumps(r) for r in audit)
            )
            with self.assertRaisesRegex(ValueError, "B receiver root"):
                runner.closed_loop_proof(plan, "migrate")

    def test_stage_two_zero_migration_is_not_claimed_as_migration_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder))
            active = {
                r["instance_id"]: dict(lifecycle="ACTIVE", inventory={})
                for r in plan["instances"]["instances"]
            }
            requests = [
                dict(
                    job_id="one",
                    source_done=True,
                    source_error=None,
                    response=dict(emitted=[1]),
                    request=dict(max_tokens=1),
                )
            ]

            def read(path):
                return (
                    dict(instances=active)
                    if path.name == "layout_status.json"
                    else requests
                )

            with patch.object(six.pc, "read_json", side_effect=read):
                result = six.assess(plan, dict(sessions=[]), 1)
                self.assertTrue(result["passed"])
                self.assertFalse(result["committed_migration_observed"])
                self.assertFalse(result["migration_required"])
                with self.assertRaisesRegex(ValueError, "outside B"):
                    six.assess(
                        plan,
                        dict(
                            sessions=[
                                dict(state="CANCELLED", target_instance="A-g3-tp2")
                            ]
                        ),
                        1,
                    )


if __name__ == "__main__":
    unittest.main()
