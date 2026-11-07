# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for canonical plans and the unchanged single-B bridge."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests.bridge_tp.test_qwen3_six_card import build
from tools.bridge_tp import migration_plan as mp


def snapshot(plan):
    # Registry owns an RLock; copy the launch data, retain the registry identity.
    return {
        key: value if key == "registry" else copy.deepcopy(value)
        for key, value in plan.items()
    }


class PortTests(unittest.TestCase):
    def allocate(self, **overrides):
        defaults = dict(gpu=30400, delta=29900, delivery=30000, ready=30300)
        defaults.update(overrides)
        return mp.allocate_ports(
            ["C-g1-tp1", "C-g4-tp1"],
            [("B-g2-tp2", 2), ("A-g3-tp2", 2)],
            **defaults,
        )

    def test_two_targets_have_distinct_blocks_and_keep_B_ports(self):
        targets, blocks = self.allocate()
        self.assertEqual(
            [[c["base_port"] for c in t["connector_sources"]] for t in targets],
            [[30400, 30410], [30500, 30510]],
        )
        self.assertEqual(
            [t["ready_notification_port"] for t in targets], [30300, 30301]
        )
        flat = [port for b in blocks for port in b["ports"]]
        self.assertEqual(len(flat), len(set(flat)))

    def test_gpu_sender_control_port_is_reserved(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.allocate(ready=30402)

    def test_cross_family_collisions_are_rejected(self):
        for overrides in (
            dict(delta=30000),
            dict(ready=29901),
            dict(reserved=[dict(owner="http", ports=[30512])]),
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.allocate(**overrides)

    def test_port_bounds_include_entire_rank_range(self):
        for overrides in (dict(gpu=65535), dict(ready=1023), dict(delivery=65535)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.allocate(**overrides)

    def test_repeated_identity_or_bad_tp_is_rejected(self):
        for sources, targets in (
            (["s", "s"], [("t", 2)]),
            (["s"], [("t", 2), ("t", 2)]),
            (["s"], [("s", 2)]),
            (["s"], [("t", 3)]),
            (["s"], [("t", True)]),
        ):
            with (
                self.subTest(sources=sources, targets=targets),
                self.assertRaises(ValueError),
            ):
                mp.allocate_ports(
                    sources,
                    targets,
                    gpu=30400,
                    delta=29900,
                    delivery=30000,
                    ready=30300,
                )


class CompatibilityTests(unittest.TestCase):
    def test_multi_target_configuration_fails_before_launch(self):
        for scheduler in ("pressure", "unified"):
            with self.subTest(scheduler=scheduler), self.assertRaises(ValueError):
                mp.install(
                    dict(
                        scheduler=scheduler, static_cards=6, migration_targets=[{}, {}]
                    )
                )

    def test_install_changes_no_existing_plan_fields(self):
        for layout, source_count in (("L1AC", 4), ("L2C", 2)):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as folder:
                plan = build(Path(folder), layout)
                plan["scheduler"] = "unified"
                before = snapshot(plan)
                mp.install(plan)
                additions = set(plan) - set(before)
                self.assertEqual(
                    additions,
                    {"migration_plan", "migration_sources", "migration_targets"},
                )
                self.assertEqual({k: plan[k] for k in before}, before)
                description = plan["migration_plan"]
                self.assertEqual(len(description["migration_sources"]), source_count)
                target = description["migration_targets"][0]
                self.assertEqual(target["instance_id"], "B-g2-tp2")
                self.assertEqual(target["runtime_pool_id"], "tp4")
                self.assertEqual(target["stager_argv"], before["stager_argv"])
                self.assertEqual(
                    target["mechanism_dir"],
                    before["target_env"]["BRIDGETP_TAKEOVER_RUN_DIR"],
                )
                self.assertFalse(description["multi_target_execution"])
                self.assertEqual(json.loads(json.dumps(description)), description)

    def test_pressure_and_four_card_paths_are_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "L2C")
            before = snapshot(plan)
            mp.install(plan)
            self.assertEqual(plan, before)
            plan.update(scheduler="unified", static_cards=4)
            before = snapshot(plan)
            mp.install(plan)
            self.assertEqual(plan, before)

    def test_single_class_layouts_have_no_migration_endpoints(self):
        for layout in ("L0", "L3"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as folder:
                plan = build(Path(folder), layout)
                plan["scheduler"] = "unified"
                mp.install(plan)
                self.assertEqual(plan["migration_targets"], [])
                self.assertEqual(plan["migration_sources"], [])

    def test_mismatched_source_file_generation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "L2C")
            next(iter(plan["runtime_target_files"].values()))[0]["generation"] += 1
            with self.assertRaisesRegex(ValueError, "identity differs"):
                mp.describe(plan)

    def test_source_file_must_not_add_A_yet(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "L2C")
            entry = next(iter(plan["runtime_target_files"].values()))
            entry.append(dict(entry[0], target="A-g3-tp2"))
            with self.assertRaisesRegex(ValueError, "B only"):
                mp.describe(plan)

    def test_stager_directory_must_match_receiver(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "L2C")
            index = plan["stager_argv"].index("--run-dir") + 1
            plan["stager_argv"][index] = "different"
            with self.assertRaisesRegex(ValueError, "directories differ"):
                mp.describe(plan)

    def test_existing_channel_ports_must_match_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build(Path(folder), "L2C")
            plan["channel_sources"][-1]["base_port"] += 1
            with self.assertRaisesRegex(ValueError, "existing single-B ports"):
                mp.describe(plan)
