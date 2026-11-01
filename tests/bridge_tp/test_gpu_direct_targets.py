# SPDX-License-Identifier: Apache-2.0
"""S1-d2a CPU contracts: one TP1 source preconnected to several TP-k targets."""

import ast
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from vllm.bridge_tp.gpu_direct_targets import (
    GpuDirectTarget,
    TargetFileFollower,
    merge_targets,
    parse_targets,
    preconnect_plan,
    read_target_file,
    select_target,
)
from vllm.bridge_tp.runtime_control import RuntimeControl

KV_STREAM = Path(__file__).resolve().parents[2] / "vllm/bridge_tp/kv_stream.py"


@dataclass(frozen=True)
class Config:
    """The single-target fields that kv_stream's config carries."""

    gpu_direct_host: str = "127.0.0.1"
    gpu_direct_base_port: int = 30400
    target_tp_size: int = 2
    channel_generation: int = 1
    gpu_direct_targets: tuple = ()
    target_instance: str | None = None


def entry(target, base_port, tp=2, generation=1, host="127.0.0.1"):
    return dict(
        target=target, host=host, base_port=base_port, tp=tp, generation=generation
    )


TWO = json.dumps([entry("B-g2-tp2", 30400), entry("B-g5-tp2", 30410, generation=3)])


class ParseTests(unittest.TestCase):
    def test_absent_variable_keeps_single_target(self):
        self.assertEqual(parse_targets("", 8), ())
        self.assertEqual(parse_targets("  ", 8), ())

    def test_valid_targets(self):
        targets = parse_targets(TWO, 8)
        self.assertEqual(
            [(t.target, t.base_port, t.generation) for t in targets],
            [("B-g2-tp2", 30400, 1), ("B-g5-tp2", 30410, 3)],
        )
        self.assertEqual(list(targets[0].ports), [30400, 30401, 30402])

    def test_rejections(self):
        bad = [
            [],
            [entry("B-g2-tp2", 30400), entry("B-g2-tp2", 30410)],
            [entry("B-g2-tp2", 30400), entry("B-g5-tp2", 30402)],
            [entry("B-g2-tp2", 30400, tp=3)],
            [entry("b-g2-tp2", 30400)],
            [entry("B-g2-tp2", 80)],
            [entry("B-g2-tp2", 30400, generation=-1)],
            [entry("B-g2-tp2", 30400, host="")],
            [dict(entry("B-g2-tp2", 30400), extra=1)],
        ]
        for rows in bad:
            with self.assertRaises(ValueError, msg=rows):
                parse_targets(json.dumps(rows), 8)
        with self.assertRaises(ValueError):
            parse_targets(json.dumps([entry("B-g2-tp4", 30400, tp=4)]), 6)


class SelectTests(unittest.TestCase):
    def config(self):
        return Config(gpu_direct_targets=parse_targets(TWO, 8))

    def test_single_target_config_is_untouched(self):
        base = Config()
        self.assertIs(select_target(base, None, True), base)
        self.assertIs(select_target(base, "B-g2-tp2", True), base)
        self.assertEqual(preconnect_plan(base), [("source", base)])

    def test_session_target_overlays_single_target_fields(self):
        chosen = select_target(self.config(), "B-g5-tp2", True)
        self.assertEqual(
            (
                chosen.gpu_direct_base_port,
                chosen.channel_generation,
                chosen.target_tp_size,
                chosen.target_instance,
            ),
            (30410, 3, 2, "B-g5-tp2"),
        )

    def test_armed_session_must_name_a_preconnected_target(self):
        config = self.config()
        self.assertIs(select_target(config, None, False), config)
        with self.assertRaises(ValueError):
            select_target(config, None, True)
        with self.assertRaises(ValueError):
            select_target(config, "B-g9-tp2", False)

    def test_single_listed_target_is_implied(self):
        config = Config(
            gpu_direct_targets=parse_targets(json.dumps([entry("B-g2-tp2", 30410)]), 8)
        )
        chosen = select_target(config, None, True)
        self.assertEqual(
            (chosen.target_instance, chosen.gpu_direct_base_port), ("B-g2-tp2", 30410)
        )
        self.assertEqual(
            [label for label, _ in preconnect_plan(config)], ["source-B-g2-tp2"]
        )

    def test_preconnect_opens_every_target_with_its_own_receipt(self):
        plan = preconnect_plan(self.config())
        self.assertEqual(
            [
                (label, c.gpu_direct_base_port, c.channel_generation)
                for label, c in plan
            ],
            [("source-B-g2-tp2", 30400, 1), ("source-B-g5-tp2", 30410, 3)],
        )

    def test_targets_get_distinct_persistent_sender_keys(self):
        tree = ast.parse(KV_STREAM.read_text(encoding="utf-8"))
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_persistent_sender_key"
        )
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        ns = {}
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), str(KV_STREAM), "exec"), ns
        )

        class Device:
            index = 0

        keys = {
            ns["_persistent_sender_key"](c, Device())
            for _, c in preconnect_plan(self.config())
        }
        self.assertEqual(len(keys), 2)


class WiringTests(unittest.TestCase):
    def test_kv_stream_uses_the_target_helpers(self):
        tree = ast.parse(KV_STREAM.read_text(encoding="utf-8"))
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

        def calls(name):
            return {
                n.func.id
                for n in ast.walk(functions[name])
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }

        self.assertIn("select_target", calls("get_bridge_tp_stream_config"))
        self.assertIn("preconnect_plan", calls("preconnect_persistent_gpu_sender"))
        self.assertIn("_preconnect_one", calls("preconnect_persistent_gpu_sender"))

    def test_target_file_is_followed_beside_serving(self):
        """S5b: kv_stream starts the follower and merges file targets."""
        tree = ast.parse(KV_STREAM.read_text(encoding="utf-8"))
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

        def calls(name):
            return {
                n.func.id
                for n in ast.walk(functions[name])
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }

        self.assertIn(
            "_start_target_file_follower", calls("preconnect_persistent_gpu_sender")
        )
        self.assertIn("merge_targets", calls("get_bridge_tp_stream_config"))
        follower = calls("_start_target_file_follower")
        self.assertIn("TargetFileFollower", follower)

    def test_runtime_control_carries_the_target_instance(self):
        with tempfile.TemporaryDirectory() as folder:
            RuntimeControl(armed=True, target_instance="B-g5-tp2").write(folder)
            self.assertEqual(RuntimeControl.load(folder).target_instance, "B-g5-tp2")
            RuntimeControl(armed=False).write(folder)
            self.assertIsNone(RuntimeControl.load(folder).target_instance)

    def test_dataclass_target_is_hashable_and_frozen(self):
        target = GpuDirectTarget("B-g2-tp2", "127.0.0.1", 30400, 2, 1)
        with self.assertRaises(AttributeError):
            target.base_port = 1


def row(target="B-g6-tp2", base_port=30400, generation=1):
    return dict(
        target=target,
        host="127.0.0.1",
        base_port=base_port,
        tp=2,
        generation=generation,
    )


class TargetFileTests(unittest.TestCase):
    """S5b: a source's targets follow the conversion's file."""

    def test_missing_or_empty_file_has_no_target(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "targets.json"
            self.assertEqual(read_target_file(path, 8), ())
            path.write_text("[]")
            self.assertEqual(read_target_file(path, 8), ())
            path.write_text(json.dumps([row()]))
            (target,) = read_target_file(path, 8)
            self.assertEqual((target.target, target.generation), ("B-g6-tp2", 1))
            path.write_text(json.dumps([row(), row()]))
            with self.assertRaises(ValueError):
                read_target_file(path, 8)

    def test_merge_keeps_startup_entries(self):
        listed = parse_targets(json.dumps([row("B-g2-tp2")]), 8)
        runtime = parse_targets(json.dumps([row("B-g2-tp2", 30500), row()]), 8)
        merged = merge_targets(listed, runtime)
        self.assertEqual([t.target for t in merged], ["B-g2-tp2", "B-g6-tp2"])
        self.assertEqual(merged[0].base_port, 30400)

    def test_follower_opens_new_and_closes_departed_idle_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "targets.json"
            calls, busy = [], set()

            def drop(target):
                calls.append(("drop", target.target))
                return target.target not in busy

            follower = TargetFileFollower(
                path, 8, lambda t: calls.append(("connect", t.target)), drop
            )
            self.assertEqual(follower.poll()["opened"], [])
            path.write_text(json.dumps([row("B-g9-tp2", 30410), row()]))
            # Id order, as the target's handshakes.
            self.assertEqual(follower.poll()["opened"], ["B-g6-tp2", "B-g9-tp2"])
            self.assertEqual(follower.poll()["opened"], [])
            busy.add("B-g6-tp2")
            path.write_text(json.dumps([row("B-g9-tp2", 30410)]))
            self.assertEqual(follower.poll()["deferred"], ["B-g6-tp2"])
            busy.clear()
            self.assertEqual(follower.poll()["closed"], ["B-g6-tp2"])
            self.assertEqual(set(follower.connected), {"B-g9-tp2"})
            # A half-written file is retried, not applied.
            path.write_text("[{")
            self.assertEqual(follower.poll(), dict(opened=[], closed=[], deferred=[]))
            self.assertIsNotNone(follower.error)
            path.write_text(json.dumps([row("B-g9-tp2", 30420)]))
            with self.assertRaises(ValueError):
                follower.poll()
            self.assertEqual(
                calls,
                [
                    ("connect", "B-g6-tp2"),
                    ("connect", "B-g9-tp2"),
                    ("drop", "B-g6-tp2"),
                    ("drop", "B-g6-tp2"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
