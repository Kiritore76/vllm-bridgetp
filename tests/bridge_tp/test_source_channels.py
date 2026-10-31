# SPDX-License-Identifier: Apache-2.0
"""S1-d2b CPU contracts: per-source receivers on one target rank."""

import unittest
from types import SimpleNamespace as NS

from vllm.bridge_tp.source_channels import SourceChannelSet, parse_plan


class Receiver:
    def __init__(self, channel, log):
        self.channel, self.log = channel, log
        self.lifecycle = NS(state=NS(value="IDLE"), active_session=None)
        log.append(("bind", channel.source))

    def open(self):
        self.log.append(("open", self.channel.source))

    def busy(self):
        self.lifecycle = NS(state=NS(value="ACTIVE"), active_session=object())

    def idle(self):
        self.lifecycle = NS(state=NS(value="IDLE"), active_session=None)

    def close(self):
        self.log.append(("close", self.channel.source))


def row(source, base_port, generation):
    return dict(source=source, base_port=base_port, generation=generation)


A = row("A-g1-tp1", 30400, 1)
A3 = row("A-g3-tp1", 30410, 3)


class PlanTests(unittest.TestCase):
    def test_valid_and_rejected_plans(self):
        plan = parse_plan([A, A3], 2)
        self.assertEqual([c.generation for c in plan], [1, 3])
        self.assertEqual(parse_plan([], 2), ())
        for bad in (
            [A, row("A-g1-tp1", 30410, 3)],
            [A, row("A-g3-tp1", 30410, 1)],
            [A, row("A-g3-tp1", 30402, 3)],
            [row("B-g2-tp2", 30400, 1)],
            [row("A-g1-tp1", 30400, -1)],
            [dict(A, extra=1)],
            "A",
        ):
            with self.assertRaises(ValueError, msg=bad):
                parse_plan(bad, 2)


class ChannelSetTests(unittest.TestCase):
    def setUp(self):
        self.log = []
        self.channels = SourceChannelSet(
            lambda c: Receiver(c, self.log), lambda r, c: r.open()
        )

    def test_opens_every_source_in_id_order_and_selects_by_generation(self):
        result = self.channels.reconcile(parse_plan([A3, A], 2))
        self.assertEqual(result["opened"], [1, 3])
        # Every listener is bound before the first handshake.
        self.assertEqual(
            self.log,
            [
                ("bind", "A-g1-tp1"),
                ("bind", "A-g3-tp1"),
                ("open", "A-g1-tp1"),
                ("open", "A-g3-tp1"),
            ],
        )
        receiver = self.channels.select(3)
        self.assertEqual(receiver.channel.source, "A-g3-tp1")
        self.assertIs(self.channels.select(3), receiver)
        with self.assertRaises(ValueError):
            self.channels.select(9)

    def test_no_switch_while_the_active_session_runs(self):
        self.channels.reconcile(parse_plan([A, A3], 2))
        self.channels.select(1).busy()
        with self.assertRaises(RuntimeError):
            self.channels.select(3)
        self.channels.channels[1][1].idle()
        self.assertEqual(self.channels.select(3).channel.generation, 3)

    def test_retiring_a_source_closes_only_its_idle_receiver(self):
        self.channels.reconcile(parse_plan([A, A3], 2))
        self.channels.select(1).busy()
        result = self.channels.reconcile(parse_plan([A3], 2))
        self.assertEqual(result, dict(opened=[], closed=[], deferred=[1]))
        with self.assertRaises(ValueError):
            self.channels.select(1)
        self.channels.channels[1][1].idle()
        result = self.channels.reconcile(parse_plan([A3], 2))
        self.assertEqual(result["closed"], [1])
        self.assertIsNone(self.channels.active_generation)
        self.assertNotIn(("close", "A-g3-tp1"), self.log)
        self.assertEqual(sorted(self.channels.channels), [3])

    def test_rebuild_adds_a_new_generation_without_touching_others(self):
        self.channels.reconcile(parse_plan([A3], 2))
        before = self.channels.channels[3][1]
        result = self.channels.reconcile(parse_plan([A3, row("A-g7-tp1", 30420, 7)], 2))
        self.assertEqual(result["opened"], [7])
        self.assertIs(self.channels.channels[3][1], before)

    def test_a_generation_cannot_change_its_source_or_port(self):
        self.channels.reconcile(parse_plan([A], 2))
        with self.assertRaises(ValueError):
            self.channels.reconcile(parse_plan([row("A-g1-tp1", 30500, 1)], 2))

    def test_failed_handshake_releases_every_new_listener(self):
        def fail_second(receiver, channel):
            if channel.generation == 3:
                raise RuntimeError("handshake failed")

        channels = SourceChannelSet(lambda c: Receiver(c, self.log), fail_second)
        with self.assertRaises(RuntimeError):
            channels.reconcile(parse_plan([A, A3], 2))
        self.assertIn(("close", "A-g3-tp1"), self.log)
        self.assertNotIn(("close", "A-g1-tp1"), self.log)
        self.assertEqual(sorted(channels.channels), [1])

    def test_close_all(self):
        self.channels.reconcile(parse_plan([A, A3], 2))
        self.channels.close_all()
        self.assertEqual(self.channels.channels, {})
        self.assertEqual(sum(1 for op, _ in self.log if op == "close"), 2)


if __name__ == "__main__":
    unittest.main()


class ConnectorWiringTests(unittest.TestCase):
    """The real connector methods, run against a stand-in connector object."""

    PATH = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "vllm/bridge_tp/streaming_connector.py"
    )

    def connector_class(self):
        import ast

        tree = ast.parse(self.PATH.read_text(encoding="utf-8"))
        return next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "BridgeTPStreamingConnector"
        )

    def method(self, name):
        import ast
        import threading

        cls = self.connector_class()
        fn = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        ns = dict(threading=threading)
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), str(self.PATH), "exec"),
            ns,
        )
        return ns[name]

    def module_function(self, name):
        """A module-level helper of the real connector file."""
        import ast
        import json

        tree = ast.parse(self.PATH.read_text(encoding="utf-8"))
        fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        ns = dict(json=json)
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), str(self.PATH), "exec"),
            ns,
        )
        return ns[name]

    def test_retired_receiver_leaves_the_set_before_its_teardown(self):
        """S3b-3 GPU (10-31): teardown runs through ``retire`` (background
        in the connector), never as a blocking close inside reconcile."""
        log, retired = [], []
        channels = None

        def retire(receiver, channel):
            retired.append((channel.generation, set(channels.channels)))

        channels = SourceChannelSet(
            lambda c: Receiver(c, log), lambda r, c: r.open(), retire
        )
        channels.reconcile(parse_plan([A, A3], 2))
        channels.channels[3][1].busy()
        result = channels.reconcile(parse_plan([], 2))
        self.assertEqual((result["closed"], result["deferred"]), ([1], [3]))
        self.assertEqual(retired, [(1, {3})])  # already unselectable
        self.assertNotIn(("close", "A-g1-tp1"), log)
        channels.channels[3][1].idle()
        channels.reconcile(parse_plan([], 2))
        self.assertEqual([g for g, _ in retired], [1, 3])
        self.assertEqual(channels.channels, {})

    def test_connector_retires_a_source_channel_on_a_background_thread(self):
        import json
        import tempfile
        import threading
        import time
        from pathlib import Path

        retire = self.method("_retire_source_receiver")
        retire.__globals__.update(
            time=time,
            logger=NS(warning=lambda *a: None, exception=lambda *a: None),
            _atomic_json_dump=lambda value, path: (
                path.parent.mkdir(parents=True, exist_ok=True),
                path.write_text(json.dumps(value)),
            ),
        )
        release = threading.Event()

        class Blocking:
            aborted = False

            def abort(self):
                release.wait(5)
                self.aborted = True

            def close(self):
                raise AssertionError("synchronous destroy must not be used")

        with tempfile.TemporaryDirectory() as folder:
            connector = NS(manifest_path=Path(folder) / "s" / "manifest.json")
            receiver = Blocking()
            channel = parse_plan([A], 2)[0]
            retire(connector, receiver, channel, 1)  # returns while abort blocks
            path = Path(folder) / "s/gpu_channel_plan_receipts/tp_rank_1-retired-1.json"
            self.assertEqual(json.loads(path.read_text())["status"], "TEARDOWN_STARTED")
            release.set()
            deadline = time.time() + 5
            while json.loads(path.read_text())["status"] != "ABORTED":
                self.assertLess(time.time(), deadline)
                time.sleep(0.01)
            self.assertTrue(receiver.aborted)

    def test_plan_file_is_read_as_a_list_and_receipted_on_first_load(self):
        """S3b-3 GPU (10-31): the target read the plan with the JSON-object
        loader, rejected it, and never closed a retired source's channel."""
        import json
        import tempfile
        from pathlib import Path

        from vllm.bridge_tp import source_channels
        from vllm.bridge_tp.channel_runtime import ManifestPollCache

        reconcile = self.method("_reconcile_source_channels")
        receipts = []
        reconcile.__globals__.update(
            parse_plan=source_channels.parse_plan,
            _load_json_list=self.module_function("_load_json_list"),
            _atomic_json_dump=lambda value, path: receipts.append((path.name, value)),
            get_tp_group=lambda: NS(rank_in_group=1),
            time=__import__("time"),
        )
        log = []
        channels = SourceChannelSet(lambda c: Receiver(c, log), lambda r, c: r.open())
        channels.reconcile(parse_plan([A, A3], 2))
        with tempfile.TemporaryDirectory() as folder:
            plan_path = Path(folder) / "plan.json"
            plan_path.write_text(json.dumps([A, A3]))
            connector = NS(
                _source_channels=channels,
                _source_plan_path=plan_path,
                _channel_plan_cache=ManifestPollCache(),
                _channel_plan_acknowledged=False,
                _target_tp_size=2,
                manifest_path=Path(folder) / "session" / "manifest.json",
            )
            reconcile(connector)
            self.assertEqual(len(receipts), 1)
            name, value = receipts[0]
            self.assertEqual(name, "tp_rank_1.json")
            self.assertEqual(value["planned"], [1, 3])
            self.assertEqual(value["closed"], [])
            reconcile(connector)
            self.assertEqual(len(receipts), 1)  # unchanged plan: no new receipt
            plan_path.write_text(json.dumps([A3]))
            reconcile(connector)
            self.assertEqual(receipts[-1][1]["closed"], [1])
            self.assertEqual(receipts[-1][1]["planned"], [3])
            self.assertEqual(set(channels.channels), {3})
            plan_path.write_text(json.dumps(dict(sources=[A3])))
            connector._channel_plan_cache = ManifestPollCache()
            with self.assertRaises(TypeError):
                reconcile(connector)

    def test_activate_points_every_single_receiver_path_at_the_source(self):
        import threading

        activate = self.method("_activate_source_channel")
        log = []
        channels = SourceChannelSet(lambda c: Receiver(c, log), lambda r, c: r.open())
        channels.reconcile(parse_plan([A, A3], 2))
        connector = NS(
            _source_channels=channels,
            _prebound_gpu_receiver_lock=threading.Lock(),
            _persistent_gpu_receiver_lock=threading.Lock(),
            _prebound_gpu_receiver=None,
            _persistent_gpu_receiver=None,
            channel_generation=0,
        )
        activate(connector, 3)
        receiver = channels.channels[3][1]
        self.assertIs(connector._prebound_gpu_receiver, receiver)
        self.assertIs(connector._persistent_gpu_receiver, receiver)
        self.assertEqual(connector.channel_generation, 3)
        receiver.busy()
        with self.assertRaises(RuntimeError):
            activate(connector, 1)
        self.assertEqual(connector.channel_generation, 3)
        with self.assertRaises(RuntimeError):
            activate(NS(_source_channels=None), 1)

    def test_scheduler_role_accepts_planned_generations_without_receivers(self):
        """G2 (10-31): the scheduler-role connector has no receivers."""
        import json
        import tempfile

        from vllm.bridge_tp import source_channels

        check = self.method("_check_source_generation")
        check.__globals__.update(
            parse_plan=source_channels.parse_plan,
            _load_json_list=self.module_function("_load_json_list"),
        )
        with tempfile.TemporaryDirectory() as folder:
            plan_path = __import__("pathlib").Path(folder) / "plan.json"
            plan_path.write_text(json.dumps([A, A3, row("A-g7-tp1", 30420, 7)]))
            scheduler = NS(
                _source_channels=None,
                _source_plan=parse_plan([A, A3], 2),
                _source_plan_path=plan_path,
                _target_tp_size=2,
            )
            for generation in (1, 3, 7):
                check(scheduler, generation)
            with self.assertRaises(ValueError):
                check(scheduler, 9)
            self.assertIsNone(scheduler._source_channels)
        log = []
        channels = SourceChannelSet(lambda c: Receiver(c, log), lambda r, c: r.open())
        channels.reconcile(parse_plan([A, A3], 2))
        activated = []
        worker = NS(
            _source_channels=channels,
            _activate_source_channel=activated.append,
        )
        check(worker, 3)
        self.assertEqual(activated, [3])

    def test_connector_calls_the_channel_set(self):
        import ast

        cls = self.connector_class()
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

        def called(node):
            return {
                n.func.attr
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            } | {
                n.func.id
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }

        self.assertIn("parse_plan", called(methods["__init__"]))
        self.assertIn("_check_source_generation", called(methods["_load_manifest"]))
        prebind = called(methods["_start_gpu_direct_receiver_prebind"])
        for name in (
            "SourceChannelSet",
            "_reconcile_source_channels",
            "_activate_source_channel",
            "discard",
        ):
            self.assertIn(name, prebind)
        text = ast.unparse(methods["__init__"])
        for key in ("bridgetp_gpu_direct_sources", "bridgetp_gpu_direct_channel_plan"):
            self.assertIn(key, text)
