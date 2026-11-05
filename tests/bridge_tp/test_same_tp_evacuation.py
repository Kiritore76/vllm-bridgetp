# SPDX-License-Identifier: Apache-2.0
"""CPU contracts of leaving L2: long requests move out before routing closes.

No GPU, NCCL or real controller here: this pins the decisions of the conversion
side and that the migration policy can never choose a same-TP source.
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from tools.bridge_tp import conversion_sequence as seq
from tools.bridge_tp import pressure_conversion as pc
from tools.bridge_tp.conversion_decider import RULES

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "A-g4-tp2"
TARGET = "B-g2-tp2"


def request(name, generated, prompt=16384, available=True):
    return dict(
        request_id=name,
        prompt_tokens=prompt,
        generated=generated,
        max_output_tokens=8192,
        action_available=available,
    )


class EvacuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now = 100.0
        self.requests = []
        self.sessions = 0
        self.assist_result = None
        backend = object.__new__(seq.SequenceBackend)
        backend.clock = lambda: self.now
        backend.safety = dict(pc.SAFETY)
        backend.assist_at = -float("inf")
        backend.assist_seq = 0
        backend.assist_step_start_seq = 0
        backend.run_dir = self.root
        backend.channel_plan = self.root / "channels.json"
        backend.channel_plan.write_text("[]")
        backend.context = dict(folder=str(self.root), target_host="127.0.0.1")
        backend.next_channel_generation = 5
        backend.next_channel_port = 31000
        backend.rows = {
            SOURCE: dict(legacy_pool=None, targets_file=str(self.root / "targets.json"))
        }
        backend.refs = lambda plan: [NS(instance_id=SOURCE)]
        backend.fresh = lambda status: True
        backend.status = self.status
        backend.write_layout = Mock()
        backend.decision = Mock()
        backend.ops = NS(event=Mock())
        self.backend = backend
        self.run = dict(state="IDLE", target=TARGET, channels={}, last_reason=None)
        backend.evacuation_run = self.run

    def status(self):
        return dict(
            assist_result=self.assist_result,
            instances={
                SOURCE: dict(requests=self.requests, sessions=self.sessions),
                TARGET: dict(requests=[], sessions=0),
            },
        )

    def tick(self):
        self.now += 2
        self.backend.evacuate_tick({}, self.run)

    def events(self):
        return [c.args[0] for c in self.backend.ops.event.call_args_list]

    def assists(self):
        return [
            c.kwargs["assist"]
            for c in self.backend.write_layout.call_args_list
            if c.kwargs.get("assist")
        ]

    def ready(self, generation=5):
        folder = self.root / "gpu_channel_preconnect_receipts"
        folder.mkdir(exist_ok=True)
        label = f"source-{SOURCE}-{TARGET}"
        for name in (
            f"tp_rank_0-{SOURCE}",
            f"tp_rank_1-{SOURCE}",
            label,
            f"{label}_rank_1",
        ):
            (folder / f"{name}.json").write_text(
                json.dumps(
                    dict(status="WARMUP_COMPLETE", channel_generation=generation)
                )
            )

    def test_without_long_requests_nothing_connects(self):
        self.requests = [request("short", 100, prompt=512)]
        self.tick()
        self.assertEqual(self.run["state"], "DONE")
        self.assertEqual(json.loads(self.backend.channel_plan.read_text()), [])

    def test_the_channel_opens_only_now_and_new_long_requests_stay_away(self):
        self.requests = [request("long", 10)]
        self.tick()
        self.assertEqual(self.run["state"], "CHANNEL")
        rows = json.loads(self.backend.channel_plan.read_text())
        self.assertEqual([r["source"] for r in rows], [SOURCE])
        targets = json.loads((self.root / "targets.json").read_text())
        self.assertEqual(targets[0]["target"], TARGET)
        self.assertEqual(targets[0]["tp"], 2)
        self.backend.write_layout.assert_called_once_with({}, tilt=[SOURCE])
        self.assertEqual(self.assists(), [])

    def test_routing_stays_open_while_long_requests_leave(self):
        self.requests = [request("long", 10)]
        with patch.object(pc.ControllerConversionBackend, "gate") as parent:
            self.now += 2
            self.assertFalse(self.backend.gate({}))
            parent.assert_not_called()
            self.requests = []
            self.now += 2
            parent.return_value = True
            self.assertTrue(self.backend.gate({}))
            parent.assert_called_once()

    def test_each_rank_of_both_ends_must_be_ready(self):
        self.requests = [request("long", 10)]
        self.tick()
        self.ready()
        folder = self.root / "gpu_channel_preconnect_receipts"
        (folder / f"source-{SOURCE}-{TARGET}_rank_1.json").unlink()
        self.tick()
        self.assertEqual(self.run["state"], "CHANNEL")
        self.ready()
        self.tick()
        self.assertEqual(self.run["state"], "MIGRATING")

    def test_a_receipt_of_an_old_channel_does_not_count(self):
        self.requests = [request("long", 10)]
        self.tick()
        self.ready(generation=4)
        self.tick()
        self.assertEqual(self.run["state"], "CHANNEL")

    def test_one_at_a_time_least_generated_first(self):
        self.requests = [request("old", 5000), request("new", 100)]
        self.tick()
        self.ready()
        self.tick()
        self.assertEqual(self.assists()[0]["request_id"], "new")
        self.assertTrue(self.assists()[0]["before_quiesce"])
        self.tick()  # Not answered yet.
        self.assertEqual(len(self.assists()), 1)
        self.assist_result = dict(seq=1, state="STARTED", reason=None)
        self.sessions = 1
        self.tick()  # Session running.
        self.assertEqual(len(self.assists()), 1)
        self.sessions = 0
        self.requests = [request("old", 5100)]
        self.tick()
        self.assertEqual(self.assists()[1]["request_id"], "old")

    def test_a_refusal_retries_without_quiescing(self):
        self.requests = [request("long", 10)]
        self.tick()
        self.ready()
        self.tick()
        self.assist_result = dict(seq=1, state="REFUSED", reason="target KV")
        self.tick()
        self.assertEqual(len(self.assists()), 2)
        self.assertEqual(self.run["state"], "MIGRATING")

    def test_a_channel_that_never_opens_falls_back_to_waiting(self):
        self.requests = [request("long", 10)]
        self.tick()
        self.now += seq.EVACUATION_CHANNEL_S + 1
        self.tick()
        self.assertEqual(self.run["state"], "NATURAL")
        self.assertEqual(json.loads(self.backend.channel_plan.read_text()), [])
        self.assertEqual(json.loads((self.root / "targets.json").read_text()), [])
        self.tick()
        self.assertEqual(self.assists(), [])
        self.assertEqual(self.run["state"], "NATURAL")
        self.requests = []
        self.tick()
        self.assertEqual(self.run["state"], "DONE")

    def test_a_cancelled_evacuation_disconnects(self):
        self.requests = [request("long", 10)]
        self.tick()
        self.backend.evacuation_close(self.run)
        self.assertEqual(json.loads(self.backend.channel_plan.read_text()), [])
        self.assertEqual(self.run["channels"], {})

    def test_it_counts_what_moved(self):
        self.requests = [request("long", 10)]
        self.tick()
        self.ready()
        self.tick()
        self.assist_result = dict(seq=1, state="STARTED", reason=None)
        self.requests = []
        self.tick()
        self.assertEqual(self.run["state"], "DONE")
        self.assertIn("EVACUATION_DONE", self.events())


class NewTp2Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        backend = object.__new__(seq.SequenceBackend)
        backend._specs = None
        backend.evacuate_enabled = True
        backend.join = None
        backend.rows = {}
        backend.next_channel_generation = 5
        backend.next_channel_port = 31000
        backend.next_service_port = 8400
        backend.context = dict(folder=str(self.root), target_host="127.0.0.1")
        backend.registry = NS(
            instances={
                TARGET: NS(lifecycle="ACTIVE", bundle_id="B", tp_degree=2),
                "A-g1-tp1": NS(lifecycle="ACTIVE", bundle_id="A", tp_degree=1),
            }
        )
        self.backend = backend

    def specs(self, plan):
        original = [(SOURCE, 4, dict(port=8400, gpus=["2", "3"]), {})]
        with patch.object(
            pc.ControllerConversionBackend,
            "replacement_specs",
            return_value=original,
        ):
            return self.backend.replacement_specs(plan)

    def test_it_can_later_be_a_source_but_is_connected_to_nothing(self):
        plan = dict(bundle_id="A", target_bundle_mode="TP2")
        ((_, _, extra, joined),) = self.specs(plan)
        self.assertEqual(joined["migration_channel"]["target"], TARGET)
        self.assertEqual(joined["migration_channel"]["target_tp"], 2)
        self.assertEqual(json.loads(Path(extra["targets_file"]).read_text()), [])
        self.assertNotIn("predictor_events", extra)

    def test_without_the_option_nothing_changes(self):
        self.backend.evacuate_enabled = False
        plan = dict(bundle_id="A", target_bundle_mode="TP2")
        ((_, _, extra, joined),) = self.specs(plan)
        self.assertEqual(joined, {})
        self.assertNotIn("targets_file", extra)

    def test_its_engine_has_the_publisher_but_no_predictor(self):
        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        plan = dict(
            source_argv=["serve", "--port", "1"],
            target_argv=[
                "serve",
                "--port",
                "2",
                "--tensor-parallel-size",
                "2",
                "--kv-transfer-config",
                "{}",
            ],
            source_env=dict(
                BRIDGETP_STREAM_ENABLED="1",
                BRIDGETP_PREDICTOR_LIVE_CHECKPOINT="x",
                BRIDGETP_PREDICTOR_LIVE_EVENTS="y",
            ),
            target_env={},
        )
        extra = dict(
            instance_id=SOURCE,
            tp=2,
            port=8400,
            gpus=["2", "3"],
            inventory_path="inventory.jsonl",
            targets_file="targets.json",
            migration_channel=dict(base_port=31000, generation=5),
        )
        argv, env = launch.extra_engine(plan, extra)
        self.assertNotIn("--kv-transfer-config", argv)
        self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(env["BRIDGETP_STREAM_ENABLED"], "1")
        self.assertEqual(env["BRIDGETP_GPU_DIRECT_TARGETS_FILE"], "targets.json")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,3")
        self.assertFalse([k for k in env if "_PREDICTOR_" in k])


class PolicyStaysOutTests(unittest.TestCase):
    def function(self, name):
        path = ROOT / "tools/bridge_tp/run_pressure_balanced_pilot.py"
        function = next(
            n
            for n in ast.parse(path.read_text("utf-8")).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        namespace = {}
        exec(compile(ast.Module([function], []), "pilot", "exec"), namespace)
        return namespace[name]

    def test_the_policy_never_sees_a_same_tp_source(self):
        policy_sources = self.function("policy_sources")
        chosen = policy_sources(
            {"tp1": ["a"], SOURCE: ["b"]}, [(SOURCE, "inventory", "predictor")]
        )
        self.assertEqual(chosen, {"tp1": ["a"]})

    def test_the_policy_tick_goes_through_the_filter(self):
        source = (ROOT / "tools/bridge_tp/run_pressure_balanced_pilot.py").read_text(
            "utf-8"
        )
        call = source[source.index('"policy.tick"') :][:700]
        self.assertEqual(call.count("policy_sources("), 2)
        self.assertNotIn("candidates_by_source.items()\n                    }", call)


class DeciderTests(unittest.TestCase):
    def test_default_still_waits_for_the_group(self):
        self.assertIs(RULES["evacuate_busy"], False)


if __name__ == "__main__":
    unittest.main()
