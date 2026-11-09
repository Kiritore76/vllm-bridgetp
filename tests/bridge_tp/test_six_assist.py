# SPDX-License-Identifier: Apache-2.0
"""Optional six-card assists never impose aggregate pressure gates."""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from tools.bridge_tp import run_pressure_balanced_pilot as pilot
from tools.bridge_tp import six_conversion as six


class AssistTests(unittest.TestCase):
    def setUp(self):
        b = self.backend = object.__new__(six.SixBackend)
        b.fresh = lambda status: True
        b.refs = lambda plan: [NS(instance_id="source")]
        b.assist_seq = b.assist_step_start_seq = 0
        b.assisted = []
        b.assist_committed_baseline = 0
        b.safety = dict(assist_max_committed=4, assist_max_attempts=8, assist_factor=3)
        b.speed_margin = 0.8
        requests = [
            dict(
                request_id=name,
                tokens_s=speed,
                remaining_bins=[(remaining, 1)],
                action_available=action,
            )
            for name, speed, remaining, action in (
                ("short", 20, 100, True),
                ("long", 20, 3200, True),
                ("unknown", None, 4000, False),
            )
        ]
        self.status = dict(
            release_budget_s=5,
            migration_target="target",
            instances={
                "source": dict(tp_degree=1, sessions=0, requests=requests),
                "target": dict(
                    tp_degree=2,
                    lifecycle="ACTIVE",
                    sessions=0,
                    pressure=dict(zone="OVERLOAD"),
                ),
                "other": dict(tp_degree=1, sessions=0, pressure=dict(capacity=100)),
            },
        )

    def test_longest_startable_despite_unrelated_pressure(self):
        reason, choice = self.backend.assist_choice({}, self.status)
        self.assertIsNone(reason)
        self.assertEqual(choice["request_id"], "long")
        self.assertNotIn("target", choice)
        self.assertTrue(choice["final_length"])
        self.assertNotIn("evacuate", choice)

    def test_refusal_retries_but_started_request_is_not_reissued(self):
        self.backend.assist_seq = 1
        self.status["assist_result"] = dict(seq=1, state="REFUSED", request_id="long")
        self.assertIsNotNone(self.backend.assist_choice({}, self.status)[1])
        self.status["assist_result"]["state"] = "STARTED"
        self.assertIsNone(self.backend.assist_choice({}, self.status)[1])

    def test_wrong_target_and_session_refuse_only_assist(self):
        self.status["instances"]["target"]["tp_degree"] = 1
        self.assertIsNone(self.backend.assist_choice({}, self.status)[1])
        self.status["instances"]["target"]["tp_degree"] = 2
        self.status["instances"]["target"]["sessions"] = 1
        self.assertIsNone(self.backend.assist_choice({}, self.status)[1])


class ControllerBindingTests(unittest.TestCase):
    def test_shared_order_and_fallback_choose_a_safe_second_target(self):
        from tools.bridge_tp.pressure_cpu import load

        runtime = load("pressure_runtime")
        readings = {
            t: NS(unallocated_prompt_blocks=n)
            for t, n in (("A", 10), ("B", 5), ("C", 0))
        }
        order = runtime.rank_migration_targets(
            list(readings),
            {"A": dict(value=0.2), "B": dict(value=0.2), "C": dict(value=0.8)},
            readings,
            {},
        )
        self.assertEqual(order, ["B", "A", "C"])
        calls = []

        def start(target):
            calls.append(target)
            return "cannot hold final length" if target == "B" else None

        reason, target, attempts = pilot.try_ranked_assist_targets(order, start)
        self.assertIsNone(reason)
        self.assertEqual(target, "A")
        self.assertEqual(calls, ["B", "A"])
        self.assertEqual(len(attempts), 2)

    def test_all_targets_refuse_without_starting_or_cancelling_conversion(self):
        reason, target, attempts = pilot.try_ranked_assist_targets(
            ["A", "B"], lambda target: "no capacity"
        )
        self.assertIsNone(target)
        self.assertEqual(reason, "no capacity")
        self.assertEqual(len(attempts), 2)

    def test_real_start_binding_reserves_final_length_and_refuses_low_space(self):
        tree = ast.parse(Path(pilot.__file__).read_text(encoding="utf-8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "start_assist"
        )
        calls = []
        candidate = NS(
            request_id="r",
            migrating=False,
            finished=False,
            prompt_tokens=16,
            generated=1,
            max_output_tokens=64,
        )
        cfg = NS(
            migration_current_length=True,
            capacity_guard_blocks=125,
            health_service=True,
        )
        target = NS(pool_id="target", free_blocks=1000)
        ledger = NS(pending=lambda _: (0, 0, 0), routing_reading=lambda _: NS())
        co = NS(
            policy=NS(cfg=cfg),
            ledger=ledger,
            assist_start=lambda *args, **kwargs: calls.append(args),
        )
        scope = dict(
            alias_of={},
            co=co,
            cfg=cfg,
            math=math,
            controller=NS(block_size=16),
            session=NS(groups={}),
            p=NS(classify=lambda *args: NS()),
            assist_service_refusal=lambda *args: None,
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "start_assist", "exec"),
            scope,
        )
        assist = dict(
            source="source", request_id="r", target="target", final_length=True
        )
        action = object()
        start = scope["start_assist"]
        self.assertIsNone(
            start(assist, 1, target, {"source": [candidate]}, {("r", "target"): action})
        )
        self.assertIs(calls[0][4], action)
        self.assertEqual(calls[0][6], 5)
        target.free_blocks = 129
        self.assertIn(
            "final length",
            start(
                assist, 1, target, {"source": [candidate]}, {("r", "target"): action}
            ),
        )
        self.assertEqual(len(calls), 1)
