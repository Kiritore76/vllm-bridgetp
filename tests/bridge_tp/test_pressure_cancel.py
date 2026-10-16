# SPDX-License-Identifier: Apache-2.0
"""Pre-freeze policy, cleanup ownership and route boundaries without CUDA."""

import ast
import io
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from tests.bridge_tp import test_pressure_health_service as health_tests
from tests.bridge_tp import test_pressure_session as session_tests
from tests.bridge_tp import test_pressure_sessions as sequence_tests
from tests.bridge_tp.test_pressure_policy import candidate, config, length, p
from tools.bridge_tp.pressure_cpu import load

c = load("pressure_cancel")
ROOT = Path(__file__).resolve().parents[2]


def reading_at(name="tp1", now=10, **changes):
    return replace(health_tests.HealthPolicyTests().reading(name, now), **changes)


class CancelPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(health_tests.HealthPolicyTests().cfg(), shadow_cancel=True)
        self.policy = c.ShadowCancellation()

    def decide(self, now, source=None, target=None, cand=None, reversible=True):
        reading = reading_at
        return self.policy.decide(
            now,
            reading(now=now, **(source or {})),
            reading("tp4", now=now, **(target or {})),
            candidate(sampled_at=now, published_at=now, **(cand or {})),
            self.cfg,
            reversible=reversible,
        )

    def sustained(self, **kwargs):
        for i in range(16):
            result = self.decide(10 + i * 0.2, **kwargs)
            if i < 15:
                self.assertIsNone(result)
        return result

    def test_recovery_requires_health_and_full_KV_projection(self):
        self.assertIn("recovered", self.sustained(source=dict(health_eta=1)))
        for source in (
            dict(health_eta=0.95),
            dict(health_eta=1, free_blocks=250, growth_blocks_s=10),
            dict(health_eta=1, growth_blocks_s=None),
            dict(health_eta=1, unallocated_prompt_blocks=None),
            dict(health_eta=1, health_status="UNKNOWN"),
            dict(health_eta=1, waiting=1),
        ):
            self.policy = c.ShadowCancellation()
            self.assertIsNone(self.sustained(source=source))

    def test_target_pressure_cannot_cancel_while_KV_warning_remains(self):
        self.assertIsNone(
            self.sustained(
                source=dict(free_blocks=250, growth_blocks_s=10),
                target=dict(health_eta=0.7),
            ),
        )

    def test_unknown_target_does_not_mean_pressure_increased(self):
        self.assertIsNone(self.sustained(target=dict(health_status="UNKNOWN")))

    def test_short_prediction_never_overrides_source_pressure(self):
        self.assertIsNone(
            self.sustained(cand=dict(remaining=length(mean=20))),
        )
        # Ambiguous bin straddles the horizon: lower=0, upper=1.
        ambiguous = load("bounded_length").bounded_length(
            (0, 1, 0),
            (0, 4096),
            generated=20,
            predicted_at=20,
            max_output_tokens=6000,
        )
        self.policy = c.ShadowCancellation()
        self.assertIsNone(self.sustained(cand=dict(remaining=ambiguous)))

    def test_stop_cap_short_does_not_cancel_while_source_needs_relief(self):
        self.assertIsNone(
            self.sustained(cand=dict(generated=5900, prediction_status="STALE")),
        )

    def test_spike_gap_unknown_and_freeze_reset_confirmation(self):
        for now in (10, 10.2, 10.4):
            self.assertIsNone(self.decide(now, source=dict(health_eta=1)))
        self.assertIsNone(self.decide(10.6))
        self.assertIsNone(self.decide(13, source=dict(health_eta=1)))
        self.assertIsNone(self.decide(13.2, reversible=False))
        self.assertIsNone(self.decide(15, source=dict(health_eta=1)))
        self.assertIsNone(self.decide(18.1, source=dict(health_eta=1)))

    def test_disabled_and_frozen_modes_never_cancel(self):
        self.assertIsNone(self.sustained(source=dict(health_eta=1), reversible=False))
        self.cfg = replace(self.cfg, shadow_cancel=False)
        self.assertIsNone(self.sustained(source=dict(health_eta=1)))

    def test_same_closed_window_cannot_confirm_recovery_by_repolling(self):
        for now in (10 + i * 0.2 for i in range(21)):
            self.assertIsNone(
                self.decide(
                    now,
                    source=dict(
                        health_eta=1,
                        health_sampled_at=10,
                    ),
                )
            )
        self.assertEqual(
            self.decide(
                14.2,
                source=dict(
                    health_eta=1,
                    health_sampled_at=14,
                ),
            ),
            "source service and projected KV recovered",
        )

    def test_mixed_window_or_bad_decode_never_proves_recovery(self):
        for changes in (
            dict(health_eta=1, health_mixed_window=True),
            dict(health_eta=1, decode_ratio=1.1),
            dict(health_eta=1, decode_ratio=None),
        ):
            self.policy = c.ShadowCancellation()
            self.assertIsNone(self.sustained(source=changes))

    def test_target_worse_or_short_output_still_cancels_only_after_source_recovers(
        self,
    ):
        self.assertIn(
            "recovered",
            self.sustained(
                source=dict(health_eta=1),
                target=dict(health_eta=0.7),
                cand=dict(remaining=length(mean=20)),
            ),
        )

    def test_own_shadow_queue_is_excluded_but_external_prefill_is_not(self):
        target = reading_at("tp4", waiting=2, pending_prompt_tokens=4500)
        inventory = dict(
            sampled_at=10,
            requests=[
                dict(
                    request_id="shadow-1",
                    generated=0,
                    running=False,
                    prompt_tokens=4000,
                    computed_tokens=1000,
                )
            ],
        )
        adjusted = c.without_shadow_demand(target, inventory, "shadow-")
        self.assertEqual(adjusted.waiting, 1)
        self.assertEqual(adjusted.pending_prompt_tokens, 1500)
        self.assertEqual(adjusted.free_blocks, target.free_blocks)
        only_shadow = replace(target, waiting=1, pending_prompt_tokens=3000)
        adjusted = c.without_shadow_demand(only_shadow, inventory, "shadow-")
        self.assertEqual((adjusted.waiting, adjusted.pending_prompt_tokens), (0, 0))
        self.assertEqual(
            c.without_shadow_demand(target, dict(inventory, sampled_at=9), "shadow-"),
            target,
        )

    def test_stale_request_does_not_prove_short_life(self):
        for now in (20, 21, 22, 23):
            result = self.policy.decide(
                now,
                reading_at(now=now),
                reading_at("tp4", now=now),
                candidate(generated=5900),
                self.cfg,
                reversible=True,
            )
            self.assertIsNone(result)


class RoutingBoundaries(unittest.TestCase):
    def route(self, prompt, *, source=None, target=None, output=1024):
        reading = reading_at
        return p.admit(
            "r",
            prompt,
            output,
            (reading(**(source or {})), reading("tp4", **(target or {}))),
            health_tests.HealthPolicyTests().cfg(),
            p.ReservationLedger(),
            10,
            "key",
        )

    def test_both_prompt_boundaries_and_pressure_boundary(self):
        for prompt, eta, expected in (
            (2047, 0.7, "tp1"),
            (2048, 0.7, "tp4"),
            (2049, 0.7, "tp4"),
            (2048, 0.9, "tp1"),
            (4095, 1, "tp1"),
            (4096, 1, "tp4"),
            (4097, 1, "tp4"),
        ):
            with self.subTest(prompt=prompt, eta=eta):
                self.assertEqual(
                    self.route(prompt, source=dict(health_eta=eta)).pool_id, expected
                )

    def test_target_boundaries_unknown_stale_and_waiting(self):
        for target, expected in (
            (dict(health_eta=0.9, health_load=0.8), "tp4"),
            (dict(health_eta=0.899), "tp1"),
            (dict(health_load=0.801), "tp1"),
            (dict(health_status="UNKNOWN"), "tp1"),
            (dict(health_sampled_at=-10), "tp1"),
            (dict(waiting=1), "tp1"),
            (dict(pending_prompt_tokens=1), "tp1"),
        ):
            result = self.route(4096, target=target)
            self.assertEqual(result.pool_id, expected)
            if expected == "tp1":
                self.assertEqual(result.routing_reason, "prefer TP1")

    def test_context_exact_boundary_fallback_and_no_admission_gate(self):
        self.assertEqual(self.route(4096, output=4096).pool_id, "tp4")
        result = self.route(4096, target=dict(context_limit=5000), output=1024)
        self.assertEqual(result.pool_id, "tp1")
        with self.assertRaises(p.RoutingError):
            self.route(4096, output=4097)
        result = self.route(
            2047, source=dict(free_blocks=0), target=dict(free_blocks=0)
        )
        self.assertEqual(result.pool_id, "tp1")
        self.assertEqual(result.reservation_basis, "none")


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.fixture = session_tests.SessionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.driver = d = self.fixture.driver
        d.persistent_cleanup = True
        for name in ("cutover_manifest.json", "rolling_freeze_selection.json"):
            (d.run_dir / name).unlink()
        d.executor = NS(submit=Mock(side_effect=AssertionError("SSE pool occupied")))
        d.coordinator.policy = NS(note_cancelled=Mock())

    def write(self, name, value):
        path = self.driver.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def setup_receipts(self):
        d = self.driver
        d.cancel_reason, d.cancel_started_at = "recovered", 10
        d.cancel_future = Future()
        d.cancel_future.set_result(dict(updated_unix_s=11))
        for name, component in (
            ("source_cleanup_receipt.json", "source_delta_mirror"),
            ("stager_cleanup_receipt.json", "cpu_stager"),
        ):
            self.write(
                name,
                dict(
                    status="CLEANED",
                    component=component,
                    migration_id="s1",
                    reason="recovered",
                    updated_unix_s=11,
                ),
            )
        common = dict(
            channel_create_count=1,
            channel_destroy_count=0,
            channel_session_count=1,
            migration_id="s1",
        )
        self.write(
            "gpu_direct_sender.json",
            dict(
                **common,
                communicator_lifecycle="PERSISTENT_CHANNEL_IDLE",
                source_request_id="request",
            ),
        )
        for rank in range(4):
            self.write(
                f"persistent_channel_receipts/tp_rank_{rank}.json",
                dict(
                    **common,
                    status="PERSISTENT_CHANNEL_IDLE",
                    tp_rank=rank,
                    session_request_id="request",
                ),
            )
        return dict(pool_id="tp4", sampled_at=12, requests=[])

    def test_cleanup_runs_off_dispatch_thread_and_never_commits(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)

        def cleanup(reason):
            entered.set()
            release.wait(2)
            return {}

        self.driver._cleanup_shadow = cleanup
        self.driver.cancel("source recovered")
        self.assertTrue(entered.wait(1))
        self.assertFalse(self.driver.cancel_future.done())
        self.driver.tick(self.fixture.view, self.fixture.snapshot)
        self.assertEqual(self.driver.adapter.commits, 0)
        self.assertFalse(self.fixture.settlements)
        release.set()
        self.driver.cancel_future.result(timeout=2)

    def test_settle_only_after_fresh_absence_all_ranks_and_stager(self):
        inventory = self.setup_receipts()
        d = self.driver
        path = d.run_dir / "persistent_channel_receipts/tp_rank_3.json"
        original = path.read_text()
        path.unlink()
        d.poll_cancel(inventory, 12)
        self.assertFalse(d.settled)
        path.write_text(original)
        d.poll_cancel(dict(inventory, sampled_at=10.5), 12)
        self.assertFalse(d.settled)
        d.poll_cancel(
            dict(
                inventory,
                requests=[
                    dict(
                        request_id=f"cmpl-bridgetp-phase9-target-{d.run_dir.name}-0-abc"
                    )
                ],
            ),
            12,
        )
        self.assertFalse(d.settled)
        d.poll_cancel(inventory, 12)
        self.assertTrue(d.settled)
        self.assertFalse(self.fixture.source.future.done())
        self.assertEqual(
            d.record.state, self.fixture.driver.old.MigrationState.CANCELLED
        )
        self.assertEqual(self.fixture.settlements[0][0][-1], "CANCELLED")
        d.coordinator.policy.note_cancelled.assert_called_once_with("request", 12)

    def test_wrong_receipt_and_failed_cleanup_keep_commitment(self):
        inventory = self.setup_receipts()
        self.write(
            "stager_cleanup_receipt.json", dict(status="CLEANED", migration_id="other")
        )
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.driver.poll_cancel(inventory, 12)
        self.assertFalse(self.fixture.settlements)
        self.driver.cancel_future = Future()
        self.driver.cancel_future.set_exception(RuntimeError("HTTP failed"))
        with self.assertRaisesRegex(RuntimeError, "HTTP failed"):
            self.driver.poll_cancel(inventory, 12)

    def test_target_abort_requires_terminal_stream_and_native_receipt(self):
        inventory = self.setup_receipts()
        d = self.driver
        d.target_future = Future()
        d.target_future.set_result(dict(token_ids=[], finish_reason="abort"))
        d.poll_cancel(inventory, 12)
        self.assertFalse(d.settled)
        self.write(
            "target_abort_receipt.json",
            dict(
                status="ABORTED",
                migration_id="s1",
                completed_unix_s=11,
                target_request_id=f"cmpl-bridgetp-phase9-target-{d.run_dir.name}-0-abc",
            ),
        )
        d.poll_cancel(inventory, 12)
        self.assertTrue(d.settled)

    def test_unexpected_empty_disconnect_is_still_failure(self):
        inventory = self.setup_receipts()
        d = self.driver
        d.target_future = Future()
        d.target_future.set_exception(
            RuntimeError("streaming response ended before completion")
        )
        with self.assertRaisesRegex(RuntimeError, "ended before completion"):
            d.poll_cancel(inventory, 12)
        self.assertFalse(d.settled)
        self.assertFalse(self.fixture.settlements)

    def test_source_drain_precedes_target_abort(self):
        order = []
        adapter = self.driver.adapter
        adapter.refresh_binding = lambda: object()
        adapter.cancel = lambda *a, **kw: (
            order.append(("source", kw))
            or dict(
                state="CANCELLED",
                migration_id="s1",
                source_request_id="request",
                source_abort_dispatched=False,
            )
        )
        adapter.wait_for_source_cleanup = lambda: order.append("drained")
        adapter.cancel_shadow_target = lambda reason: order.append("target")
        self.driver._cleanup_shadow("recovered")
        self.assertEqual(
            order, [("source", dict(abort_source=False)), "drained", "target"]
        )

    def test_freeze_blocks_new_policy_cancellation(self):
        self.assertTrue(self.driver.reversible())
        self.write("request_freeze_control.json", dict(action="FREEZE"))
        self.assertFalse(self.driver.reversible())


class CancelReuseTests(unittest.TestCase):
    def test_cancelled_session_rotates_while_source_continues(self):
        fixture = sequence_tests.RepeatedSessions()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        s = fixture.sequence.sessions[0]
        fixture.entry.state = "CANCELLED"
        fixture.entry.service_observed = False
        s.stream.future = Future()
        s.target_future = None
        fixture.write(
            "stager_cleanup_receipt.json",
            dict(migration_id="run-0001", status="CLEANED"),
        )
        with patch("tools.bridge_tp.pressure_sessions.publish_active"):
            self.assertTrue(fixture.sequence.rotate())
        self.assertFalse(s.stream.future.done())
        self.assertEqual(fixture.sequence.migration_id, "run-0002")

    def test_cancelled_request_cooldown_does_not_block_another_KV_candidate(self):
        cfg = config(capacity_lookahead_s=16, sustain_s=3)
        ledger = p.ReservationLedger()
        policy = p.PressurePolicy(cfg, ledger)
        policy.note_cancelled("a", 10)
        from tests.bridge_tp.test_pressure_policy import pool

        source = pool(free_blocks=250, growth_blocks_s=10, unallocated_prompt_blocks=0)
        a = candidate(estimated_release_s=16, releasable_blocks=200, owned_blocks=200)
        b = replace(a, request_id="b")
        result = policy.decide(10, source, pool("tp4"), [a, b])
        self.assertEqual(result.request_id, "b")
        self.assertIn(
            "cancelled request cooling down", result.candidates[0]["rejections"]
        )


class ReceiverCancelTests(unittest.TestCase):
    def test_native_abort_receipt_only_matches_cancelled_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            control = root / "takeover_state.json"
            session_tests.io.atomic_json_dump(
                dict(state="CANCELLED", migration_id="m1"), control
            )
            connector = NS(
                gpu_resident_shadow=True,
                takeover_control_path=control,
                manifest_path=root / "migration_active" / "session_manifest.json",
                _release_persistent_payload_buffer=Mock(),
                _pending_requests={},
                _active_requests={},
                _load_threads={},
                _load_errors={},
                _gpu_kv_lock=threading.RLock(),
                _copy_done_events={},
                _model_wait_streams=set(),
                _model_wait_pending_requests=set(),
                _gpu_restore_quiescence_lock=threading.RLock(),
                _gpu_restore_quiescence={},
                _completed_recvs=set(),
                _reported_recvs=set(),
            )
            method = actual_method(
                "vllm/bridge_tp/streaming_connector.py",
                "request_finished",
                dict(
                    Request=object,
                    Any=object,
                    time=time,
                    _atomic_json_dump=session_tests.io.atomic_json_dump,
                    _load_json=session_tests.io.load_json,
                ),
            )
            request = NS(
                request_id=f"cmpl-bridgetp-phase9-target-{root.name}-0-abc",
                status=NS(name="FINISHED_ABORTED"),
            )
            method(connector, request, [])
            proof = session_tests.io.load_json(root / "target_abort_receipt.json")
            self.assertEqual(proof["target_request_id"], request.request_id)
            method(
                connector, NS(request_id="other-background", status=request.status), []
            )
            self.assertEqual(
                session_tests.io.load_json(root / "target_abort_receipt.json"), proof
            )

    def test_actual_receiver_drains_end_session_without_closing_channel(self):
        path = ROOT / "vllm/bridge_tp/streaming_connector.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_finish_cancelled_receiver"
        )
        context = {}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            context,
        )
        lifecycle = NS(active_session=object())
        receiver = NS(lifecycle=lifecycle, close=Mock(), acknowledge_delta=Mock())
        results = [NS(start_token=10, end_token=20), None]

        def receive(**kwargs):
            value = results.pop(0)
            if value is None:
                lifecycle.active_session = None
            return value

        receiver.receive_delta = receive
        connector = NS(persistent_channel=True, _record_persistent_receiver_idle=Mock())
        context["_finish_cancelled_receiver"](
            connector,
            receiver,
            request_id="target",
            source_request_id="source",
            migration_id="m1",
            tp_rank=0,
        )
        receiver.acknowledge_delta.assert_called_once_with(start_token=10, end_token=20)
        receiver.close.assert_not_called()
        connector._record_persistent_receiver_idle.assert_called_once()


class CancelSmokePlanTests(unittest.TestCase):
    def test_selects_only_original_KV_recipe_and_separates_mechanism_test(self):
        from tools.bridge_tp import pressure_health_screen as screen

        bundle = ROOT / "experiments/phase9/pressure_balanced_v2"
        original = json.loads((bundle / "health_sustained.json").read_text())
        selected = screen.select_kv_test(original)
        self.assertEqual(selected["arms"], ["stay", "migrate"])
        self.assertEqual(len(selected["recipes"]), 1)
        self.assertEqual(selected["recipes"][0], original["recipes"][-1])
        smoke = screen.select_kv_test(original, cancel_smoke=True)
        self.assertEqual(smoke["arms"], ["migrate"])
        self.assertTrue(smoke["test_only_cancel_smoke"])
        self.assertEqual(len(original["recipes"]), 5)

    def test_injection_waits_for_natural_START_admitted_target_and_four_ranks(self):
        from tools.bridge_tp import pressure_health_screen as screen

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            s = NS(
                run_dir=root,
                record=NS(request_id="source"),
                migration_id="m1",
                settled=False,
                cancel_future=None,
                target_future=Future(),
                reversible=lambda: True,
                cancel=Mock(),
            )
            sequence = NS(sessions=[s])
            inventory = dict(
                sampled_at=10,
                requests=[
                    dict(
                        request_id=f"cmpl-bridgetp-phase9-target-{root.name}-0-abc",
                        generated=0,
                    )
                ],
            )
            audit = NS(write=Mock())
            self.assertFalse(screen.inject_cancel_smoke(sequence, inventory, 10, audit))
            for rank in range(4):
                session_tests.io.atomic_json_dump(
                    dict(status="STAGED", migration_id="m1"),
                    root / "initial_stage_receipts" / f"tp_rank_{rank}.json",
                )
            self.assertTrue(screen.inject_cancel_smoke(sequence, inventory, 10, audit))
            s.cancel.assert_called_once()
            self.assertTrue(audit.write.call_args.args[0]["test_only"])
            s.cancel.reset_mock()
            s.reversible = lambda: False
            self.assertFalse(screen.inject_cancel_smoke(sequence, inventory, 10, audit))
            s.cancel.assert_not_called()

    def test_no_injection_is_coverage_gap_not_pass(self):
        from tools.bridge_tp import pressure_health_screen as screen

        result = screen.assess_cancel_smoke([], ROOT)
        self.assertEqual(result["coverage"], "COVERAGE_GAP")
        self.assertFalse(result["natural_cancellation_validated"])
        self.assertFalse(result["benefit_comparison"])

    def test_mechanism_acceptance_requires_release_after_confirmed_cancel(self):
        from tools.bridge_tp import pressure_health_screen as screen

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session_tests.io.atomic_json_dump(
                dict(target_request_id="target1"),
                root / "sessions/migration-0001/target_abort_receipt.json",
            )
            audit = [
                dict(
                    kind="pressure_test_cancel_injected",
                    migration_id="m1",
                    target_request_id="target1",
                ),
                dict(kind="pressure_cancel_confirmed", migration_id="m1", unix_s=20),
                dict(
                    kind="pressure_channel_reused",
                    previous_migration_id="m1",
                    previous_outcome="CANCELLED",
                ),
            ]
            self.assertEqual(
                screen.assess_cancel_smoke(audit, root)["coverage"], "COVERAGE_GAP"
            )
            audit.append(
                dict(
                    kind="pressure_source_release_confirmed",
                    migration_id="m2",
                    unix_s=30,
                )
            )
            self.assertEqual(
                screen.assess_cancel_smoke(audit, root)["coverage"],
                "CANCEL_AND_REUSE_OBSERVED",
            )

    def test_actual_plan_propagates_diagnostic_flag_only_in_smoke(self):
        from tools.bridge_tp import pressure_health_screen as screen
        from tools.bridge_tp import run_pressure_comparison as compare

        bundle = ROOT / "experiments/phase9/pressure_balanced_v2"
        config = json.loads((bundle / "health_sustained.json").read_text())
        inputs = json.loads((bundle / "calibration.json").read_text())

        def build(spec, scene, root, *args):
            return dict(
                run_dir=root / scene["name"] / "controller",
                controller_argv=["python", "controller"],
                source_env={},
                target_env={},
            )

        for smoke, count in ((False, 2), (True, 1)):
            with (
                tempfile.TemporaryDirectory() as temp,
                patch.object(compare.pilot, "build_scene", side_effect=build),
            ):
                common = NS(
                    write_json=lambda path, value: path.write_text(json.dumps(value))
                )
                plans = compare.build_plans(
                    dict(policy_config=str(bundle / "policy.json")),
                    screen.select_kv_test(config, cancel_smoke=smoke),
                    inputs,
                    Path(temp),
                    common,
                    None,
                )
                self.assertEqual(len(plans), count)
                for _, arm, _, plan in plans:
                    self.assertEqual(
                        "--test-only-cancel-first-shadow" in plan["controller_argv"],
                        smoke,
                    )
                    self.assertEqual(
                        "--no-migration" in plan["controller_argv"], arm == "stay"
                    )


class EmptyTerminalStreamTests(unittest.TestCase):
    def stream(self, finish_reason):
        path = ROOT / "vllm/entrypoints/openai/completion/serving.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "completion_stream_generator"
        )
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None

        class Message(NS):
            def model_dump_json(self, **kwargs):
                return json.dumps(vars(self), default=vars)

        context = dict(
            CompletionStreamResponse=Message,
            CompletionResponseStreamChoice=Message,
            UsageInfo=Message,
            should_include_usage=lambda *a: (False, False),
            as_list=list,
            GenerationError=RuntimeError,
            logger=Mock(),
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            context,
        )

        async def results():
            for reason in (None, finish_reason):
                yield (
                    0,
                    NS(
                        prompt_token_ids=[1, 2],
                        prompt_logprobs=None,
                        num_cached_tokens=0,
                        prompt="",
                        outputs=[
                            NS(
                                index=0,
                                text="",
                                token_ids=[],
                                logprobs=None,
                                finish_reason=reason,
                                stop_reason=None,
                            )
                        ],
                    ),
                )

        server = NS(
            enable_force_include_usage=False,
            enable_prompt_tokens_details=False,
            system_fingerprint=None,
            _raise_if_error=Mock(),
            create_streaming_error_response=lambda error: json.dumps(
                dict(error=str(error))
            ),
        )
        request = NS(
            n=1,
            stream_options=None,
            max_tokens=100,
            echo=False,
            return_token_ids=True,
            logprobs=None,
        )

        async def collect():
            return [
                chunk
                async for chunk in context["completion_stream_generator"](
                    server,
                    request,
                    [],
                    results(),
                    "cmpl-target",
                    0,
                    "model",
                    1,
                    None,
                    NS(),
                )
            ]

        # The mocked engine yields synchronously. Drive the real async
        # generator without opening the Windows event loop's network sockets.
        coroutine = collect()
        try:
            coroutine.send(None)
        except StopIteration as done:
            return done.value
        finally:
            coroutine.close()
        raise AssertionError("unexpected asynchronous IO in stream fixture")

    def test_empty_abort_keeps_terminal_choice_and_strict_reader_accepts(self):
        chunks = self.stream("abort")
        self.assertEqual(len(chunks), 2)  # Intermediate prefill stays suppressed.
        response = json.loads(chunks[0][6:])
        self.assertEqual(response["choices"][0]["finish_reason"], "abort")
        self.assertEqual(response["choices"][0]["token_ids"], [])
        self.assertEqual(chunks[1], "data: [DONE]\n\n")
        sink = Mock()
        wire = io.BytesIO("".join(chunks).encode())
        with patch("urllib.request.urlopen", return_value=wire):
            result = session_tests.io.post_streaming_completion(
                "http://localhost", dict(prompt=[1], max_tokens=100), 1, sink
            )
        self.assertEqual(result["finish_reason"], "abort")
        self.assertEqual(result["token_ids"], [])
        sink.assert_not_called()

    def test_empty_stop_keeps_terminal_choice(self):
        response = json.loads(self.stream("stop")[0][6:])
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")

    def test_done_without_finish_reason_is_still_rejected(self):
        wire = io.BytesIO(b"data: [DONE]\n\n")
        with (
            patch("urllib.request.urlopen", return_value=wire),
            self.assertRaisesRegex(RuntimeError, "ended before completion"),
        ):
            session_tests.io.post_streaming_completion(
                "http://localhost", dict(prompt=[1], max_tokens=100), 1, Mock()
            )


def actual_method(path, name, context):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), context)
    return context[name]


class SourceCancelRaceTests(unittest.TestCase):
    def test_explicit_cancel_racing_finalizer_is_not_a_transfer_failure(self):
        abandoned = type("_ShadowAbandoned", (RuntimeError,), {})
        state = NS(
            finalized=False,
            finalizing=True,
            lifecycle_lock=threading.RLock(),
            stop_workers=Mock(),
        )
        method = actual_method(
            "vllm/bridge_tp/phase8_source.py",
            "_background_finalize_cutover",
            dict(
                Any=object,
                _Phase8SourceState=object,
                _ShadowAbandoned=abandoned,
                logger=Mock(),
                _finalize_cutover=Mock(side_effect=abandoned()),
            ),
        )
        method(state=state)
        self.assertTrue(state.finalized)
        self.assertFalse(state.finalizing)
        state.stop_workers.assert_called_once()

    def test_actual_watcher_waits_finalizer_then_drains_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            io = session_tests.io
            io.atomic_json_dump(
                dict(reason="recover", abort_source=False),
                root / "cleanup_request.json",
            )
            order = []
            lock_entered = Event()

            class NotifyingLock:
                def __enter__(self):
                    lock_entered.set()

                def __exit__(self, *args):
                    return False

            state = NS(
                config=NS(run_dir=root, migration_id="m1", socket_timeout_s=2),
                request_id="s1",
                finalized=False,
                finalizing=True,
                lifecycle_lock=NotifyingLock(),
                wait_for_acks=lambda: order.append("acks"),
                stop_workers=lambda: order.append("end_session"),
                cancel_deferred_history=lambda reason: 0,
                delta_batches=1,
                delta_tokens=4,
            )
            method = actual_method(
                "vllm/bridge_tp/phase8_source.py",
                "_watch_cleanup",
                dict(
                    json=json,
                    time=time,
                    _atomic_json_dump=io.atomic_json_dump,
                    logger=Mock(),
                ),
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(method, state)
                self.assertTrue(lock_entered.wait(1))
                self.assertFalse((root / "source_cleanup_receipt.json").exists())
                state.finalizing = False
                future.result(timeout=2)
            self.assertEqual(order, ["acks", "end_session", "acks"])
            self.assertEqual(
                io.load_json(root / "request_freeze_control.json")["action"], "RESUME"
            )
            self.assertEqual(
                io.load_json(root / "source_cleanup_receipt.json")["status"], "CLEANED"
            )

    def test_actual_republish_requires_old_cancel_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            context = dict(json=json, _published_sessions={"s1": ("m1", old)})
            method = actual_method(
                "vllm/bridge_tp/kv_stream.py", "_allow_cancelled_republish", context
            )
            config = NS(migration_id="m2", run_dir=new)
            self.assertFalse(method(config, "s1"))
            session_tests.io.atomic_json_dump(
                dict(
                    status="CLEANED", migration_id="m1", component="source_delta_mirror"
                ),
                old / "source_cleanup_receipt.json",
            )
            state = dict(
                state="CANCELLED",
                migration_id="m1",
                source_request_id="s1",
                source_abort_dispatched=False,
            )
            session_tests.io.atomic_json_dump(state, old / "takeover_state.json")
            self.assertTrue(method(config, "s1"))
            session_tests.io.atomic_json_dump(
                dict(state, source_abort_dispatched=True), old / "takeover_state.json"
            )
            self.assertFalse(method(config, "s1"))

    def test_freeze_gate_resets_same_request_after_resume(self):
        from vllm.bridge_tp.request_freeze import RequestFreezeGate

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gate = RequestFreezeGate(root)
            session_tests.io.atomic_json_dump(
                dict(action="FREEZE", request_id="same"),
                root / "request_freeze_control.json",
            )
            self.assertTrue(gate.is_frozen("same"))
            gate._recorded_frozen_request_id = "same"
            session_tests.io.atomic_json_dump(
                dict(action="RESUME"), root / "request_freeze_control.json"
            )
            gate._mtime_ns = -1
            self.assertFalse(gate.is_frozen("same"))
            self.assertIsNone(gate._recorded_frozen_request_id)

    def test_rotation_does_not_keep_a_cancelled_source_frozen(self):
        from vllm.bridge_tp.request_freeze import RequestFreezeGate

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            gate = RequestFreezeGate(old)
            session_tests.io.atomic_json_dump(
                dict(action="FREEZE", request_id="same"),
                old / "request_freeze_control.json",
            )
            self.assertTrue(gate.is_frozen("same"))
            # Equivalent control-path stat results to an active symlink rotation.
            gate.control_path = new / "request_freeze_control.json"
            self.assertFalse(gate.is_frozen("same"))


if __name__ == "__main__":
    unittest.main()
