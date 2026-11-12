# SPDX-License-Identifier: Apache-2.0
"""A migrating source request and vLLM preemption (11-12 directkv run 3).

C-g4 preempted the request it was migrating; the stream hook's "exactly one
pending token" check then killed the engine. Now: the scheduler never picks
the migrating request as the preemption victim; before the session starts a
non-decoding request only defers it (refused after a timeout); after the
start a regression refuses the session instead of waiting or failing.
"""

import dataclasses
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from vllm.bridge_tp import kv_stream, phase8_source
from vllm.bridge_tp.preemption_guard import MigrationSourceGuard, pick_victim
from vllm.bridge_tp.runtime_control import RuntimeControl
from vllm.bridge_tp.source_ranks import read_failure

SCHEDULER = Path(__file__).resolve().parents[2] / "vllm/v1/core/sched/scheduler.py"


def arm(run, prefix="pressure-abc", armed=True, state=None):
    RuntimeControl(
        armed=armed, migration_id="m", source_request_id_prefix=prefix
    ).write(run)
    if state is not None:
        (run / "takeover_state.json").write_text(json.dumps(dict(state=state)))


class GuardTests(unittest.TestCase):
    def test_protects_the_armed_request_until_the_session_ends(self):
        running = ["cmpl-short-0-1", "cmpl-pressure-abc-0-9f", "cmpl-other-0-2"]
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            guard = MigrationSourceGuard(run)
            self.assertIsNone(guard.protected_request_id(running))  # no control
            arm(run)
            self.assertEqual(
                guard.protected_request_id(running), "cmpl-pressure-abc-0-9f"
            )
            arm(run, state="PREPARING")
            self.assertEqual(
                guard.protected_request_id(running), "cmpl-pressure-abc-0-9f"
            )
            for state in ("COMMITTED", "CANCELLED"):
                arm(run, state=state)
                self.assertIsNone(guard.protected_request_id(running))
            arm(run, armed=False)
            self.assertIsNone(guard.protected_request_id(running))

    def test_broken_files_protect_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            arm(run)
            (run / "takeover_state.json").write_text("{not json")
            guard = MigrationSourceGuard(run)
            self.assertIsNone(guard.protected_request_id(["cmpl-pressure-abc-0"]))

    def test_enabled_only_for_bridgetp_sources(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(MigrationSourceGuard.from_env())
        env = {"BRIDGETP_STREAM_ENABLED": "1", "BRIDGETP_STREAM_RUN_DIR": "/x"}
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNotNone(MigrationSourceGuard.from_env())
        with patch.dict(
            os.environ, dict(env, BRIDGETP_PROTECT_MIGRATING_SOURCE="0"), clear=True
        ):
            self.assertIsNone(MigrationSourceGuard.from_env())


class VictimTests(unittest.TestCase):
    def setUp(self):
        self.reqs = [
            NS(request_id="a", priority=0, arrival_time=1),
            NS(request_id="m", priority=0, arrival_time=3),
            NS(request_id="b", priority=0, arrival_time=2),
        ]

    def test_fcfs_takes_the_last_unprotected(self):
        self.assertEqual(pick_victim(self.reqs, None).request_id, "b")
        self.assertEqual(pick_victim(self.reqs[:2], "m").request_id, "a")

    def test_priority_skips_the_protected_maximum(self):
        key = lambda r: (r.priority, r.arrival_time)  # noqa: E731
        self.assertEqual(pick_victim(self.reqs, None, key).request_id, "m")
        self.assertEqual(pick_victim(self.reqs, "m", key).request_id, "b")

    def test_only_the_protected_request_left(self):
        self.assertIsNone(pick_victim([self.reqs[1]], "m"))


class SchedulerWiringTests(unittest.TestCase):
    def test_preemption_consults_the_guard_and_keeps_vllm_paths(self):
        text = SCHEDULER.read_text(encoding="utf-8")
        start = text.index("# Preempt the lowest-priority request.")
        block = text[start : text.index("self._preempt_request(preempted_req", start)]
        self.assertIn("guard.protected_request_id(", block)
        self.assertIn("pick_victim(", block)
        # Only the protected request left: stop scheduling, never preempt it.
        self.assertIn("if preempted_req is None:", block)
        # Without a protected request vLLM's own choices are unchanged.
        self.assertIn("preempted_req = self.running.pop()", block)
        self.assertIn("elif self.policy == SchedulingPolicy.PRIORITY:", block)
        self.assertIn("MigrationSourceGuard.from_env()", text)


@dataclasses.dataclass(frozen=True)
class Config:
    run_dir: Path
    migration_id: str = "m-start"
    target_tp_size: int = 2
    persistent_channel: bool = True
    gpu_direct_delta: bool = True
    gpu_direct_history: bool = True
    stop_and_copy: bool = False
    after_output_tokens: int = 0


def publish(config, *, computed, known, scheduled=1):
    request = NS(output_token_ids=[1] * 10, num_tokens=known)
    kv_stream._publish_request(
        config=config,
        request_id="r",
        kv_caches=[object()],
        requests={"r": request},
        input_batch=NS(req_id_to_index={"r": 0}, num_computed_tokens_cpu=[computed]),
        scheduler_output=NS(
            num_scheduled_tokens={"r": scheduled}, scheduled_spec_decode_tokens={}
        ),
        kv_cache_config=None,
        attn_groups=[],
        cache_dtype="auto",
        model_name="m",
        tp_rank=0,
        tp_world_size=1,
        async_scheduling=False,
    )


class StartDeferralTests(unittest.TestCase):
    def tearDown(self):
        kv_stream._start_deferred_since.pop("m-start", None)
        kv_stream._refused_sessions.discard("m-start")

    def test_a_recomputing_request_defers_the_start_instead_of_failing(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Config(Path(folder))
            # 11-12: preempted, first 2048-token recompute chunk of 18697.
            publish(config, computed=0, known=18697, scheduled=2048)
            receipt = json.loads(
                (Path(folder) / "source_start_deferred_receipt.json").read_text()
            )
            self.assertEqual(receipt["status"], "DEFERRED_NOT_DECODING")
            self.assertEqual(receipt["pending_known_tokens"], 16649)
            self.assertIsNone(read_failure(Path(folder)))
            self.assertFalse((Path(folder) / "session_manifest.json").exists())

    def test_a_start_deferred_beyond_the_timeout_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Config(Path(folder))
            with patch.dict(os.environ, {"BRIDGETP_SOURCE_START_DEFER_TIMEOUT_S": "5"}):
                kv_stream._start_deferred_since["m-start"] = time.monotonic() - 6
                publish(config, computed=0, known=18697, scheduled=2048)
            failure = read_failure(Path(folder))
            self.assertIn("request not decoding unavailable", failure["error"])
            self.assertIn("m-start", kv_stream._refused_sessions)

    def test_the_timeout_counts_from_the_first_deferral(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Config(Path(folder))
            role = kv_stream.SourceRole()
            with patch.dict(os.environ, {"BRIDGETP_SOURCE_START_DEFER_TIMEOUT_S": "5"}):
                self.assertFalse(kv_stream._start_deferred_too_long(config, role, "x"))
                self.assertFalse(kv_stream._start_deferred_too_long(config, role, "x"))
                kv_stream._start_deferred_since["m-start"] -= 6
                self.assertTrue(kv_stream._start_deferred_too_long(config, role, "x"))
            with patch.dict(os.environ, {"BRIDGETP_SOURCE_START_DEFER_TIMEOUT_S": "0"}):
                kv_stream._start_deferred_since["m-start"] -= 600
                self.assertFalse(kv_stream._start_deferred_too_long(config, role, "x"))

    def test_memory_deferral_uses_the_same_timeout(self):
        text = Path(kv_stream.__file__).read_text(encoding="utf-8")
        branch = text[text.index("        if not allowed:") :]
        branch = branch[: branch.index("            return\n")]
        self.assertIn('_start_deferred_too_long(config, role, "GPU memory', branch)


class SessionRegressionTests(unittest.TestCase):
    def tearDown(self):
        kv_stream._refused_sessions.discard("m-delta")

    def test_preempted_after_the_start_refuses_the_session(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = NS(run_dir=root, migration_id="m-delta")
            enqueued = []
            state = NS(
                config=config,
                role=phase8_source.SourceRole(),
                request_id="anchor",
                last_computed_token=2174,
                finalizing=False,
                finalized=False,
                stopped=False,
                lifecycle_lock=threading.RLock(),
                enqueue_gpu_delta=lambda **k: enqueued.append(k) or True,
            )
            request = NS(output_token_ids=[1] * 128, num_tokens=18697)
            with patch.object(phase8_source, "_state", state):
                phase8_source.maybe_publish_phase8_delta(
                    config=config,
                    request_id="anchor",
                    kv_caches=[],
                    requests={"anchor": request},
                    input_batch=NS(
                        req_id_to_index={"anchor": 0}, num_computed_tokens_cpu=[0]
                    ),
                    scheduler_output=NS(num_scheduled_tokens={"anchor": 2048}),
                    cache_dtype="auto",
                    attn_groups=[],
                )
            failure = read_failure(root)
            self.assertIn("preempted during migration", failure["error"])
            self.assertIn("m-delta", kv_stream._refused_sessions)
            self.assertEqual(enqueued, [])
            # The controller's cancel runs the normal source cleanup.
            self.assertFalse(state.finalized)


if __name__ == "__main__":
    unittest.main()
