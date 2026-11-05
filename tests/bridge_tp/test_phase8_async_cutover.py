# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm.bridge_tp import phase8_source
from vllm.bridge_tp.request_freeze import RequestFreezeGate


class TestBackgroundCutover(unittest.TestCase):
    def _setup(self, root: Path, wait_for_acks):
        config = SimpleNamespace(
            run_dir=root,
            phase8_cutover_output_tokens=128,
            gpu_direct_delta=True,
            gpu_direct_delta_batch_tokens=16,
            gpu_direct_delta_flush_ms=25.0,
            stop_and_copy=False,
            socket_timeout_s=2.0,
            shadow_strategy="S_NEW_OLD",
            migration_id="async-cutover-test",
            target_tp_size=4,
        )
        state = SimpleNamespace(
            config=config,
            role=phase8_source.SourceRole(),
            request_id="anchor",
            session_token="session",
            last_computed_token=2174,
            last_flush_monotonic=time.monotonic(),
            block_size=16,
            block_axis=0,
            delta_batches=1,
            delta_tokens=1,
            delta_payload_bytes=1024,
            d2h_ms=0.0,
            finalizing=False,
            finalized=False,
            stopped=False,
            lifecycle_lock=threading.RLock(),
            enqueue_gpu_delta=lambda **kwargs: True,
            wait_for_acks=wait_for_acks,
            start_history_transfer=lambda: None,
        )

        def stop_workers():
            state.stopped = True

        state.stop_workers = stop_workers
        request = SimpleNamespace(
            request_id="anchor",
            num_prompt_tokens=2048,
            num_tokens=2176,
            num_computed_tokens=2175,
            output_token_ids=[1] * 128,
            get_token_id=lambda index: index,
        )
        input_batch = SimpleNamespace(
            req_id_to_index={"anchor": 0},
            num_computed_tokens_cpu=[2174],
        )
        scheduler_output = SimpleNamespace(num_scheduled_tokens={"anchor": 1})
        (root / "session_manifest.json").write_text(
            json.dumps({"num_computed_tokens": 2174}), encoding="utf-8"
        )
        return config, state, request, input_batch, scheduler_output

    def _publish(self, config, state, request, input_batch, scheduler_output):
        with (
            patch.object(phase8_source, "_state", state),
            patch.object(
                phase8_source,
                "_get_request_block_ids",
                return_value=([0] * 136, 16),
            ),
            patch.dict("os.environ", {"BRIDGETP_REQUEST_FREEZE_ENABLED": "1"}),
        ):
            phase8_source.maybe_publish_phase8_delta(
                config=config,
                request_id="anchor",
                kv_caches=[],
                requests={"anchor": request},
                input_batch=input_batch,
                scheduler_output=scheduler_output,
                cache_dtype="auto",
                attn_groups=[],
            )

    def test_model_step_returns_before_freeze_receipt_and_delta_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entered = threading.Event()
            release = threading.Event()

            def wait_for_acks():
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test delta ACK never arrived")

            config, state, request, batch, output = self._setup(
                root, wait_for_acks
            )
            started = time.monotonic()
            self._publish(config, state, request, batch, output)
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertTrue(state.finalizing)
            self.assertFalse((root / "cutover_manifest.json").exists())
            self.assertFalse(entered.is_set())

            RequestFreezeGate(root).record_frozen(request, scheduler_step=1)
            self.assertTrue(entered.wait(1))
            self.assertFalse((root / "cutover_manifest.json").exists())
            release.set()
            for _ in range(200):
                if (root / "cutover_manifest.json").exists():
                    break
                time.sleep(0.01)
            cutover = json.loads(
                (root / "cutover_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                cutover["final_delta_finalize_mode"],
                "BACKGROUND_AFTER_REQUEST_FREEZE",
            )
            self.assertEqual(cutover["num_computed_tokens"], 2175)
            self.assertTrue(state.finalized)
            self.assertTrue(state.stopped)

    def test_delta_failure_resumes_only_source_request_without_cutover(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def wait_for_acks():
                raise RuntimeError("rank 2 delta rejected")

            config, state, request, batch, output = self._setup(
                root, wait_for_acks
            )
            self._publish(config, state, request, batch, output)
            RequestFreezeGate(root).record_frozen(request, scheduler_step=1)
            for _ in range(200):
                control_path = root / "request_freeze_control.json"
                if (
                    (root / "cutover_finalize_error.json").exists()
                    and json.loads(control_path.read_text(encoding="utf-8"))[
                        "action"
                    ] == "RESUME"
                ):
                    break
                time.sleep(0.01)
            self.assertFalse((root / "cutover_manifest.json").exists())
            control = json.loads(
                (root / "request_freeze_control.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(control["action"], "RESUME")
            self.assertFalse(RequestFreezeGate(root).is_frozen("anchor"))

    def test_wrong_freeze_boundary_never_drains_or_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            waited = threading.Event()

            def wait_for_acks():
                waited.set()

            config, state, request, batch, output = self._setup(
                root, wait_for_acks
            )
            self._publish(config, state, request, batch, output)
            request.num_computed_tokens = 2174
            RequestFreezeGate(root).record_frozen(request, scheduler_step=1)
            for _ in range(200):
                if (root / "cutover_finalize_error.json").exists():
                    break
                time.sleep(0.01)
            self.assertFalse(waited.is_set())
            self.assertFalse((root / "cutover_manifest.json").exists())
            error = json.loads(
                (root / "cutover_finalize_error.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("wrong token boundary", error["error"])

    def _wait_for(self, predicate):
        for _ in range(300):
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_follower_reports_and_never_freezes_or_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, state, request, batch, output = self._setup(root, lambda: None)
            state.role = phase8_source.SourceRole(2, 1)
            self._publish(config, state, request, batch, output)
            self.assertFalse((root / "request_freeze_control.json").exists())
            report = root / "source_rank_reports" / "final_delta_rank_1.json"
            self.assertFalse(report.exists())
            RequestFreezeGate(root).record_frozen(request, scheduler_step=1)
            self.assertTrue(self._wait_for(report.exists))
            row = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(row["output_tokens"], 128)
            self.assertEqual(row["num_computed_tokens"], 2175)
            self.assertFalse((root / "cutover_manifest.json").exists())
            self.assertTrue(state.stopped)

    def test_leader_publishes_only_after_every_rank_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, state, request, batch, output = self._setup(root, lambda: None)
            state.role = phase8_source.SourceRole(2, 0)
            self._publish(config, state, request, batch, output)
            self.assertTrue((root / "request_freeze_control.json").exists())
            RequestFreezeGate(root).record_frozen(request, scheduler_step=1)
            time.sleep(0.2)
            self.assertFalse((root / "cutover_manifest.json").exists())
            phase8_source.report_stage(
                root,
                phase8_source.SourceRole(2, 1),
                "final_delta",
                output_tokens=128,
                num_computed_tokens=2175,
            )
            manifest = root / "cutover_manifest.json"
            self.assertTrue(self._wait_for(manifest.exists))

    def test_leader_resumes_when_a_rank_fails_at_the_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, state, request, batch, output = self._setup(root, lambda: None)
            state.role = phase8_source.SourceRole(2, 0)
            self._publish(config, state, request, batch, output)
            RequestFreezeGate(root).record_frozen(request, scheduler_step=1)
            phase8_source.report_failure(
                root, phase8_source.SourceRole(2, 1), "delta failed"
            )
            control = root / "request_freeze_control.json"
            self.assertTrue(
                self._wait_for(
                    lambda: json.loads(control.read_text(encoding="utf-8"))["action"]
                    == "RESUME"
                )
            )
            self.assertTrue((root / "cutover_finalize_error.json").exists())
            self.assertFalse((root / "cutover_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
