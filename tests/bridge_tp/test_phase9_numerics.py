# SPDX-License-Identifier: Apache-2.0
"""Phase 9 numerical-fidelity unit tests."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.measure_agreement import load_token_ids
from tools.bridge_tp.probe_logit_ulp import analyze_side, token_value
from tools.bridge_tp.run_fixed_prefix_continuation import request_payload
from tools.bridge_tp.summarize_agreement import validate_pairs
from vllm.bridge_tp.block_layout import snapshot_target_block_ids
from vllm.bridge_tp.config import BridgeTPDumpConfig
from vllm.bridge_tp.controller.numerics import (
    agreement_length,
    analyze_candidate_gap,
    paired_bootstrap_mean_difference,
    summarize_samples,
    ulp_at,
)
from vllm.bridge_tp.logit_capture import (
    LogitCaptureConfig,
    _resolve_target_request_id,
    token_ids_sha256,
)


def _streaming_connector_stub(connector_type):
    """Initialize the state used by connector methods without starting workers."""
    connector = object.__new__(connector_type)
    connector.gpu_resident_shadow = False
    connector.persistent_channel = False
    connector._pending_requests = {}
    connector._active_requests = {}
    connector._model_wait_pending_requests = set()
    connector._gpu_restore_quiescence_lock = threading.Lock()
    connector._gpu_restore_quiescence = {}
    return connector


class TestRecordedDtypeUlp(unittest.TestCase):
    def test_bfloat16_spacing_depends_on_binade(self):
        self.assertEqual(ulp_at(8.0, "bfloat16"), 0.0625)
        self.assertEqual(ulp_at(16.0, "bfloat16"), 0.125)
        self.assertEqual(ulp_at(32.0, "bfloat16"), 0.25)

    def test_actual_raw_values_drive_the_band(self):
        result = analyze_candidate_gap(
            stage="raw",
            dtype="bfloat16",
            first_token_id=8381,
            second_token_id=1372,
            first_value=20.0,
            second_value=19.875,
        )
        self.assertEqual(result.gap_ulps, 1.0)
        self.assertEqual(result.descriptive_band, "WITHIN_ONE_RECORDED_DTYPE_ULP")

    def test_gap_band_does_not_assign_causality(self):
        result = analyze_candidate_gap(
            stage="processed",
            dtype="float32",
            first_token_id=1,
            second_token_id=2,
            first_value=1.0,
            second_value=0.5,
        ).to_json()
        self.assertNotIn("verdict", result)
        self.assertNotIn("migration", result)


class TestAgreementStatistics(unittest.TestCase):
    def test_target_local_agreement(self):
        self.assertEqual(agreement_length([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(agreement_length([1, 2], [1, 2, 3], budget=8), 2)

    def test_summary_reports_full_budget_fraction(self):
        result = summarize_samples([8, 8, 4, 2], budget=8)
        self.assertEqual(result["median"], 6.0)
        self.assertEqual(result["fully_agreeing_fraction"], 0.5)

    def test_paired_bootstrap_preserves_pairing(self):
        result = paired_bootstrap_mean_difference(
            [20, 30, 40],
            [10, 20, 30],
            resamples=200,
            seed=7,
        )
        self.assertEqual(result["estimate"], 10)
        self.assertEqual(result["ci_low"], 10)
        self.assertEqual(result["ci_high"], 10)
        self.assertEqual(result["paired_differences"], [10, 10, 10])

    def test_abcd_pairing_requires_same_prefix_k_and_metadata(self):
        base = {
            "request_id": "r0",
            "boundary_k": 53,
            "budget": 256,
            "fixed_prefix_sha256": "abc",
            "metadata": {
                "model": "qwen",
                "gpu_platform": "A100 PCIe",
                "vllm_commit": "deadbeef",
                "cutover_rule": "fixed",
            },
        }
        groups = {name: {"r0": {**base, "group": name}} for name in "ABCD"}
        self.assertEqual(validate_pairs(groups), ["r0"])
        groups["B"]["r0"]["fixed_prefix_sha256"] = "different"
        with self.assertRaises(SystemExit):
            validate_pairs(groups)


class TestEvidenceTools(unittest.TestCase):
    def test_pending_tail_block_is_excluded_from_snapshot_restore(self):
        selected = snapshot_target_block_ids(
            (list(range(100, 113)),),
            request_num_tokens=193,
            block_size=16,
            snapshot_blocks=12,
            error_message="bad allocation",
        )
        self.assertEqual(selected, list(range(100, 112)))
        self.assertNotIn(112, selected)

    def test_pending_tail_block_can_be_allocated_on_first_target_step(self):
        # The failed O129 smoke had 2048 prompt + 129 output tokens: 2176
        # computed tokens fill 136 blocks, and the pending token starts block 137.
        selected = snapshot_target_block_ids(
            (list(range(136)),),
            request_num_tokens=2177,
            block_size=16,
            snapshot_blocks=136,
            error_message="bad allocation",
        )
        self.assertEqual(selected, list(range(136)))

    def test_unexpected_extra_tail_block_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bad allocation"):
            snapshot_target_block_ids(
                (list(range(14)),),
                request_num_tokens=193,
                block_size=16,
                snapshot_blocks=12,
                error_message="bad allocation",
            )
        with self.assertRaisesRegex(ValueError, "bad allocation"):
            snapshot_target_block_ids(
                (list(range(13)),),
                request_num_tokens=193,
                block_size=16,
                snapshot_blocks=10,
                error_message="bad allocation",
            )

    def test_measurement_reads_fixed_prefix_provenance_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixed_prefix.json"
            path.write_text(json.dumps({"fixed_token_ids": [3, 4, 5]}))
            self.assertEqual(load_token_ids(path), [3, 4, 5])

    def test_logit_probe_uses_actual_candidate_values(self):
        capture = {
            "stages": {
                name: {
                    "dtype": "bfloat16",
                    "candidate_values": {"1": 20.0, "2": 19.875},
                }
                for name in ("raw", "processed")
            }
        }
        result = analyze_side("control", capture, [1, 2])
        self.assertEqual(result["raw"]["gap_ulps"], 1.0)
        self.assertEqual(
            result["processed"]["descriptive_band"],
            "WITHIN_ONE_RECORDED_DTYPE_ULP",
        )

    def test_logit_probe_falls_back_to_saved_tensor(self):
        stage = {
            "candidate_values": {},
            "top_token_ids": [1],
            "top_values": [20.0],
            "tensor_file": "raw_logits.pt",
        }
        with patch(
            "tools.bridge_tp.probe_logit_ulp._tensor_token_value",
            return_value=19.75,
        ) as loader:
            value = token_value(stage, 2, "control/raw", Path("capture"))
        self.assertEqual(value, 19.75)
        loader.assert_called_once_with(stage, 2, "control/raw", Path("capture"))

    def test_fixed_prefix_request_freezes_strict_greedy_contract(self):
        payload = request_payload(
            model="bridgetp-model",
            prompt=[10, 11],
            request_id="r0",
            max_tokens=8,
        )
        self.assertEqual(payload["prompt"], [10, 11])
        self.assertFalse(payload["add_special_tokens"])
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["repetition_penalty"], 1.0)
        self.assertTrue(payload["return_token_ids"])


class TestOptInConfiguration(unittest.TestCase):
    def test_kv_dump_defaults_to_tp1_only(self):
        with patch.dict(os.environ, {}, clear=True):
            config = BridgeTPDumpConfig.from_env()
        self.assertEqual(config.allowed_tp_world_sizes, (1,))

    def test_tp4_dump_requires_explicit_opt_in(self):
        with patch.dict(
            os.environ,
            {"BRIDGETP_DUMP_TP_WORLD_SIZES": "1,4"},
            clear=True,
        ):
            config = BridgeTPDumpConfig.from_env()
        self.assertEqual(config.allowed_tp_world_sizes, (1, 4))

    def test_logit_capture_parses_global_indices_and_candidates(self):
        with patch.dict(
            os.environ,
            {
                "BRIDGETP_LOGIT_CAPTURE_ENABLED": "1",
                "BRIDGETP_LOGIT_CAPTURE_INDICES": "98,99",
                "BRIDGETP_LOGIT_CAPTURE_CANDIDATE_TOKEN_IDS": "8381,1372",
                "BRIDGETP_LOGIT_CAPTURE_GLOBAL_OFFSET": "53",
            },
            clear=True,
        ):
            config = LogitCaptureConfig.from_env()
        self.assertEqual(config.global_indices, (98, 99))
        self.assertEqual(config.candidate_token_ids, (8381, 1372))
        self.assertEqual(config.global_index_offset, 53)

    def test_prefix_hash_is_stable_and_order_sensitive(self):
        self.assertEqual(token_ids_sha256([1, 2, 3]), token_ids_sha256([1, 2, 3]))
        self.assertNotEqual(token_ids_sha256([1, 2, 3]), token_ids_sha256([3, 2, 1]))

    def test_logit_filter_accepts_public_completion_request_id(self):
        self.assertEqual(
            _resolve_target_request_id("control", ["cmpl-control-0-deadbeef"]),
            "cmpl-control-0-deadbeef",
        )
        self.assertIsNone(
            _resolve_target_request_id("control", ["cmpl-different-0"])
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_streaming_connector_refuses_unmarked_target_recompute(self):
        from vllm.bridge_tp.streaming_connector import BridgeTPStreamingConnector

        connector = _streaming_connector_stub(BridgeTPStreamingConnector)
        connector._manifest = {
            "migration_id": "migration",
            "all_known_token_ids": [1, 2, 3],
            "num_computed_tokens": 2,
        }
        request = types.SimpleNamespace(
            request_id="cmpl-bridgetp-phase9-target-run-0",
            kv_transfer_params=None,
            prompt_token_ids=[1, 2, 3],
            num_tokens=3,
        )
        with self.assertRaisesRegex(ValueError, "refusing local recomputation"):
            connector._request_matches(request)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_streaming_connector_allows_unmarked_background_before_manifest(
        self,
    ) -> None:
        from vllm.bridge_tp.streaming_connector import BridgeTPStreamingConnector

        connector = _streaming_connector_stub(BridgeTPStreamingConnector)
        connector._manifest = None
        connector.manifest_path = Path("missing-staging-manifest.json")
        request = types.SimpleNamespace(
            request_id="cmpl-bridgetp-cap0-load-target_000",
            kv_transfer_params=None,
            prompt_token_ids=[1, 2, 3],
            num_tokens=3,
        )
        self.assertFalse(connector._request_matches(request))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_streaming_connector_rejects_wrong_migration_id(self):
        from vllm.bridge_tp.streaming_connector import BridgeTPStreamingConnector

        connector = _streaming_connector_stub(BridgeTPStreamingConnector)
        connector._manifest = {
            "migration_id": "migration",
            "all_known_token_ids": [1, 2, 3],
            "num_computed_tokens": 2,
        }
        request = types.SimpleNamespace(
            request_id="cmpl-bridgetp-phase9-target-run-0",
            kv_transfer_params={"bridgetp_migration_id": "wrong"},
            prompt_token_ids=[1, 2, 3],
            num_tokens=3,
        )
        with self.assertRaisesRegex(ValueError, "migration id differs"):
            connector._request_matches(request)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_streaming_connector_keeps_pending_tail_block_out_of_restore(self):
        from vllm.bridge_tp.stream_protocol import MIGRATION_PARAM
        from vllm.bridge_tp.streaming_connector import BridgeTPStreamingConnector

        connector = _streaming_connector_stub(BridgeTPStreamingConnector)
        connector._manifest = {
            "migration_id": "migration",
            "source_request_id": "source",
            "all_known_token_ids": list(range(193)),
            "num_computed_tokens": 192,
            "num_blocks": 12,
            "block_size": 16,
        }
        connector._pending_requests = {}
        connector._claimed_target_request_id = None
        request = types.SimpleNamespace(
            request_id="target",
            kv_transfer_params={MIGRATION_PARAM: "migration"},
            prompt_token_ids=list(range(193)),
            num_tokens=193,
        )
        allocated = (list(range(100, 113)),)
        blocks = types.SimpleNamespace(get_block_ids=lambda: allocated)

        connector.update_state_after_alloc(request, blocks, 192)
        scheduler_output = types.SimpleNamespace(
            scheduled_new_reqs=[
                types.SimpleNamespace(
                    req_id="target",
                    num_computed_tokens=192,
                    block_ids=allocated,
                )
            ]
        )
        metadata = connector.build_connector_meta(scheduler_output)

        self.assertEqual(len(metadata.requests), 1)
        self.assertEqual(
            metadata.requests[0].target_block_ids,
            list(range(100, 112)),
        )
        self.assertNotIn(112, metadata.requests[0].target_block_ids)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_streaming_connector_rearms_wait_only_for_migrated_request(self):
        from vllm.bridge_tp.streaming_connector import (
            BridgeTPStreamingConnector,
            BridgeTPStreamMetadata,
        )

        connector = _streaming_connector_stub(BridgeTPStreamingConnector)
        connector._pending_requests = {}
        connector._model_wait_pending_requests = {"migrated"}

        unrelated = types.SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=["background"]),
        )
        unrelated_metadata = connector.build_connector_meta(unrelated)
        self.assertEqual(unrelated_metadata.model_wait_request_ids, [])
        self.assertEqual(connector._model_wait_pending_requests, {"migrated"})

        migrated = types.SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=["migrated"]),
        )
        migrated_metadata = connector.build_connector_meta(migrated)
        self.assertEqual(
            migrated_metadata.model_wait_request_ids, ["migrated"]
        )
        self.assertEqual(connector._model_wait_pending_requests, set())

        worker = _streaming_connector_stub(BridgeTPStreamingConnector)
        worker._connector_metadata = BridgeTPStreamMetadata(
            model_wait_request_ids=["migrated"]
        )
        with patch.object(worker, "_arm_model_stream_waits") as arm:
            worker.start_load_kv(None)
        arm.assert_called_once_with(worker._connector_metadata)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_persistent_earliest_ready_prebind_receives_second_session(self):
        from vllm.bridge_tp.streaming_connector import BridgeTPStreamingConnector

        class Receiver:
            def __init__(self, **kwargs):
                self.calls = []
                self.closed = False

            def receive(self, **kwargs):
                if self.closed:
                    raise RuntimeError("receiver was closed before reuse")
                self.calls.append(kwargs["migration_id"])
                return object()

            def close(self):
                self.closed = True

        connector = _streaming_connector_stub(BridgeTPStreamingConnector)
        connector.gpu_resident_shadow = True
        connector.persistent_channel = True
        connector.shadow_cutover_output_tokens = 0
        connector.defer_communicator_destroy = False
        connector.post_takeover_communicator_destroy = False
        connector._registered_kv_caches = {
            "layer": types.SimpleNamespace(device="cuda:0")
        }
        connector._prebound_gpu_receiver = None
        connector._prebound_gpu_receiver_lock = threading.Lock()
        connector._persistent_gpu_receiver = None
        connector._persistent_gpu_receiver_lock = threading.Lock()
        connector._prebound_gpu_history = None
        connector._prebound_gpu_history_lock = threading.Lock()
        connector._prebind_receiver_thread = None
        connector._prebind_receiver_stop = threading.Event()
        connector._prebind_receiver_ready = threading.Event()

        with tempfile.TemporaryDirectory() as directory:
            connector.manifest_path = Path(directory) / "session_manifest.json"

            def publish(migration_id):
                connector.manifest_path.write_text(
                    json.dumps({
                        "history_transport": "NCCL_P2P_GPU_DIRECT",
                        "delta_transport": "NCCL_P2P_GPU_DIRECT_PERSISTENT",
                        "migration_id": migration_id,
                        "source_request_id": "source",
                        "ranks": [{"host": "localhost", "port": 30400}],
                        "layers": [],
                    }),
                    encoding="utf-8",
                )

            def receipt_is(migration_id):
                path = Path(directory) / "gpu_initial_receipts" / "tp_rank_0.json"
                return path.is_file() and json.loads(path.read_text())[
                    "migration_id"
                ] == migration_id

            publish("first")
            with (
                patch(
                    "vllm.bridge_tp.gpu_direct_history.GpuDirectHistoryReceiver",
                    Receiver,
                ),
                patch(
                    "vllm.bridge_tp.streaming_connector.get_tp_group",
                    return_value=types.SimpleNamespace(rank_in_group=0),
                ),
            ):
                connector._start_gpu_direct_receiver_prebind()
                try:
                    for migration_id in ("first", "second"):
                        if migration_id == "second":
                            publish(migration_id)
                        for _ in range(200):
                            if receipt_is(migration_id):
                                break
                            connector._prebind_receiver_stop.wait(0.01)
                        self.assertTrue(receipt_is(migration_id))
                    self.assertEqual(
                        connector._prebound_gpu_receiver.calls,
                        ["first", "second"],
                    )
                    self.assertFalse(connector._prebound_gpu_receiver.closed)
                finally:
                    connector._prebind_receiver_stop.set()
                    connector._prebind_receiver_thread.join(timeout=2)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_manifest_switch_accepts_prebound_current_session_only(self):
        from vllm.bridge_tp.stream_protocol import PROTOCOL_VERSION
        from vllm.bridge_tp.streaming_connector import BridgeTPStreamingConnector

        connector = _streaming_connector_stub(BridgeTPStreamingConnector)
        connector.persistent_channel = True
        connector._manifest = {"migration_id": "first"}
        connector._persistent_gpu_receiver_lock = threading.Lock()
        connector._claimed_target_request_id = "old-target"
        connector.expected_phase = "BridgeTP D3 Phase 7"
        connector._target_model = "model"
        connector._target_block_size = 16
        connector.channel_generation = 1
        lifecycle = types.SimpleNamespace(
            state=types.SimpleNamespace(value="ACTIVE"),
            active_session=types.SimpleNamespace(migration_id="first"),
        )
        connector._persistent_gpu_receiver = types.SimpleNamespace(
            lifecycle=lifecycle
        )
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "source_tp_size": 1,
            "target_tp_size": 4,
            "pending_known_tokens": 1,
            "migration_id": "second",
            "phase": connector.expected_phase,
            "model": connector._target_model,
            "block_size": connector._target_block_size,
            "ranks": [{"target_tp_rank": rank} for rank in range(4)],
            "all_known_token_ids": [1, 2],
            "computed_token_ids": [1],
            "pending_token_ids": [2],
            "num_computed_tokens": 1,
            "persistent_channel": True,
            "channel_generation": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            connector.manifest_path = Path(directory) / "session_manifest.json"
            connector.manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "before the previous"):
                connector._load_manifest("second")
            lifecycle.active_session.migration_id = "second"
            self.assertEqual(connector._load_manifest("second"), manifest)
            self.assertIsNone(connector._claimed_target_request_id)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_streaming_connector_rejects_unexpected_extra_tail_blocks(self):
        from vllm.bridge_tp.streaming_connector import BridgeTPStreamingConnector

        connector = _streaming_connector_stub(BridgeTPStreamingConnector)
        connector._manifest = {
            "num_blocks": 12,
            "block_size": 16,
        }
        request = types.SimpleNamespace(num_tokens=193)

        with self.assertRaisesRegex(ValueError, "differs from live snapshot"):
            connector._snapshot_target_block_ids(
                request,
                (list(range(14)),),
                "Target block allocation differs from live snapshot",
            )


if __name__ == "__main__":
    unittest.main()
