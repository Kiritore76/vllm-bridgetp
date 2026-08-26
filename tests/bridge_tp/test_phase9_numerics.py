# SPDX-License-Identifier: Apache-2.0
"""Phase 9 numerical-fidelity unit tests."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.measure_agreement import load_token_ids
from tools.bridge_tp.probe_logit_ulp import analyze_side, token_value
from tools.bridge_tp.run_fixed_prefix_continuation import request_payload
from tools.bridge_tp.summarize_agreement import validate_pairs
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

        connector = object.__new__(BridgeTPStreamingConnector)
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
    def test_streaming_connector_rejects_wrong_migration_id(self):
        from vllm.bridge_tp.streaming_connector import BridgeTPStreamingConnector

        connector = object.__new__(BridgeTPStreamingConnector)
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


if __name__ == "__main__":
    unittest.main()
