# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from vllm.bridge_tp.remote_attention_protocol import (
    BlockTransferRecord,
    KVPartition,
    RemoteAttentionGeometry,
    history_transfer_bytes,
    make_partition,
    one_layer_query_wire_bytes,
    one_layer_rank_transfer_bytes,
    one_layer_statistics_wire_bytes,
    projected_token_wire_bytes,
    validate_block_records,
    validate_measurements,
)


class TestRemoteAttentionProtocol(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = RemoteAttentionGeometry()

    def test_qwen25_geometry(self) -> None:
        self.geometry.validate()
        self.assertEqual(self.geometry.query_heads_per_rank, 10)
        self.assertEqual(self.geometry.kv_heads_per_rank, 2)
        self.assertEqual(self.geometry.query_heads_per_kv_head, 5)
        self.assertEqual(self.geometry.aggregate_kv_bytes_per_token, 196608)

    def test_partition_rounds_remote_suffix_to_blocks(self) -> None:
        partition = make_partition(1000, 0.25, 16)
        self.assertEqual(partition.remote_tokens, 240)
        self.assertEqual(partition.local_tokens, 760)
        self.assertEqual(partition.ownership_boundary, 760)

    def test_partition_rejects_incomplete_coverage(self) -> None:
        partition = KVPartition(100, 60, 32, 16)
        with self.assertRaisesRegex(ValueError, "cover the context"):
            partition.validate()

    def test_partition_rejects_partial_remote_block(self) -> None:
        partition = KVPartition(100, 68, 32, 16)
        partition.validate()
        invalid = KVPartition(100, 70, 30, 16)
        with self.assertRaisesRegex(ValueError, "block boundary"):
            invalid.validate()

    def test_history_transfer_volume_uses_all_layers(self) -> None:
        partition = make_partition(1024, 0.5, 16)
        self.assertEqual(
            history_transfer_bytes(partition, self.geometry),
            512 * 196608,
        )
        self.assertEqual(
            one_layer_rank_transfer_bytes(partition, self.geometry),
            2 * 2 * 512 * 128 * 2,
        )

    def test_remote_attention_wire_volume(self) -> None:
        self.assertEqual(one_layer_query_wire_bytes(self.geometry), 10240)
        self.assertEqual(
            one_layer_statistics_wire_bytes(self.geometry),
            40 * 130 * 4,
        )
        self.assertEqual(
            projected_token_wire_bytes(self.geometry),
            48 * (10240 + 40 * 130 * 4),
        )

    def test_shadow_record_cannot_release_source(self) -> None:
        record = BlockTransferRecord(
            request_id="request-1",
            migration_epoch="epoch-1",
            block_id=0,
            token_start=0,
            token_end=16,
            kind="HISTORY",
            state="SHADOW",
            source_present=False,
            target_verified=True,
            source_released=True,
            verified_unix_s=1.0,
            released_unix_s=2.0,
        )
        with self.assertRaisesRegex(ValueError, "Shadow cannot release"):
            record.validate()

    def test_bridge_release_requires_verified_target(self) -> None:
        record = BlockTransferRecord(
            request_id="request-1",
            migration_epoch="epoch-1",
            block_id=0,
            token_start=0,
            token_end=16,
            kind="HISTORY",
            state="BRIDGE",
            source_present=False,
            target_verified=False,
            source_released=True,
            released_unix_s=2.0,
        )
        with self.assertRaisesRegex(ValueError, "verified TP4 ownership"):
            record.validate()

    def test_valid_bridge_release_and_duplicate_detection(self) -> None:
        record = BlockTransferRecord(
            request_id="request-1",
            migration_epoch="epoch-1",
            block_id=3,
            token_start=48,
            token_end=64,
            kind="HISTORY",
            state="BRIDGE",
            source_present=False,
            target_verified=True,
            source_released=True,
            queued_unix_s=1.0,
            sent_unix_s=2.0,
            verified_unix_s=3.0,
            released_unix_s=4.0,
        )
        record.validate()
        self.assertEqual(validate_block_records([record])["status"], "PASS")
        duplicate = validate_block_records([record, record])
        self.assertEqual(duplicate["status"], "FAIL")
        self.assertIn("duplicate block identity", duplicate["errors"][0])

    def test_acceptance_passes_complete_measured_case(self) -> None:
        row = {
            "status": "PASS",
            "finite": True,
            "max_abs_error": 1e-5,
            "mean_abs_error": 1e-6,
            "staged_kv_bytes": 4096,
            "expected_staged_kv_bytes": 4096,
            "step_p50_ms": 1.0,
        }
        acceptance = validate_measurements(
            [row],
            expected_cases=1,
            max_abs_tolerance=1e-4,
            mean_abs_tolerance=1e-5,
        )
        self.assertEqual(acceptance["status"], "PASS")
        self.assertEqual(acceptance["errors"], [])

    def test_acceptance_fails_closed(self) -> None:
        row = {
            "status": "FAIL",
            "finite": False,
            "max_abs_error": 0.1,
            "mean_abs_error": 0.01,
            "staged_kv_bytes": 1,
            "expected_staged_kv_bytes": 2,
            "step_p50_ms": 0,
        }
        acceptance = validate_measurements(
            [row],
            expected_cases=2,
            max_abs_tolerance=1e-4,
            mean_abs_tolerance=1e-5,
        )
        self.assertEqual(acceptance["status"], "FAIL")
        self.assertGreaterEqual(len(acceptance["errors"]), 6)


if __name__ == "__main__":
    unittest.main()
