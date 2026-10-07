# SPDX-License-Identifier: Apache-2.0
"""Exact history evidence rejects partial and mismatched allocations."""

import copy
import unittest

from vllm.bridge_tp.history_coverage import (
    history_block_coverage,
    valid_history_block_coverage,
)


class TestHistoryCoverage(unittest.TestCase):
    def setUp(self):
        self.blocks = [7, 2, 10]
        self.validation = {
            "exact_readback": True,
            "num_target_blocks": 3,
            "num_layers": 48,
        }
        self.receipt = {
            "status": "INITIAL_HISTORY_GPU_RESIDENT",
            "migration_id": "m1",
            "tp_rank": 2,
            "target_request_id": "target1",
            "end_token": 35,
            "exact_readback": True,
            "history_block_coverage": history_block_coverage(
                block_size=16,
                end_token=35,
                target_block_ids=self.blocks,
                validation=self.validation,
            ),
        }

    def valid(self, receipt):
        return valid_history_block_coverage(
            receipt,
            migration_id="m1",
            rank=2,
            end_token=35,
            block_size=16,
            expected_blocks=3,
            target_block_ids=[*self.blocks, 12],
            target_request_id="target1",
        )

    def test_partial_last_block_and_noncontiguous_allocation(self):
        self.assertTrue(self.valid(self.receipt))
        self.blocks[0] = 99
        self.assertEqual(
            self.receipt["history_block_coverage"]["target_block_ids"], [7, 2, 10]
        )

    def test_requires_successful_full_rank_readback(self):
        for patch in (
            {"exact_readback": False},
            {"num_target_blocks": 2},
            {"num_layers": 0},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                history_block_coverage(
                    block_size=16,
                    end_token=35,
                    target_block_ids=[7, 2, 10],
                    validation={**self.validation, **patch},
                )

    def test_rejects_missing_duplicate_or_different_physical_blocks(self):
        for blocks in ([7, 2], [7, 2, 2], [7, 2, 11], [7, 2, True]):
            receipt = copy.deepcopy(self.receipt)
            receipt["history_block_coverage"]["target_block_ids"] = blocks
            with self.subTest(blocks=blocks):
                self.assertFalse(self.valid(receipt))

    def test_rejects_identity_or_boundary_mismatch(self):
        for patch in (
            {"migration_id": "other"},
            {"tp_rank": 1},
            {"tp_rank": True},
            {"target_request_id": "other"},
            {"end_token": 34},
            {"exact_readback": False},
            {"status": "INITIAL_HISTORY_GPU_BUFFERED"},
        ):
            with self.subTest(patch=patch):
                self.assertFalse(self.valid({**self.receipt, **patch}))

    def test_rejects_truncated_coverage_and_unverified_scope(self):
        for patch in (
            {"logical_block_end_exclusive": 2},
            {"end_token": 34},
            {"verification_scope": "RECEIVED_ONLY"},
            {"exact_readback": False},
            {"format_version": True},
        ):
            receipt = copy.deepcopy(self.receipt)
            receipt["history_block_coverage"].update(patch)
            with self.subTest(patch=patch):
                self.assertFalse(self.valid(receipt))

    def test_rejects_missing_proof(self):
        receipt = dict(self.receipt)
        del receipt["history_block_coverage"]
        self.assertFalse(self.valid(receipt))
