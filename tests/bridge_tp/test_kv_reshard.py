# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

import torch

from vllm.bridge_tp.kv_reshard import (
    iter_tp_rank_shards,
    validate_exact_roundtrip,
    validate_tp1_layers,
)


class TestKVReshard(unittest.TestCase):
    def setUp(self) -> None:
        self.layers = {
            "model.layers.0.self_attn.attn": torch.arange(
                3 * 2 * 4 * 8 * 5, dtype=torch.float32
            ).reshape(3, 2, 4, 8, 5),
            "model.layers.1.self_attn.attn": torch.arange(
                3 * 2 * 4 * 8 * 5, dtype=torch.float32
            ).reshape(3, 2, 4, 8, 5)
            + 1,
        }

    def test_tp1_to_tp4_roundtrip_is_exact(self) -> None:
        rank_layers = [
            shard
            for _, shard in iter_tp_rank_shards(
                self.layers,
                head_axis=3,
                target_tp_size=4,
                expected_source_kv_heads=8,
            )
        ]

        self.assertEqual(len(rank_layers), 4)
        for shard in rank_layers:
            for tensor in shard.values():
                self.assertEqual(tensor.shape, (3, 2, 4, 2, 5))
                self.assertTrue(tensor.is_contiguous())

        result = validate_exact_roundtrip(self.layers, rank_layers, head_axis=3)
        self.assertTrue(result["exact_roundtrip"])
        self.assertEqual(result["num_layers"], 2)

    def test_rejects_nondivisible_head_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            validate_tp1_layers(
                self.layers,
                head_axis=3,
                target_tp_size=3,
                expected_source_kv_heads=8,
            )

    def test_rejects_wrong_observed_head_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 4"):
            validate_tp1_layers(
                self.layers,
                head_axis=3,
                target_tp_size=2,
                expected_source_kv_heads=4,
            )


if __name__ == "__main__":
    unittest.main()
