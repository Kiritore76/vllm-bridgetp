# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

import torch

from vllm.bridge_tp.kv_restore import inject_rank_shard


class TestKVRestore(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "layer.0": torch.arange(3 * 2 * 4 * 2 * 5, dtype=torch.float32).reshape(
                3, 2, 4, 2, 5
            ),
            "layer.1": torch.arange(3 * 2 * 4 * 2 * 5, dtype=torch.float32).reshape(
                3, 2, 4, 2, 5
            )
            + 1,
        }
        self.destination = {
            name: torch.full((12, 2, 4, 2, 5), -1.0) for name in self.source
        }

    def test_injects_noncontiguous_target_blocks_with_exact_readback(self) -> None:
        block_ids = [7, 2, 10]
        result = inject_rank_shard(
            self.destination,
            self.source,
            block_ids,
            block_axis=0,
        )

        self.assertTrue(result["exact_readback"])
        self.assertEqual(result["num_layers"], 2)
        for layer_name, source in self.source.items():
            self.assertTrue(
                torch.equal(self.destination[layer_name][block_ids], source)
            )
            self.assertTrue(torch.all(self.destination[layer_name][0] == -1.0))

    def test_rejects_duplicate_target_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "not unique"):
            inject_rank_shard(
                self.destination,
                self.source,
                [2, 2, 3],
                block_axis=0,
            )

    def test_rejects_layer_shape_mismatch(self) -> None:
        bad_destination = dict(self.destination)
        bad_destination["layer.0"] = torch.empty(12, 2, 4, 3, 5)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            inject_rank_shard(
                bad_destination,
                self.source,
                [1, 2, 3],
                block_axis=0,
            )

    def test_rejects_layer_name_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer names differ"):
            inject_rank_shard(
                {"different": torch.empty(12, 2, 4, 2, 5)},
                self.source,
                [1, 2, 3],
                block_axis=0,
            )


if __name__ == "__main__":
    unittest.main()
