# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tools.bridge_tp.phase8_stager import _assemble_rank


class TestPhase8Staging(unittest.TestCase):
    def _initial(self) -> dict:
        tensor = torch.zeros((1, 2, 4, 1, 2), dtype=torch.float32)
        tensor[0, :, 0:3, :, :] = 1
        return {"layers": {"layer.0": tensor}}

    def test_contiguous_deltas_expand_and_fill_blocks(self) -> None:
        deltas = {
            3: {
                "end_token": 4,
                "layers": {
                    "layer.0": torch.full((1, 2, 1, 2), 3.0)
                },
            },
            4: {
                "end_token": 6,
                "layers": {
                    "layer.0": torch.full((2, 2, 1, 2), 4.0)
                },
            },
        }
        layers, coverage = _assemble_rank(
            initial=self._initial(),
            deltas=deltas,
            initial_end=3,
            final_end=6,
            block_axis=0,
            block_size=4,
        )
        result = layers["layer.0"]
        self.assertEqual(tuple(result.shape), (2, 2, 4, 1, 2))
        self.assertTrue(torch.all(result[0, :, 3, :, :] == 3))
        self.assertTrue(torch.all(result[1, :, 0:2, :, :] == 4))
        self.assertEqual(coverage, [[3, 4], [4, 6]])

    def test_gap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "coverage gap/overlap"):
            _assemble_rank(
                initial=self._initial(),
                deltas={
                    4: {
                        "end_token": 5,
                        "layers": {
                            "layer.0": torch.zeros((1, 2, 1, 2))
                        },
                    }
                },
                initial_end=3,
                final_end=5,
                block_axis=0,
                block_size=4,
            )


if __name__ == "__main__":
    unittest.main()
