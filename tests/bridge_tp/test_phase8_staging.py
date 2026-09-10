# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep

import torch

from tools.bridge_tp.phase8_stager import (
    _assemble_rank,
    _compact_history_block_layers,
    _wait_for_final_watermark,
)
from tools.bridge_tp.run_phase9_cap0_calibration import write_json


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

    def test_history_block_has_compact_independent_storage(self) -> None:
        snapshot = torch.arange(4 * 2 * 4 * 1 * 2, dtype=torch.float32).reshape(
            4, 2, 4, 1, 2
        )
        block = _compact_history_block_layers(
            {"layer.0": snapshot}, block_axis=0, logical_block=2
        )["layer.0"]

        self.assertEqual(tuple(block.shape), (1, 2, 4, 1, 2))
        self.assertEqual(
            block.untyped_storage().nbytes(),
            block.numel() * block.element_size(),
        )
        expected = snapshot[2:3].clone()
        snapshot[2].zero_()
        self.assertTrue(torch.equal(block, expected))

    def test_final_watermark_waits_past_streaming_file(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            watermark = root / "watermark.json"
            cleanup = root / "cleanup.json"
            write_json(
                watermark,
                {"status": "STREAMING", "end_token": 164},
            )

            def finish() -> None:
                sleep(0.05)
                write_json(
                    watermark,
                    {
                        "status": "TARGET_READY",
                        "end_token": 196,
                        "target_request_id": "target",
                    },
                )

            thread = Thread(target=finish)
            thread.start()
            result = _wait_for_final_watermark(
                watermark, cleanup, monotonic() + 1, 196
            )
            thread.join()
            self.assertIsNotNone(result)
            self.assertEqual(result["status"], "TARGET_READY")

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
