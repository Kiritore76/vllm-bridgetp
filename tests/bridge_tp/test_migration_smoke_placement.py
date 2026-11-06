# SPDX-License-Identifier: Apache-2.0
"""CPU placement contracts for the isolated migration smoke."""

import contextlib
import io
import json
import unittest

from tools.bridge_tp.run_tp2_migration_smoke import main, smoke_gpus


class PlacementTests(unittest.TestCase):
    def test_existing_defaults(self):
        self.assertEqual(smoke_gpus(4, None), ("2", "0,1"))
        self.assertEqual(smoke_gpus(4, 1), ("2", "3"))
        self.assertEqual(smoke_gpus(4, 2), ("2,3", "0,1"))

    def test_six_card_new_group_source(self):
        self.assertEqual(smoke_gpus(6, None, "4", "0,1"), ("4", "0,1"))

    def test_invalid_placement_rejected(self):
        for source, target in (("4", "0,1"), ("2", "2,3"), ("2", "0,0"),
                               ("2,3", "0,1"), ("-1", "0,1")):
            with (self.subTest(source=source, target=target),
                  self.assertRaises(ValueError)):
                smoke_gpus(4, None, source, target)

    def test_validate_only_resolves_actual_devices_without_gpu(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main([
                "--model-path", "unused", "--out-root", "unused",
                "--platform", "qwen3", "--cards", "6",
                "--source-gpus", "4", "--target-gpus", "0,1",
                "--expanded-context", "--validate-only",
            ])
        self.assertEqual(code, 0)
        row = json.loads(output.getvalue())
        self.assertEqual((row["source_gpu"], row["target_gpus"]), ("4", "0,1"))
        self.assertTrue(row["expanded_context"])

    def test_a100_override_and_tp1_expansion_rejected(self):
        base = ["--model-path", "unused", "--out-root", "unused", "--validate-only"]
        for extra in (["--cards", "6"],
                      ["--platform", "qwen3", "--same-tp", "1",
                       "--expanded-context"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                main(base + extra)
