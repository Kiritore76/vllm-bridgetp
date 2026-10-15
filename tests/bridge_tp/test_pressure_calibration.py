# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for output-length collection and durable partial evidence."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp import run_pressure_calibration as c


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = c.pilot.read(
            c.ROOT / "experiments/phase9/pressure_balanced_v2/calibration.json"
        )

    def test_inputs_and_context_caps(self):
        c.validate_config(self.config)
        for size in (512, 1024, 2048, 4096, 6144):
            request = c.payload(self.config, size, 16, "id")
            self.assertEqual(len(request["prompt"]), size)
            self.assertTrue(request["ignore_eos"])
            self.assertEqual(request["seed"], 20261015)
        with self.assertRaises(ValueError):
            c.payload(self.config, 6144, 6144, "too-long")

    def test_capacity_counts_round_up_per_request_and_include_injection(self):
        scene = dict(
            prompt_tokens=17,
            max_tokens=16,
            source_count=3,
            target_count=3,
            injection_prompt_tokens=[17],
        )
        inventory = dict(block_size=16, free_blocks=12)
        receipt = c.capacity_check(scene, [inventory, inventory])
        self.assertEqual(receipt["source"]["required_blocks"], 12)
        with self.assertRaises(ValueError):
            c.capacity_check(scene, [dict(inventory, free_blocks=11), inventory])

    def test_inventory_partial_append_and_sticky_error_evidence(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT) as directory:
            path = Path(directory) / "inventory.jsonl"
            row = dict(schema="PRESSURE_INVENTORY_V1", generation=1)
            path.write_bytes(json.dumps(row).encode() + b'\n{"generation":')
            self.assertEqual(c.latest_inventory(path), row)
            row["dropped_before"] = 1
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(RuntimeError):
                c.latest_inventory(path)

    def test_idle_inventory_needs_generation_advance_not_heartbeat(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT) as directory:
            root = Path(directory)
            for pool in ("source", "target"):
                (root / f"{pool}_inventory.jsonl").write_text(
                    json.dumps(
                        dict(
                            schema="PRESSURE_INVENTORY_V1",
                            generation=7,
                            sampled_at=time.time() - 30,
                            requests=[],
                            running=0,
                            waiting=0,
                        )
                    )
                    + "\n"
                )
            self.assertEqual(len(c.wait_empty(root, after_generations=[6, 6])), 2)
            with self.assertRaises(TimeoutError):
                c.wait_empty(root, timeout_s=0.01, after_generations=[7, 7])

    def fake_stream(self, _url, request, _timeout, sink, *_args, **_kwargs):
        for index in range(request["max_tokens"]):
            sink(index, 42, time.time())
        return dict(
            token_ids=[42] * request["max_tokens"],
            finish_reason="length",
            response_id=request["request_id"],
            chunks=[],
        )

    def test_complete_output_and_durable_token_indices(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT) as directory:
            root = Path(directory)
            work = c.Workload(self.config, root, {"source": "unused"}, 1)
            try:
                with patch.object(c.io, "post_streaming_completion", self.fake_stream):
                    _, future = work.submit("source", 2048, 128, "short")
                    future.result(timeout=2)
                result = c.pilot.read(root / "short.result.json")
                self.assertTrue(result["done"])
                self.assertEqual(result["tokens"], 128)
                self.assertIsNone(result["error"])
                rows = [
                    json.loads(line)
                    for line in (root / "short.tokens.jsonl").read_text().splitlines()
                ]
                self.assertEqual([r["index"] for r in rows], list(range(128)))
            finally:
                work.executor.shutdown(wait=True)

    def test_short_or_failed_stream_is_not_success(self):
        def fail(_url, _request, _timeout, sink, *_args, **_kwargs):
            sink(0, 42, time.time())
            raise RuntimeError("server stopped")

        with tempfile.TemporaryDirectory(dir=c.ROOT) as directory:
            root = Path(directory)
            work = c.Workload(self.config, root, {"source": "unused"}, 1)
            try:
                with patch.object(c.io, "post_streaming_completion", fail):
                    _, future = work.submit("source", 2048, 128, "failed")
                    with self.assertRaises(RuntimeError):
                        future.result(timeout=2)
                result = c.pilot.read(root / "failed.result.json")
                self.assertIn("server stopped", result["error"])
                self.assertEqual(result["tokens"], 1)
                self.assertTrue((root / "failed.tokens.jsonl").exists())
            finally:
                work.executor.shutdown(wait=True)

    def test_early_eos_cannot_pass_forced_length_collection(self):
        def early(*_args, **_kwargs):
            return dict(token_ids=[42], finish_reason="stop", chunks=[])

        with tempfile.TemporaryDirectory(dir=c.ROOT) as directory:
            root = Path(directory)
            work = c.Workload(self.config, root, {"source": "unused"}, 1)
            try:
                with patch.object(c.io, "post_streaming_completion", early):
                    _, future = work.submit("source", 2048, 128, "early")
                    with self.assertRaisesRegex(RuntimeError, "stop cap"):
                        future.result(timeout=2)
                self.assertIsNotNone(c.pilot.read(root / "early.result.json")["error"])
            finally:
                work.executor.shutdown(wait=True)

    def test_ended_background_cannot_be_reported_as_mixed_coverage(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT) as directory:
            root = Path(directory)
            work = c.Workload(
                self.config, root, {"source": "unused", "target": "unused"}, 2
            )
            scene = dict(self.config["scenes"][-1])
            scene.update(source_count=1, target_count=1, recovery_s=0)
            try:
                with patch.object(c.io, "post_streaming_completion", self.fake_stream):
                    c.drive_guarded(work, scene, 1, root)
                result = c.pilot.read(root / "request_results.json")
                self.assertEqual(result["status"], "TECHNICAL_FAILURE")
                self.assertIn("background ended", result["error"])
            finally:
                work.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
