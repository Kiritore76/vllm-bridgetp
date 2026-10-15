# SPDX-License-Identifier: Apache-2.0
"""CPU evidence contracts for actual allocation and online stream observations."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tools.bridge_tp.pressure_cpu import load
from tools.bridge_tp.summarize_pressure_evidence import (
    cell_key,
    lookup_cells,
    rolling_pressure,
)

o = load("pressure_observation")
p = load("pressure_policy")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "pressure_inventory_test", ROOT / "vllm/bridge_tp/pressure_inventory.py"
)
inv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inv)


def scheduler(groups=1):
    shared, unique, null = (
        NS(block_id=1, ref_cnt=2, is_null=False),
        NS(block_id=2, ref_cnt=1, is_null=False),
        NS(block_id=0, ref_cnt=0, is_null=True),
    )
    request = NS(
        request_id="a",
        num_output_tokens=20,
        num_prompt_tokens=2048,
        num_computed_tokens=2068,
        max_tokens=6144,
        sampling_params=NS(ignore_eos=False),
    )
    kv = NS(
        block_pool=NS(get_num_free_blocks=lambda: 100, num_gpu_blocks=200),
        get_blocks=lambda rid: NS(
            blocks=tuple([shared, unique, null] for _ in range(groups))
        ),
    )
    return NS(
        kv_cache_manager=kv,
        running=[request],
        requests={"a": request},
        waiting=[],
        skipped_waiting=[],
        block_size=16,
        max_model_len=8192,
    )


class InventoryTests(unittest.TestCase):
    def test_real_refcounts_not_context_approximation(self):
        row = inv.inventory_snapshot(scheduler(), "tp1", 10)
        request = row["requests"][0]
        self.assertEqual(request["owned_blocks"], 2)
        self.assertEqual(request["releasable_blocks"], 1)
        self.assertEqual(request["max_output_tokens"], 6144)
        self.assertEqual(row["free_blocks"], 100)
        self.assertFalse(request["ignore_eos"])
        self.assertNotEqual(
            request["owned_blocks"] * 16,
            request["prompt_tokens"] + request["generated"],
        )

    def test_unsupported_groups_keep_release_unknown(self):
        row = inv.inventory_snapshot(scheduler(2), "tp1", 10)
        self.assertIsNone(row["requests"][0]["releasable_blocks"])

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(inv.InventoryPublisher.from_env())

    def test_request_end_is_published_even_inside_sampling_interval(self):
        publisher = object.__new__(inv.InventoryPublisher)
        saved = []
        publisher.pool_id, publisher.interval_s = "tp1", 0.2
        publisher.last_sample, publisher.last_membership = -float("inf"), None
        publisher.generation = 0
        publisher.journal = NS(dropped=0, error=None, submit=saved.append)
        current = scheduler()
        with patch.object(inv.time, "time", side_effect=[10, 10.05, 10.1]):
            publisher.sample(current)
            current.requests, current.running = {}, []
            publisher.sample(current)
            publisher.sample(current)
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[-1]["requests"], [])

    def test_only_complete_published_records_are_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inventory.jsonl"
            row = inv.inventory_snapshot(scheduler(), "tp1", 10)
            row.update(generation=1, dropped_before=0)
            encoded = json.dumps(row).encode()
            path.write_bytes(encoded)
            reader = o.InventoryReader(path, "tp1")
            self.assertIsNone(reader.poll(10))
            with path.open("ab") as stream:
                stream.write(b"\n")
            self.assertEqual(reader.poll(10)["free_blocks"], 100)
            self.assertEqual(reader.observed_at, 10)
            row.update(generation=2, sampled_at=11)
            with path.open("ab") as stream:
                stream.write(json.dumps(row).encode() + b"\n")
            self.assertEqual(reader.poll(10.1)["generation"], 1)
            self.assertIsNone(reader.error)
            self.assertEqual(reader.deferred["generation"], 2)
            self.assertEqual(reader.poll(11)["generation"], 2)
            self.assertIsNone(reader.deferred)

    def test_concurrent_append_after_poll_start_recovers_on_next_poll(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inventory.jsonl"
            row = inv.inventory_snapshot(scheduler(), "tp1", 10.01)
            row.update(generation=1, dropped_before=0)
            path.write_text(json.dumps(row) + "\n")
            reader = o.InventoryReader(path, "tp1")
            self.assertIsNone(reader.poll(10))
            self.assertEqual(reader.offset, 0)
            self.assertIsNone(reader.error)
            row.update(generation=2, sampled_at=10.02)
            with path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            self.assertEqual(reader.poll(10.03)["generation"], 2)
            self.assertIsNone(reader.error)

    def test_corruption_remains_fatal_and_is_not_cleared_by_later_valid_rows(self):
        for change in ({"pool_id": "tp4"}, {"dropped_before": 1}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "inventory.jsonl"
                row = inv.inventory_snapshot(scheduler(), "tp1", 10)
                row.update(generation=1, dropped_before=0)
                path.write_text(json.dumps(row | change) + "\n")
                reader = o.InventoryReader(path, "tp1")
                self.assertIsNone(reader.poll(10))
                row.update(generation=2, sampled_at=11)
                with path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                self.assertIsNone(reader.poll(11))
                self.assertIsNotNone(reader.error)
                self.assertEqual(reader.diagnostics()["error_detail"]["generation"], 1)


class NativeTests(unittest.TestCase):
    def test_live_samples_do_not_require_completed_request(self):
        window = o.NativeServiceWindow()
        window.arrive("a", "tp1", 0)
        for i in range(31):
            window.token("a", i * 0.02)
        row = window.snapshot("tp1", 0.6)
        self.assertEqual(row["samples"], 30)
        self.assertAlmostEqual(row["p95_s"], 0.02)
        self.assertAlmostEqual(row["ratio"], 0.2)
        window.finish("a")
        self.assertEqual(window.snapshot("tp1", 4)["samples"], 0)

    def test_slow_fraction_and_open_gap_are_explicit(self):
        window = o.NativeServiceWindow()
        window.arrive("a", "tp1", 0)
        window.token("a", 0)
        window.token("a", 0.2)
        row = window.snapshot("tp1", 1)
        self.assertEqual(row["slow_fraction"], 1)
        self.assertEqual(row["ratio"], 100)
        self.assertAlmostEqual(row["max_active_gap_s"], 0.8)
        self.assertEqual(window.snapshot("tp4", 1)["ratio"], None)


class LookupTests(unittest.TestCase):
    def test_future_token_intervals_cannot_change_earlier_pressure_prefix(self):
        rows = [(n / 100, 0.01) for n in range(25)]
        before = rolling_pressure(rows, 0, 2)
        after = rolling_pressure(rows + [(100, 99)], 0, 2)
        self.assertEqual(before, after)

    def test_lookup_separates_context_and_prefill_and_marks_single_run(self):
        row = dict(
            pool="source",
            prefill_tokens=0,
            recompute_tokens=0,
            decode_requests=2,
            decode_context_sum=4096,
            run="a",
            gpu_ms=25,
            engine_ms=28,
        )
        other = row | {"decode_context_sum": 8192}
        self.assertNotEqual(cell_key(row), cell_key(other))
        result = lookup_cells([row, row, other])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["support"], "descriptive_only")
        self.assertEqual(result[0]["gpu_ms"]["p50"], 25)


if __name__ == "__main__":
    unittest.main()
