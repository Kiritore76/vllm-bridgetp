# SPDX-License-Identifier: Apache-2.0
"""Exercise idle bootstrap through real allocator usage and metrics parsing."""

import ast
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp.test_pressure_observation import inv, scheduler
from tools.bridge_tp.pressure_cpu import load
from tools.bridge_tp.pressure_progress import ProgressWatch, save_progress

o, p, telemetry = (
    load(name) for name in ("pressure_observation", "pressure_policy", "telemetry")
)
ROOT = Path(__file__).resolve().parents[2]
CFG = p.PressureConfig(
    **json.loads(
        (
            ROOT / "experiments/phase9/controller/pressure_balanced_v1_draft.json"
        ).read_text()
    )
)


def idle_inventory(total=1963):
    current = scheduler()
    current.requests, current.running = {}, []
    current.kv_cache_manager.block_pool = NS(
        num_gpu_blocks=total, get_num_free_blocks=lambda: total - 1
    )
    return inv.inventory_snapshot(current, "tp1", 10)


def real_heartbeat(raw, now=20):
    tree = ast.parse((ROOT / "vllm/v1/core/block_pool.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BlockPool"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_usage"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "allocator_usage", "exec"),
        namespace,
    )
    usage = namespace["get_usage"](
        NS(
            num_gpu_blocks=raw["total_blocks"],
            get_num_free_blocks=lambda: raw["free_blocks"],
        )
    )
    text = (
        f"vllm:kv_cache_usage_perc {usage}\n"
        "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
    ).encode()
    scraper = telemetry.MetricsScraper("http://unused", 16, raw["total_blocks"])
    with (
        patch.object(
            telemetry.urllib.request, "urlopen", return_value=io.BytesIO(text)
        ),
        patch.object(telemetry.time, "time", return_value=now),
    ):
        return o.scrape_inventory_metrics(scraper, raw)


class BootstrapTests(unittest.TestCase):
    def test_real_inventory_adapter_keeps_step_and_backlog_separate(self):
        current = scheduler()
        current.requests["a"].num_computed_tokens = 1024
        raw = inv.inventory_snapshot(current, "tp1", 20)
        raw["last_scheduled_batch"] = dict(prefill_tokens=2048, sampled_at=20)
        window = o.NativeServiceWindow()
        reading = o.pool_reading(raw, 20, window.snapshot("tp1", 20), CFG, 20)
        self.assertEqual(reading.pending_prompt_tokens, 1024)
        self.assertEqual(reading.prefill_batch_tokens, 2048)
        self.assertEqual(p.classify(reading, CFG, 20).prefill, 1)
        raw["sampled_at"] = 23
        reading = o.pool_reading(raw, 23, window.snapshot("tp1", 23), CFG, 23)
        self.assertIsNone(reading.prefill_batch_tokens)
        self.assertIn("prefill step", p.classify(reading, CFG, 23).missing)

    def test_actual_allocator_parser_idle_heartbeat_admission(self):
        for total in (1963, 35739):
            with self.subTest(total=total):
                raw = idle_inventory(total)
                heartbeat = real_heartbeat(raw)
                self.assertEqual(heartbeat["free_kv_blocks"], total - 1)
                window = o.NativeServiceWindow()
                reading = o.pool_reading(
                    raw, 10, window.snapshot("tp1", 20), CFG, 20, heartbeat=heartbeat
                )
                self.assertTrue(reading.idle)
                self.assertEqual(p.classify(reading, CFG, 20).zone, "SPARE")
                ledger = p.ReservationLedger()
                admission = p.admit(
                    "a",
                    1024,
                    1024,
                    (reading, replace(reading, pool_id="tp4")),
                    CFG,
                    ledger,
                    20,
                    "admit-a",
                    {"tp1": 0.25, "tp4": 0.2},
                )
                self.assertEqual(admission.pool_id, "tp1")
                self.assertEqual(ledger.entries["admit-a"].blocks, 64)
                # A real request update replaces idle evidence, without trusting
                # the older zero-running /metrics sample.
                active = inv.inventory_snapshot(scheduler(), "tp1", 20)
                window.arrive("a", "tp1", 20)
                active_reading = o.pool_reading(
                    active, 20, window.snapshot("tp1", 20), CFG, 20, heartbeat=heartbeat
                )
                self.assertFalse(active_reading.idle)
                self.assertEqual(active_reading.free_blocks, 100)

    def test_unmatched_stale_future_or_active_heartbeat_cannot_refresh(self):
        raw = idle_inventory()
        window = o.NativeServiceWindow()
        base = real_heartbeat(raw)
        for change in (
            {"free_kv_blocks": raw["free_blocks"] + 1},
            {"num_running": 1},
            {"num_waiting": 1},
            {"sampled_unix_s": 17},
            {"sampled_unix_s": 21},
        ):
            reading = o.pool_reading(
                raw, 10, window.snapshot("tp1", 20), CFG, 20, heartbeat=base | change
            )
            self.assertEqual(p.classify(reading, CFG, 20).zone, "UNKNOWN")
        window.arrive("new-arrival", "tp1", 19)
        reading = o.pool_reading(
            raw, 10, window.snapshot("tp1", 20), CFG, 20, heartbeat=base
        )
        self.assertFalse(reading.idle)
        self.assertEqual(p.classify(reading, CFG, 20).zone, "UNKNOWN")

    def test_old_schema_is_not_silently_accepted_as_same_denominator(self):
        raw = idle_inventory()
        raw.pop("usage_denominator_blocks")
        with self.assertRaisesRegex(ValueError, "denominator"):
            o.scrape_inventory_metrics(NS(scrape=lambda: self.fail()), raw)

    def test_metrics_outage_does_not_manufacture_idle_heartbeat(self):
        raw = idle_inventory()
        scraper = telemetry.MetricsScraper("http://unused", 16, 1963)
        with (
            patch.object(
                telemetry.urllib.request, "urlopen", side_effect=TimeoutError("offline")
            ),
            self.assertRaises(telemetry.TelemetryError),
        ):
            o.scrape_inventory_metrics(scraper, raw)
        window = o.NativeServiceWindow()
        reading = o.pool_reading(
            raw, 10, window.snapshot("tp1", 20), CFG, 20, heartbeat=None
        )
        self.assertEqual(p.classify(reading, CFG, 20).zone, "UNKNOWN")
        self.assertEqual(real_heartbeat(raw)["free_kv_blocks"], raw["free_blocks"])


class ProgressTests(unittest.TestCase):
    def test_due_work_without_progress_has_deadline(self):
        watch = ProgressWatch(60)
        watch.update(0, (0, 0), work_due=True, reason="stale")
        watch.update(59, (0, 0), work_due=True, reason="stale")
        with self.assertRaisesRegex(TimeoutError, "stale"):
            watch.update(60, (0, 0), work_due=True, reason="stale")

    def test_future_arrivals_and_real_token_progress_do_not_timeout(self):
        watch = ProgressWatch(60)
        watch.update(0, 0, work_due=False, reason="future arrival")
        watch.update(80, 0, work_due=False, reason="future arrival")
        watch.update(100, 1, work_due=True, reason="token arrived")
        watch.update(159, 1, work_due=True, reason="decoding")
        watch.update(160, 2, work_due=True, reason="token arrived")

    def test_progress_snapshot_replaces_prior_valid_json(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "progress.json"
            save_progress(path, stage="queued", submitted=0)
            save_progress(path, stage="running", submitted=1)
            self.assertEqual(json.loads(path.read_text())["submitted"], 1)
            self.assertFalse(path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
