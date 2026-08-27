# SPDX-License-Identifier: Apache-2.0
"""Phase 9: telemetry parsing and the runtime control block.

python -m unittest tests.bridge_tp.test_phase9_telemetry_control
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from vllm.bridge_tp.controller import telemetry as tel
from vllm.bridge_tp.runtime_control import (
    ControlCache,
    RuntimeControl,
    effective_config,
)

# A realistic scrape, including the ``_created`` companion series that a naive
# reader would mistake for a preemption count.
SCRAPE = """
# HELP vllm:num_requests_running Number of running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="qwen"} 7.0
vllm:num_requests_waiting{model_name="qwen"} 2.0
vllm:gpu_cache_usage_perc{model_name="qwen"} 0.63
vllm:num_preemptions_total{model_name="qwen"} 195.0
vllm:num_preemptions_created{model_name="qwen"} 1787422490.0
vllm:time_per_output_token_seconds_bucket{le="0.01"} 10.0
vllm:time_per_output_token_seconds_bucket{le="0.02"} 50.0
vllm:time_per_output_token_seconds_bucket{le="0.04"} 90.0
vllm:time_per_output_token_seconds_bucket{le="0.08"} 99.0
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 100.0
vllm:time_per_output_token_seconds_sum 2.4
vllm:time_per_output_token_seconds_count 100.0
"""


class TestPrometheusParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = tel.parse_prometheus(SCRAPE)

    def test_created_series_are_never_returned(self):
        """``*_created`` holds a counter creation timestamp, not a count.

        Reading it as preemptions yields ~1.7e9 and pins the risk signal at
        its ceiling forever, which is silent rather than loud. The parser
        drops those series so no caller can make that mistake.
        """
        self.assertTrue(all(not s.name.endswith("_created") for s in self.samples))
        self.assertEqual(
            tel.first_value(self.samples, "vllm:num_preemptions_created", -1.0), -1.0
        )
        self.assertEqual(
            tel.first_value(self.samples, "vllm:num_preemptions_total"), 195.0
        )

    def test_labels_are_parsed(self):
        running = next(s for s in self.samples if s.name == "vllm:num_requests_running")
        self.assertEqual(running.labels["model_name"], "qwen")
        self.assertEqual(running.value, 7.0)

    def test_comments_and_blank_lines_are_ignored(self):
        self.assertTrue(all(not s.name.startswith("#") for s in self.samples))

    def test_histogram_quantile_interpolates(self):
        p99 = tel.histogram_quantile(
            self.samples, "vllm:time_per_output_token_seconds", 0.99
        )
        p50 = tel.histogram_quantile(
            self.samples, "vllm:time_per_output_token_seconds", 0.50
        )
        self.assertTrue(0.04 < p99 <= 0.08)
        self.assertTrue(0.01 < p50 <= 0.02)
        self.assertLess(p50, p99)

    def test_histogram_quantile_on_missing_metric_is_zero(self):
        self.assertEqual(tel.histogram_quantile([], "nope", 0.99), 0.0)

    def test_quantile_rejects_out_of_range_q(self):
        with self.assertRaises(ValueError):
            tel.histogram_quantile(self.samples, "x", 1.0)

    def test_pool_from_samples_derives_free_blocks_and_tail(self):
        p = tel.pool_from_samples(self.samples, block_size=16, total_kv_blocks=1000)
        self.assertEqual(p.num_running, 7)
        self.assertEqual(p.num_waiting, 2)
        self.assertAlmostEqual(p.kv_usage_frac, 0.63, places=9)
        self.assertEqual(p.free_kv_blocks, 370)
        self.assertEqual(p.free_kv_tokens, 370 * 16)
        self.assertEqual(p.preemptions_total, 195)
        self.assertAlmostEqual(p.mean_tpot_s, 0.024, places=9)
        self.assertGreater(p.p99_tpot_s, p.mean_tpot_s)

    def test_current_request_tpot_metric_is_preferred(self):
        current = tel.parse_prometheus(
            "vllm:request_time_per_output_token_seconds_bucket{le=\"0.04\"} 8\n"
            "vllm:request_time_per_output_token_seconds_bucket{le=\"+Inf\"} 10\n"
            "vllm:request_time_per_output_token_seconds_sum 0.3\n"
            "vllm:request_time_per_output_token_seconds_count 10\n"
            "vllm:time_per_output_token_seconds_bucket{le=\"+Inf\"} 99\n"
        )
        self.assertEqual(
            tel.request_tpot_metric(current),
            "vllm:request_time_per_output_token_seconds",
        )
        pool = tel.pool_from_samples(current, block_size=16, total_kv_blocks=100)
        self.assertAlmostEqual(pool.mean_tpot_s, 0.03, places=9)

    def test_interval_pool_accepts_lazily_created_current_metric(self):
        current = tel.parse_prometheus(
            "vllm:num_requests_running 2\n"
            "vllm:num_requests_waiting 0\n"
            "vllm:kv_cache_usage_perc 0.1\n"
            "vllm:request_time_per_output_token_seconds_bucket{le=\"0.04\"} 3\n"
            "vllm:request_time_per_output_token_seconds_bucket{le=\"+Inf\"} 4\n"
            "vllm:request_time_per_output_token_seconds_sum 0.12\n"
            "vllm:request_time_per_output_token_seconds_count 4\n"
        )
        pool, count = tel.interval_pool_from_samples(
            [], current, block_size=16, total_kv_blocks=100
        )
        self.assertEqual(count, 4)
        self.assertAlmostEqual(pool.mean_tpot_s, 0.03, places=9)

    def test_percent_style_cache_usage_is_normalized(self):
        p = tel.pool_from_samples(
            tel.parse_prometheus("vllm:gpu_cache_usage_perc 63.0\n"),
            block_size=16,
            total_kv_blocks=100,
        )
        self.assertAlmostEqual(p.kv_usage_frac, 0.63, places=9)

    def test_new_kv_cache_metric_name_is_preferred(self):
        samples = tel.parse_prometheus(
            "vllm:gpu_cache_usage_perc 0.25\n"
            "vllm:kv_cache_usage_perc 0.75\n"
        )
        pool = tel.pool_from_samples(samples, block_size=16, total_kv_blocks=100)
        self.assertAlmostEqual(pool.kv_usage_frac, 0.75, places=9)

    def test_interval_histogram_uses_counter_deltas(self):
        previous = tel.parse_prometheus(
            """
vllm:time_per_output_token_seconds_bucket{le="0.01"} 10
vllm:time_per_output_token_seconds_bucket{le="0.02"} 20
vllm:time_per_output_token_seconds_bucket{le="0.04"} 30
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 30
vllm:time_per_output_token_seconds_sum 0.60
vllm:time_per_output_token_seconds_count 30
"""
        )
        current = tel.parse_prometheus(
            """
vllm:time_per_output_token_seconds_bucket{le="0.01"} 10
vllm:time_per_output_token_seconds_bucket{le="0.02"} 25
vllm:time_per_output_token_seconds_bucket{le="0.04"} 40
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 40
vllm:time_per_output_token_seconds_sum 0.90
vllm:time_per_output_token_seconds_count 40
"""
        )
        count, mean, p99 = tel.interval_histogram_stats(
            previous,
            current,
            "vllm:time_per_output_token_seconds",
        )
        self.assertEqual(count, 10)
        self.assertAlmostEqual(mean, 0.03, places=9)
        self.assertGreater(p99, 0.02)
        self.assertLessEqual(p99, 0.04)

    def test_interval_pool_retains_current_gauges(self):
        previous = tel.parse_prometheus(SCRAPE)
        current = tel.parse_prometheus(
            SCRAPE.replace(
                'vllm:num_requests_running{model_name="qwen"} 7.0',
                'vllm:num_requests_running{model_name="qwen"} 9.0',
            ).replace(
                'vllm:time_per_output_token_seconds_count 100.0',
                'vllm:time_per_output_token_seconds_count 110.0',
            ).replace(
                'vllm:time_per_output_token_seconds_sum 2.4',
                'vllm:time_per_output_token_seconds_sum 2.7',
            ).replace(
                'vllm:time_per_output_token_seconds_bucket{le="+Inf"} 100.0',
                'vllm:time_per_output_token_seconds_bucket{le="+Inf"} 110.0',
            )
        )
        pool, count = tel.interval_pool_from_samples(
            previous,
            current,
            block_size=16,
            total_kv_blocks=1000,
        )
        self.assertEqual(pool.num_running, 9)
        self.assertAlmostEqual(pool.kv_usage_frac, 0.63, places=9)
        self.assertEqual(count, 10)
        self.assertAlmostEqual(pool.mean_tpot_s, 0.03, places=9)


class TestRuntimeControl(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_roundtrip_and_generation_bump(self):
        written = RuntimeControl(armed=True, trigger_output_tokens=128).write(
            self.run_dir
        )
        self.assertEqual(written.generation, 1)
        back = RuntimeControl.load(self.run_dir)
        self.assertTrue(back.armed)
        self.assertEqual(back.trigger_output_tokens, 128)
        self.assertEqual(written.write(self.run_dir).generation, 2)

    def test_missing_control_block_loads_as_none(self):
        self.assertIsNone(RuntimeControl.load(self.run_dir))

    def test_corrupt_control_block_loads_as_none(self):
        RuntimeControl.path(self.run_dir).write_text("{not json", encoding="utf-8")
        self.assertIsNone(RuntimeControl.load(self.run_dir))

    def test_env_wins_when_no_control_block_exists(self):
        """Every existing Phase 6/7/8 run script must keep working unchanged."""
        self.assertEqual(effective_config(128, 160, 0.05, None), (128, 160, 0.05, True))

    def test_control_block_overlays_env_field_by_field(self):
        control = RuntimeControl(armed=True, trigger_output_tokens=64, rate_gib_s=0.9)
        trigger, cutover, rate, armed = effective_config(128, 160, 0.05, control)
        self.assertEqual(trigger, 64)
        self.assertEqual(cutover, 160)  # unset field falls through to env
        self.assertAlmostEqual(rate, 0.9, places=9)
        self.assertTrue(armed)

    def test_cutover_must_follow_trigger(self):
        control = RuntimeControl(trigger_output_tokens=200)
        with self.assertRaisesRegex(ValueError, "strictly after trigger"):
            effective_config(128, 160, 0.05, control)

    def test_negative_rate_is_rejected(self):
        control = RuntimeControl(rate_gib_s=-1.0)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            effective_config(128, 160, 0.05, control)

    def test_cache_rereads_only_after_a_write(self):
        """The hot decode path must not re-parse JSON every iteration."""
        RuntimeControl(armed=False, rate_gib_s=0.1).write(self.run_dir)
        cache = ControlCache(self.run_dir)
        first = cache.get()
        self.assertIs(cache.get(), first)
        time.sleep(0.01)
        RuntimeControl(armed=True, rate_gib_s=0.8).write(self.run_dir)
        updated = cache.get()
        self.assertIsNot(updated, first)
        self.assertAlmostEqual(updated.rate_gib_s, 0.8, places=9)

    def test_cache_on_missing_file(self):
        self.assertIsNone(ControlCache(self.run_dir).get())


if __name__ == "__main__":
    unittest.main()
