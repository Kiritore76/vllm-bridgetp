# SPDX-License-Identifier: Apache-2.0
import math
import unittest
from types import SimpleNamespace as NS

from tools.bridge_tp.guard_forecast import (
    guard_time,
    snapshot_guard_time,
    unallocated_prefill_tokens,
)


class TestGuardForecast(unittest.TestCase):
    def test_completed_prefill_does_not_extrapolate_old_speed(self):
        self.assertEqual(snapshot_guard_time({
            "source_prefill_pending_kv_tokens": 0,
            "source_decode_growth_tokens_s": 100,
            "source_prefill_growth_tokens_s": 6500,
        }, 8704), 87.04)

    def test_finite_demand_and_prefill_only(self):
        self.assertEqual(guard_time(2000, 0, 10000, 5000), 0.2)
        self.assertEqual(guard_time(6000, 100, 10000, 5000), 10)
        self.assertEqual(guard_time(6000, 0, 10000, 5000), math.inf)
        self.assertIsNone(guard_time(2000, 0, None, 5000))
        self.assertIsNone(snapshot_guard_time({
            "source_prefill_pending_kv_tokens": 5000,
            "source_decode_growth_tokens_s": 100,
            "source_prefill_growth_tokens_s": 6500,
        }, 8704))

    def test_allocated_blocks_are_not_counted_again(self):
        requests = [NS(request_id="a", num_prompt_tokens=33,
                       num_computed_tokens=0)]
        manager = NS(get_blocks=lambda _: NS(blocks=([1, 2, 3],)))
        self.assertEqual(unallocated_prefill_tokens(requests, manager, 16, 100), 0)
        manager.get_blocks = lambda _: NS(blocks=([1],))
        self.assertEqual(unallocated_prefill_tokens(requests, manager, 16, 100), 32)
        manager.get_blocks = lambda _: NS(blocks=([1], [2]))
        self.assertIsNone(unallocated_prefill_tokens(requests, manager, 16, 100))
