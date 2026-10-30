# SPDX-License-Identifier: Apache-2.0
"""Urgent KV ignores performance vetoes but retains executable-transfer checks."""

import dataclasses
import hashlib
import json
import unittest
from dataclasses import replace

from tests.bridge_tp.test_pressure_policy import candidate as base_candidate
from tests.bridge_tp.test_pressure_policy import config, length, pool
from tools.bridge_tp.pressure_cpu import load

p = load("pressure_policy")


def candidate(*args, **kwargs):
    return base_candidate(*args, **(dict(source_tokens_s=32) | kwargs))


def decide(source=None, target=None, candidates=None, cfg=None, ledger=None):
    return p.PressurePolicy(
        cfg or config(kv_emergency_priority=True), ledger or p.ReservationLedger()
    ).decide(
        10.0,
        source or pool(free_blocks=100),
        target or pool("tp4"),
        candidates if candidates is not None else [candidate(source_tokens_s=32)],
    )


class KVPriorityTests(unittest.TestCase):
    def test_busy_target_is_warning_not_emergency_veto(self):
        target = pool("tp4", decode_ratio=2.0, pending_prompt_tokens=14338, waiting=20)
        legacy = decide(target=target, cfg=config())
        self.assertEqual(legacy.action, "STAY")
        result = decide(target=target)
        self.assertEqual(result.action, "START_SHADOW")
        selected = next(r for r in result.candidates if r["selected"])
        self.assertEqual(selected["rejections"], [])
        self.assertIn(
            "emergency target service budget unavailable", selected["soft_warnings"]
        )

    def test_short_or_missing_prediction_still_rejected_for_urgent_kv(self):
        for c in (
            candidate(max_output_tokens=200, remaining=length(32, cap=200)),
            candidate(
                max_output_tokens=200, remaining=None, prediction_status="MISSING"
            ),
        ):
            self.assertEqual(decide(candidates=[c], cfg=config()).action, "STAY")
            result = decide(candidates=[c])
            self.assertEqual(result.action, "STAY")
            self.assertTrue(result.candidates[0]["rejections"])

    def test_large_kv_does_not_override_insufficient_remaining_work(self):
        short = candidate(
            "large-short", max_output_tokens=200,
            remaining=length(32, cap=200), owned_blocks=500,
            releasable_blocks=500, transfer_bytes=1,
        )
        durable = candidate("durable", transfer_bytes=2000)
        result = decide(candidates=[short, durable])
        self.assertEqual(result.request_id, "durable")
        self.assertIn("insufficient sustained work probability",
                      result.candidates[0]["rejections"])

    def test_long_stop_cap_alone_does_not_prove_remaining_generation(self):
        missing = candidate(remaining=None, prediction_status="MISSING")
        self.assertEqual(decide(candidates=[missing]).action, "STAY")
        short = candidate(remaining=length(200))
        self.assertEqual(decide(candidates=[short]).action, "STAY")

    def test_cost_adjusted_horizon_filters_work_that_would_end_in_flight(self):
        c = candidate(remaining=length(400))
        self.assertEqual(decide(candidates=[c]).action, "START_SHADOW")
        result = decide(candidates=[c], cfg=config(
            kv_emergency_priority=True, survival_horizon=480))
        self.assertEqual(result.action, "STAY")
        self.assertEqual(result.candidates[0]["remaining"]["mean"], 400)

    def test_durable_work_still_preferred_when_both_can_release(self):
        short = candidate(
            "short",
            max_output_tokens=200,
            remaining=length(32, cap=200),
            transfer_bytes=1,
        )
        durable = candidate("durable", transfer_bytes=2000)
        self.assertEqual(decide(candidates=[short, durable]).request_id, "durable")

    def test_physical_ownership_and_deadline_guards_not_removed(self):
        for c in (
            candidate(owner="tp4"),
            candidate(finished=True),
            candidate(migrating=True),
            candidate(releasable_blocks=None),
            candidate(releasable_blocks=20),
            candidate(transfer_bytes=None),
            candidate(target_decode_cost=None),
        ):
            self.assertEqual(decide(candidates=[c]).action, "STAY")
        self.assertEqual(decide(target=pool("tp4", free_blocks=100)).action, "STAY")
        self.assertEqual(decide(source=pool(free_blocks=1)).action, "STAY")
        self.assertEqual(decide(target=pool("tp4", sampled_at=7.0)).action, "STAY")
        ledger = p.ReservationLedger()
        ledger.reserve(
            p.Reservation("busy", "another", "tp4", 500, 0.1, 0, "migration"),
            pool("tp4"),
        )
        self.assertEqual(decide(ledger=ledger).reason, "migration channel occupied")

    def test_joint_score_can_choose_neither_largest_kv_nor_longest_request(self):
        # All survive the same horizon and satisfy physical relief. Products:
        # 400*(600/32-1.3)=6980; 80*(4000/32-1.3)=9896;
        # 200*(2000/32-1.3)=12240 block-seconds.
        choices = [
            candidate("largest", owned_blocks=400, releasable_blocks=400,
                      remaining=length(600)),
            candidate("longest", remaining=length(4000)),
            candidate("balanced", owned_blocks=200, releasable_blocks=200,
                      remaining=length(2000)),
        ]
        result = decide(candidates=choices)
        self.assertEqual(result.request_id, "balanced")
        self.assertAlmostEqual(result.candidates[2]["kv_joint_benefit"]
                               ["relief_block_seconds"], 12240)
        for choice in (choices, list(reversed(choices))):
            self.assertEqual(decide(candidates=choice).request_id, "balanced")

    def test_joint_score_counts_releasable_blocks_not_all_owned_blocks(self):
        c = candidate(owned_blocks=1000, releasable_blocks=80)
        result = decide(candidates=[c])
        score = result.candidates[0]["kv_joint_benefit"]
        self.assertEqual(score["releasable_blocks"], 80)
        self.assertAlmostEqual(score["relief_block_seconds"],
                               80 * (c.remaining.mean / 32 - 1.3))

    def test_joint_score_penalizes_transfer_delay_and_uses_request_rate(self):
        a = candidate("a", remaining=length(1000))
        b = candidate("b", remaining=length(1000), estimated_release_s=10)
        self.assertEqual(decide(candidates=[b, a]).request_id, "a")
        b = replace(b, estimated_release_s=1.3, source_tokens_s=16)
        self.assertEqual(decide(candidates=[a, b]).request_id, "b")
        self.assertEqual(decide(candidates=[replace(a, source_tokens_s=1000)])
                         .action, "STAY")

    def test_missing_or_invalid_rate_never_uses_output_cap_as_score(self):
        for rate in (None, 0, -1, float("inf"), float("nan")):
            result = decide(candidates=[candidate(source_tokens_s=rate)])
            self.assertEqual(result.action, "STAY")
            self.assertIsNone(result.candidates[0]["kv_joint_benefit"])

    def test_joint_score_preserves_existing_probability_weighted_mean(self):
        # A two-point PMF with mean 1000, median 600 and cap 5980.
        # The scorer uses the mean, not either of those other quantities.
        template = length(1000)
        pmf = [0.0] * (template.cap + 1)
        pmf[600] = pmf[1400] = 0.5
        remaining = replace(template, pmf=tuple(pmf), bins=(
            (0.5, 600, 600, True), (0.5, 1400, 1400, True),
        ))
        result = decide(candidates=[candidate(remaining=remaining)])
        score = result.candidates[0]["kv_joint_benefit"]
        self.assertEqual(score["remaining_tokens_mean"], remaining.mean)
        self.assertAlmostEqual(score["remaining_s"], remaining.mean / 32)
        self.assertAlmostEqual(score["relief_block_seconds"],
                               80 * (remaining.mean / 32 - 1.3))

    def test_ordinary_service_path_unchanged_and_legacy_hash_stable(self):
        source = pool(decode_ratio=1.5)
        target = pool("tp4", decode_ratio=2.0)
        old = decide(source=source, target=target, cfg=config())
        new = decide(source=source, target=target)
        self.assertFalse(new.capacity_path)
        self.assertEqual((old.action, old.reason), (new.action, new.reason))
        selections = []
        for enabled in (False, True):
            policy = p.PressurePolicy(
                config(kv_emergency_priority=enabled), p.ReservationLedger()
            )
            for index in range(7):
                result = policy.decide(
                    10.0 + index * 0.2, source, pool("tp4"), [candidate("long")]
                )
            selections.append(result)
        self.assertEqual(selections[0].action, "START_SHADOW")
        self.assertEqual(selections[0].request_id, selections[1].request_id)
        self.assertEqual(selections[0].reason, selections[1].reason)
        old_values = dataclasses.asdict(config())
        old_values.pop("kv_emergency_priority")
        old_values.pop("kv_routing_priority")
        expected = hashlib.sha256(
            json.dumps(old_values, sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(config().sha256, expected)
        self.assertNotEqual(config().sha256, config(kv_emergency_priority=True).sha256)
        with self.assertRaises(ValueError):
            config(kv_emergency_priority=1)
