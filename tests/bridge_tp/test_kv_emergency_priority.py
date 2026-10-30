# SPDX-License-Identifier: Apache-2.0
"""Urgent KV ignores performance vetoes but retains executable-transfer checks."""

import dataclasses
import hashlib
import json
import unittest

from tests.bridge_tp.test_pressure_policy import candidate, config, length, pool
from tools.bridge_tp.pressure_cpu import load

p = load("pressure_policy")


def decide(source=None, target=None, candidates=None, cfg=None, ledger=None):
    return p.PressurePolicy(
        cfg or config(kv_emergency_priority=True), ledger or p.ReservationLedger()
    ).decide(
        10.0,
        source or pool(free_blocks=100),
        target or pool("tp4"),
        candidates if candidates is not None else [candidate()],
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
        expected = hashlib.sha256(
            json.dumps(old_values, sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(config().sha256, expected)
        self.assertNotEqual(config().sha256, config(kv_emergency_priority=True).sha256)
        with self.assertRaises(ValueError):
            config(kv_emergency_priority=1)
