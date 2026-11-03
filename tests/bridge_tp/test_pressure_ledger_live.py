# SPDX-License-Identifier: Apache-2.0
"""The ledger's live index answers exactly what a scan of every row would."""

import random
import unittest
from dataclasses import replace

from tests.bridge_tp.test_pressure_workload import reading
from tools.bridge_tp.pressure_cpu import load

p = load("pressure_policy")

POOLS = ("a", "b", "c")


def scan_pending(ledger, pool_id):
    rows = [
        r
        for r in ledger.entries.values()
        if r.pool_id == pool_id and r.state in {"ACTIVE", "COMMITTED"}
    ]
    return (
        sum(r.blocks - r.allocated_blocks for r in rows),
        sum(r.decode_ratio for r in rows if not r.service_observed),
        sum(r.prefill_tokens for r in rows if not r.prefill_observed),
    )


def scan_unobserved(ledger, pool_id):
    rows = [
        r
        for r in ledger.entries.values()
        if r.pool_id == pool_id
        and r.kind == "admission"
        and r.state in {"ACTIVE", "COMMITTED"}
        and not r.prefill_observed
    ]
    return len(rows), sum(r.prefill_tokens for r in rows)


def scan_channel(ledger, pool_id):
    return not any(
        r.pool_id == pool_id and r.kind == "migration" and r.state == "ACTIVE"
        for r in ledger.entries.values()
    )


class LedgerLiveIndexTests(unittest.TestCase):
    def test_random_lifecycle_matches_a_full_scan(self):
        rng = random.Random(20261103)
        ledger = p.ReservationLedger()
        keys = []
        for step in range(3000):
            action = rng.random()
            if action < 0.45 or not keys:
                key = f"k{step}"
                pool = rng.choice(POOLS)
                if rng.random() < 0.9:
                    ledger.track_ingress(key, f"r{step}", pool, rng.randrange(1, 900))
                else:
                    ledger.entries[key] = p.Reservation(
                        key, f"r{step}", pool, 64, 0.1, 128, "migration"
                    )
                keys.append(key)
            else:
                row = ledger.entries[rng.choice(keys)]
                if row.state not in p.LIVE_STATES:
                    continue
                choice = rng.random()
                if row.kind == "admission":
                    if choice < 0.3:
                        ledger.observe(
                            row.key,
                            allocated_blocks=0,
                            service_observed=False,
                            prefill_observed=True,
                        )
                    else:
                        ledger.retire_admission(
                            row.key,
                            "COMPLETED" if choice < 0.8 else "MOVED",
                            evidence=True,
                        )
                elif row.state == "ACTIVE":
                    ledger.settle(
                        row.key,
                        "COMMITTED" if choice < 0.5 else "CANCELLED",
                        evidence=True,
                    )
                else:
                    ledger.complete(row.key, evidence=True)
            if step % 50 == 0:
                for pool in POOLS:
                    self.assertEqual(ledger.pending(pool), scan_pending(ledger, pool))
                    self.assertEqual(
                        ledger.unobserved_ingress(pool), scan_unobserved(ledger, pool)
                    )
                    self.assertEqual(
                        ledger.channel_available(pool), scan_channel(ledger, pool)
                    )
                self.assertEqual(
                    {r["key"] for r in ledger.live_snapshot()},
                    {
                        r.key
                        for r in ledger.entries.values()
                        if r.state in p.LIVE_STATES
                    },
                )

    def test_settled_rows_stay_for_audit_but_leave_the_live_index(self):
        ledger = p.ReservationLedger()
        for i in range(2000):
            ledger.track_ingress(f"k{i}", f"r{i}", "a", 32)
            ledger.retire_admission(f"k{i}", "COMPLETED", evidence=True)
        ledger.track_ingress("open", "r-open", "a", 32)
        self.assertEqual(len(ledger.entries), 2001)
        self.assertEqual(len(ledger.snapshot()), 2001)
        self.assertEqual([r.key for r in ledger.live_rows()], ["open"])
        self.assertEqual(len(ledger.entries._live), 1)

    def test_identity_is_never_reused_even_after_settlement(self):
        ledger = p.ReservationLedger()
        ledger.track_ingress("k", "r", "a", 32)
        ledger.retire_admission("k", "COMPLETED", evidence=True)
        with self.assertRaises(ValueError):
            ledger.track_ingress("other", "r", "a", 32)
        with self.assertRaises(ValueError):
            ledger.track_ingress("k", "other", "a", 32)

    def test_added_since_reports_only_new_rows_in_order(self):
        ledger = p.ReservationLedger()
        ledger.track_ingress("k0", "r0", "a", 32)
        mark = ledger.mark()
        ledger.track_ingress("k1", "r1", "a", 32)
        ledger.track_ingress("k2", "r2", "b", 32)
        self.assertEqual([r["key"] for r in ledger.added_since(mark)], ["k1", "k2"])

    def test_routing_reading_counts_only_unobserved_live_ingress(self):
        ledger = p.ReservationLedger()
        ledger.track_ingress("done", "r0", "tp1", 512)
        ledger.retire_admission("done", "COMPLETED", evidence=True)
        ledger.track_ingress("open", "r1", "tp1", 512)
        pool = replace(reading("tp1", now=10), waiting=0, unallocated_prompt_blocks=0)
        value = ledger.routing_reading(pool)
        self.assertEqual(value.waiting, 1)
        self.assertEqual(value.unallocated_prompt_blocks, 32)


if __name__ == "__main__":
    unittest.main()
