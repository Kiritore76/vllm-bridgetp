# SPDX-License-Identifier: Apache-2.0
"""Exercise real ingress boundary with a deferred transport, not a GPU mock claim."""

import unittest
from concurrent.futures import Future
from dataclasses import replace

from tools.bridge_tp.pressure_cpu import load

p, r, o, ingress, b = (
    load(name)
    for name in (
        "pressure_policy",
        "pressure_runtime",
        "pressure_observation",
        "pressure_ingress",
        "bounded_length",
    )
)


class DeferredExecutor:
    def submit(self, function):
        self.function = function
        self.future = Future()
        return self.future


def reading(name, ratio=0.4):
    return p.PoolReading(name, 10, 10, 500, 16, 8192, ratio, 30, 10, False, 0, 0, 1)


class IngressTests(unittest.TestCase):
    def setUp(self):
        cfg = p.PressureConfig(
            prompt_boundary=2048,
            prefill_pending_budget=2048,
            prefill_waiting_budget=4,
            capacity_guard_blocks=125,
            sustain_s=0,
            survival_horizon=64,
        )
        self.rows = []
        self.co = r.PressureCoordinator(cfg, self.rows.append)
        self.window = o.NativeServiceWindow()
        self.executor = DeferredExecutor()
        self.http_calls = []

        def post(url, payload, timeout, sink):
            self.http_calls.append((url, payload.copy()))
            sink(0, 42, 10.01)
            return {"completed_unix_s": 11}

        self.ingress = ingress.PressureIngress(
            self.co,
            self.window,
            self.executor,
            "source",
            "target",
            post=post,
            clock=lambda: 10,
        )
        self.stream = self.ingress.submit(
            dict(model="qwen", prompt=[1] * 64, max_tokens=256),
            (reading("tp1"), reading("tp4")),
            {"tp1": 0.1, "tp4": 0.1},
        )
        self.server_id = f"cmpl-{self.stream.client_id}-0-abcdef"

    def snapshot(self, **kwargs):
        row = dict(
            request_id=self.server_id,
            owner="tp1",
            max_output_tokens=256,
            prompt_tokens=64,
            generated=20,
            ignore_eos=False,
            owned_blocks=10,
        )
        row.update(kwargs)
        return dict(pool_id="tp1", sampled_at=10, generation=1, requests=[row])

    def test_shared_reservation_precedes_http_and_freezes_cap(self):
        self.assertEqual(len(self.co.ledger.entries), 1)
        self.assertEqual(self.http_calls, [])
        self.executor.function()
        self.assertEqual(self.http_calls[0][0], "source")
        self.assertEqual(self.http_calls[0][1]["max_tokens"], 256)

    def test_server_binding_and_prefill_commitment_not_double_counted(self):
        self.ingress.reconcile(self.snapshot(), 10)
        entry = self.co.ledger.entries[self.stream.reservation_key]
        self.assertEqual(entry.request_id, self.server_id)
        self.assertEqual(entry.state, "COMMITTED")
        self.assertEqual(self.co.ledger.pending("tp1"), (10, 0, 0))
        self.assertIs(self.ingress.for_server_id(self.server_id), self.stream)
        with self.assertRaises(RuntimeError):
            self.ingress.refuse_restart_if_active()

    def test_admitted_active_request_can_migrate_without_relaunch(self):
        self.ingress.reconcile(self.snapshot(), 10)
        remaining = b.bounded_length(
            (0, 0, 1), (0, 8192), generated=20, predicted_at=20, max_output_tokens=256
        )
        candidate = p.Candidate(
            self.server_id,
            "tp1",
            10,
            10,
            20,
            64,
            256,
            10,
            10,
            1000,
            remaining,
            "AVAILABLE",
            0.2,
            0.2,
            1,
        )
        starts = []
        self.co.tick(
            10,
            reading("tp1", 1.5),
            reading("tp4"),
            [candidate],
            {
                self.server_id: r.BoundAction(
                    self.server_id, "migration", lambda: starts.append(self.server_id)
                )
            },
        )
        self.assertEqual(starts, [self.server_id])
        self.assertEqual(self.http_calls, [])  # no second source request
        self.assertEqual(self.co.ledger.pending("tp4")[0], 20)
        self.co.finish(
            "migration",
            self.server_id,
            "COMMITTED",
            ownership_confirmed=True,
            source_release_confirmed=True,
        )
        self.assertEqual(self.co.ledger.pending("tp1"), (0, 0, 0))
        self.assertEqual(
            self.co.ledger.entries[self.stream.reservation_key].state, "MOVED"
        )

    def test_cap_mismatch_and_ambiguous_binding_rejected(self):
        with self.assertRaises(ValueError):
            self.ingress.reconcile(self.snapshot(max_output_tokens=257), 10)
        snapshot = self.snapshot()
        snapshot["requests"].append(snapshot["requests"][0].copy())
        with self.assertRaises(ValueError):
            self.ingress.reconcile(snapshot, 10)

    def test_preemption_restores_future_block_commitment(self):
        self.ingress.reconcile(self.snapshot(owned_blocks=20), 10)
        self.assertEqual(self.co.ledger.pending("tp1"), (0, 0, 0))
        self.ingress.reconcile(self.snapshot(owned_blocks=2), 10)
        self.assertEqual(self.co.ledger.pending("tp1")[0], 18)

    def test_long_prompt_can_enter_idle_pool_instead_of_queueing_forever(self):
        source = replace(reading("tp1"), idle=True, decode_samples=0, decode_ratio=None)
        target = replace(reading("tp4"), idle=True, decode_samples=0, decode_ratio=None)
        result = p.admit(
            "long",
            3072,
            1024,
            (source, target),
            self.co.policy.cfg,
            p.ReservationLedger(),
            10,
            "long-key",
            {"tp1": 0.1, "tp4": 0.1},
        )
        self.assertEqual(result.pool_id, "tp1")
        self.assertEqual(result.max_output_tokens, 1024)


if __name__ == "__main__":
    unittest.main()
