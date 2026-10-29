# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the actual pilot parser and supported local service cells."""

import json
import tempfile
import unittest
from pathlib import Path

from tests.bridge_tp.test_pressure_policy import config, pool
from tools.bridge_tp import run_pressure_balanced_pilot as pilot
from tools.bridge_tp.pressure_cpu import load
from tools.bridge_tp.summarize_pressure_evidence import lookup_cells


def evidence():
    rows = []
    for run in ("one", "two"):
        for _ in range(5):
            for pool_name, n, kd, engine in (
                ("source", 1, 2048, 25),
                ("source", 2, 4096, 30),
                ("target", 1, 2048, 20),
            ):
                rows.append(
                    dict(
                        pool=pool_name,
                        run=run,
                        decode_requests=n,
                        decode_context_sum=kd,
                        prefill_tokens=0,
                        recompute_tokens=0,
                        engine_ms=engine,
                        gpu_ms=engine - 2,
                    )
                )
    return dict(
        schema="PRESSURE_EVIDENCE_V1",
        service_lookup=lookup_cells(rows),
        source_release_summary={"start_to_release_s": {"max": 17}},
    )


class ServiceTests(unittest.TestCase):
    def test_missing_shape_does_not_extrapolate(self):
        lookup = load("pressure_service").LocalServiceLookup(evidence())
        self.assertEqual(lookup.estimate("source", 1, 2048), 0.025)
        self.assertIsNone(lookup.estimate("source", 1, 8192))
        self.assertIsNone(lookup.estimate("source", 1, 2047))
        self.assertIsNone(lookup.estimate("target", 8, 4096))

    def test_delta_is_local_engine_proxy_not_benefit_score(self):
        lookup = load("pressure_service").LocalServiceLookup(evidence())
        source = dict(
            decode_requests=2,
            decode_context_sum=4096,
            prefill_tokens=0,
            recompute_tokens=0,
            requests=[
                dict(
                    request_id="a",
                    computed_before=2048,
                    decode_tokens=1,
                    prefill_tokens=0,
                    recompute_tokens=0,
                )
            ],
        )
        target = dict(
            decode_requests=0,
            decode_context_sum=0,
            prefill_tokens=0,
            recompute_tokens=0,
        )
        relief, cost = lookup.deltas(source, target, "a")
        self.assertAlmostEqual(relief, 0.05)
        self.assertAlmostEqual(cost, 0.2)


class AdmissionPressureTests(unittest.TestCase):
    def setUp(self):
        self.p = load("pressure_policy")
        root = Path(__file__).resolve().parents[2]
        summary = json.loads(
            (
                root / "experiments/phase9/pressure_balanced_v2/service_evidence.json"
            ).read_text()
        )
        self.lookup = load("pressure_service").LocalServiceLookup(summary)

    def admit(self, ledger, request_id="new", source=None):
        return self.p.admit(
            request_id,
            4096,
            4096,
            (source or pool(), pool("tp4")),
            config(),
            ledger,
            10,
            request_id,
            allowed_pool_id="tp1",
        )

    def test_zero_free_blocks_and_overload_do_not_queue_ingress(self):
        ledger = self.p.ReservationLedger()
        source = pool(
            free_blocks=0, decode_ratio=2, waiting=20, pending_prompt_tokens=32000
        )
        for request_id in ("first", "second", "third", "fourth"):
            result = self.admit(ledger, request_id, source=source)
            self.assertEqual(result.pool_id, "tp1")
            self.assertEqual(result.max_output_tokens, 4096)
        self.assertEqual(ledger.pending("tp1"), (0, 0, 16384))
        self.assertEqual(ledger.unobserved_ingress("tp1"), (4, 16384))
        self.assertTrue(
            all(row.blocks == row.decode_ratio == 0 for row in ledger.entries.values())
        )

    def test_migration_commitment_does_not_block_new_ingress(self):
        ledger = self.p.ReservationLedger()
        ledger.reserve(
            self.p.Reservation("migration", "moving", "tp1", 1800, 0.1, 0, "migration"),
            pool(),
        )
        result = self.admit(ledger)
        self.assertEqual(result.pool_id, "tp1")
        self.assertEqual(ledger.pending("tp1"), (1800, 0.1, 4096))
        self.assertEqual(ledger.entries["migration"].state, "ACTIVE")

    def test_missing_inventory_does_not_create_ingress_capacity_promises(self):
        ledger = self.p.ReservationLedger()
        result = self.admit(
            ledger,
            source=pool(
                free_blocks=None, sampled_at=0, decode_ratio=None, decode_samples=0
            ),
        )
        self.assertEqual(result.pool_id, "tp1")
        self.assertEqual(result.reservation_basis, "none")
        self.assertEqual(ledger.pending("tp1"), (0, 0, 4096))

    def test_context_infeasible_is_an_error_instead_of_capacity_queue(self):
        ledger = self.p.ReservationLedger()
        with self.assertRaises(self.p.RoutingError):
            self.p.admit(
                "oversize",
                8192,
                8192,
                (pool(), pool("tp4")),
                config(),
                ledger,
                10,
                "oversize",
            )
        self.assertEqual(ledger.entries, {})

    def route(self, prompt, cap=1024, source=None, target=None):
        return self.p.admit(
            "route",
            prompt,
            cap,
            (source or pool(), target or pool("tp4", context_limit=16384)),
            config(),
            self.p.ReservationLedger(),
            10,
            "route",
        )

    def test_prefer_tp1_even_for_medium_prompt_when_not_pressured(self):
        self.assertEqual(self.route(3072).pool_id, "tp1")
        self.assertEqual(self.route(512, source=pool(decode_ratio=2)).pool_id, "tp1")

    def test_long_prompt_prefers_tp4_only_when_tp1_pressured(self):
        result = self.route(3072, source=pool(decode_ratio=1.5))
        self.assertEqual(result.pool_id, "tp4")
        self.assertIn("TP1 pressure", result.routing_reason)

    def test_very_long_prompt_and_context_stop_budget_route_without_cap_shrink(self):
        self.assertEqual(self.route(4096).pool_id, "tp4")
        result = self.route(512, cap=8192)
        self.assertEqual(result.pool_id, "tp4")
        self.assertEqual(result.max_output_tokens, 8192)

    def test_unknown_or_stale_observations_do_not_block_routing(self):
        source = pool(
            decode_ratio=None,
            decode_samples=0,
            waiting=10,
            pending_prompt_tokens=12000,
            prefill_batch_tokens=None,
        )
        self.assertEqual(self.route(512, source=source).pool_id, "tp1")
        stale = pool(sampled_at=0)
        result = self.route(512, source=stale)
        self.assertEqual(result.pool_id, "tp1")

    def test_prefill_chunk_and_backlog_are_separate_pressure_axes(self):
        step = pool(pending_prompt_tokens=4096, prefill_batch_tokens=2048)
        reading = self.p.classify(step, config(), 10)
        self.assertEqual(reading.prefill, 1.0)
        self.assertEqual(reading.zone, "CRITICAL")
        missing = pool(pending_prompt_tokens=4096, prefill_batch_tokens=None)
        self.assertIn("prefill step", self.p.classify(missing, config(), 10).missing)
        heavy = pool(pending_prompt_tokens=16384, prefill_batch_tokens=0)
        self.assertEqual(self.p.classify(heavy, config(), 10).zone, "OVERLOAD")

    def test_third_decode_request_uses_observed_heavier_shapes(self):
        batch = dict(decode_requests=2, decode_context_sum=7000, prefill_tokens=0)
        self.assertIsNone(self.lookup.estimate("source", 3, 9048))
        detail = self.lookup.admission_details("source", batch, 2048)
        self.assertGreater(detail["cost"], 0)
        self.assertIn("not a hard bound", detail["basis"])
        self.assertGreaterEqual(detail["tail_max_s"], detail["upper_s"])
        ledger = self.p.ReservationLedger()
        result = self.p.admit(
            "peer",
            2048,
            2048,
            (pool(), pool("tp4")),
            config(),
            ledger,
            10,
            "peer",
            allowed_pool_id="tp1",
        )
        self.assertEqual(result.pool_id, "tp1")

    def test_envelope_never_extrapolates_beyond_observed_domain(self):
        self.assertIsNone(self.lookup.decode_envelope("source", 7, 14000))
        self.assertIsNone(self.lookup.decode_envelope("source", 3, 60000))
        batch = dict(decode_requests=6, decode_context_sum=24000, prefill_tokens=0)
        self.assertIsNone(self.lookup.admission_cost("source", batch, 2048))
        batch.update(decode_requests=2, prefill_tokens=999)
        self.assertIsNone(self.lookup.admission_cost("source", batch, 2048))

    def test_sparse_move_estimate_cannot_invent_source_relief(self):
        source = dict(
            decode_requests=3,
            decode_context_sum=9000,
            prefill_tokens=0,
            requests=[
                dict(
                    request_id="a",
                    computed_before=3000,
                    decode_tokens=1,
                    prefill_tokens=0,
                    recompute_tokens=0,
                )
            ],
        )
        target = dict(decode_requests=0, decode_context_sum=0, prefill_tokens=0)
        relief, cost = self.lookup.deltas(source, target, "a")
        self.assertEqual(relief, 0)
        self.assertGreater(cost, 0)


class PreflightTests(unittest.TestCase):
    def test_actual_full_argv_is_parsed_without_gpu_import_or_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = dict(
                prompt_boundary=2048,
                prefill_pending_budget=2048,
                prefill_waiting_budget=4,
                capacity_guard_blocks=125,
            )
            control = (
                load("config")
                .ControllerConfig(tp1_total_kv_blocks=2000, tp4_total_kv_blocks=30000)
                .to_json()
            )
            for name, value in (
                ("policy.json", cfg),
                ("controller.json", control),
                ("evidence.json", evidence()),
                (
                    "arrivals.json",
                    [
                        dict(
                            id="a",
                            at_s=0,
                            request=dict(model="qwen", prompt=[1, 2], max_tokens=128),
                        )
                    ],
                ),
            ):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            argv = [
                "--policy",
                "pressure-balanced-v1",
                "--policy-config",
                str(root / "policy.json"),
                "--controller-config",
                str(root / "controller.json"),
                "--service-evidence",
                str(root / "evidence.json"),
                "--arrivals",
                str(root / "arrivals.json"),
                "--run-dir",
                str(root / "run"),
                "--source-inventory",
                str(root / "source.jsonl"),
                "--target-inventory",
                str(root / "target.jsonl"),
                "--predictor-events",
                str(root / "predictor.jsonl"),
                "--predictor-sha256",
                "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7",
                "--migration-id",
                "cpu",
                "--max-seconds",
                "120",
                "--tick-s",
                "0.2",
                "--validate-only",
            ]
            args = pilot.parse_args(argv)
            prepared = pilot.preflight(args)
            self.assertFalse(prepared[-1]["gpu_validated"])
            self.assertEqual(prepared[-1]["jobs"], 1)
            self.assertEqual(len(prepared[-1]["input_sha256"]), 4)
            self.assertFalse((root / "run").exists())
            # Exercise the actual controller preflight, not only the outer plan.
            assets = Path("experiments/platforms/qwen3_pressure_observation")
            trial_policy = json.loads((assets / "policy.json").read_text())
            trial_policy["capacity_guard_blocks"] = 384
            (root / "policy.json").write_text(json.dumps(trial_policy))
            control["target_tp_size"] = 2
            (root / "controller.json").write_text(json.dumps(control))
            from tools.bridge_tp.qwen3_online_validation import CHECKPOINT_SHA

            argv[argv.index("--predictor-sha256") + 1] = CHECKPOINT_SHA
            argv[argv.index("--service-evidence") + 1] = str(
                assets / "service_evidence.json"
            )
            argv += [
                "--platform",
                "qwen3",
                "--qwen3-execute-migrations",
                "--health-reference",
                str(assets / "health_reference.json"),
            ]
            prepared = pilot.preflight(pilot.parse_args(argv))
            self.assertEqual(prepared[-1]["capacity_guard_blocks"], 384)
            self.assertEqual(prepared[-1]["capacity_guard_tokens"], 6144)
            self.assertFalse((root / "run").exists())


if __name__ == "__main__":
    unittest.main()
