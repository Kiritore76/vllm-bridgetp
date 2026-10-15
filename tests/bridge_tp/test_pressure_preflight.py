# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the actual pilot parser and supported local service cells."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.bridge_tp import run_pressure_balanced_pilot as pilot
from tools.bridge_tp.pressure_cpu import load
from tools.bridge_tp.summarize_pressure_evidence import lookup_cells


def evidence():
    rows = []
    for run in ("one", "two"):
        for _ in range(5):
            for pool, n, kd, engine in (
                ("source", 1, 2048, 25),
                ("source", 2, 4096, 30),
                ("target", 1, 2048, 20),
            ):
                rows.append(
                    dict(
                        pool=pool,
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


if __name__ == "__main__":
    unittest.main()
