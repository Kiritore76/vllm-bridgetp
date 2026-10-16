# SPDX-License-Identifier: Apache-2.0
"""Fixed supplement scope, native window coverage and host evidence checks."""

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp import monitor_pressure_host as monitor
from tools.bridge_tp import pressure_health_nd2 as nd2
from tools.bridge_tp import run_pressure_calibration as calibration


class HealthNd2Tests(unittest.TestCase):
    def setUp(self):
        base = calibration.ROOT / "experiments/phase9/pressure_balanced_v2"
        self.plan = json.loads((base / "health_nd2.json").read_text())
        self.inputs = json.loads((base / "calibration.json").read_text())

    def test_frozen_scope_and_kv_bounds(self):
        nd2.validate(self.plan, self.inputs)
        result = calibration.capacity_check(
            self.plan["scenes"][0],
            [
                dict(block_size=16, free_blocks=1962),
                dict(block_size=16, free_blocks=35000),
            ],
        )
        self.assertEqual(result["source"]["required_blocks"], 893)
        self.assertEqual(result["target"]["required_blocks"], 1149)
        for field, value in (
            ("seed", 42),
            ("repetitions", 1),
            ("migration_enabled", True),
        ):
            changed = copy.deepcopy(self.plan)
            changed[field] = value
            with self.assertRaises(ValueError):
                nd2.validate(changed, self.inputs)

    def check_coverage(self, mutate=lambda rows: rows):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "measurement_window.json").write_text(
                json.dumps(dict(started_unix_s=0))
            )
            (root / "request_results.json").write_text(
                json.dumps(
                    dict(
                        requests=[
                            dict(pool=pool, first_token_unix_s=0, last_token_unix_s=80)
                            for pool in ("source", "target")
                            for _ in range(2)
                        ]
                    )
                )
            )
            rows = [
                dict(
                    kind="engine_batch",
                    scheduled_unix_s=i / 2,
                    decode_requests=2,
                    decode_context_sum=(1 + i // 60) * 4096,
                    prefill_tokens=0,
                    recompute_tokens=0,
                )
                for i in range(160)
            ]
            with patch(
                "tools.bridge_tp.verify_performance_trace.read_journal",
                return_value=mutate(rows),
            ):
                return nd2.coverage(root, self.plan["scenes"][0])

    def test_native_context_coverage(self):
        self.assertEqual(self.check_coverage()["status"], "COVERED")

    def test_nominal_concurrency_cannot_cover_missing_bin(self):
        def change(rows):
            for row in rows:
                row["decode_context_sum"] = 4096
            return rows

        self.assertEqual(self.check_coverage(change)["status"], "COVERAGE_GAP")

    def test_sparse_samples_and_prefill_cannot_supply_windows(self):
        self.assertEqual(
            self.check_coverage(lambda rows: rows[::10])["status"], "COVERAGE_GAP"
        )

        def change(rows):
            for row in rows:
                row["prefill_tokens"] = 1
            return rows

        self.assertEqual(self.check_coverage(change)["status"], "COVERAGE_GAP")

    def test_monitor_retains_stalls_and_query_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "monitor.jsonl"
            rows = [
                dict(schema="PRESSURE_HOST_MONITOR_V1"),
                dict(sample_gap_s=17, gpu=dict(error="timeout")),
                dict(kind="monitor_footer", status="COMPLETED", samples=1),
            ]
            path.write_text("\n".join(map(json.dumps, rows)))
            receipt = monitor.validate_journal(path)
            self.assertEqual(receipt["largest_sample_gap_s"], 17)
            self.assertEqual(receipt["gpu_query_errors"], 1)
            path.write_text("\n".join(map(json.dumps, rows[:-1])))
            with self.assertRaises(RuntimeError):
                monitor.validate_journal(path)

    def test_gpu_query_timeout_is_evidence(self):
        with patch.object(
            monitor.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 3),
        ) as run:
            self.assertIn("TimeoutExpired", monitor.gpu_sample()["error"])
            self.assertEqual(run.call_args.kwargs["timeout"], 3)


if __name__ == "__main__":
    unittest.main()
