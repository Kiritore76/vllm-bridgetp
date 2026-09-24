# SPDX-License-Identifier: Apache-2.0
"""Checks for the A4-P workload built from an existing target-only manifest."""

from __future__ import annotations

import unittest

from tools.bridge_tp.build_experiment_a4_pressure_manifest import build_manifest


class TestA4PressureManifest(unittest.TestCase):
    def test_preserves_target_flow_and_adds_bounded_tp1_contexts(self) -> None:
        target = {
            "job_id": "target_000",
            "pool": "target",
            "start_after_s": 0.0,
            "request": {
                "model": "bridgetp-model",
                "prompt": [100] * 2048,
                "max_tokens": 6000,
            },
        }
        base = {"format_version": 1, "jobs": [target]}
        manifest = build_manifest(
            base,
            source_jobs=4,
            source_prompt_tokens=4096,
            source_output_tokens=3500,
            source_start_after_s=2.0,
            source_start_interval_s=0.1,
            source_prompt_token_id=100,
            max_model_len=8192,
        )
        self.assertEqual(manifest["jobs"][0], target)
        self.assertEqual(len(manifest["jobs"]), 5)
        self.assertEqual(
            {row["job_id"] for row in manifest["jobs"]},
            {"target_000", *(f"a4_source_{i:03d}" for i in range(4))},
        )
        for row in manifest["jobs"][1:]:
            self.assertEqual(row["pool"], "source")
            self.assertEqual(len(row["request"]["prompt"]), 4096)
            self.assertLessEqual(4096 + row["request"]["max_tokens"], 8192)

    def test_rejects_source_request_over_context_limit(self) -> None:
        base = {
            "format_version": 1,
            "jobs": [{
                "job_id": "target_000",
                "pool": "target",
                "request": {"model": "bridgetp-model"},
            }],
        }
        with self.assertRaisesRegex(ValueError, "exceeds max model length"):
            build_manifest(
                base,
                source_jobs=1,
                source_prompt_tokens=6000,
                source_output_tokens=3500,
                source_start_after_s=0.0,
                source_start_interval_s=0.0,
                source_prompt_token_id=100,
                max_model_len=8192,
            )


if __name__ == "__main__":
    unittest.main()
