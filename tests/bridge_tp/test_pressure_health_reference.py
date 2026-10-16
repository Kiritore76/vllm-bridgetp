# SPDX-License-Identifier: Apache-2.0
"""KV warnings must survive healthy/unknown service and precede depletion."""

import json
import unittest
from pathlib import Path

from tools.bridge_tp import pressure_health_reference as h


def warning(**changes):
    values = dict(
        free_blocks=250,
        block_size=16,
        sampled_at=10,
        published_at=10,
        now=10,
        growth_blocks_s=10,
        release_budget_s=16,
        guard_tokens=2000,
        max_age_s=2,
    )
    return h.kv_warning(**(values | changes))


class HealthReferenceTests(unittest.TestCase):
    def test_exact_repeated_shapes_and_diagnostic_quarantine(self):
        reference = dict(
            context_bin_tokens=2048,
            cells=[
                dict(
                    pool="source",
                    nd=8,
                    context_bin=1,
                    health_status="REFERENCE",
                    window_goodoutput=dict(median=268.35),
                ),
                dict(
                    pool="target",
                    nd=32,
                    context_bin=2,
                    health_status="DIAGNOSTIC_ONLY",
                    health_reason="delivery anomaly unresolved",
                ),
            ],
        )
        result = h.health_efficiency(reference, "source", 8, 3000, 268.35)
        self.assertEqual(result["eta"], 1)
        self.assertEqual(
            h.healthy_lookup(reference, "source", 9, 3000)["status"], "UNKNOWN"
        )
        self.assertEqual(
            h.healthy_lookup(reference, "source", 8, 4096)["status"], "UNKNOWN"
        )
        self.assertEqual(
            h.healthy_lookup(reference, "target", 32, 5000)["status"], "UNKNOWN"
        )
        # Service lookup is deliberately absent from the physical warning API.
        self.assertTrue(warning()["needs_relief"])

    def test_forecast_warns_above_guard_using_full_release_not_handoff(self):
        value = warning()
        self.assertEqual(value["warning_blocks"], 285)
        self.assertEqual(value["projected_free_at_release"], 90)
        self.assertEqual(value["required_release_blocks"], 35)
        self.assertEqual(value["action"], "EVALUATE_CAPACITY_CANDIDATES")
        self.assertFalse(warning(release_budget_s=0.4)["needs_relief"])
        self.assertFalse(warning(free_blocks=500)["needs_relief"])

    def test_low_free_warns_with_flat_or_unknown_growth(self):
        self.assertTrue(warning(free_blocks=100, growth_blocks_s=0)["needs_relief"])
        unknown = warning(free_blocks=100, growth_blocks_s=None)
        self.assertTrue(unknown["needs_relief"])
        self.assertIsNone(unknown["required_release_blocks"])
        self.assertFalse(warning(free_blocks=300, growth_blocks_s=None)["needs_relief"])

    def test_prompt_growth_is_not_a_second_subtraction_of_allocated_prefill(self):
        rows = [dict(prompt_tokens=4096, owned_blocks=256)]
        self.assertEqual(h.unallocated_prompt_blocks(rows, 16), 0)
        rows[0]["owned_blocks"] = 128
        self.assertEqual(h.unallocated_prompt_blocks(rows, 16), 128)
        self.assertFalse(
            warning(growth_blocks_s=0, pending_prompt_blocks=124)["needs_relief"]
        )
        value = warning(growth_blocks_s=0, pending_prompt_blocks=130)
        self.assertTrue(value["needs_relief"])
        self.assertEqual(value["free_blocks"], 250)
        self.assertEqual(value["required_release_blocks"], 5)
        rows[0]["owned_blocks"] = None
        self.assertIsNone(h.unallocated_prompt_blocks(rows, 16))

    def test_exported_reference_keeps_known_and_quarantined_cells_separate(self):
        root = Path(__file__).resolve().parents[2]
        reference = json.loads(
            (
                root / "experiments/phase9/pressure_balanced_v2/"
                "offline_health_reference.json"
            ).read_text()
        )
        self.assertEqual(len(reference["cells"]), 22)
        self.assertEqual(
            sum(c["health_status"] == "REFERENCE" for c in reference["cells"]), 20
        )
        self.assertAlmostEqual(
            h.healthy_lookup(reference, "source", 8, 3000)["reference_goodoutput_s"],
            268.35,
        )
        for context in (5000, 7000):
            self.assertEqual(
                h.healthy_lookup(reference, "target", 32, context)["status"], "UNKNOWN"
            )
        self.assertEqual(
            h.healthy_lookup(reference, "source", 8, None)["status"], "UNKNOWN"
        )
        self.assertFalse(reference["deployment_ready"])

    def test_age_quantization_and_stale_physical_evidence(self):
        value = warning(now=11, free_blocks=295)
        self.assertTrue(value["needs_relief"])
        self.assertEqual(value["warning_blocks"], 295)
        self.assertEqual(warning(now=13)["status"], "UNKNOWN")
        self.assertEqual(warning(published_at=11)["status"], "UNKNOWN")
        self.assertEqual(warning(free_blocks=-1)["status"], "UNKNOWN")

    def test_physical_exhaustion_relief_and_target_promises_still_gate(self):
        options = dict(
            releasable_blocks=40,
            target_free_blocks=500,
            target_pending_blocks=0,
            target_required_blocks=450,
            target_evidence_fresh=True,
        )
        result = h.capacity_candidate(warning(), **options)
        self.assertTrue(result["physical_candidate"])
        self.assertEqual(result["action"], "NO_START_IN_OFFLINE_REPLAY")
        for changes in (
            dict(releasable_blocks=34),
            dict(target_pending_blocks=51),
            dict(target_evidence_fresh=False),
        ):
            self.assertFalse(
                h.capacity_candidate(warning(), **(options | changes))[
                    "physical_candidate"
                ]
            )
        self.assertFalse(
            h.capacity_candidate(warning(free_blocks=138), **options)[
                "physical_candidate"
            ]
        )
        self.assertFalse(
            h.capacity_candidate(warning(free_blocks=0), **options)[
                "physical_candidate"
            ]
        )


if __name__ == "__main__":
    unittest.main()
