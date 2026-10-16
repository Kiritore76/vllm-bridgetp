# SPDX-License-Identifier: Apache-2.0
"""Sustained fixed-input preflight and drain-aware maximum-gap score contracts."""

import copy
import unittest

from tests.bridge_tp import test_pressure_kv_pilot as previous
from tools.bridge_tp import pressure_sustained as s
from tools.bridge_tp import run_pressure_comparison as compare


class SustainedInputs(previous.KvInputTests):
    def setUp(self):
        super().setUp()
        self.config = compare.pilot.read(previous.BUNDLE / "sustained.json")

    def test_frozen_inputs_preserve_measured_shapes_and_sampling(self):
        s.validate(self.config, self.inputs)
        for recipe in self.config["recipes"]:
            jobs = compare.arrivals(self.inputs, recipe)
            self.assertEqual(len(jobs), 17)
            self.assertEqual(sum(j["route"] == "tp1" for j in jobs), 13)
            self.assertEqual(sum(j["route"] == "tp4" for j in jobs), 4)
            self.assertEqual(jobs[-1]["at_s"], 278)
            self.assertTrue(all(j["request"]["ignore_eos"] for j in jobs))

    def test_only_sustain_can_be_changed(self):
        for key, value in (("maximum_gap_s", 2), ("horizon_s", 100)):
            changed = copy.deepcopy(self.config)
            changed[key] = value
            with self.assertRaises(ValueError):
                s.validate(changed, self.inputs)
        changed = copy.deepcopy(self.config)
        changed["recipes"][0]["arrivals"][-1]["at_s"] = 280
        with self.assertRaises(ValueError):
            s.validate(changed, self.inputs)

    def test_runtime_capacity_must_allow_one_offload_to_relieve_pressure(self):
        jobs = compare.arrivals(self.inputs, self.config["recipes"][0])
        initial = [j for j in jobs if j["at_s"] < 10]
        result = previous.kv.capacity_shape(
            [
                dict(block_size=16, free_blocks=1962),
                dict(block_size=16, free_blocks=35738),
            ],
            initial,
        )
        self.assertEqual(result["status"], "VALID")


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.config = compare.pilot.read(previous.BUNDLE / "sustained.json")

    def row(self, times, migrated=False):
        return dict(
            job_id="a",
            source_error=None,
            initial_pool="tp1",
            request=dict(max_tokens=len(times)),
            source_transport=dict(request_started_unix_s=100),
            response=dict(
                committed=migrated,
                emitted=[
                    dict(
                        index=i,
                        unix_s=t,
                        origin="target" if migrated and i else "source",
                    )
                    for i, t in enumerate(times)
                ],
            ),
        )

    def test_maximum_gap_includes_seam_but_excludes_ttft(self):
        score = s.score([self.row([110, 110.1, 112.1], True)], self.config)
        row = score["request_rows"][0]
        self.assertAlmostEqual(row["maximum_gap_s"], 2)
        self.assertEqual(row["ttft_s"], 10)
        self.assertAlmostEqual(row["maximum_gap_weight"], 0.5)
        self.assertAlmostEqual(
            score["actual_pool_G"]["tp1"] + score["actual_pool_G"]["tp4"],
            score["goodoutput_tokens_s"],
        )

    def test_drain_tokens_excluded_but_drain_gap_affects_window_weight(self):
        score = s.score([self.row([101, 101.01, 400, 401])], self.config)
        row = score["request_rows"][0]
        self.assertEqual(row["window_tokens"], 2)
        self.assertEqual(row["drain_tokens"], 2)
        self.assertAlmostEqual(row["maximum_gap_weight"], 1 / 298.99)
        self.assertAlmostEqual(score["good_tokens"], 2 / 298.99)
        self.assertFalse(score["drain_tokens_counted"])
        self.assertEqual(score["load_coverage"], "COVERAGE_GAP")

    def test_one_second_is_unpenalized_and_weights_take_minimum(self):
        times = [101 + i * 0.001 for i in range(2000)]
        times += [times[-1] + 1]
        score = s.score([self.row(times)], self.config)
        self.assertEqual(score["request_rows"][0]["final_weight"], 1)
        times[-1] += 1
        score = s.score([self.row(times)], self.config)
        self.assertAlmostEqual(score["request_rows"][0]["final_weight"], 0.5)

    def test_invalid_output_is_technical_failure(self):
        row = self.row([101, 102])
        row["response"]["emitted"][-1]["index"] = 7
        with self.assertRaises(ValueError):
            s.score([row], self.config)

    def test_coverage_requires_output_in_every_bin(self):
        times = [101 + i * 0.05 for i in range(5981)]
        score = s.score([self.row(times)], self.config)
        self.assertEqual(score["output_10s_bins"], 30)
        self.assertEqual(score["load_coverage"], "COVERED")


if __name__ == "__main__":
    unittest.main()
