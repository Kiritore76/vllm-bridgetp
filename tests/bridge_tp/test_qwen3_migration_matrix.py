# SPDX-License-Identifier: Apache-2.0
"""Migration benefit workloads, path coverage and genuine target overlap."""

import copy
import unittest

from tools.bridge_tp import qwen3_migration_matrix as matrix


def config():
    return matrix.config(dict(requests={"2048": dict(prompt=list(range(2048)))}))


def fixture(path="capacity", background=True, committed=True, released=True):
    recipe = dict(
        expected_path="capacity",
        minimum_matching_migrations=1,
        require_target_background=background,
        jobs=[
            dict(id="s", comparison_role="source-stress"),
            dict(id="b", comparison_role="target-background"),
        ],
    )
    raw = [
        dict(
            job_id="s",
            server_id="s1",
            response=dict(
                committed=committed, emitted=[dict(unix_s=1.0), dict(unix_s=12.0)]
            ),
        ),
        dict(
            job_id="b",
            server_id="b1",
            response=dict(
                committed=False, emitted=[dict(unix_s=1.0), dict(unix_s=10.0)]
            ),
        ),
    ]
    audit = [
        dict(
            kind="pressure_decision",
            execution="RESERVED",
            unix_s=4.0,
            decision=dict(
                request_id="s1",
                reason="capacity relief"
                if path == "capacity"
                else "sustained pressure; service",
            ),
        ),
        dict(
            kind="pressure_actual_start", migration_id="m", request_id="s1", unix_s=5.0
        ),
    ]
    if released:
        audit.append(dict(kind="pressure_source_release_confirmed", migration_id="m"))
    quality = [
        dict(
            request_id=j,
            ttft_s=0.1,
            tpot_s=0.02,
            e2e_s=2.0,
            slow_fraction=0.0,
            maximum_gap_s=0.2,
            slo_pass=True,
        )
        for j in ("s", "b")
    ]
    return recipe, raw, audit, dict(request_rows=quality, load_coverage="COVERED")


class MigrationMatrixTests(unittest.TestCase):
    def test_one_second_handoff_is_reported_separately_from_path_coverage(self):
        recipe, raw, audit, score = fixture()
        raw[0]["response"]["handoff_stall_s"] = 1.001
        result = matrix.assess(recipe, "migrate", raw, audit, score)
        self.assertTrue(result["mechanism_coverage_pass"])
        self.assertFalse(result["handoff_slo_pass"])
        raw[0]["response"]["handoff_stall_s"] = 1.0
        self.assertTrue(
            matrix.assess(recipe, "migrate", raw, audit, score)["handoff_slo_pass"]
        )

    def test_eighteen_runs_fixed_seed_and_same_work(self):
        cfg = config()
        self.assertEqual(cfg, config())
        self.assertEqual(len(cfg["recipes"]) * len(cfg["arms"]), 18)
        self.assertEqual(cfg["arms"], ["stay", "migrate"])
        self.assertEqual(cfg["policy_overrides"], {})
        self.assertEqual(cfg["maximum_migrations"], 0)
        self.assertEqual({r["scenario"] for r in cfg["recipes"]}, set(matrix.SCENARIOS))
        for r in cfg["recipes"]:
            jobs = r["jobs"]
            self.assertEqual([j["at_s"] for j in jobs], sorted(j["at_s"] for j in jobs))
            self.assertTrue(all(0 <= j["at_s"] < 300 for j in jobs))
            self.assertTrue(all(j["request"]["ignore_eos"] for j in jobs))
            self.assertEqual(len(jobs), len({j["id"] for j in jobs}))
            self.assertEqual(matrix.matched.digest(jobs), r["input_sha256"])
            self.assertTrue(
                all(
                    len(j["request"]["prompt"]) + j["request"]["max_tokens"] <= 8192
                    for j in jobs
                )
            )
            self.assertEqual({j["route"] for j in jobs}, {"tp1", "tp4"})

    def test_service_wave_fits_cache_and_background_uses_eight_slots(self):
        recipes = config()["recipes"]
        service = next(r for r in recipes if r["scenario"] == "source_decode")
        first = [
            j
            for j in service["jobs"]
            if j["comparison_role"] == "source-stress" and j["at_s"] < 100
        ]
        self.assertEqual(len(first), 16)
        self.assertEqual(
            sum(
                (len(j["request"]["prompt"]) + j["request"]["max_tokens"]) / 16
                for j in first
            ),
            2048,
        )
        target = next(r for r in recipes if r["scenario"] == "target_background")
        self.assertEqual(
            sum(j["comparison_role"] == "target-background" for j in target["jobs"]), 8
        )

    def test_service_cannot_be_credited_for_a_kv_move(self):
        recipe, raw, audit, score = fixture()
        recipe["expected_path"] = "service"
        result = matrix.assess(recipe, "migrate", raw, audit, score)
        self.assertFalse(result["mechanism_coverage_pass"])
        self.assertEqual(result["migrations"][0]["path"], "capacity")

    def test_background_must_be_emitting_at_actual_start(self):
        recipe, raw, audit, score = fixture()
        self.assertTrue(
            matrix.assess(recipe, "migrate", raw, audit, score)[
                "mechanism_coverage_pass"
            ]
        )
        raw[1]["response"]["emitted"][-1]["unix_s"] = 4.0
        self.assertFalse(
            matrix.assess(recipe, "migrate", raw, audit, score)[
                "mechanism_coverage_pass"
            ]
        )

    def test_queued_background_and_uncommitted_or_unreleased_start_not_coverage(self):
        for committed, released in ((False, True), (True, False)):
            recipe, raw, audit, score = fixture(committed=committed, released=released)
            self.assertFalse(
                matrix.assess(recipe, "migrate", raw, audit, score)[
                    "mechanism_coverage_pass"
                ]
            )
        recipe, raw, audit, score = fixture()
        raw[1]["response"]["emitted"][0]["unix_s"] = 6.0
        self.assertFalse(
            matrix.assess(recipe, "migrate", raw, audit, score)[
                "mechanism_coverage_pass"
            ]
        )

    def test_zero_move_stays_visible_and_stay_is_not_migration_coverage(self):
        recipe, raw, audit, score = fixture()
        self.assertFalse(
            matrix.assess(recipe, "migrate", raw, [], score)["mechanism_coverage_pass"]
        )
        self.assertTrue(
            matrix.assess(recipe, "stay", raw, [], score)["mechanism_coverage_pass"]
        )

    def test_aggregate_target_metrics_pool_requests_not_p95s(self):
        from tools.bridge_tp import qwen3_four_mode as full

        # Use the real scorer for the shared horizon, then test role accounting.
        payload = dict(prompt=[1], max_tokens=2)
        data = [
            dict(
                job_id="s",
                initial_pool="tp1",
                source_error=None,
                request=payload,
                source_transport=dict(request_started_unix_s=0.0),
                response=dict(
                    committed=False,
                    emitted=[
                        dict(index=0, token_id=1, unix_s=0.1, origin="source"),
                        dict(index=1, token_id=2, unix_s=0.12, origin="source"),
                    ],
                ),
            )
        ]
        score = full.full_score(data, config(), 0.0, [])
        e = dict(
            arm="stay",
            scenario="target_background",
            score=score,
            roles=dict(s="target-background"),
        )
        result = matrix.aggregate([e, copy.deepcopy(e)])["target_background"]["stay"]
        self.assertEqual(result["denominator_s"], 600)
        self.assertEqual(result["role_metrics"]["target-background"]["requests"], 2)
        self.assertAlmostEqual(
            result["role_metrics"]["target-background"]["soft_goodoutput"], 4 / 600
        )
