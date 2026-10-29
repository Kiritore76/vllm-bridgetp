# SPDX-License-Identifier: Apache-2.0
"""Six-run guard384 screening; preserve the production default and safety gates."""

import copy

from tools.bridge_tp import qwen3_migration_matrix as matrix
from tools.bridge_tp import qwen3_stay_migrate as matched


def config(inputs):
    cfg = matrix.config(inputs)
    burst = copy.deepcopy(cfg["recipes"][0])
    assert burst["scenario"] == "kv_growth" and burst["repetition"] == 1
    smooth = copy.deepcopy(burst)
    # The first six-source batch grows KV continuously without the later waves.
    # Keep target probes and the first background; do not infer steady capacity.
    smooth["jobs"] = [
        j
        for j in smooth["jobs"]
        if j["comparison_role"] == "short-probe" or j["at_s"] < 2
    ]
    smooth.update(
        name="guard384-growth-r1",
        scenario="continuous_growth",
        minimum_matching_migrations=1,
        evaluation_scope="single initial batch; continuous KV growth; finite load",
    )
    burst.update(name="guard384-burst-r1", scenario="fixed_burst")
    routed = copy.deepcopy(burst)
    for job in routed["jobs"]:
        if job["comparison_role"] == "source-stress":
            job.pop("route")
    routed.update(
        name="guard384-routed-r1",
        scenario="routed_burst",
        minimum_matching_migrations=0,
        evaluation_scope="original pressure router; no migration required if relieved",
        coverage_note="routing may relieve pressure without migration; zero moves "
        "are retained, not migration benefit evidence",
    )
    recipes = [smooth, burst, routed]
    for recipe in recipes:
        recipe["input_sha256"] = matched.digest(recipe["jobs"])
        recipe["routing_scope"] = (
            "stress arrivals use original pressure router; target anchors fixed"
            if recipe["scenario"] == "routed_burst"
            else "fixed initial pools isolate migration"
        )
    cfg.update(
        schema="QWEN3_KV_GUARD384_SCREEN_V1",
        recipes=recipes,
        policy_overrides={"capacity_guard_blocks": 384},
        calibration_status="offline-selected candidate; not GPU calibrated",
        note="six matched runs; same guard384 in STAY/MIGRATE; "
        "no contemporaneous guard125 arm; no forced migration",
    )
    return cfg
