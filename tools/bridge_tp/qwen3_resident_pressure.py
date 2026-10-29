# SPDX-License-Identifier: Apache-2.0
"""Resident source work followed by routable arrivals, with paired guard arms."""

import copy

from tools.bridge_tp import qwen3_migration_matrix as matrix
from tools.bridge_tp import qwen3_stay_migrate as matched


def config(inputs):
    cfg = matrix.config(inputs)
    base = cfg["recipes"][0]
    recipes = []
    for busy in (False, True):
        jobs = copy.deepcopy(base["jobs"])
        for job in jobs:
            if job["comparison_role"] == "source-stress" and job["at_s"] >= 100:
                job.pop("route")
        if busy:
            background = next(
                j for j in jobs if j["comparison_role"] == "target-background"
            )
            jobs = [j for j in jobs if j["comparison_role"] != "target-background"]
            for index in range(8):
                job = copy.deepcopy(background)
                job.update(id=f"resident-target-background-{index}", at_s=index * 0.15)
                job["request"]["max_tokens"] = 6144
                jobs.append(job)
            jobs.sort(key=lambda j: (j["at_s"], j["id"]))
        load = "busy" if busy else "spare"
        for guard in (384, 125) if busy else (125, 384):
            recipe = copy.deepcopy(base)
            recipe.update(
                name=f"resident-{load}-g{guard}-r1",
                scenario=f"resident_{load}_g{guard}",
                jobs=copy.deepcopy(jobs),
                input_sha256=matched.digest(jobs),
                policy_overrides={"capacity_guard_blocks": guard},
                guard_blocks=guard,
                comparison_family=f"resident_{load}",
                require_target_background=busy,
                minimum_matching_migrations=1,
                matching_job_ids=[
                    j["id"]
                    for j in jobs
                    if j["comparison_role"] == "source-stress" and j["at_s"] < 2
                ],
                routing_scope="six initial source residents fixed; twelve later "
                "stress arrivals use original pressure router; target anchors fixed",
                evaluation_scope="same-batch guard comparison; single seed; "
                "no maximum-capacity or statistical-significance claim",
                coverage_note="resident work must actually migrate; busy case "
                "requires original target background emitting at START",
            )
            recipes.append(recipe)
    cfg.update(
        schema="QWEN3_RESIDENT_PRESSURE_GUARD_COMPARISON_V1",
        recipes=recipes,
        policy_overrides={},
        calibration_status="same-batch 125/384 screening; original policy unchanged",
        note="eight runs; same jobs across guards and arms; retain negative or "
        "zero-migration cases; no forced START or future-output reservation",
    )
    return cfg
