# SPDX-License-Identifier: Apache-2.0
"""Three autonomous routing pressure workloads, four original strategies each."""

import copy
import random

from tools.bridge_tp import qwen3_four_mode as full
from tools.bridge_tp import qwen3_stay_migrate as matched


def config(inputs):
    recipes = []
    seed = 20261030
    for offset, scenario in enumerate(
        ("decode_growth", "long_short_shift", "prefill_competition")
    ):
        jobs = []

        def add(at, prompt, cap, role, phase, *, jobs=jobs, scenario=scenario):
            key = "6144" if prompt > 2048 else "2048"
            request = copy.deepcopy(inputs["requests"][key])
            tokens = request["prompt"]
            request["prompt"] = tokens[: prompt - 32] + tokens[-32:]
            request.pop("request_id", None)
            request.update(
                max_tokens=cap,
                ignore_eos=True,
                seed=seed,
                stream=True,
                return_token_ids=True,
            )
            if len(request["prompt"]) != prompt or prompt + cap > 8192:
                raise ValueError("pressure workload exceeds common context")
            jobs.append(
                dict(
                    id=f"{scenario}-{len(jobs)}",
                    at_s=at,
                    route=None,
                    request=request,
                    workload_role=role,
                    workload_phase=phase,
                )
            )

        if scenario == "decode_growth":
            order = ["long"] * 60 + ["short"] * 15
            random.Random(seed).shuffle(order)
            for index, kind in enumerate(order):
                add(
                    index * 4,
                    512,
                    5120 if kind == "long" else 128,
                    "long_decode" if kind == "long" else "short_probe",
                    "sustained",
                )
        elif scenario == "long_short_shift":
            for index in range(36):
                add(index * 2.5, 1536, 5120, "long_decode", "long_accumulation")
            for index in range(120):
                add(100 + index * 0.75, 512, 128, "short_probe", "short_burst")
            for index in range(12):
                add(200 + index * 8, 2048, 2048, "medium_decode", "recovery")
        else:
            order = [0, 0, 0, 1, 2] * 75
            random.Random(seed).shuffle(order)
            for index, kind in enumerate(order):
                prompt, cap, role = (
                    (6144, 512, "long_prefill"),
                    (4096, 1024, "long_prefill"),
                    (512, 128, "short_probe"),
                )[kind]
                add(index * 0.8, prompt, cap, role, "sustained")
        recipes.append(
            dict(
                name=f"pressure-four-{scenario}-r1",
                scenario=scenario,
                repetition=1,
                arm_order_offset=offset,
                fixed_output=True,
                jobs=jobs,
                input_sha256=matched.digest(jobs),
                planned_output_tokens=sum(j["request"]["max_tokens"] for j in jobs),
                routing_scope="all formal arrivals use their strategy; no forced pool",
            )
        )
    return dict(
        schema="QWEN3_PRESSURE_FOUR_MODE_12_V1",
        arms=full.ARMS,
        recipes=recipes,
        maximum_migrations=0,
        policy_overrides={"capacity_guard_blocks": 125},
        pool_comparison=True,
        routing_weights=full.measured_weights(
            full.observation.read(full.observation.ASSETS / "health_reference.json")
        )[0],
        horizon_s=300.0,
        max_dispatch_lag_s=1.0,
        natural_eos=False,
        metric=full.sustained.METRIC,
        itl_s=0.1,
        slow_fraction=0.01,
        maximum_gap_s=1.0,
        note="single-seed pressure screening, not calibrated saturation; "
        "all negative/zero-migration results retained; no forced START",
    )


def decorate(score, recipe):
    """Preserve individual short requests and changing workload phases."""
    for field in ("workload_role", "workload_phase"):
        labels = {j["id"]: j[field] for j in recipe["jobs"]}
        score[field + "_groups"] = {
            label: full.request_group(
                [
                    r
                    for r in score["request_rows"]
                    if r["in_arrival_cohort"] and labels[r["request_id"]] == label
                ]
            )
            for label in dict.fromkeys(labels.values())
        }
    return score


def updated_config(inputs):
    """Repeat the two affected workloads with identical updated serving paths."""
    cfg = config(inputs)
    cfg["recipes"] = [r for r in cfg["recipes"] if r["scenario"] != "decode_growth"]
    cfg.update(
        schema="QWEN3_UPDATED_PRESSURE_FOUR_MODE_8_V1",
        policy_overrides=dict(
            capacity_guard_blocks=125,
            kv_emergency_priority=True,
            kv_routing_priority=True,
        ),
        urgent_target_admission=dict(
            ordinary_admission_slots=15,
            total_running_limit=16,
            first_token_small_batch=True,
            async_load_bypasses_running_cap=True,
            handoff_pending_prefill_quantum=256,
        ),
        note="two frozen workloads, four strategies, current machine; "
        "same serving configuration in every arm; STAY/MIGRATE differ only "
        "in migration execution; retain zero migrations and negative outcomes",
    )
    return cfg


def aggregate(evaluations):
    return {
        scenario: full.aggregate([e for e in evaluations if e["scenario"] == scenario])
        for scenario in dict.fromkeys(e["scenario"] for e in evaluations)
    }
