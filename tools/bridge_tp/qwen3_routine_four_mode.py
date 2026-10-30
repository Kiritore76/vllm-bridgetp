# SPDX-License-Identifier: Apache-2.0
"""Predeclared mixed, decode and transient workloads; no migration-based search."""

import copy
import random

from tools.bridge_tp import qwen3_pressure_four_mode as pressure
from tools.bridge_tp import qwen3_stay_migrate as matched


def config(inputs):
    cfg = pressure.updated_config(inputs)
    recipes = []
    seed = 20261030
    scenarios = ("routine_mixed", "moderate_decode", "transient_burst")
    for offset, scenario in enumerate(scenarios):
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
                raise ValueError("routine workload exceeds common context")
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

        if scenario == "routine_mixed":
            shapes = [(512, 128), (1024, 512), (1536, 1536), (2048, 2048), (512, 4096)]
            order = list(range(5)) * 15
            random.Random(seed).shuffle(order)
            for i, kind in enumerate(order):
                prompt, cap = shapes[kind]
                role = (
                    "short_probe"
                    if kind == 0
                    else ("long_decode" if kind == 4 else "medium_decode")
                )
                add(i * 4, prompt, cap, role, "sustained")
        elif scenario == "moderate_decode":
            order = ["long"] * 60 + ["short"] * 15
            random.Random(seed).shuffle(order)
            for i, kind in enumerate(order):
                add(
                    i * 4,
                    512,
                    3072 if kind == "long" else 128,
                    "long_decode" if kind == "long" else "short_probe",
                    "sustained",
                )
        else:
            for i in range(60):
                at = i * 5
                phase = (
                    "baseline"
                    if at < 100
                    else ("burst_overlap" if at < 120 else "recovery")
                )
                add(at, 512, 512, "background", phase)
            for i in range(12):
                add(100 + i * 1.5, 1024, 4096, "long_decode", "burst")
            for i in range(12):
                add(200 + i * 8, 1536, 1024, "recovery_probe", "recovery")
        jobs.sort(key=lambda j: (j["at_s"], j["id"]))
        recipes.append(
            dict(
                name=f"routine-four-{scenario}-r1",
                scenario=scenario,
                repetition=1,
                arm_order_offset=offset,
                fixed_output=True,
                jobs=jobs,
                input_sha256=matched.digest(jobs),
                planned_output_tokens=sum(j["request"]["max_tokens"] for j in jobs),
                routing_scope="all arrivals use their strategy; no forced pool",
            )
        )
    cfg.update(
        schema="QWEN3_ROUTINE_FOUR_MODE_12_V1",
        recipes=recipes,
        note="predeclared workloads, no STAY pilot or migration-based selection; "
        "normal/moderate labels are design intent, not measured service health; "
        "same policy/serving in four arms; retain zero migrations and losses",
    )
    return cfg
