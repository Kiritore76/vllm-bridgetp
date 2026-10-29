# SPDX-License-Identifier: Apache-2.0
"""Frozen migration-benefit pairs; preserve the original policy and safety gates."""

import copy
import random

from tools.bridge_tp import qwen3_four_mode as full
from tools.bridge_tp import qwen3_stay_migrate as matched

SCENARIOS = ("kv_growth", "source_decode", "target_background")


def config(inputs):
    recipes = []
    for repetition in (1, 2, 3):
        for scenario in SCENARIOS:
            rng = random.Random(20261029 + repetition)
            jobs = []

            def add(
                at,
                prompt,
                cap,
                pool,
                role,
                *,
                repetition=repetition,
                scenario=scenario,
                jobs=jobs,
            ):
                request = copy.deepcopy(inputs["requests"]["2048"])
                tokens = request["prompt"]
                request["prompt"] = tokens[: prompt - 32] + tokens[-32:]
                request.pop("request_id", None)
                request.update(
                    max_tokens=cap,
                    ignore_eos=True,
                    stream=True,
                    return_token_ids=True,
                    seed=20261029 + repetition,
                )
                if len(request["prompt"]) != prompt or prompt + cap > 8192:
                    raise ValueError("matrix prompt or common context differs")
                jobs.append(
                    dict(
                        id=f"{scenario}-r{repetition}-{role}-{len(jobs)}",
                        at_s=at,
                        route=pool,
                        comparison_role=role,
                        stop_scope="fixed-length migration comparison",
                        request=request,
                    )
                )

            if scenario == "kv_growth":
                # Repeat the six-source KV pattern already measured on this GPU.
                for wave in (0, 100, 200):
                    for k in range(6):
                        add(
                            wave + k * 0.3 + rng.uniform(0, 0.02),
                            1536,
                            5120,
                            "tp1",
                            "source-stress",
                        )
                    add(wave, 2048, 4096, "tp4", "target-background")
                for k in range(15):
                    add(30 + k * 18, 512, 64, "tp4", "short-probe")
            elif scenario == "source_decode":
                # Fully occupy 16 source slots with bounded contexts. Unlike the
                # KV scenario, each complete wave fits 2048 KV blocks.
                for wave in (0, 110, 220):
                    for k in range(16):
                        add(
                            wave + k * 0.08 + rng.uniform(0, 0.01),
                            512,
                            1536,
                            "tp1",
                            "source-stress",
                        )
                    for k in range(2):
                        add(wave + k * 0.1, 2048, 6144, "tp4", "target-background")
                for k in range(15):
                    add(15 + k * 19, 512, 64, "tp4", "short-probe")
            else:
                for k in range(8):
                    add(k * 0.15, 2048, 6144, "tp4", "target-background")
                for k in range(6):
                    add(
                        2 + k * 0.3 + rng.uniform(0, 0.02),
                        1536,
                        5120,
                        "tp1",
                        "source-stress",
                    )
                for k in range(9):
                    add(20 + k * 15, 512, 64, "tp4", "short-probe")
            jobs.sort(key=lambda r: (r["at_s"], r["id"]))
            recipes.append(
                dict(
                    name=f"{scenario}-r{repetition}",
                    scenario=scenario,
                    repetition=repetition,
                    fixed_output=True,
                    jobs=jobs,
                    input_sha256=matched.digest(jobs),
                    expected_path="service"
                    if scenario == "source_decode"
                    else "capacity",
                    minimum_matching_migrations=2 if scenario == "kv_growth" else 1,
                    require_target_background=scenario == "target_background",
                    evaluation_scope="finite target-impact pair"
                    if scenario == "target_background"
                    else "repeated pressure waves; audit sustained coverage",
                )
            )
    cfg = matched.config(recipes)
    cfg.update(
        schema="QWEN3_MIGRATION_BENEFIT_MATRIX_V1",
        metric=full.sustained.METRIC,
        itl_s=0.1,
        slow_fraction=0.01,
        maximum_gap_s=1.0,
        routing_weights=full.measured_weights(
            full.observation.read(full.observation.ASSETS / "health_reference.json")
        )[0],
        note="fixed initial pools isolate migration; no forced START, "
        "no policy changes or bandwidth throttling; zero migration retained",
    )
    return cfg


def assess(recipe, arm, raw, audit, score):
    roles = {r["id"]: r["comparison_role"] for r in recipe["jobs"]}
    quality = score["request_rows"]
    groups = {
        role: full.request_group([r for r in quality if roles[r["request_id"]] == role])
        for role in ("source-stress", "target-background", "short-probe")
    }
    backgrounds = [r for r in raw if roles[r["job_id"]] == "target-background"]
    rows = []
    for start in (r for r in audit if r["kind"] == "pressure_actual_start"):
        decisions = [
            r
            for r in audit
            if r["kind"] == "pressure_decision"
            and r.get("execution") == "RESERVED"
            and r["unix_s"] <= start["unix_s"]
            and r["decision"].get("request_id") == start["request_id"]
        ]
        if not decisions:
            raise ValueError("actual migration lacks a matching reserved decision")
        reason = decisions[-1]["decision"]["reason"]
        path = (
            "capacity"
            if reason == "capacity relief"
            else ("service" if reason.startswith("sustained pressure;") else "unknown")
        )
        committed = [
            r
            for r in raw
            if r["server_id"] == start["request_id"] and r["response"]["committed"]
        ]
        released = any(
            r["kind"] == "pressure_source_release_confirmed"
            and r["migration_id"] == start["migration_id"]
            for r in audit
        )
        active = [
            r["job_id"]
            for r in backgrounds
            if r["response"]["emitted"][0]["unix_s"]
            <= start["unix_s"]
            < r["response"]["emitted"][-1]["unix_s"]
        ]
        rows.append(
            dict(
                migration_id=start["migration_id"],
                path=path,
                reason=reason,
                committed=bool(committed),
                source_released=released,
                active_target_background=active,
            )
        )
    matching = [
        r
        for r in rows
        if r["path"] == recipe["expected_path"]
        and r["committed"]
        and r["source_released"]
        and (not recipe["require_target_background"] or r["active_target_background"])
    ]
    handoffs = [
        r["response"].get("handoff_stall_s") for r in raw if r["response"]["committed"]
    ]
    return dict(
        expected_path=recipe["expected_path"],
        migrations=rows,
        matching_committed_released=len(matching),
        mechanism_coverage_pass=(len(matching) >= recipe["minimum_matching_migrations"])
        if arm == "migrate"
        else not rows,
        role_metrics=groups,
        load_coverage=score["load_coverage"],
        handoff_slo_pass=all(v is not None and v <= 1.0 for v in handoffs)
        if handoffs
        else None,
        note=recipe.get(
            "coverage_note",
            "service case counts only service moves; target-impact case "
            "requires visible background generation at START",
        ),
    )


def aggregate(evaluations):
    output = {}
    for scenario in dict.fromkeys(e["scenario"] for e in evaluations):
        selected = [e for e in evaluations if e["scenario"] == scenario]
        arms = full.aggregate(selected)
        for arm, summary in arms.items():
            group = [e for e in selected if e["arm"] == arm]
            duration = sum(e["score"]["horizon_s"] for e in group)
            roles = {}
            for role in ("source-stress", "target-background", "short-probe"):
                rows = [
                    r
                    for e in group
                    for r in e["score"]["request_rows"]
                    if r["in_arrival_cohort"] and e["roles"][r["request_id"]] == role
                ]
                roles[role] = dict(
                    full.request_group(rows),
                    soft_goodoutput=sum(r["good_tokens"] for r in rows) / duration,
                    hard_token_goodput=sum(
                        r["window_tokens"] for r in rows if r["slo_pass"]
                    )
                    / duration,
                )
            summary["role_metrics"] = roles
        output[scenario] = arms
    return output
