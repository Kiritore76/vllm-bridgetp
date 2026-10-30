# SPDX-License-Identifier: Apache-2.0
"""One real MIGRATE rerun of the worst prefill workload; no forced route/START."""

import json

from tools.bridge_tp import qwen3_pressure_four_mode as pressure
from tools.bridge_tp import qwen3_stay_migrate as matched


def enable_urgent_target(plan):
    argv = plan["target_argv"]
    position = argv.index("--kv-transfer-config") + 1
    kv = json.loads(argv[position])
    kv.setdefault("kv_connector_extra_config", {})[
        "bridgetp_urgent_shadow_admission"
    ] = True
    argv[position] = json.dumps(kv)


def config(inputs, prior=None, *, kv_routing=False):
    cfg = pressure.config(inputs)
    recipe = next(r for r in cfg["recipes"] if r["scenario"] == "prefill_competition")
    baseline = dict(
        prior=None,
        input_sha256=matched.digest(recipe["jobs"]),
        comparison_scope="current-machine standalone verification; "
        "no historical environment or cross-machine performance comparison",
    )
    if prior is not None:
        previous = pressure.full.observation.read(prior / "result.json")
        old = next(
            r
            for r in previous["runs"]
            if r["name"] == "pressure-four-prefill_competition-r1-migrate"
        )
        if old["status"] != "COMPLETED" or old["summary"]["migration_starts"] != 0:
            raise ValueError("requires the completed zero-migration prefill baseline")
        if matched.digest(recipe["jobs"]) != old["input_sha256"]:
            raise ValueError("prefill baseline workload differs")
        baseline = dict(
            prior=str(prior),
            name=old["name"],
            input_sha256=old["input_sha256"],
            summary=old["summary"],
            comparison_scope="historical same workload; no contemporaneous STAY",
        )
    recipe["name"] = "kv-priority-prefill-r1"
    recipe["arm_order_offset"] = 0
    cfg.update(
        schema="QWEN3_KV_PRIORITY_VERIFY_V1",
        arms=["migrate"],
        recipes=[recipe],
        policy_overrides=dict(capacity_guard_blocks=125, kv_emergency_priority=True),
        urgent_target_admission=dict(
            ordinary_admission_slots=15, total_running_limit=16,
            first_token_small_batch=True, async_load_bypasses_running_cap=True,
            handoff_pending_prefill_quantum=256,
        ),
        note="one prefill MIGRATE; frozen workload; no forced migration; "
        "physical release/deadline gates unchanged",
    )
    if kv_routing:
        cfg["policy_overrides"]["kv_routing_priority"] = True
        cfg["note"] += "; KV-first ingress only during KV warning/recovery"
    return cfg, baseline


def assess(raw, audit):
    moves = []
    for start in (r for r in audit if r["kind"] == "pressure_actual_start"):
        decision = next(
            r
            for r in reversed(audit)
            if r["kind"] == "pressure_decision"
            and r.get("execution") == "RESERVED"
            and r["unix_s"] <= start["unix_s"]
            and r["decision"]["request_id"] == start["request_id"]
        )
        selected = next(c for c in decision["decision"]["candidates"] if c["selected"])
        if not decision["decision"]["capacity_path"] or not selected.get(
            "kv_priority_override"
        ):
            raise ValueError("verification START did not use urgent KV policy")
        response = next(
            r["response"] for r in raw if r["server_id"] == start["request_id"]
        )
        released = any(
            r["kind"] == "pressure_source_release_confirmed"
            and r["migration_id"] == start["migration_id"]
            for r in audit
        )
        benefit = selected.get("kv_joint_benefit")
        eligible_scores = [
            c.get("kv_joint_benefit", {}).get("relief_block_seconds")
            for c in decision["decision"]["candidates"]
            if c.get("selection_eligible") and c.get("kv_joint_benefit")
        ]
        joint_maximal = bool(
            benefit
            and benefit.get("method")
            == "OBSERVED_KV_X_MEAN_EARLY_RELEASE_SECONDS_V1"
            and eligible_scores
            and benefit["relief_block_seconds"] > 0
            and benefit["relief_block_seconds"] == max(eligible_scores)
        )
        moves.append(
            dict(
                migration_id=start["migration_id"],
                soft_warnings=selected["soft_warnings"],
                kv_joint_benefit=benefit,
                joint_score_maximal=joint_maximal,
                committed=response["committed"],
                source_released=released,
                handoff_s=response.get("handoff_stall_s"),
            )
        )
    committed = [m for m in moves if m["committed"] and m["source_released"]]
    mechanism = bool(committed) and any(
        "emergency target service budget unavailable" in m["soft_warnings"]
        for m in committed
    )
    handoff = (
        all(m["handoff_s"] is not None and m["handoff_s"] <= 1 for m in committed)
        if committed
        else None
    )
    joint_scoring = bool(moves) and all(m["joint_score_maximal"] for m in moves)
    return dict(
        migrations=moves,
        joint_scoring_pass=joint_scoring,
        mechanism_pass=mechanism,
        handoff_slo_pass=handoff,
        verification_pass=mechanism and handoff is True and joint_scoring,
        note="single fix/mechanism verification; "
        "does not mean all output SLOs or performance passed",
    )


def assess_routing(audit):
    arrivals = [r["admission"] for r in audit if r["kind"] == "pressure_admission"]
    checked = [a for a in arrivals if a.get("routing_kv") is not None]
    changed = [a for a in checked if a["routing_kv"]["changed_route"]]
    normal = [a for a in checked if a["routing_kv"]["mode"] == "SERVICE_PRESSURE"]
    valid = all(
        a["routing_kv"]["redirect"]
        and a["pool_id"] == "tp4"
        and a["routing_kv"]["target"]["pressure"]
        < a["routing_kv"]["source"]["pressure"]
        for a in changed
    ) and all(a["pool_id"] == a["routing_kv"]["previous_pool_id"] for a in normal)
    return dict(
        checked_arrivals=len(checked), changed_routes=len(changed),
        normal_service_routes=len(normal), invariants_pass=valid,
        verification_pass=bool(changed) and valid and len(checked)==len(arrivals),
        note="routing fix coverage only; zero migrations is allowed; "
        "no output-quality or comparative performance acceptance",
    )
