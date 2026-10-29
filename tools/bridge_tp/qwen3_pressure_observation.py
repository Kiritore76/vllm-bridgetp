# SPDX-License-Identifier: Apache-2.0
"""Adapt measured TP2 inputs to the existing pressure policy, never execute moves."""

import json
from collections import Counter
from pathlib import Path

from tools.bridge_tp.pressure_cpu import load

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "experiments/platforms/qwen3_pressure_observation"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def profiles():
    return load("pressure_workload").build_profiles(
        read(ASSETS / "health_reference.json"), read(ASSETS / "service_evidence.json")
    )


def scenes():
    return [
        "short_burst_recovery",
        "natural_long_pressure",
        "target_busy",
        "kv_pressure",
    ]


def jobs(scene, inputs):
    """Fixed arrivals; final outputs never enter policy input or expectations."""
    if scene not in scenes():
        raise ValueError("unknown observation scene")
    output = []

    def add(at, size, cap, label, route=None, natural=False):
        source = inputs["requests"]["2048"]
        prompt = source["prompt"][: size - 32] + source["prompt"][-32:]
        if len(prompt) != size or size + cap > 8192:
            raise ValueError("observation request exceeds shared context support")
        text = dict(
            source,
            prompt=prompt,
            max_tokens=cap,
            ignore_eos=not natural,
            return_token_ids=True,
            stream=True,
        )
        text.pop("request_id", None)
        output.append(
            dict(
                id=f"{scene}-{label}-{len(output)}",
                at_s=at,
                request=text,
                route=route,
                stop_scope="natural EOS"
                if natural
                else "forced-length stress; not PMF accuracy",
            )
        )

    if scene == "short_burst_recovery":
        for k in range(3):
            add(k * 0.3, 1536, 4096, "held-source", "tp1")
        for k in range(30):
            add(5 + k * 0.15, 512, 128, "burst")
        for k in range(5):
            add(100 + k * 3, 512, 128, "recovery")
    elif scene == "natural_long_pressure":
        # The caller replaces these marked payloads with real chat templates.
        for k in range(16):
            add(k * 1.5, 2048, 6144, "natural-code-design", natural=True)
            output[-1]["natural_template"] = k % 2
    elif scene == "target_busy":
        for k in range(16):
            add(k * 0.2, 2048, 4096, "target-load", "tp4")
        for k in range(6):
            add(8 + k * 0.3, 1536, 4096, "source-load", "tp1")
        for k in range(10):
            add(15 + k, 512, 128, "unforced")
    else:
        for k in range(6):
            add(k * 0.3, 1536, 5120, "source-kv-load", "tp1")
        add(0, 2048, 4096, "target-background", "tp4")
        for k in range(8):
            add(45 + k * 3, 512, 128, "unforced")
    return output


def natural_jobs(values, tokenizer):
    from tools.bridge_tp.qwen3_online_validation import probe_requests

    templates = probe_requests(tokenizer)[1:]
    for row in values:
        if "natural_template" not in row:
            continue
        request = dict(templates[row.pop("natural_template")]["payload"])
        request.pop("request_id", None)
        request["max_tokens"] = min(6144, 8192 - len(request["prompt"]))
        row["request"] = request
    return values


def reactions():
    """Deterministic synthetic evidence through the unchanged coordinator.

    This checks reactions, not GPU behavior or predictor accuracy. There are
    no action callbacks and no HTTP clients. All PMFs here are labeled fixtures.
    """
    p, runtime, bounded = (
        load(n) for n in ("pressure_policy", "pressure_runtime", "bounded_length")
    )
    cfg = p.PressureConfig(**read(ASSETS / "policy.json"))
    work = profiles()

    def pool(name, now, *, load_value, ratio, free=2000, pending=0, age=0):
        return p.PoolReading(
            pool_id=name,
            sampled_at=now - age,
            published_at=now - age,
            free_blocks=free,
            block_size=16,
            context_limit=8192 if name == "tp1" else 16384,
            decode_ratio=ratio,
            decode_samples=1000,
            decode_sampled_at=now - age,
            idle=False,
            pending_prompt_tokens=pending,
            waiting=0,
            growth_blocks_s=10.0,
            prefill_batch_tokens=0,
            unallocated_prompt_blocks=0,
            prefill_generation=1,
            service_evidence=dict(
                interval_count=int((now - 9) * 1000),
                slow_interval_count=int((now - 9) * 100),
                window_s=3.0,
                slow_budget=0.01,
            ),
            workload=dict(
                status="OBSERVED",
                sampled_at=now - age,
                load=load_value,
                prefill_s=pending * work[name]["prefill_s_per_token"],
                profile=work[name],
            ),
        )

    def candidate(now, short=False, missing=False, identity="long"):
        remaining = bounded.bounded_length(
            (0.0, 0.0, 1.0, 0.0),
            (0, 15 if short else 4095, 16 if short else 4096),
            generated=100,
            predicted_at=100,
            max_output_tokens=6144,
        )
        return p.Candidate(
            identity,
            "tp1",
            now,
            now,
            100,
            1024,
            6144,
            100,
            100,
            100 * 16 * 147456,
            remaining,
            "UNAVAILABLE" if missing else "AVAILABLE",
            0.25,
            0.0625,
            cfg.capacity_lookahead_s,
        )

    cases = [
        ("healthy_stay", "STAY"),
        ("sustained_source_degradation", "START_SHADOW"),
        ("transient_recovery", "STAY"),
        ("target_busy", "STAY"),
        ("prefill_not_drained", "STAY"),
        ("stale_inventory", "STAY"),
        ("stale_prediction", "STAY"),
        ("short_candidate", "STAY"),
        ("kv_warning", "START_SHADOW"),
        ("kv_exhaustion_too_soon", "STAY"),
        ("choose_longest_tier", "START_SHADOW"),
    ]
    results = []
    for name, expected in cases:
        log = []
        co = runtime.PressureCoordinator(cfg, log.append)
        for tick in range(30):
            now = 10 + tick * 0.2
            healthy = name == "healthy_stay" or (
                name == "transient_recovery" and tick > 4
            )
            source = pool(
                "tp1",
                now,
                load_value=0.5 if healthy else 1.5,
                ratio=0.2 if healthy else 2.0,
                free=170
                if name == "kv_warning"
                else 40
                if name == "kv_exhaustion_too_soon"
                else 2000,
                pending=512 if name == "prefill_not_drained" else 0,
                age=4 if name == "stale_inventory" else 0,
            )
            target = pool(
                "tp4",
                now,
                load_value=1.2 if name == "target_busy" else 0.125,
                ratio=2.0 if name == "target_busy" else 0.2,
                free=11000,
            )
            candidates = [
                candidate(
                    now,
                    short=name == "short_candidate",
                    missing=name == "stale_prediction",
                )
            ]
            if name == "choose_longest_tier":
                candidates += [candidate(now, short=True, identity="short")]
            co.tick(now, source, target, candidates, actions=None)
        selected = [r for r in log if r["decision"]["action"] == "START_SHADOW"]
        observed = "START_SHADOW" if selected else "STAY"
        assert observed == expected, (name, observed, log[-1]["decision"])
        assert all(r["execution"] == "ADVISORY" for r in log)
        assert not co.ledger.snapshot()
        results.append(
            dict(
                name=name,
                expected=expected,
                observed=observed,
                selected_ids=sorted({r["decision"]["request_id"] for r in selected}),
                last_decision=log[-1]["decision"],
                passed=True,
            )
        )
    for name, expected, free in (
        ("route_prefer_tp1", "tp1", 2000),
        ("route_source_prompt_cannot_fit", "tp4", 1),
        ("route_target_busy", "tp1", 2000),
    ):
        co = runtime.PressureCoordinator(cfg, lambda row: None)
        source = pool("tp1", 10.0, load_value=0.5, ratio=0.2, free=free)
        target = pool(
            "tp4",
            10.0,
            load_value=1.2 if name == "route_target_busy" else 0.125,
            ratio=2.0 if name == "route_target_busy" else 0.2,
            free=11000,
        )
        admission = co.admission(name, 2048, 512, (source, target), now=10.0, key=name)
        assert admission.pool_id == expected, (name, admission)
        assert all(
            row["kind"] == "admission" and row["blocks"] == 0
            for row in co.ledger.snapshot()
        )
        results.append(
            dict(name=name, expected=expected, observed=admission.pool_id, passed=True)
        )
    return dict(
        scope="CPU synthetic fixtures, unchanged policy, NOT GPU acceptance",
        migration_execution_enabled=False,
        cases=results,
    )


def summarize(directory):
    from tools.bridge_tp.qwen3_migration_diagnostics import stream_quality

    rows = [
        json.loads(line)
        for line in (directory / "pressure_audit.jsonl").read_text().splitlines()
    ]
    decisions = [r for r in rows if r.get("kind") == "pressure_decision"]
    admissions = [r for r in rows if r.get("kind") == "pressure_admission"]
    final = [r for r in rows if r.get("kind") == "pressure_final"][-1]
    if (
        not decisions
        or final["outcome"] != "COMPLETED"
        or any(s["started"] for s in final["sessions"])
    ):
        raise ValueError("observation incomplete or migration unexpectedly started")
    if any(r["execution"] != "ADVISORY" for r in decisions):
        raise ValueError("observation executed a migration action")
    predictions = [v for r in decisions for v in r.get("prediction_inputs", [])]
    request_results = read(directory / "pressure_request_results.json")
    if len(request_results) != len(admissions) or any(
        not r["source_done"] or r["source_error"] for r in request_results
    ):
        raise ValueError("traffic completion evidence differs")
    streams = {}
    for line in (directory / "native_tokens.jsonl").read_text().splitlines():
        value = json.loads(line)
        if value.get("kind") == "visible_token":
            streams.setdefault(value["job_id"], []).append(value)
    if len(streams) != len(request_results):
        raise ValueError("visible token streams do not cover all dispatched jobs")
    quality = [
        dict(job_id=key, tokens=len(values), **stream_quality(values))
        for key, values in streams.items()
        if len(values) >= 2
    ]
    return dict(
        decision_ticks=len(decisions),
        migration_actions_executed=0,
        proposed_starts=sum(
            r["decision"]["action"] == "START_SHADOW" for r in decisions
        ),
        reasons=dict(Counter(r["decision"]["reason"] for r in decisions)),
        routed=dict(Counter(r["admission"]["pool_id"] for r in admissions)),
        traffic_admissions=len(admissions),
        output_tokens=sum(len(v) for v in streams.values()),
        request_quality=quality,
        slo_pass=sum(v["passed"] for v in quality),
        slo_evaluable=len(quality),
        source_zones=dict(Counter(r["decision"]["source"]["zone"] for r in decisions)),
        target_zones=dict(Counter(r["decision"]["target"]["zone"] for r in decisions)),
        relief_phases=dict(Counter(r["routing_relief"]["phase"] for r in decisions)),
        prediction_status=dict(Counter(v["status"] for v in predictions)),
        available_point_estimates=sum(
            v["status"] == "AVAILABLE" and v["remaining_mean"] is not None
            for v in predictions
        ),
        per_candidate_rejections=dict(
            Counter(
                reason
                for r in decisions
                for c in r["decision"]["candidates"]
                for reason in c["rejections"]
            )
        ),
        interpretation=(
            "repeated advice is not multiple migrations; "
            "absence of proposals can be correct"
        ),
    )
