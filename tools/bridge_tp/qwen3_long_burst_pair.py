# SPDX-License-Identifier: Apache-2.0
"""Short background plus predeclared long-input/long-output bursts, paired only."""

import copy

from tools.bridge_tp import qwen3_pressure_four_mode as pressure
from tools.bridge_tp import qwen3_stay_migrate as matched


def request(inputs, prompt_tokens, output_tokens):
    """Repeat the frozen token body while preserving its template boundaries."""
    row = copy.deepcopy(inputs["requests"]["6144"])
    tokens = row["prompt"]
    if (
        len(tokens) < 65
        or not prompt_tokens >= 64
        or (prompt_tokens + output_tokens > 32768 or not 0 < output_tokens <= 8192)
    ):
        raise ValueError("long burst exceeds native context/output contract")
    middle = tokens[32:-32]
    needed = prompt_tokens - 64
    row["prompt"] = (
        tokens[:32]
        + (middle * ((needed + len(middle) - 1) // len(middle)))[:needed]
        + tokens[-32:]
    )
    row.pop("request_id", None)
    row.update(
        max_tokens=output_tokens,
        ignore_eos=True,
        seed=20261030,
        stream=True,
        return_token_ids=True,
    )
    return row


def config(inputs):
    cfg = pressure.updated_config(inputs)
    recipes = []
    for offset, scenario in enumerate(("mixed_long_burst", "deep_long_burst")):
        jobs = []

        def add(at, prompt, cap, role, phase, *, jobs=jobs, scenario=scenario):
            jobs.append(
                dict(
                    id=f"{scenario}-{len(jobs)}",
                    at_s=at,
                    route=None,
                    request=request(inputs, prompt, cap),
                    workload_role=role,
                    workload_phase=phase,
                )
            )

        for i in range(150):
            at = i * 2
            add(
                at,
                512,
                128,
                "short_probe",
                "baseline"
                if at < 90
                else ("burst_overlap" if at < 120 else "recovery"),
            )
        if scenario == "mixed_long_burst":
            for i in range(8):
                prompt, cap = (8192, 4096) if i % 2 == 0 else (12288, 6144)
                add(90 + i * 2, prompt, cap, "long_input_output", "burst")
        else:
            for i in range(6):
                add(90 + i * 2, 16384, 8192, "long_input_output", "burst")
        jobs.sort(key=lambda j: (j["at_s"], j["id"]))
        recipes.append(
            dict(
                name=f"long-burst-{scenario}-r1",
                scenario=scenario,
                repetition=1,
                arm_order_offset=offset,
                fixed_output=True,
                jobs=jobs,
                input_sha256=matched.digest(jobs),
                planned_output_tokens=sum(j["request"]["max_tokens"] for j in jobs),
                routing_scope="autonomous pressure router; no forced pool or START",
            )
        )
    cfg.update(
        schema="QWEN3_LONG_BURST_STAY_MIGRATE_4_V1",
        arms=["stay", "migrate"],
        recipes=recipes,
        service_context_limits=dict(tp1=32768, tp2=32768),
        note="two predeclared short-background/long-I/O bursts, STAY/MIGRATE only; "
        "frozen synthetic token body, fixed output, not generation quality; "
        "zero migrations and losses retained; long-context reference extrapolation "
        "is not newly calibrated evidence",
    )
    return cfg


def prepare_native(plan, common, inputs):
    """Validate both engines' 32K execution before arming the controller."""
    from tools.bridge_tp.run_atomic_takeover import _post_json
    from tools.bridge_tp.run_qwen3_stay_migrate import prepare_native as check_native

    check_native(plan, common)
    records = []
    receipt = plan["run_dir"].parent / "context_boundary_receipt.json"
    for role in ("source", "target"):
        row = dict(role=role, status="STARTED", prompt_tokens=32767, max_tokens=1)
        records.append(row)
        common.write_json(receipt, dict(status="RUNNING", pools=records))
        body = request(inputs, 32767, 1)
        body["stream"] = False
        result = _post_json(plan["config"][role + "_url"], "/v1/completions", body, 600)
        row["response"] = result
        common.write_json(receipt, dict(status="RECEIVED", pools=records))
        usage = result.get("usage", {})
        if usage.get("prompt_tokens") != 32767 or usage.get("completion_tokens") != 1:
            raise ValueError("32K boundary did not execute the full token contract")
        row["status"] = "PASSED"
    check_native(plan, common)
    common.write_json(
        receipt,
        dict(
            status="PASSED",
            pools=records,
            note="boundary capacity, not concurrency or quality",
        ),
    )


def assess_stay(summary, recipe):
    """Accept full fixed work with valid arrivals, coverage and no migration."""
    return (
        summary["requests"] == len(recipe["jobs"])
        and summary["output_tokens"] == recipe["planned_output_tokens"]
        and summary["fixed_arrivals_valid"]
        and summary["load_coverage_pass"]
        and summary["migration_starts"] == 0
        and summary["committed_requests"] == 0
        and summary["source_releases"] == 0
    )


def assess_memory(plan, summary):
    """Require actual bounded transfers, commit/free and <=1s handoff samples."""
    import json

    rows = []
    for folder in sorted((plan["run_dir"] / "sessions").glob("migration-*")):
        state_path = folder / "takeover_state.json"
        if not state_path.exists():
            continue

        def read(p):
            return json.loads(p.read_text()) if p.exists() else {}

        state = read(state_path)
        sender = read(folder / "gpu_direct_sender.json")
        memory = read(folder / "source_history_memory_receipt.json")
        ranks = sender.get("ranks", [])
        valid = (
            state["state"] == "COMMITTED"
            and len(ranks) == 2
            and memory.get("status") == "READY_FOR_BOUNDED_HISTORY"
            and memory["cuda_free_bytes"]
            >= memory["source_pack_peak_bound_bytes"] + memory["forward_reserve_bytes"]
            and all(
                r.get("source_bounded_history")
                and r.get("source_history_payload_released")
                and type(r.get("source_pack_chunk_buffer_bytes")) is int
                and r["source_pack_chunk_buffer_bytes"] > 0
                and r["source_pack_chunk_buffer_bytes"] <= 16 * 1024 * 1024
                for r in ranks
            )
        )
        rows.append(
            dict(
                migration_id=state["migration_id"],
                pass_memory=valid,
                raw_history_bytes=memory.get("raw_history_bytes"),
                source_peak_bound_bytes=memory.get("source_pack_peak_bound_bytes"),
            )
        )
    accepted = (
        bool(rows)
        and all(r["pass_memory"] for r in rows)
        and len(rows) == summary["committed_requests"] == summary["source_releases"]
        and summary["maximum_handoff_s"] <= 1
        and summary["fixed_arrivals_valid"]
        and summary["load_coverage_pass"]
    )
    return {
        "pass": accepted,
        "sessions": rows,
        "note": "memory mechanism verification; not comparative performance proof",
    }
