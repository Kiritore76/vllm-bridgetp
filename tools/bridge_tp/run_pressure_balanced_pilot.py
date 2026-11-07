#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pressure pilot with optional serialized persistent-channel migrations.

The workload generator knows arrivals; the policy receives only due admissions,
published inventory, current stream windows and per-request frozen PMFs. Run
--validate-only on CPU before starting the existing five-GPU server mechanism.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.bridge_tp import conversion_decider as decision  # noqa: E402
from tools.bridge_tp import pressure_health_reference as health  # noqa: E402
from tools.bridge_tp.pressure_cpu import load  # noqa: E402
from tools.bridge_tp.pressure_progress import (  # noqa: E402
    DueJobs,
    ProgressWatch,
    StreamTracker,
    completed_streams,
    inventory_progress,
    save_progress,
    workload_progress,
)

p, r, o, i, s = (
    load(name)
    for name in (
        "pressure_policy",
        "pressure_runtime",
        "pressure_observation",
        "pressure_ingress",
        "pressure_service",
    )
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["pressure-balanced-v1"], required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--controller-config", type=Path, required=True)
    parser.add_argument("--service-evidence", type=Path, required=True)
    parser.add_argument("--health-reference", type=Path)
    parser.add_argument("--arrivals", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mechanism-root", type=Path)
    parser.add_argument("--migration-targets", choices=("B",))
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--target-inventory", type=Path, required=True)
    parser.add_argument("--source-context-limit", type=int, default=8192)
    parser.add_argument("--target-context-limit", type=int, default=16384)
    parser.add_argument("--predictor-events", type=Path, required=True)
    parser.add_argument("--predictor-sha256", required=True)
    parser.add_argument("--migration-id", required=True)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--tick-s", type=float, default=0.2)
    parser.add_argument("--no-progress-s", type=float, default=60)
    parser.add_argument("--no-migration", action="store_true")
    # S5b: a single-class start whose conversion creates the other class;
    # migration runs once the pair target and a channel source both serve.
    parser.add_argument("--deferred-migration", action="store_true")
    parser.add_argument("--platform", choices=["a100", "qwen3"], default="a100")
    parser.add_argument("--qwen3-execute-migrations", action="store_true")
    parser.add_argument(
        "--routing-strategy",
        default="pressure",
        choices=["pressure", "weighted-round-robin", "least-inflight"],
    )
    parser.add_argument("--routing-weights", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument(
        "--maximum-migrations",
        type=int,
        default=1,
        help="0 permits repeated serialized migrations",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        help="launcher instance manifest; recorded and checked against tp1/tp4",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--thread-profile", action="store_true")
    parser.add_argument(
        "--hold-unstarted",
        type=int,
        default=0,
        help="hold arrivals in the controller while each pool has this many "
        "requests sent but not started; 0 sends every arrival when due",
    )
    parser.add_argument(
        "--hold-tp1-long-kv",
        type=float,
        default=0.0,
        help="a TP1 takes no long prompt with a long request on it or this share "
        "of its KV in use; 0 applies neither",
    )
    parser.add_argument(
        "--hold-long-wait-s",
        type=float,
        default=300.0,
        help="a held long prompt is sent after this long without fresh KV evidence "
        "or engine room; measured too little KV for it keeps it waiting",
    )
    parser.add_argument("--test-only-cancel-first-shadow", action="store_true")
    parser.add_argument(
        "--scheduler",
        choices=("pressure", "unified"),
        default="pressure",
        help="unified (user 11-06): engine queue limit and prompt fit, one "
        "fullness for routing, KV/pressure triggered migration",
    )
    parser.add_argument("--tp1-max-num-seqs", type=int, default=16)
    parser.add_argument("--tp2-max-num-seqs", type=int, default=16)
    parser.add_argument(
        "--unified-itl-reference-ms",
        type=float,
        default=None,
        help="speed share denominator; default the scheduler module's value",
    )
    args = parser.parse_args(argv)
    if args.scheduler == "unified" and (
        args.hold_unstarted <= 0
        or args.hold_tp1_long_kv
        or args.tp1_max_num_seqs <= 0
        or args.tp2_max_num_seqs <= 0
    ):
        parser.error(
            "the unified scheduler needs the hold queue, positive slot counts "
            "and no TP1 long-prompt rule"
        )
    if args.maximum_migrations < 0:
        parser.error("maximum migrations must be nonnegative")
    if args.hold_unstarted < 0 or args.hold_long_wait_s < 0:
        parser.error("hold settings must be nonnegative")
    if min(args.source_context_limit, args.target_context_limit) <= 0:
        parser.error("positive configured context limits required")
    if any(
        not math.isfinite(v) or v <= 0
        for v in (args.tick_s, args.max_seconds, args.no_progress_s)
    ):
        parser.error("positive tick and runtime required")
    expected = "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7"
    if args.platform == "qwen3":
        from tools.bridge_tp.qwen3_online_validation import CHECKPOINT_SHA

        expected = CHECKPOINT_SHA
        if not args.no_migration and not args.qwen3_execute_migrations:
            parser.error("Qwen3 migration execution must be explicitly enabled")
    if args.deferred_migration and args.no_migration:
        parser.error("deferred migration is a migrating run")
    if args.qwen3_execute_migrations and (
        args.platform != "qwen3" or args.no_migration
    ):
        parser.error("Qwen3 execution flag requires the Qwen3 migrate arm")
    if args.predictor_sha256 != expected:
        parser.error("frozen platform checkpoint identity required")
    return args


def validate_guard_geometry(cfg, controller, platform):
    """Retain the A100 contract; allow only the scoped TP2 guard384 trial."""
    allowed = (
        {125, 384}
        if platform == "qwen3" and controller.target_tp_size == 2 and cfg.health_service
        else {125}
    )
    if (
        controller.block_size != 16
        or cfg.capacity_guard_blocks not in allowed
        or controller.tp1_total_kv_blocks <= cfg.capacity_guard_blocks
        or controller.tp4_total_kv_blocks <= cfg.capacity_guard_blocks
        or controller.policy.kv_bytes_per_token <= 0
    ):
        raise ValueError("pilot geometry/scoped KV guard contract differs")


def preflight(args):
    if any(not math.isfinite(w) or w <= 0 for w in args.routing_weights):
        raise ValueError("routing weights must be positive and finite")
    if args.routing_strategy != "pressure" and not args.no_migration:
        raise ValueError("baseline routing requires migration disabled")
    cfg = p.PressureConfig(**read(args.policy_config))
    if args.test_only_cancel_first_shadow and (
        not cfg.shadow_cancel or args.maximum_migrations != 0 or args.no_migration
    ):
        raise ValueError("cancel smoke requires active repeated shadow cleanup")
    if args.maximum_migrations != 1 and not cfg.health_service:
        raise ValueError("repeated migrations require the health policy")
    if cfg.health_service:
        if args.health_reference is None:
            raise ValueError("health service mode requires a frozen V2 reference")
        load("pressure_health").HealthTracker(read(args.health_reference), 0, None)
        if cfg.sustain_s < 3 or not cfg.capacity_lookahead_s:
            raise ValueError(
                "health mode requires >=3s persistence and independent KV lookahead"
            )
    raw = read(args.controller_config)
    # Reuse only mechanism geometry/transport rates. The legacy config loader
    # enforces benefit-model calibration and must not gate this new policy.
    controller = SimpleNamespace(
        **{
            key: raw[key]
            for key in (
                "source_url",
                "target_url",
                "block_size",
                "tp1_total_kv_blocks",
                "tp4_total_kv_blocks",
            )
        },
        rate=SimpleNamespace(**raw["rate"]),
        transfer_mode=raw.get("transfer_mode", "ADAPTIVE"),
        target_tp_size=load("tp_geometry").validate_session_tp(
            raw.get("target_tp_size", 4)
        ),
        policy=SimpleNamespace(kv_bytes_per_token=raw["policy"]["kv_bytes_per_token"]),
        ready_notification_mode=raw.get("ready_notification_mode", "FILE_POLL"),
        ready_notification_port=raw.get("ready_notification_port", 0),
        source_pool_id=raw.get("source_pool_id", "tp1"),
        target_pool_id=raw.get("target_pool_id", "tp4"),
        target_scoped_receivers=raw.get("target_scoped_receivers", False),
    )
    validate_guard_geometry(cfg, controller, getattr(args, "platform", "a100"))
    if controller.transfer_mode not in {"FULL_SPEED", "ADAPTIVE"}:
        raise ValueError("unknown pressure transfer_mode")
    if controller.transfer_mode == "ADAPTIVE":
        load("manager_m2").M2RateConfig(
            low_bytes_s=controller.rate.b_min_bytes_s,
            medium_bytes_s=controller.rate.b_start_bytes_s,
            high_bytes_s=controller.rate.b_max_bytes_s,
        ).validate()
    evidence = read(args.service_evidence)
    if controller.target_tp_size == 2 and (
        evidence.get("target_tp_size") != 2
        or (
            args.health_reference
            and read(args.health_reference).get("target_tp_size") != 2
        )
    ):
        raise ValueError("TP2 requires explicitly identified TP2 service profiles")
    lookup = s.LocalServiceLookup(evidence)
    if cfg.workload_pressure:
        load("pressure_workload").build_profiles(
            read(args.health_reference), evidence, cfg.health_min_efficiency
        )
    lookup.release_budget = evidence["source_release_summary"]["start_to_release_s"][
        "max"
    ]
    if cfg.capacity_lookahead_s and cfg.capacity_lookahead_s != lookup.release_budget:
        raise ValueError("KV pilot must use the frozen START-to-source-release budget")
    jobs = read(args.arrivals)
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("arrivals must be a nonempty list")
    manifest = instance_manifest(args, controller)
    mechanism_root = getattr(args, "mechanism_root", None)
    if mechanism_root is not None:
        if not (
            manifest and manifest.get("runtime_pool_ids")
            and getattr(args, "migration_targets", None) == "B"
            and controller.target_scoped_receivers
            and mechanism_root.resolve() == (
                args.run_dir / "migration" / manifest["legacy_pool_aliases"]["tp4"]
            ).resolve()
        ):
            raise ValueError("mechanism root must be the owned B receiver")
    elif getattr(args, "migration_targets", None):
        raise ValueError("receiver selection requires its owned mechanism root")
    pair = pair_pools(manifest)
    pool_names = {"tp1": "tp1", "tp4": "tp4"} | (
        manifest.get("runtime_pool_ids", {}) if manifest else {}
    )
    ids = set()
    for job in jobs:
        req = job["request"]
        if (
            job["id"] in ids
            or not math.isfinite(job["at_s"])
            or job["at_s"] < 0
            or not isinstance(req["prompt"], list)
            or not req["prompt"]
            or any(type(t) is not int or t < 0 for t in req["prompt"])
            or not 0 < req["max_tokens"] <= 8192
            or job.get("route") not in {None, *pair}
        ):
            raise ValueError("invalid/duplicate arrival or stop contract")
        ids.add(job["id"])
        limits = {
            name: limit
            for name, limit in (
                (pool_names["tp1"], args.source_context_limit),
                (pool_names["tp4"], args.target_context_limit),
            )
            if name in pair
        }
        routes = [job["route"]] if job.get("route") else limits
        if not any(
            len(req["prompt"]) + req["max_tokens"] <= limits[name] for name in routes
        ):
            raise ValueError("arrival has no context-feasible route")
    paths = (
        args.policy_config,
        args.controller_config,
        args.service_evidence,
        args.arrivals,
    ) + ((args.health_reference,) if cfg.health_service else ())
    receipt = dict(
        policy=p.VERSION,
        policy_config_sha256=cfg.sha256,
        input_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        jobs=len(jobs),
        maximum_migrations=args.maximum_migrations,
        migration_execution_enabled=not args.no_migration,
        mode="CPU_PREFLIGHT",
        test_only_cancel_first_shadow=args.test_only_cancel_first_shadow,
        routing_strategy=args.routing_strategy,
        routing_weights=args.routing_weights,
        service_estimator=(
            "observed engine-span local proxy; causal ITL delta unmeasured"
        ),
        gpu_validated=False,
        platform=getattr(args, "platform", "a100"),
        capacity_guard_blocks=cfg.capacity_guard_blocks,
        capacity_guard_tokens=cfg.capacity_guard_blocks * controller.block_size,
        kv_emergency_priority=cfg.kv_emergency_priority,
        calibration_scope=(
            "Qwen3 development experiment; original nominal KV "
            "guard is unvalidated; release maximum is empirical only"
        )
        if getattr(args, "platform", "a100") == "qwen3"
        else "original A100 contract",
        routing_entry="multi-instance (class ranking, then two-pool admit/decide)",
    )
    if manifest is not None:
        receipt["instance_manifest"] = manifest
    return cfg, controller, lookup, jobs, receipt


def batch(inventory, now, cfg, reading):
    if reading.idle:
        return dict(
            decode_requests=0,
            decode_context_sum=0,
            prefill_tokens=0,
            recompute_tokens=0,
            requests=[],
        )
    value = inventory.get("last_scheduled_batch")
    if (
        not value
        or value.get("error")
        or not 0 <= now - value["sampled_at"] <= cfg.max_age_s
    ):
        return None
    return value


def instance_manifest(args, controller):
    """Validate the pair, extra sources and the opt-in canonical runtime IDs."""
    if args.instances is None:
        return None
    manifest = read(args.instances)
    aliases = manifest.get("legacy_pool_aliases", {})
    rows = {row["instance_id"]: row for row in manifest.get("instances", ())}
    expected = dict(
        tp1=(1, args.source_context_limit),
        tp4=(controller.target_tp_size, args.target_context_limit),
    )
    if (
        manifest.get("schema") != "BRIDGETP_INSTANCE_MANIFEST_V1"
        or not aliases
        or not set(aliases) <= set(expected)
        or not set(aliases.values()) <= set(rows)
        or any(
            (rows[aliases[pool]]["tp_degree"], rows[aliases[pool]]["context_limit"])
            != expected[pool]
            for pool in aliases
        )
    ):
        raise ValueError("controller needs one TP1 source and one TP-k target pair")
    runtime_ids = manifest.get("runtime_pool_ids")
    if runtime_ids is not None and (
        runtime_ids != aliases
        or getattr(args, "scheduler", "pressure") != "unified"
        or len({gpu for row in rows.values() for gpu in row["physical_gpu_ids"]}) != 6
        or any(
            getattr(
                controller, {"tp1": "source_pool_id", "tp4": "target_pool_id"}[k], v
            )
            != v
            for k, v in aliases.items()
        )
    ):
        raise ValueError(
            "canonical runtime IDs require a matching six-card unified plan"
        )
    deferred = getattr(args, "deferred_migration", False)
    if len(aliases) == 1 and (
        not (args.no_migration or deferred)
        or args.no_migration
        and any(row.get("migration_channel") for row in rows.values())
        # S5b: only TP1s (an L0 that will merge the target) own channels.
        or "tp4" in aliases
        and any(row.get("migration_channel") for row in rows.values())
        or {row["tp_degree"] for row in rows.values()}
        != {expected[next(iter(aliases))][0]}
    ):
        # Single-class layout (L0/L2): one side of the pair; it routes only,
        # or (S5b) waits for the conversion that creates the other side.
        raise ValueError("single-class layout must be one class without migration")
    if deferred and len(aliases) != 1:
        raise ValueError("deferred migration starts single-class")
    endpoints = [row["endpoint"] for row in rows.values()]
    for row in extra_instances(manifest):
        if (
            not load("pressure_inventory_process").POOL_ID.fullmatch(row["instance_id"])
            or row["tp_degree"] not in {1, controller.target_tp_size}
            or row["context_limit"]
            != (
                args.source_context_limit
                if row["tp_degree"] == 1
                else args.target_context_limit
            )
            or not row.get("inventory_path")
            or endpoints.count(row["endpoint"]) != 1
            or row.get("migration_channel")
            and (row["tp_degree"] != 1 or not row.get("predictor_events"))
        ):
            raise ValueError("invalid routing-only instance " + row["instance_id"])
    return manifest


def acknowledged_blocks(entry, owned_blocks):
    """High-water mark of a migrated request's allocation on its target.

    A preemption there drops its owned blocks to zero (GPU 11-02); that is not
    a retraction of what was allocated. Its recompute is demand the inventory
    already reports as unallocated prompt, not this promise.
    """
    return max(entry.allocated_blocks, min(owned_blocks, entry.blocks))


def pair_pools(manifest):
    """Pool IDs of present pair sides; canonical when explicitly configured."""
    if manifest is None:
        return ("tp1", "tp4")
    return tuple(
        manifest.get("runtime_pool_ids", {}).get(p, p)
        for p in ("tp1", "tp4")
        if p in manifest["legacy_pool_aliases"]
    )


# Remaining-output quantiles published per request for the conversion planner.
REMAINING_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


# Reservation states the controller still waits on before it can finish; an
# ACCOUNTED admission completes only on its pool's next inventory sample.
UNSETTLED = frozenset({"ACTIVE", "COMMITTED", "ACCOUNTED"})


def unsettled_on(entries, pool_id):
    """Reservations of ``pool_id`` whose settlement needs its inventory.

    S5a GPU drain arm (11-01): an ACCOUNTED admission was not counted, the
    pool was retired one inventory sample early, and the controller then
    waited for it forever.
    """
    return sum(r.pool_id == pool_id and r.state in UNSETTLED for r in entries)


def assist_service_refusal(policy_cfg, reading, after, pending_decode, candidate):
    """Target service after a drain-assist move, or None if it holds.

    The active policy's own non-emergency rule (PressurePolicy.decide). With
    workload pressure the measured health reference is not consulted: it has
    no long-context cells (bins 1-3), and requiring it refused every long
    request (S5a GPU, 11-01).
    """
    cost = candidate.target_decode_cost
    if not policy_cfg.workload_pressure and not candidate.target_health_after_supported:
        return "target health not supported after one more decode"
    if (
        cost is None
        or not math.isfinite(cost)
        or cost < 0
        or after.decode is None
        or after.prefill is None
        or after.decode + pending_decode + cost > policy_cfg.leave_ratio
        or reading.pending_prompt_tokens
        or reading.waiting
    ):
        return "target service budget exceeded after the move"
    return None


def evacuation_refusal(
    policy_cfg,
    reading,
    after,
    *,
    candidates,
    block_size,
    guard_blocks,
    pending_blocks,
    pending_decode,
    service,
):
    """Whole-evacuation check before its first START (S5a all-or-nothing).

    Every request must fit at its final length (prompt + max output) on top
    of pending promises while the guard stays free, and, with the health
    service on, the target must stay within the policy's post-move service
    budget with every added decode. With ``migration_current_length`` it only
    has to fit at its current length, under the policy's target checks.
    """
    if policy_cfg.migration_current_length:
        return p.target_refusal(
            policy_cfg,
            reading,
            pending_blocks,
            sum(
                math.ceil((c.prompt_tokens + c.generated) / block_size)
                for c in candidates
            ),
        )
    required = sum(
        math.ceil((c.prompt_tokens + c.max_output_tokens) / block_size)
        for c in candidates
    )
    free = reading.free_blocks
    if free is None or free - pending_blocks - required < guard_blocks:
        return "target KV cannot hold every final length plus guard"
    if not service:
        return None
    costs = [c.target_decode_cost for c in candidates]
    combined = SimpleNamespace(
        target_decode_cost=None if None in costs else sum(costs),
        target_health_after_supported=all(
            c.target_health_after_supported for c in candidates
        ),
    )
    return assist_service_refusal(policy_cfg, reading, after, pending_decode, combined)


def remaining_bins(row, candidate):
    """Conservative remaining-token bins [(upper_tokens, mass), ...] of a request.

    The hard bound (max output minus generated) always holds; a fresh
    predictor distribution tightens it. Each quantile slice's mass is put at
    its upper edge, so the bins never understate the remaining output.
    """
    cap = max(0, row["max_output_tokens"] - row["generated"])
    remaining = getattr(candidate, "remaining", None)
    if remaining is None or remaining.deterministic:
        return [(cap, 1.0)], "HARD_CAP" if remaining is None else "DETERMINISTIC"
    bins, previous = [], 0.0
    for q in REMAINING_QUANTILES:
        upper = min(cap, remaining.quantile(q))
        if bins and bins[-1][0] == upper:
            bins[-1] = (upper, q - previous + bins[-1][1])
        else:
            bins.append((upper, q - previous))
        previous = q
    return [(u, round(m, 6)) for u, m in bins], "PREDICTED"


class NoPrediction:
    """Predictor of a source that only migrates when its group is retired."""

    @staticmethod
    def advisory(*args, **kwargs):
        return dict(status="UNAVAILABLE")


def channel_sources(manifest):
    """Routing-only TP1 rows that also own a migration channel (S1-d2c)."""
    return [r for r in extra_instances(manifest) if r.get("migration_channel")]


def policy_sources(candidates_by_source, evacuation_sources):
    """What the migration policy may choose from: never a same-TP source."""
    excluded = {row[0] for row in evacuation_sources}
    return {s: cs for s, cs in candidates_by_source.items() if s not in excluded}


def extra_instances(manifest):
    """Routing-only instances: every manifest row outside the migration pair."""
    if manifest is None:
        return []
    paired = set(manifest["legacy_pool_aliases"].values())
    return [r for r in manifest["instances"] if r["instance_id"] not in paired]


def unknown_routing_pools(
    args, block_size, extras=(), pair=("tp1", "tp4"), pool_names=None
):
    """Use configured context contracts while pressure telemetry is absent."""
    pool_names = pool_names or {"tp1": "tp1", "tp4": "tp4"}
    return tuple(
        p.PoolReading(
            pool_id=name,
            sampled_at=0,
            published_at=0,
            free_blocks=None,
            block_size=block_size,
            context_limit=limit,
            decode_ratio=None,
            decode_samples=0,
            decode_sampled_at=None,
            idle=False,
            pending_prompt_tokens=None,
            waiting=None,
            growth_blocks_s=None,
        )
        for name, limit in (
            *(
                (name, limit)
                for name, limit in (
                    (pool_names["tp1"], args.source_context_limit),
                    (pool_names["tp4"], args.target_context_limit),
                )
                if name in pair
            ),
            *((row["instance_id"], row["context_limit"]) for row in extras),
        )
    )


def dispatch_due(
    jobs, dispatched, elapsed, ingress, pools, tokens, audit, *, origin_unix_s=None
):
    """Submit every due legal arrival once; engine capacity is not a gate."""
    for value in dispatched.values():
        if value.future.done():
            value.future.result()
    for job in jobs:
        if job["id"] in dispatched or elapsed < job["at_s"]:
            continue
        value = ingress.submit(
            job["request"],
            tuple(pools),
            allowed_pool_id=job.get("route"),
            emission_sink=lambda token, job_id=job["id"]: tokens.submit(
                {"kind": "visible_token", "job_id": job_id, **token}
            ),
        )
        dispatched[job["id"]] = value
        value.held_s = job.get("held_s", 0.0)
        audit.write(
            dict(
                kind="pressure_dispatch",
                job_id=job["id"],
                at_s=job["at_s"],
                actual_s=(
                    value.dispatched_at - origin_unix_s
                    if origin_unix_s is not None
                    else elapsed
                ),
                client_id=value.client_id,
                pool=value.pool_id,
                frozen_cap=value.payload["max_tokens"],
                ingress_capacity_gate="held_s" in job,
                held_s=value.held_s,
                dispatch_entered_unix_s=value.dispatch_entered_at,
                admission_finished_unix_s=value.admission_finished_at,
                executor_submitted_unix_s=value.executor_submitted_at,
            )
        )


def run(args, prepared):
    diagnostic_path = os.environ.get("BRIDGETP_PRESSURE_DIAGNOSTICS")
    if not diagnostic_path:
        return _run(args, prepared)
    from vllm.bridge_tp.async_journal import AsyncJournal
    from vllm.bridge_tp.controller import pause_diagnostics as native_diagnostics

    args.run_dir.mkdir(parents=True, exist_ok=True)
    observer = load("pause_diagnostics").TimingObserver(
        AsyncJournal(Path(diagnostic_path))
    )
    try:
        with (
            load("pause_diagnostics").installed(observer),
            native_diagnostics.installed(observer),
            load("pause_diagnostics").GCMonitor(observer),
        ):
            if not args.thread_profile:
                return _run(args, prepared)
            with load("thread_profile").ThreadProfile(
                args.run_dir / "thread_profile.jsonl"
            ):
                return _run(args, prepared)
    finally:
        observer.close()


def _run(args, prepared):
    from tools.bridge_tp.pressure_sessions import PressureSessions
    from vllm.bridge_tp.async_journal import AsyncJournal

    cfg, controller, lookup, jobs, receipt = prepared
    receipt = receipt | {"mode": "ONLINE_FUNCTION_PILOT"}
    receipt["stream_storage"] = dict(
        token_history="lossless_primitive_arrays",
        transport_chunks_retained=False,
        full_visible_token_journal=True,
        gc_policy="unchanged",
    )
    if cfg.health_service:
        receipt["observation_sampling"] = dict(
            metrics_interval_s=1.0,
            metrics_socket_timeout_s=0.75,
            metrics_log_interval_s=1.0,
            unchanged_reconcile_log_interval_s=1.0,
            observation_flush_interval_s=1.0,
            inventory_and_migration_tick_s=args.tick_s,
            arrival_scheduler="independent_monotonic_thread",
            inventory_cache_poll_s=0.05,
            inventory_observer="exec_cpu_process_per_pool; compact_latest_snapshot",
            arrival_observation_io="async_identity_only; migration remains durable",
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (args.run_dir / name).exists()
        for name in (
            "pressure_audit.jsonl",
            "pressure_audit_durable.jsonl",
            "pressure_io_receipt.json",
            "native_tokens.jsonl",
            "pressure_request_results.json",
        )
    ):
        raise FileExistsError(
            "pressure pilot restart requires a fresh, reconciled run directory"
        )
    audit_path = args.run_dir / "pressure_audit.jsonl"
    audit = (
        load("pressure_io").PressureAudit(
            audit_path, AsyncJournal(audit_path, flush_interval_s=1.0), receipt
        )
        if cfg.health_service
        else load("audit").AuditLog(audit_path, receipt)
    )
    tokens = AsyncJournal(args.run_dir / "native_tokens.jsonl")
    co = r.PressureCoordinator(
        cfg,
        audit.write,
        routing_strategy=args.routing_strategy,
        routing_weights=tuple(args.routing_weights),
    )
    health_module = load("pressure_health") if cfg.health_service else None
    health_tokens = health_module.TokenWindows() if cfg.health_service else None
    service = o.NativeServiceWindow(health_observer=health_tokens)
    executor = ThreadPoolExecutor(max_workers=max(4, len(jobs) + 1))
    # Fixed pair through the multi-instance entry points; identical behaviour
    # and records for one instance per TP class.
    manifest = receipt.get("instance_manifest")
    canonical_ids = bool(manifest and manifest.get("runtime_pool_ids"))
    pool_names = {"tp1": "tp1", "tp4": "tp4"} | (
        manifest.get("runtime_pool_ids", {}) if manifest else {}
    )
    source_pool, target_pool = pool_names["tp1"], pool_names["tp4"]
    tp_of = {source_pool: 1, target_pool: getattr(controller, "target_tp_size", 4)}
    # A single-class layout starts with one side of the pair (S3b-2).
    pair = pair_pools(receipt.get("instance_manifest"))
    tp_of = {pool: tp for pool, tp in tp_of.items() if pool in pair}
    extras = extra_instances(receipt.get("instance_manifest"))
    tp_of.update({row["instance_id"]: row["tp_degree"] for row in extras})
    # Workload/health references exist per TP class, not per instance.
    profile_key = {pool: ("tp1" if tp == 1 else "tp4") for pool, tp in tp_of.items()}
    if getattr(args, "scheduler", "pressure") == "unified":
        # Conversions add instances at run time; unified runs are static.
        co.scheduler = "unified"
        if getattr(args, "unified_itl_reference_ms", None):
            co.itl_reference_s = args.unified_itl_reference_ms / 1000
        co.slots = {
            pool: args.tp1_max_num_seqs if tp == 1 else args.tp2_max_num_seqs
            for pool, tp in tp_of.items()
        }
        audit.write(
            dict(
                kind="pressure_scheduler",
                scheduler="unified",
                slots=co.slots,
                itl_reference_s=co.itl_reference_s,
                parameters={
                    k: getattr(r.unified, k)
                    for k in dir(r.unified)
                    if k.isupper() and not k.startswith("_")
                },
            )
        )
    context_of = {
        **{
            pool: limit
            for pool, limit in (
                (source_pool, args.source_context_limit),
                (target_pool, args.target_context_limit),
            )
            if pool in pair
        },
        **{row["instance_id"]: row["context_limit"] for row in extras},
    }
    for pool in tp_of:
        service.add_pool(pool)
    ingress = i.PressureIngress(
        co,
        service,
        executor,
        controller.source_url,
        controller.target_url,
        reconcile_log_interval_s=1.0 if cfg.health_service else 0,
        tp_of=tp_of,
        **({"pair_ids": pool_names} if canonical_ids else {}),
    )
    ingress.urls.update({row["instance_id"]: row["endpoint"] for row in extras})
    for pool in {source_pool, target_pool} - set(pair):
        ingress.urls.pop(pool, None)
    session = PressureSessions(
        persistent_cleanup=cfg.shadow_cancel,
        maximum_migrations=args.maximum_migrations,
        run_dir=getattr(args, "mechanism_root", None) or args.run_dir,
        migration_id=args.migration_id,
        config=controller,
        coordinator=co,
        ingress=ingress,
        executor=executor,
        audit=audit,
    )

    def scraper_for(url, tp):
        return load("telemetry").MetricsScraper(
            url,
            controller.block_size,
            controller.tp1_total_kv_blocks
            if tp == 1
            else controller.tp4_total_kv_blocks,
            timeout_s=0.75 if cfg.health_service else 2.0,
        )

    predictor = load("manager_m5").PredictorEventReader(
        args.predictor_events,
        args.predictor_sha256,
        align_history=getattr(args, "platform", "a100") == "qwen3",
    )

    def predictor_for(path):
        return load("manager_m5").PredictorEventReader(
            Path(path),
            args.predictor_sha256,
            align_history=getattr(args, "platform", "a100") == "qwen3",
        )

    evacuation_sources = []
    # Migration sources: the pair source, then every TP1 with its own channel.
    migration_sources = [(source_pool, args.source_inventory, predictor)] * (
        source_pool in pair
    ) + [
        (
            row["instance_id"],
            Path(row["inventory_path"]),
            predictor_for(row["predictor_events"]),
        )
        for row in channel_sources(receipt.get("instance_manifest"))
    ]
    manifest = receipt.get("instance_manifest")
    # Compatibility boundary: canonical runs keep actual IDs on both sides.
    alias_of = (
        {
            v: (v if canonical_ids else k)
            for k, v in manifest["legacy_pool_aliases"].items()
        }
        if manifest
        else {}
    )
    dispatched, previous = {}, {}
    growth_readers = {pool: health.NativeDecodeGrowth() for pool in tp_of}
    started = time.monotonic()
    origin_unix_s = time.time()
    health_tracker = (
        health_module.HealthTracker(
            read(args.health_reference),
            origin_unix_s,
            health_tokens,
            cache_closed_windows=cfg.workload_pressure,
        )
        if cfg.health_service
        else None
    )
    if health_tracker is not None:
        for row in (manifest["instances"] if canonical_ids else extras):
            health_tracker.add_pool(
                row["instance_id"], "source" if row["tp_degree"] == 1 else "target"
            )
    workload_module = load("pressure_workload") if cfg.workload_pressure else None
    workload_profiles = (
        workload_module.build_profiles(
            read(args.health_reference),
            read(args.service_evidence),
            cfg.health_min_efficiency,
        )
        if workload_module is not None
        else None
    )
    if workload_profiles is not None:
        audit.write(
            dict(
                kind="pressure_workload_profiles",
                profiles=workload_profiles,
                basis="existing repeated evidence; no probing requests",
            )
        )
    audit.write(dict(kind="pressure_clock_origin", unix_s=origin_unix_s))
    last_view = None
    outcome = "FAILED_OR_INTERRUPTED"
    watch = ProgressWatch(args.no_progress_s)
    progress_path = args.run_dir / "pressure_progress.json"
    last_status_at = -math.inf
    last_metrics_error = {}
    last_metrics_logged = {}
    background_metrics = []
    background_inventory = []
    dispatcher = None
    last_inventory_status = {}
    last_inventory_progress = ()
    failure = None
    # Per-instance observation resources keyed by pool id (S3b-1). Initial
    # instances gate start-up on inventory; layout-control additions join the
    # readings once their first inventory arrives.
    slots = {}

    def metrics_for(scraper):
        return load("pressure_io").BackgroundMetrics(
            lambda raw, scraper=scraper: load("pause_diagnostics").call(
                "metrics.scrape", o.scrape_inventory_metrics, scraper, raw
            )
        )

    def add_slot(pool_id, path, url, tp, initial, metrics=False):
        reader = load("pressure_inventory_process").ProcessInventory(
            path, pool_id, args.run_dir
        )
        background_inventory.append(reader)
        slot = dict(reader=reader, scraper=scraper_for(url, tp), initial=initial)
        slot["metrics"] = metrics_for(slot["scraper"]) if metrics else None
        if slot["metrics"] is not None:
            background_metrics.append(slot["metrics"])
        slots[pool_id] = slot

    def remove_slot(pool_id):
        slot = slots.pop(pool_id)
        for sampler, owner in (
            (slot["reader"], background_inventory),
            (slot["metrics"], background_metrics),
        ):
            if sampler is not None:
                sampler.stop.set()
                sampler.close()
                owner.remove(sampler)

    layout_state = dict(
        signature=None,
        migration_target=target_pool if target_pool in pair else None,
        revision=None,
        status_at=-math.inf,
        candidates={},
        pending=False,
        pressure={},
        # S5a: the conversion's drain-assist request and its last outcome.
        assist=None,
        assist_seen=None,
        assist_result=None,
        # (unix_s, source) of migration starts, for the conversion decider.
        migration_starts=[],
        tilt=frozenset(),
        tilted=0,
    )

    def note_migration_start(source, value):
        layout_state["migration_starts"].append((time.time(), source))
        return value

    instance_of = (
        {pool_names[k]: v for k, v in manifest["legacy_pool_aliases"].items()}
        if manifest
        else {}
    )
    if canonical_ids:
        co.pool_sort_ids = {v: k for k, v in manifest["legacy_pool_aliases"].items()}
        audit.write(
            dict(kind="pressure_runtime_identities", pool_ids=pool_names,
                 compatibility_sort_ids=co.pool_sort_ids)
        )

    def request_rows(raw):
        rows = []
        for row in raw["requests"]:
            candidate = layout_state["candidates"].get(row["request_id"])
            bins, basis = remaining_bins(row, candidate)
            rows.append(
                dict(
                    request_id=row["request_id"],
                    prompt_tokens=row["prompt_tokens"],
                    generated=row["generated"],
                    max_output_tokens=row["max_output_tokens"],
                    remaining_bins=bins,
                    remaining_basis=basis,
                    tokens_s=getattr(candidate, "source_tokens_s", None),
                    transfer_bytes=getattr(candidate, "transfer_bytes", None),
                    # Only a migration source's candidate can be started.
                    action_available=candidate is not None
                    and candidate.action_available
                    and not candidate.migrating
                    and not candidate.finished,
                )
            )
        return rows

    def start_assist(assist, now, target_pool, candidates_by_source, actions):
        """Final safety checks of one conversion-assist START (S5a).

        Hard: the target keeps its KV guard after the request reaches its
        final length (prompt + max output), on top of every pending promise.
        Service: with the health service on, the target shape after one more
        decode must be a measured supported one. Then the coordinator's
        single-session reservation path starts it.
        """
        source = alias_of.get(assist["source"], assist["source"])
        candidate = next(
            (
                c
                for c in candidates_by_source.get(source, ())
                if c.request_id == assist["request_id"]
            ),
            None,
        )
        if candidate is None:
            return "request is not a current candidate of the source"
        if candidate.migrating or candidate.finished:
            return "request is migrating or finished"
        if actions is None:
            return "no start binding (session not ready or no observed rates)"
        if "evacuate" in assist:
            by_id = {
                c.request_id: c for cs in candidates_by_source.values() for c in cs
            }
            if not set(assist["evacuate"]) <= set(by_id):
                return "an evacuation request is not a current candidate"
            pending_blocks, pending_decode, _ = co.ledger.pending(target_pool.pool_id)
            reading = co.ledger.routing_reading(target_pool)
            refusal = evacuation_refusal(
                co.policy.cfg,
                reading,
                p.classify(reading, co.policy.cfg, now),
                candidates=[by_id[i] for i in assist["evacuate"]],
                block_size=controller.block_size,
                guard_blocks=cfg.capacity_guard_blocks,
                pending_blocks=pending_blocks,
                pending_decode=pending_decode,
                service=cfg.health_service,
            )
            if refusal:
                return refusal
        current = co.policy.cfg.migration_current_length
        required = math.ceil(
            (
                candidate.prompt_tokens
                + (candidate.generated if current else candidate.max_output_tokens)
            )
            / controller.block_size
        )
        pending, pending_decode, _ = co.ledger.pending(target_pool.pool_id)
        free = target_pool.free_blocks
        if current:
            refusal = p.target_refusal(
                co.policy.cfg,
                co.ledger.routing_reading(target_pool),
                pending,
                required,
            )
            if refusal:
                return refusal
        elif free is None or free - pending - required < cfg.capacity_guard_blocks:
            return "target KV cannot hold the final length plus guard"
        if cfg.health_service and not current:
            reading = co.ledger.routing_reading(target_pool)
            refusal = assist_service_refusal(
                co.policy.cfg,
                reading,
                p.classify(reading, co.policy.cfg, now),
                pending_decode,
                candidate,
            )
            if refusal:
                return refusal
        return co.assist_start(
            now,
            source,
            target_pool.pool_id,
            candidate,
            actions.get(candidate.request_id),
            target_pool,
            required,
            before_quiesce=bool(assist.get("before_quiesce")),
        )

    def publish_layout_status(now, polled, dispatched, *, force=False):
        """Controller-side drain evidence per instance for the conversion executor.

        Counts what only the controller knows (open admissions, streams it still
        relays, migration sessions); the executor pairs it with fresh native
        evidence before declaring an instance drained.
        """
        if not force and now - layout_state["status_at"] < 0.5:
            return False
        layout_state["status_at"] = now
        entries = co.ledger.live_rows()
        streams, sessions = {}, {}
        for value in tracker.open.values():
            if not value.future.done():
                # A committed migration moves ownership (keyed by server id).
                owner = co._owners.get(
                    getattr(value, "server_id", None),
                    co._owners.get(value.client_id, value.pool_id),
                )
                streams[owner] = streams.get(owner, 0) + 1
        for session, (_, target) in tuple(co._sessions.items()):
            if co.ledger.entries[session].state == "ACTIVE":
                for pool in {target, co._session_source.get(session)} - {None}:
                    sessions[pool] = sessions.get(pool, 0) + 1
        instances = {}
        for pool in tp_of:
            raw = polled.get(pool)
            instances[instance_of.get(pool, pool)] = dict(
                pool_id=pool,
                lifecycle=co.lifecycle(pool),
                open_admissions=unsettled_on(entries, pool),
                streams_pending=streams.get(pool, 0),
                sessions=sessions.get(pool, 0),
                requests=[] if raw is None else request_rows(raw),
                pressure=layout_state["pressure"].get(pool),
                tp_degree=tp_of[pool],
                inventory=None
                if raw is None
                else dict(
                    sampled_at=raw["sampled_at"],
                    generation=raw["generation"],
                    running=raw["running"],
                    waiting=raw["waiting"],
                    requests=len(raw["requests"]),
                    free_blocks=raw["free_blocks"],
                    total_blocks=raw["total_blocks"],
                    block_size=raw["block_size"],
                ),
            )
        # Conversion decision signals (record-only decider): arrivals and
        # migration starts of the last window.
        window = decision.WINDOW_S
        streams = tracker.recent(now, window)
        recent = [
            len(v.payload["prompt"]) + int(v.payload["max_tokens"]) for v in streams
        ]
        long_requests = [n for n in recent if n >= decision.LONG_TOKENS]
        short_window = decision.SHORT_WINDOW_S
        in_short_window = [
            n
            for v, n in zip(streams, recent, strict=True)
            if now - v.dispatched_at <= short_window
        ]
        starts = layout_state["migration_starts"]
        while starts and now - starts[0][0] > window:
            starts.pop(0)
        hold = dispatcher.hold if dispatcher is not None else None
        workload = dict(
            window_s=window,
            arrivals=len(recent)
            if hold is None
            else hold.arrivals_since(time.monotonic() - window),
            long_arrivals=len(long_requests),
            long_final_tokens=sum(long_requests),
            short_window_s=short_window,
            short_window_arrivals=len(in_short_window)
            if hold is None
            else hold.arrivals_since(time.monotonic() - short_window),
            short_window_long=sum(
                n >= decision.LONG_TOKENS for n in in_short_window
            ),
            migration_starts=[[round(now - t, 3), source] for t, source in starts],
            tilt=sorted(layout_state["tilt"]),
            tilted_long_requests=layout_state["tilted"],
        )
        if hold is not None:
            workload["held_long_prompt_tokens"] = list(hold.held_long)
        path = args.run_dir / "layout_status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                dict(
                    schema="PRESSURE_LAYOUT_STATUS_V1",
                    unix_s=time.time(),
                    workload_origin_unix_s=origin_unix_s,
                    layout_revision=layout_state["revision"],
                    instances=instances,
                    # Measured START-to-source-release budget (S4 planner).
                    release_budget_s=lookup.release_budget,
                    migration_target=instance_of.get(
                        layout_state["migration_target"],
                        layout_state["migration_target"],
                    )
                    if polled.get(layout_state["migration_target"]) is not None
                    else None,
                    assist_result=layout_state["assist_result"],
                    workload=workload,
                    # Ledger state of every drain-assist session (S5a cap).
                    assist_sessions={
                        sid: co.ledger.entries[sid].state
                        for sid in tuple(co._assist_sessions)
                    },
                )
            ),
            encoding="utf-8",
        )
        # A Windows reader may hold it; the next tick retries.
        try:
            os.replace(temporary, path)
        except PermissionError:
            return False
        return True

    def apply_layout_control():
        """Reconcile instances with the conversion executor's layout file."""
        path = args.run_dir / "layout_control.json"
        if not path.is_file():
            return
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if signature == layout_state["signature"]:
            return
        try:
            control = read(path)
        except (json.JSONDecodeError, PermissionError):
            # A writer without atomic replace, or (Windows) a replace in
            # progress; retry on the next tick.
            return
        layout_state["pending"] = bool(control.get("conversion_pending"))
        layout_state["assist"] = control.get("assist")
        # Decider pre-tilt: long requests avoid these pools (replaced whole,
        # read by the arrival thread).
        layout_state["tilt"] = frozenset(
            alias_of.get(i, i) for i in control.get("tilt") or ()
        )
        changes, deferred = [], []
        for row in control["instances"]:
            if row.get("legacy_pool") == "tp4" and row["instance_id"] not in alias_of:
                # S5b: the conversion's new TP-k takes the absent pair target.
                if not args.deferred_migration or "tp4" in tp_of:
                    raise ValueError("layout control cannot add a pair target")
                alias_of[row["instance_id"]] = "tp4"
                instance_of["tp4"] = row["instance_id"]
                layout_state["migration_target"] = "tp4"
            pool_id = alias_of.get(row["instance_id"], row["instance_id"])
            state = row["lifecycle"]
            if row.get("migration_target") and state == "ACTIVE":
                layout_state["migration_target"] = pool_id
            if state == "RETIRED" and pool_id == layout_state["migration_target"]:
                layout_state["migration_target"] = None
            if state not in {"ACTIVE", "QUIESCING", "RETIRED"}:
                raise ValueError("layout control lifecycle " + state)
            if pool_id not in tp_of:
                if state != "ACTIVE":
                    raise ValueError("layout control adds instances as ACTIVE")
                if (
                    not load("pressure_inventory_process").POOL_ID.fullmatch(pool_id)
                    or row["tp_degree"]
                    not in {1, getattr(controller, "target_tp_size", 4)}
                    or row["context_limit"]
                    != (
                        args.source_context_limit
                        if row["tp_degree"] == 1
                        else args.target_context_limit
                    )
                ):
                    raise ValueError("invalid layout control instance " + pool_id)
                tp = row["tp_degree"]
                tp_of[pool_id] = tp
                profile_key[pool_id] = "tp1" if tp == 1 else "tp4"
                context_of[pool_id] = row["context_limit"]
                service.add_pool(pool_id)
                if health_tracker is not None:
                    health_tracker.add_pool(pool_id, "source" if tp == 1 else "target")
                growth_readers[pool_id] = health.NativeDecodeGrowth()
                ingress.urls[pool_id] = row["endpoint"]
                add_slot(
                    pool_id,
                    Path(row["inventory_path"]),
                    row["endpoint"],
                    tp,
                    initial=False,
                    metrics=cfg.health_service,
                )
                if row.get("migration_channel") and tp == 1:
                    migration_sources.append(
                        (
                            pool_id,
                            Path(row["inventory_path"]),
                            predictor_for(row["predictor_events"]),
                        )
                    )
                elif row.get("migration_channel"):
                    # Same-TP source: only a conversion retiring it may migrate
                    # from it, so the migration policy never sees this list.
                    evacuation_sources.append(
                        (pool_id, Path(row["inventory_path"]), NoPrediction())
                    )
                changes.append(dict(pool_id=pool_id, change="ADDED"))
            # The file may skip a step between polls (ACTIVE straight to
            # RETIRED); walk the coordinator's transitions in order.
            # Re-reading a layout is idempotent; a retired id never comes back
            # (a rebuilt engine gets a new generation in its id).
            current = co.lifecycle(pool_id)
            if current == "RETIRED":
                if state != "RETIRED":
                    raise ValueError("layout control revives retired " + pool_id)
                continue
            steps = {"ACTIVE": ["ACTIVE"], "QUIESCING": ["QUIESCING"]}.get(
                state, ["QUIESCING", "RETIRED"]
            )
            # Its inventory is the only completion evidence for admissions
            # still open there, and the coordinator refuses to retire an
            # instance an unsettled migration names (a QUIESCING source may
            # still migrate out): stay QUIESCING until both settle.
            if state == "RETIRED" and (
                unsettled_on(co.ledger.live_rows(), pool_id)
                or any(
                    pool_id in (target, co._session_source.get(session))
                    and co.ledger.entries[session].state == "ACTIVE"
                    for session, (_, target) in tuple(co._sessions.items())
                )
            ):
                steps, state = ["QUIESCING"], "QUIESCING"
                deferred.append(pool_id)
            for step in steps:
                if co.lifecycle(pool_id) != step:
                    co.set_lifecycle(pool_id, step)
                    changes.append(dict(pool_id=pool_id, change=step))
            if state == "RETIRED" and pool_id in slots:
                remove_slot(pool_id)
                migration_sources[:] = [m for m in migration_sources if m[0] != pool_id]
                evacuation_sources[:] = [
                    m for m in evacuation_sources if m[0] != pool_id
                ]
        # A deferred retirement re-reads the same file on the next tick.
        layout_state.update(
            signature=None if deferred else signature,
            revision=control.get("revision"),
        )
        if changes or deferred != layout_state.get("deferred"):
            audit.write(
                dict(
                    kind="pressure_layout_applied",
                    unix_s=time.time(),
                    revision=control.get("revision"),
                    changes=changes,
                    retire_deferred=deferred,
                )
            )
        layout_state["deferred"] = deferred

    try:
        for path, pool_id, url, tp in (
            *(
                row
                for row in (
                    (args.source_inventory, source_pool, controller.source_url, 1),
                    (args.target_inventory, target_pool, controller.target_url, 2),
                )
                if row[1] in pair
            ),
            *(
                (
                    Path(row["inventory_path"]),
                    row["instance_id"],
                    row["endpoint"],
                    row["tp_degree"],
                )
                for row in extras
            ),
        ):
            add_slot(pool_id, path, url, tp, initial=True)

        def submit_arrival(job, pools):
            submitted = {}
            request = job["request"]
            tilt = layout_state["tilt"]
            if (
                tilt
                and job.get("route") is None
                and len(request["prompt"])
                + (
                    0
                    if getattr(args, "hold_unstarted", 0)
                    else int(request["max_tokens"])
                )
                >= decision.LONG_TOKENS
            ):
                # Keep new long requests off a bundle the decider expects to
                # convert, while any other instance remains.
                kept = [p for p in pools if p.pool_id not in tilt]
                if any(co.lifecycle(p.pool_id) == "ACTIVE" for p in kept):
                    pools = kept
                    layout_state["tilted"] += 1
            dispatch_due(
                [job],
                submitted,
                time.monotonic() - started,
                ingress,
                pools,
                tokens,
                audit,
                origin_unix_s=origin_unix_s,
            )
            return submitted[job["id"]]

        dispatcher = load("pressure_dispatch").ScheduledDispatch(
            jobs,
            submit_arrival,
            unknown_routing_pools(
                args, controller.block_size, extras, pair, pool_names
            ),
            started,
            args.max_seconds,
            hold=load("pressure_hold").HoldQueue(
                args.hold_unstarted,
                args.hold_long_wait_s,
                decision.LONG_TOKENS,
                cfg.max_age_s,
                lambda pool_id: co.lifecycle(pool_id) == "ACTIVE",
                degree=tp_of.get,
                tp1_long_kv=getattr(args, "hold_tp1_long_kv", 0.0),
                unified_guard_blocks=r.unified.GUARD_BLOCKS
                if getattr(args, "scheduler", "pressure") == "unified"
                else None,
            )
            if getattr(args, "hold_unstarted", 0)
            else None,
        )
        dispatcher.start()
        tracker, due_jobs = StreamTracker(), DueJobs(jobs)
        if cfg.health_service:
            for slot in slots.values():
                slot["metrics"] = metrics_for(slot["scraper"])
                background_metrics.append(slot["metrics"])
        while time.monotonic() - started < args.max_seconds:
            load("pause_diagnostics").tick()
            apply_layout_control()
            dispatched = dispatcher.snapshot()
            tracker.update(dispatched, session)
            now, elapsed = time.time(), time.monotonic() - started
            readers = [slot["reader"] for slot in slots.values()]
            polled = {pool_id: s["reader"].poll(now) for pool_id, s in slots.items()}
            publish_layout_status(now, polled, dispatched)
            for reader in readers:
                diagnostic = reader.diagnostics()
                status_key = (
                    reader.error,
                    reader.deferred.get("generation") if reader.deferred else None,
                    reader.latest is None,
                )
                if last_inventory_status.get(reader.pool_id) != status_key:
                    audit.write(dict(kind="pressure_inventory_status", **diagnostic))
                    last_inventory_status[reader.pool_id] = status_key
                if reader.error:
                    raise RuntimeError(
                        "inventory evidence invalid: " + json.dumps(diagnostic)
                    )
            if any(polled[p] is None for p, s in slots.items() if s["initial"]):
                dispatcher.publish(
                    unknown_routing_pools(
                        args, controller.block_size, extras, pair, pool_names
                    )
                )
                save_progress(
                    progress_path,
                    stage="waiting_inventory",
                    elapsed_s=elapsed,
                    submitted=len(dispatched),
                    completed=completed_streams(dispatched, session),
                    total=len(jobs),
                    visible_tokens=sum(
                        v.recorder.progress()[0] for v in dispatched.values()
                    ),
                    inventory_status=[reader.diagnostics() for reader in readers],
                    reason="requests routed; migration waiting for valid inventory",
                )
                watch.update(
                    time.monotonic(),
                    workload_progress(
                        dispatched,
                        last_inventory_progress,
                        session,
                        co.ledger.live_rows(),
                    ),
                    work_due=any(elapsed >= job["at_s"] for job in jobs),
                    reason="waiting for inventory reconciliation; "
                    + json.dumps([reader.diagnostics() for reader in readers]),
                )
                time.sleep(args.tick_s)
                continue
            live = [p for p in slots if polled[p] is not None]
            readers = [slots[p]["reader"] for p in live]
            inventories = [polled[p] for p in live]
            pools, shapes = [], []
            for pool_id in live:
                reader, raw = slots[pool_id]["reader"], polled[pool_id]
                scraper, sampler = slots[pool_id]["scraper"], slots[pool_id]["metrics"]
                if raw["block_size"] != controller.block_size:
                    raise ValueError(
                        "live KV block geometry differs from pilot contract"
                    )
                if raw["context_limit"] != context_of[raw["pool_id"]]:
                    raise ValueError("live context limit differs from serving contract")
                try:
                    if cfg.health_service:
                        sampler.update_inventory(raw)
                        metrics = sampler.snapshot(time.time(), cfg.max_age_s)
                    else:
                        metrics = load("pause_diagnostics").call(
                            "metrics.scrape", o.scrape_inventory_metrics, scraper, raw
                        )
                except load("telemetry").TelemetryError as error:
                    metrics = None
                    if now - last_metrics_error.get(raw["pool_id"], -math.inf) >= 5:
                        audit.write(
                            dict(
                                kind="pressure_metrics_error",
                                pool=raw["pool_id"],
                                error=str(error),
                                action="no heartbeat; do not refresh stale inventory",
                            )
                        )
                        last_metrics_error[raw["pool_id"]] = now
                now = time.time()
                measured = service.snapshot(raw["pool_id"], now)
                growth = None
                prior = previous.get(raw["pool_id"])
                if prior and raw["sampled_at"] > prior["sampled_at"]:
                    dt = raw["sampled_at"] - prior["sampled_at"]
                    if dt <= cfg.max_age_s:
                        growth = max(
                            0, (prior["free_blocks"] - raw["free_blocks"]) / dt
                        )
                previous[raw["pool_id"]] = raw
                if cfg.capacity_lookahead_s > 0:
                    growth = growth_readers[raw["pool_id"]].update(raw)
                    if (
                        not cfg.health_service
                        or now - last_metrics_logged.get(raw["pool_id"], -math.inf) >= 1
                    ):
                        last_metrics_logged[raw["pool_id"]] = now
                        audit.write(
                            dict(
                                kind="pressure_metrics_sample",
                                unix_s=time.time(),
                                pool=raw["pool_id"],
                                generation=raw["generation"],
                                metrics=metrics,
                            )
                        )
                pool = o.pool_reading(
                    raw,
                    reader.observed_at,
                    measured,
                    cfg,
                    now,
                    growth_blocks_s=growth,
                    heartbeat=metrics,
                )
                if health_tracker is not None:
                    health_tracker.observe(raw, now)
                    pool = replace(
                        pool, **health_tracker.snapshot(raw["pool_id"], now, pool.idle)
                    )
                if workload_module is not None:
                    pool = replace(
                        pool,
                        workload=workload_module.observe(
                            workload_profiles[profile_key[raw["pool_id"]]],
                            raw | {"sampled_at": pool.sampled_at},
                            now,
                        ),
                    )
                if cfg.capacity_lookahead_s > 0:
                    pool = replace(
                        pool,
                        unallocated_prompt_blocks=health.unallocated_prompt_blocks(
                            raw["requests"], raw["block_size"]
                        ),
                    )
                pools.append(pool)
                shapes.append(batch(raw, now, cfg, pool))
                if 0 <= now - pool.sampled_at <= cfg.max_age_s:
                    ingress.reconcile(raw | {"sampled_at": pool.sampled_at}, now)
            now = time.time()
            if co.scheduler == "unified":
                co.observe_itl(service, [pool.pool_id for pool in pools], now)
            # Request rates of every migration source (client ids are unique).
            source_rates = {}
            for source_id, _, _ in migration_sources:
                source_rates.update(service.snapshot(source_id, now)["request_rates"])
            if source_rates:
                horizon = min(
                    8192,
                    cfg.survival_horizon
                    + math.ceil(lookup.release_budget * max(source_rates.values())),
                )
                if horizon != co.policy.cfg.survival_horizon:
                    co.set_config(replace(cfg, survival_horizon=horizon))
                    audit.write(
                        {
                            "kind": "pressure_cost_horizon",
                            "unix_s": now,
                            "config": asdict(co.policy.cfg),
                            "release_budget_s": lookup.release_budget,
                            "observed_source_request_rates": source_rates,
                            "scope": "empirical budget plus configured recovery tokens",
                        }
                    )
            dispatcher.publish(pools)
            dispatched = dispatcher.snapshot()
            index_of = {pool.pool_id: n for n, pool in enumerate(pools)}
            # Migration needs the pair target and a live source; retired or
            # not-yet-observed instances simply drop out (S3b-1).
            target_pool_id = layout_state["migration_target"]
            t = index_of.get(target_pool_id)
            if (
                t is not None
                and session.ready_for_start
                and hasattr(session, "bind_target")
            ):
                session.bind_target(
                    target_pool_id,
                    ingress.urls[target_pool_id],
                    instance_of.get(target_pool_id, target_pool_id),
                )
            candidates_by_source, runtimes, rates_by_source = {}, {}, {}
            # Published (next tick) with each request's planner row.
            layout_state["candidates"] = {}
            for source_id, inventory_path, source_predictor in (
                migration_sources + evacuation_sources
            ):
                if t is None or source_id not in index_of:
                    continue
                n = index_of[source_id]
                rates = service.snapshot(source_id, now)["request_rates"]
                rates_by_source[source_id] = rates
                candidates, eos = [], {}
                for row in inventories[n]["requests"]:
                    try:
                        stream = ingress.for_server_id(row["request_id"])
                    except ValueError:
                        stream = None
                    if workload_module is not None:
                        relief, cost = (
                            tuple(
                                workload_module.slot_cost(
                                    workload_profiles[name],
                                    max(row["prompt_tokens"], row["computed_tokens"]),
                                )[0]
                                for name in ("tp1", "tp4")
                            )
                            if all(
                                workload_module.valid(pool, co.policy.cfg, now)
                                for pool in (pools[n], pools[t])
                            )
                            else (None, None)
                        )
                    else:
                        relief, cost = (
                            lookup.deltas(shapes[n], shapes[t], row["request_id"])
                            if shapes[n] and shapes[t]
                            else (None, None)
                        )
                    release_budget = lookup.release_budget
                    candidates.append(
                        p.Candidate(
                            row["request_id"],
                            co._owners.get(row["request_id"], source_id),
                            inventories[n]["sampled_at"],
                            readers[n].observed_at,
                            row["generated"],
                            row["prompt_tokens"],
                            row["max_output_tokens"],
                            row["releasable_blocks"],
                            row["owned_blocks"],
                            row["owned_blocks"]
                            * controller.block_size
                            * controller.policy.kv_bytes_per_token
                            if row["owned_blocks"] is not None
                            else None,
                            None,
                            "UNAVAILABLE",
                            relief,
                            cost,
                            release_budget,
                            source_tokens_s=(
                                rates.get(stream.client_id)
                                if stream is not None
                                else None
                            ),
                            finished=stream.future.done()
                            if stream is not None
                            else False,
                            action_available=stream is not None
                            and session.ready_for_start,
                            inventory_reference=(
                                f"{inventory_path}:{inventories[n]['generation']}"
                            ),
                            migrating=bool(
                                session.record
                                and session.record.request_id == row["request_id"]
                            ),
                            source_tp=tp_of[source_id],
                            target_health_after_supported=(
                                health_module.supported_after(
                                    health_tracker.reference,
                                    shapes[t]["decode_requests"] + 1,
                                    (
                                        shapes[t]["decode_context_sum"]
                                        + row["computed_tokens"]
                                    )
                                    / (shapes[t]["decode_requests"] + 1),
                                )
                                if health_tracker is not None and shapes[t]
                                else False
                            ),
                        )
                    )
                    eos[row["request_id"]] = row["ignore_eos"]
                candidates_by_source[source_id] = load("pause_diagnostics").call(
                    "predictor.read",
                    co.with_predictions,
                    candidates,
                    source_predictor,
                    now,
                    (pools[n].free_blocks - cfg.capacity_guard_blocks)
                    * pools[n].block_size,
                    eos,
                )
                layout_state["candidates"].update(
                    (c.request_id, c) for c in candidates_by_source[source_id]
                )
                growth = pools[n].growth_blocks_s
                guard_time = (
                    max(0, pools[n].free_blocks - cfg.capacity_guard_blocks) / growth
                    if growth is not None and growth > 0
                    else None
                )
                pools[n] = replace(
                    pools[n],
                    guard_time_s=guard_time,
                    urgency=lookup.release_budget / guard_time
                    if guard_time is not None and guard_time > 0
                    else None,
                )
                runtimes[source_id] = load("manager_m0").RuntimeSnapshot(
                    unix_s=now,
                    migration_id=session.migration_id,
                    request_id=None,
                    state="LOCAL",
                    generated_tokens=None,
                    source_sampled_unix_s=pools[n].sampled_at,
                    target_sampled_unix_s=pools[t].sampled_at,
                    source_free_kv_tokens=pools[n].free_blocks * pools[n].block_size,
                    source_guard_free_kv_tokens=cfg.capacity_guard_blocks
                    * pools[n].block_size,
                    source_pool_growth_tokens_s=pools[n].growth_blocks_s
                    * pools[n].block_size
                    if pools[n].growth_blocks_s is not None
                    else None,
                    target_running=inventories[t]["running"],
                    target_waiting=co.ledger.routing_reading(pools[t]).waiting,
                    target_kv_usage_frac=1
                    - pools[t].free_blocks / inventories[t]["total_blocks"],
                )
            candidates = [c for cs in candidates_by_source.values() for c in cs]
            # A source without observed request rates may not start a migration;
            # with one source this is the historical "no rates, advisory" rule.
            actions = (
                None
                if args.no_migration
                or not session.ready_for_start
                or not any(rates_by_source.values())
                else {
                    c.request_id: r.BoundAction(
                        c.request_id,
                        session.migration_id,
                        lambda c=c, runtime=runtimes[source], source=source: (
                            note_migration_start(source, session.start(c, runtime))
                        ),
                    )
                    for source, cs in candidates_by_source.items()
                    if rates_by_source[source]
                    for c in cs
                }
            )
            assist = layout_state["assist"]
            assist_started = False
            if (
                t is not None
                and assist is not None
                and assist["seq"] != layout_state["assist_seen"]
            ):
                layout_state["assist_seen"] = assist["seq"]
                reason = start_assist(
                    assist, now, pools[t], candidates_by_source, actions
                )
                assist_started = reason is None
                layout_state["assist_result"] = dict(
                    seq=assist["seq"],
                    request_id=assist["request_id"],
                    state="STARTED" if assist_started else "REFUSED",
                    reason=reason,
                    unix_s=now,
                )
                audit.write(
                    dict(kind="pressure_assist_result", **layout_state["assist_result"])
                )
            if t is not None and not assist_started:
                load("pause_diagnostics").call(
                    "policy.tick",
                    co.tick_multi,
                    now,
                    {target_pool_id: pools[t]}
                    | {
                        source: pools[index_of[source]]
                        for source in policy_sources(
                            candidates_by_source, evacuation_sources
                        )
                    },
                    tp_of,
                    {
                        (source, target_pool_id): cs
                        for source, cs in policy_sources(
                            candidates_by_source, evacuation_sources
                        ).items()
                    },
                    actions,
                )
            session_source = co._session_source.get(session.migration_id, source_pool)
            runtime = runtimes.get(session_source)
            if t is not None and session.record is not None and not session.settled:
                session.poll_cancel(inventories[t], now, cfg.max_age_s)
                if args.test_only_cancel_first_shadow:
                    from tools.bridge_tp.pressure_health_screen import (
                        inject_cancel_smoke,
                    )

                    inject_cancel_smoke(session, inventories[t], now, audit)
                # "Source recovered" ends a relief migration; a drain assist
                # exists precisely because its quiescing source is idle
                # (S5a GPU 11-01: it cancelled 2 of 3 assists).
                if session.migration_id not in co._assist_sessions:
                    session.reconsider(
                        now,
                        co.ledger.routing_reading(pools[index_of[session_source]]),
                        co.ledger.routing_reading(pools[t]),
                        next(
                            (
                                c
                                for c in candidates
                                if c.request_id == session.record.request_id
                            ),
                            None,
                        ),
                        co.policy.cfg,
                        target_inventory=inventories[t],
                    )
                progress = session.old.read_source_progress(
                    session.run_dir, session.stream.payload, now
                )
                if (
                    progress is not None
                    and progress.request_id == session.record.request_id
                ):
                    last_view = progress
                if last_view is not None:
                    load("pause_diagnostics").call(
                        "migration.tick", session.tick, last_view, runtime
                    )
                elif session.stream.future.done():
                    session.cancel(
                        "source ended before selected progress was published"
                    )
            for previous_session in session.sessions if t is not None else ():
                if previous_session.record is None:
                    continue
                entry = co.ledger.entries[previous_session.migration_id]
                if previous_session.target_future is not None and (
                    previous_session.target_future.done()
                ):
                    previous_session.target_future.result()
                    if entry.state in {"COMMITTED", "ACCOUNTED", "COMPLETED"}:
                        service.finish(
                            previous_session.stream.recorder.target_service_id
                        )
                prefix = (
                    f"cmpl-bridgetp-phase9-target-{previous_session.run_dir.name}-0-"
                )
                targets = [
                    row
                    for row in inventories[t]["requests"]
                    if row["request_id"].startswith(prefix)
                ]
                if len(targets) > 1:
                    raise ValueError("ambiguous target inventory identity")
                if (
                    targets
                    and entry.state in {"ACTIVE", "COMMITTED"}
                    and 0 <= now - pools[t].sampled_at <= cfg.max_age_s
                    and targets[0]["owned_blocks"] is not None
                ):
                    co.observe_migration(
                        entry.key,
                        inventories[t]["pool_id"],
                        allocated_blocks=acknowledged_blocks(
                            entry, targets[0]["owned_blocks"]
                        ),
                        service_observed=entry.service_observed
                        or targets[0]["generated"] >= 2,
                        prefill_observed=True,
                    )
                if (
                    not targets
                    and previous_session.settled
                    and previous_session.target_future is not None
                    and previous_session.target_future.done()
                    and not previous_session.target_future.exception()
                    and pools[t].sampled_at
                    > previous_session.target_future.result()["completed_unix_s"]
                    and entry.state in {"COMMITTED", "ACCOUNTED"}
                ):
                    with co._lock:
                        co.ledger.complete(entry.key, evidence=True)
            if session.rotate():
                last_view = None
            tracker.update(dispatched, session)
            due_queued = due_jobs.queued(elapsed, dispatched)
            completed = tracker.completed(session)
            last_inventory_progress = inventory_progress(inventories)
            marker = workload_progress(
                tracker.open,
                last_inventory_progress,
                session,
                co.ledger.live_rows(),
            )
            pressures = [
                p.classify(co.ledger.routing_reading(pool), co.policy.cfg, now)
                for pool in pools
            ]
            by_pool = dict(zip(index_of, pressures))
            # Per-instance pressure for the conversion's gate and safety valve.
            layout_state["pressure"] = {
                pool: asdict(value) for pool, value in by_pool.items()
            }
            source_p, target_p = by_pool.get(source_pool), by_pool.get(target_pool_id)
            if time.monotonic() - last_status_at >= 5:
                last_status_at = time.monotonic()
                status = dict(
                    stage="running" if len(dispatched) > completed else "queued",
                    elapsed_s=round(elapsed, 1),
                    submitted=len(dispatched),
                    completed=completed,
                    total=len(jobs),
                    queued=due_queued,
                    source_zone=source_p.zone if source_p else None,
                    target_zone=target_p.zone if target_p else None,
                    source_missing=source_p.missing if source_p else ["not serving"],
                    target_missing=target_p.missing if target_p else ["not serving"],
                    active_generated=sum(row[2] for row in marker[2]),
                    migration=marker[3],
                    migrations_started=sum(
                        s.record is not None for s in session.sessions
                    ),
                    migrations_released=sum(
                        s.settled
                        and co.ledger.entries[s.migration_id].state
                        in {"COMMITTED", "ACCOUNTED", "COMPLETED"}
                        for s in session.sessions
                    ),
                    migrations_cancelled=sum(
                        s.record is not None
                        and co.ledger.entries[s.migration_id].state == "CANCELLED"
                        for s in session.sessions
                    ),
                    migration_cancelling=session.cancel_future is not None
                    and not session.settled,
                    engine_waiting={pool.pool_id: pool.waiting for pool in pools},
                    unobserved_ingress={
                        pool.pool_id: co.ledger.unobserved_ingress(pool.pool_id)
                        for pool in pools
                    },
                    prefill={
                        pool.pool_id: dict(
                            step_tokens=pool.prefill_batch_tokens,
                            backlog_tokens=pool.pending_prompt_tokens,
                            waiting=pool.waiting,
                        )
                        for pool in pools
                    },
                )
                if cfg.health_service:
                    status["health"] = {
                        pool.pool_id: dict(
                            status=pool.health_status,
                            eta=pool.health_eta,
                            load=pool.health_load,
                            reason=pool.health_reason,
                        )
                        for pool in pools
                    }
                if cfg.workload_pressure:
                    status["routing_relief"] = co.policy.relief.snapshot()
                    status["workload_pressure"] = {
                        pool.pool_id: dict(
                            status=(pool.workload or {}).get("status", "UNKNOWN"),
                            reason=(pool.workload or {}).get("reason"),
                            load=(pool.workload or {}).get("load"),
                            prefill_s=(pool.workload or {}).get("prefill_s"),
                            kv_pressure=pressures[index].capacity,
                        )
                        for index, pool in enumerate(pools)
                    }
                save_progress(progress_path, **status)
                audit.write(dict(kind="pressure_progress", **status))
            watch.update(
                time.monotonic(),
                (tracker.retired, len(dispatched), marker),
                work_due=bool(
                    due_queued
                    or len(dispatched) > completed
                    or any(
                        row.state in {"ACTIVE", "COMMITTED", "ACCOUNTED"}
                        and not (
                            session.cleanup_requires_shutdown
                            and row.kind == "migration"
                        )
                        for row in co.ledger.live_rows()
                    )
                    or session.record
                    and not session.settled
                    and not session.cleanup_requires_shutdown
                ),
                reason=json.dumps(
                    dict(
                        queued=due_queued,
                        source=source_p.zone if source_p else None,
                        target=target_p.zone if target_p else None,
                    )
                ),
            )
            if (
                len(dispatched) == len(jobs)
                and tracker.all_done()
                # A live conversion keeps the controller serving until it ends.
                and not layout_state["pending"]
                and not any(
                    row.state in {"ACTIVE", "COMMITTED", "ACCOUNTED"}
                    and not (
                        session.cleanup_requires_shutdown and row.kind == "migration"
                    )
                    for row in co.ledger.live_rows()
                )
                and all(
                    previous_session.record is None
                    or (
                        (
                            previous_session.settled
                            or previous_session.cleanup_requires_shutdown
                        )
                        and (
                            previous_session.target_future is None
                            or previous_session.target_future.done()
                        )
                    )
                    for previous_session in session.sessions
                )
            ):
                # Layout additions use asynchronous inventory readers. An
                # applied ACTIVE row alone does not prove they have observed
                # the serving engine, especially after the last request.
                if any(
                    polled.get(pool) is None
                    for pool in slots
                    if co.lifecycle(pool) == "ACTIVE"
                ):
                    time.sleep(args.tick_s)
                    continue
                # A conversion can publish its last revision immediately after
                # a periodic status sample. Flush the applied layout before
                # exit, retrying next tick if a Windows reader holds the file.
                if not publish_layout_status(
                    time.time(), polled, dispatched, force=True
                ):
                    time.sleep(args.tick_s)
                    continue
                outcome = (
                    "COMPLETED_PENDING_OWNER_CLEANUP"
                    if session.cleanup_requires_shutdown
                    else "COMPLETED"
                )
                return
            time.sleep(args.tick_s)
        raise TimeoutError(
            "pilot timed out; preserve inventory/audit and reconcile active promises"
        )
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        dispatch_stop_error = None
        if dispatcher is not None:
            try:
                dispatcher.close()
            except RuntimeError as error:
                dispatch_stop_error = error
                failure = failure or repr(error)
                outcome = "FAILED_OR_INTERRUPTED"
            dispatched = dispatcher.snapshot(check=False)
        close_errors = []
        for sampler in background_metrics + background_inventory:
            sampler.stop.set()
        for sampler in background_metrics + background_inventory:
            try:
                sampler.close()
            except RuntimeError as error:
                close_errors.append(error)
        if close_errors:
            failure = failure or repr(close_errors[0])
            outcome = "FAILED_OR_INTERRUPTED"
        save_progress(
            progress_path,
            stage=outcome,
            error=failure,
            submitted=len(dispatched),
            completed=completed_streams(dispatched, session),
            total=len(jobs),
            elapsed_s=round(time.monotonic() - started, 1),
        )
        results = []
        for job_id, value in dispatched.items():
            results.append(
                dict(
                    job_id=job_id,
                    client_id=value.client_id,
                    server_id=value.server_id,
                    initial_pool=value.pool_id,
                    request=value.payload,
                    source_done=value.future.done(),
                    source_error=str(value.future.exception())
                    if value.future.done() and value.future.exception()
                    else None,
                    response=value.recorder.stats(),
                    source_transport={
                        key: value.future.result().get(key)
                        for key in (
                            "request_started_unix_s",
                            "completed_unix_s",
                            "ttft_ms",
                            "e2e_ms",
                            "finish_reason",
                        )
                    }
                    if value.future.done() and value.future.exception() is None
                    else None,
                    dispatch_timing=dict(
                        planned_unix_s=origin_unix_s
                        + next(job["at_s"] for job in jobs if job["id"] == job_id),
                        dispatch_entered_unix_s=value.dispatch_entered_at,
                        admission_finished_unix_s=value.admission_finished_at,
                        executor_submitted_unix_s=value.executor_submitted_at,
                        http_thread_started_unix_s=value.http_thread_started_at,
                        held_s=value.held_s,
                    ),
                )
            )
        (args.run_dir / "pressure_request_results.json").write_text(
            json.dumps(results), encoding="utf-8"
        )
        audit.write(
            dict(
                kind="pressure_final",
                reservations=co.ledger.snapshot(),
                dispatched=list(dispatched),
                elapsed_s=time.monotonic() - started,
                session_settled=session.settled,
                maximum_migrations=args.maximum_migrations,
                sessions=[
                    dict(
                        migration_id=s.migration_id,
                        started=s.record is not None,
                        settled=s.settled,
                        run_dir=str(s.run_dir),
                    )
                    for s in session.sessions
                ],
                outcome=outcome,
                error=failure,
                shutdown_cleanup_required=session.cleanup_requires_shutdown,
            )
        )
        executor.shutdown(wait=False, cancel_futures=True)
        if all(v.future.done() for v in dispatched.values()) and (
            all(
                s.target_future is None or s.target_future.done()
                for s in session.sessions
            )
        ):
            tokens.close()
        else:
            atexit.register(tokens.close)
        try:
            session.close()
        finally:
            audit.close()
        if close_errors:
            raise RuntimeError("background observation shutdown failed") from (
                close_errors[0]
            )
        if dispatch_stop_error is not None:
            raise dispatch_stop_error


def main(argv=None):
    args = parse_args(argv)
    prepared = preflight(args)
    if args.validate_only:
        print(json.dumps(prepared[-1], indent=2))
        return 0
    run(args, prepared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
