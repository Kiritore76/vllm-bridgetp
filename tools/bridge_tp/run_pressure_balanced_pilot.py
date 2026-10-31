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
from tools.bridge_tp import pressure_health_reference as health  # noqa: E402
from tools.bridge_tp.pressure_cpu import load  # noqa: E402
from tools.bridge_tp.pressure_progress import (  # noqa: E402
    ProgressWatch,
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
    parser.add_argument("--test-only-cancel-first-shadow", action="store_true")
    args = parser.parse_args(argv)
    if args.maximum_migrations < 0:
        parser.error("maximum migrations must be nonnegative")
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
        target_tp_size=load("tp_geometry").validate_target_tp(
            raw.get("target_tp_size", 4)
        ),
        policy=SimpleNamespace(kv_bytes_per_token=raw["policy"]["kv_bytes_per_token"]),
        ready_notification_mode=raw.get("ready_notification_mode", "FILE_POLL"),
        ready_notification_port=raw.get("ready_notification_port", 0),
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
            or job.get("route") not in {None, "tp1", "tp4"}
        ):
            raise ValueError("invalid/duplicate arrival or stop contract")
        ids.add(job["id"])
        limits = dict(tp1=args.source_context_limit, tp4=args.target_context_limit)
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
    manifest = instance_manifest(args, controller)
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
    """Fixed migration pair (tp1/tp4 aliases) plus optional routing-only instances.

    Only the aliased pair has a migration channel and session; other instances
    serve routed requests and never migrate (multi-source channels: S1-d2).
    """
    if args.instances is None:
        return None
    manifest = read(args.instances)
    aliases = manifest.get("legacy_pool_aliases", {})
    rows = {row["instance_id"]: row for row in manifest.get("instances", ())}
    if (
        manifest.get("schema") != "BRIDGETP_INSTANCE_MANIFEST_V1"
        or set(aliases) != {"tp1", "tp4"}
        or not set(aliases.values()) <= set(rows)
        or rows[aliases["tp1"]]["tp_degree"] != 1
        or rows[aliases["tp4"]]["tp_degree"] != controller.target_tp_size
        or rows[aliases["tp1"]]["context_limit"] != args.source_context_limit
        or rows[aliases["tp4"]]["context_limit"] != args.target_context_limit
    ):
        raise ValueError("controller needs one TP1 source and one TP-k target pair")
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


def channel_sources(manifest):
    """Routing-only TP1 rows that also own a migration channel (S1-d2c)."""
    return [r for r in extra_instances(manifest) if r.get("migration_channel")]


def extra_instances(manifest):
    """Routing-only instances: every manifest row outside the migration pair."""
    if manifest is None:
        return []
    paired = set(manifest["legacy_pool_aliases"].values())
    return [r for r in manifest["instances"] if r["instance_id"] not in paired]


def unknown_routing_pools(args, block_size, extras=()):
    """Use configured context contracts while pressure telemetry is absent."""
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
            ("tp1", args.source_context_limit),
            ("tp4", args.target_context_limit),
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
                ingress_capacity_gate=False,
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
    tp_of = {"tp1": 1, "tp4": getattr(controller, "target_tp_size", 4)}
    extras = extra_instances(receipt.get("instance_manifest"))
    tp_of.update({row["instance_id"]: row["tp_degree"] for row in extras})
    # Workload/health references exist per TP class, not per instance.
    profile_key = {pool: ("tp1" if tp == 1 else "tp4") for pool, tp in tp_of.items()}
    context_of = {
        "tp1": args.source_context_limit,
        "tp4": args.target_context_limit,
        **{row["instance_id"]: row["context_limit"] for row in extras},
    }
    for row in extras:
        service.add_pool(row["instance_id"])
    ingress = i.PressureIngress(
        co,
        service,
        executor,
        controller.source_url,
        controller.target_url,
        reconcile_log_interval_s=1.0 if cfg.health_service else 0,
        tp_of=tp_of,
    )
    ingress.urls.update({row["instance_id"]: row["endpoint"] for row in extras})
    session = PressureSessions(
        persistent_cleanup=cfg.shadow_cancel,
        maximum_migrations=args.maximum_migrations,
        run_dir=args.run_dir,
        migration_id=args.migration_id,
        config=controller,
        coordinator=co,
        ingress=ingress,
        executor=executor,
        audit=audit,
    )
    scrapers = tuple(
        load("telemetry").MetricsScraper(
            url,
            controller.block_size,
            total,
            timeout_s=0.75 if cfg.health_service else 2.0,
        )
        for url, total in (
            (controller.source_url, controller.tp1_total_kv_blocks),
            (controller.target_url, controller.tp4_total_kv_blocks),
            *(
                (
                    row["endpoint"],
                    controller.tp1_total_kv_blocks
                    if row["tp_degree"] == 1
                    else controller.tp4_total_kv_blocks,
                )
                for row in extras
            ),
        )
    )
    predictor = load("manager_m5").PredictorEventReader(
        args.predictor_events,
        args.predictor_sha256,
        align_history=getattr(args, "platform", "a100") == "qwen3",
    )
    # Migration sources: the pair source, then every TP1 with its own channel.
    migration_sources = [("tp1", args.source_inventory, predictor)] + [
        (
            row["instance_id"],
            Path(row["inventory_path"]),
            load("manager_m5").PredictorEventReader(
                Path(row["predictor_events"]),
                args.predictor_sha256,
                align_history=getattr(args, "platform", "a100") == "qwen3",
            ),
        )
        for row in channel_sources(receipt.get("instance_manifest"))
    ]
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
        for row in extras:
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
    try:
        for path, pool_id in (
            (args.source_inventory, "tp1"),
            (args.target_inventory, "tp4"),
            *((Path(row["inventory_path"]), row["instance_id"]) for row in extras),
        ):
            background_inventory.append(
                load("pressure_inventory_process").ProcessInventory(
                    path,
                    pool_id,
                    args.run_dir,
                )
            )
        readers = tuple(background_inventory)

        def submit_arrival(job, pools):
            submitted = {}
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
            unknown_routing_pools(args, controller.block_size, extras),
            started,
            args.max_seconds,
        )
        dispatcher.start()
        if cfg.health_service:
            for scraper in scrapers:
                background_metrics.append(
                    load("pressure_io").BackgroundMetrics(
                        lambda raw, scraper=scraper: load("pause_diagnostics").call(
                            "metrics.scrape", o.scrape_inventory_metrics, scraper, raw
                        )
                    )
                )
        while time.monotonic() - started < args.max_seconds:
            load("pause_diagnostics").tick()
            dispatched = dispatcher.snapshot()
            now, elapsed = time.time(), time.monotonic() - started
            inventories = [reader.poll(now) for reader in readers]
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
            if any(v is None for v in inventories):
                dispatcher.publish(
                    unknown_routing_pools(args, controller.block_size, extras)
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
                        tuple(co.ledger.entries.values()),
                    ),
                    work_due=any(elapsed >= job["at_s"] for job in jobs),
                    reason="waiting for inventory reconciliation; "
                    + json.dumps([reader.diagnostics() for reader in readers]),
                )
                time.sleep(args.tick_s)
                continue
            pools, shapes = [], []
            for pool_index, (reader, raw, scraper) in enumerate(
                zip(readers, inventories, scrapers)
            ):
                if raw["block_size"] != controller.block_size:
                    raise ValueError(
                        "live KV block geometry differs from pilot contract"
                    )
                if raw["context_limit"] != context_of[raw["pool_id"]]:
                    raise ValueError("live context limit differs from serving contract")
                try:
                    if cfg.health_service:
                        sampler = background_metrics[pool_index]
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
            candidates_by_source, runtimes, rates_by_source = {}, {}, {}
            for source_id, inventory_path, source_predictor in migration_sources:
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
                                for pool in (pools[n], pools[1])
                            )
                            else (None, None)
                        )
                    else:
                        relief, cost = (
                            lookup.deltas(shapes[n], shapes[1], row["request_id"])
                            if shapes[n] and shapes[1]
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
                                if stream is not None else None
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
                            target_health_after_supported=(
                                health_module.supported_after(
                                    health_tracker.reference,
                                    shapes[1]["decode_requests"] + 1,
                                    (
                                        shapes[1]["decode_context_sum"]
                                        + row["computed_tokens"]
                                    )
                                    / (shapes[1]["decode_requests"] + 1),
                                )
                                if health_tracker is not None and shapes[1]
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
                    target_sampled_unix_s=pools[1].sampled_at,
                    source_free_kv_tokens=pools[n].free_blocks * pools[n].block_size,
                    source_guard_free_kv_tokens=cfg.capacity_guard_blocks
                    * pools[n].block_size,
                    source_pool_growth_tokens_s=pools[n].growth_blocks_s
                    * pools[n].block_size
                    if pools[n].growth_blocks_s is not None
                    else None,
                    target_running=inventories[1]["running"],
                    target_waiting=co.ledger.routing_reading(pools[1]).waiting,
                    target_kv_usage_frac=1
                    - pools[1].free_blocks / inventories[1]["total_blocks"],
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
                        lambda c=c, runtime=runtimes[source]: session.start(
                            c, runtime
                        ),
                    )
                    for source, cs in candidates_by_source.items()
                    if rates_by_source[source]
                    for c in cs
                }
            )
            load("pause_diagnostics").call(
                "policy.tick",
                co.tick_multi,
                now,
                {"tp4": pools[1]}
                | {source: pools[index_of[source]] for source in candidates_by_source},
                tp_of,
                {(source, "tp4"): cs for source, cs in candidates_by_source.items()},
                actions,
            )
            session_source = co._session_source.get(session.migration_id, "tp1")
            runtime = runtimes.get(session_source, runtimes["tp1"])
            if session.record is not None and not session.settled:
                session.poll_cancel(inventories[1], now, cfg.max_age_s)
                if args.test_only_cancel_first_shadow:
                    from tools.bridge_tp.pressure_health_screen import (
                        inject_cancel_smoke,
                    )

                    inject_cancel_smoke(session, inventories[1], now, audit)
                session.reconsider(
                    now,
                    co.ledger.routing_reading(pools[index_of[session_source]]),
                    co.ledger.routing_reading(pools[1]),
                    next(
                        (
                            c
                            for c in candidates
                            if c.request_id == session.record.request_id
                        ),
                        None,
                    ),
                    co.policy.cfg,
                    target_inventory=inventories[1],
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
            for previous_session in session.sessions:
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
                    for row in inventories[1]["requests"]
                    if row["request_id"].startswith(prefix)
                ]
                if len(targets) > 1:
                    raise ValueError("ambiguous target inventory identity")
                if (
                    targets
                    and entry.state in {"ACTIVE", "COMMITTED"}
                    and 0 <= now - pools[1].sampled_at <= cfg.max_age_s
                    and targets[0]["owned_blocks"] is not None
                ):
                    co.observe_migration(
                        entry.key,
                        inventories[1]["pool_id"],
                        allocated_blocks=min(targets[0]["owned_blocks"], entry.blocks),
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
                    and pools[1].sampled_at
                    > previous_session.target_future.result()["completed_unix_s"]
                    and entry.state in {"COMMITTED", "ACCOUNTED"}
                ):
                    with co._lock:
                        co.ledger.complete(entry.key, evidence=True)
            if session.rotate():
                last_view = None
            for value in dispatched.values():
                if value.future.done():
                    value.future.result()
            due_queued = [
                job["id"]
                for job in jobs
                if elapsed >= job["at_s"] and job["id"] not in dispatched
            ]
            completed = completed_streams(dispatched, session)
            last_inventory_progress = inventory_progress(inventories)
            marker = workload_progress(
                dispatched,
                last_inventory_progress,
                session,
                tuple(co.ledger.entries.values()),
            )
            pressures = [
                p.classify(co.ledger.routing_reading(pool), co.policy.cfg, now)
                for pool in pools
            ]
            if time.monotonic() - last_status_at >= 5:
                last_status_at = time.monotonic()
                status = dict(
                    stage="running" if len(dispatched) > completed else "queued",
                    elapsed_s=round(elapsed, 1),
                    submitted=len(dispatched),
                    completed=completed,
                    total=len(jobs),
                    queued=due_queued,
                    source_zone=pressures[0].zone,
                    target_zone=pressures[1].zone,
                    source_missing=pressures[0].missing,
                    target_missing=pressures[1].missing,
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
                marker,
                work_due=bool(
                    due_queued
                    or len(dispatched) > completed
                    or any(
                        row.state in {"ACTIVE", "COMMITTED", "ACCOUNTED"}
                        and not (
                            session.cleanup_requires_shutdown
                            and row.kind == "migration"
                        )
                        for row in tuple(co.ledger.entries.values())
                    )
                    or session.record
                    and not session.settled
                    and not session.cleanup_requires_shutdown
                ),
                reason=json.dumps(
                    dict(
                        queued=due_queued,
                        source=pressures[0].zone,
                        target=pressures[1].zone,
                    )
                ),
            )
            if (
                len(dispatched) == len(jobs)
                and all(v.future.done() for v in dispatched.values())
                and not any(
                    row.state in {"ACTIVE", "COMMITTED", "ACCOUNTED"}
                    and not (
                        session.cleanup_requires_shutdown and row.kind == "migration"
                    )
                    for row in tuple(co.ledger.entries.values())
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
