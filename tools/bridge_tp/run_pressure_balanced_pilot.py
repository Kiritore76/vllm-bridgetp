#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Experimental single-session pressure pilot with multiple existing source streams.

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
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.source_context_limit, args.target_context_limit) <= 0:
        parser.error("positive configured context limits required")
    if any(
        not math.isfinite(v) or v <= 0
        for v in (args.tick_s, args.max_seconds, args.no_progress_s)
    ):
        parser.error("positive tick and runtime required")
    if (
        args.predictor_sha256
        != "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7"
    ):
        parser.error("frozen four-layer76 checkpoint identity required")
    return args


def preflight(args):
    cfg = p.PressureConfig(**read(args.policy_config))
    if cfg.health_service:
        if args.health_reference is None:
            raise ValueError("health service mode requires a frozen V2 reference")
        load("pressure_health").HealthTracker(read(args.health_reference), 0, None)
        if cfg.sustain_s < 10 or not cfg.capacity_lookahead_s:
            raise ValueError(
                "health mode requires 10s persistence and independent KV lookahead"
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
        policy=SimpleNamespace(kv_bytes_per_token=raw["policy"]["kv_bytes_per_token"]),
        ready_notification_mode=raw.get("ready_notification_mode", "FILE_POLL"),
        ready_notification_port=raw.get("ready_notification_port", 0),
    )
    if (
        controller.block_size != 16
        or cfg.capacity_guard_blocks * 16 != 2000
        or controller.tp1_total_kv_blocks <= 0
        or controller.tp4_total_kv_blocks <= 0
        or controller.policy.kv_bytes_per_token <= 0
    ):
        raise ValueError("pilot geometry/2000-token guard contract differs")
    load("manager_m2").M2RateConfig(
        low_bytes_s=controller.rate.b_min_bytes_s,
        medium_bytes_s=controller.rate.b_start_bytes_s,
        high_bytes_s=controller.rate.b_max_bytes_s,
    ).validate()
    evidence = read(args.service_evidence)
    lookup = s.LocalServiceLookup(evidence)
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
        maximum_migrations=1,
        migration_execution_enabled=not args.no_migration,
        mode="CPU_PREFLIGHT",
        service_estimator=(
            "observed engine-span local proxy; causal ITL delta unmeasured"
        ),
        gpu_validated=False,
    )
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


def unknown_routing_pools(args, block_size):
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
            )
        )


def run(args, prepared):
    from tools.bridge_tp.pressure_session import RollingPressureSession
    from vllm.bridge_tp.async_journal import AsyncJournal

    cfg, controller, lookup, jobs, receipt = prepared
    receipt = receipt | {"mode": "ONLINE_FUNCTION_PILOT"}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (args.run_dir / name).exists()
        for name in (
            "pressure_audit.jsonl",
            "native_tokens.jsonl",
            "pressure_request_results.json",
        )
    ):
        raise FileExistsError(
            "pressure pilot restart requires a fresh, reconciled run directory"
        )
    audit = load("audit").AuditLog(args.run_dir / "pressure_audit.jsonl", receipt)
    tokens = AsyncJournal(args.run_dir / "native_tokens.jsonl")
    co = r.PressureCoordinator(cfg, audit.write)
    health_module = load("pressure_health") if cfg.health_service else None
    health_tokens = health_module.TokenWindows() if cfg.health_service else None
    service = o.NativeServiceWindow(health_observer=health_tokens)
    executor = ThreadPoolExecutor(max_workers=max(4, len(jobs) + 1))
    ingress = i.PressureIngress(
        co, service, executor, controller.source_url, controller.target_url
    )
    session = RollingPressureSession(
        run_dir=args.run_dir,
        migration_id=args.migration_id,
        config=controller,
        coordinator=co,
        ingress=ingress,
        executor=executor,
        audit=audit,
    )
    readers = (
        o.InventoryReader(args.source_inventory, "tp1"),
        o.InventoryReader(args.target_inventory, "tp4"),
    )
    scrapers = tuple(
        load("telemetry").MetricsScraper(url, controller.block_size, total)
        for url, total in (
            (controller.source_url, controller.tp1_total_kv_blocks),
            (controller.target_url, controller.tp4_total_kv_blocks),
        )
    )
    predictor = load("manager_m5").PredictorEventReader(
        args.predictor_events, args.predictor_sha256
    )
    dispatched, previous = {}, {}
    growth_readers = {pool: health.NativeDecodeGrowth() for pool in ("tp1", "tp4")}
    started = time.monotonic()
    origin_unix_s = time.time()
    health_tracker = (
        health_module.HealthTracker(
            read(args.health_reference), origin_unix_s, health_tokens
        )
        if cfg.health_service
        else None
    )
    audit.write(dict(kind="pressure_clock_origin", unix_s=origin_unix_s))
    last_view = None
    outcome = "FAILED_OR_INTERRUPTED"
    watch = ProgressWatch(args.no_progress_s)
    progress_path = args.run_dir / "pressure_progress.json"
    last_status_at = -math.inf
    last_metrics_error = {}
    last_inventory_status = {}
    last_inventory_progress = ()
    failure = None
    try:
        while time.monotonic() - started < args.max_seconds:
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
                dispatch_due(
                    jobs,
                    dispatched,
                    elapsed,
                    ingress,
                    unknown_routing_pools(args, controller.block_size),
                    tokens,
                    audit,
                    origin_unix_s=origin_unix_s,
                )
                save_progress(
                    progress_path,
                    stage="waiting_inventory",
                    elapsed_s=elapsed,
                    submitted=len(dispatched),
                    completed=completed_streams(
                        dispatched, session.stream, session.target_future
                    ),
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
                        co.ledger.entries.values(),
                    ),
                    work_due=any(elapsed >= job["at_s"] for job in jobs),
                    reason="waiting for inventory reconciliation; "
                    + json.dumps([reader.diagnostics() for reader in readers]),
                )
                time.sleep(args.tick_s)
                continue
            pools, shapes = [], []
            for reader, raw, scraper in zip(readers, inventories, scrapers):
                if raw["block_size"] != controller.block_size:
                    raise ValueError(
                        "live KV block geometry differs from pilot contract"
                    )
                expected_context = (
                    args.source_context_limit
                    if raw["pool_id"] == "tp1"
                    else args.target_context_limit
                )
                if raw["context_limit"] != expected_context:
                    raise ValueError("live context limit differs from serving contract")
                try:
                    metrics = o.scrape_inventory_metrics(scraper, raw)
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
            source_rates = service.snapshot("tp1", now)["request_rates"]
            if source_rates:
                horizon = min(
                    8192,
                    cfg.survival_horizon
                    + math.ceil(lookup.release_budget * max(source_rates.values())),
                )
                if horizon != co.policy.cfg.survival_horizon:
                    co.policy.cfg = replace(cfg, survival_horizon=horizon)
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
            dispatch_due(
                jobs,
                dispatched,
                time.monotonic() - started,
                ingress,
                pools,
                tokens,
                audit,
                origin_unix_s=origin_unix_s,
            )
            candidates, eos = [], {}
            for row in inventories[0]["requests"]:
                try:
                    stream = ingress.for_server_id(row["request_id"])
                except ValueError:
                    stream = None
                relief, cost = (
                    lookup.deltas(shapes[0], shapes[1], row["request_id"])
                    if all(shapes)
                    else (None, None)
                )
                release_budget = lookup.release_budget
                candidates.append(
                    p.Candidate(
                        row["request_id"],
                        co._owners.get(row["request_id"], "tp1"),
                        inventories[0]["sampled_at"],
                        readers[0].observed_at,
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
                        finished=stream.future.done() if stream is not None else False,
                        action_available=stream is not None,
                        inventory_reference=f"{args.source_inventory}:{inventories[0]['generation']}",
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
            candidates = co.with_predictions(
                candidates,
                predictor,
                now,
                pools[0].free_blocks * pools[0].block_size - 2000,
                eos,
            )
            growth = pools[0].growth_blocks_s
            guard_time = (
                max(0, pools[0].free_blocks - cfg.capacity_guard_blocks) / growth
                if growth is not None and growth > 0
                else None
            )
            pools[0] = replace(
                pools[0],
                guard_time_s=guard_time,
                urgency=lookup.release_budget / guard_time
                if guard_time is not None and guard_time > 0
                else None,
            )
            runtime = load("manager_m0").RuntimeSnapshot(
                unix_s=now,
                migration_id=args.migration_id,
                request_id=None,
                state="LOCAL",
                generated_tokens=None,
                source_sampled_unix_s=pools[0].sampled_at,
                target_sampled_unix_s=pools[1].sampled_at,
                source_free_kv_tokens=pools[0].free_blocks * pools[0].block_size,
                source_guard_free_kv_tokens=2000,
                source_pool_growth_tokens_s=pools[0].growth_blocks_s
                * pools[0].block_size
                if pools[0].growth_blocks_s is not None
                else None,
                target_running=inventories[1]["running"],
                target_waiting=co.ledger.routing_reading(pools[1]).waiting,
                target_kv_usage_frac=1
                - pools[1].free_blocks / inventories[1]["total_blocks"],
            )
            actions = (
                None
                if args.no_migration or session.record is not None or not source_rates
                else {
                    c.request_id: r.BoundAction(
                        c.request_id,
                        args.migration_id,
                        lambda c=c, runtime=runtime: session.start(c, runtime),
                    )
                    for c in candidates
                }
            )
            co.tick(now, pools[0], pools[1], candidates, actions)
            if session.record is not None and not session.settled:
                progress = session.old.read_source_progress(
                    args.run_dir, session.stream.payload, now
                )
                if (
                    progress is not None
                    and progress.request_id == session.record.request_id
                ):
                    last_view = progress
                if last_view is not None:
                    session.tick(last_view, runtime)
                elif session.stream.future.done():
                    session.cancel(
                        "source ended before selected progress was published"
                    )
            if session.target_future is not None and session.target_future.done():
                session.target_future.result()
                service.finish(session.stream.recorder.target_service_id)
            if session.record is not None:
                entry = co.ledger.entries[args.migration_id]
                prefix = f"cmpl-bridgetp-phase9-target-{args.run_dir.name}-0-"
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
                    co.ledger.observe(
                        entry.key,
                        allocated_blocks=min(targets[0]["owned_blocks"], entry.blocks),
                        service_observed=entry.service_observed
                        or targets[0]["generated"] >= 2,
                        prefill_observed=True,
                    )
                if (
                    not targets
                    and session.settled
                    and session.target_future is not None
                    and session.target_future.done()
                    and not session.target_future.exception()
                    and pools[1].sampled_at
                    > session.target_future.result()["completed_unix_s"]
                    and entry.state in {"COMMITTED", "ACCOUNTED"}
                ):
                    co.ledger.complete(entry.key, evidence=True)
            for value in dispatched.values():
                if value.future.done():
                    value.future.result()
            due_queued = [
                job["id"]
                for job in jobs
                if elapsed >= job["at_s"] and job["id"] not in dispatched
            ]
            completed = completed_streams(
                dispatched, session.stream, session.target_future
            )
            last_inventory_progress = inventory_progress(inventories)
            marker = workload_progress(
                dispatched,
                last_inventory_progress,
                session,
                co.ledger.entries.values(),
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
                        )
                        for pool in pools
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
                        for row in co.ledger.entries.values()
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
                    for row in co.ledger.entries.values()
                )
                and (
                    session.record is None
                    or (
                        (session.settled or session.cleanup_requires_shutdown)
                        and (
                            session.target_future is None
                            or session.target_future.done()
                        )
                    )
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
        save_progress(
            progress_path,
            stage=outcome,
            error=failure,
            submitted=len(dispatched),
            completed=completed_streams(
                dispatched, session.stream, session.target_future
            ),
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
                outcome=outcome,
                error=failure,
                shutdown_cleanup_required=session.cleanup_requires_shutdown,
            )
        )
        executor.shutdown(wait=False, cancel_futures=True)
        if all(v.future.done() for v in dispatched.values()) and (
            session.target_future is None or session.target_future.done()
        ):
            tokens.close()
        else:
            atexit.register(tokens.close)
        session.adapter.close()
        audit.close()


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
