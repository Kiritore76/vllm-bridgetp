#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Own four fresh function-pilot server groups; never run paired benefit fitting.

The spec freezes input/code/model/checkpoint SHA. Repository bundles record the
actual clean committed HEAD at runtime, avoiding self-referential manifests.
Linux dependency/rolling checks precede GPU startup.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import run_pressure_balanced_pilot as controller  # noqa: E402
from tools.bridge_tp.pressure_cpu import load  # noqa: E402
from tools.bridge_tp.pressure_progress import save_progress  # noqa: E402


def announce(scene, stage, **details):
    stamp = time.strftime("%H:%M:%S")
    fields = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[{stamp}] [{scene}] {stage} {fields}", flush=True)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_spec(path):
    spec = read(path)
    if spec.get("identity_mode") != "git-tracked-bundle-v1":
        return spec
    relative = path.resolve().relative_to(ROOT).as_posix()
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    spec["expected_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    spec["bundle_spec_path"] = relative
    spec["bundle_spec_sha256"] = sha(path)

    def resolve(value):
        resolved = (ROOT / value).resolve()
        resolved.relative_to(ROOT)
        return str(resolved)

    for key in (
        "policy_config",
        "service_evidence",
        "controller_config_template",
        "checkpoint",
    ):
        spec[key] = resolve(spec[key])
    subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "--",
            Path(spec["checkpoint"]).relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    if spec.get("health_reference"):
        spec["health_reference"] = resolve(spec["health_reference"])
    for scene in spec["scenarios"]:
        scene["arrivals"] = resolve(scene["arrivals"])
    return spec


def verify_spec(spec):
    if (
        spec["model_config_sha256"]
        != "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e"
        or spec["checkpoint_sha256"]
        != "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7"
    ):
        raise ValueError("frozen Qwen/predictor identities differ")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if head != spec["expected_head"]:
        raise ValueError("checkout HEAD differs from frozen pilot spec")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != spec["expected_branch"]:
        raise ValueError("checkout branch differs from frozen pilot spec")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("pilot requires committed code; preserve dirty work and stop")
    for name, expected in spec["sha256"].items():
        path = ROOT / name if not Path(name).is_absolute() else Path(name)
        if sha(path) != expected:
            raise ValueError(f"pilot file SHA differs: {name}")
    if sha(Path(spec["model_path"]) / "config.json") != spec["model_config_sha256"]:
        raise ValueError("model configuration SHA differs")
    if sha(spec["checkpoint"]) != spec["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA differs")
    if {r["name"] for r in spec["scenarios"]} != {
        "both-light",
        "source-decode",
        "target-prefill",
        "capacity-arrival",
    } or len(spec["scenarios"]) != 4:
        raise ValueError("first pilot requires exactly four function scenarios")


def devices_of(args):
    return args.tp4_gpus.split(",")


def arguments(spec):
    # Existing transport helper is imported only in the actual server environment.
    from tools.bridge_tp import run_phase9_cap0_calibration as common

    values = copy.deepcopy(spec["server_settings"])
    values.update(model_path=Path(spec["model_path"]), python_bin=Path(sys.executable))
    args = SimpleNamespace(**values)
    target_tp = getattr(args, "target_tp_size", 4)
    if target_tp not in (2, 4):
        raise ValueError("target TP must be 2 or 4")
    devices = devices_of(args)
    if (
        len(devices) != target_tp
        or len(set(devices)) != target_tp
        or args.tp1_gpu in devices
    ):
        raise ValueError("source/target GPUs must be disjoint and match TP")
    context_profile = spec.get("context_profile")
    limits = (8192, 16384)
    if context_profile is not None:
        if context_profile != "QWEN3_NATIVE_32K_BOTH_V1" or target_tp != 2:
            raise ValueError("unknown or incompatible context profile")
        limits = (32768, 32768)
    if (
        args.dtype != "bfloat16"
        or (args.max_model_len, args.tp4_max_model_len) != limits
        or (target_tp == 4 and args.tp1_gpu != "0")
        or (target_tp == 4 and args.tp4_gpus != "1,2,3,4")
        or not args.persistent_channel
        or not args.preconnect_persistent_channel
        or not args.gpu_direct_history
        or not args.gpu_direct_delta
    ):
        raise ValueError("pilot transport/hardware scope differs from frozen mechanism")
    return common, args


def build_scene(spec, scene, run_root, common, args):
    target_tp = getattr(args, "target_tp_size", 4)
    run_dir = run_root / scene["name"] / "controller"
    multiple = spec.get("maximum_migrations", 1) != 1
    mechanism_dir = run_dir / "migration_active" if multiple else run_dir
    config_path = run_root / scene["name"] / "controller_config.json"
    config = read(spec["controller_config_template"])
    config.update(
        transfer_mode="FULL_SPEED",
        target_tp_size=target_tp,
        source_url=f"http://127.0.0.1:{args.tp1_port}",
        target_url=f"http://127.0.0.1:{args.tp4_port}",
        run_dir=str(run_dir),
    )
    config.update(
        ready_notification_mode="UDP",
        ready_notification_port=args.ready_notification_port,
    )
    inherited = {k: v for k, v in os.environ.items() if k.startswith("BRIDGETP_")}
    try:
        for key in inherited:
            os.environ.pop(key)
        source = common.source_environment(args, scene["name"], mechanism_dir)
    finally:
        os.environ.update(inherited)
    source.update(
        BRIDGETP_STREAM_TARGET_TP=str(target_tp),
        VLLM_DISABLE_COMPILE_CACHE="1",
        BRIDGETP_SHADOW_STRATEGY="S_NEW_OLD",
        BRIDGETP_REQUEST_FREEZE_ENABLED="1",
        BRIDGETP_GPU_DIRECT_HISTORY_PACING="0",
        BRIDGETP_STREAM_RATE_GIB_S="0",
        BRIDGETP_PREDICTOR_LIVE_CHECKPOINT=spec["checkpoint"],
        BRIDGETP_PREDICTOR_LIVE_SHA256=spec["checkpoint_sha256"],
        BRIDGETP_PREDICTOR_LIVE_EVENTS=str(run_dir / "predictor_events.jsonl"),
        BRIDGETP_PRESSURE_INVENTORY_PATH=str(run_dir / "source_inventory.jsonl"),
        BRIDGETP_PRESSURE_POOL="tp1",
        BRIDGETP_ROLLING_MIN_LEAD_TOKENS="128",
    )
    target = os.environ.copy()
    # Never leak the source publisher into TP4 workers.
    for key in list(target):
        if key.startswith("BRIDGETP_"):
            target.pop(key)
    target.update(
        BRIDGETP_STREAM_TARGET_TP=str(target_tp),
        CUDA_VISIBLE_DEVICES=args.tp4_gpus,
        OMP_NUM_THREADS="1",
        BRIDGETP_TAKEOVER_ENABLED="1",
        BRIDGETP_TAKEOVER_RUN_DIR=str(mechanism_dir),
        BRIDGETP_TAKEOVER_MIGRATION_ID=scene["name"],
        BRIDGETP_PRESSURE_INVENTORY_PATH=str(run_dir / "target_inventory.jsonl"),
        BRIDGETP_PRESSURE_POOL="tp4",
    )
    if multiple:
        target["BRIDGETP_TAKEOVER_PERSISTENT_CHANNEL"] = "1"
    target_argv = common.server_command(args, target_tp, args.tp4_port) + [
        "--kv-transfer-config",
        common.target_connector(
            mechanism_dir,
            gpu_resident_shadow=True,
            persistent_channel=True,
            ready_sync_mode="STREAM_EVENT",
            ready_notification_mode="UDP",
            ready_notification_port=args.ready_notification_port,
            channel_generation=args.channel_generation,
            preconnect_gpu_direct=True,
            gpu_direct_base_port=args.gpu_direct_base_port,
        ),
    ]
    if multiple:
        connector = json.loads(target_argv[-1])
        # Reuse one bounded receive buffer. Old decode completions must not
        # release or write cleanup receipts for a newer active transfer.
        connector["kv_connector_extra_config"][
            "bridgetp_release_persistent_payload_buffer"
        ] = False
        target_argv[-1] = json.dumps(connector)
    source_argv = common.server_command(args, 1, args.tp1_port)
    stager_argv = [
        sys.executable,
        str(ROOT / "tools/bridge_tp/pressure_sessions.py")
        if multiple
        else str(common.STAGER),
        "--run-dir",
        str(mechanism_dir),
        "--delta-host",
        "127.0.0.1",
        "--delta-base-port",
        str(args.delta_port),
        "--delivery-host",
        "127.0.0.1",
        "--delivery-base-port",
        str(args.delivery_port),
        "--timeout-s",
        str(spec["run_timeout_s"]),
        "--gpu-resident-shadow",
    ]
    argv = [
        sys.executable,
        str(ROOT / "tools/bridge_tp/run_pressure_balanced_pilot.py"),
        "--policy",
        "pressure-balanced-v1",
        "--policy-config",
        spec["policy_config"],
        "--controller-config",
        str(config_path),
        "--service-evidence",
        spec["service_evidence"],
        "--arrivals",
        scene["arrivals"],
        "--run-dir",
        str(run_dir),
        "--source-inventory",
        str(run_dir / "source_inventory.jsonl"),
        "--target-inventory",
        str(run_dir / "target_inventory.jsonl"),
        "--source-context-limit",
        str(args.max_model_len),
        "--target-context-limit",
        str(args.tp4_max_model_len),
        "--predictor-events",
        str(run_dir / "predictor_events.jsonl"),
        "--predictor-sha256",
        spec["checkpoint_sha256"],
        "--migration-id",
        scene["name"],
        "--max-seconds",
        str(spec["run_timeout_s"]),
        "--tick-s",
        "0.2",
        "--no-progress-s",
        str(spec.get("no_progress_s", 60)),
    ]
    if spec.get("health_reference"):
        argv.extend(["--health-reference", spec["health_reference"]])
    if multiple:
        argv.extend(["--maximum-migrations", str(spec["maximum_migrations"])])
    instances_path = run_root / scene["name"] / "instances.json"
    argv.extend(["--instances", str(instances_path)])
    # Instance identities for the fixed pair; the controller still runs on the
    # historical tp1/tp4 pool ids, recorded here as aliases.
    instance_pool = load("instance_pool")
    pair_bundles, extras = layout_extras(spec, args, target_tp, run_dir)
    registry, aliases = instance_pool.fixed_pair_registry(
        source_gpu=args.tp1_gpu,
        target_gpus=devices_of(args),
        source_url=config["source_url"],
        target_url=config["target_url"],
        identity=str(args.model_path),
        source_context_limit=args.max_model_len,
        target_context_limit=args.tp4_max_model_len,
        source_bundle=pair_bundles[0],
        target_bundle=pair_bundles[1],
        extras=extras,
    )
    instances = instance_pool.registry_manifest(registry, aliases)
    paired = set(aliases.values())
    by_endpoint = {e["endpoint"]: e for e in extras}
    for row in instances["instances"]:
        if row["instance_id"] not in paired:
            extra = by_endpoint[row["endpoint"]]
            extra["instance_id"] = row["instance_id"]
            extra["inventory_path"] = row["inventory_path"] = str(
                run_dir / f"inventory_{row['instance_id']}.jsonl"
            )
            if "migration_channel" in extra:
                # Channel generation = source instance generation: unique per
                # run, and a rebuilt instance gets a new one.
                extra["migration_channel"]["generation"] = row["instance_generation"]
                extra["migration_channel"]["target"] = aliases["tp4"]
                extra["migration_channel"]["target_tp"] = target_tp
                row["migration_channel"] = dict(extra["migration_channel"])
                extra["predictor_events"] = row["predictor_events"] = str(
                    run_dir / f"predictor_events_{row['instance_id']}.jsonl"
                )
    if set(aliases) != {"tp1", "tp4"}:
        # Single-class layout: routing only, nothing to migrate between.
        argv.append("--no-migration")
    sources = (
        source_channel_list(args, aliases["tp1"], extras) if "tp1" in aliases else []
    )
    request = conversion_request(spec, registry)
    channel_plan = None
    if request is not None and sources:
        # Conversion may retire a source: the target follows a channel plan
        # file (same rows as the static list) so its channel can be closed.
        channel_plan = run_root / scene["name"] / "gpu_channel_plan.json"
    if len(sources) > 1 or channel_plan is not None:
        connector = json.loads(target_argv[-1])
        extra_config = connector["kv_connector_extra_config"]
        extra_config["bridgetp_gpu_direct_sources"] = sources
        if channel_plan is not None:
            extra_config["bridgetp_gpu_direct_channel_plan"] = str(channel_plan)
        target_argv[-1] = json.dumps(connector)
    return dict(
        registry=registry,
        conversion=request,
        context_limits=(args.max_model_len, args.tp4_max_model_len),
        channel_plan=channel_plan,
        channel_sources=sources,
        run_dir=run_dir,
        instances=instances,
        instances_path=instances_path,
        extra_engines=extras,
        config=config,
        config_path=config_path,
        source_env=source,
        target_env=target,
        source_argv=source_argv,
        target_argv=target_argv,
        stager_argv=stager_argv,
        controller_argv=argv,
    )


def layout_extras(spec, args, target_tp, run_dir):
    """Routing-only engines of ``spec["layout"]`` beside the migration pair.

    A mixed layout must contain the pair (TP1 on ``tp1_gpu``, TP-k on
    ``tp4_gpus``). A single-class layout (L0: all TP1, L2: all TP-k) keeps only
    the side of its class, on the same GPUs, and runs without migration.
    Returns the pair's bundle ids (``None`` for an absent side) and one dict
    per extra engine.
    """
    layout = spec.get("layout")
    if not layout:
        return ("A", "B"), []
    channels = bool(layout.get("migration_sources"))
    if channels and int(args.channel_generation) != 1:
        raise ValueError("per-source channels need the pair source on generation 1")
    rows = load("instance_pool").layouts(target_tp, list(layout["bundle_gpus"]))[
        layout["name"]
    ]
    source = [r for r in rows if r["tp_degree"] == 1 and r["gpu_ids"] == [args.tp1_gpu]]
    target = [
        r
        for r in rows
        if r["tp_degree"] == target_tp and r["gpu_ids"] == devices_of(args)
    ]
    classes = {r["tp_degree"] for r in rows}
    if len(classes) == 1 and channels:
        raise ValueError("single-class layouts have no migration channels")
    if len(source) + len(target) != (2 if len(classes) == 2 else 1) or (
        classes == {1} and not source or classes == {target_tp} and not target
    ):
        raise ValueError("layout must contain the TP1 source and TP-k target pair")
    extras, sources = [], 0
    pair = source + target
    for index, row in enumerate(r for r in rows if r not in pair):
        port = int(layout["base_port"]) + index
        extras.append(
            dict(
                bundle=row["bundle_id"],
                tp=row["tp_degree"],
                gpus=row["gpu_ids"],
                port=port,
                endpoint=f"http://127.0.0.1:{port}",
                context_limit=args.max_model_len
                if row["tp_degree"] == 1
                else args.tp4_max_model_len,
            )
        )
        if channels and row["tp_degree"] == 1:
            # Port blocks [base, base + tp] of 10 apart never overlap for TP<=4.
            sources += 1
            extras[-1]["migration_channel"] = dict(
                base_port=int(args.gpu_direct_base_port) + 10 * sources
            )
    return (
        source[0]["bundle_id"] if source else None,
        target[0]["bundle_id"] if target else None,
    ), extras


def live_pair(plan):
    """Pair sides still serving at the end: not absent, not retired mid-run."""
    aliases = plan.get("instances", {}).get("legacy_pool_aliases", {})
    status_path = plan["run_dir"] / "layout_status.json"
    status = read(status_path)["instances"] if status_path.exists() else {}
    return [
        side
        for side, pool in (("source", "tp1"), ("target", "tp4"))
        if side not in absent_pair(plan)
        and status.get(aliases.get(pool), {}).get("lifecycle") != "RETIRED"
    ]


def conversion_request(spec, registry):
    """Validated explicit conversion of one bundle (S3b-3), or None."""
    request = spec.get("conversion")
    if request is None:
        return None
    if not spec.get("layout"):
        raise ValueError("conversion needs an explicit layout")
    bundles = {row.bundle_id for row in registry.instances.values()}
    if (
        set(request)
        != {"bundle", "target_mode", "trigger_s", "deadline_s", "base_port"}
        or request["bundle"] not in bundles
        or request["target_mode"] not in {"TP1", "TP2"}
        or not 0 <= request["trigger_s"] < request["deadline_s"]
        or int(request["base_port"]) <= 0
    ):
        raise ValueError("invalid conversion request")
    return dict(request)


def absent_pair(plan):
    """Pair sides ("source"/"target") a single-class layout does not start."""
    aliases = plan.get("instances", {}).get("legacy_pool_aliases", {"tp1": 0, "tp4": 0})
    return [
        side
        for side, pool in (("source", "tp1"), ("target", "tp4"))
        if pool not in aliases
    ]


def source_channel_list(args, pair_source_id, extras):
    """Target connector source list, sorted by source id (handshake order)."""
    rows = [
        dict(
            source=pair_source_id,
            base_port=int(args.gpu_direct_base_port),
            generation=int(args.channel_generation),
        )
    ] + [
        dict(
            source=e["instance_id"],
            base_port=e["migration_channel"]["base_port"],
            generation=e["migration_channel"]["generation"],
        )
        for e in extras
        if "migration_channel" in e
    ]
    return sorted(rows, key=lambda row: row["source"])


def extra_engine(plan, extra):
    """Serving argv/env of one routing-only engine, from the final pair plan.

    Same serving flags as the pair engine of its TP class (whatever the runner
    appended), without the migration connector or any source publisher.
    """
    template = plan["source_argv"] if extra["tp"] == 1 else plan["target_argv"]
    argv = list(template)
    if "--kv-transfer-config" in argv:
        index = argv.index("--kv-transfer-config")
        del argv[index : index + 2]
    argv[argv.index("--port") + 1] = str(extra["port"])
    channel = extra.get("migration_channel")
    if channel:
        # A second migration source: the pair source's publisher environment
        # with its own channel, predictor journal, inventory and trace.
        env = dict(plan["source_env"])
        env.update(
            BRIDGETP_GPU_DIRECT_BASE_PORT=str(channel["base_port"]),
            BRIDGETP_CHANNEL_GENERATION=str(channel["generation"]),
            BRIDGETP_PREDICTOR_LIVE_EVENTS=extra["predictor_events"],
            # One listed target: its own preconnect receipt name, implied per
            # session (the pair source keeps the legacy source.json receipt).
            BRIDGETP_GPU_DIRECT_TARGETS=json.dumps(
                [
                    dict(
                        target=channel["target"],
                        host=env.get("BRIDGETP_GPU_DIRECT_HOST", "127.0.0.1"),
                        base_port=channel["base_port"],
                        tp=channel["target_tp"],
                        generation=channel["generation"],
                    )
                ]
            ),
        )
    else:
        env = {
            k: v
            for k, v in plan["target_env"].items()
            if not k.startswith(("BRIDGETP_TAKEOVER", "BRIDGETP_STREAM_"))
        }
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(extra["gpus"]),
        OMP_NUM_THREADS="1",
        BRIDGETP_PRESSURE_INVENTORY_PATH=extra["inventory_path"],
        BRIDGETP_PRESSURE_POOL=extra["instance_id"],
    )
    if "BRIDGETP_PERFORMANCE_TRACE_DIR" in env:
        env["BRIDGETP_PERFORMANCE_TRACE_DIR"] = str(
            Path(env["BRIDGETP_PERFORMANCE_TRACE_DIR"]).parent / extra["instance_id"]
        )
    return argv, env


def write_instances(plan):
    """The controller checks --instances in every preflight, so write it first."""
    path = plan.get("instances_path")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan["instances"], indent=2) + "\n", "utf-8")


def verify_full_speed(run_dir):
    """Validate actual controls and history receipts, not only launch intent."""
    starts = 0
    with (run_dir / "pressure_audit.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "pressure_actual_start":
                rate = row["rate"]
                starts += 1
            elif row.get("kind") == "pressure_m2_rate":
                rate = row["decision"]
            else:
                continue
            if rate["profile"] != "FULL_SPEED" or rate["rate_bytes_s"] != 0:
                raise RuntimeError("full-speed run published a rate limit")
    controls = list(run_dir.glob("sessions/*/runtime_control.json"))
    controls += list(run_dir.glob("runtime_control.json"))
    for path in controls:
        if read(path).get("rate_gib_s") not in (None, 0):
            raise RuntimeError("full-speed runtime control contains a rate limit")
    senders = list(run_dir.glob("sessions/*/gpu_direct_sender.json"))
    senders += list(run_dir.glob("gpu_direct_sender.json"))
    ranks = 0
    for path in senders:
        for rank in read(path)["ranks"]:
            if (
                rank.get("history_pacing_enabled") is not False
                or rank.get("history_pacing_sleep_ms") != 0
                or rank.get("history_pacing_chunk_count") != 0
            ):
                raise RuntimeError("full-speed history receipt reports pacing")
            ranks += 1
    return dict(
        status="COMPLETED",
        transfer_mode="FULL_SPEED",
        actual_starts=starts,
        history_sender_receipts=len(senders),
        history_rank_receipts=ranks,
        note="zero means unlimited; no migration is not transport coverage",
    )


def scene_result(run_dir, name):
    rows = [
        json.loads(v)
        for v in (run_dir / "pressure_audit.jsonl").read_text().splitlines()
    ]
    starts = [r for r in rows if r.get("kind") == "pressure_actual_start"]
    released = [r for r in rows if r.get("kind") == "pressure_source_release_confirmed"]
    final = next(r for r in reversed(rows) if r.get("kind") == "pressure_final")
    shutdown = read(run_dir.parent / "owned_shutdown_receipt.json")
    pending_shutdown = final.get("shutdown_cleanup_required", False)
    if final["outcome"] not in {"COMPLETED", "COMPLETED_PENDING_OWNER_CLEANUP"}:
        raise RuntimeError("controller did not complete; keep full evidence")
    maximum = final.get("maximum_migrations", 1)
    if (
        (maximum and len(starts) > maximum)
        or len({r["migration_id"] for r in starts}) != len(starts)
        or any(
            r["state"] in {"ACTIVE", "COMMITTED", "ACCOUNTED"}
            and not (
                pending_shutdown
                and shutdown["all_owned_groups_ended"]
                and r["kind"] == "migration"
            )
            for r in final["reservations"]
        )
    ):
        raise RuntimeError("duplicate START or outstanding reservation")
    decisions = [r for r in rows if r.get("kind") == "pressure_decision"]
    light = [
        r
        for r in decisions
        if r["decision"]["source"]["zone"] == "SPARE"
        and r["decision"]["target"]["zone"] == "SPARE"
    ]
    if any(
        r["execution"] == "RESERVED"
        and r["decision"].get("service_path") not in {"health", "workload"}
        for r in light
    ):
        raise RuntimeError("actually light pools started migration")
    if name == "both-light":
        covered = bool(light) and not starts
    elif name == "source-decode":
        eligible = [
            r
            for r in decisions
            if r["execution"] == "RESERVED"
            and not r["decision"]["capacity_path"]
            and (r["decision"]["source"]["decode"] or 0) >= 1.2
            and sum(
                c["request"].get("action_available", True) and not c["rejections"]
                for c in r["decision"]["candidates"]
            )
            >= 2
        ]
        covered = bool(eligible and starts and released)
    elif name == "target-prefill":
        observed = [
            r
            for r in decisions
            if r["target_reading"]["pending_prompt_tokens"]
            and (r["decision"]["source"]["decode"] or 0) >= 1.2
        ]
        if any(r["target_reading"]["idle"] for r in observed):
            raise RuntimeError("target prefill was treated as idle")
        covered = bool(observed)
    else:
        capacity = [
            r
            for r in decisions
            if r["execution"] == "RESERVED" and r["decision"]["capacity_path"]
        ]
        begun = next(
            (r["unix_s"] for r in rows if r.get("kind") == "pressure_start_published"),
            None,
        )
        ended = next(
            (r["unix_s"] for r in rows if r.get("kind") == "pressure_settlement"), None
        )
        arrival = (
            begun is not None
            and ended is not None
            and any(
                r.get("kind") == "pressure_dispatch"
                and r["pool"] == "tp4"
                and begun <= r["unix_s"] < ended
                for r in rows
            )
        )
        covered = bool(capacity and released and arrival)
    return dict(
        status="FUNCTION_OBSERVED" if covered else "COVERAGE_GAP",
        actual_starts=len(starts),
        source_releases=len(released),
        note="coverage gap/STAY is not technical failure; inspect actual strata",
    )


def timing_receipt(path, scene_failed):
    from tools.bridge_tp.pressure_cpu import load

    if scene_failed and not path.exists():
        return dict(
            status="UNAVAILABLE_AFTER_SCENE_FAILURE",
            reason="controller timing journal was not created",
        )
    return load("pause_diagnostics").verify(path)


def start_conversion(spec, scene, plan, common, processes, publish, requests):
    """Bind the conversion backend to this scene's owned engines and start it."""
    from tools.bridge_tp import pressure_conversion as pc
    from tools.bridge_tp.conversion_planner import ShadowPlanner, load_profile

    root, run_dir = plan["run_dir"].parent, plan["run_dir"]
    aliases = plan["instances"]["legacy_pool_aliases"]
    by_name = {p.name: p for p in processes}
    owned = {
        **({aliases["tp1"]: by_name["source TP1"]} if "tp1" in aliases else {}),
        **(
            {aliases["tp4"]: next(p for p in processes if p.name.startswith("target"))}
            if "tp4" in aliases
            else {}
        ),
        **{e["instance_id"]: by_name[e["instance_id"]] for e in plan["extra_engines"]},
    }
    manifest = json.loads(json.dumps(plan["instances"]))
    for row in manifest["instances"]:
        pool_id = row.get("legacy_pool")
        if pool_id is not None:
            row["inventory_path"] = (
                str(run_dir / ("source" if pool_id == "tp1" else "target"))
                + "_inventory.jsonl"
            )

    def event(kind, **values):
        row = dict(kind=kind, unix_s=time.time(), **values)
        with (root / "conversion_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        announce(scene["name"], "转换", **{"event": kind, **values})

    def start(extra):
        argv, env = extra_engine(plan, extra)
        process = common.start_process(
            extra["instance_id"], argv, env, root / f"{extra['instance_id']}.log"
        )
        processes.append(process)
        owned[extra["instance_id"]] = process
        publish()

    def warm(extra):
        from tools.bridge_tp.run_phase9_cap0_noop import warmup_prompt_shapes

        process = owned[extra["instance_id"]]
        common.wait_healthy(extra["endpoint"], process, spec["startup_timeout_s"])
        warmup_prompt_shapes(
            extra["endpoint"],
            requests,
            root / f"{extra['instance_id']}_warmup.json",
            spec["run_timeout_s"],
        )

    ops = SimpleNamespace(
        stop=lambda ids: common.stop_processes([owned[i] for i in ids]),
        alive=lambda i: common.process_group_alive(owned[i]),
        gpus_released=pc.nvidia_gpus_released,
        scrape=pc.scrape_metrics,
        start=start,
        warm=warm,
        event=event,
    )
    target_tp = plan["config"].get("target_tp_size", 4)
    backend = pc.ControllerConversionBackend(
        plan["registry"],
        manifest,
        run_dir,
        ops,
        base_port=plan["conversion"]["base_port"],
        context_limits=plan["context_limits"],
        channel_plan=plan.get("channel_plan"),
        target_tp_ranks=target_tp if "tp4" in aliases else 0,
        shadow_planner=ShadowPlanner(load_profile(), manifest["instances"]),
    )
    driver = pc.ConversionDriver(
        backend, plan["conversion"], root / "conversion_plan.json"
    )
    driver.start()
    return driver


def finish_conversion(driver, root):
    outcome = driver.stop()
    (root / "conversion_result.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
    )
    driver.thread = None
    driver.outcome = outcome
    return outcome


def run_scene(
    spec,
    scene,
    plan,
    common,
    args,
    *,
    before_shutdown=None,
    before_workload=None,
    diagnose_pauses=False,
):
    from tools.bridge_tp.run_phase9_cap0_noop import warmup_prompt_shapes
    from tools.bridge_tp.run_phase9_controller import RuntimeControl

    root, processes = plan["run_dir"].parent, []
    monitor = driver = None

    def publish_processes():
        if diagnose_pauses:
            common.write_json(
                root / "owned_processes.json",
                [
                    dict(name=p.name, pid=p.process.pid)
                    for p in processes
                    if p is not monitor
                ],
            )

    plan["run_dir"].mkdir(parents=True, exist_ok=False)
    mechanism_dir = plan["run_dir"]
    multiple = "--maximum-migrations" in plan["controller_argv"]
    if multiple:
        from tools.bridge_tp.pressure_sessions import initialize

        mechanism_dir = initialize(plan["run_dir"])
    common.write_json(plan["config_path"], plan["config"])
    write_instances(plan)
    if plan["config"].get("transfer_mode") == "FULL_SPEED":
        announce(scene["name"], "迁移全速：历史pacing关闭，M2动态限速关闭")
    RuntimeControl(
        armed=False,
        rate_gib_s=0 if plan["config"].get("transfer_mode") == "FULL_SPEED" else None,
        migration_id=scene["name"],
        source_request_id_prefix="pressure-unselected-" + secrets.token_hex(16),
    ).write(mechanism_dir)
    engines = {
        extra["instance_id"]: extra_engine(plan, extra)
        for extra in plan.get("extra_engines", ())
    }
    common.write_json(
        root / "effective_argv.json",
        {k: v for k, v in plan.items() if k.endswith("_argv")}
        | ({"extra_argv": {k: v[0] for k, v in engines.items()}} if engines else {}),
    )

    def recorded(env):
        return {
            key: value
            for key, value in env.items()
            if key.startswith(("BRIDGETP_", "CUDA_", "NCCL_", "VLLM_", "OMP_"))
            and not key.endswith("_TOKEN")
        }

    common.write_json(
        root / "effective_environment.json",
        {name: recorded(plan[name + "_env"]) for name in ("source", "target")}
        | (
            {"extra": {k: recorded(v[1]) for k, v in engines.items()}}
            if engines
            else {}
        ),
    )
    try:
        if diagnose_pauses:
            publish_processes()
            monitor = common.start_process(
                "host monitor",
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "tools/bridge_tp/monitor_pressure_host.py"),
                    "--output",
                    str(root / "host_monitor.jsonl"),
                    "--stop-file",
                    str(root / "host_monitor.stop"),
                    "--process-file",
                    str(root / "owned_processes.json"),
                ],
                os.environ.copy(),
                root / "host_monitor.log",
            )
            processes.append(monitor)
        absent = absent_pair(plan)
        if plan.get("channel_plan") is not None:
            common.write_json(plan["channel_plan"], plan["channel_sources"])
        target_label = f"TP{plan['config'].get('target_tp_size', 4)}"
        if "target" not in absent:
            announce(scene["name"], f"启动{target_label}服务，等待就绪")
            target = common.start_process(
                f"target {target_label}",
                plan["target_argv"],
                plan["target_env"],
                root / "target.log",
            )
            processes.append(target)
            publish_processes()
            common.wait_healthy(
                plan["config"]["target_url"], target, spec["startup_timeout_s"]
            )
        if "source" not in absent:
            announce(scene["name"], "启动TP1服务")
            source = common.start_process(
                "source TP1",
                plan["source_argv"],
                plan["source_env"],
                root / "source.log",
            )
            processes.append(source)
            publish_processes()
            common.wait_healthy(
                plan["config"]["source_url"], source, spec["startup_timeout_s"]
            )
        extras = []
        for extra in plan.get("extra_engines", ()):
            argv, env = engines[extra["instance_id"]]
            announce(scene["name"], "启动只路由实例", instance=extra["instance_id"])
            extras.append(
                (
                    extra,
                    common.start_process(
                        extra["instance_id"],
                        argv,
                        env,
                        root / f"{extra['instance_id']}.log",
                    ),
                )
            )
            processes.append(extras[-1][1])
        publish_processes()
        for extra, process in extras:
            common.wait_healthy(extra["endpoint"], process, spec["startup_timeout_s"])
        announce(scene["name"], "实例就绪；开始warmup")
        requests = [job["request"] for job in read(scene["arrivals"])]
        for pool in ("source", "target"):
            if pool in absent:
                continue
            announce(scene["name"], "warmup", pool=pool)
            warmup_prompt_shapes(
                plan["config"][pool + "_url"],
                requests,
                root / (pool + "_warmup.json"),
                spec["run_timeout_s"],
            )
        for extra, _ in extras:
            announce(scene["name"], "warmup", pool=extra["instance_id"])
            warmup_prompt_shapes(
                extra["endpoint"],
                requests,
                root / f"{extra['instance_id']}_warmup.json",
                spec["run_timeout_s"],
            )
        if before_workload is not None:
            before_workload(plan)
        if absent:
            # Single-class layout: no pair to migrate between, so no stager.
            announce(scene["name"], "warmup完成；单类布局不启动stager；启动正式调度器")
        else:
            announce(scene["name"], "warmup完成；启动stager和正式调度器")
            stager = common.start_process(
                "stager", plan["stager_argv"], os.environ.copy(), root / "stager.log"
            )
            processes.append(stager)
            publish_processes()
        if spec.get("require_stager_process_ready") and not absent:
            from tools.bridge_tp.qwen3_migration_diagnostics import wait_stager_ready

            wait_stager_ready(mechanism_dir, stager)
        checked = controller.parse_args(
            plan["controller_argv"][2:] + ["--validate-only"]
        )
        controller.preflight(checked)
        (root / "controller_config.sha256").write_text(sha(plan["config_path"]) + "\n")
        worker_env = os.environ.copy()
        if diagnose_pauses:
            worker_env["BRIDGETP_PRESSURE_DIAGNOSTICS"] = str(
                plan["run_dir"] / "pressure_timing.jsonl"
            )
        worker = common.start_process(
            "pressure controller",
            plan["controller_argv"],
            worker_env,
            root / "controller.log",
        )
        processes.append(worker)
        publish_processes()
        if plan.get("conversion") is not None:
            driver = start_conversion(
                spec, scene, plan, common, processes, publish_processes, requests
            )
        deadline = time.monotonic() + spec["run_timeout_s"] + 60
        last_progress_at = -float("inf")
        while worker.process.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError("owned controller deadline reached")
            if time.monotonic() - last_progress_at >= 10:
                last_progress_at = time.monotonic()
                path = plan["run_dir"] / "pressure_progress.json"
                if path.exists():
                    status = read(path)
                    save_progress(
                        root.parent / "pilot_progress.json",
                        scene=scene["name"],
                        **status,
                    )
                    announce(
                        scene["name"],
                        status["stage"],
                        elapsed=status.get("elapsed_s"),
                        submitted=(
                            f"{status.get('submitted', 0)}/{status.get('total', '?')}"
                        ),
                        completed=status.get("completed", 0),
                        pending_dispatch=len(status.get("queued", [])),
                        engine_waiting=status.get("engine_waiting", {}),
                        pools=(
                            f"{status.get('source_zone', '?')}/"
                            f"{status.get('target_zone', '?')}"
                        ),
                        active_tokens=status.get(
                            "active_generated", status.get("visible_tokens", 0)
                        ),
                        migration=status.get("migration", "none"),
                        migrated=(
                            f"{status.get('migrations_released', 0)}/"
                            f"{status.get('migrations_started', 0)}"
                        ),
                        cancelled=status.get("migrations_cancelled", 0),
                        relief=(status.get("routing_relief") or {}).get(
                            "phase", "NORMAL"
                        ),
                    )
                    if status.get("queue_reasons"):
                        announce(
                            scene["name"], "排队原因", reasons=status["queue_reasons"]
                        )
                    if status.get("workload_pressure"):
                        announce(
                            scene["name"],
                            "负载压力",
                            pools={
                                key: {k: v for k, v in value.items() if k != "reason"}
                                for key, value in status["workload_pressure"].items()
                            },
                        )
                    elif status.get("health"):
                        announce(scene["name"], "健康参考", pools=status["health"])
                else:
                    announce(scene["name"], "等待controller初始化")
            final_path = plan["run_dir"] / "pressure_request_results.json"
            if final_path.exists() and worker.process.poll() is None:
                # A failing Python process can wait for in-flight HTTP threads.
                # The owner will stop its servers in finally, releasing them.
                tail = (plan["run_dir"] / "pressure_audit.jsonl").read_text()
                if '"outcome": "FAILED_OR_INTERRUPTED"' in tail:
                    status_path = plan["run_dir"] / "pressure_progress.json"
                    status = read(status_path) if status_path.exists() else {}
                    raise RuntimeError(
                        f"controller failed: {status.get('error')}; "
                        f"submitted={status.get('submitted', '?')}"
                    )
            time.sleep(0.2)
        if worker.process.returncode:
            raise subprocess.CalledProcessError(
                worker.process.returncode, plan["controller_argv"]
            )
        if driver is not None:
            outcome = finish_conversion(driver, root)
            if outcome["state"] != "COMPLETED":
                raise RuntimeError(f"conversion ended in {outcome['state']}")
        if plan["config"].get("transfer_mode") == "FULL_SPEED":
            common.write_json(
                plan["run_dir"] / "full_speed_receipt.json",
                verify_full_speed(plan["run_dir"]),
            )
            announce(scene["name"], "全速检查通过：历史pacing关闭，运行时限速为0")
        if before_shutdown is not None:
            before_shutdown(plan)
    except BaseException:
        (root / "scene_failure_traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        scene_failed = sys.exc_info()[0] is not None
        if driver is not None and driver.thread is not None:
            finish_conversion(driver, root)
        if monitor:
            (root / "host_monitor.stop").touch()
            with contextlib.suppress(subprocess.TimeoutExpired):
                monitor.process.wait(timeout=8)
        announce(scene["name"], "清理本场受管进程，核对退出")
        common.stop_processes(processes)
        ended = not any(common.process_group_alive(process) for process in processes)
        common.write_json(
            root / "owned_shutdown_receipt.json",
            dict(
                all_owned_groups_ended=ended,
                method="owned process-group termination",
                processes=[
                    dict(
                        name=process.name,
                        pid=process.process.pid,
                        returncode=process.process.poll(),
                    )
                    for process in processes
                ],
            ),
        )
        if not ended:
            raise RuntimeError(
                "owned server groups did not terminate; commitments unresolved"
            )
        if multiple:
            active = plan["run_dir"] / "migration_active"
            if active.is_symlink():
                common.write_json(
                    plan["run_dir"] / "migration_active_final.json",
                    dict(target=str(active.resolve()), owned_groups_ended=True),
                )
                active.unlink()
        if monitor:
            from tools.bridge_tp.monitor_pressure_host import validate_journal

            if monitor.process.returncode != 0:
                raise RuntimeError("host monitor failed; preserve evidence")
            common.write_json(
                root / "host_monitor_receipt.json",
                validate_journal(root / "host_monitor.jsonl"),
            )
            timing_path = plan["run_dir"] / "pressure_timing.jsonl"
            common.write_json(
                root / "pressure_timing_receipt.json",
                timing_receipt(timing_path, scene_failed),
            )
    return scene_result(plan["run_dir"], scene["name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    announce("preflight", "核对HEAD、分支和代码/输入/模型SHA")
    verify_spec(spec)
    common, settings = arguments(spec)
    plans = [
        build_scene(spec, scene, args.out_root, common, settings)
        for scene in spec["scenarios"]
    ]
    from tools.bridge_tp import phase8_stager
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    for plan in plans:
        argvs = [plan["source_argv"], plan["target_argv"]] + [
            extra_engine(plan, extra)[0] for extra in plan.get("extra_engines", ())
        ]
        for argv in argvs:
            actual = make_arg_parser(FlexibleArgumentParser()).parse_args(argv[3:])
            validate_parsed_serve_args(actual)
        phase8_stager.parse_args(plan["stager_argv"][2:])
        write_instances(plan)
        actual = controller.parse_args(
            plan["controller_argv"][2:] + ["--validate-only"]
        )
        actual.controller_config = Path(spec["controller_config_template"])
        controller.preflight(actual)
    if args.validate_only:
        announce("preflight", "真实serving/stager/controller参数检查通过；尚未启动GPU")
        return 0
    archive_path = args.out_root.parent / (args.out_root.name + ".tar.gz")
    if args.out_root.exists() or archive_path.exists():
        raise FileExistsError("refuse to reuse a pilot result directory")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(names) != 5 or any(
        name.strip() != "NVIDIA A100-PCIE-40GB" for name in names
    ):
        raise ValueError("five A100-PCIE-40GB GPUs required; UUIDs are not matched")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPUs already have compute processes; no server was started")
    announce("preflight", "核对GPU通过；运行rolling和会话CPU回归")
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "tests.bridge_tp.test_rolling_cutover",
            "tests.bridge_tp.test_pressure_session",
            "tests.bridge_tp.test_pressure_sessions",
            "tests.bridge_tp.test_pressure_bootstrap",
            "tests.bridge_tp.test_pressure_observation",
            "tests.bridge_tp.test_delta_batching",
            "tests.bridge_tp.test_pressure_preflight",
            "tests.bridge_tp.test_pressure_policy",
            "tests.bridge_tp.test_pressure_dispatch",
            "tests.bridge_tp.test_pressure_inventory_process",
            "tests.bridge_tp.test_pressure_compact_history",
            "tests.bridge_tp.test_phase9_state_and_proxy",
        ],
        cwd=ROOT,
        check=True,
    )
    args.out_root.mkdir(parents=True)
    (args.out_root / "gpu_inventory.txt").write_text(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv"],
            text=True,
        )
    )
    common.write_json(args.out_root / "frozen_spec.json", spec)
    results, failures, first_error_rc = [], 0, 0
    selected = list(zip(spec["scenarios"], plans))
    if args.smoke_only:
        selected = [
            (scene, plan) for scene, plan in selected if scene["name"] == "both-light"
        ]
    for index, (scene, plan) in enumerate(selected, 1):
        announce(
            "pilot", "开始场景", scenario=f"{index}/{len(selected)} {scene['name']}"
        )
        try:
            result = run_scene(spec, scene, plan, common, settings)
        except Exception as error:
            failures += 1
            real_rc = getattr(error, "returncode", 1)
            first_error_rc = first_error_rc or real_rc
            result = dict(
                status="TECHNICAL_FAILURE", error=repr(error), returncode=real_rc
            )
        results.append(dict(scene=scene["name"], **result))
        announce(
            scene["name"],
            "场景结束",
            status=result["status"],
            failures=failures,
            error=result.get("error", "none"),
        )
        common.write_json(args.out_root / "scene_results.json", results)
        if failures >= 3:
            break
    try:
        announce("pilot", "打包原始证据", scenes=len(results), failures=failures)
        from tools.bridge_tp.result_archive import pack

        packed = pack(args.out_root, archive_path)
        print(json.dumps(packed), flush=True)
    except Exception as error:
        print(f"evidence packaging failed: {error!r}", file=sys.stderr, flush=True)
        if not first_error_rc:
            raise
    return first_error_rc


if __name__ == "__main__":
    raise SystemExit(main())
