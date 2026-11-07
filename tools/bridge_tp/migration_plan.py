# SPDX-License-Identifier: Apache-2.0
"""Canonical migration plan with an explicit, single-B runtime bridge.

Stage 1 describes existing channels; it neither installs new receivers nor
changes a running controller's pool IDs, paths or source control discovery.
"""

import copy
from pathlib import Path
from urllib.parse import urlsplit


def validate_ports(blocks):
    """Reject overlaps across every supplied family, including control ports."""
    occupied = {}
    for row in blocks:
        for port in row["ports"]:
            if type(port) is not int or not 1024 <= port <= 65535:
                raise ValueError("migration port outside 1024..65535")
            if port in occupied:
                raise ValueError(
                    f"migration port {port} overlaps: {occupied[port]} / {row['owner']}"
                )
            occupied[port] = row["owner"]


def allocate_ports(sources, targets, *, gpu, delta, delivery, ready, reserved=()):
    """Allocate target-major blocks; put existing B first to retain its ports.

    Source ordering must match the existing channel base-port order. Each
    target is (instance ID, TP degree). The GPU block includes base+TP, the
    sender control socket; stager blocks contain TP rank ports.
    """
    ids = [target for target, _ in targets]
    if (
        len(set(sources)) != len(sources)
        or len(set(ids)) != len(ids)
        or set(sources) & set(ids)
        or any(not isinstance(i, str) or not i for i in [*sources, *ids])
        or any(type(tp) is not int or tp not in (1, 2, 4) for _, tp in targets)
    ):
        raise ValueError("invalid migration source/target identities or TP")
    blocks, result = list(reserved), []
    for index, (target, tp) in enumerate(targets):
        channels = []
        for source_index, source in enumerate(sources):
            base = gpu + 100 * index + 10 * source_index
            channels.append(dict(source=source, base_port=base))
            blocks.append(
                dict(
                    owner=f"gpu:{source}->{target}",
                    ports=list(range(base, base + tp + 1)),
                )
            )
        row = dict(
            instance_id=target,
            delta_base_port=delta + 10 * index,
            delivery_base_port=delivery + 10 * index,
            ready_notification_port=ready + index,
            connector_sources=channels,
        )
        result.append(row)
        for family, base, count in (
            ("delta", row["delta_base_port"], tp),
            ("delivery", row["delivery_base_port"], tp),
            ("ready", row["ready_notification_port"], 1),
        ):
            blocks.append(
                dict(owner=f"{family}:{target}", ports=list(range(base, base + count)))
            )
    validate_ports(blocks)
    return result, blocks


def _argument(argv, name):
    return argv[argv.index(name) + 1]


def describe(plan):
    """Describe actual single-B launch state, without mutating legacy fields."""
    aliases = plan["instances"]["legacy_pool_aliases"]
    rows = plan["instances"]["instances"]
    inverse = {instance: pool for pool, instance in aliases.items()}
    canonical_ids = bool(plan["instances"].get("runtime_pool_ids"))
    canonical = [
        dict(
            instance_id=row["instance_id"],
            runtime_pool_id=row["instance_id"]
            if canonical_ids
            else inverse.get(row["instance_id"], row["instance_id"]),
            bundle=row["bundle_id"],
            gpus=list(row["physical_gpu_ids"]),
            tp=row["tp_degree"],
            endpoint=row["endpoint"],
            context_limit=row["context_limit"],
        )
        for row in rows
    ]
    for original, row in zip(rows, canonical):
        side = {"tp1": "source", "tp4": "target"}.get(inverse.get(row["instance_id"]))
        row["inventory_path"] = (
            original["inventory_path"]
            if canonical_ids or not side
            else str(Path(plan["run_dir"]) / f"{side}_inventory.jsonl")
        )
        row["trace_name"] = row["instance_id"]
        row["runtime_trace_name"] = (
            row["instance_id"] if canonical_ids else side or row["instance_id"]
        )
    result = dict(
        schema="BRIDGETP_MIGRATION_PLAN_V1",
        stage="SINGLE_B_INSTANCE_IDS" if canonical_ids else "SINGLE_B_COMPATIBILITY",
        scheduler=plan.get("scheduler", "pressure"),
        instances=canonical,
        migration_sources=[],
        migration_targets=[],
        runtime_aliases=dict(aliases),
        globally_serial=True,
        multi_target_execution=False,
    )
    if set(aliases) != {"tp1", "tp4"}:
        return result
    target_id = aliases["tp4"]
    target = next(row for row in canonical if row["instance_id"] == target_id)
    if target["bundle"] != "B":
        raise ValueError("stage 1 supports the existing B receiver only")
    channels = sorted(plan["channel_sources"], key=lambda row: row["base_port"])
    source_ids = [row["source"] for row in channels]
    expected = {row["instance_id"] for row in canonical if row["tp"] == 1}
    if set(source_ids) != expected or not source_ids:
        raise ValueError("missing or unexpected single-B migration sources")
    ports, blocks = allocate_ports(
        source_ids,
        [(target_id, target["tp"])],
        gpu=channels[0]["base_port"],
        delta=int(_argument(plan["stager_argv"], "--delta-base-port")),
        delivery=int(_argument(plan["stager_argv"], "--delivery-base-port")),
        ready=plan["config"]["ready_notification_port"],
        reserved=[
            dict(
                owner=f"http:{row['instance_id']}",
                ports=[urlsplit(row["endpoint"]).port],
            )
            for row in canonical
        ]
        + [
            dict(
                owner="snapshot",
                ports=list(
                    range(
                        int(plan["source_env"]["BRIDGETP_STREAM_BASE_PORT"]),
                        int(plan["source_env"]["BRIDGETP_STREAM_BASE_PORT"])
                        + target["tp"],
                    )
                ),
            )
        ],
    )
    if [row["base_port"] for row in ports[0]["connector_sources"]] != [
        row["base_port"] for row in channels
    ]:
        raise ValueError("allocated channels differ from existing single-B ports")
    mechanism = plan["target_env"]["BRIDGETP_TAKEOVER_RUN_DIR"]
    if _argument(plan["stager_argv"], "--run-dir") != mechanism:
        raise ValueError("single-B stager and receiver directories differ")
    sources = {row["instance_id"]: row for row in rows}
    source_views = {row["instance_id"]: row for row in canonical}
    for channel in channels:
        row = sources[channel["source"]]
        path = row["targets_file"]
        entries = plan["runtime_target_files"][path]
        if len(entries) != 1 or entries[0]["target"] != target_id:
            raise ValueError("stage 1 source must list B only")
        if any(entries[0][key] != channel[key] for key in ("base_port", "generation")):
            raise ValueError("source target-file channel identity differs")
        result["migration_sources"].append(
            dict(
                source_views[channel["source"]],
                targets_file=path,
                targets=[
                    dict(
                        instance_id=target_id,
                        mechanism_dir=mechanism,
                        base_port=channel["base_port"],
                        generation=channel["generation"],
                    )
                ],
            )
        )
    result["migration_targets"] = [
        dict(
            target,
            port=urlsplit(target["endpoint"]).port,
            mechanism_dir=mechanism,
            stager_argv=list(plan["stager_argv"]),
        )
        | ports[0]
        | dict(connector_sources=copy.deepcopy(channels))
    ]
    result["port_blocks"] = blocks
    return result


def install(plan):
    """Opt the six-card unified plan into the canonical description only."""
    if len(plan.get("migration_targets", ())) > 1:
        raise ValueError("multi-target execution is not enabled in stage 1")
    if plan.get("static_cards") != 6 or plan.get("scheduler") != "unified":
        return
    description = describe(plan)
    plan["migration_plan"] = description
    plan["migration_targets"] = copy.deepcopy(description["migration_targets"])
    plan["migration_sources"] = copy.deepcopy(description["migration_sources"])


def activate_instance_ids(plan):
    """Use actual IDs on the static six-card unified runtime boundary only."""
    if (
        plan.get("static_cards") != 6
        or plan.get("scheduler") != "unified"
        or "migration_plan" not in plan
    ):
        return
    if plan["instances"].get("runtime_pool_ids"):
        raise ValueError("instance IDs already activated")
    aliases = plan["instances"]["legacy_pool_aliases"]
    plan["instances"]["runtime_pool_ids"] = dict(aliases)
    plan["canonical_instance_ids"] = True
    by_id = {row["instance_id"]: row for row in plan["instances"]["instances"]}
    for side, alias in (("source", "tp1"), ("target", "tp4")):
        if alias not in aliases:
            continue
        instance = aliases[alias]
        path = str(plan["run_dir"] / f"inventory_{instance}.jsonl")
        by_id[instance]["inventory_path"] = path
        env = plan[side + "_env"]
        env.update(
            BRIDGETP_PRESSURE_POOL=instance, BRIDGETP_PRESSURE_INVENTORY_PATH=path
        )
        if "BRIDGETP_PERFORMANCE_TRACE_DIR" in env:
            env["BRIDGETP_PERFORMANCE_TRACE_DIR"] = str(
                plan["run_dir"] / "performance_trace" / instance
            )
        argv = plan["controller_argv"]
        argv[argv.index("--" + side + "-inventory") + 1] = path
        plan["config"][side + "_pool_id"] = instance
    # Rebuild the description with the actual filenames and identities.
    install(plan)
