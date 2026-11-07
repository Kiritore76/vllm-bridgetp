# SPDX-License-Identifier: Apache-2.0
"""Stage 2: isolated A/B receivers and warm channels; execution stays on B."""

import copy
import json
from pathlib import Path
from urllib.parse import urlsplit

from tools.bridge_tp.migration_plan import _argument, allocate_ports


def _remap(value, old, new):
    if isinstance(value, dict):
        return {k: _remap(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_remap(v, old, new) for v in value]
    if isinstance(value, str) and value.startswith(old):
        return new + value[len(old) :]
    return value


def normalized_receiver_command(command, target):
    """Compare serving flags while allowing only owned identities and ports."""
    command = list(command)
    command[command.index("--port") + 1] = "HTTP_PORT"
    index = command.index("--kv-transfer-config") + 1
    connector = _remap(json.loads(command[index]), target["mechanism_dir"], "MECHANISM")
    settings = connector["kv_connector_extra_config"]
    for key in (
        "bridgetp_preconnect_gpu_direct_base_port",
        "bridgetp_ready_notification_port",
    ):
        settings[key] = "PORT"
    for channel in settings["bridgetp_gpu_direct_sources"]:
        channel["base_port"] = "PORT"
    command[index] = json.dumps(connector, sort_keys=True)
    return command


def configure(plan):
    """Install receive capability for A and B, without selecting A."""
    if (
        not plan.get("canonical_instance_ids")
        or plan.get("scheduler") != "unified"
        or plan.get("static_cards") != 6
        or "--maximum-migrations" not in plan["controller_argv"]
        or plan.get("conversion")
    ):
        raise ValueError(
            "receiver stage needs static six-card unified rolling sessions"
        )
    if _argument(plan["controller_argv"], "--maximum-migrations") == "1":
        raise ValueError("receiver stage requires rolling session directories")
    rows = plan["instances"]["instances"]
    targets = {row["bundle_id"]: row for row in rows if row["tp_degree"] == 2}
    sources = sorted(plan["channel_sources"], key=lambda r: r["base_port"])
    if set(targets) != {"A", "B"} or len(sources) != 2:
        raise ValueError("receiver stage currently supports L2C only")
    env = plan["source_env"]
    if any(
        env.get(k) != "1"
        for k in ("BRIDGETP_GPU_DIRECT_HISTORY", "BRIDGETP_GPU_DIRECT_DELTA")
    ):
        raise ValueError("target-scoped receivers require GPU-direct history and delta")
    old = plan["target_env"]["BRIDGETP_TAKEOVER_RUN_DIR"]
    roots = plan["run_dir"] / "migration"
    target_rows = [targets["B"], targets["A"]]
    ports, blocks = allocate_ports(
        [s["source"] for s in sources],
        [(t["instance_id"], t["tp_degree"]) for t in target_rows],
        gpu=sources[0]["base_port"],
        delta=int(_argument(plan["stager_argv"], "--delta-base-port")),
        delivery=int(_argument(plan["stager_argv"], "--delivery-base-port")),
        ready=plan["config"]["ready_notification_port"],
        reserved=[
            b
            for b in plan["migration_plan"]["port_blocks"]
            if b["owner"].startswith(("http:", "snapshot"))
        ],
    )
    base_argv, base_env = list(plan["target_argv"]), dict(plan["target_env"])
    extras = {row["instance_id"]: row for row in plan["extra_engines"]}
    descriptions = []
    for row, allocated in zip(target_rows, ports):
        name = row["instance_id"]
        root = roots / name
        active = str(root / "migration_active")
        argv, receiver_env = list(base_argv), dict(base_env)
        index = argv.index("--kv-transfer-config") + 1
        connector = _remap(json.loads(argv[index]), old, active)
        settings = connector["kv_connector_extra_config"]
        channels = [
            dict(c, generation=sources[n]["generation"])
            for n, c in enumerate(allocated["connector_sources"])
        ]
        settings.update(
            bridgetp_gpu_direct_sources=channels,
            bridgetp_preconnect_gpu_direct_base_port=channels[0]["base_port"],
            bridgetp_ready_notification_port=allocated["ready_notification_port"],
            bridgetp_release_persistent_payload_buffer=False,
        )
        argv[index] = json.dumps(connector)
        argv[argv.index("--port") + 1] = str(urlsplit(row["endpoint"]).port)
        receiver_env.update(
            CUDA_VISIBLE_DEVICES=",".join(row["physical_gpu_ids"]),
            BRIDGETP_TAKEOVER_RUN_DIR=active,
            BRIDGETP_TAKEOVER_PERSISTENT_CHANNEL="1",
            BRIDGETP_PRESSURE_POOL=name,
            BRIDGETP_PRESSURE_INVENTORY_PATH=row["inventory_path"],
            BRIDGETP_PERFORMANCE_TRACE_DIR=str(
                plan["run_dir"] / "performance_trace" / name
            ),
        )
        stager = list(plan["stager_argv"])
        for option, value in (
            ("--run-dir", active),
            ("--delta-base-port", allocated["delta_base_port"]),
            ("--delivery-base-port", allocated["delivery_base_port"]),
        ):
            stager[stager.index(option) + 1] = str(value)
        description = dict(
            instance_id=name,
            bundle=row["bundle_id"],
            tp=2,
            gpus=list(row["physical_gpu_ids"]),
            endpoint=row["endpoint"],
            port=urlsplit(row["endpoint"]).port,
            runtime_pool_id=name,
            inventory_path=row["inventory_path"],
            trace_name=name,
            mechanism_root=str(root),
            mechanism_dir=active,
            stager_argv=stager,
            connector_sources=channels,
            delta_base_port=allocated["delta_base_port"],
            delivery_base_port=allocated["delivery_base_port"],
            ready_notification_port=allocated["ready_notification_port"],
            execution_allowed=row["bundle_id"] == "B",
        )
        descriptions.append(description)
        row["migration_receiver"] = True
        if row["bundle_id"] == "B":
            plan.update(target_argv=argv, target_env=receiver_env, stager_argv=stager)
        else:
            extras[name].update(receiver_argv=argv, receiver_env=receiver_env)
    plan["source_links"] = {}
    for source in sources:
        name = source["source"]
        path = plan["run_dir"] / "source_active" / name
        plan["source_links"][name] = str(path)
        row = next(r for r in rows if r["instance_id"] == name)
        entries = []
        for target in descriptions:
            channel = next(
                c for c in target["connector_sources"] if c["source"] == name
            )
            entries.append(
                dict(
                    target=target["instance_id"],
                    host="127.0.0.1",
                    tp=2,
                    base_port=channel["base_port"],
                    generation=channel["generation"],
                    mechanism_dir=target["mechanism_dir"],
                )
            )
        plan["runtime_target_files"][row["targets_file"]] = entries
        source_view = next(
            r for r in plan["migration_sources"] if r["instance_id"] == name
        )
        source_view["targets"] = [
            dict(
                instance_id=e["target"],
                **{k: e[k] for k in ("base_port", "generation", "mechanism_dir")},
            )
            for e in entries
        ]
        source_view["source_active"] = str(path)
        if name == plan["instances"]["legacy_pool_aliases"]["tp1"]:
            plan["source_env"].update(
                BRIDGETP_STREAM_RUN_DIR=str(path), BRIDGETP_TAKEOVER_RUN_DIR=str(path)
            )
        else:
            extras[name]["source_active"] = str(path)
    plan["source_env"]["BRIDGETP_GPU_DIRECT_MECHANISM_ROOT"] = str(roots)
    selected = descriptions[0]
    plan["config"]["target_scoped_receivers"] = True
    plan["controller_argv"] += [
        "--mechanism-root",
        selected["mechanism_root"],
        "--migration-targets",
        "B",
    ]
    plan["receiver_stage"] = 2
    plan["migration_targets"] = descriptions
    plan["migration_plan"].update(
        stage="AB_RECEIVERS_B_ONLY",
        migration_targets=copy.deepcopy(descriptions),
        migration_sources=copy.deepcopy(plan["migration_sources"]),
        port_blocks=blocks,
        execution_targets=[selected["instance_id"]],
        multi_target_execution=False,
    )


def initialize(plan):
    """Create owned target sessions, then the stable source entry points."""
    from tools.bridge_tp.pressure_sessions import initialize as initial_session
    from tools.bridge_tp.run_phase9_controller import RuntimeControl

    selected = None
    for target in plan["migration_targets"]:
        root = Path(target["mechanism_root"])
        if (
            root.parent != plan["run_dir"] / "migration"
            or root.name != target["instance_id"]
        ):
            raise ValueError("receiver escaped owned migration root")
        root.mkdir(parents=True, exist_ok=False)
        session = initial_session(root)
        RuntimeControl(armed=False, target_instance=target["instance_id"]).write(
            session
        )
        if target["execution_allowed"]:
            selected = session
    if selected is None:
        raise ValueError("no selected B receiver")
    target = next(t for t in plan["migration_targets"] if t["execution_allowed"])
    for value in plan["source_links"].values():
        link = Path(value)
        if link.parent != plan["run_dir"] / "source_active":
            raise ValueError("source entry escaped owned directory")
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(Path(target["mechanism_dir"]), target_is_directory=True)
    return selected


def warm_channels(plan, receipts, timeout_s=30):
    import time

    deadline = time.monotonic() + timeout_s
    while True:
        found, missing = {}, []
        for target in plan["migration_targets"]:
            rows = receipts(Path(target["mechanism_root"]))
            for source in target["connector_sources"]:
                names = [f"source-{source['source']}-{target['instance_id']}.json"] + [
                    f"tp_rank_{rank}-{source['source']}.json"
                    for rank in range(target["tp"])
                ]
                for name in names:
                    row = rows.get(name, {})
                    if (
                        row.get("status") != "WARMUP_COMPLETE"
                        or row.get("channel_generation") != source["generation"]
                    ):
                        missing.append((target["instance_id"], name))
                    else:
                        found[target["instance_id"] + "/" + name] = row
        if not missing:
            return found
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"target-scoped warm channel proof incomplete: {missing}"
            )
        time.sleep(0.2)


def cleanup_links(plan, write):
    """Called only after owned processes stop; preserve all actual sessions."""
    for target in plan["migration_targets"]:
        active = Path(target["mechanism_dir"])
        if (
            active
            != plan["run_dir"]
            / "migration"
            / target["instance_id"]
            / "migration_active"
        ):
            raise ValueError("receiver cleanup escaped owned directory")
        if active.is_symlink():
            write(
                Path(target["mechanism_root"]) / "migration_active_final.json",
                dict(target=str(active.resolve()), owned_groups_ended=True),
            )
            active.unlink()
    for value in plan["source_links"].values():
        link = Path(value)
        if link.parent != plan["run_dir"] / "source_active":
            raise ValueError("source cleanup escaped owned directory")
        if link.is_symlink():
            link.unlink()
