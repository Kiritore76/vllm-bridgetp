# SPDX-License-Identifier: Apache-2.0
"""Six-card conversions using the existing executor and owned process lifecycle."""

import copy
import json
from pathlib import Path

from tools.bridge_tp import pressure_conversion as pc
from tools.bridge_tp.conversion_channels import TargetChannelGraph
from tools.bridge_tp.conversion_sequence import SequenceBackend

ports = pc.load("tcp_ports")

PATH = {
    "L0": {"A": 1, "B": 1, "C": 1},
    "L1AC": {"A": 1, "B": 2, "C": 1},
    "L2C": {"A": 2, "B": 2, "C": 1},
    "L3": {"A": 2, "B": 2, "C": 2},
}
EDGES = {
    f"{before}-{after}": (before, after, bundle, f"TP{PATH[after][bundle]}")
    for before, after in (
        ("L0", "L1AC"),
        ("L1AC", "L0"),
        ("L1AC", "L2C"),
        ("L2C", "L1AC"),
        ("L2C", "L3"),
        ("L3", "L2C"),
    )
    for bundle in "ABC"
    if PATH[before][bundle] != PATH[after][bundle]
}


def controller_timeout_s(request, current):
    """Outwait the conversion deadline and its existing resource-ACK budget."""
    latest = max(s["trigger_s"] + s["deadline_s"] for s in request["sequence"])
    return max(current, latest + SixBackend.ack_timeout_s + 60)


def take(state, width=10):
    return ports.take(state, width)


def receiver(state, row, run):
    name = row["instance_id"]
    root = Path(run) / "migration" / name
    return dict(
        instance_id=name,
        tp=2,
        mechanism_root=str(root),
        mechanism_dir=str(root / "migration_active"),
        channel_plan=str(root / "gpu_channel_plan.json"),
        ready_notification_port=take(state),
        delta_base_port=take(state),
        delivery_base_port=take(state),
        connector_sources=[],
        endpoint=row["endpoint"],
        execution_allowed=True,
    )


def connect(state, sources, targets):
    """Preserve retained edges, allocate only new TP1->TP2 edges."""
    for target in targets.values():
        existing = {r["source"]: r for r in target["connector_sources"]}
        target["connector_sources"] = []
        for source in sorted(sources):
            channel = existing.get(source)
            if channel is None:
                state["generation"] += 1
                channel = dict(
                    source=source, base_port=take(state), generation=state["generation"]
                )
            target["connector_sources"].append(channel)


def source_fields(source, run, targets):
    run = Path(run)
    channel = next(
        (
            c
            for t in targets.values()
            for c in t["connector_sources"]
            if c["source"] == source
        ),
        dict(base_port=33900, generation=0),
    )
    return dict(
        migration_channel=dict(channel),
        targets_file=str(run / f"gpu_targets_{source}.json"),
        predictor_events=str(run / f"predictor_events_{source}.jsonl"),
        source_active=str(run / "source_active" / source),
    )


def source_entries(source, targets):
    return [
        dict(
            target=name,
            host="127.0.0.1",
            tp=2,
            base_port=c["base_port"],
            generation=c["generation"],
            mechanism_dir=t["mechanism_dir"],
        )
        for name, t in targets.items()
        for c in t["connector_sources"]
        if c["source"] == source
    ]


def receiver_engine(plan, extra, target):
    from tools.bridge_tp.migration_receivers import _remap

    argv, env = list(plan["target_argv"]), dict(plan["target_env"])
    index = argv.index("--kv-transfer-config") + 1
    old = plan["target_env"]["BRIDGETP_TAKEOVER_RUN_DIR"]
    connector = _remap(json.loads(argv[index]), old, target["mechanism_dir"])
    connector["kv_connector_extra_config"].update(
        bridgetp_gpu_direct_sources=(
            []
            if target.get("defer_channels_until_model_ready")
            else target["connector_sources"]
        ),
        bridgetp_gpu_direct_channel_plan=target["channel_plan"],
        bridgetp_ready_notification_port=target["ready_notification_port"],
        bridgetp_release_persistent_payload_buffer=False,
    )
    argv[index] = json.dumps(connector)
    argv[argv.index("--port") + 1] = str(extra["port"])
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(extra["gpus"]),
        BRIDGETP_TAKEOVER_RUN_DIR=target["mechanism_dir"],
        BRIDGETP_TAKEOVER_PERSISTENT_CHANNEL="1",
        BRIDGETP_PRESSURE_POOL=extra["instance_id"],
        BRIDGETP_PRESSURE_INVENTORY_PATH=extra["inventory_path"],
    )
    if "BRIDGETP_PERFORMANCE_TRACE_DIR" in env:
        env["BRIDGETP_PERFORMANCE_TRACE_DIR"] = str(
            Path(env["BRIDGETP_PERFORMANCE_TRACE_DIR"]).parent / extra["instance_id"]
        )
    return argv, env


def stager(plan, target):
    argv = list(plan["stager_argv"])
    for option, value in (
        ("--run-dir", target["mechanism_dir"]),
        ("--delta-base-port", target["delta_base_port"]),
        ("--delivery-base-port", target["delivery_base_port"]),
    ):
        argv[argv.index(option) + 1] = str(value)
    return argv


def initialize_target(target):
    from tools.bridge_tp.pressure_sessions import initialize
    from tools.bridge_tp.run_phase9_controller import RuntimeControl

    root = Path(target["mechanism_root"])
    if not (root / "migration_active").is_symlink():
        directory = initialize(root)
        RuntimeControl(armed=False, target_instance=target["instance_id"]).write(
            directory
        )


def initialize_links(run, sources, targets):
    from tools.bridge_tp.pressure_sessions import initialize
    from tools.bridge_tp.run_phase9_controller import RuntimeControl

    run = Path(run)
    if targets:
        active = Path(next(iter(targets.values()))["mechanism_dir"])
    else:
        parking = run / "migration" / "parked"
        active = parking / "migration_active"
        if not active.is_symlink():
            directory = initialize(parking)
            RuntimeControl(armed=False).write(directory)
    (run / "source_active").mkdir(exist_ok=True)
    for source in sources:
        link = run / "source_active" / source
        if not link.is_symlink():
            link.symlink_to(active, target_is_directory=True)


def configure(plan):
    """Run after canonical identities and serving flags are finalized."""
    if plan.get("scheduler") != "unified" or plan.get("static_cards") != 6:
        raise ValueError("six-card conversion requires six-card unified scheduler")
    rows = plan["instances"]["instances"]
    state = dict(ports.state(), generation=1000)
    targets = {
        r["instance_id"]: receiver(state, r, plan["run_dir"])
        for r in sorted(rows, key=lambda r: (r["bundle_id"] != "B", r["instance_id"]))
        if r["tp_degree"] == 2
    }
    sources = {r["instance_id"]: r for r in rows if r["tp_degree"] == 1}
    connect(state, sources, targets)
    plan.update(
        dynamic_conversion=True,
        receiver_stage=3,
        migration_targets=list(targets.values()),
        source_links={},
        six_state=state,
        six_targets=targets,
    )
    plan["config"].update(target_scoped_receivers=True, dynamic_conversion=True)
    plan["source_env"]["BRIDGETP_GPU_DIRECT_MECHANISM_ROOT"] = str(
        Path(plan["run_dir"]) / "migration"
    )
    argv = plan["controller_argv"]
    for flag in ("--no-migration", "--deferred-migration"):
        if flag in argv:
            argv.remove(flag)
    argv += ["--migration-targets", "all"]
    for target in targets.values():
        target["stager_argv"] = stager(plan, target)
        argv += [
            "--migration-target",
            f"{target['instance_id']},{target['ready_notification_port']}",
        ]
    extras = {r["instance_id"]: r for r in plan["extra_engines"]}
    aliases = plan["instances"]["legacy_pool_aliases"]
    for row in rows:
        name = row["instance_id"]
        if row["tp_degree"] == 1:
            row.update(source_fields(name, plan["run_dir"], targets))
            plan["source_links"][name] = row["source_active"]
            if name == aliases.get("tp1"):
                plan["source_env"].update(
                    BRIDGETP_GPU_DIRECT_TARGETS_FILE=row["targets_file"],
                    BRIDGETP_STREAM_RUN_DIR=row["source_active"],
                    BRIDGETP_TAKEOVER_RUN_DIR=row["source_active"],
                    BRIDGETP_PREDICTOR_LIVE_EVENTS=row["predictor_events"],
                )
                plan["source_env"].pop("BRIDGETP_GPU_DIRECT_TARGETS", None)
            else:
                extras[name].update(
                    {k: row[k] for k in source_fields(name, plan["run_dir"], targets)}
                )
        else:
            target = targets[name]
            extra = dict(
                instance_id=name,
                tp=2,
                gpus=row["physical_gpu_ids"],
                port=int(row["endpoint"].rsplit(":", 1)[1]),
                inventory_path=row["inventory_path"],
            )
            argv2, env = receiver_engine(plan, extra, target)
            if name == aliases.get("tp4"):
                plan["target_argv"], plan["target_env"] = argv2, env
            else:
                extras[name].update(receiver_argv=argv2, receiver_env=env)
    plan["sequence_context"] = dict(profile="six_card")
    plan["migration_plan"].update(
        stage="SIX_CARD_DYNAMIC_CONVERSION",
        migration_targets=copy.deepcopy(list(targets.values())),
        migration_sources=[
            dict(
                instance_id=n,
                source_active=plan["source_links"][n],
                targets=source_entries(n, targets),
            )
            for n in sources
        ],
        port_blocks=copy.deepcopy(state.get("port_blocks", [])),
        port_policy=copy.deepcopy(state["port_policy"]),
        port_probe_failures=copy.deepcopy(state.get("port_probe_failures", [])),
    )


def initialize(plan):
    targets = plan["six_targets"]
    for target in targets.values():
        initialize_target(target)
        pc.atomic_json(target["channel_plan"], target["connector_sources"])
    sources = plan["source_links"]
    initialize_links(plan["run_dir"], sources, targets)
    for source in sources:
        pc.atomic_json(
            Path(plan["run_dir"]) / f"gpu_targets_{source}.json",
            source_entries(source, targets),
        )
    return (
        Path(next(iter(targets.values()))["mechanism_dir"]).resolve()
        if targets
        else Path(plan["run_dir"]) / "migration/parked/migration_active"
    )


def cleanup(plan, write):
    from tools.bridge_tp.migration_receivers import cleanup_links

    cleanup_links(plan, write)
    active = Path(plan["run_dir"]) / "migration/parked/migration_active"
    if active.is_symlink():
        write(
            active.parent / "migration_active_final.json",
            dict(target=str(active.resolve()), owned_groups_ended=True),
        )
        active.unlink()


class SixBackend(SequenceBackend):
    ack_timeout_s = 180

    def __init__(self, *args, resources, **kwargs):
        self.resources = resources
        self.targets = copy.deepcopy(resources["six_targets"])
        self.alloc = dict(resources["six_state"])
        super().__init__(*args, **kwargs)
        self.context = dict(profile="six_card")
        self.channel_plan = None
        self.next_instance_generation = (
            max(r.ref.instance_generation for r in self.registry.instances.values()) + 1
        )
        self.channel_graph = TargetChannelGraph(
            self.run_dir,
            {
                n: dict(
                    mechanism_root=t["mechanism_root"], channel_plan=t["channel_plan"]
                )
                for n, t in self.targets.items()
            },
            {
                r["instance_id"]: r["targets_file"]
                for r in self.rows.values()
                if "targets_file" in r
            },
            clock=self.clock,
            event=self.ops.event,
        )
        self.write_layout(dict.fromkeys(self.active(), "ACTIVE"))

    def begin_step(self, plan_id, step):
        modes = {r.bundle_id: r.tp_degree for r in self.active().values()}
        after = dict(modes, **{step["bundle"]: int(step["target_mode"][-1])})
        if not any(
            modes == PATH[b] and after == PATH[a] for b, a, _, _ in EDGES.values()
        ):
            raise ValueError("not an adjacent canonical six-card conversion")
        if step.get("method", "DRAIN") != "DRAIN" or step.get("evacuate"):
            raise ValueError(
                "six-card conversions use natural drain and normal TP1 migration"
            )
        self.reset_step(plan_id)
        self.join = None
        self.evacuation_run = None
        self.gate_mode = "UNIFIED"
        self.ops.event("STEP_BEGIN", before=modes, after=after)
        self.write_layout({}, tilt=[])

    def replacement_specs(self, plan):
        if self._specs is not None:
            return self._specs
        self._specs = []
        for _, _, extra, joined in pc.ControllerConversionBackend.replacement_specs(
            self, plan
        ):
            generation = self.next_instance_generation
            self.next_instance_generation += 1
            name = f"{plan['bundle_id']}-g{generation}-tp{extra['tp']}"
            extra.update(
                instance_id=name,
                inventory_path=str(self.run_dir / f"inventory_{name}.jsonl"),
            )
            if extra["tp"] == 1:
                joined.update(source_fields(name, self.run_dir, {}))
                extra.update(joined)
            else:
                target = receiver(
                    self.alloc,
                    dict(instance_id=name, endpoint=extra["endpoint"]),
                    self.run_dir,
                )
                extra["migration_receiver"] = target
            self.next_service_port = max(self.next_service_port, extra["port"] + 1)
            self._specs.append((name, generation, extra, joined))
        return self._specs

    def track_receiver(self, target):
        target["stager_argv"] = stager(self.resources, target)
        if not any(
            t["instance_id"] == target["instance_id"]
            for t in self.resources["migration_targets"]
        ):
            self.resources["migration_targets"].append(target)

    def prestart(self, plan):
        if not self.prestart_enabled or self.prestarted is not None:
            return
        retiring = {r.instance_id for r in self.refs(plan)}
        retained = {n: t for n, t in self.targets.items() if n not in retiring}
        for name, _, extra, _ in self.replacement_specs(plan):
            if extra["tp"] == 2:
                target = extra["migration_receiver"]
                initialize_target(target)
                pc.atomic_json(target["channel_plan"], [])
                self.track_receiver(target)
            else:
                initialize_links(self.run_dir, {name: extra}, retained)
                pc.atomic_json(extra["targets_file"], [])
                self.resources["source_links"][name] = extra["source_active"]
        super().prestart(plan)

    def start_replacements(self, plan):
        retiring = {r.instance_id for r in self.refs(plan)}
        for target in list(self.targets):
            if target in retiring:
                self.channel_graph.remove_target(target)
                del self.targets[target]
        specs = self.replacement_specs(plan)
        first_target = not self.targets and any(e["tp"] == 2 for _, _, e, _ in specs)
        for name, _, extra, _ in specs:
            if extra["tp"] == 2:
                target = extra["migration_receiver"]
                self.targets[name] = target
                if first_target:
                    target["defer_channels_until_model_ready"] = True
                    self.ops.event("CHANNEL_TOPOLOGY_DEFERRED", instance=name)
                initialize_target(target)
                pc.atomic_json(target["channel_plan"], [])
                self.channel_graph.add_target(
                    name,
                    mechanism_root=target["mechanism_root"],
                    channel_plan=target["channel_plan"],
                )
        sources = {
            n: r
            for n, r in self.rows.items()
            if n not in retiring
            and r["tp_degree"] == 1
            and self.registry.instances[n].lifecycle == "ACTIVE"
        }
        sources.update(
            {
                n: dict(tp_degree=1, **joined)
                for n, _, e, joined in specs
                if e["tp"] == 1
            }
        )
        connect(self.alloc, sources, self.targets)
        initialize_links(self.run_dir, sources, self.targets)
        for source, row in sources.items():
            self.resources["source_links"][source] = row["source_active"]
            self.channel_graph.source_files[source] = Path(row["targets_file"])
            visible = {
                n: t
                for n, t in self.targets.items()
                if not t.get("defer_channels_until_model_ready")
            }
            pc.atomic_json(row["targets_file"], source_entries(source, visible))
        for target in self.targets.values():
            pc.atomic_json(
                target["channel_plan"],
                []
                if target.get("defer_channels_until_model_ready")
                else target["connector_sources"],
            )
        for _, _, extra, _ in specs:
            if extra["tp"] == 2:
                target = extra["migration_receiver"]
                self.track_receiver(target)
                self.ops.start_receiver(target)
        self.joined_at = self.clock()
        super().start_replacements(plan)

    def retire_channels(self, plan):
        ids = {r.instance_id for r in self.refs(plan)}
        if any(self.layout.get(i, {}).get("lifecycle") != "RETIRED" for i in ids):
            self.write_layout(dict.fromkeys(ids, "RETIRED"))
        status = self.status()
        if (
            not self.fresh(status)
            or status.get("migration_resource_targets") is None
            or ids & set(status["migration_resource_targets"])
            or status.get("migration_resource_sources") is None
            or ids & set(status["migration_resource_sources"])
        ):
            return False
        return pc.ControllerConversionBackend.retire_channels(self, plan)

    def settle_before_abort(self, plan):
        # Session identities are permanently retired by controller ACK. Once
        # that boundary is published, reviving the old id is not a safe abort.
        if any(
            self.layout.get(r.instance_id, {}).get("lifecycle") == "RETIRED"
            for r in self.refs(plan)
        ):
            plan.update(state="RECOVERY_REQUIRED", destruction_started=True)
            self.ops.event(
                "LOGICAL_RETIREMENT_CANNOT_ROLL_BACK", plan_id=plan["plan_id"]
            )
            return False
        return True

    def step_acknowledged(self):
        if not super().step_acknowledged():
            return False
        desired = pc.read_json(self.run_dir / "layout_control.json")[
            "migration_resources"
        ]
        status = self.status()
        return set(status.get("migration_resource_targets") or ()) == set(
            desired["targets"]
        ) and set(status.get("migration_resource_sources") or ()) == set(
            desired["source_links"]
        )

    def warm_and_verify(self, plan):
        if not super().warm_and_verify(plan):
            return False
        # Only the first receiver starts with an empty topology. Its model/KV
        # warmup must finish before live sources begin communicator creation.
        for name, target in self.targets.items():
            if target.get("defer_channels_until_model_ready"):
                pc.atomic_json(target["channel_plan"], target["connector_sources"])
                for source, row in self.rows.items():
                    if (
                        row.get("targets_file")
                        and self.registry.instances[source].lifecycle == "ACTIVE"
                    ):
                        pc.atomic_json(
                            row["targets_file"], source_entries(source, self.targets)
                        )
                target["defer_channels_until_model_ready"] = False
                self.ops.event(
                    "CHANNEL_TOPOLOGY_PUBLISHED",
                    instance=name,
                    sources=[c["source"] for c in target["connector_sources"]],
                )
        missing = []
        for name, target in self.targets.items():
            receipts = pc.preconnect_receipts(Path(target["mechanism_root"]))
            for channel in target["connector_sources"]:
                source = channel["source"]
                names = [f"source-{source}-{name}.json"] + [
                    f"tp_rank_{r}-{source}.json" for r in range(2)
                ]
                if any(
                    receipts.get(k, {}).get("status") != "WARMUP_COMPLETE"
                    or receipts[k].get("channel_generation") != channel["generation"]
                    for k in names
                ):
                    diagnostics = {}
                    for path in Path(target["mechanism_root"]).glob(
                        "sessions/*/gpu_channel_connection_receipts/"
                        f"tp_rank_*-{channel['generation']}.json"
                    ):
                        value = pc.read_json(path) or {}
                        rank = value.get("target_tp_rank")
                        if value.get("updated_unix_s", 0) >= diagnostics.get(
                            rank, {}
                        ).get("updated_unix_s", 0):
                            diagnostics[rank] = value
                    sender_paths = list(
                        Path(target["mechanism_root"]).glob(
                            f"sessions/*/gpu_channel_connection_receipts/source-{source}-{name}*.json"
                        )
                    )
                    sender_progress = [pc.read_json(p) for p in sender_paths]
                    missing.append(
                        dict(
                            sender_progress=sender_progress,
                            target=name,
                            source=source,
                            generation=channel["generation"],
                            base_port=channel["base_port"],
                            missing_receipts=[
                                k
                                for k in names
                                if receipts.get(k, {}).get("status")
                                != "WARMUP_COMPLETE"
                                or receipts[k].get("channel_generation")
                                != channel["generation"]
                            ],
                            connection_progress=list(diagnostics.values()),
                        )
                    )
        if missing:
            if self.clock() - self.joined_at > 780:
                raise TimeoutError(
                    "six-card new channel warmup incomplete: "
                    + json.dumps(missing, sort_keys=True)
                )
            return False
        return self.channel_graph.acknowledged()

    def write_layout(self, changes, **kwargs):
        revision = super().write_layout(changes, **kwargs)
        value = pc.read_json(self.run_dir / "layout_control.json")
        value["migration_resources"] = dict(
            targets={
                n: t
                for n, t in self.targets.items()
                if self.layout.get(n, {}).get("lifecycle") in {"ACTIVE", "QUIESCING"}
            },
            source_links={
                n: r["source_active"]
                for n, r in self.rows.items()
                if r.get("source_active")
                and self.layout.get(n, {}).get("lifecycle") in {"ACTIVE", "QUIESCING"}
            },
        )
        pc.atomic_json(self.run_dir / "layout_control.json", value)
        return revision


def assess(scene, expected_requests, require_conversion=True, *, split_probe=False):
    from tools.bridge_tp.conversion_sequence import assess as sequence_assess
    from tools.bridge_tp.six_split_probe import PROFILE, coverage

    value = sequence_assess(scene, expected_requests, decided=True)
    sequence = pc.read_json(Path(scene) / "conversion_sequence_result.json") or {}
    steps = sequence.get("steps", [])
    value["all_steps_completed"] = (bool(steps) or not require_conversion) and all(
        s["plan"]["state"] == "COMPLETED" and s.get("layout") in PATH.values()
        for s in steps
    )
    value["mechanism_passed"] = value["passed"] and value["all_steps_completed"]
    scene = Path(scene)
    manifest = pc.read_json(scene / "instances.json") or {}
    split_probe = split_probe or manifest.get("diagnostic_arrival_profile") == PROFILE
    audit = [
        json.loads(line)
        for line in (scene / "controller/pressure_audit.jsonl").read_text().splitlines()
    ]
    observed = coverage(
        sequence,
        [
            json.loads(line)
            for line in (scene / "conversion_events.jsonl").read_text().splitlines()
        ],
        pc.read_json(scene / "controller/pressure_request_results.json"),
        [e for e in audit if e["kind"] == "pressure_actual_start"],
        (pc.read_json(scene / "closed_loop_acceptance.json") or {}).get("sessions", []),
    )
    value.update(observed)
    value["migration_coverage"] = observed["actual_migration_coverage"]
    value["migration_coverage_required"] = split_probe
    value["coverage_passed"] = not split_probe or observed["split_probe_coverage"]
    value["passed"] = value["mechanism_passed"] and value["coverage_passed"]
    return value
