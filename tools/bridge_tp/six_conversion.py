# SPDX-License-Identifier: Apache-2.0
"""Six-card conversions using the existing executor and owned process lifecycle."""

import copy
import json
import threading
from pathlib import Path

from tools.bridge_tp import pressure_conversion as pc
from tools.bridge_tp.conversion_channels import TargetChannelGraph
from tools.bridge_tp.conversion_sequence import SequenceBackend

ports = pc.load("tcp_ports")

# Engine reuse (user 11-11). A GPU whose engines all sleep keeps their CUDA
# context and buffers: 525-1214 MiB in the two-GPU probe of 11-11, plus the
# BridgeTP connector here. More than this means the sleep did not free it.
SLEEP_RESIDUAL_MIB = 4096
SLEEP_TIMEOUT_S = 120

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
    if (plan.get("conversion") or {}).get("engine_reuse") == "sleep":
        # Engine reuse needs every engine sleep-capable from its first start
        # (cumem allocator); the dev endpoints stay on the local host.
        for key in ("source_argv", "target_argv"):
            if "--enable-sleep-mode" not in plan[key]:
                plan[key].append("--enable-sleep-mode")
        for key in ("source_env", "target_env"):
            plan[key]["VLLM_SERVER_DEV_MODE"] = "1"
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


def recompute_choice(instances, refs, waited_s, safety, stalled=frozenset()):
    """(reason, None) or (None, choice): which requests a merge recomputes.

    Every request still on the group goes, or none: the drain ends with the
    last one. Triggers only on what is observed now (drain time, request
    count, context so far, whose output stopped growing); it never predicts a
    remaining length. ``stalled`` requests go too but count toward neither
    the request cap nor the token budget.
    """
    if waited_s < safety["recompute_after_s"]:
        return "drain younger than the recompute delay", None
    if any(instances[i].get("sessions") for i in refs):
        return "a migration session touches the group", None
    rows = [
        dict(
            request_id=r["request_id"],
            instance=i,
            context_tokens=r["prompt_tokens"] + r["generated"],
            stalled=r["request_id"] in stalled,
        )
        for i in refs
        for r in instances[i]["requests"]
    ]
    if not rows:
        return "group already empty", None
    if sum(instances[i].get("streams_pending", 0) for i in refs) != len(rows):
        return "controller streams differ from native inventory", None
    moving = [r for r in rows if not r["stalled"]]
    if len(moving) > safety["recompute_max_requests"]:
        return (
            f"{len(moving)} progressing requests exceed the recompute cap "
            f"{safety['recompute_max_requests']}: natural drain"
        ), None
    tokens = sum(r["context_tokens"] for r in moving)
    if tokens > safety["recompute_max_tokens"]:
        return f"{tokens} context tokens exceed the recompute budget", None
    return None, dict(
        requests=rows,
        context_tokens=sum(r["context_tokens"] for r in rows),
        counted_tokens=tokens,
        stalled=sum(r["stalled"] for r in rows),
    )


class SixBackend(SequenceBackend):
    ack_timeout_s = 180

    def __init__(self, *args, resources, **kwargs):
        self.resources = resources
        self.targets = copy.deepcopy(resources["six_targets"])
        self.alloc = dict(resources["six_state"])
        super().__init__(*args, **kwargs)
        self.context = dict(profile="six_card")
        self.channel_plan = None
        # Engine reuse: None, or "sleep" (set by the launcher from the request).
        self.engine_reuse = None
        self.sleeping = {}  # instance id -> what its wake needs
        self.sleep_job = None
        self.sleep_fallback = set()  # plan ids whose sleep failed
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
        if (
            step.get("method", "DRAIN") not in {"DRAIN", "SELECTIVE_ROLL"}
            or step.get("evacuate")
            or (step.get("method") == "SELECTIVE_ROLL" and step["target_mode"] != "TP2")
        ):
            raise ValueError(
                "six-card conversions support drain or TP1-to-TP2 assistance only"
            )
        self.reset_step(plan_id)
        self.join = None
        self.evacuation_run = None
        self.gate_mode = "UNIFIED"
        self.assist_priority_sources = (
            [i for i, row in self.active().items() if row.bundle_id == step["bundle"]]
            if step.get("method") == "SELECTIVE_ROLL"
            else []
        )
        self.ops.event("STEP_BEGIN", before=modes, after=after)
        self.write_layout({}, tilt=[])

    def recompute_step(self, plan, now):
        """Request the group's recompute once (see ``recompute_choice``)."""
        if not getattr(self, "recompute_enabled", False) or getattr(
            self, "layout_recompute", None
        ):
            return
        status = self.status()
        if not self.fresh(status):
            return
        instances = status["instances"]
        refs = [r.instance_id for r in self.refs(plan)]
        if any(instances.get(i, {}).get("tp_degree") != 1 for i in refs):
            return  # only a TP1-to-TP2 merge drains behind long requests
        clock = self.clock()
        if self.recompute_since is None:
            self.recompute_since = clock
        # Last time each group request's output grew (observed every tick).
        progress = getattr(self, "recompute_progress", None)
        if progress is None:
            progress = self.recompute_progress = {}
        seen = {}
        for i in refs:
            for row in instances[i]["requests"]:
                old = progress.get(row["request_id"])
                grew = old is None or row["generated"] != old[0]
                seen[row["request_id"]] = (
                    (row["generated"], clock) if grew else old
                )
        progress.clear()
        progress.update(seen)
        stalled = frozenset(
            k
            for k, (_, since) in progress.items()
            if clock - since >= self.safety["recompute_stall_s"]
        )
        result = status.get("assist_result")
        waited = clock - self.recompute_since
        if waited >= self.safety["recompute_after_s"] and (
            self.assist_seq > self.assist_step_start_seq
            and (result is None or result["seq"] != self.assist_seq)
        ):
            reason, choice = "an assist START is unanswered", None
        else:
            reason, choice = recompute_choice(
                instances, refs, waited, self.safety, stalled
            )
        if choice is None:
            if reason != self.last_recompute_reason:
                self.decision("RECOMPUTE", choice=None, reason=reason)
                self.last_recompute_reason = reason
            return
        self.recompute_seq += 1
        self.layout_recompute = dict(
            seq=self.recompute_seq,
            plan_id=plan["plan_id"],
            bundle=plan["bundle_id"],
            drained_s=round(clock - self.recompute_since, 3),
            **choice,
        )
        self.decision("RECOMPUTE", choice=self.layout_recompute, reason=None)
        self.write_layout({})
        self.ops.event("RECOMPUTE_REQUESTED", **self.layout_recompute)

    def assist_choice(self, plan, status):
        """Optional TP1-to-TP2 help; refusal never gates natural drainage."""
        if not self.fresh(status):
            return "status stale", None
        if getattr(self, "layout_recompute", None) is not None:
            return "group requests are being recomputed", None
        result = status.get("assist_result")
        if self.assist_seq > self.assist_step_start_seq:
            if result is None or result["seq"] != self.assist_seq:
                return "previous assist not answered yet", None
            if (
                result["state"] == "STARTED"
                and result["request_id"] not in self.assisted
            ):
                self.assisted.append(result["request_id"])
        instances = status["instances"]
        refs = [r.instance_id for r in self.refs(plan)]
        if any(instances[i].get("tp_degree") != 1 for i in refs):
            return "assist only supports TP1 sources", None
        targets = [
            i
            for i, row in instances.items()
            if i not in refs
            and row.get("tp_degree") == 2
            and row.get("lifecycle") == "ACTIVE"
        ]
        if not targets:
            return "no active retained TP2 target", None
        if any(v.get("sessions") for v in instances.values()):
            return "migration session in progress", None
        states = list((status.get("assist_sessions") or {}).values())
        committed = max(
            0,
            sum(s in {"COMMITTED", "ACCOUNTED", "COMPLETED"} for s in states)
            - self.assist_committed_baseline,
        )
        if (
            committed >= self.safety["assist_max_committed"]
            or len(self.assisted) >= self.safety["assist_max_attempts"]
        ):
            return "assist budget reached; natural drain continues", None
        if status.get("release_budget_s") is None:
            return "no measured transfer budget", None
        candidates, _ = self.long_requests(plan, status)
        candidates = [c for c in candidates if c["request_id"] not in self.assisted]
        if not candidates:
            return "no startable long remaining request", None
        # The controller ranks all live target-specific candidates at START.
        return None, dict(candidates[0], current_length=True)

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
        if self.wake_set(plan) is not None:
            return  # nothing to pre-start: sleeping engines wake instead
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

    # ---------------------------------------------------------- engine reuse
    def reuse_on(self, plan):
        return (
            self.engine_reuse == "sleep" and plan["plan_id"] not in self.sleep_fallback
        )

    def retire_label(self, plan):
        return "SLEEPING" if self.reuse_on(plan) else "RETIRED"

    def reclaimed_state(self, plan):
        return self.retire_label(plan)

    def wake_set(self, plan):
        """Sleeping engines that are exactly this conversion's target form."""
        if self.engine_reuse != "sleep":
            return None
        tp = int(plan["target_bundle_mode"][2:])
        gpus = sorted(
            g for r in self.refs(plan) for g in self.registry.get(r).physical_gpu_ids
        )
        names = sorted(
            n
            for n, e in self.sleeping.items()
            if e["bundle"] == plan["bundle_id"] and e["tp"] == tp
        )
        found = sorted(g for n in names for g in self.sleeping[n]["gpus"])
        if not names or found != gpus or len(names) != (1 if tp > 1 else len(gpus)):
            return None
        return names

    def stop_owned_engines(self, plan):
        if not self.reuse_on(plan):
            return super().stop_owned_engines(plan)
        ids = [r.instance_id for r in self.refs(plan)]
        job = dict(plan_id=plan["plan_id"], ids=ids, done=False, error=None)
        job["started"] = self.clock()
        self.sleep_job = job

        def run():
            try:
                job["seconds"] = self.ops.sleep(
                    {i: self.registry.instances[i].endpoint for i in ids}
                )
            except Exception as error:  # the next poll falls back to a stop
                job["error"] = repr(error)
            job["done"] = True

        threading.Thread(target=run, daemon=True).start()
        self.ops.event("SLEEP_START", plan_id=plan["plan_id"], instances=ids)

    def resources_released(self, plan):
        job = self.sleep_job
        if job is None or job["plan_id"] != plan["plan_id"]:
            return super().resources_released(plan)
        if not job["done"]:
            if self.clock() - job["started"] <= SLEEP_TIMEOUT_S:
                return False
            job["error"] = "sleep timed out"
        refs = self.refs(plan)
        gpus = sorted(g for r in refs for g in self.registry.get(r).physical_gpu_ids)
        used = None
        if job["error"] is None:
            try:
                used = self.ops.gpu_memory_used(gpus)
            except Exception as error:
                job["error"] = repr(error)
        if job["error"] is None and any(used[g] > SLEEP_RESIDUAL_MIB for g in gpus):
            job["error"] = "GPU memory not released by sleep"
        if job["error"] is not None:
            # Never two awake engines on a GPU: stop the old ones and continue
            # as a cold conversion (the controller ends them RETIRED).
            self.ops.event(
                "SLEEP_FALLBACK",
                plan_id=plan["plan_id"],
                reason=job["error"],
                used_mib=used,
            )
            self.sleep_fallback.add(plan["plan_id"])
            self.sleep_job = None
            self.write_layout(dict.fromkeys(job["ids"], "RETIRED"))
            self.ops.stop(job["ids"])
            return False
        for ref in refs:
            row = self.registry.get(ref)
            self.sleeping[ref.instance_id] = dict(
                bundle=row.bundle_id,
                tp=row.tp_degree,
                gpus=list(row.physical_gpu_ids),
                endpoint=row.endpoint,
                context_limit=row.context_limit,
                inventory_path=self.rows[ref.instance_id]["inventory_path"],
                target=self.targets.get(ref.instance_id),
            )
        self.sleep_job = None
        self.ops.event(
            "RESOURCES_RELEASED", plan_id=plan["plan_id"], gpus=gpus, slept=True
        )
        self.ops.event(
            "SLEPT",
            plan_id=plan["plan_id"],
            instances=job["ids"],
            sleep_s=job.get("seconds"),
            wait_s=round(self.clock() - job["started"], 3),
            used_mib=used,
        )
        return True

    def wake_replacements(self, plan, names):
        """Wake the sleeping target form; channels open once each is awake."""
        retiring = {r.instance_id for r in self.refs(plan)}
        for target in list(self.targets):
            if target in retiring:
                self.channel_graph.remove_target(target)
                del self.targets[target]
        for name in names:
            entry = self.sleeping.pop(name)
            self.registry.transition(self.registry.instances[name].ref, "WAKING")
            if entry["tp"] == 2:
                target = entry["target"]
                # Fresh edges only: connect() gives every source a new generation.
                target["connector_sources"] = []
                target["defer_channels_until_model_ready"] = True
                self.targets[name] = target
                pc.atomic_json(target["channel_plan"], [])
                self.channel_graph.add_target(
                    name,
                    mechanism_root=target["mechanism_root"],
                    channel_plan=target["channel_plan"],
                    wake=True,
                )
            self.started[name] = dict(
                instance_id=name,
                bundle=entry["bundle"],
                tp=entry["tp"],
                gpus=entry["gpus"],
                endpoint=entry["endpoint"],
                port=int(entry["endpoint"].rsplit(":", 1)[1]),
                context_limit=entry["context_limit"],
                inventory_path=entry["inventory_path"],
                wake=True,
                channels_published=entry["tp"] == 2,
            )
            self.ops.event("WAKE_START", instance=name, gpus=entry["gpus"])
        sources = {
            n: r
            for n, r in self.rows.items()
            if n not in retiring
            and r["tp_degree"] == 1
            and self.registry.instances[n].lifecycle in {"ACTIVE", "WAKING"}
        }
        connect(self.alloc, sources, self.targets)
        for source, row in sources.items():
            self.resources["source_links"][source] = row["source_active"]
            self.channel_graph.source_files[source] = Path(row["targets_file"])
            waking = self.registry.instances[source].lifecycle == "WAKING"
            visible = {
                n: t
                for n, t in self.targets.items()
                if not t.get("defer_channels_until_model_ready")
            }
            pc.atomic_json(
                row["targets_file"], [] if waking else source_entries(source, visible)
            )
        for target in self.targets.values():
            pc.atomic_json(
                target["channel_plan"],
                []
                if target.get("defer_channels_until_model_ready")
                else target["connector_sources"],
            )
        self.joined_at = self.clock()

    def start_replacements(self, plan):
        names = self.wake_set(plan)
        if names is not None:
            return self.wake_replacements(plan, names)
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
        label = self.retire_label(plan)
        if any(self.layout.get(i, {}).get("lifecycle") != label for i in ids):
            self.write_layout(dict.fromkeys(ids, label))
        status = self.status()
        # A sleeping group keeps its controller migration group and source
        # entry (a retired identity can never return); lifecycle gates it.
        if (
            not self.fresh(status)
            or status.get("migration_resource_targets") is None
            or status.get("migration_resource_sources") is None
            or label == "RETIRED"
            and (
                ids & set(status["migration_resource_targets"])
                or ids & set(status["migration_resource_sources"])
            )
        ):
            return False
        return pc.ControllerConversionBackend.retire_channels(self, plan)

    def settle_before_abort(self, plan):
        # Session identities are permanently retired by controller ACK. Once
        # that boundary is published, reviving the old id is not a safe abort.
        if any(
            self.layout.get(r.instance_id, {}).get("lifecycle")
            in {"RETIRED", "SLEEPING"}
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
                    if row.get("targets_file") and self.registry.instances[
                        source
                    ].lifecycle in {"ACTIVE", "WAKING"}:
                        pc.atomic_json(
                            row["targets_file"], source_entries(source, self.targets)
                        )
                target["defer_channels_until_model_ready"] = False
                self.ops.event(
                    "CHANNEL_TOPOLOGY_PUBLISHED",
                    instance=name,
                    sources=[c["source"] for c in target["connector_sources"]],
                )
        for name, extra in getattr(self, "started", {}).items():
            # A woken TP1 source reconnects only after its engine is awake.
            if extra.get("wake") and not extra["channels_published"]:
                pc.atomic_json(
                    self.rows[name]["targets_file"],
                    source_entries(name, self.targets),
                )
                extra["channels_published"] = True
                self.ops.event(
                    "SOURCE_CHANNELS_PUBLISHED",
                    instance=name,
                    targets=sorted(self.targets),
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
        live = {"ACTIVE", "QUIESCING", "SLEEPING"}
        asleep = {
            n: e["target"]
            for n, e in getattr(self, "sleeping", {}).items()
            if e.get("target") is not None
        }
        value["migration_resources"] = dict(
            targets={
                n: t
                for n, t in {**asleep, **self.targets}.items()
                if self.layout.get(n, {}).get("lifecycle") in live
            },
            source_links={
                n: r["source_active"]
                for n, r in self.rows.items()
                if r.get("source_active")
                and self.layout.get(n, {}).get("lifecycle") in live
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
    # The underlying decided-sequence audit verifies ownership, controller ACKs
    # and safe ABORTED outcomes. Automatic runs may legitimately make no plan
    # or abandon one; explicit edge probes still require a completed edge.
    value["automatic_terminal_steps_valid"] = all(
        s["plan"]["state"] in {"COMPLETED", "ABORTED"}
        and s.get("layout") in PATH.values()
        for s in steps
    )
    value["mechanism_passed"] = value["passed"] and (
        value["all_steps_completed"]
        if require_conversion
        else value["automatic_terminal_steps_valid"]
    )
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
