# SPDX-License-Identifier: Apache-2.0
"""Serialized conversion plans, explicit or proposed by the decider."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

from tools.bridge_tp import pressure_conversion as pc


def prepare_plan(plan):
    """Bind startup senders to files so a later receiver may be replaced."""
    by_source = {c["source"]: c for c in plan["channel_sources"]}
    aliases = plan["instances"]["legacy_pool_aliases"]
    target_id = aliases["tp4"]
    folder = plan["run_dir"].parent
    extra_by_id = {e["instance_id"]: e for e in plan["extra_engines"]}
    files = {}
    for row in plan["instances"]["instances"]:
        if row["tp_degree"] != 1:
            continue
        source = row["instance_id"]
        c = by_source[source]
        channel = dict(
            base_port=c["base_port"],
            generation=c["generation"],
            target=target_id,
            target_tp=2,
        )
        path = str(folder / f"gpu_targets_{source}.json")
        row.update(
            migration_channel=channel,
            targets_file=path,
            predictor_events=(
                plan["source_env"]["BRIDGETP_PREDICTOR_LIVE_EVENTS"]
                if row.get("legacy_pool") == "tp1"
                else extra_by_id[source]["predictor_events"]
            ),
        )
        files[path] = [
            dict(
                target=target_id,
                host="127.0.0.1",
                tp=2,
                base_port=c["base_port"],
                generation=c["generation"],
            )
        ]
        if row.get("legacy_pool") == "tp1":
            plan["source_env"]["BRIDGETP_GPU_DIRECT_TARGETS_FILE"] = path
            plan["source_env"].pop("BRIDGETP_GPU_DIRECT_TARGETS", None)
        else:
            extra_by_id[source].update(targets_file=path)
    plan["sequence_target_files"] = files
    plan["sequence_context"] = dict(
        target_tp=2,
        target_host="127.0.0.1",
        folder=str(folder),
        target_slot=dict(endpoint=plan["config"]["target_url"]),
        source_env=plan["source_env"],
    )


class SequenceBackend(pc.ControllerConversionBackend):
    """Per-plan transient state over one run's registry and owned resources."""

    def __init__(self, *args, sequence_context, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = sequence_context
        self.next_service_port = self.base_port
        channels = list(pc.read_json(self.channel_plan) or [])
        self.next_channel_generation = (
            max((c["generation"] for c in channels), default=0) + 1
        )
        self.next_channel_port = (
            max((c["base_port"] for c in channels), default=30400) + 100
        )
        self.stager_started = any(
            r.bundle_id == "B" and r.tp_degree == 2 and r.lifecycle == "ACTIVE"
            for r in self.registry.instances.values()
        )
        self.current_plan_id = None
        self.ops = copy.copy(self.ops)
        event = self.ops.event
        self.ops.event = lambda kind, **values: event(
            kind, **{"plan_id": self.current_plan_id, **values}
        )

    def active(self):
        return {
            i: r for i, r in self.registry.instances.items() if r.lifecycle == "ACTIVE"
        }

    def begin_step(self, plan_id, step):
        active = self.active()
        modes = {r.bundle_id: r.tp_degree for r in active.values()}
        bundle, target = step["bundle"], int(step["target_mode"][2:])
        if set(modes) != {"A", "B"} or modes[bundle] == target:
            raise ValueError("sequence edge does not match the current active layout")
        after = dict(modes, **{bundle: target})
        if after == {"A": 2, "B": 1}:
            raise ValueError("L1B is outside this explicit sequence")
        self.current_plan_id = plan_id
        self.started, self.warm = {}, {}
        self.base_port = self.next_service_port
        self.channel_closing = self.joining = self.prestarted = None
        if hasattr(self, "_close_generations"):
            del self._close_generations
        self.gate_since = self.valve_since = self.refused_since = None
        self.planned = self.evacuation = self.last_shadow = None
        self.last_assist_reason = None
        self.assisted = []
        self.assist_step_start_seq = self.assist_seq
        self.assist_committed_baseline = sum(
            state in {"COMMITTED", "ACCOUNTED", "COMPLETED"}
            for state in (self.status().get("assist_sessions") or {}).values()
        )
        self.assist_at = self.shadow_at = -float("inf")
        self.valve_continued = self.reported_unacknowledged = False
        self.service_gate_passed = False
        self.layout_assist = None
        self.layout_tilt = []
        self._specs = None
        target_id = next(
            (i for i, r in active.items() if r.bundle_id == "B" and r.tp_degree == 2),
            None,
        )
        self.target_tp_ranks = 2 if target_id else 0
        self.join = None
        if bundle == "B" and target == 2:
            channels, files = {}, {}
            for source, r in sorted(active.items()):
                if r.bundle_id != "A" or r.tp_degree != 1:
                    continue
                channels[source] = self.channel(source)
                files[source] = self.rows[source]["targets_file"]
            self.join = dict(
                self.context,
                mode="MERGE_TARGET",
                bundle=bundle,
                channels=channels,
                target_files=files,
            )
        elif bundle == "A" and target == 1 and target_id:
            self.join = dict(
                self.context,
                mode="SPLIT_SOURCES",
                bundle=bundle,
                channels={},
                target_files={},
                target_id=target_id,
                channel_base_port=self.next_channel_port,
            )
        self.ops.event("STEP_BEGIN", plan_id=plan_id, before=modes, after=after)
        self.write_layout({}, assist=None, tilt=[])

    def channel(self, source):
        generation, port = self.next_channel_generation, self.next_channel_port
        self.next_channel_generation += 1
        self.next_channel_port += 10
        if port + 9 >= 65536:
            raise ValueError("sequence channel port budget exhausted")
        return dict(
            pool=self.rows[source].get("legacy_pool") or source,
            base_port=port,
            generation=generation,
        )

    def joined_sources(self):
        return [i for i in self.active() if "migration_channel" in self.rows[i]]

    def replacement_specs(self, plan):
        if self._specs is not None:
            return self._specs
        specs = super().replacement_specs(plan)
        self._specs = []
        for instance_id, generation, extra, joined in specs:
            if self.join_mode(plan) == "MERGE_TARGET":
                joined = dict(migration_target=True)
                extra["migration_target"] = True
                extra["inventory_path"] = str(
                    self.run_dir / f"inventory_{instance_id}.jsonl"
                )
            elif "migration_channel" in joined:
                # Channel generations do not reuse an old target's counters.
                c = self.channel_for_new(instance_id)
                joined["migration_channel"].update(c)
                extra["migration_channel"].update(c)
            self.next_service_port = max(self.next_service_port, extra["port"] + 1)
            self._specs.append((instance_id, generation, extra, joined))
        return self._specs

    def channel_for_new(self, instance_id):
        self.rows.setdefault(instance_id, {})
        return {k: v for k, v in self.channel(instance_id).items() if k != "pool"}

    def prestart(self, plan):
        if not self.prestart_enabled or self.prestarted is not None:
            return
        self.prestarted = {}
        for instance_id, _, extra, _ in self.replacement_specs(plan):
            gated = dict(
                extra,
                start_gate=str(
                    self.run_dir / f"start_gate_{plan['plan_id']}_{instance_id}"
                ),
            )
            self.prestarted[instance_id] = gated
            self.ops.start(gated)
        self.ops.event(
            "PRESTART", plan_id=plan["plan_id"], instances=sorted(self.prestarted)
        )

    def retire_channels(self, plan):
        # Closing a receiver closes every source edge; closing a source only
        # removes that source. Graph edges, not instance generations, select it.
        retiring = {r.instance_id for r in self.refs(plan)}
        receiver = any(
            self.registry.get(r).bundle_id == "B"
            and self.registry.get(r).tp_degree == 2
            for r in self.refs(plan)
        )
        if self.channel_closing is None:
            rows = pc.read_json(self.channel_plan) or []
            selected = [c for c in rows if receiver or c["source"] in retiring]
            self._close_generations = {c["generation"] for c in selected}
            for c in selected:
                path = self.rows[c["source"]].get("targets_file")
                if path:
                    pc.atomic_json(path, [])
            # The base close protocol uses this selection hook.
        return super().retire_channels(plan)

    def start_replacements(self, plan):
        super().start_replacements(plan)
        for instance_id, extra in self.started.items():
            if extra.get("targets_file"):
                self.rows[instance_id]["targets_file"] = extra["targets_file"]

    def step_acknowledged(self):
        status = self.status()
        if not self.fresh(status) or status.get("layout_revision", -1) < self.revision:
            return False
        for instance_id, r in self.registry.instances.items():
            if r.lifecycle not in {"ACTIVE", "RETIRED"}:
                return False
            row = status["instances"].get(instance_id)
            if row is None or row["lifecycle"] != r.lifecycle:
                return False
            if r.lifecycle == "ACTIVE" and row.get("inventory") is None:
                return False
        return True


def validate(request):
    """Validate the sequence without creating processes or mutating a registry."""
    allowed = {"sequence", "base_port", "prestart", "safety"}
    if not set(request) <= allowed or not {"sequence", "base_port"} <= set(request):
        raise ValueError("invalid conversion sequence fields")
    if (
        type(request["base_port"]) is not int
        or not 1024 <= request["base_port"] <= 64000
    ):
        raise ValueError("invalid sequence service port")
    if type(request.get("prestart", False)) is not bool:
        raise ValueError("invalid prestart flag")
    if not set(request.get("safety", {})) <= set(pc.SAFETY):
        raise ValueError("invalid sequence safety overrides")
    steps = request["sequence"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 32:
        raise ValueError("sequence requires 1..32 explicit steps")
    previous = -1
    for step in steps:
        required = {"bundle", "target_mode", "trigger_s", "deadline_s"}
        if not required <= set(step) <= required | {"method", "on_abort"}:
            raise ValueError("invalid sequence step fields")
        if step["bundle"] not in {"A", "B"} or step["target_mode"] not in {
            "TP1",
            "TP2",
        }:
            raise ValueError("sequence supports four-card A/B TP1/TP2 only")
        if any(
            type(step[k]) not in (int, float) or not math.isfinite(step[k])
            for k in ("trigger_s", "deadline_s")
        ):
            raise ValueError("sequence times must be finite numbers")
        if (
            step["trigger_s"] < 0
            or step["trigger_s"] < previous
            or step["deadline_s"] <= 0
        ):
            raise ValueError(
                "sequence triggers must be ordered; deadline is a duration"
            )
        previous = step["trigger_s"]
        if step.get("method", "DRAIN") not in {"DRAIN", "SELECTIVE_ROLL"}:
            raise ValueError("unsupported sequence method")
        if step.get("method") == "SELECTIVE_ROLL" and step["target_mode"] != "TP2":
            raise ValueError("only merges support drain assistance")
        if step.get("on_abort", "STOP") not in {"STOP", "CONTINUE"}:
            raise ValueError("invalid sequence abort policy")
    return request


class ConversionSequenceDriver(pc.ConversionDriver):
    """Keep the controller alive across explicitly scheduled, isolated plans.

    Safe ABORTED steps may continue only when explicitly requested. Destructive
    uncertainty always stops. Every plan has its own durable journal; no plan
    or migration receipt is deleted or reused.
    """

    def __init__(self, backend, request, journal_path, interval_s=0.2):
        validate(request)
        super().__init__(backend, request, journal_path, interval_s)
        self.sequence_request = request
        self.root = Path(journal_path).parent
        self.history = []
        self.sequence_state = "RUNNING"

    def state(self):
        if self.sequence_state != "RUNNING":
            return self.sequence_state
        return super().state()

    def save_sequence(self):
        pc.atomic_json(
            self.root / "conversion_sequence_result.json",
            dict(state=self.sequence_state, error=self.error, steps=self.history),
        )

    def wait_for_clock(self):
        """False when stopped first."""
        if not hasattr(self.backend, "status"):
            return True
        until = time.monotonic() + 900
        while True:
            status = self.backend.status()
            if (
                self.backend.fresh(status)
                and status.get("workload_origin_unix_s") is not None
            ):
                self.origin = time.monotonic() - (
                    time.time() - status["workload_origin_unix_s"]
                )
                return True
            if time.monotonic() >= until:
                raise TimeoutError("controller workload clock not published")
            if self.stop_event.wait(self.interval_s):
                return False

    def after_begin(self, step):
        """A step has begun; its plan does not exist yet."""

    def run_step(self, plan_id, step):
        """One plan to its end and acknowledgement; None when stopped first."""
        self.request = step
        self.backend.begin_step(plan_id, step)
        self.after_begin(step)
        self.executor = pc.conversion.ConversionExecutor(
            self.backend.registry,
            self.backend,
            self.root / "conversions" / plan_id / "plan.json",
            clock=time.time,
        )
        self.triggered_at = time.time()
        self.backend.ops.event("TRIGGER", plan_id=plan_id, request=step)
        self.executor.plan_create(
            plan_id,
            self.backend.registry.registry_revision,
            step["bundle"],
            step["target_mode"],
            step.get("method", "DRAIN"),
            "ACTIVE",
            time.time() + step["deadline_s"],
            "NO_PROFILE",
        )
        while not self.stop_event.is_set():
            self.end_expired_tilt()
            before = self.executor.plan["state"]
            after = self.executor.plan_execute(time.time())["state"]
            if after != before:
                self.backend.ops.event("PLAN_STATE", plan_id=plan_id, state=after)
            if after in self.TERMINAL:
                break
            self.stop_event.wait(self.interval_s)
        if self.stop_event.is_set():
            return None
        self.history.append(
            dict(
                plan_id=plan_id,
                request=step,
                triggered_unix_s=self.triggered_at,
                completed_unix_s=time.time(),
                plan=self.executor.plan_status(),
                layout_revision=self.backend.revision,
                controller_acknowledged=False,
                layout=(
                    {r.bundle_id: r.tp_degree for r in self.backend.active().values()}
                    if hasattr(self.backend, "active")
                    else None
                ),
                active_instances=(
                    {
                        i: dict(
                            tp_degree=r.tp_degree,
                            generation=r.ref.instance_generation,
                            bundle=r.bundle_id,
                            endpoint=r.endpoint,
                        )
                        for i, r in self.backend.active().items()
                    }
                    if hasattr(self.backend, "active")
                    else None
                ),
            )
        )
        self.save_sequence()
        if after != "RECOVERY_REQUIRED":
            until = time.monotonic() + 30
            while not self.backend.step_acknowledged():
                if time.monotonic() >= until:
                    raise TimeoutError("controller did not acknowledge conversion")
                if self.stop_event.wait(self.interval_s):
                    return None
            self.history[-1].update(
                controller_acknowledged=True, completed_unix_s=time.time()
            )
            self.save_sequence()
        return after

    def _loop(self):
        try:
            if not self.wait_for_clock():
                return
            for number, step in enumerate(self.sequence_request["sequence"], 1):
                while time.monotonic() - self.origin < step["trigger_s"]:
                    if self.stop_event.wait(self.interval_s):
                        return
                if self.stop_event.is_set():
                    return
                plan_id = f"sequence-{number:04d}-{step['bundle']}"
                after = self.run_step(plan_id, step)
                if after is None:
                    return
                if after == "RECOVERY_REQUIRED" or (
                    after == "ABORTED" and step.get("on_abort", "STOP") != "CONTINUE"
                ):
                    self.sequence_state = after
                    return
            self.sequence_state = "COMPLETED"
        except Exception as error:
            self.error = repr(error)
            self.sequence_state = "FAILED"
            self.backend.ops.event("DRIVER_ERROR", error=self.error)
        finally:
            if self.sequence_state == "RUNNING":
                self.sequence_state = "INTERRUPTED"
            self.save_sequence()
            self.backend.write_layout({}, pending=False)

    def stop(self, timeout_s=30):
        result = super().stop(timeout_s)
        return dict(result, steps=self.history, sequence=True)


class DecidedSequenceDriver(ConversionSequenceDriver):
    """Serial plans proposed by the decider while the workload runs.

    A safely ABORTED plan is not final: the decider may propose again after its
    cooldown. RECOVERY_REQUIRED or an error stops all conversion. Between plans
    the controller is free to finish; the plan itself holds it.
    """

    prepare_prestart = False  # A plan pre-starts its own replacements.

    def __init__(self, backend, request, journal_path, interval_s=0.2):
        pc.ConversionDriver.__init__(self, backend, request, journal_path, interval_s)
        self.sequence_request = request
        self.root = Path(journal_path).parent
        self.history = []
        self.sequence_state = "RUNNING"
        self.number = 0
        self.tilt_before_step = []

    def _loop(self):
        try:
            if not self.wait_for_clock():
                self.sequence_state = "COMPLETED"
                return
            while (
                not self.stop_event.is_set() and self.sequence_state == "RUNNING"
            ):
                self.decide()
                self.stop_event.wait(self.interval_s)
            if self.sequence_state == "RUNNING":
                self.sequence_state = "COMPLETED"
        except Exception as error:
            self.error = repr(error)
            self.sequence_state = "FAILED"
            self.backend.ops.event("DRIVER_ERROR", error=self.error)
        finally:
            if self.sequence_state == "RUNNING":
                self.sequence_state = "INTERRUPTED"
            self.save_sequence()
            self.backend.write_layout({}, pending=False)

    def after_begin(self, step):
        # Beginning clears the tilt, but routing keeps it until the bundle closes;
        # the start gate's service check ran before the load the tilt added.
        tilt = self.tilt_before_step
        if tilt:
            self.backend.write_layout({}, tilt=tilt)
            self.backend.service_gate_passed = True
            self.tilt_since = time.monotonic()

    def start_proposal(self, proposal):
        limit = self.sequence_request.get("max_conversions", 4)
        if self.number >= limit:
            self.backend.ops.event("DECIDER_LIMIT", limit=limit, **proposal)
            return
        self.number += 1
        bundle, mode = proposal["bundle"], proposal["target_mode"]
        step = dict(
            bundle=bundle,
            target_mode=mode,
            deadline_s=self.sequence_request["deadline_s"],
            # Only a merge towards L2 has a TP-k to move requests to.
            method="SELECTIVE_ROLL"
            if self.sequence_request.get("assist") and (bundle, mode) == ("A", "TP2")
            else "DRAIN",
            held_s=proposal["held_s"],
        )
        self.tilt_before_step = list(self.backend.layout_tilt)
        after = self.run_step(f"decided-{self.number:04d}-{bundle}", step)
        self.decider.converted(time.time())
        self.prepared_key = None
        if after is None:
            return
        self.backend.write_layout({}, pending=False)
        if after == "RECOVERY_REQUIRED":
            self.sequence_state = after


def assess(scene, expected_requests=None, decided=False):
    """Separate cycle correctness, migration coverage and planned-time latency.

    ``decided``: the decider chose the steps, so their number and layouts are
    results, not expectations, and a safely aborted step is allowed.
    """
    from tools.bridge_tp.qwen3_four_mode import distribution

    scene = Path(scene)
    result = pc.read_json(scene / "conversion_result.json")
    run = scene / "controller"
    events = [
        json.loads(v)
        for v in (scene / "conversion_events.jsonl").read_text().splitlines()
    ]
    audit = [
        json.loads(v) for v in (run / "pressure_audit.jsonl").read_text().splitlines()
    ]
    rows = pc.read_json(run / "pressure_request_results.json")
    proof = pc.read_json(scene / "closed_loop_acceptance.json")
    admission_closed, late_dispatches = {}, []
    # The audit's closing evidence_footer carries no timestamp.
    audit = [row for row in audit if "unix_s" in row]
    for event in sorted(audit, key=lambda row: row["unix_s"]):
        if event["kind"] == "pressure_instance_lifecycle":
            admission_closed[event["instance_id"]] = (
                event["unix_s"] if event["to"] != "ACTIVE" else None
            )
        elif event["kind"] == "pressure_dispatch":
            closed_at = admission_closed.get(event["pool"])
            if closed_at is not None and event["dispatch_entered_unix_s"] > closed_at:
                late_dispatches.append(event["job_id"])
    steps = result.get("steps", [])
    expected = [{"A": 2, "B": 2}, {"A": 1, "B": 2}, {"A": 1, "B": 1}, {"A": 1, "B": 2}]
    phases = []
    applied = {
        v["revision"]: v["unix_s"]
        for v in audit
        if v["kind"] == "pressure_layout_applied"
    }
    for step in steps:
        group = [v for v in events if v.get("plan_id") == step["plan_id"]]

        def at(kind, group=group):
            return next((v["unix_s"] for v in group if v["kind"] == kind), None)

        closed = next(
            (
                v["unix_s"]
                for v in group
                if v["kind"] == "LAYOUT_CONTROL"
                and "QUIESCING" in v.get("changes", {}).values()
            ),
            None,
        )
        fenced, released, published = (
            at("FENCED"),
            at("RESOURCES_RELEASED"),
            at("ROUTE_PUBLISHED"),
        )
        ordered = all(
            v is not None for v in (closed, fenced, released, published)
        ) and (closed <= fenced <= released <= published <= step["completed_unix_s"])
        phases.append(
            dict(
                plan_id=step["plan_id"],
                state=step["plan"]["state"],
                layout=step["layout"],
                evidence_ordered=ordered,
                controller_applied=step["layout_revision"] in applied,
                drain_s=fenced - closed if ordered else None,
                rebuild_s=published - released if ordered else None,
                admission_closed_s=published - closed if ordered else None,
                trigger_to_route_s=published - step["triggered_unix_s"]
                if ordered
                else None,
            )
        )

    def latency(selected):
        ttft, e2e, dispatch, gaps = [], [], [], []
        for row in selected:
            e = row["response"]["emitted"]
            planned = row["dispatch_timing"]["planned_unix_s"]
            ttft.append(e[0]["unix_s"] - planned)
            e2e.append(e[-1]["unix_s"] - planned)
            dispatch.append(
                row["dispatch_timing"]["http_thread_started_unix_s"] - planned
            )
            gaps.extend(b["unix_s"] - a["unix_s"] for a, b in zip(e, e[1:]))
        return dict(
            requests=len(selected),
            ttft_s=distribution(ttft),
            e2e_s=distribution(e2e),
            dispatch_delay_s=distribution(dispatch),
            maximum_gap_s=max(gaps, default=None),
            gaps_over_10s=sum(g > 10 for g in gaps),
        )

    starts = [v for v in audit if v["kind"] == "pressure_actual_start"]
    committed = {
        v["migration_id"]
        for v in (proof or {}).get("sessions", [])
        if v["state"] == "COMMITTED" and v.get("source_release")
    }
    windows = []
    for number in (1, 3):
        if number >= len(steps):
            continue
        left = steps[number]["completed_unix_s"]
        right = (
            steps[number + 1]["triggered_unix_s"]
            if number + 1 < len(steps)
            else float("inf")
        )
        windows.append(
            dict(
                after_plan=steps[number]["plan_id"],
                committed_migrations=sum(
                    left <= v["unix_s"] < right
                    and v["migration_id"] in committed
                    and v.get("target_epoch")
                    in (steps[number].get("active_instances") or {})
                    for v in starts
                ),
            )
        )
    complete = (
        result.get("state") == "COMPLETED"
        and result.get("error") is None
        and (decided or (len(steps) == 4 and [v["layout"] for v in steps] == expected))
        and all(
            v["state"] == "ABORTED"
            if decided and v["state"] != "COMPLETED"
            else v["evidence_ordered"] and v["controller_applied"]
            for v in phases
        )
        and all(v.get("controller_acknowledged") for v in steps)
        and not late_dispatches
        and len({r["job_id"] for r in rows}) == len(rows)
        and (expected_requests is None or len(rows) == expected_requests)
        and all(
            r["source_done"]
            and r["source_error"] is None
            and len(r["response"]["emitted"]) == r["request"]["max_tokens"]
            for r in rows
        )
    )
    coverage = decided or (
        len(windows) == 2 and all(v["committed_migrations"] > 0 for v in windows)
    )
    for phase, step in zip(phases, steps):
        phase["planned_latency"] = latency(
            [
                r
                for r in rows
                if step["triggered_unix_s"]
                <= r["dispatch_timing"]["planned_unix_s"]
                <= step["completed_unix_s"]
            ]
        )
    return dict(
        passed=complete and coverage,
        conversion_correct=complete,
        migration_coverage=coverage,
        late_dispatches=late_dispatches,
        migration_windows=windows,
        phases=phases,
        planned_latency=latency(rows),
        note="a mechanism check, not automatic-policy performance evidence; "
        "inspect host_monitor controller CPU before interpreting latencies; "
        "commit-filled token timestamps make handoff incomparable with older runs",
    )


def finalize(plan, common, *, timeout_s=30):
    """Finalize the current engines, rather than the retired startup pair."""
    from tools.bridge_tp.run_atomic_takeover import _post_json
    from tools.bridge_tp.verify_performance_trace import verify_pool

    directory = plan["run_dir"]
    status = pc.read_json(directory / "layout_status.json")["instances"]
    control = pc.read_json(directory / "layout_control.json")
    active = [r for r in control["instances"] if r["lifecycle"] == "ACTIVE"]
    if not active or any(
        status.get(r["instance_id"], {}).get("lifecycle") != "ACTIVE" for r in active
    ):
        raise ValueError("sequence finalization lacks acknowledged active engines")
    deadline = time.monotonic() + timeout_s
    while True:
        inventories = {
            r["instance_id"]: pc.load("native_journal").latest_inventory(
                Path(r["inventory_path"])
            )
            for r in active
        }
        if all(
            value
            and value["sampled_at"] <= time.time()
            and not value["requests"]
            and not value["running"]
            and not value["waiting"]
            for value in inventories.values()
        ):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("current sequence engines did not settle empty")
        time.sleep(0.2)
    common.write_json(directory.parent / "native_empty_receipt.json", inventories)
    for row in active:
        side = "source" if row["tp_degree"] == 1 else "target"
        environment = plan[side + "_env"]
        receipt = _post_json(
            row["endpoint"],
            "/bridge_tp/v1/performance_trace/finalize",
            {"token": environment["BRIDGETP_PERFORMANCE_FINALIZE_TOKEN"]},
            60,
        )
        name = row["instance_id"]
        common.write_json(directory.parent / f"{name}_trace_finalization.json", receipt)
        if receipt.get("status") != "FINALIZED":
            raise RuntimeError("current engine trace did not finalize: " + name)
        trace = Path(environment["BRIDGETP_PERFORMANCE_TRACE_DIR"]).parent / name
        common.write_json(
            directory.parent / f"{name}_trace_acceptance.json",
            verify_pool(trace, row["tp_degree"]),
        )
