# SPDX-License-Identifier: Apache-2.0
"""S3b-3: explicit TP bundle conversion while the fast controller serves.

``ConversionExecutor`` owns the step order and its durable journal. This backend
binds those steps to the live pilot:

- routing: ``layout_control.json`` (QUIESCING, RETIRED, new ACTIVE instances),
  read by the controller every tick (S3b-1);
- drain evidence: the controller's ``layout_status.json`` (open admissions,
  relayed streams, migration sessions) plus a fresh native scrape;
- channels: the target connector's channel plan file; a retired TP1 source's
  channel is closed on every target rank before its engine stops;
- engines: injected process operations owned by the scene runner.

New instances join routing-only: after a conversion the migration pair is
the original one if both sides survive, otherwise migration stays disabled.
With ``join`` (S5b, a single-class start) the new instances join migration
instead: the conversion closes the channels of what it retires (as above)
and opens every TP1-to-TP-k channel of the new layout before publishing it.
The conversion is triggered explicitly; nothing here selects a layout.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from tools.bridge_tp.pressure_cpu import load

pool = load("instance_pool")
conversion = load("reconfiguration_executor")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    for attempt in range(50):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:  # Windows: a reader holds the file right now
            if attempt == 49:
                raise
            time.sleep(0.01)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


PLAN_RECEIPT = re.compile(r"tp_rank_\d+\.json")
# S5b channel readiness: both ends of every channel the conversion opened.
CHANNEL_READY_S = 180.0
TEARDOWN_RECEIPT = re.compile(r"(tp_rank_\d+)-retired-(\d+)\.json")


def channel_receipts(root, pattern=PLAN_RECEIPT):
    """Target-rank channel plan receipts below ``root`` (no symlinks)."""
    rows = []
    for directory, _, names in os.walk(root):
        if Path(directory).name != "gpu_channel_plan_receipts":
            continue
        for name in sorted(names):
            if pattern.fullmatch(name):
                value = read_json(Path(directory) / name)
                if value is not None:
                    rows.append((name, value))
    return rows


def preconnect_receipts(root):
    """{file name: receipt} of channel warmups below ``root`` (both ends)."""
    rows = {}
    for directory, _, names in os.walk(root):
        if Path(directory).name != "gpu_channel_preconnect_receipts":
            continue
        for name in names:
            if name.endswith(".json"):
                value = read_json(Path(directory) / name)
                if value is not None:
                    rows[name] = value
    return rows


def teardown_receipts(root):
    """{(rank, generation): status} of retired channels' background teardown."""
    return {
        (match[1], int(match[2])): value.get("status")
        for name, value in channel_receipts(root, TEARDOWN_RECEIPT)
        for match in [TEARDOWN_RECEIPT.fullmatch(name)]
    }


# S5a safety, two tiers (first values; tuned from GPU runs, every decision is
# logged). Hard safety is never relaxed: KV not driven into the guard (the
# policy's capacity ratio, 1.0 = free down to guard + projected growth), no
# request lost. Service quality (zone SPARE) gates the start comfortably, is
# waived after ``relax_after_s``, and always gates assist migrations.
SAFETY = dict(
    relax_after_s=180.0,
    gate_capacity_max=0.8,
    valve_capacity=1.0,
    valve_sustain_s=3.0,
    # Valve trips but the conversion is nearly done: finish instead of
    # cancelling (user 11-01); the estimate is the SHADOW planner's
    # conservative drain-only READY (remaining drain + later stages).
    continue_if_remaining_s=60.0,
    assist_factor=3.0,
    # All-or-nothing evacuation (user 11-01): a partial one left the drain
    # as long and still cost the target. S4/S5a: received migrations grow
    # the TP2 by ~1.2 GB once, not per migration, so the cap is the size of
    # a TP1 pair's long requests seen so far, not a memory budget.
    assist_max_committed=4,
    assist_max_attempts=8,
    evacuation_patience_s=30.0,
    decision_interval_s=1.0,
)


class ControllerConversionBackend:
    """Backend of one explicit conversion inside a pilot scene.

    ``ops`` (scene runner) provides: ``stop(ids)``, ``alive(id)``,
    ``gpus_released(gpu_ids)``, ``scrape(endpoint, native)`` (fresh metrics
    dict or None), ``start(extra)`` (spawn, non-blocking) and ``warm(extra)``
    (blocking health wait + warmup), ``event(kind, **values)``.
    """

    def __init__(
        self,
        registry,
        manifest,
        run_dir,
        ops,
        *,
        base_port,
        context_limits,
        channel_plan=None,
        target_tp_ranks=0,
        clock=time.time,
        shadow_planner=None,
        safety=None,
        speed_margin=0.8,
        join=None,
        prestart=False,
    ):
        self.registry, self.ops, self.clock = registry, ops, clock
        self.run_dir = Path(run_dir)
        self.base_port, self.context_limits = base_port, context_limits
        self.channel_plan, self.target_tp_ranks = channel_plan, target_tp_ranks
        self.rows = {row["instance_id"]: dict(row) for row in manifest["instances"]}
        self.revision = 0
        self.layout = {}
        self.started = {}
        self.warm = {}
        self.channel_closing = None
        self.layout_assist = None
        self.layout_tilt = []
        self.safety = dict(SAFETY, **(safety or {}))
        self.speed_margin = speed_margin
        self.gate_since = None
        self.valve_since = None
        self.assist_at = -float("inf")
        self.assist_seq = 0
        self.assist_step_start_seq = 0
        self.assisted = []  # request ids whose assist START was published
        self.assist_committed_baseline = 0
        self.last_assist_reason = None
        self.planned = None  # evacuation proposed, waiting for the controller
        self.evacuation = None  # accepted plan; [] once it gave up
        self.refused_since = None
        # S4: read-only planner consulted while draining; never acts.
        self.shadow_planner, self.shadow_at = shadow_planner, -float("inf")
        self.last_shadow = None
        self.valve_continued = False
        self.reported_unacknowledged = False
        # S5b: launcher join plan (mode, bundle, channels, target slot...).
        self.join = join
        self.joining = None  # channels this conversion opens, once written
        self.stager_started = False
        # Pre-start the replacements behind a start gate while draining.
        self.prestart_enabled = prestart
        self.prestarted = None
        self.service_gate_passed = False
        self.lock = threading.Lock()
        # Engines are serving when the controller starts: the manifest's
        # PREPARING records become ACTIVE before the first plan.
        for row in registry.instances.values():
            if row.lifecycle == "PREPARING":
                registry.transition(row.ref, "ACTIVE")

    # ------------------------------------------------------------- helpers
    def refs(self, plan):
        return [pool.InstanceRef(**r) for r in plan["refs"]]

    def write_layout(self, changes, *, pending=True, assist=None, tilt=None):
        """Publish lifecycle changes; rows accumulate across revisions.

        ``conversion_pending`` keeps the controller serving after its workload
        is done, until the driver publishes the plan's terminal state.
        """
        for instance_id, lifecycle in changes.items():
            row = self.rows[instance_id]
            self.layout[instance_id] = dict(
                instance_id=instance_id,
                tp_degree=row["tp_degree"],
                context_limit=row["context_limit"],
                endpoint=row["endpoint"],
                inventory_path=row.get("inventory_path", ""),
                lifecycle=lifecycle,
                # S5b: how a new instance joins migration.
                **{
                    k: row[k]
                    for k in (
                        "legacy_pool",
                        "migration_channel",
                        "predictor_events",
                        "migration_target",
                    )
                    if k in row
                },
            )
        self.revision += 1
        if assist is not None:
            self.layout_assist = assist
        if tilt is not None:
            self.layout_tilt = sorted(tilt)
        atomic_json(
            self.run_dir / "layout_control.json",
            dict(
                revision=self.revision,
                instances=list(self.layout.values()),
                conversion_pending=pending,
                assist=assist or self.layout_assist,
                # Decider pre-tilt: instances that get no new long request.
                tilt=self.layout_tilt,
            ),
        )
        self.ops.event("LAYOUT_CONTROL", revision=self.revision, changes=changes)
        return self.revision

    def status(self):
        return read_json(self.run_dir / "layout_status.json")

    def controller_states(self, ids):
        value = self.status()
        if value is None:
            return None
        return {i: value["instances"].get(i) for i in ids}

    # ------------------------------------------------------ executor steps
    def decision(self, kind, **values):
        row = dict(kind=kind, unix_s=self.clock(), **values)
        with (self.run_dir / "conversion_decisions.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(row) + "\n")
        return row

    def retained(self, plan, status):
        refs = {r.instance_id for r in self.refs(plan)}
        return {
            i: v
            for i, v in status["instances"].items()
            if i not in refs and v["lifecycle"] == "ACTIVE"
        }

    def fresh(self, status):
        return status is not None and 0 <= self.clock() - status["unix_s"] <= 2

    @staticmethod
    def hard_ok(value, limit):
        pressure = value.get("pressure") or {}
        capacity = pressure.get("capacity")
        return capacity is not None and capacity < limit

    def gate(self, plan):
        """Start gate: the retained instances must carry the load meanwhile."""
        now = self.clock()
        if self.gate_since is None:
            self.gate_since = now
        status = self.status()
        waited = now - self.gate_since
        if not self.fresh(status):
            self.decision("GATE", open=False, waited_s=waited, reason="status stale")
            return False
        retained = self.retained(plan, status)
        if not retained:
            reason = "no retained instance"
        elif not all(
            self.hard_ok(v, self.safety["gate_capacity_max"]) for v in retained.values()
        ):
            reason = "retained KV headroom (hard)"
        elif waited >= self.safety["relax_after_s"] or self.service_gate_passed:
            # Passed: the decider checked the others were quiet before the
            # tilt it added to them; only hard safety is left to check.
            reason = None
        elif not all(
            (v.get("pressure") or {}).get("zone") == "SPARE" for v in retained.values()
        ):
            reason = "retained not SPARE (waiting for a quieter time)"
        else:
            reason = None
        opened = reason is None
        self.decision(
            "GATE",
            open=opened,
            waited_s=waited,
            relaxed=opened and waited >= self.safety["relax_after_s"],
            reason=reason,
            retained={
                i: (v.get("pressure") or {}).get("zone") for i, v in retained.items()
            },
        )
        if opened:
            self.ops.event(
                "GATE_OPEN",
                plan_id=plan["plan_id"],
                waited_s=round(waited, 3),
                relaxed=waited >= self.safety["relax_after_s"],
            )
        return opened

    def precheck(self, plan):
        if not all(
            self.registry.get(ref).lifecycle == "ACTIVE" for ref in self.refs(plan)
        ):
            return False
        if self.channel_plan is not None:
            # Every target rank must follow the plan file before anything is
            # quiesced; otherwise a retired source's channel could never close
            # (S3b-3 GPU, 10-31: the connector had rejected the file).
            planned = {r["generation"] for r in read_json(self.channel_plan) or []}
            acknowledged = {
                name
                for name, receipt in channel_receipts(self.run_dir)
                if "planned" in receipt and planned <= set(receipt["planned"])
            }
            if len(acknowledged) < self.target_tp_ranks:
                if not self.reported_unacknowledged:
                    self.reported_unacknowledged = True
                    self.ops.event(
                        "CHANNEL_PLAN_NOT_ACKNOWLEDGED",
                        plan_id=plan["plan_id"],
                        ranks=sorted(acknowledged),
                    )
                return False
        return self.gate(plan)

    def quiesce(self, plan):
        # Routing closes for the bundle: the tilt has nothing left to do.
        self.write_layout(
            {r.instance_id: "QUIESCING" for r in self.refs(plan)}, tilt=[]
        )
        self.prestart(plan)

    def remaining_s(self):
        """Conservative time left to READY from the latest planner snapshot.

        None (abort as before) unless a fresh, supported drain-only estimate
        exists: stale, failed or UNKNOWN planner output never keeps the
        retained side under pressure.
        """
        record = self.last_shadow
        if record is None or "error" in record:
            return None
        if not 0 <= self.clock() - record["evaluated_unix_s"] <= 3:
            return None
        drain = record["drain_only"]
        if drain["support_status"] != "SUPPORTED":
            return None
        return drain["estimated_ready_s"]

    def kv_emergency(self, plan):
        """Safety valve: retained KV about to hit its guard, sustained.

        Only hard safety aborts (before any engine stops): the plan is then
        cancelled and routing restored. Service-quality drops do not abort,
        or a conversion under load would cancel and retry forever. When the
        conservative remaining time is short the conversion finishes instead:
        cancelling then would throw away the drain for a few seconds' relief.
        """
        status = self.status()
        if not self.fresh(status):
            return False
        now = self.clock()
        unsafe = {
            i: (v.get("pressure") or {}).get("capacity")
            for i, v in self.retained(plan, status).items()
            if not self.hard_ok(v, self.safety["valve_capacity"])
            and (v.get("pressure") or {}).get("capacity") is not None
        }
        if not unsafe:
            self.valve_since = None
            return False
        if self.valve_since is None:
            self.valve_since = now
            self.decision("VALVE_WARNING", instances=unsafe)
        if now - self.valve_since < self.safety["valve_sustain_s"]:
            return False
        remaining = self.remaining_s()
        if (
            remaining is not None
            and remaining <= self.safety["continue_if_remaining_s"]
        ):
            if not self.valve_continued:
                self.valve_continued = True
                self.decision("VALVE_CONTINUE", instances=unsafe, remaining_s=remaining)
                self.ops.event(
                    "SAFETY_CONTINUE",
                    plan_id=plan["plan_id"],
                    remaining_s=round(remaining, 3),
                )
            return False
        self.decision("VALVE_ABORT", instances=unsafe, remaining_s=remaining)
        self.ops.event("SAFETY_ABORT", plan_id=plan["plan_id"], instances=unsafe)
        return True

    def assisted_step(self, plan, now):
        """S5a: evacuate every long request of the bundle, or none of them.

        A request is long when its conservative remaining time exceeds
        ``assist_factor`` x the measured START-to-release budget; shorter ones
        finish on their own. The drain ends with the last request, so moving
        only some long requests saves nothing and still loads the target
        (S5a GPU 11-01: 3 of 4 moved, drain 181s vs 184s). The evacuation
        starts only if all long requests are startable, fit the cap and the
        target is SPARE, and the controller accepts the whole set (KV for
        every final length, service budget for every added decode). It then
        moves them one at a time, longest first; each START is re-checked by
        the controller, and a refused one is retried until refusals have
        lasted ``evacuation_patience_s``.
        """
        if self.clock() - self.assist_at < self.safety["decision_interval_s"]:
            return
        self.assist_at = self.clock()
        status = self.status()
        reason, choice = self.assist_choice(plan, status)
        if reason != self.last_assist_reason or choice is not None:
            self.decision("ASSIST", choice=choice, reason=reason)
            self.last_assist_reason = reason
        if choice is None:
            return
        self.assist_seq += 1
        assist = dict(
            seq=self.assist_seq,
            request_id=choice["request_id"],
            source=choice["instance"],
        )
        if "evacuate" in choice:
            assist["evacuate"] = choice["evacuate"]
            self.planned = dict(seq=self.assist_seq, requests=choice["evacuate"])
        self.write_layout({}, assist=assist)
        self.ops.event("ASSIST_REQUESTED", seq=self.assist_seq, **choice)

    def long_requests(self, plan, status):
        """(long, blocked): rows worth moving, and why not all can be."""
        budget = status.get("release_budget_s")
        threshold = self.safety["assist_factor"] * budget
        long, blocked = [], []
        for instance in [r.instance_id for r in self.refs(plan)]:
            for row in status["instances"][instance]["requests"]:
                speed = row.get("tokens_s")
                remaining = max(u for u, _ in row["remaining_bins"])
                if not speed:
                    # Unknown speed: its time is unknown, not short.
                    blocked.append(row["request_id"])
                    continue
                natural = remaining / (speed * self.speed_margin)
                if natural <= threshold:
                    continue
                if not row.get("action_available"):
                    blocked.append(row["request_id"])
                    continue
                long.append(
                    dict(
                        request_id=row["request_id"],
                        instance=instance,
                        natural_s=round(natural, 3),
                        transfer_s=budget,
                    )
                )
        long.sort(key=lambda r: (-r["natural_s"], r["request_id"]))
        return long, blocked

    def assist_choice(self, plan, status):
        if not self.fresh(status):
            return "status stale", None
        result = status.get("assist_result")
        if self.assist_seq > self.assist_step_start_seq and (
            result is None or result["seq"] != self.assist_seq
        ):
            return "previous assist not answered yet", None
        if (
            self.assist_seq > self.assist_step_start_seq
            and result is not None
            and result["seq"] == self.assist_seq
        ):
            started = result["state"] == "STARTED"
            if started and result["request_id"] not in self.assisted:
                self.assisted.append(result["request_id"])
            planned = self.planned
            if planned is not None and planned["seq"] == result["seq"]:
                # The controller accepted (or refused) the whole evacuation.
                self.evacuation = planned["requests"] if started else None
                self.planned = None
            elif self.evacuation:
                # A refused step is "not now" (S5a GPU 11-01: one 512-token
                # prompt queued for 0.2s ended an evacuation 3/4 done); it is
                # retried under the same checks and ends only if refusals
                # persist for evacuation_patience_s.
                if started:
                    self.refused_since = None
                elif self.refused_since is None:
                    self.refused_since = self.clock()
                elif (
                    self.clock() - self.refused_since
                    > self.safety["evacuation_patience_s"]
                ):
                    self.evacuation = []
        target = status.get("migration_target")
        instances = status["instances"]
        if target is None or instances.get(target, {}).get("lifecycle") != "ACTIVE":
            return "no active migration target", None
        refs = [r.instance_id for r in self.refs(plan)]
        if any(instances[i]["sessions"] for i in refs) or instances[target]["sessions"]:
            return "a migration session is active", None
        states = list((status.get("assist_sessions") or {}).values())
        committed = max(
            0,
            sum(s in {"COMMITTED", "ACCOUNTED", "COMPLETED"} for s in states)
            - self.assist_committed_baseline,
        )
        if committed >= self.safety["assist_max_committed"]:
            return "committed assists at cap for this conversion", None
        if len(self.assisted) >= self.safety["assist_max_attempts"]:
            return "assist attempts at cap for this conversion", None
        if status.get("release_budget_s") is None:
            return "no measured transfer budget", None
        long, blocked = self.long_requests(plan, status)
        evacuation = self.evacuation
        if evacuation is not None:
            if evacuation == []:
                return "evacuation ended: steps refused for too long", None
            remaining = [r for r in long if r["request_id"] in evacuation]
            if not remaining:
                return "evacuation complete", None
            return None, remaining[0]
        if not long:
            return "no request worth migrating (natural <= factor x transfer)", None
        if blocked:
            return "a long request is not startable yet (all or none)", None
        room = self.safety["assist_max_committed"] - committed
        if len(long) > room:
            return f"{len(long)} long requests exceed the cap {room}: drain", None
        pressure = instances[target].get("pressure") or {}
        if pressure.get("zone") != "SPARE":
            return f"target not SPARE ({pressure.get('zone')})", None
        return None, dict(long[0], evacuate=[r["request_id"] for r in long])

    def shadow(self, plan, status):
        now = self.clock()
        if self.shadow_planner is None or now - self.shadow_at < 1:
            return
        self.shadow_at = now
        try:
            record = self.shadow_planner(plan, status, now)
        except Exception as error:  # a planner fault must never stop the drain
            record = dict(evaluated_unix_s=now, error=repr(error))
        self.last_shadow = record
        with (self.run_dir / "conversion_planner_shadow.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(record) + "\n")

    def drain_evidence(self, plan):
        value = self.status()
        if value is None:
            return []
        self.shadow(plan, value)
        evidence = []
        for ref in self.refs(plan):
            state = value["instances"].get(ref.instance_id)
            row = self.rows[ref.instance_id]
            native = (
                read_latest(Path(row["inventory_path"]))
                if row.get("inventory_path")
                else None
            )
            if state is None or native is None:
                return []
            observed = self.ops.scrape(row["endpoint"], native) or {}
            stamp = min(value["unix_s"], native["sampled_at"])
            idle = (
                not native["requests"]
                and native["running"] == native["waiting"] == 0
                and observed.get("num_running") == observed.get("num_waiting") == 0
                and observed.get("free_kv_blocks") == native["free_blocks"]
            )
            if idle:
                # Native inventory is event driven; revalidate an idle engine by
                # a fresh scrape instead of rewriting the native sample's age.
                stamp = min(value["unix_s"], observed["sampled_unix_s"])
            # The controller must have applied QUIESCING (routing closed).
            closed = state["lifecycle"] == "QUIESCING"
            evidence.append(
                conversion.DrainEvidence(
                    ref.instance_id,
                    ref.instance_generation,
                    stamp,
                    max(native["running"], observed.get("num_running", 1))
                    if closed
                    else None,
                    max(native["waiting"], observed.get("num_waiting", 1)),
                    max(state["open_admissions"], len(native["requests"])),
                    state["streams_pending"],
                    state["sessions"],
                    state["sessions"],
                    0,
                    state["open_admissions"],
                    0,
                    state["streams_pending"],
                    state["streams_pending"],
                    True,
                )
            )
        with (self.run_dir / "conversion_drain_observations.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    dict(
                        collected_unix_s=self.clock(), rows=[vars(r) for r in evidence]
                    )
                )
                + "\n"
            )
        return evidence

    def fence(self, plan):
        states = self.controller_states([r.instance_id for r in self.refs(plan)])
        if states is None or any(
            s is None or s["open_admissions"] or s["streams_pending"] or s["sessions"]
            for s in states.values()
        ):
            return False
        self.ops.event("FENCED", plan_id=plan["plan_id"])
        return True

    def retire_channels(self, plan):
        """Retire in the controller, then close their channels on the target."""
        ids = [r.instance_id for r in self.refs(plan)]
        if any(self.layout.get(i, {}).get("lifecycle") != "RETIRED" for i in ids):
            self.write_layout(dict.fromkeys(ids, "RETIRED"))
        states = self.controller_states(ids)
        if states is None or any(
            s is None or s["lifecycle"] != "RETIRED" for s in states.values()
        ):
            return False
        if self.channel_plan is None:
            self.ops.event("CHANNEL_RETIRE", plan_id=plan["plan_id"], closed=[])
            return True
        if self.channel_closing is None:
            rows = read_json(self.channel_plan) or []
            generations = {r.instance_generation for r in self.refs(plan)}
            selected = getattr(self, "_close_generations", generations)
            closing = sorted(
                r["generation"] for r in rows if r["generation"] in selected
            )
            written_at = self.clock()
            if closing:
                atomic_json(
                    self.channel_plan,
                    [r for r in rows if r["generation"] not in selected],
                )
            self.channel_closing = dict(
                generations=closing, written_at=written_at, closed={}
            )
            self.ops.event(
                "CHANNEL_PLAN_WRITTEN", plan_id=plan["plan_id"], closing=closing
            )
        closing = self.channel_closing
        if not closing["generations"]:
            self.ops.event("CHANNEL_RETIRE", plan_id=plan["plan_id"], closed=[])
            return True
        for name, receipt in channel_receipts(self.run_dir):
            if receipt.get("completed_unix_s", 0) >= closing["written_at"]:
                closing["closed"].setdefault(name, set()).update(receipt["closed"])
        done = len(closing["closed"]) >= self.target_tp_ranks and all(
            set(closing["generations"]) <= seen for seen in closing["closed"].values()
        )
        if not done:
            if self.clock() - closing["written_at"] > 60:
                # Fail fast (before any engine stops) instead of idling in
                # FENCED until the workload ends.
                raise RuntimeError(
                    "target ranks did not close retired channels within 60s: "
                    + json.dumps({k: sorted(v) for k, v in closing["closed"].items()})
                )
            return False
        # The channels are unselectable now; their communicators are torn
        # down in the background. Wait briefly to record the outcome, then
        # proceed either way: stopping the source ends the peer regardless.
        closing.setdefault("closed_at", self.clock())
        teardown = {
            f"{rank}/{generation}": status
            for (rank, generation), status in teardown_receipts(self.run_dir).items()
            if generation in closing["generations"]
        }
        finished = sum(s in {"ABORTED", "FAILED"} for s in teardown.values())
        expected = self.target_tp_ranks * len(closing["generations"])
        if finished < expected and self.clock() - closing["closed_at"] < 10:
            return False
        self.ops.event(
            "CHANNEL_RETIRE",
            plan_id=plan["plan_id"],
            closed=closing["generations"],
            ranks=sorted(closing["closed"]),
            teardown=teardown,
            teardown_complete=finished == expected,
            teardown_wait_s=self.clock() - closing["closed_at"],
        )
        return True

    def stop_owned_engines(self, plan):
        self.ops.stop([r.instance_id for r in self.refs(plan)])
        self.ops.event("STOP_RETURN", plan_id=plan["plan_id"])

    def resources_released(self, plan):
        refs = self.refs(plan)
        if any(self.ops.alive(r.instance_id) for r in refs):
            return False
        gpus = [g for r in refs for g in self.registry.get(r).physical_gpu_ids]
        if not self.ops.gpus_released(gpus):
            return False
        self.ops.event("RESOURCES_RELEASED", plan_id=plan["plan_id"], gpus=gpus)
        return True

    def replacement_specs(self, plan):
        """(instance id, generation, launch extra, joined row fields) per new
        engine; deterministic until they are registered, so a pre-start at
        quiesce and the start after release agree."""
        refs = self.refs(plan)
        gpus = sorted(g for r in refs for g in self.registry.get(r).physical_gpu_ids)
        tp = int(plan["target_bundle_mode"][2:])
        generation = max(
            r.ref.instance_generation for r in self.registry.instances.values()
        )
        port = self.base_port + len(self.started)
        mode = self.join_mode(plan)
        sources = len(self.joined_sources())
        specs = []
        for devices in [gpus] if tp > 1 else [[g] for g in gpus]:
            generation += 1
            instance_id = f"{plan['bundle_id']}-g{generation}-tp{tp}"
            extra = dict(
                instance_id=instance_id,
                bundle=plan["bundle_id"],
                tp=tp,
                gpus=devices,
                port=port,
                endpoint=f"http://127.0.0.1:{port}",
                context_limit=self.context_limits[0 if tp == 1 else 1],
                inventory_path=str(self.run_dir / f"inventory_{instance_id}.jsonl"),
            )
            port += 1
            joined = {}
            if mode == "MERGE_TARGET":
                # The absent pair target: its planned port, inventory and run
                # directory (the stager and sessions already point there).
                slot = self.join["target_slot"]
                extra.update(
                    slot,
                    port=int(slot["endpoint"].rsplit(":", 1)[1]),
                    pair_slot="target",
                )
                joined = dict(legacy_pool="tp4")
            elif mode == "SPLIT_SOURCES":
                channel = dict(
                    # 10 apart per channel source, existing ones first.
                    base_port=self.join["channel_base_port"] + 10 * sources,
                    generation=generation,
                    target=self.join["target_id"],
                    target_tp=self.join["target_tp"],
                )
                sources += 1
                joined = dict(
                    migration_channel=channel,
                    predictor_events=str(
                        self.run_dir / f"predictor_events_{instance_id}.jsonl"
                    ),
                )
                extra.update(
                    joined,
                    targets_file=str(
                        Path(self.join["folder"]) / f"gpu_targets_{instance_id}.json"
                    ),
                )
            specs.append((instance_id, generation, extra, joined))
        return specs

    def prestart(self, plan):
        """Start the replacements behind their start gate while draining.

        Process start, imports and worker spawn (about two thirds of a
        rebuild, S5b GPU 11-01) overlap the drain; each worker waits before
        touching CUDA until ``start_replacements`` opens its gate.
        """
        if not self.prestart_enabled or self.prestarted is not None:
            return
        self.prestarted = {}
        for instance_id, _, extra, _ in self.replacement_specs(plan):
            gated = dict(
                extra, start_gate=str(self.run_dir / f"start_gate_{instance_id}")
            )
            self.prestarted[instance_id] = gated
            self.ops.start(gated)
        self.ops.event(
            "PRESTART", plan_id=plan["plan_id"], instances=sorted(self.prestarted)
        )

    def drop_prestarted(self, reason):
        if self.prestarted:
            self.ops.stop(sorted(self.prestarted))
            self.ops.event(
                "PRESTART_DROPPED", instances=sorted(self.prestarted), reason=reason
            )
        self.prestarted = None

    def start_replacements(self, plan):
        refs = self.refs(plan)
        tp = int(plan["target_bundle_mode"][2:])
        identity = self.registry.get(refs[0]).identity
        mode = self.join_mode(plan)
        specs = self.replacement_specs(plan)
        prestarted = self.prestarted
        if prestarted is not None and (
            set(prestarted) != {i for i, _, _, _ in specs}
            or not all(self.ops.alive(i) for i in prestarted)
        ):
            # A gated engine died (or the plan changed): start afresh.
            self.drop_prestarted("not alive at release")
            prestarted = None
        new = []
        for instance_id, generation, extra, joined in specs:
            if prestarted is not None:
                extra = prestarted[instance_id]
            new.append((instance_id, extra, joined))
            self.registry.register(
                pool.InstanceRecord(
                    pool.InstanceRef(instance_id, generation),
                    tp,
                    tuple(extra["gpus"]),
                    plan["bundle_id"],
                    extra["endpoint"],
                    identity,
                    extra["context_limit"],
                )
            )
            self.rows[instance_id] = dict(
                instance_id=instance_id,
                tp_degree=tp,
                context_limit=extra["context_limit"],
                endpoint=extra["endpoint"],
                inventory_path=extra["inventory_path"],
                **joined,
            )
        if mode is not None:
            # Both ends of every new channel are written before any engine
            # touches its GPU: senders retry until the target listens (600s).
            self.open_channels(plan, mode, new)
        for instance_id, extra, _ in new:
            self.started[instance_id] = extra
            self.ops.event(
                "LOAD_START",
                instance=instance_id,
                gpus=extra["gpus"],
                tp=tp,
                prestarted=prestarted is not None,
            )
            if prestarted is not None:
                Path(extra["start_gate"]).touch()
            else:
                self.ops.start(extra)
        self.prestarted = None

    # ------------------------------------------------------- S5b channels
    def join_mode(self, plan):
        if self.join is None or plan["bundle_id"] != self.join["bundle"]:
            return None
        return self.join["mode"]

    def joined_sources(self):
        return [i for i, row in self.rows.items() if "migration_channel" in row]

    def open_channels(self, plan, mode, new):
        """Write the channels of the new layout: TP1 sources -> the TP-k.

        MERGE_TARGET: every retained TP1 channel source and the new target.
        SPLIT_SOURCES: the new TP1s and the retained pair target. The target
        follows the channel plan file; each source follows its target file.
        """
        retired = {r.instance_id for r in self.refs(plan)}
        if mode == "MERGE_TARGET":
            target_id = new[0][0]
            channels = {
                source: value
                for source, value in self.join["channels"].items()
                if source not in retired
                and self.registry.instances[source].lifecycle == "ACTIVE"
            }
            files = {s: self.join["target_files"][s] for s in channels}
        else:
            target_id = self.join["target_id"]
            channels = {
                instance_id: dict(
                    base_port=joined["migration_channel"]["base_port"],
                    generation=joined["migration_channel"]["generation"],
                    pool=instance_id,
                )
                for instance_id, _, joined in new
            }
            files = {i: extra["targets_file"] for i, extra, _ in new}
        rows = (
            [] if mode == "MERGE_TARGET" else list(read_json(self.channel_plan) or [])
        )
        rows += [
            dict(source=s, base_port=v["base_port"], generation=v["generation"])
            for s, v in channels.items()
        ]
        atomic_json(self.channel_plan, sorted(rows, key=lambda r: r["source"]))
        for source, value in channels.items():
            atomic_json(
                files[source],
                [
                    dict(
                        target=target_id,
                        host=self.join["target_host"],
                        base_port=value["base_port"],
                        tp=self.join["target_tp"],
                        generation=value["generation"],
                    )
                ],
            )
        self.joining = dict(
            target=target_id,
            channels={
                s: dict(generation=v["generation"], pool=v.get("pool", s))
                for s, v in channels.items()
            },
            written_at=self.clock(),
            ready_at=None,
        )
        self.ops.event(
            "CHANNEL_PLAN_JOINED",
            plan_id=plan["plan_id"],
            target=target_id,
            sources=sorted(channels),
        )

    def channels_ready(self, plan):
        """Both ends warmed for every channel this conversion opened."""
        joining = self.joining
        receipts = preconnect_receipts(self.run_dir)
        missing = []
        for source, value in joining["channels"].items():
            names = [
                f"tp_rank_{rank}-{source}.json"
                for rank in range(self.join["target_tp"])
            ] + [f"source-{value['pool']}-{joining['target']}.json"]
            for name in names:
                row = receipts.get(name) or {}
                if (
                    row.get("status") != "WARMUP_COMPLETE"
                    or row.get("channel_generation") != value["generation"]
                ):
                    missing.append(name)
        if missing:
            if self.clock() - joining["written_at"] > CHANNEL_READY_S + 600:
                raise RuntimeError("joined channels not ready: " + ", ".join(missing))
            return False
        if joining["ready_at"] is None:
            joining["ready_at"] = self.clock()
            self.ops.event(
                "CHANNELS_READY",
                plan_id=plan["plan_id"],
                target=joining["target"],
                sources=sorted(joining["channels"]),
                ready_s=round(joining["ready_at"] - joining["written_at"], 3),
            )
        return True

    def warm_and_verify(self, plan):
        """Health + warmup of every replacement, off the executor thread."""
        new = [
            i
            for i, extra in self.started.items()
            if extra["bundle"] == plan["bundle_id"]
        ]
        with self.lock:
            for instance_id in new:
                if instance_id not in self.warm:
                    self.warm[instance_id] = dict(done=False, error=None)
                    threading.Thread(
                        target=self._warm, args=(instance_id,), daemon=True
                    ).start()
            rows = [self.warm[i] for i in new]
        errors = [r["error"] for r in rows if r["error"] is not None]
        if errors:
            raise RuntimeError(f"replacement warmup failed: {errors[0]}")
        if not (bool(new) and all(r["done"] for r in rows)):
            return False
        mode = self.join_mode(plan)
        if mode == "MERGE_TARGET" and not self.stager_started:
            # The merged TP-k is the pair target: its stager starts with it.
            self.ops.start_stager()
            self.stager_started = True
            self.ops.event("STAGER_STARTED", plan_id=plan["plan_id"])
        return mode is None or self.channels_ready(plan)

    def _warm(self, instance_id):
        try:
            self.ops.warm(self.started[instance_id])
            self.ops.event("VERIFIED", instance=instance_id)
            with self.lock:
                self.warm[instance_id]["done"] = True
        except Exception as error:  # surfaced by the next executor step
            with self.lock:
                self.warm[instance_id]["error"] = repr(error)

    def publish_ready(self, plan):
        new = [
            i
            for i, extra in self.started.items()
            if extra["bundle"] == plan["bundle_id"]
        ]
        for instance_id in new:
            self.registry.transition(self.registry.instances[instance_id].ref, "ACTIVE")
        self.write_layout(dict.fromkeys(new, "ACTIVE"))
        self.ops.event("ROUTE_PUBLISHED", plan_id=plan["plan_id"], instances=new)

    def settle_before_abort(self, plan):
        return True

    def restore_admission(self, plan):
        self.drop_prestarted("conversion aborted")
        self.write_layout({r.instance_id: "ACTIVE" for r in self.refs(plan)})
        self.ops.event("ADMISSION_RESTORED", plan_id=plan["plan_id"])


def nvidia_gpus_released(gpu_ids):
    """No compute process remains on these physical GPUs (nvidia-smi)."""
    import subprocess

    uuids = {
        index.strip(): uuid.strip()
        for index, uuid in (
            line.split(",")
            for line in subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                text=True,
            ).splitlines()
        )
    }
    if not set(gpu_ids) <= set(uuids):
        raise RuntimeError("GPU UUID mapping incomplete")
    busy = {
        line.split(",")[0].strip()
        for line in subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()
        if line.strip()
    }
    return not busy & {uuids[g] for g in gpu_ids}


def scrape_metrics(endpoint, native):
    """Fresh engine metrics for drain revalidation; None when unreachable."""
    telemetry = load("telemetry")
    scraper = telemetry.MetricsScraper(
        endpoint,
        native["block_size"],
        native["usage_denominator_blocks"],
        timeout_s=1,
    )
    try:
        return load("pressure_observation").scrape_inventory_metrics(scraper, native)
    except telemetry.TelemetryError:
        return None


def read_latest(path):
    from tools.bridge_tp.run_pressure_calibration import latest_inventory

    return latest_inventory(path) if Path(path).exists() else None


class ConversionDriver:
    """Advance one explicit plan from a thread at a fixed cadence.

    ``request`` = dict(bundle, target_mode, trigger_s, deadline_s); the trigger
    counts from ``start()``. The driver never creates a second plan.

    With ``decider`` (RECORD_ONLY or EXECUTE, plus ``edges``) the rule-based
    decider is evaluated every ``decision_interval_s`` and every record goes
    to ``conversion_decider.jsonl``. RECORD_ONLY never converts; EXECUTE
    converts the first proposal (one plan per run for now). In the decider's
    prepare stage, EXECUTE with ``prestart`` starts the replacements and
    ``pre_tilt`` keeps new long requests off the bundle; ``assist`` drains a
    merging bundle with SELECTIVE_ROLL.
    """

    decision_interval_s = 1.0

    TERMINAL = {"COMPLETED", "ABORTED", "RECOVERY_REQUIRED"}

    def __init__(self, backend, request, journal_path, interval_s=0.2):
        self.backend, self.request = backend, request
        self.executor = conversion.ConversionExecutor(
            backend.registry, backend, journal_path, clock=time.time
        )
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.error = None
        self.thread = None
        self.triggered_at = None
        self.decider = None
        if "decider" in request:
            from tools.bridge_tp.conversion_decider import ConversionDecider

            self.decider = ConversionDecider(
                request["edges"],
                request.get("rules"),
                execute=request["decider"] == "EXECUTE",
            )
        self.decided_at = -float("inf")
        self.decisions = 0
        self.proposals = []
        self.tilt_since = None
        self.prepared_key = None

    def start(self):
        self.origin = time.monotonic()
        # The controller must not finish before this plan reaches its end; a
        # decider holds it only once it has created a plan.
        self.backend.write_layout({}, pending=self.decider is None)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def state(self):
        return self.executor.plan["state"] if self.executor.plan else "NOT_TRIGGERED"

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                if self.executor.plan is None and self.decider is not None:
                    self.decide()
                elif self.executor.plan is None:
                    if time.monotonic() - self.origin >= self.request["trigger_s"]:
                        self.triggered_at = time.time()
                        self.backend.ops.event("TRIGGER", request=self.request)
                        self.executor.plan_create(
                            "manual-" + self.request["bundle"],
                            self.backend.registry.registry_revision,
                            self.request["bundle"],
                            self.request["target_mode"],
                            self.request.get("method", "DRAIN"),
                            "ACTIVE",
                            time.time() + self.request["deadline_s"],
                            "NO_PROFILE",
                        )
                else:
                    self.end_expired_tilt()
                    before = self.state()
                    after = self.executor.plan_execute(time.time())["state"]
                    if after != before:
                        self.backend.ops.event("PLAN_STATE", state=after)
                    if after in self.TERMINAL:
                        return
                self.stop_event.wait(self.interval_s)
        except Exception as error:
            self.error = repr(error)
            self.backend.ops.event("DRIVER_ERROR", error=self.error)
        finally:
            # Terminal, failed or stopped: release the controller.
            self.backend.write_layout({}, pending=False)

    def decide(self):
        """One decider evaluation; EXECUTE turns the first proposal into a plan."""
        if time.monotonic() - self.decided_at < self.decision_interval_s:
            return
        self.decided_at = time.monotonic()
        status = self.backend.status()
        if not self.backend.fresh(status):
            return
        from tools.bridge_tp.conversion_decider import registry_layout

        bundles, members = registry_layout(self.backend.registry)
        record = self.decider.observe(status, bundles, members, time.time())
        self.decisions += 1
        with (self.backend.run_dir / "conversion_decider.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(record) + "\n")
        self.apply_prepared(record, members)
        proposal = record["proposal"]
        if proposal is None:
            return
        self.proposals.append(dict(proposal, unix_s=record["evaluated_unix_s"]))
        self.backend.ops.event("DECIDER_PROPOSAL", mode=record["mode"], **proposal)
        if not self.decider.execute:
            return
        self.triggered_at = time.time()
        self.request = dict(
            self.request, bundle=proposal["bundle"], target_mode=proposal["target_mode"]
        )
        self.backend.ops.event("TRIGGER", request=self.request)
        # The start gate's service check ran before the load this tilt added.
        self.backend.service_gate_passed = bool(self.backend.layout_tilt)
        # Hold the controller until terminal; the tilt stays until routing
        # closes for the bundle at quiesce.
        self.backend.write_layout({})
        self.executor.plan_create(
            "decided-" + proposal["bundle"],
            self.backend.registry.registry_revision,
            proposal["bundle"],
            proposal["target_mode"],
            "SELECTIVE_ROLL"
            if self.request.get("assist") and proposal["target_mode"] != "TP1"
            else "DRAIN",
            "ACTIVE",
            time.time() + self.request["deadline_s"],
            "NO_PROFILE",
        )
        self.decider.converted(time.time())

    def apply_prepared(self, record, members):
        """Tilt and pre-start of the decider's prepare stage, while it lasts."""
        request, backend = self.request, self.backend
        if request.get("pre_tilt"):
            tilt = sorted(i for b in record["tilt"] for i in members.get(b, ()))
            if tilt != backend.layout_tilt:
                backend.write_layout({}, pending=False, tilt=tilt)
                backend.ops.event("PRE_TILT", instances=tilt)
                self.tilt_since = time.monotonic() if tilt else None
        if not (self.decider.execute and backend.prestart_enabled):
            return
        prepared = record["prepared"]
        key = prepared["key"] if prepared else None
        if key == self.prepared_key:
            return
        backend.drop_prestarted("the decider's candidate stopped holding")
        self.prepared_key = key
        if prepared is not None:
            refs = [
                dict(
                    instance_id=i,
                    instance_generation=backend.registry.instances[
                        i
                    ].ref.instance_generation,
                )
                for i in members[prepared["bundle"]]
            ]
            backend.prestart(
                dict(
                    plan_id="prepare-" + prepared["bundle"],
                    bundle_id=prepared["bundle"],
                    target_bundle_mode=prepared["target_mode"],
                    refs=refs,
                )
            )

    def end_expired_tilt(self):
        """The tilt's time limit also covers a start gate still waiting."""
        if (
            self.tilt_since is not None
            and self.backend.layout_tilt
            and time.monotonic() - self.tilt_since > self.decider.rules["tilt_max_s"]
        ):
            self.tilt_since = None
            self.backend.write_layout({}, tilt=[])
            self.backend.ops.event("PRE_TILT", instances=[])

    def stop(self, timeout_s=30):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout_s)
        return dict(
            state=self.state(),
            error=self.error,
            triggered_unix_s=self.triggered_at,
            plan=self.executor.plan_status() if self.executor.plan else None,
            driver_alive=self.thread is not None and self.thread.is_alive(),
            **(
                dict(decisions=self.decisions, proposals=self.proposals)
                if self.decider is not None
                else {}
            ),
        )
