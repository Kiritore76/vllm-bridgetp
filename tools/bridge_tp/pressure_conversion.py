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
    assist_factor=3.0,
    assist_max_starts=2,
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
        self.safety = dict(SAFETY, **(safety or {}))
        self.speed_margin = speed_margin
        self.gate_since = None
        self.valve_since = None
        self.assist_at = -float("inf")
        self.assist_seq = 0
        self.assisted = []  # request ids whose assist START was published
        self.last_assist_reason = None
        # S4: read-only planner consulted while draining; never acts.
        self.shadow_planner, self.shadow_at = shadow_planner, -float("inf")
        self.reported_unacknowledged = False
        self.lock = threading.Lock()
        # Engines are serving when the controller starts: the manifest's
        # PREPARING records become ACTIVE before the first plan.
        for row in registry.instances.values():
            if row.lifecycle == "PREPARING":
                registry.transition(row.ref, "ACTIVE")

    # ------------------------------------------------------------- helpers
    def refs(self, plan):
        return [pool.InstanceRef(**r) for r in plan["refs"]]

    def write_layout(self, changes, *, pending=True, assist=None):
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
            )
        self.revision += 1
        if assist is not None:
            self.layout_assist = assist
        atomic_json(
            self.run_dir / "layout_control.json",
            dict(
                revision=self.revision,
                instances=list(self.layout.values()),
                conversion_pending=pending,
                assist=assist or self.layout_assist,
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
        elif waited >= self.safety["relax_after_s"]:
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
        self.write_layout({r.instance_id: "QUIESCING" for r in self.refs(plan)})

    def kv_emergency(self, plan):
        """Safety valve: retained KV about to hit its guard, sustained.

        Only hard safety aborts (before any engine stops): the plan is then
        cancelled and routing restored. Service-quality drops do not abort,
        or a conversion under load would cancel and retry forever.
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
        self.decision("VALVE_ABORT", instances=unsafe)
        self.ops.event("SAFETY_ABORT", plan_id=plan["plan_id"], instances=unsafe)
        return True

    def assisted_step(self, plan, now):
        """S5a: migrate the next long request out, or leave the drain natural.

        Rolling: at most one assist START at a time and a fresh decision after
        each. Eligible: a startable request whose conservative remaining time
        exceeds ``assist_factor`` x the measured START-to-release budget,
        longest first. The target must be SPARE (service quality is never
        traded for a faster drain here), at most ``assist_max_starts`` assists
        start per conversion, and the controller re-checks KV for the final
        length and health before starting.
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
        self.write_layout(
            {},
            assist=dict(
                seq=self.assist_seq,
                request_id=choice["request_id"],
                source=choice["instance"],
            ),
        )
        self.ops.event("ASSIST_REQUESTED", seq=self.assist_seq, **choice)

    def assist_choice(self, plan, status):
        if not self.fresh(status):
            return "status stale", None
        result = status.get("assist_result")
        if self.assist_seq and (result is None or result["seq"] != self.assist_seq):
            return "previous assist not answered yet", None
        if (
            result is not None
            and result["seq"] == self.assist_seq
            and (
                result["state"] == "STARTED"
                and result["request_id"] not in self.assisted
            )
        ):
            self.assisted.append(result["request_id"])
        target = status.get("migration_target")
        instances = status["instances"]
        if target is None or instances.get(target, {}).get("lifecycle") != "ACTIVE":
            return "no active migration target", None
        refs = [r.instance_id for r in self.refs(plan)]
        if any(instances[i]["sessions"] for i in refs) or instances[target]["sessions"]:
            return "a migration session is active", None
        # S4: two received migrations cost ~1.3 GB on the 24 GB TP2 (about
        # 1 GB left), so the total of assist starts per conversion is capped.
        if len(self.assisted) >= self.safety["assist_max_starts"]:
            return "assist starts at cap for this conversion", None
        pressure = instances[target].get("pressure") or {}
        if pressure.get("zone") != "SPARE":
            return f"target not SPARE ({pressure.get('zone')})", None
        budget = status.get("release_budget_s")
        if budget is None:
            return "no measured transfer budget", None
        best = None
        for instance in refs:
            for row in instances[instance]["requests"]:
                speed = row.get("tokens_s")
                if not row.get("action_available") or not speed:
                    continue
                remaining = max(u for u, _ in row["remaining_bins"])
                natural = remaining / (speed * self.speed_margin)
                if natural <= self.safety["assist_factor"] * budget:
                    continue
                if best is None or natural > best["natural_s"]:
                    best = dict(
                        request_id=row["request_id"],
                        instance=instance,
                        natural_s=round(natural, 3),
                        transfer_s=budget,
                    )
        if best is None:
            return "no request worth migrating (natural <= factor x transfer)", None
        return None, best

    def shadow(self, plan, status):
        now = self.clock()
        if self.shadow_planner is None or now - self.shadow_at < 1:
            return
        self.shadow_at = now
        try:
            record = self.shadow_planner(plan, status, now)
        except Exception as error:  # a planner fault must never stop the drain
            record = dict(evaluated_unix_s=now, error=repr(error))
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
            closing = sorted(
                r["generation"] for r in rows if r["generation"] in generations
            )
            written_at = self.clock()
            if closing:
                atomic_json(
                    self.channel_plan,
                    [r for r in rows if r["generation"] not in generations],
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

    def start_replacements(self, plan):
        refs = self.refs(plan)
        gpus = sorted(g for r in refs for g in self.registry.get(r).physical_gpu_ids)
        tp = int(plan["target_bundle_mode"][2:])
        generation = max(
            r.ref.instance_generation for r in self.registry.instances.values()
        )
        port = self.base_port + len(self.started)
        identity = self.registry.get(refs[0]).identity
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
            self.registry.register(
                pool.InstanceRecord(
                    pool.InstanceRef(instance_id, generation),
                    tp,
                    tuple(devices),
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
            )
            self.started[instance_id] = extra
            self.ops.event("LOAD_START", instance=instance_id, gpus=devices, tp=tp)
            self.ops.start(extra)

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
        return bool(new) and all(r["done"] for r in rows)

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
    """

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

    def start(self):
        self.origin = time.monotonic()
        # The controller must not finish before this plan reaches its end.
        self.backend.write_layout({})
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def state(self):
        return self.executor.plan["state"] if self.executor.plan else "NOT_TRIGGERED"

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                if self.executor.plan is None:
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
        )
