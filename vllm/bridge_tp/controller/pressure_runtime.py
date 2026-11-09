# SPDX-License-Identifier: Apache-2.0
"""Serialized pressure-policy harness and explicitly bound action callbacks.

The callbacks integrate the existing M2/M3/M4 lifecycle; they must not launch a
fresh source request, alter its stop contract, or bypass four-rank readiness.
The experimental real pilot ingress calls this coordinator. Replay is CPU-only.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

from . import pressure_multi as multi
from . import unified_scheduler as unified
from .bounded_length import from_advisory
from .pressure_policy import (
    Admission,
    Candidate,
    PoolReading,
    PressureConfig,
    PressurePolicy,
    Reservation,
    ReservationLedger,
    RoutingError,
    admit,
    classify,
)


def rank_migration_targets(names, fullness, readings, sort_ids):
    """Shared normal/assist ordering: fullness, pending prompts, stable ID."""

    def promised(target):
        value = readings[target].unallocated_prompt_blocks
        return value if type(value) is int and value > 0 else 0

    return sorted(
        names, key=lambda t: (fullness[t]["value"], promised(t), sort_ids.get(t, t))
    )


@dataclass(frozen=True)
class BoundAction:
    request_id: str
    session_id: str
    # START delegates to the existing mechanism and returns only after its
    # control publication. Exceptions may have happened AFTER publication.
    start: Callable[[], None]


class PressureCoordinator:
    def __init__(
        self,
        cfg: PressureConfig,
        audit: Callable[[dict], None],
        *,
        routing_strategy="pressure",
        routing_weights=(1.0, 1.0),
        scheduler="pressure",
        slots=None,
    ) -> None:
        if scheduler not in {"pressure", "unified"}:
            raise ValueError("unknown scheduler")
        if scheduler == "unified" and not slots:
            raise ValueError("the unified scheduler needs every instance's slots")
        # Unified (user 11-06): one fullness for admission and migration; the
        # pressure scheduler below stays unchanged behind this switch.
        self.scheduler = scheduler
        self.slots = dict(slots or {})
        self.itl = unified.SmoothedItl()
        self.itl_reference_s = unified.ITL_REFERENCE_S
        self.migration_state = unified.MigrationState()
        self.ledger = ReservationLedger()
        self.policy = PressurePolicy(cfg, self.ledger)
        self.audit = audit
        self._lock = threading.RLock()
        self._sessions: dict[str, tuple[str, str]] = {}
        self._owners: dict[str, str] = {}
        self.routing_strategy, self.routing_weights = routing_strategy, routing_weights
        # Per-source trigger state; the first source reuses self.policy so a
        # single-source run is the historical coordinator.
        self._policies: dict[str, PressurePolicy] = {}
        # Instance ids carry their generation (<bundle>-g<gen>-tp<k>); unknown
        # ids are ACTIVE so the fixed pair needs no registration.
        self._lifecycle: dict[str, str] = {}
        self._session_source: dict[str, str] = {}
        # S5a: sessions started for a conversion's drain, not for relief.
        self._assist_sessions: set[str] = set()

    def lifecycle(self, instance_id: str) -> str:
        return self._lifecycle.get(instance_id, "ACTIVE")

    def set_lifecycle(self, instance_id: str, state: str) -> None:
        """ACTIVE <-> QUIESCING -> RETIRED.

        QUIESCING closes new admission and migration INTO the instance; it may
        still migrate its own requests out. RETIRED needs every migration that
        names it to be settled and drops its trigger state.
        """
        allowed = {
            "ACTIVE": {"QUIESCING"},
            "QUIESCING": {"ACTIVE", "RETIRED"},
            "RETIRED": set(),
        }
        with self._lock:
            current = self.lifecycle(instance_id)
            if state == current:
                return
            if state not in allowed[current]:
                raise ValueError("invalid instance lifecycle transition")
            if state == "RETIRED" and any(
                instance_id in (target, self._session_source.get(session))
                and self.ledger.entries[session].state == "ACTIVE"
                for session, (_, target) in self._sessions.items()
            ):
                raise ValueError("unsettled migration prevents retirement")
            self._lifecycle[instance_id] = state
            if state == "RETIRED":
                self.forget_source(instance_id)
        self.audit(
            {
                "kind": "pressure_instance_lifecycle",
                "instance_id": instance_id,
                "from": current,
                "to": state,
            }
        )

    def _bound_target(self, session_id: str, target_id: str | None) -> str:
        actual = self._sessions[session_id][1]
        if target_id is not None and target_id != actual:
            raise ValueError("receipt names another target instance generation")
        return actual

    def observe_migration(self, session_id: str, target_id: str, **evidence) -> None:
        """Target allocation evidence, accepted only from the bound instance."""
        with self._lock:
            self._bound_target(session_id, target_id)
            self.ledger.observe(session_id, **evidence)

    def set_config(self, cfg: PressureConfig) -> None:
        """Apply a configuration change (e.g. survival horizon) to every source."""
        with self._lock:
            self.policy.cfg = cfg
            for policy in self._policies.values():
                policy.cfg = cfg

    def policy_for(self, source_id: str) -> PressurePolicy:
        policy = self._policies.get(source_id)
        if policy is None:
            policy = (
                self.policy
                if not self._policies
                else PressurePolicy(self.policy.cfg, self.ledger, self.policy.shared)
            )
            self._policies[source_id] = policy
        return policy

    def forget_source(self, source_id: str) -> None:
        """Drop a retired source generation; a replacement starts fresh."""
        with self._lock:
            policy = self._policies.pop(source_id, None)
            if policy is self.policy:
                self.policy = PressurePolicy(policy.cfg, self.ledger, policy.shared)

    def admission(self, *args, **kwargs):
        """Route and track identity atomically, without reserving ingress KV."""
        with self._lock:
            result, record = self._admit_pair(self.policy, args, kwargs)
        self.audit(record)
        return result

    def _admit_pair(self, policy, args, kwargs):
        # Admission only adds rows. Recording the whole ever-growing ledger
        # twice made each arrival O(all past requests) and capped ingress.
        before, known = self.ledger.live_snapshot(), self.ledger.mark()
        pools = args[3] if len(args) > 3 else kwargs["pools"]
        now = kwargs.get("now", args[4] if len(args) > 4 else None)
        policy.observe_relief(now, pools[0])
        if self.routing_strategy == "pressure":
            policy.relief.acknowledge(now)
        relief = policy.relief.snapshot()
        kv_before = policy.kv_route.snapshot()
        result = admit(
            *args,
            cfg=policy.cfg,
            ledger=self.ledger,
            routing_strategy=self.routing_strategy,
            routing_weights=self.routing_weights,
            routing_relief=relief,
            kv_routing_state=kv_before,
            **kwargs,
        )
        if result.routing_kv is not None:
            policy.kv_route.observe(result.routing_kv["source"], policy.cfg, now)
        if policy.cfg.routing_relief:
            policy.relief.admitted(
                result.pool_id,
                args[1] if len(args) > 1 else kwargs["prompt_tokens"],
                now,
                result.routing_reason.startswith("router relief epoch")
                or bool(result.routing_kv and result.routing_kv["redirect"]),
                source_pool_id=pools[0].pool_id,
            )
            if result.pool_id == pools[0].pool_id:
                policy._since = None
                policy._service_confirmation = None
        self._owners[result.request_id] = result.pool_id
        record = {
            "kind": "pressure_admission",
            "admission": asdict(result),
            "pool_readings": [asdict(pool) for pool in pools],
            "routing_config": asdict(policy.cfg),
            "routing_strategy": self.routing_strategy,
            "routing_weights": self.routing_weights,
            "routing_relief": relief,
            "routing_relief_after": policy.relief.snapshot(),
            "decision_unix_s": now,
            "config_sha256": policy.cfg.sha256,
            "reservation_scope": "live",
            "reservation_before": before,
            "reservation_added": self.ledger.added_since(known),
        }
        if policy.cfg.kv_routing_priority:
            record["kv_routing_state"] = kv_before
            record["kv_routing_state_after"] = policy.kv_route.snapshot()
        return result, record

    def admission_multi(
        self,
        request_id: str,
        prompt_tokens: int,
        requested_output: int,
        readings: list[PoolReading],
        tp_of: dict[str, int],
        now: float,
        key: str,
        allowed_pool_id: str | None = None,
    ):
        """Rank inside each TP class, then the unchanged two-pool ``admit``."""
        with self._lock:
            readings = [r for r in readings if self.lifecycle(r.pool_id) == "ACTIVE"]
            if not readings or (
                allowed_pool_id is not None
                and allowed_pool_id not in {r.pool_id for r in readings}
            ):
                raise RoutingError({"admission": dict(reasons=["no ACTIVE instance"])})
            if self.scheduler == "unified":
                result, record = self._admit_unified(
                    request_id,
                    prompt_tokens,
                    requested_output,
                    readings,
                    key,
                    now,
                    allowed_pool_id,
                )
                self.audit(record)
                return result
            classes = (
                [r for r in readings if tp_of[r.pool_id] == 1],
                [r for r in readings if tp_of[r.pool_id] > 1],
            )
            rankings = [
                multi.class_ranking(
                    rows,
                    prompt_tokens,
                    requested_output,
                    self.ledger,
                    self.policy.cfg,
                    now,
                    self.routing_strategy,
                )
                for rows in classes
            ]
            by_id = {r.pool_id: r for r in readings}
            leaders = [
                by_id[allowed_pool_id]
                if allowed_pool_id in {row["pool_id"] for row in ranking}
                else by_id[ranking[0]["pool_id"]]
                for ranking in rankings
                if ranking
            ]
            if len(leaders) == 2:
                result, record = self._admit_pair(
                    self.policy_for(leaders[0].pool_id),
                    (request_id, prompt_tokens, requested_output, tuple(leaders)),
                    dict(now=now, key=key, allowed_pool_id=allowed_pool_id),
                )
            else:
                result, record = self._admit_class_only(
                    request_id,
                    prompt_tokens,
                    requested_output,
                    next(r for r in rankings if r),
                    key,
                    now,
                    allowed_pool_id,
                )
            record.update(
                class_ranking=rankings,
                unified_choice=multi.unified_choice(rankings),
                instance_readings=[asdict(r) for r in readings],
            )
        self.audit(record)
        return result

    def observe_itl(self, service, pool_ids, now):
        """Advance every instance's smoothed inter-token interval once."""
        with self._lock:
            for pool_id in pool_ids:
                last = self.itl.snapshot().get(pool_id, {}).get("updated_at", -math.inf)
                seconds, tokens, active = service.decode_progress(pool_id, last, now)
                self.itl.update(pool_id, now, seconds, tokens, active)

    def unified_fullness(self, reading, kind, *, extra_requests=0, extra_blocks=0):
        """Fullness of a routing-adjusted reading; ``kind`` picks the smoothing."""
        unobserved, _ = self.ledger.unobserved_ingress(reading.pool_id)
        return unified.fullness(
            reading,
            self.slots[reading.pool_id],
            self.itl.value(reading.pool_id, kind),
            extra_requests=unobserved + extra_requests,
            extra_blocks=extra_blocks,
            itl_reference_s=self.itl_reference_s,
        )

    def cancellation_source_need(self, source, now, expected_pool_id):
        """Evidence from the same adjusted source and migrate EMA as START.

        ``source`` is already ledger.routing_reading-adjusted by the session
        caller; applying that adjustment twice would double count promises.
        Instance IDs include their generation and must match the bound owner.
        """
        with self._lock:
            value = dict(slots=None, kv=None, speed=None, value=None)
            if source.pool_id in self.slots:
                value = self.unified_fullness(source, "migrate")
            capacity = classify(source, self.policy.cfg, now).capacity
            exhausting = (
                capacity >= 1
                if capacity is not None and math.isfinite(capacity)
                else None
            )
            snapshot = dict(
                value,
                kind="migrate",
                pool_id=source.pool_id,
                expected_pool_id=expected_pool_id,
                observed_at=now,
                sampled_at=source.sampled_at,
                published_at=source.published_at,
                itl_sampled_at=self.itl.snapshot()
                .get(source.pool_id, {})
                .get("updated_at"),
                legacy_capacity=capacity,
                exhausting=exhausting,
                complete=False,
                state="UNKNOWN",
                need=None,
                reason="incomplete, stale or mismatched source evidence",
            )
            complete = (
                self.scheduler == "unified"
                and source.pool_id == expected_pool_id
                and self.lifecycle(source.pool_id) in {"ACTIVE", "QUIESCING"}
                and type(source.engine_requests) is int
                and source.engine_requests >= 0
                and type(source.unallocated_prompt_blocks) is int
                and source.unallocated_prompt_blocks >= 0
                and type(source.total_blocks) is int
                and source.total_blocks > 0
                and type(source.free_blocks) is int
                and 0 <= source.free_blocks <= source.total_blocks
                and exhausting is not None
                and source.growth_blocks_s is not None
                and math.isfinite(source.growth_blocks_s)
                and source.growth_blocks_s >= 0
                and all(
                    type(v) in (int, float) and math.isfinite(v) and v >= 0
                    for v in value.values()
                )
                and all(
                    type(snapshot[k]) in (int, float)
                    and math.isfinite(snapshot[k])
                    and 0 <= now - snapshot[k] <= self.policy.cfg.max_age_s
                    for k in (
                        "observed_at",
                        "sampled_at",
                        "published_at",
                        "itl_sampled_at",
                    )
                )
            )
            if complete:
                need = unified.source_migration_need(value, exhausting)
                snapshot.update(
                    complete=True,
                    need=need,
                    state="NEEDED" if need else "CLEAR",
                    reason="source migration demand remains"
                    if need
                    else "source below migration triggers",
                )
            return snapshot

    def route_fullness_snapshot(self, readings, now):
        """Export current load for conversion, without a hypothetical new prompt.

        Routing tolerates missing components while ranking. Conversion must
        instead wait for all three components and their original timestamps.
        """
        with self._lock:
            itl = self.itl.snapshot()
            result = {}
            for reading in readings:
                pool_id = reading.pool_id
                value = self.unified_fullness(
                    self.ledger.routing_reading(reading), "route"
                )
                result[pool_id] = dict(
                    value,
                    kind="route",
                    observed_at=now,
                    sampled_at=reading.sampled_at,
                    published_at=reading.published_at,
                    itl_sampled_at=itl.get(pool_id, {}).get("updated_at"),
                    complete=(
                        type(reading.engine_requests) is int
                        and reading.engine_requests >= 0
                        and reading.free_blocks is not None
                        and reading.total_blocks is not None
                        and 0 <= reading.free_blocks <= reading.total_blocks
                        and all(v is not None for v in value.values())
                    ),
                )
            return result

    def _admit_unified(
        self, request_id, prompt_tokens, requested_output, readings, key, now, allowed
    ):
        """Least full instance that holds the prompt; one rule for every TP."""
        if (
            type(prompt_tokens) is not int
            or prompt_tokens < 0
            or type(requested_output) is not int
            or requested_output <= 0
        ):
            raise ValueError("invalid request lengths")
        output_cap = min(8192, requested_output)
        rows = []
        for raw in readings:
            reading = self.ledger.routing_reading(raw)
            prompt_blocks = (
                math.ceil(prompt_tokens / reading.block_size)
                if reading.block_size > 0
                else None
            )
            room = unified.prompt_room(reading, self.ledger.pending(raw.pool_id)[0])
            rows.append(
                dict(
                    pool_id=raw.pool_id,
                    allowed=allowed in (None, raw.pool_id),
                    context_feasible=reading.context_limit > 0
                    and prompt_tokens + output_cap <= reading.context_limit,
                    fits=room is not None
                    and prompt_blocks is not None
                    and room >= prompt_blocks,
                    room_blocks=room,
                    waiting=reading.waiting,
                    fullness=self.unified_fullness(
                        reading,
                        "route",
                        extra_requests=1,
                        extra_blocks=prompt_blocks or 0,
                    ),
                )
            )
        feasible = [r for r in rows if r["allowed"] and r["context_feasible"]]
        if not feasible:
            reason = dict(reasons=["context/stop budget exhausted"])
            raise RoutingError({r["pool_id"]: reason for r in rows})
        # The hold queue sends a request only where its prompt fits; without
        # one (fixed routes, stale evidence) the least full feasible one gets it.
        fitting = [r for r in feasible if r["fits"]]
        chosen = unified.rank_admission(
            fitting or feasible, getattr(self, "pool_sort_ids", None)
        )[0]
        before, known = self.ledger.live_snapshot(), self.ledger.mark()
        self.ledger.track_ingress(key, request_id, chosen["pool_id"], prompt_tokens)
        result = Admission(
            request_id,
            chosen["pool_id"],
            output_cap,
            key,
            "fixed pilot route"
            if allowed
            else "unified: least full instance holding the prompt"
            if fitting
            else "unified: least full feasible instance; prompt fit unknown/none",
            routing_capacity={r["pool_id"]: r["fits"] for r in rows},
            routing_pressure={r["pool_id"]: r["fullness"] for r in rows},
        )
        self._owners[request_id] = result.pool_id
        record = {
            "kind": "pressure_admission",
            "scheduler": "unified",
            "admission": asdict(result),
            "unified_rows": rows,
            "smoothed_itl": self.itl.snapshot(),
            "routing_strategy": self.routing_strategy,
            "decision_unix_s": now,
            "reservation_scope": "live",
            "reservation_before": before,
            "reservation_added": self.ledger.added_since(known),
        }
        return result, record

    def _admit_class_only(
        self, request_id, prompt_tokens, requested_output, ranking, key, now, allowed
    ):
        """Single-class layout: the class leader, with admit's length contract."""
        if (
            type(prompt_tokens) is not int
            or prompt_tokens < 0
            or type(requested_output) is not int
            or requested_output <= 0
        ):
            raise ValueError("invalid request lengths")
        feasible = [
            row
            for row in ranking
            if row["context_feasible"] and allowed in (None, row["pool_id"])
        ]
        if not feasible:
            raise RoutingError(
                {
                    row["pool_id"]: dict(reasons=["context/stop budget exhausted"])
                    for row in ranking
                }
            )
        # Admission only adds rows. Recording the whole ever-growing ledger
        # twice made each arrival O(all past requests) and capped ingress.
        before, known = self.ledger.live_snapshot(), self.ledger.mark()
        chosen = feasible[0]
        self.ledger.track_ingress(key, request_id, chosen["pool_id"], prompt_tokens)
        result = Admission(
            request_id,
            chosen["pool_id"],
            min(8192, requested_output),
            key,
            "fixed pilot route" if allowed else "class-only: " + chosen["basis"],
            routing_capacity={row["pool_id"]: row["fit"] for row in ranking},
            routing_pressure={
                row["pool_id"]: dict(
                    service=row["service"],
                    capacity=row["capacity"],
                    score=row["score"],
                )
                for row in ranking
            },
        )
        self._owners[request_id] = result.pool_id
        record = {
            "kind": "pressure_admission",
            "admission": asdict(result),
            "routing_config": asdict(self.policy.cfg),
            "routing_strategy": self.routing_strategy,
            "decision_unix_s": now,
            "config_sha256": self.policy.cfg.sha256,
            "reservation_scope": "live",
            "reservation_before": before,
            "reservation_added": self.ledger.added_since(known),
        }
        return result, record

    @staticmethod
    def with_predictions(
        candidates: list[Candidate],
        reader,
        now: float,
        headroom_tokens: int,
        ignore_eos: dict[str, bool],
    ) -> list[Candidate]:
        """Query M5 for EVERY live candidate, preserving identity/age/lag."""
        result = []
        for candidate in candidates:
            row = reader.advisory(
                candidate.request_id,
                candidate.generated,
                headroom_tokens,
                now_ns=int(now * 1e9),
                max_output_tokens=candidate.max_output_tokens,
                ignore_eos=ignore_eos[candidate.request_id],
            )
            remaining = None
            status = row["status"]
            captured = row.get("captured_unix_ns")
            published = row.get("published_unix_ns")
            if status == "AVAILABLE" and (
                captured is None
                or published is None
                or captured > now * 1e9
                or published > now * 1e9
                or published < captured
            ):
                status = "PUBLICATION_NOT_PROVEN"
            if status == "AVAILABLE":
                try:
                    remaining = from_advisory(row, candidate.max_output_tokens)
                except ValueError:
                    status = "INVALID_STOP_ALIGNMENT"
            result.append(
                replace(
                    candidate,
                    remaining=remaining,
                    prediction_status=status,
                    prediction_age_s=row.get("age_s"),
                    prediction_lag=row.get("prediction_lag_tokens"),
                    pmf_reference=json_reference(row),
                )
            )
        return result

    def tick(
        self,
        now: float,
        source: PoolReading,
        target: PoolReading,
        candidates: list[Candidate],
        actions: dict[str, BoundAction] | None = None,
    ):
        """Decide once, reserve, audit, then execute the SAME request if enabled.

        actions=None is advisory replay. Sessions that may have STARTed remain
        reserved after exceptions until authoritative cleanup is supplied.
        """
        action = None
        with self._lock:
            for c in candidates:
                owner = self._owners.get(c.request_id)
                if owner is not None and owner != c.owner:
                    raise ValueError("inventory contradicts confirmed ownership")
            # Each tick recording every past row made it O(run length) and the
            # audit hundreds of MB (GPU 11-03); settlements keep the full ledger.
            before = self.ledger.live_snapshot()
            self.policy.observe_relief(now, source)
            if self.routing_strategy == "pressure":
                self.policy.relief.acknowledge(now)
            decision = self.policy.decide(now, source, target, candidates)
            record = {
                "kind": "pressure_decision",
                "unix_s": now,
                "decision": asdict(decision),
                "source_reading": asdict(self.ledger.routing_reading(source)),
                "target_reading": asdict(self.ledger.routing_reading(target)),
                "reservation_scope": "live",
                "reservation_before": before,
                "execution": "ADVISORY",
                "routing_relief": self.policy.relief.snapshot(),
                "prediction_inputs": prediction_inputs(candidates),
            }
            if decision.request_id is None or actions is None:
                record["reservation_after"] = self.ledger.live_snapshot()
            else:
                selected = next(
                    c for c in candidates if c.request_id == decision.request_id
                )
                action = actions.get(selected.request_id)
                if action is None or action.request_id != selected.request_id:
                    raise ValueError("selected request has no matching action binding")
                if action.session_id in self._sessions:
                    raise ValueError("session already used")
                evidence = next(
                    r
                    for r in decision.candidates
                    if r["request"]["request_id"] == selected.request_id
                )
                self.ledger.reserve(
                    Reservation(
                        action.session_id,
                        selected.request_id,
                        target.pool_id,
                        evidence["target_required_blocks"],
                        selected.target_decode_cost,
                        0,
                        "migration",
                    ),
                    target,
                )
                self._sessions[action.session_id] = (
                    selected.request_id,
                    target.pool_id,
                )
                self._session_source[action.session_id] = source.pool_id
                self._owners.setdefault(selected.request_id, source.pool_id)
                record.update(
                    {
                        "session_id": action.session_id,
                        "execution": "RESERVED",
                        "reservation_after": self.ledger.live_snapshot(),
                    }
                )
        # Audit errors prevent START and retain the commitment for recovery.
        self.audit(record)
        if action is None:
            return decision
        try:
            action.start()
        except BaseException:
            self.audit(
                {
                    "kind": "pressure_start_uncertain",
                    "session_id": action.session_id,
                    "request_id": selected.request_id,
                }
            )
            raise
        with self._lock:
            self.policy.note_started(now)
        self.audit(
            {
                "kind": "pressure_start_published",
                "unix_s": now,
                "session_id": action.session_id,
                "request_id": selected.request_id,
            }
        )
        return decision

    def tick_multi(
        self,
        now: float,
        readings: dict[str, PoolReading],
        tp_of: dict[str, int],
        candidates: dict[tuple[str, str], list[Candidate]],
        actions: dict[str, BoundAction] | None = None,
    ):
        """Evaluate every (TP1 source, TP-k target) pair; execute at most one.

        ``candidates[(s, t)]`` are the requests owned by s, with target-side
        costs for t. Each source's trigger state advances once per tick: pairs
        are evaluated on copies and the chosen target's outcome is kept.
        Returns the executed (source, target, decision) or None.
        """
        if self.scheduler == "unified":
            return self._tick_unified(now, readings, tp_of, candidates, actions)
        chosen = action = None
        with self._lock:
            for rows in candidates.values():
                for c in rows:
                    owner = self._owners.get(c.request_id)
                    if owner is not None and owner != c.owner:
                        raise ValueError("inventory contradicts confirmed ownership")
            before = self.ledger.live_snapshot()
            # A quiescing source may still migrate out; targets must be ACTIVE.
            sources = sorted(
                i
                for i in readings
                if tp_of[i] == 1 and self.lifecycle(i) in {"ACTIVE", "QUIESCING"}
            )
            targets = sorted(
                i for i in readings if tp_of[i] > 1 and self.lifecycle(i) == "ACTIVE"
            )
            session_active = any(
                r.kind == "migration" and r.state == "ACTIVE"
                for r in self.ledger.live_rows()
            )
            evaluations = []
            for s in sources if targets else ():
                policy = self.policy_for(s)
                policy.observe_relief(now, readings[s])
                if self.routing_strategy == "pressure":
                    policy.relief.acknowledge(now)
                base = policy.source_state()
                pairs = []
                for t in targets:
                    policy.restore_source_state(base)
                    decision = policy.decide(
                        now, readings[s], readings[t], candidates.get((s, t), [])
                    )
                    headroom = multi.target_headroom(readings[t], self.ledger)
                    pairs.append(
                        dict(
                            target=t,
                            decision=decision,
                            headroom=headroom,
                            state=policy.source_state(),
                            key=multi.pair_key(decision, headroom, s, t)
                            if decision.request_id is not None
                            else None,
                        )
                    )
                executable = [p for p in pairs if p["key"] is not None]
                best = (
                    min(executable, key=lambda p: p["key"])
                    if executable
                    else min(
                        pairs,
                        key=lambda p: (
                            -p["headroom"] if p["headroom"] is not None else 0,
                            p["target"],
                        ),
                    )
                )
                policy.restore_source_state(best["state"])
                evaluations.append(dict(source=s, best=best, pairs=pairs))
                if best["key"] is not None and (
                    chosen is None or best["key"] < chosen["best"]["key"]
                ):
                    chosen = evaluations[-1]
            record = {
                "kind": "pressure_decision_multi",
                "unix_s": now,
                "readings": {
                    i: asdict(self.ledger.routing_reading(r))
                    for i, r in readings.items()
                },
                "evaluations": [
                    dict(
                        source=e["source"],
                        best_target=e["best"]["target"],
                        pairs=[
                            dict(
                                target=p["target"],
                                headroom=p["headroom"],
                                key=list(p["key"]) if p["key"] else None,
                                decision=asdict(p["decision"]),
                            )
                            for p in e["pairs"]
                        ],
                    )
                    for e in evaluations
                ],
                "chosen": [chosen["source"], chosen["best"]["target"]]
                if chosen
                else None,
                "deferred": [
                    [e["source"], e["best"]["target"]]
                    for e in evaluations
                    if e is not chosen and e["best"]["key"] is not None
                ],
                "global_session_active": session_active,
                "reservation_scope": "live",
                "reservation_before": before,
                "execution": "ADVISORY",
            }
            if len(evaluations) == 1 and len(targets) == 1:
                # Fixed pair: keep the historical record that analysis reads.
                s, t = sources[0], targets[0]
                record.update(
                    kind="pressure_decision",
                    decision=asdict(evaluations[0]["best"]["decision"]),
                    source_reading=asdict(self.ledger.routing_reading(readings[s])),
                    target_reading=asdict(self.ledger.routing_reading(readings[t])),
                    routing_relief=self.policy_for(s).relief.snapshot(),
                    prediction_inputs=prediction_inputs(candidates.get((s, t), [])),
                )
            target_fresh = (
                chosen is not None
                and 0
                <= now - readings[chosen["best"]["target"]].sampled_at
                <= self.policy.cfg.max_age_s
            )
            if chosen is not None and not target_fresh:
                record["execution_blocked"] = "target capacity evidence stale"
            if (
                chosen is not None
                and actions is not None
                and not session_active
                and target_fresh
                and chosen["best"]["decision"].request_id not in actions
                and len(evaluations) > 1
            ):
                # Several sources: one without observed rates has no bindings.
                record["execution_blocked"] = "chosen source has no action binding"
            elif (
                chosen is not None
                and actions is not None
                and not session_active
                and target_fresh
            ):
                s, t = chosen["source"], chosen["best"]["target"]
                decision = chosen["best"]["decision"]
                selected = next(
                    c for c in candidates[(s, t)] if c.request_id == decision.request_id
                )
                action = actions.get(selected.request_id)
                if action is None or action.request_id != selected.request_id:
                    raise ValueError("selected request has no matching action binding")
                if action.session_id in self._sessions:
                    raise ValueError("session already used")
                evidence = next(
                    r
                    for r in decision.candidates
                    if r["request"]["request_id"] == selected.request_id
                )
                self.ledger.reserve(
                    Reservation(
                        action.session_id,
                        selected.request_id,
                        t,
                        evidence["target_required_blocks"],
                        selected.target_decode_cost,
                        0,
                        "migration",
                    ),
                    readings[t],
                )
                self._sessions[action.session_id] = (selected.request_id, t)
                self._session_source[action.session_id] = s
                self._owners.setdefault(selected.request_id, s)
                record.update(session_id=action.session_id, execution="RESERVED")
            record["reservation_after"] = self.ledger.live_snapshot()
        self.audit(record)
        if action is None:
            return None
        try:
            action.start()
        except BaseException:
            self.audit(
                {
                    "kind": "pressure_start_uncertain",
                    "session_id": action.session_id,
                    "request_id": action.request_id,
                }
            )
            raise
        with self._lock:
            self.policy_for(s).note_started(now)
        self.audit(
            {
                "kind": "pressure_start_published",
                "unix_s": now,
                "session_id": action.session_id,
                "request_id": action.request_id,
            }
        )
        return s, t, decision

    def _tick_unified(self, now, readings, tp_of, candidates, actions):
        """KV or pressure trigger; the least full target that takes a request.

        Pairs are the (TP1 source, TP-k target) ones passed in ``candidates``
        (warm channels); targets are tried from the least full, the next one
        only when none of its requests fits or passes (design 11-06 section 3).
        """
        action = chosen = None
        with self._lock:
            for rows in candidates.values():
                for c in rows:
                    owner = self._owners.get(c.request_id)
                    if owner is not None and owner != c.owner:
                        raise ValueError("inventory contradicts confirmed ownership")
            before = self.ledger.live_snapshot()
            sort_ids = getattr(self, "pool_sort_ids", {})
            sources = sorted(
                (
                    i
                    for i in readings
                    if tp_of[i] == 1 and self.lifecycle(i) in {"ACTIVE", "QUIESCING"}
                ),
                key=lambda i: sort_ids.get(i, i),
            )
            targets = sorted(
                i for i in readings if tp_of[i] > 1 and self.lifecycle(i) == "ACTIVE"
            )
            session_active = any(
                r.kind == "migration" and r.state == "ACTIVE"
                for r in self.ledger.live_rows()
            )
            adjusted = {i: self.ledger.routing_reading(r) for i, r in readings.items()}
            full = {i: self.unified_fullness(adjusted[i], "migrate") for i in readings}
            fresh = {
                i: 0 <= now - r.sampled_at <= self.policy.cfg.max_age_s
                for i, r in readings.items()
            }
            evaluations = []
            open_targets = [
                t for t in targets if fresh[t] and self.ledger.channel_available(t)
            ]

            def by_fullness(names):
                # Least full first (design 11-06 section 3); ties by prompt
                # blocks promised to pending prefill, then the instance order.
                return rank_migration_targets(names, full, adjusted, sort_ids)

            # KV is safety and ignores cooldowns; pressure respects them.
            kv_order = by_fullness(open_targets)
            pressure_order = by_fullness(
                [t for t in open_targets if not self.migration_state.cooling(t, now)]
            )

            def pick(s, target, trigger, entry):
                """The old evaluation's choice for one target that has room."""
                # Request selection is the old scheduler's (user 11-06): stop
                # and survival from the predictor, release before exhaustion,
                # enough relief, then joint KV block-seconds. KV maps to its
                # capacity path, pressure to its overload path.
                policy = self.policy_for(s)
                source_reading = adjusted[s]
                target_reading = adjusted[target]
                guard = self.policy.cfg.capacity_guard_blocks
                required = max(1, guard - source_reading.free_blocks)
                pool = list(candidates.get((s, target), []))

                def reject(request_id, reasons):
                    entry["rejected"].append(
                        dict(request_id=request_id, reasons=reasons, target=target)
                    )

                while pool:
                    rows, selected, basis = policy.evaluate_candidates(
                        now,
                        source_reading,
                        target_reading,
                        classify(source_reading, self.policy.cfg, now),
                        classify(target_reading, self.policy.cfg, now),
                        pool,
                        True,
                        trigger == "PRESSURE",
                        required,
                    )
                    for r in rows:
                        if r["rejections"]:
                            reject(r["request"]["request_id"], r["rejections"])
                    if selected is None:
                        return None
                    c = next(x for x in pool if x.request_id == selected.request_id)
                    evidence = next(
                        r for r in rows if r["request"]["request_id"] == c.request_id
                    )
                    blocks = evidence["target_required_blocks"]
                    # The target's free blocks already promised to prompts
                    # waiting for prefill are not room: count them, and keep
                    # the admission guard, as routing does (GPU 11-06: two KV
                    # moves into B overcommitted it and B preempted 10 times).
                    room = unified.prompt_room(
                        target_reading, self.ledger.pending(target)[0]
                    )
                    if room is None or room < blocks:
                        reject(
                            c.request_id, ["target room promised to pending prompts"]
                        )
                        pool = [x for x in pool if x.request_id != c.request_id]
                        continue
                    if trigger == "PRESSURE":
                        after = self.unified_fullness(
                            target_reading,
                            "migrate",
                            extra_requests=1,
                            extra_blocks=blocks,
                        )
                        if after["value"] >= unified.TARGET_AFTER_MAX:
                            reject(
                                c.request_id,
                                ["target would not stay below hysteresis"],
                            )
                            pool = [x for x in pool if x.request_id != c.request_id]
                            continue
                    return c, blocks, basis
                return None

            for s in sources:
                capacity = classify(adjusted[s], self.policy.cfg, now).capacity
                exhausting = fresh[s] and (capacity or 0) >= 1
                trigger, reason = unified.migration_trigger(
                    full[s],
                    exhausting,
                    full[pressure_order[0]] if pressure_order else None,
                    self.migration_state,
                    s,
                    now,
                )
                if trigger == "KV" and not kv_order:
                    trigger, reason = None, "no fresh target with a free channel"
                order = kv_order if trigger == "KV" else pressure_order
                entry = dict(
                    source=s,
                    fullness=full[s],
                    exhausting=exhausting,
                    trigger=trigger,
                    reason=reason,
                    target=None,
                    request_id=None,
                    rejected=[],
                    targets=[],
                )
                evaluations.append(entry)
                if trigger is None or not fresh[s]:
                    continue
                # Least full target first; the next one only when no request
                # fits or passes there. No fit anywhere: no move this tick.
                for target in order:
                    found = pick(s, target, trigger, entry)
                    entry["targets"].append(
                        dict(
                            target=target,
                            fullness=full[target]["value"],
                            candidates=len(candidates.get((s, target), [])),
                            request_id=found[0].request_id if found else None,
                        )
                    )
                    if found is None:
                        continue
                    c, blocks, basis = found
                    entry.update(
                        target=target,
                        request_id=c.request_id,
                        required_blocks=blocks,
                        selection_policy=basis,
                    )
                    key = (trigger != "KV", -full[s]["value"], sort_ids.get(s, s))
                    if chosen is None or key < chosen["key"]:
                        chosen = dict(
                            key=key,
                            source=s,
                            target=target,
                            candidate=c,
                            required=blocks,
                            target_reading=readings[target],
                            entry=entry,
                        )
                    break
            record = {
                "kind": "pressure_decision_unified",
                "unix_s": now,
                "readings": {i: asdict(r) for i, r in adjusted.items()},
                "smoothed_itl": self.itl.snapshot(),
                "evaluations": evaluations,
                "chosen": [
                    chosen["source"],
                    chosen["target"],
                    chosen["candidate"].request_id,
                ]
                if chosen
                else None,
                "global_session_active": session_active,
                "reservation_scope": "live",
                "reservation_before": before,
                "execution": "ADVISORY",
            }
            if chosen is not None and actions is not None and not session_active:
                selected = chosen["candidate"]
                # Several targets: one binding per (request, target), so START
                # runs on the session of the chosen target.
                action = actions.get((selected.request_id, chosen["target"]))
                if action is None:
                    action = actions.get(selected.request_id)
                if action is None or action.request_id != selected.request_id:
                    record["execution_blocked"] = "chosen request has no action binding"
                    action = None
                elif action.session_id in self._sessions:
                    raise ValueError("session already used")
                else:
                    s, t = chosen["source"], chosen["target"]
                    self.ledger.reserve(
                        Reservation(
                            action.session_id,
                            selected.request_id,
                            t,
                            chosen["required"],
                            selected.target_decode_cost,
                            0,
                            "migration",
                        ),
                        chosen["target_reading"],
                    )
                    self._sessions[action.session_id] = (selected.request_id, t)
                    self._session_source[action.session_id] = s
                    self._owners.setdefault(selected.request_id, s)
                    record.update(session_id=action.session_id, execution="RESERVED")
            record["reservation_after"] = self.ledger.live_snapshot()
        self.audit(record)
        if action is None:
            return None
        try:
            action.start()
        except BaseException as error:
            if getattr(error, "migration_not_started", False):
                # Refused before anything was published: undo the reservation,
                # cool the request down and keep serving (user 11-06).
                with self._lock:
                    self.ledger.discard_unstarted(action.session_id)
                    self._sessions.pop(action.session_id, None)
                    self._session_source.pop(action.session_id, None)
                    policy = self.policy_for(s)
                    policy._retry_after[action.request_id] = now + policy.cfg.cooldown_s
                self.audit(
                    {
                        "kind": "pressure_start_refused",
                        "unix_s": now,
                        "session_id": action.session_id,
                        "request_id": action.request_id,
                        "reason": str(error),
                    }
                )
                return None
            self.audit(
                {
                    "kind": "pressure_start_uncertain",
                    "session_id": action.session_id,
                    "request_id": action.request_id,
                }
            )
            raise
        with self._lock:
            self.policy_for(s).note_started(now)
            self.migration_state.started(s, t, now)
        self.audit(
            {
                "kind": "pressure_start_published",
                "unix_s": now,
                "session_id": action.session_id,
                "request_id": action.request_id,
                "scheduler": "unified",
                "trigger": chosen["entry"]["trigger"],
            }
        )
        return s, t, chosen["entry"]

    def assist_start(
        self,
        now: float,
        source: str,
        target: str,
        candidate: Candidate,
        action: BoundAction | None,
        target_reading: PoolReading,
        required_blocks: int,
        before_quiesce: bool = False,
    ) -> str | None:
        """S5a: start one conversion-assist migration chosen outside the policy.

        The request is picked by the conversion's drain plan, not by pressure;
        everything else is the policy path's: one global session, a fresh
        target reading, an exact action binding and an atomic ledger
        reservation (sized by the caller's length rule). Returns None
        once started, otherwise the refusal reason; nothing is retried here.
        """
        with self._lock:
            # Leaving a same-TP group: it keeps serving until its long
            # requests are out, then quiesces (user 11-05).
            allowed = ("QUIESCING", "ACTIVE") if before_quiesce else ("QUIESCING",)
            if self.lifecycle(source) not in allowed:
                return "source is not quiescing"
            if self.lifecycle(target) != "ACTIVE":
                return "target is not active"
            if any(
                r.kind == "migration" and r.state == "ACTIVE"
                for r in self.ledger.live_rows()
            ):
                return "a migration session is active"
            if not 0 <= now - target_reading.sampled_at <= self.policy.cfg.max_age_s:
                return "target capacity evidence stale"
            if action is None or action.request_id != candidate.request_id:
                return "no start binding for the request"
            if action.session_id in self._sessions:
                return "session already used"
            owner = self._owners.get(candidate.request_id)
            if owner is not None and owner != source:
                return "request is not owned by the source"
            try:
                self.ledger.reserve(
                    Reservation(
                        action.session_id,
                        candidate.request_id,
                        target,
                        required_blocks,
                        candidate.target_decode_cost or 0.0,
                        0,
                        "migration",
                    ),
                    target_reading,
                )
            except ValueError as error:
                return str(error)
            self._sessions[action.session_id] = (candidate.request_id, target)
            self._session_source[action.session_id] = source
            self._owners.setdefault(candidate.request_id, source)
            self._assist_sessions.add(action.session_id)
            snapshot = self.ledger.snapshot()
        self.audit(
            {
                "kind": "pressure_conversion_assist",
                "unix_s": now,
                "session_id": action.session_id,
                "request_id": candidate.request_id,
                "source": source,
                "target": target,
                "required_blocks": required_blocks,
                "execution": "RESERVED",
                "reservation_after": snapshot,
            }
        )
        try:
            action.start()
        except BaseException:
            self.audit(
                {
                    "kind": "pressure_start_uncertain",
                    "session_id": action.session_id,
                    "request_id": action.request_id,
                }
            )
            raise
        with self._lock:
            self.policy_for(source).note_started(now)
        self.audit(
            {
                "kind": "pressure_start_published",
                "unix_s": now,
                "session_id": action.session_id,
                "request_id": action.request_id,
                "reason": "conversion assist",
            }
        )
        return None

    def finish(
        self,
        session_id: str,
        request_id: str,
        outcome: str,
        *,
        ownership_confirmed: bool = False,
        source_release_confirmed: bool = False,
        cleanup_confirmed: bool = False,
        target_id: str | None = None,
    ) -> None:
        """Settle after existing lifecycle receipts; never infer safety from PMF.

        ``target_id`` names the instance the receipt came from; a receipt from
        another target generation is rejected.
        """
        with self._lock:
            self._bound_target(session_id, target_id)
            actual, target = self._sessions[session_id]
            if actual != request_id:
                raise ValueError("completion request/session mismatch")
            evidence = (
                ownership_confirmed and source_release_confirmed
                if outcome == "COMMITTED"
                else cleanup_confirmed
            )
            self.ledger.settle(session_id, outcome, evidence=evidence)
            if outcome == "COMMITTED":
                self._owners[request_id] = target
                for row in self.ledger.live_rows():
                    if (
                        row.request_id == request_id
                        and row.kind == "admission"
                        and row.pool_id != target
                        and row.state in {"ACTIVE", "COMMITTED", "ACCOUNTED"}
                    ):
                        self.ledger.retire_admission(
                            row.key, "MOVED", evidence=evidence
                        )
            record = {
                "kind": "pressure_settlement",
                "session_id": session_id,
                "request_id": request_id,
                "outcome": outcome,
                "ownership_confirmed": ownership_confirmed,
                "source_release_confirmed": source_release_confirmed,
                "cleanup_confirmed": cleanup_confirmed,
                "reservation_after": self.ledger.snapshot(),
            }
        self.audit(record)


def prediction_inputs(candidates: list[Candidate]) -> list[dict]:
    return [
        dict(
            request_id=c.request_id,
            status=c.prediction_status,
            generated=c.generated,
            max_output_tokens=c.max_output_tokens,
            remaining_mean=c.remaining.mean if c.remaining is not None else None,
            prediction_age_s=c.prediction_age_s,
            prediction_lag=c.prediction_lag,
            point_method=(
                "probability-weighted integer-bin means; survival/lag/stop aligned"
            ),
        )
        for c in candidates
    ]


def json_reference(row: dict) -> str:
    """Compact original journal identity, not repeated raw feature payloads."""
    return ":".join(
        str(row.get(k, ""))
        for k in ("checkpoint_sha256", "request_id", "captured_unix_ns", "feature_id")
    )
