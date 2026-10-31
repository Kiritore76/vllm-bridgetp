# SPDX-License-Identifier: Apache-2.0
"""Serialized pressure-policy harness and explicitly bound action callbacks.

The callbacks integrate the existing M2/M3/M4 lifecycle; they must not launch a
fresh source request, alter its stop contract, or bypass four-rank readiness.
The experimental real pilot ingress calls this coordinator. Replay is CPU-only.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

from . import pressure_multi as multi
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
    ) -> None:
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
        before = self.ledger.snapshot()
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
            "reservation_before": before,
            "reservation_after": self.ledger.snapshot(),
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
        before = self.ledger.snapshot()
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
            "reservation_before": before,
            "reservation_after": self.ledger.snapshot(),
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
            before = self.ledger.snapshot()
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
                "reservation_before": before,
                "execution": "ADVISORY",
                "routing_relief": self.policy.relief.snapshot(),
                "prediction_inputs": prediction_inputs(candidates),
            }
            if decision.request_id is None or actions is None:
                record["reservation_after"] = self.ledger.snapshot()
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
                        "reservation_after": self.ledger.snapshot(),
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
        chosen = action = None
        with self._lock:
            for rows in candidates.values():
                for c in rows:
                    owner = self._owners.get(c.request_id)
                    if owner is not None and owner != c.owner:
                        raise ValueError("inventory contradicts confirmed ownership")
            before = self.ledger.snapshot()
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
                for r in self.ledger.entries.values()
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
            record["reservation_after"] = self.ledger.snapshot()
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
                for row in self.ledger.entries.values():
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
