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

from .bounded_length import from_advisory
from .pressure_policy import (
    Candidate,
    PoolReading,
    PressureConfig,
    PressurePolicy,
    Reservation,
    ReservationLedger,
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
    def __init__(self, cfg: PressureConfig, audit: Callable[[dict], None]) -> None:
        self.ledger = ReservationLedger()
        self.policy = PressurePolicy(cfg, self.ledger)
        self.audit = audit
        self._lock = threading.RLock()
        self._sessions: dict[str, tuple[str, str]] = {}
        self._owners: dict[str, str] = {}

    def admission(self, *args, **kwargs):
        """Route and track identity atomically, without reserving ingress KV."""
        with self._lock:
            before = self.ledger.snapshot()
            result = admit(*args, cfg=self.policy.cfg, ledger=self.ledger, **kwargs)
            self._owners[result.request_id] = result.pool_id
            self.audit(
                {
                    "kind": "pressure_admission",
                    "admission": asdict(result),
                    "pool_readings": [
                        asdict(pool)
                        for pool in (args[3] if len(args) > 3 else kwargs["pools"])
                    ],
                    "routing_config": asdict(self.policy.cfg),
                    "decision_unix_s": kwargs.get(
                        "now", args[4] if len(args) > 4 else None
                    ),
                    "config_sha256": self.policy.cfg.sha256,
                    "reservation_before": before,
                    "reservation_after": self.ledger.snapshot(),
                }
            )
            return result

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
        with self._lock:
            for c in candidates:
                owner = self._owners.get(c.request_id)
                if owner is not None and owner != c.owner:
                    raise ValueError("inventory contradicts confirmed ownership")
            before = self.ledger.snapshot()
            decision = self.policy.decide(now, source, target, candidates)
            record = {
                "kind": "pressure_decision",
                "unix_s": now,
                "decision": asdict(decision),
                "source_reading": asdict(self.ledger.routing_reading(source)),
                "target_reading": asdict(self.ledger.routing_reading(target)),
                "reservation_before": before,
                "execution": "ADVISORY",
            }
            if decision.request_id is None or actions is None:
                record["reservation_after"] = self.ledger.snapshot()
                self.audit(record)
                return decision
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
            self._sessions[action.session_id] = (selected.request_id, target.pool_id)
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

    def finish(
        self,
        session_id: str,
        request_id: str,
        outcome: str,
        *,
        ownership_confirmed: bool = False,
        source_release_confirmed: bool = False,
        cleanup_confirmed: bool = False,
    ) -> None:
        """Settle after existing lifecycle receipts; never infer safety from PMF."""
        with self._lock:
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
            self.audit(
                {
                    "kind": "pressure_settlement",
                    "session_id": session_id,
                    "request_id": request_id,
                    "outcome": outcome,
                    "ownership_confirmed": ownership_confirmed,
                    "source_release_confirmed": source_release_confirmed,
                    "cleanup_confirmed": cleanup_confirmed,
                    "reservation_after": self.ledger.snapshot(),
                }
            )


def json_reference(row: dict) -> str:
    """Compact original journal identity, not repeated raw feature payloads."""
    return ":".join(
        str(row.get(k, ""))
        for k in ("checkpoint_sha256", "request_id", "captured_unix_ns", "feature_id")
    )
