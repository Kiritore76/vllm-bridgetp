# SPDX-License-Identifier: Apache-2.0
"""Idempotent migration state machine with an append-only audit trail.

Two properties are enforced here rather than left to the caller, because both
are correctness gates that Phase 7 already relies on:

1. TAKEOVER is reachable only from HANDOFF. The controller can never commit a
   migration whose four ranks have not reported exact readback, because
   entering HANDOFF requires that evidence.
2. Transitions are idempotent and keyed by migration ID. Replaying the same
   transition is a no-op that returns the existing record rather than an error,
   so a controller retry after a timeout cannot double-commit.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from .events import (
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    MigrationState,
    TriggerPath,
)


class IllegalTransition(RuntimeError):
    pass


@dataclass
class MigrationRecord:
    migration_id: str
    request_id: str
    state: MigrationState = MigrationState.LOCAL
    history: list[tuple[float, MigrationState, str]] = field(default_factory=list)

    # evidence gates
    ranks_ready: set[int] = field(default_factory=set)
    expected_ranks: int = 4

    # boundaries
    trigger_output_tokens: int | None = None
    cutover_output_tokens: int | None = None
    trigger_path: TriggerPath | None = None

    # timings, unix seconds
    t_decision: float | None = None
    t_shadow_start: float | None = None
    t_cutover: float | None = None
    t_target_ready: float | None = None
    t_committed: float | None = None

    @property
    def all_ranks_ready(self) -> bool:
        return len(self.ranks_ready) >= self.expected_ranks

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES


class MigrationStateMachine:
    """Owns migration records. Thread-safe; one lock for the whole table."""

    def __init__(self, audit_sink: Callable[[dict], None] | None = None) -> None:
        self._records: dict[str, MigrationRecord] = {}
        self._lock = threading.RLock()
        self._audit = audit_sink or (lambda _record: None)

    # ---- lifecycle ----------------------------------------------------
    def create(self, migration_id: str, request_id: str) -> MigrationRecord:
        with self._lock:
            existing = self._records.get(migration_id)
            if existing is not None:
                if existing.request_id != request_id:
                    raise IllegalTransition(
                        f"migration {migration_id} already bound to "
                        f"{existing.request_id}"
                    )
                return existing
            record = MigrationRecord(migration_id, request_id)
            self._records[migration_id] = record
            return record

    def get(self, migration_id: str) -> MigrationRecord | None:
        with self._lock:
            return self._records.get(migration_id)

    def active(self) -> list[MigrationRecord]:
        with self._lock:
            return [r for r in self._records.values() if not r.is_terminal]

    def count_active(self) -> int:
        return len(self.active())

    # ---- evidence -----------------------------------------------------
    def mark_rank_ready(self, migration_id: str, rank: int) -> MigrationRecord:
        with self._lock:
            record = self._require(migration_id)
            record.ranks_ready.add(int(rank))
            return record

    # ---- transitions --------------------------------------------------
    def transition(
        self,
        migration_id: str,
        to: MigrationState,
        now_unix_s: float,
        reason: str = "",
    ) -> MigrationRecord:
        with self._lock:
            record = self._require(migration_id)
            if record.state is to:
                return record  # idempotent replay

            allowed = LEGAL_TRANSITIONS.get(record.state, frozenset())
            if to not in allowed:
                raise IllegalTransition(
                    f"migration {migration_id}: {record.state.value} -> {to.value} "
                    f"is not legal (allowed: "
                    f"{sorted(s.value for s in allowed) or 'none'})"
                )

            if to is MigrationState.TAKEOVER and not record.all_ranks_ready:
                raise IllegalTransition(
                    f"migration {migration_id}: refusing commit with "
                    f"{len(record.ranks_ready)}/{record.expected_ranks} ranks ready"
                )

            previous = record.state
            record.state = to
            record.history.append((now_unix_s, to, reason))

            if to is MigrationState.SHADOW:
                record.t_shadow_start = now_unix_s
            elif to is MigrationState.HANDOFF:
                record.t_cutover = now_unix_s
            elif to is MigrationState.TAKEOVER:
                record.t_committed = now_unix_s

            self._audit(
                {
                    "kind": "transition",
                    "unix_s": now_unix_s,
                    "migration_id": migration_id,
                    "request_id": record.request_id,
                    "from": previous.value,
                    "to": to.value,
                    "reason": reason,
                    "ranks_ready": sorted(record.ranks_ready),
                    "trigger_path": (
                        record.trigger_path.value
                        if record.trigger_path is not None
                        else None
                    ),
                }
            )
            return record

    def _require(self, migration_id: str) -> MigrationRecord:
        record = self._records.get(migration_id)
        if record is None:
            raise KeyError(f"unknown migration {migration_id}")
        return record
