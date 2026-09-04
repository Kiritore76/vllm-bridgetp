# SPDX-License-Identifier: Apache-2.0
"""BridgeTP Phase 9 event and record schemas.

Every quantity in this package uses explicit units:

    * time      -> seconds (float)
    * size      -> bytes (int)
    * length    -> tokens (int)
    * rate      -> bytes per second (float)

Rates are converted to GiB/s only at the boundary with Phase 6/8 code, which
expects ``BRIDGETP_STREAM_RATE_GIB_S``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

GIB = 1024.0**3


class MigrationState(str, Enum):
    """Paper Section 3.3 states, plus terminal outcomes.

    The mapping to the Phase 7/8 on-disk ``takeover_state.json`` values is:

        LOCAL     -> (no takeover state file yet)
        SHADOW    -> "PREPARING"
        HANDOFF   -> "PREPARING" with all four TARGET_READY receipts present
        TAKEOVER  -> "COMMITTING" then "COMMITTED"
        ROLLED_BACK -> "ROLLED_BACK"
        CANCELLED   -> "CANCELLED"
    """

    LOCAL = "LOCAL"
    SHADOW = "SHADOW"
    HANDOFF = "HANDOFF"
    TAKEOVER = "TAKEOVER"

    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    CANCELLED = "CANCELLED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"
    COMPLETED_ON_TP1 = "COMPLETED_ON_TP1"


TERMINAL_STATES = frozenset(
    {
        MigrationState.NOT_ELIGIBLE,
        MigrationState.CANCELLED,
        MigrationState.ROLLED_BACK,
        MigrationState.FAILED,
        MigrationState.COMPLETED_ON_TP1,
        MigrationState.TAKEOVER,
    }
)

# Legal transitions. Deliberately strict: the controller must never be able to
# reach TAKEOVER without passing through HANDOFF, because HANDOFF is where the
# four-rank exact-readback gate lives.
LEGAL_TRANSITIONS: dict[MigrationState, frozenset[MigrationState]] = {
    MigrationState.LOCAL: frozenset(
        {
            MigrationState.SHADOW,
            MigrationState.NOT_ELIGIBLE,
            MigrationState.COMPLETED_ON_TP1,
        }
    ),
    MigrationState.SHADOW: frozenset(
        {
            MigrationState.HANDOFF,
            MigrationState.CANCELLED,
            MigrationState.FAILED,
            MigrationState.COMPLETED_ON_TP1,
        }
    ),
    MigrationState.HANDOFF: frozenset(
        {
            MigrationState.TAKEOVER,
            MigrationState.ROLLED_BACK,
            MigrationState.FAILED,
        }
    ),
    MigrationState.TAKEOVER: frozenset(),
    MigrationState.NOT_ELIGIBLE: frozenset(),
    MigrationState.CANCELLED: frozenset(),
    MigrationState.ROLLED_BACK: frozenset(),
    MigrationState.FAILED: frozenset(),
    MigrationState.COMPLETED_ON_TP1: frozenset(),
}


class Action(str, Enum):
    """What the policy asks the action adapter to do."""

    STAY = "STAY"
    START_SHADOW = "START_SHADOW"
    SET_RATE = "SET_RATE"
    CUTOVER = "CUTOVER"
    COMMIT = "COMMIT"
    ROLLBACK = "ROLLBACK"
    CANCEL = "CANCEL"


class TriggerPath(str, Enum):
    """Why a migration entered Shadow; persisted in every run artifact."""

    PERFORMANCE_OPPORTUNITY = "PERFORMANCE_OPPORTUNITY"
    POLICY_OOM_RISK = "POLICY_OOM_RISK"
    CAPACITY_PILOT = "CAPACITY_PILOT"
    DIAGNOSTIC_FIXED_BOUNDARY = "DIAGNOSTIC_FIXED_BOUNDARY"


@dataclass(frozen=True)
class SourceRequestView:
    """What the controller knows about one live TP1 request."""

    request_id: str
    prompt_tokens: int
    output_tokens: int  # tokens emitted so far
    computed_tokens: int  # tokens represented in KV
    pending_tokens: int  # sampled, KV not yet written
    arrival_unix_s: float
    last_token_unix_s: float
    group_id: str | None = None
    is_group_longest: bool = False

    @property
    def kv_tokens(self) -> int:
        return self.computed_tokens


@dataclass(frozen=True)
class PoolTelemetry:
    """Instantaneous view of one TP pool."""

    num_running: int
    num_waiting: int
    kv_usage_frac: float  # 0..1
    preemptions_total: int  # monotone counter, NOT *_created
    p99_tpot_s: float
    mean_tpot_s: float
    free_kv_blocks: int
    block_size: int
    sampled_unix_s: float = 0.0

    @property
    def free_kv_tokens(self) -> int:
        return self.free_kv_blocks * self.block_size


@dataclass(frozen=True)
class Decision:
    """One policy decision, fully auditable.

    Every field that entered the comparison is recorded, so a reviewer can
    replay why a request was or was not escalated.
    """

    request_id: str
    unix_s: float
    action: Action
    from_state: MigrationState
    to_state: MigrationState
    trigger_path: TriggerPath | None = None

    # inputs
    output_tokens: int = 0
    expected_remaining_tokens: float = 0.0
    p_oom: float = 0.0
    p_worth: float = 0.0
    theta_esc: float = 0.0
    risk_tp1: float = 0.0
    break_even_tokens: float = 0.0

    # benefit / cost, all in seconds
    benefit_s: float = 0.0
    cost_s: float = 0.0
    cost_breakdown: dict[str, float] = field(default_factory=dict)

    # rate
    rate_bytes_s: float = 0.0

    reason: str = ""
    forced: bool = False

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["action"] = self.action.value
        out["from_state"] = self.from_state.value
        out["to_state"] = self.to_state.value
        out["trigger_path"] = (
            self.trigger_path.value if self.trigger_path is not None else None
        )
        return out
