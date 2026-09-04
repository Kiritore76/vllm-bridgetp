# SPDX-License-Identifier: Apache-2.0
"""BridgeTP Phase 9: online fast/slow controller.

Layering, deliberately strict:

    policy / predictor / rate_controller   pure functions of telemetry
    state_machine                          legality and idempotence gates
    action_adapter                         the ONLY writer of migration state
    response_proxy                         the client-visible token stream

No module in this package touches a KV tensor. All state movement is performed
by the Phase 6/7/8 mechanism, which already carries its own correctness
evidence; Phase 9 only decides when to invoke it.
"""

from .audit import AuditLog, read_audit
from .events import (
    Action,
    Decision,
    MigrationState,
    PoolTelemetry,
    SourceRequestView,
    TriggerPath,
)
from .policy import (
    FastPolicy,
    InterferenceModel,
    PolicyConfig,
    RiskTracker,
    TpotModel,
)
from .predictor import SurvivalTable
from .rate_controller import RateConfig, RateController
from .response_proxy import ProxyMode, ResponseProxy, StreamViolation
from .sampling_contract import (
    STRICT_GREEDY_SAMPLING_CONTRACT,
    freeze_strict_greedy_sampling,
    strict_greedy_sampling_errors,
)
from .state_machine import IllegalTransition, MigrationStateMachine

__all__ = [
    "Action",
    "AuditLog",
    "Decision",
    "FastPolicy",
    "IllegalTransition",
    "InterferenceModel",
    "MigrationState",
    "MigrationStateMachine",
    "PolicyConfig",
    "PoolTelemetry",
    "ProxyMode",
    "RateConfig",
    "RateController",
    "ResponseProxy",
    "RiskTracker",
    "SourceRequestView",
    "STRICT_GREEDY_SAMPLING_CONTRACT",
    "StreamViolation",
    "SurvivalTable",
    "TpotModel",
    "TriggerPath",
    "freeze_strict_greedy_sampling",
    "read_audit",
    "strict_greedy_sampling_errors",
]
