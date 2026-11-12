# SPDX-License-Identifier: Apache-2.0
"""Keep the migrating source request out of vLLM's preemption (BridgeTP, 11-12).

directkv run 3: C-g4's KV filled while it migrated a long request; vLLM
preempted that very request (recompute), and the source stream hook, which
assumes a decoding request (exactly one pending token), killed the engine.
The request being moved out is the one that relieves the pressure, so
preempting it only wastes the recompute and stalls the migration.

The guard names the request from the source's live session: the armed
runtime control's request prefix, until the takeover state is terminal.
It is read only when the scheduler has to preempt (rare), straight from the
files of the current session (``source_active`` rotates per session), and
any error means "protect nothing" rather than a scheduling failure.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

TERMINAL_STATES = frozenset({"COMMITTED", "CANCELLED"})


class MigrationSourceGuard:
    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.run_dir = Path(run_dir)
        self.protected_events = 0
        self._warned = False

    @classmethod
    def from_env(cls) -> MigrationSourceGuard | None:
        if os.getenv("BRIDGETP_STREAM_ENABLED", "0") != "1":
            return None
        if os.getenv("BRIDGETP_PROTECT_MIGRATING_SOURCE", "1") == "0":
            return None
        run_dir = os.getenv("BRIDGETP_STREAM_RUN_DIR", "").strip()
        return cls(run_dir) if run_dir else None

    def protected_request_id(self, request_ids: Iterable[str]) -> str | None:
        """The running request of the live migration, if any."""
        try:
            from vllm.bridge_tp.controller.anchor_selector import (
                select_source_request_id,
            )
            from vllm.bridge_tp.runtime_control import RuntimeControl

            control = RuntimeControl.load(self.run_dir)
            if (
                control is None
                or not control.armed
                or not control.source_request_id_prefix
            ):
                return None
            state_path = self.run_dir / "takeover_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("state") in TERMINAL_STATES:
                    return None
            return select_source_request_id(
                list(request_ids), control.source_request_id_prefix
            )
        except Exception:  # never fail scheduling over a diagnostic guard
            if not self._warned:
                self._warned = True
                logger.exception("BridgeTP preemption guard disabled for this read")
            return None


def pick_victim(running: list, protected_id: str | None, key=None):
    """vLLM's preemption choice without the protected request.

    FCFS (``key`` None) takes the last running request, PRIORITY the max by
    ``key``; None when only the protected request is left.
    """
    candidates = [r for r in running if r.request_id != protected_id]
    if not candidates:
        return None
    if key is None:
        return candidates[-1]
    return max(candidates, key=key)
