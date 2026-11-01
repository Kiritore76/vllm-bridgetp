# SPDX-License-Identifier: Apache-2.0
"""Start gate of a pre-started replacement engine (BridgeTP conversion).

A TP conversion starts the new engine while the old one still drains on the
same GPUs: process start, imports and worker spawn (about two thirds of a
rebuild, S5b GPU 11-01) run on the CPU meanwhile. The worker then waits here,
before it touches CUDA, until the conversion opens the gate once the old
engine has released the GPUs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

ENV = "BRIDGETP_START_GATE"
TIMEOUT_ENV = "BRIDGETP_START_GATE_TIMEOUT_S"


def wait_for_start_gate(poll_s: float = 0.05) -> float | None:
    """Block until the gate file exists; seconds waited, or None (no gate)."""
    path = os.environ.get(ENV)
    if not path:
        return None
    timeout = float(os.environ.get(TIMEOUT_ENV, "1800"))
    started = time.monotonic()
    gate = Path(path)
    while not gate.exists():
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"start gate {gate} not opened within {timeout}s")
        time.sleep(poll_s)
    return time.monotonic() - started
