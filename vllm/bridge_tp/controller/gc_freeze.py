# SPDX-License-Identifier: Apache-2.0
"""Keep the controller's full garbage collections short (BridgeTP, 11-13).

guard40 R4/split5: every admission that waited >0.5 s overlapped a Python
generation-2 collection in the controller process (0.4 s at the start,
0.7-1.1 s after 1700 s as the per-token records and audit rows piled up);
the collection stops the whole process, so arrivals waited in admission.

``gc.freeze()`` moves every object the collector currently tracks into a
permanent generation that later collections skip, so a full collection only
scans what was allocated since the last freeze. The controller freezes once
after setup and then periodically; a cheap young collection first keeps
short-lived cycles from being frozen. Frozen cyclic garbage is never
reclaimed (non-cyclic objects are still freed by reference counting): a
bounded memory cost for a run of tens of minutes.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any, Callable

ENV = "BRIDGETP_CONTROLLER_GC_FREEZE_S"
DEFAULT_INTERVAL_S = 30.0


class GcFreezer:
    def __init__(
        self,
        interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        record: Callable[[dict[str, Any]], None] | None = None,
        collector: Any = gc,
    ) -> None:
        if interval_s < 0:
            raise ValueError("gc freeze interval cannot be negative")
        self.interval_s = interval_s
        self.clock = clock
        self.record = record
        self.gc = collector
        self.last = None
        self.freezes = 0

    @classmethod
    def from_env(cls, **kwargs: Any) -> GcFreezer:
        return cls(float(os.environ.get(ENV, str(DEFAULT_INTERVAL_S))), **kwargs)

    @property
    def enabled(self) -> bool:
        return self.interval_s > 0

    def _freeze(self, reason: str, generation: int) -> None:
        started = time.perf_counter()
        collected = self.gc.collect(generation)
        frozen_before = self.gc.get_freeze_count()
        self.gc.freeze()
        self.freezes += 1
        self.last = self.clock()
        if self.record is not None:
            self.record(
                dict(
                    kind="controller_gc_freeze",
                    reason=reason,
                    collected=collected,
                    newly_frozen=self.gc.get_freeze_count() - frozen_before,
                    frozen_total=self.gc.get_freeze_count(),
                    duration_s=time.perf_counter() - started,
                    unix_s=time.time(),
                )
            )

    def start(self) -> None:
        """After setup, before arrivals: one full collection, then freeze."""
        if self.enabled:
            self._freeze("start", 2)

    def close(self) -> None:
        """Unfreeze when the controller loop ends (in-process callers)."""
        if self.freezes:
            self.gc.unfreeze()
            self.freezes = 0
            self.last = None

    def tick(self) -> None:
        """Freeze what accumulated since the last freeze, once per interval."""
        if not self.enabled or self.last is None:
            return
        if self.clock() - self.last >= self.interval_s:
            self._freeze("periodic", 1)
