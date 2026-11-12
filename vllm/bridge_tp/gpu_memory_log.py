# SPDX-License-Identifier: Apache-2.0
"""Per-rank CUDA memory samples (BridgeTP, 11-12).

guard40 standby hung when one TP2 rank reached 24,076/24,564 MiB; nvidia-smi
only showed totals per GPU. Each TP worker that runs the BridgeTP connector
appends one JSON line per sample: PyTorch allocated/reserved, their peaks
since the previous periodic sample, and the device's free/total memory.
Periodic samples come from the pre-forward hook; migration milestones add
event samples (which do not reset the peaks).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable


class GpuMemoryLog:
    def __init__(
        self,
        path: Path,
        *,
        interval_s: float,
        device: Any,
        pool: str | None,
        tp_rank: int | None,
        cuda: Any = None,
        clock: Callable[[], float] = time.time,
    ):
        if interval_s <= 0:
            raise ValueError("memory log interval must be positive")
        if cuda is None:
            import torch

            cuda = torch.cuda
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_s = interval_s
        self.device = device
        self.pool, self.tp_rank = pool, tp_rank
        self.cuda, self.clock = cuda, clock
        self.last = -float("inf")
        self.samples = 0
        self._lock = threading.Lock()

    def maybe_sample(self) -> bool:
        """Periodic sample if the interval elapsed; resets the peaks."""
        now = self.clock()
        if now - self.last < self.interval_s:
            return False
        self.sample("periodic", reset_peaks=True, now=now)
        return True

    def sample(
        self,
        event: str,
        *,
        reset_peaks: bool = False,
        now: float | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        with self._lock:
            now = self.clock() if now is None else now
            free, total = self.cuda.mem_get_info(self.device)
            row = {
                "unix_s": now,
                "event": event,
                "pool": self.pool,
                "tp_rank": self.tp_rank,
                "device": str(self.device),
                "allocated_bytes": int(self.cuda.memory_allocated(self.device)),
                "reserved_bytes": int(self.cuda.memory_reserved(self.device)),
                "max_allocated_bytes": int(
                    self.cuda.max_memory_allocated(self.device)
                ),
                "max_reserved_bytes": int(self.cuda.max_memory_reserved(self.device)),
                "device_free_bytes": int(free),
                "device_total_bytes": int(total),
                **extra,
            }
            if reset_peaks:
                self.cuda.reset_peak_memory_stats(self.device)
                self.last = now
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            self.samples += 1
            return row
