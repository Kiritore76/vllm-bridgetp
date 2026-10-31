# SPDX-License-Identifier: Apache-2.0
"""Per-source persistent receivers on one TP-k target rank (S1-d2b).

A target keeps one preconnected receiver per TP1 source. Channel generations
are unique across every channel of a run, so the ``channel_generation`` a
source writes into its session manifest names the receiver to use; only one
migration session is active globally, so a single receiver is "active" at a
time and the connector's existing single-receiver code runs against it.

``reconcile`` applies a channel plan: it opens receivers for new sources and
closes receivers whose source left the plan, but never a receiver that is not
IDLE (those are deferred to the next reconcile). This is how a retained target
rebuilds channels when a TP conversion retires or adds sources.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_SOURCE_ID = re.compile(r"tp1|[A-Z]-g[0-9]+-tp1")


@dataclass(frozen=True)
class SourceChannel:
    source: str
    base_port: int
    generation: int


def parse_plan(rows: Any, target_tp: int) -> tuple[SourceChannel, ...]:
    """Validate a plan: unique sources/generations, disjoint per-rank ports."""
    if not isinstance(rows, list):
        raise ValueError("source channel plan must be a list")
    plan = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "source",
            "base_port",
            "generation",
        }:
            raise ValueError("source channel needs source/base_port/generation")
        channel = SourceChannel(
            str(row["source"]), int(row["base_port"]), int(row["generation"])
        )
        if (
            not _SOURCE_ID.fullmatch(channel.source)
            or channel.generation < 0
            or not 1024 <= channel.base_port <= 65535 - target_tp
        ):
            raise ValueError(f"invalid source channel {row!r}")
        plan.append(channel)
    if len({c.source for c in plan}) != len(plan) or len(
        {c.generation for c in plan}
    ) != len(plan):
        raise ValueError("source channel plan repeats a source or generation")
    ports = [c.base_port + rank for c in plan for rank in range(target_tp + 1)]
    if len(set(ports)) != len(ports):
        raise ValueError("source channel port blocks overlap")
    return tuple(plan)


def _idle(receiver: Any) -> bool:
    lifecycle = getattr(receiver, "lifecycle", None)
    return lifecycle is None or (
        lifecycle.state.value == "IDLE" and lifecycle.active_session is None
    )


class SourceChannelSet:
    """Receivers of one target rank, keyed by channel generation.

    ``bind(channel)`` returns a receiver whose listener is bound;
    ``open(receiver, channel)`` completes the source handshake for that
    generation and warms it up; ``receiver.close()`` releases it. All new
    listeners are bound before any handshake, and handshakes run in source-id
    order: a source connecting early is queued, never refused, and with
    sources also preconnecting targets in id order no wait cycle can form.

    ``retire(receiver, channel)`` tears down a departed idle channel after it
    has left the set (default: ``receiver.close()``). The connector passes a
    background teardown: a synchronous communicator destroy with the source
    still alive never returned and stalled plan polling (S3b-3 GPU, 10-31).
    """

    def __init__(
        self,
        bind: Callable[[SourceChannel], Any],
        open: Callable[[Any, SourceChannel], None],
        retire: Callable[[Any, SourceChannel], None] | None = None,
    ) -> None:
        self._bind, self._open = bind, open
        self._retire = retire or (lambda receiver, channel: receiver.close())
        self._lock = threading.RLock()
        self.channels: dict[int, tuple[SourceChannel, Any]] = {}
        self.active_generation: int | None = None
        self.deferred_close: set[int] = set()

    def reconcile(self, plan: tuple[SourceChannel, ...]) -> dict[str, list[int]]:
        """Open new sources (sorted by source id), close departed idle ones."""
        with self._lock:
            wanted = {c.generation: c for c in plan}
            for generation, (channel, _) in self.channels.items():
                if generation in wanted and wanted[generation] != channel:
                    raise ValueError("a generation cannot change source or port")
            closed, deferred = [], []
            for generation in sorted(set(self.channels) - set(wanted)):
                channel, receiver = self.channels[generation]
                if not _idle(receiver):
                    deferred.append(generation)
                    continue
                # Unselectable first, then torn down.
                del self.channels[generation]
                if self.active_generation == generation:
                    self.active_generation = None
                self._retire(receiver, channel)
                closed.append(generation)
            self.deferred_close = set(deferred)
            new = sorted(
                (c for g, c in wanted.items() if g not in self.channels),
                key=lambda c: c.source,
            )
            bound = []
            try:
                for channel in new:
                    bound.append((channel, self._bind(channel)))
                for channel, receiver in bound:
                    self._open(receiver, channel)
                    self.channels[channel.generation] = (channel, receiver)
            except BaseException:
                for channel, receiver in bound:
                    held = self.channels.get(channel.generation)
                    if held is None or held[1] is not receiver:
                        receiver.close()
                raise
            return dict(
                opened=[c.generation for c in new], closed=closed, deferred=deferred
            )

    def select(self, generation: int) -> Any:
        """Make ``generation`` the active receiver for the next session."""
        with self._lock:
            if generation not in self.channels:
                raise ValueError(f"no preconnected source channel {generation}")
            if generation in self.deferred_close:
                raise ValueError(f"source channel {generation} is being retired")
            current = self.active_generation
            if current not in (None, generation) and not _idle(
                self.channels[current][1]
            ):
                raise RuntimeError("cannot switch source while a session is active")
            self.active_generation = generation
            return self.channels[generation][1]

    def discard(self, receiver: Any) -> None:
        """Forget a receiver the connector closed after a failure."""
        with self._lock:
            for generation, (_, held) in list(self.channels.items()):
                if held is receiver:
                    del self.channels[generation]
                    if self.active_generation == generation:
                        self.active_generation = None

    def close_all(self) -> None:
        with self._lock:
            for _, receiver in self.channels.values():
                receiver.close()
            self.channels.clear()
            self.active_generation = None
