# SPDX-License-Identifier: Apache-2.0
"""Several preconnected TP-k targets for one TP1 source (S1-d2a).

``BRIDGETP_GPU_DIRECT_TARGETS`` lists every target this source keeps a
persistent channel to, as a JSON list of objects with exactly the keys
``target`` (instance id), ``host``, ``base_port``, ``tp`` and ``generation``.

Target rank r listens on ``base_port + r``; the sender control socket uses
``base_port + tp``, so blocks ``[base_port, base_port + tp]`` must not overlap.
Every target is preconnected at engine start. A session names its target in
the runtime control block (``target_instance``); the selected entry is
overlaid on the single-target fields (host, base port, TP, generation), so the
existing sender key, addresses and handshake are unchanged. Without the
variable the source keeps its single environment target.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

ENV = "BRIDGETP_GPU_DIRECT_TARGETS"
_TARGET_ID = re.compile(r"tp[14]|[A-Z]-g[0-9]+-tp[0-9]+")


@dataclass(frozen=True)
class GpuDirectTarget:
    target: str
    host: str
    base_port: int
    tp: int
    generation: int

    @property
    def ports(self) -> range:
        return range(self.base_port, self.base_port + self.tp + 1)


def parse_targets(text: str, expected_kv_heads: int) -> tuple[GpuDirectTarget, ...]:
    if not text.strip():
        return ()
    rows = json.loads(text)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{ENV} must be a non-empty JSON list")
    targets = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "target",
            "host",
            "base_port",
            "tp",
            "generation",
        }:
            raise ValueError(f"{ENV} entries need target/host/base_port/tp/generation")
        target = GpuDirectTarget(
            str(row["target"]),
            str(row["host"]),
            int(row["base_port"]),
            int(row["tp"]),
            int(row["generation"]),
        )
        if (
            not _TARGET_ID.fullmatch(target.target)
            or target.tp not in (1, 2, 4)
            or expected_kv_heads % target.tp
            or target.generation < 0
            or not 1024 <= target.base_port <= 65535 - target.tp
            or not target.host
        ):
            raise ValueError(f"invalid {ENV} entry {row!r}")
        targets.append(target)
    if len({t.target for t in targets}) != len(targets):
        raise ValueError(f"{ENV} repeats a target")
    used = [port for t in targets for port in t.ports]
    if len(set(used)) != len(used):
        raise ValueError(f"{ENV} port blocks overlap")
    return tuple(targets)


def apply_target(config, target: GpuDirectTarget):
    """The single-target view of ``config`` for one preconnected target."""
    return replace(
        config,
        gpu_direct_host=target.host,
        gpu_direct_base_port=target.base_port,
        target_tp_size=target.tp,
        channel_generation=target.generation,
        target_instance=target.target,
    )


def select_target(config, target_instance: str | None, armed: bool):
    """Overlay the session's target; an armed session must name one.

    A source with exactly one listed target uses it when the session names
    none (fixed-target sources need no per-session choice).
    """
    if not config.gpu_direct_targets:
        return config
    if target_instance is None and len(config.gpu_direct_targets) == 1:
        return apply_target(config, config.gpu_direct_targets[0])
    if target_instance is None:
        if armed:
            raise ValueError("armed multi-target session names no target_instance")
        return config
    for target in config.gpu_direct_targets:
        if target.target == target_instance:
            return apply_target(config, target)
    raise ValueError(f"target_instance {target_instance!r} is not preconnected")


def preconnect_plan(config) -> list[tuple[str, object]]:
    """(receipt label, single-target config) for every channel to open."""
    if not config.gpu_direct_targets:
        return [("source", config)]
    return [
        ("source-" + target.target, apply_target(config, target))
        for target in config.gpu_direct_targets
    ]


FILE_ENV = "BRIDGETP_GPU_DIRECT_TARGETS_FILE"


def read_target_file(path, expected_kv_heads: int) -> tuple[GpuDirectTarget, ...]:
    """Targets a TP conversion assigned to this source (S5b); () when absent.

    Same entries as ``BRIDGETP_GPU_DIRECT_TARGETS``, but the file may be empty
    (no target yet, e.g. an L0 that will merge a TP2) and is rewritten while
    the engine serves. Writers replace it atomically.
    """
    try:
        with open(path, encoding="utf-8") as stream:
            text = stream.read()
    except FileNotFoundError:
        return ()
    if not json.loads(text or "[]"):
        return ()
    return parse_targets(text, expected_kv_heads)


def merge_targets(listed, runtime) -> tuple[GpuDirectTarget, ...]:
    """Startup targets plus file targets; a listed id keeps its startup entry."""
    seen = {t.target for t in listed}
    return tuple(listed) + tuple(t for t in runtime if t.target not in seen)


class TargetFileFollower:
    """Keep this source's channels equal to its target file (S5b).

    ``connect(target)`` opens and warms a channel (blocking: the sender
    retries until the target listens); ``drop(target)`` closes an idle one and
    returns False while a session still uses it (retried next poll). Targets
    are connected in id order, matching the target's source-id handshake
    order, so preconnects cannot wait on each other in a cycle.
    """

    def __init__(self, path, expected_kv_heads, connect, drop, listed=()):
        self.path, self.heads = path, expected_kv_heads
        self.connect, self.drop = connect, drop
        self.connected = {t.target: t for t in listed}
        self.error = None

    def poll(self) -> dict[str, list[str]]:
        try:
            wanted = {t.target: t for t in read_target_file(self.path, self.heads)}
        except (OSError, ValueError) as error:  # half-written or invalid: retry
            self.error = repr(error)
            return dict(opened=[], closed=[], deferred=[])
        self.error = None
        for target_id, target in wanted.items():
            if self.connected.get(target_id) not in (None, target):
                raise ValueError(f"target {target_id} cannot change its channel")
        closed, deferred, opened = [], [], []
        for target_id in sorted(set(self.connected) - set(wanted)):
            if self.drop(self.connected[target_id]):
                del self.connected[target_id]
                closed.append(target_id)
            else:
                deferred.append(target_id)
        for target_id in sorted(set(wanted) - set(self.connected)):
            self.connect(wanted[target_id])
            self.connected[target_id] = wanted[target_id]
            opened.append(target_id)
        return dict(opened=opened, closed=closed, deferred=deferred)
