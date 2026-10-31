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
_TARGET_ID = re.compile(r"tp4|[A-Z]-g[0-9]+-tp[0-9]+")


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
            or target.tp not in (2, 4)
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
