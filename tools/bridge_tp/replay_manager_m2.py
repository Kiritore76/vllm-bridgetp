#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay M2 rate choices on the audited M0 runtime snapshots."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if importlib.util.find_spec("torch") is None and "vllm" not in sys.modules:
    package = types.ModuleType("vllm")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "vllm")]
    sys.modules["vllm"] = package

from vllm.bridge_tp.controller.manager_m0 import RuntimeSnapshot  # noqa: E402
from vllm.bridge_tp.controller.manager_m2 import (  # noqa: E402
    M2RateConfig,
    M2RateController,
)

GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--low-gib-s", type=float, required=True)
    parser.add_argument("--medium-gib-s", type=float, required=True)
    parser.add_argument("--high-gib-s", type=float, required=True)
    return parser.parse_args()


def replay(path: Path, config: M2RateConfig) -> dict:
    controller = M2RateController(config)
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    changes = []
    active_ticks = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("kind") != "manager_m0_shadow":
                continue
            snapshot = RuntimeSnapshot(**event["snapshot"])
            if snapshot.state in {"SHADOW", "READY_NOT_COMMITTED"}:
                active_ticks += 1
            decision = controller.decide(snapshot)
            counts[decision.profile] += 1
            reasons[decision.reason] += 1
            if decision.action == "SET_RATE":
                changes.append({
                    "tick": event.get("tick"),
                    "unix_s": snapshot.unix_s,
                    "profile": decision.profile,
                    "rate_gib_s": decision.rate_bytes_s / GIB,
                    "reason": decision.reason,
                })
    return {
        "format_version": 1,
        "mode": "advisory_replay_no_actuation",
        "input": str(path.resolve()),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rate_profiles_gib_s": {
            "LOW": config.low_bytes_s / GIB,
            "MEDIUM": config.medium_bytes_s / GIB,
            "HIGH": config.high_bytes_s / GIB,
        },
        "active_shadow_ticks": active_ticks,
        "profile_ticks": dict(counts),
        "reasons": dict(reasons),
        "changes": changes,
    }


def main() -> None:
    args = parse_args()
    config = M2RateConfig(
        low_bytes_s=args.low_gib_s * GIB,
        medium_bytes_s=args.medium_gib_s * GIB,
        high_bytes_s=args.high_gib_s * GIB,
    )
    print(json.dumps(replay(args.audit, config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
