#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay Phase 9 audit ticks through the read-only M0 manager.

This uses only evidence recorded in the input audit.  Historical TODO7/TODO9
traces often lack resident-byte and armed-rank fields; those remain null in
the output rather than being reconstructed from future receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if importlib.util.find_spec("torch") is None and "vllm" not in sys.modules:
    # Replay needs only pure-Python controller modules on a development host.
    package = types.ModuleType("vllm")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "vllm")]
    sys.modules["vllm"] = package

from vllm.bridge_tp.controller.audit import read_audit  # noqa: E402
from vllm.bridge_tp.controller.manager_m0 import (  # noqa: E402
    M0Proposal,
    MigrationManagerM0,
    RuntimeStateCollector,
)


def replay(path: Path, max_age_s: float = 2.0) -> list[dict[str, Any]]:
    """Return deterministic per-tick snapshots and advisory decisions."""
    manager = MigrationManagerM0(max_sample_age_s=max_age_s)
    collector = RuntimeStateCollector()
    metadata: dict[str, Any] = {}
    tick: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []

    def flush() -> None:
        if tick is None:
            return
        state = str(tick.get("state", ""))
        decision = next(
            (row for row in events if row.get("kind") == "decision"), None
        )
        rate = next((row for row in events if row.get("kind") == "rate"), None)
        abandoned = any(row.get("kind") == "abandon" for row in events)
        progress: dict[str, Any] = {}
        resident = next(
            (row for row in events if row.get("kind") == "earliest_ready_poll"),
            None,
        )
        delta = next(
            (
                row
                for row in events
                if row.get("kind") == "earliest_ready_delta_progress"
            ),
            None,
        )
        if resident is not None:
            progress["all_ranks_history_resident"] = resident.get("ready")
        if delta is not None:
            progress["delta_lag_tokens"] = delta.get("delta_lag_tokens")
        proposal = M0Proposal(
            start=(decision.get("action") == "START_SHADOW")
            if decision is not None and state == "LOCAL"
            else None,
            rate_bytes_s=rate.get("rate_bytes_s") if rate is not None else None,
            cancel=abandoned if state == "SHADOW" else None,
            origin="phase9_audit",
        )
        snapshot = collector.collect(
            tick,
            migration_id=metadata.get("migration_id"),
            request_id=metadata.get("source_request_id"),
            channel_available=True if state == "LOCAL" else None,
            expected_remaining_tokens=(
                decision.get("expected_remaining_tokens")
                if decision is not None
                and decision.get("expected_remaining_tokens", 0) > 0
                else None
            ),
            progress=progress,
        )
        result = manager.decide(snapshot, proposal)
        output.append(
            {
                "kind": "manager_m0_replay",
                "tick": tick.get("tick"),
                "snapshot": snapshot.to_json(),
                "proposal": asdict(proposal),
                "decision": result.to_json(),
            }
        )

    for row in read_audit(path):
        kind = row.get("kind")
        if kind == "run_metadata":
            metadata = row
        elif kind == "telemetry":
            flush()
            tick = row
            events = []
        elif tick is not None:
            events.append(row)
    flush()
    return output


def main() -> None:
    """Write a separate, source-hashed replay artifact and coverage summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-sample-age-s", type=float, default=2.0)
    args = parser.parse_args()
    source = args.audit.resolve(strict=True)
    target = args.out.resolve()
    if source == target:
        parser.error("output must not overwrite the source audit")
    rows = replay(source, args.max_sample_age_s)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    actions = Counter(row["decision"]["action"] for row in rows)
    missing = Counter(
        field for row in rows for field in row["decision"]["missing"]
    )
    unknown = Counter(
        field
        for row in rows
        for field, value in row["snapshot"].items()
        if value is None
    )
    summary = {
        "source_audit": str(source),
        "source_sha256": digest,
        "ticks": len(rows),
        "actions": dict(actions),
        "missing_evidence": dict(missing),
        "unknown_snapshot_fields": dict(unknown),
        "interpretation": (
            "Advisory M0 replay only; historical fixed-boundary actions are not "
            "counterfactual manager outcomes."
        ),
    }
    summary_path = target.with_suffix(target.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
