# SPDX-License-Identifier: Apache-2.0
"""Process-local event timeline for BridgeTP runtime-upgrade experiments."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def emit_event(
    run_dir: str | Path,
    component: str,
    event: str,
    *,
    request_id: str | None = None,
    migration_id: str | None = None,
    **detail: Any,
) -> dict[str, Any]:
    row = {
        "format_version": 1,
        "component": component,
        "event": event,
        "request_id": request_id,
        "migration_id": migration_id,
        "pid": os.getpid(),
        "unix_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        **detail,
    }
    root = Path(run_dir) / "timeline_parts"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{component}_{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
    return row


def merge_parts(run_dir: str | Path) -> dict[str, Any]:
    """Merge process-local timelines and verify each clock never regresses."""
    root = Path(run_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "timeline_parts").glob("*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    errors: list[str] = []
    by_process: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["component"]), int(row["pid"]))
        by_process.setdefault(key, []).append(row)
    for key, process_rows in by_process.items():
        monotonic = [int(row["monotonic_ns"]) for row in process_rows]
        if monotonic != sorted(monotonic):
            errors.append(f"monotonic clock regressed for {key[0]} pid={key[1]}")
    rows.sort(key=lambda row: (int(row["unix_ns"]), int(row["monotonic_ns"])))
    output = root / "timeline.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    result = {
        "format_version": 1,
        "status": "PASS" if rows and not errors else "FAIL",
        "events": len(rows),
        "components": sorted({str(row["component"]) for row in rows}),
        "errors": errors if rows else ["timeline contains no events"],
    }
    _atomic_json_dump(result, root / "timeline_acceptance.json")
    return result
