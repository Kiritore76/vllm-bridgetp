# SPDX-License-Identifier: Apache-2.0
"""Explicit target geometry shared by transport and CPU control checks."""

import json
from pathlib import Path


def validate_target_tp(value):
    if type(value) is not int or value not in (2, 4):
        raise ValueError("BridgeTP target TP must be 2 or 4")
    return value


def validate_session_tp(value):
    """A migration may also end on a TP1 (same-TP migration, user 11-05)."""
    if type(value) is not int or value not in (1, 2, 4):
        raise ValueError("BridgeTP session target TP must be 1, 2 or 4")
    return value


def session_target_tp(run_dir):
    """Read immutable session geometry; legacy manifests default to TP4 only."""
    path = Path(run_dir) / "session_manifest.json"
    if not path.exists():
        return 4
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return validate_session_tp(manifest.get("target_tp_size", 4))


def session_source_tp(run_dir):
    """Ranks of the source; manifests before same-TP migration were all TP1."""
    path = Path(run_dir) / "session_manifest.json"
    if not path.exists():
        return 1
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return validate_session_tp(manifest.get("source_tp_size", 1))
