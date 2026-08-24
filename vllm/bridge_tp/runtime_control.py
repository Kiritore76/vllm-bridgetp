# SPDX-License-Identifier: Apache-2.0
"""Runtime-mutable migration knobs.

WHY THIS FILE EXISTS
--------------------
Phase 6/8 read their three actuation knobs from the environment:

    BRIDGETP_STREAM_AFTER_OUTPUT_TOKENS     trigger boundary
    BRIDGETP_PHASE8_CUTOVER_OUTPUT_TOKENS   cutover boundary
    BRIDGETP_STREAM_RATE_GIB_S              migration rate

and the getter in ``vllm/bridge_tp/kv_stream.py`` is wrapped in
``@lru_cache(maxsize=1)``, so all three are frozen at the first read inside the
server process. That is correct for Phase 6/8, where a human fixes the
boundaries before launching the run. It is fatal for Phase 9: a controller that
cannot change the trigger, the cutover, or the rate at runtime has nothing to
actuate, and every policy decision it makes is unobservable.

This module replaces the frozen read with a control block that lives in the run
directory, so the controller process and the server process can share it
without an RPC. The server side stats one file per decode iteration and
re-reads only when the modification time changes, which is negligible next to a
14B decode step.

INTEGRATION (see PHASE9_EXPERIMENT_PLAN.md, step P9-0)
------------------------------------------------------
In ``vllm/bridge_tp/kv_stream.py``:

    - drop ``@lru_cache`` from the stream-config getter;
    - keep the environment variables as the defaults;
    - overlay ``RuntimeControl.load(run_dir)`` on top of them each call.

The environment stays authoritative when no control file exists, so every
existing Phase 6/7/8 run script keeps working unchanged.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path

CONTROL_FILENAME = "runtime_control.json"


def _atomic_json_payload(payload: dict[str, object], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=str(target.parent),
            delete=False,
            encoding="utf-8",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class RuntimeControl:
    """Knobs the Phase 9 controller may change while the server is running."""

    generation: int = 0
    armed: bool = False
    trigger_output_tokens: int | None = None
    cutover_output_tokens: int | None = None
    rate_gib_s: float | None = None
    target_request_admitted: bool = False
    note: str = ""

    # ---- read side (server process) -----------------------------------
    @staticmethod
    def path(run_dir: str | os.PathLike[str]) -> Path:
        return Path(run_dir) / CONTROL_FILENAME

    @classmethod
    def load(cls, run_dir: str | os.PathLike[str]) -> RuntimeControl | None:
        target = cls.path(run_dir)
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if raw.get("format_version") != 1:
            return None
        payload = {k: v for k, v in raw.items() if k in _FIELDS}
        return cls(**payload)

    # ---- write side (controller process) ------------------------------
    def write(self, run_dir: str | os.PathLike[str]) -> RuntimeControl:
        """Atomically publish this control block, bumping the generation."""
        target = self.path(run_dir)
        nxt = replace(self, generation=self.generation + 1)
        payload = {"format_version": 1, **asdict(nxt)}
        _atomic_json_payload(payload, target)
        return nxt


_FIELDS = set(RuntimeControl.__dataclass_fields__)


class ControlCache:
    """mtime-gated cache so the hot decode path does not re-parse JSON.

    One ``stat`` per decode iteration; a re-read only when the controller has
    actually published a new block.
    """

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self._run_dir = Path(run_dir)
        self._path = RuntimeControl.path(run_dir)
        self._lock = threading.Lock()
        self._mtime_ns: int | None = None
        self._value: RuntimeControl | None = None

    def get(self) -> RuntimeControl | None:
        try:
            mtime_ns = self._path.stat().st_mtime_ns
        except OSError:
            return None
        with self._lock:
            if mtime_ns != self._mtime_ns:
                self._value = RuntimeControl.load(self._run_dir)
                self._mtime_ns = mtime_ns
            return self._value


HONORED_FILENAME = "runtime_control_honored"
_honored_generation: dict[str, int] = {}
_honored_lock = threading.Lock()


def mark_control_honored(run_dir: str | os.PathLike[str], generation: int) -> None:
    """Record that the server actually consumed a control block.

    ``run_phase9_controller.py`` refuses to start until this marker exists,
    because a controller whose actions are silently ignored produces data with
    no causal relationship to its decisions -- the single most expensive way to
    waste a cluster night.

    Writes at most once per generation per process, so the decode path pays a
    dict lookup rather than a filesystem write on every iteration.
    """
    key = str(run_dir)
    with _honored_lock:
        if _honored_generation.get(key) == generation:
            return
        target = Path(run_dir) / HONORED_FILENAME
        try:
            _atomic_json_payload(
                {"format_version": 1, "generation": int(generation)},
                target,
            )
        except OSError:
            # Never let bookkeeping break a live migration. Do not cache the
            # failure, so the next decode iteration can retry the marker.
            return
        _honored_generation[key] = generation


def effective_config(
    env_trigger: int,
    env_cutover: int,
    env_rate_gib_s: float,
    control: RuntimeControl | None,
) -> tuple[int, int, float, bool]:
    """Overlay a control block on the environment defaults.

    Returns ``(trigger, cutover, rate_gib_s, armed)``. When no control block is
    present the environment wins and ``armed`` is True, which preserves the
    Phase 6/7/8 behaviour exactly.
    """
    if control is None:
        return env_trigger, env_cutover, env_rate_gib_s, True
    trigger = (
        control.trigger_output_tokens
        if control.trigger_output_tokens is not None
        else env_trigger
    )
    cutover = (
        control.cutover_output_tokens
        if control.cutover_output_tokens is not None
        else env_cutover
    )
    rate = control.rate_gib_s if control.rate_gib_s is not None else env_rate_gib_s
    if cutover <= trigger:
        raise ValueError(
            f"cutover boundary {cutover} must be strictly after trigger {trigger}"
        )
    if rate < 0:
        raise ValueError("rate_gib_s cannot be negative")
    return trigger, cutover, rate, bool(control.armed)
