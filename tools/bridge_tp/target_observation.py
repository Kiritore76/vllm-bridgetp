# SPDX-License-Identifier: Apache-2.0
"""Candidate-before native-target observations, independent of future recipes."""

import json
import math
from collections import deque
from pathlib import Path

VERSION = "TARGET_OBSERVATION_V1"


class TargetObserver:
    """Tail bounded real-arrival/ITL evidence without rescanning whole logs."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.pending = b""
        self.events = deque()
        self.integrity_errors = []
        self.last_arrival = None

    def poll(self):
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read(1024 * 1024)
            self.offset = handle.tell()
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        for line in lines:
            try:
                row = json.loads(line)
                if row.get("kind") == "evidence_footer":
                    if row.get("dropped") or row.get("writer_error"):
                        self.integrity_errors.append("target evidence incomplete")
                else:
                    self.events.append(row)
            except (ValueError, TypeError):
                self.integrity_errors.append("invalid target evidence")

    def snapshot(self, now, pool):
        """Return actual observations and explicit support/freshness states."""
        self.poll()
        for row in self.events:
            if row.get("kind") == "native_arrival" and row["unix_s"] <= now:
                self.last_arrival = max(
                    self.last_arrival or row["unix_s"], row["unix_s"]
                )
        # Keep future timestamps until their time arrives; never use them early.
        self.events = deque(
            row for row in self.events if row.get("unix_s", now) >= now - 12
        )
        arrivals = [
            row
            for row in self.events
            if row.get("kind") == "native_arrival" and now - 10 <= row["unix_s"] <= now
        ]
        intervals = [
            row
            for row in self.events
            if row.get("kind") == "native_itl" and now - 3 <= row["unix_s"] <= now
        ]
        ratios = [
            row["itl_s"] / row["tpot_reference_s"]
            for row in intervals
            if row.get("tpot_reference_s", 0) > 0
            and math.isfinite(row.get("itl_s", math.inf))
            and row["itl_s"] >= 0
        ]
        age = now - max((row["unix_s"] for row in intervals), default=now - 99)
        pending = pool.get("prefill_pending_tokens")
        sampled = pool.get("target_observation_unix_s")
        telemetry_age = now - sampled if sampled is not None else None
        fresh = telemetry_age is not None and 0 <= telemetry_age <= 2
        scrape_age = now - pool.get("sampled_unix_s", 0)
        # An idle engine need not produce new scheduler reports. Its last
        # empty report remains authoritative while the endpoint is live and
        # no native arrival occurred after that report. Busy reports expire.
        idle_report = (
            sampled is not None
            and 0 < sampled <= now
            and 0 <= scrape_age <= 2
            and (self.last_arrival is None or self.last_arrival <= sampled)
        )
        idle = (
            idle_report
            and pool.get("num_running") == 0
            and pool.get("num_waiting") == 0
            and pending == 0
        )
        pressure = None
        status = "MISSING"
        if idle:
            pressure, status = 1.0, "IDLE"
        elif fresh and len(ratios) >= 20 and 0 <= age <= 2:
            ordered = sorted(ratios)
            index = 0.95 * (len(ordered) - 1)
            lower = int(index)
            upper = min(lower + 1, len(ordered) - 1)
            pressure = ordered[lower] + (index - lower) * (
                ordered[upper] - ordered[lower]
            )
            status = "VALID"
        if not self.path.exists() or self.integrity_errors:
            status, pressure = "MISSING", None
        return {
            "version": VERSION,
            "unix_s": now,
            "prompt_arrival_window_s": 10.0,
            "prompt_arrival_tokens_s": (
                sum(row["prompt_tokens"] for row in arrivals) / 10
                if self.path.exists() and not self.integrity_errors
                else None
            ),
            "prompt_arrival_samples": len(arrivals),
            "prefill_pending_tokens": pending if fresh or idle else None,
            "decode_window_s": 3.0,
            "decode_interval_samples": len(ratios),
            "decode_latest_age_s": age if intervals else None,
            "decode_pressure_p95": pressure,
            "decode_status": status,
            "telemetry_age_s": telemetry_age,
            "scrape_age_s": scrape_age,
            "idle_report_still_current": idle,
            "running": pool.get("num_running"),
            "waiting": pool.get("num_waiting"),
            "integrity_errors": list(self.integrity_errors),
        }


def state_eligibility(observation, state, decode_min=1.0, prefill_min=1024):
    """Gate collection strata only; never replace physical migration safety."""
    if state == "any":
        return ()
    pending = observation.get("prefill_pending_tokens")
    if state == "idle":
        valid = observation.get("decode_status") == "IDLE"
    elif state == "decode":
        pressure = observation.get("decode_pressure_p95")
        valid = (
            observation.get("decode_status") == "VALID"
            and pressure is not None
            and pressure >= decode_min
            and pending == 0
        )
    elif state == "prefill":
        valid = pending is not None and pending >= prefill_min
    else:
        raise ValueError("unknown target state")
    return () if valid else ("planned target state not observed",)
