# SPDX-License-Identifier: Apache-2.0
"""Read-only consumption of frozen length-predictor events by the manager."""

from __future__ import annotations

import bisect
import json
import math
import time
from pathlib import Path


def probability_gt_bounds(
    probabilities: tuple[float, ...], edges: tuple[int, ...], horizon: int
) -> tuple[float, float]:
    """Bounds for P(remaining > horizon), including the unresolved bin."""
    if horizon < 0:
        return 1.0, 1.0
    index = bisect.bisect_left(edges, horizon)
    if index == len(edges):
        return 0.0, probabilities[-1]
    lower = sum(probabilities[index + 1 :])
    upper = lower if horizon == edges[index] else sum(probabilities[index:])
    return lower, upper


class PredictorEventReader:
    """Incrementally read one source TP1 event file; never influence actuation."""

    def __init__(
        self, path: Path, checkpoint_sha256: str, *, max_age_s: float = 30.0,
        max_output_lag_tokens: int = 40,
    ) -> None:
        if len(checkpoint_sha256) != 64 or max_age_s <= 0 or max_output_lag_tokens < 0:
            raise ValueError("invalid M5 reader configuration")
        self.path = Path(path)
        self.checkpoint_sha256 = checkpoint_sha256
        self.max_age_s = max_age_s
        self.max_output_lag_tokens = max_output_lag_tokens
        self._offset = 0
        self._header: dict | None = None
        self._latest: dict[str, dict] = {}
        self._last_position: dict[str, int] = {}
        self._unavailable: dict[str, str] = {}

    def poll(self) -> None:
        if not self.path.exists():
            return
        if self.path.stat().st_size < self._offset:
            raise ValueError("M5 event file was truncated")
        with self.path.open("rb") as source:
            source.seek(self._offset)
            while True:
                start = source.tell()
                line = source.readline()
                if not line or not line.endswith(b"\n"):
                    self._offset = start
                    return
                self._offset = source.tell()
                row = json.loads(line)
                if self._header is None:
                    self._read_header(row)
                else:
                    self._read_event(row)

    def _read_header(self, row: dict) -> None:
        edges = row.get("category_upper_edges")
        if (
            row.get("kind") != "predictor_header"
            or row.get("format_version") != 1
            or row.get("checkpoint_sha256") != self.checkpoint_sha256
            or row.get("feature_layer") != "decoder:31"
            or row.get("interval") != 20
            or not isinstance(edges, list)
            or len(edges) < 2
            or edges[0] != 0
            or any(not isinstance(v, int) or isinstance(v, bool) for v in edges)
            or any(a >= b for a, b in zip(edges, edges[1:]))
        ):
            raise ValueError("M5 predictor event header differs from frozen contract")
        self._header = row

    def _read_event(self, row: dict) -> None:
        if row.get("checkpoint_sha256") != self.checkpoint_sha256:
            raise ValueError("M5 predictor event checkpoint differs")
        request_id = row.get("request_id")
        position = row.get("generated_tokens")
        if not isinstance(request_id, str) or not request_id or (
            not isinstance(position, int) or isinstance(position, bool) or position < 0
        ):
            raise ValueError("invalid M5 predictor request identity or position")
        if position <= self._last_position.get(request_id, -1):
            raise ValueError("M5 predictor positions are not increasing")
        self._last_position[request_id] = position
        if row.get("kind") == "predictor_unavailable":
            self._latest.pop(request_id, None)
            self._unavailable[request_id] = str(row.get("reason", "unknown"))
            return
        if row.get("kind") != "predictor_prediction":
            raise ValueError("unknown M5 predictor event")
        values = row.get("probabilities")
        assert self._header is not None
        if (
            not isinstance(values, list)
            or len(values) != len(self._header["category_upper_edges"]) + 1
            or any(
                not isinstance(v, (int, float)) or isinstance(v, bool)
                or not math.isfinite(v) or v < 0
                for v in values
            )
            or abs(sum(values) - 1.0) > 1e-4
            or not isinstance(row.get("captured_unix_ns"), int)
        ):
            raise ValueError("invalid M5 predictor probability event")
        self._latest[request_id] = row
        self._unavailable.pop(request_id, None)

    def advisory(
        self, request_id: str, output_tokens: int, headroom_tokens: int,
        short_window_tokens: int = 64, *, now_ns: int | None = None,
        max_output_tokens: int | None = None, ignore_eos: bool = False,
    ) -> dict:
        """Report raw model risk and bounds under the actual output stop rule."""
        self.poll()
        if max_output_tokens is not None and max_output_tokens < 0:
            raise ValueError("M5 output cap cannot be negative")
        remaining_cap = (
            max(0, max_output_tokens - output_tokens)
            if max_output_tokens is not None else None
        )
        result: dict = {
            "kind": "manager_m5_predictor_shadow",
            "request_id": request_id,
            "output_tokens": output_tokens,
            "headroom_tokens": headroom_tokens,
            "checkpoint_sha256": self.checkpoint_sha256,
            "max_remaining_output_tokens": remaining_cap,
            "ignore_eos": ignore_eos,
            "model_applicable_to_runtime_stop_rule": not ignore_eos,
        }
        if self._header is None:
            return {**result, "status": "NO_HEADER"}
        row = self._latest.get(request_id)
        if row is None:
            return {
                **result, "status": "UNAVAILABLE",
                "reason": self._unavailable.get(request_id, "no prediction yet"),
            }
        position = row["generated_tokens"]
        age_s = ((now_ns or time.time_ns()) - row["captured_unix_ns"]) / 1e9
        if (
            position > output_tokens
            or output_tokens - position > self.max_output_lag_tokens
            or age_s < -1 or age_s > self.max_age_s
        ):
            return {
                **result, "status": "STALE", "prediction_output_tokens": position,
                "age_s": age_s,
            }
        probabilities = tuple(float(v) for v in row["probabilities"])
        edges = tuple(self._header["category_upper_edges"])
        headroom_bounds = probability_gt_bounds(
            probabilities, edges, headroom_tokens
        )
        short_bounds = probability_gt_bounds(
            probabilities, edges, short_window_tokens
        )

        def cap_aware(bounds: tuple[float, float], horizon: int) -> tuple[float, float]:
            if remaining_cap is None:
                return bounds
            if ignore_eos:
                forced = float(remaining_cap > horizon)
                return forced, forced
            return (0.0, 0.0) if horizon >= remaining_cap else bounds

        return {
            **result,
            "status": "AVAILABLE",
            "prediction_output_tokens": position,
            "age_s": age_s,
            "p_remaining_gt_headroom_bounds": headroom_bounds,
            "p_remaining_gt_short_window_bounds": short_bounds,
            "p_remaining_gt_headroom_runtime_bounds": cap_aware(
                headroom_bounds, headroom_tokens
            ),
            "p_remaining_gt_short_window_runtime_bounds": cap_aware(
                short_bounds, short_window_tokens
            ),
            "short_window_tokens": short_window_tokens,
        }
