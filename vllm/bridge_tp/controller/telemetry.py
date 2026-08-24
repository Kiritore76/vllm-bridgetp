# SPDX-License-Identifier: Apache-2.0
"""Telemetry scraping and normalization for the Phase 9 controller.

Reads the vLLM Prometheus endpoint and produces :class:`PoolTelemetry`.

Two traps are guarded here explicitly because both silently corrupt the risk
signal rather than raising:

1. ``*_created`` series. Prometheus client libraries emit a companion
   ``<name>_created`` gauge whose value is the UNIX timestamp at which the
   counter was created. Reading that as a preemption count yields ~1.7e9 and a
   permanently saturated risk signal. :func:`parse_prometheus` refuses to
   return ``_created`` series at all.

2. Histogram quantiles. vLLM exports TPOT as a histogram, not a summary, so
   there is no ready-made P99 series. :func:`histogram_quantile` interpolates
   from ``_bucket`` samples; a naive read of ``_sum / _count`` gives the mean
   and would hide exactly the tail behavior the rate controller reacts to.
"""

from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import dataclass

from .events import PoolTelemetry


class TelemetryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict[str, str]
    value: float


def parse_prometheus(text: str) -> list[Sample]:
    """Minimal Prometheus text-format parser.

    Deliberately rejects ``_created`` series so they cannot be mistaken for
    counters anywhere downstream.
    """
    samples: list[Sample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            label_blob, _, value_blob = rest.partition("}")
            labels: dict[str, str] = {}
            for pair in _split_labels(label_blob):
                if "=" not in pair:
                    continue
                key, _, val = pair.partition("=")
                labels[key.strip()] = val.strip().strip('"')
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, labels, value_blob = parts[0], {}, parts[1]
        name = name.strip()
        if name.endswith("_created"):
            continue
        try:
            value = float(value_blob.split()[0])
        except (ValueError, IndexError):
            continue
        if math.isnan(value):
            continue
        samples.append(Sample(name, labels, value))
    return samples


def _split_labels(blob: str) -> list[str]:
    out, buf, in_quote = [], [], False
    for ch in blob:
        if ch == '"':
            in_quote = not in_quote
            buf.append(ch)
        elif ch == "," and not in_quote:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def first_value(samples: list[Sample], name: str, default: float = 0.0) -> float:
    for sample in samples:
        if sample.name == name:
            return sample.value
    return default


def histogram_quantile(samples: list[Sample], metric: str, q: float) -> float:
    """Linear-interpolated quantile from ``<metric>_bucket`` samples."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0,1)")
    buckets: list[tuple[float, float]] = []
    for sample in samples:
        if sample.name != f"{metric}_bucket":
            continue
        le = sample.labels.get("le")
        if le is None:
            continue
        bound = math.inf if le in ("+Inf", "Inf") else float(le)
        buckets.append((bound, sample.value))
    if not buckets:
        return 0.0
    buckets.sort(key=lambda item: item[0])
    total = buckets[-1][1]
    if total <= 0:
        return 0.0
    target = q * total
    previous_bound, previous_count = 0.0, 0.0
    for bound, cumulative in buckets:
        if cumulative >= target:
            if math.isinf(bound):
                return previous_bound
            span = cumulative - previous_count
            if span <= 0:
                return bound
            frac = (target - previous_count) / span
            return previous_bound + frac * (bound - previous_bound)
        previous_bound, previous_count = bound, cumulative
    return previous_bound


def pool_from_samples(
    samples: list[Sample],
    block_size: int,
    total_kv_blocks: int,
    now_unix_s: float | None = None,
) -> PoolTelemetry:
    """Build a :class:`PoolTelemetry` from one scrape.

    Metric names follow vLLM V1. If a deployment renames them, override here
    rather than scattering string literals through the policy.
    """
    kv_usage = first_value(samples, "vllm:gpu_cache_usage_perc", 0.0)
    if kv_usage > 1.0:  # some builds export percent, others fraction
        kv_usage /= 100.0
    kv_usage = max(0.0, min(1.0, kv_usage))
    free_blocks = int(round(total_kv_blocks * (1.0 - kv_usage)))
    return PoolTelemetry(
        num_running=int(first_value(samples, "vllm:num_requests_running", 0.0)),
        num_waiting=int(first_value(samples, "vllm:num_requests_waiting", 0.0)),
        kv_usage_frac=kv_usage,
        preemptions_total=int(first_value(samples, "vllm:num_preemptions_total", 0.0)),
        p99_tpot_s=histogram_quantile(
            samples, "vllm:time_per_output_token_seconds", 0.99
        ),
        mean_tpot_s=_safe_mean(
            first_value(samples, "vllm:time_per_output_token_seconds_sum", 0.0),
            first_value(samples, "vllm:time_per_output_token_seconds_count", 0.0),
        ),
        free_kv_blocks=free_blocks,
        block_size=block_size,
        sampled_unix_s=now_unix_s if now_unix_s is not None else time.time(),
    )


def _safe_mean(total: float, count: float) -> float:
    return total / count if count > 0 else 0.0


class MetricsScraper:
    """Polls one vLLM ``/metrics`` endpoint."""

    def __init__(
        self,
        base_url: str,
        block_size: int,
        total_kv_blocks: int,
        timeout_s: float = 2.0,
    ) -> None:
        self.url = base_url.rstrip("/") + "/metrics"
        self.block_size = block_size
        self.total_kv_blocks = total_kv_blocks
        self.timeout_s = timeout_s

    def scrape(self) -> PoolTelemetry:
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout_s) as response:
                text = response.read().decode("utf-8", errors="replace")
        except OSError as error:  # pragma: no cover - network path
            raise TelemetryError(f"failed to scrape {self.url}: {error}") from error
        return pool_from_samples(
            parse_prometheus(text), self.block_size, self.total_kv_blocks
        )
