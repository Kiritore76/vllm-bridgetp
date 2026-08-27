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
from typing import Iterable

from .events import PoolTelemetry


# vLLM V1 currently exports request-level TPOT under the first name.  Older
# Phase 9 environments used the second name.  Do not substitute the
# inter-token-latency histogram here: it has one observation per token gap,
# whereas the controller model and calibration response are request-level
# TPOT.
REQUEST_TPOT_METRICS = (
    "vllm:request_time_per_output_token_seconds",
    "vllm:time_per_output_token_seconds",
)


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


def first_value_for_names(
    samples: list[Sample], names: Iterable[str], default: float = 0.0
) -> float:
    """Return the first available metric among ordered aliases."""
    available = {sample.name for sample in samples}
    for name in names:
        if name in available:
            return first_value(samples, name, default)
    return default


def has_metric(samples: list[Sample], name: str) -> bool:
    """Return whether a scrape contains at least one sample with ``name``."""
    return any(sample.name == name for sample in samples)


def request_tpot_metric(samples: list[Sample]) -> str | None:
    """Return the request-level TPOT histogram exported by this server."""
    for metric in REQUEST_TPOT_METRICS:
        if has_metric(samples, f"{metric}_bucket"):
            return metric
    return None


def histogram_quantile(samples: list[Sample], metric: str, q: float) -> float:
    """Linear-interpolated quantile from ``<metric>_bucket`` samples."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0,1)")
    totals_by_bound: dict[float, float] = {}
    for sample in samples:
        if sample.name != f"{metric}_bucket":
            continue
        le = sample.labels.get("le")
        if le is None:
            continue
        bound = math.inf if le in ("+Inf", "Inf") else float(le)
        totals_by_bound[bound] = totals_by_bound.get(bound, 0.0) + sample.value
    buckets = sorted(totals_by_bound.items())
    if not buckets:
        return 0.0
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


def _counter_delta(current: float, previous: float) -> float:
    """Return a non-negative Prometheus counter delta, tolerating resets."""
    if current >= previous:
        return current - previous
    return max(0.0, current)


def histogram_delta_samples(
    previous: list[Sample], current: list[Sample], metric: str
) -> list[Sample]:
    """Build an interval histogram from two cumulative scrapes.

    Histogram buckets are matched by all labels. Counter resets are handled by
    treating the current value as the new interval count. The returned samples
    can be passed to :func:`histogram_quantile`.
    """

    def key(sample: Sample) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(sample.labels.items()))

    bucket_name = f"{metric}_bucket"
    previous_buckets = {
        key(sample): sample.value
        for sample in previous
        if sample.name == bucket_name
    }
    out: list[Sample] = []
    for sample in current:
        if sample.name != bucket_name:
            continue
        before = previous_buckets.get(key(sample), 0.0)
        out.append(
            Sample(
                name=bucket_name,
                labels=sample.labels,
                value=_counter_delta(sample.value, before),
            )
        )
    return out


def interval_histogram_stats(
    previous: list[Sample],
    current: list[Sample],
    metric: str,
    q: float = 0.99,
) -> tuple[int, float, float]:
    """Return ``(count, mean, quantile)`` for one scrape interval."""
    current_count = sum(
        sample.value for sample in current if sample.name == f"{metric}_count"
    )
    previous_count = sum(
        sample.value for sample in previous if sample.name == f"{metric}_count"
    )
    current_sum = sum(
        sample.value for sample in current if sample.name == f"{metric}_sum"
    )
    previous_sum = sum(
        sample.value for sample in previous if sample.name == f"{metric}_sum"
    )
    count = _counter_delta(current_count, previous_count)
    total = _counter_delta(current_sum, previous_sum)
    delta_samples = histogram_delta_samples(previous, current, metric)
    quantile = histogram_quantile(delta_samples, metric, q) if count > 0 else 0.0
    mean = total / count if count > 0 else 0.0
    return int(round(count)), mean, quantile


def pool_from_samples(
    samples: list[Sample],
    block_size: int,
    total_kv_blocks: int,
    now_unix_s: float | None = None,
    tpot_metric: str | None = None,
) -> PoolTelemetry:
    """Build a :class:`PoolTelemetry` from one scrape.

    Metric names follow vLLM V1. If a deployment renames them, override here
    rather than scattering string literals through the policy.
    """
    # V1 renamed this metric in newer releases. Prefer the name observed on
    # current Phase 9 servers, while retaining compatibility with older P1/P2
    # environments.
    kv_usage = first_value_for_names(
        samples,
        (
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
        ),
        0.0,
    )
    if kv_usage > 1.0:  # some builds export percent, others fraction
        kv_usage /= 100.0
    kv_usage = max(0.0, min(1.0, kv_usage))
    free_blocks = int(round(total_kv_blocks * (1.0 - kv_usage)))
    selected_tpot_metric = tpot_metric or request_tpot_metric(samples)
    return PoolTelemetry(
        num_running=int(first_value(samples, "vllm:num_requests_running", 0.0)),
        num_waiting=int(first_value(samples, "vllm:num_requests_waiting", 0.0)),
        kv_usage_frac=kv_usage,
        preemptions_total=int(first_value(samples, "vllm:num_preemptions_total", 0.0)),
        p99_tpot_s=(
            histogram_quantile(samples, selected_tpot_metric, 0.99)
            if selected_tpot_metric is not None
            else 0.0
        ),
        mean_tpot_s=(
            _safe_mean(
                first_value(samples, f"{selected_tpot_metric}_sum", 0.0),
                first_value(samples, f"{selected_tpot_metric}_count", 0.0),
            )
            if selected_tpot_metric is not None
            else 0.0
        ),
        free_kv_blocks=free_blocks,
        block_size=block_size,
        sampled_unix_s=now_unix_s if now_unix_s is not None else time.time(),
    )


def interval_pool_from_samples(
    previous: list[Sample],
    current: list[Sample],
    block_size: int,
    total_kv_blocks: int,
    now_unix_s: float | None = None,
    tpot_metric: str | None = None,
) -> tuple[PoolTelemetry, int]:
    """Build pool telemetry whose TPOT fields cover one scrape interval.

    Gauges and monotone counters come from the current scrape. TPOT mean and
    P99 are computed from deltas between cumulative Prometheus histograms.
    The second return value is the number of TPOT observations in the interval.
    """
    selected_tpot_metric = (
        tpot_metric
        or request_tpot_metric(current)
        or request_tpot_metric(previous)
    )
    pool = pool_from_samples(
        current,
        block_size=block_size,
        total_kv_blocks=total_kv_blocks,
        now_unix_s=now_unix_s,
        tpot_metric=selected_tpot_metric,
    )
    if selected_tpot_metric is None:
        count, mean, p99 = 0, 0.0, 0.0
    else:
        count, mean, p99 = interval_histogram_stats(
            previous,
            current,
            selected_tpot_metric,
            0.99,
        )
    return (
        PoolTelemetry(
            num_running=pool.num_running,
            num_waiting=pool.num_waiting,
            kv_usage_frac=pool.kv_usage_frac,
            preemptions_total=pool.preemptions_total,
            p99_tpot_s=p99,
            mean_tpot_s=mean,
            free_kv_blocks=pool.free_kv_blocks,
            block_size=pool.block_size,
            sampled_unix_s=pool.sampled_unix_s,
        ),
        count,
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
