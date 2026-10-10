"""Discrete remaining-length distributions and offline risk evaluation.

Each category ends at an inclusive token bound. The final category is an
open-ended tail. Queries between bounds return probability bounds rather than
claiming knowledge of the within-category distribution.
"""

import math
from collections import Counter

import numpy as np


def nonuniform84_upper_edges() -> np.ndarray:
    """Return the agreed 83 finite bounds plus a separate open tail."""
    return np.asarray(
        [0, 8, 16, 32]
        + list(range(64, 513, 32))
        + list(range(576, 2049, 64))
        + list(range(2176, 4097, 128))
        + list(range(4352, 8193, 256))
        + list(range(9216, 16385, 1024)),
        dtype=np.int64,
    )


def aggregate_probabilities(
    probabilities: np.ndarray,
    source_edges: np.ndarray,
    target_edges: np.ndarray,
) -> np.ndarray:
    """Sum adjacent probabilities only when every new boundary already exists."""
    validate_edges(source_edges)
    validate_edges(target_edges)
    p = np.asarray(probabilities)
    if (
        not np.isin(target_edges, source_edges).all()
        or p.shape[-1] != len(source_edges) + 1
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(axis=-1), 1, atol=1e-6)
    ):
        raise ValueError("aggregation needs aligned edges and valid probabilities")
    mapping = np.append(np.searchsorted(target_edges, source_edges), len(target_edges))
    result = np.zeros((*p.shape[:-1], len(target_edges) + 1), dtype=p.dtype)
    for target in range(len(target_edges) + 1):
        result[..., target] = p[..., mapping == target].sum(axis=-1)
    return result


def default_upper_edges(max_tokens: int, step: int = 32) -> np.ndarray:
    if max_tokens <= 0 or step <= 0:
        raise ValueError("max_tokens and step must be positive")
    edges = {0, max_tokens}
    edges.update(x for x in (8, 16) if x < max_tokens)
    edges.update(range(step, max_tokens, step))
    return np.asarray(sorted(edges), dtype=np.int64)


def validate_edges(edges: np.ndarray) -> None:
    if (
        edges.ndim != 1
        or not len(edges)
        or edges[0] != 0
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError("category upper edges must start at 0 and increase")


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    exp = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def category_targets(remaining: np.ndarray, edges: np.ndarray) -> np.ndarray:
    validate_edges(edges)
    if not np.isfinite(remaining).all() or (remaining < 0).any():
        raise ValueError("remaining lengths must be finite and nonnegative")
    return np.searchsorted(edges, remaining, side="left")


def probability_gt_bounds(
    probabilities: np.ndarray, edges: np.ndarray, horizon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper bounds on P(N > horizon), including the open tail."""
    validate_edges(edges)
    p = np.asarray(probabilities)
    if (
        p.shape[-1] != len(edges) + 1
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(axis=-1), 1, atol=1e-6)
    ):
        raise ValueError("invalid category probabilities")
    if math.isnan(horizon):
        raise ValueError("horizon cannot be NaN")
    shape = p.shape[:-1]
    if horizon < 0:
        return np.ones(shape), np.ones(shape)
    if math.isinf(horizon):
        return np.zeros(shape), np.zeros(shape)
    category = int(np.searchsorted(edges, horizon, side="left"))
    if category == len(edges):
        return np.zeros(shape), p[..., -1]
    lower = p[..., category + 1 :].sum(axis=-1)
    if horizon == edges[category]:
        return lower, lower
    return lower, p[..., category:].sum(axis=-1)


def probability_remaining_gt(
    probabilities: np.ndarray, edges: np.ndarray, horizon: float
) -> np.ndarray:
    """Use the conservative upper bound for a dynamic capacity threshold."""
    return probability_gt_bounds(probabilities, edges, horizon)[1]


def request_weights(request_ids: np.ndarray) -> np.ndarray:
    counts = Counter(request_ids)
    weights = np.asarray([1 / counts[x] for x in request_ids], dtype=np.float64)
    return weights / weights.sum()


def distribution_nll(
    probabilities: np.ndarray,
    target: np.ndarray,
    censored: np.ndarray,
    request_ids: np.ndarray,
) -> float:
    """Exact category likelihood or compatible-tail likelihood for censored rows."""
    row = np.arange(len(target))
    likelihood = probabilities[row, target].copy()
    tail = np.flip(np.cumsum(np.flip(probabilities, axis=1), axis=1), axis=1)
    likelihood[censored] = tail[row[censored], target[censored]]
    return float(
        np.sum(-np.log(likelihood.clip(min=1e-12)) * request_weights(request_ids))
    )


def binary_risk_metrics(
    predicted: np.ndarray, actual: np.ndarray, request_ids: np.ndarray
) -> dict:
    weights = request_weights(request_ids)
    predicted = np.asarray(predicted, dtype=np.float64).clip(0, 1)
    actual = np.asarray(actual, dtype=np.float64)
    clipped = predicted.clip(1e-7, 1 - 1e-7)
    reliability = []
    buckets = np.minimum((predicted * 10).astype(int), 9)
    for bucket in range(10):
        lo, hi = bucket / 10, (bucket + 1) / 10
        mask = buckets == bucket
        if not mask.any():
            continue
        w = weights[mask]
        mean_p = float(np.average(predicted[mask], weights=w))
        rate = float(np.average(actual[mask], weights=w))
        reliability.append(
            {
                "lower": float(lo),
                "upper": float(hi),
                "samples": int(mask.sum()),
                "requests": int(len(set(request_ids[mask]))),
                "weight": float(w.sum()),
                "predicted_probability": mean_p,
                "observed_exceedance_rate": rate,
            }
        )
    low = predicted <= 0.05
    return {
        "samples": len(actual),
        "requests": len(set(request_ids)),
        "positive_samples": int(actual.sum()),
        "brier": float(np.sum(weights * (predicted - actual) ** 2)),
        "binary_log_loss": float(
            np.sum(
                weights
                * (-actual * np.log(clipped) - (1 - actual) * np.log1p(-clipped))
            )
        ),
        "ece": sum(
            x["weight"]
            * abs(x["predicted_probability"] - x["observed_exceedance_rate"])
            for x in reliability
        ),
        "mean_predicted_probability": float(np.sum(weights * predicted)),
        "observed_exceedance_rate": float(np.sum(weights * actual)),
        "low_risk_samples": int(low.sum()),
        "low_risk_positive_samples": int(actual[low].sum()),
        "low_risk_exceedance_rate": (
            float(np.average(actual[low], weights=weights[low])) if low.any() else None
        ),
        "reliability": reliability,
    }


def km_conditional_survival(
    output_lengths: np.ndarray,
    censored: np.ndarray,
    generated: np.ndarray,
    horizon: float,
) -> np.ndarray:
    """Training-only Kaplan-Meier P(L > g+h | L > g), retaining censoring."""
    times = np.unique(output_lengths)
    curve = []
    survival = 1.0
    for time in times:
        at_risk = int((output_lengths >= time).sum())
        events = int(((output_lengths == time) & ~censored).sum())
        survival *= 1 - events / at_risk
        curve.append(survival)
    curve = np.asarray(curve)

    def query(values):
        indices = np.searchsorted(times, values, side="right") - 1
        result = np.where(indices < 0, 1.0, curve[np.maximum(0, indices)])
        if survival > 0:
            result = np.where(values > times[-1], np.nan, result)
        return result

    denominator = query(generated)
    return np.divide(
        query(generated + horizon),
        denominator,
        out=np.full(len(generated), np.nan),
        where=(denominator > 0) & (generated < times[-1]),
    )
