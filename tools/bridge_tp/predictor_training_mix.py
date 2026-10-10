"""Balance historical replay and generation stages without split leakage."""

from collections import Counter

import numpy as np
from predictor_distribution import category_targets, request_weights


def mixed_training_weights(data: dict, legacy_ids: set[str]) -> tuple[np.ndarray, dict]:
    """Allocate 25% to legacy, 75% to new requests; stages get 30/40/30.

    Each request has equal mass within its source cohort. Exact requests use
    actual generation progress: early <25%, middle 25-75%, late >=75%.
    Missing stages redistribute their mass over present stages. Censored
    requests have unknown final length and retain flat within-request weight.
    Only original training splits receive positive loss weights.
    """
    train = data["splits"] == "train"
    weights = np.zeros(len(train), dtype=np.float64)
    stage_mass = np.array([0.3, 0.4, 0.3])
    ids = data["requests"]
    cohorts = {
        "legacy5000": train & np.isin(ids, sorted(legacy_ids)),
        "new300_and_first60": train & ~np.isin(ids, sorted(legacy_ids)),
    }
    total = data["generated"] + data["remaining"]
    fraction = np.divide(
        data["generated"], total, out=np.zeros(len(total)), where=total > 0
    )
    stages = np.digitize(fraction, [0.25, 0.75])
    # Group indices once; scanning every state for each of 4000 requests is slow.
    by_id = {}
    for index in np.flatnonzero(train):
        by_id.setdefault(str(ids[index]), []).append(index)
    report = {}
    for (name, mask), mass in zip(cohorts.items(), (0.25, 0.75)):
        members = sorted(set(ids[mask]))
        if not members:
            raise ValueError(f"empty training cohort: {name}")
        for key in members:
            indices = np.asarray(by_id[str(key)])
            request_mass = mass / len(members)
            censor = data["censored"][indices]
            if censor.any() != censor.all():
                raise ValueError("request mixes exact and censored states")
            if censor.all():
                weights[indices] = request_mass / len(indices)
                continue
            present, counts = np.unique(stages[indices], return_counts=True)
            normalized = stage_mass[present] / stage_mass[present].sum()
            for stage, count, share in zip(present, counts, normalized):
                rows = indices[stages[indices] == stage]
                weights[rows] = request_mass * share / count
        report[name] = {
            "training_requests": len(members),
            "samples": int(mask.sum()),
            "loss_mass": float(weights[mask].sum()),
        }
    return weights, {
        "cohorts": report,
        "generation_stages": ["0-25%", "25-75%", "75-100%"],
        "stage_mass": stage_mass.tolist(),
        "censored_rule": "flat within request; final progress is unknown",
        "missing_stage_rule": "renormalize over present stages",
    }


def length_metrics(p, edges, remaining, ids) -> dict:
    """Report request-equal accuracy and interval width, retaining open tails."""
    w = request_weights(ids)
    cdf = p.cumsum(1)
    q = [np.minimum((cdf < level).sum(1), len(edges)) for level in (0.05, 0.5, 0.95)]
    lower = np.r_[0, edges + 1]
    upper = np.r_[edges, np.inf]
    lo, hi = lower[q[0]], upper[q[2]]
    median_upper = upper[q[1]]
    bounded_interval, bounded_median = np.isfinite(hi), np.isfinite(median_upper)
    target = category_targets(remaining, edges)

    def bounded_mean(value, mask):
        return float(np.average(value[mask], weights=w[mask])) if mask.any() else None

    midpoint = (lower[q[1]] + median_upper) / 2
    return {
        "true_bucket_mean_probability": float(np.sum(w * p[np.arange(len(p)), target])),
        "interval90_coverage": float(
            np.sum(w * ((remaining >= lo) & (remaining <= hi)))
        ),
        "interval90_mean_finite_width_tokens": bounded_mean(hi - lo, bounded_interval),
        "interval90_open_tail_weight": float(w[~bounded_interval].sum()),
        "median_bucket_midpoint_mae_finite_tokens": bounded_mean(
            np.abs(midpoint - remaining), bounded_median
        ),
        "median_open_tail_weight": float(w[~bounded_median].sum()),
        "under_median_upper_rate": float(np.sum(w * (remaining > median_upper))),
    }


def capture_coverage(data: dict) -> dict:
    """Summarize actual EOS lengths separately from censoring and task intent."""
    result = {}
    for split in ("train", "validation", "test"):
        rows = [r for r in data["labels"] if r["split"] == split]
        lengths = [r["output_tokens"] for r in rows if r["natural_finish"]]
        result[split] = {
            "requests": len(rows),
            "natural_eos": len(lengths),
            "censored": len(rows) - len(lengths),
            "exact_output_length_counts": dict(
                Counter(
                    "0-2048"
                    if n <= 2048
                    else "2049-4096"
                    if n <= 4096
                    else "4097-8192"
                    if n <= 8192
                    else "8193-12288"
                    if n <= 12288
                    else "12289-16384"
                    if n <= 16384
                    else ">16384"
                    for n in lengths
                )
            ),
        }
    return result
