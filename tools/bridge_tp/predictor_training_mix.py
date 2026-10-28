"""Balance historical replay and generation stages without split leakage."""

from collections import Counter

import numpy as np
from predictor_distribution import category_targets, request_weights


def mixed_training_weights(
    data: dict,
    legacy_ids: set[str] | None,
    *,
    legacy_mass: float = 0.25,
    prefill_mass: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Balance source cohorts and stages; None selects one request-equal cohort.

    Each request has equal mass within its source cohort. Exact requests use
    actual generation progress: early <25%, middle 25-75%, late >=75%.
    With positive prefill_mass, one explicit PREFILL_COMPLETE state receives
    that fraction of each request; remaining decode mass follows 30/40/30.
    Missing stages redistribute decode mass over present stages. Censored
    requests have unknown final length and distribute decode mass uniformly.
    Prefill-only requests retain their full request mass. The zero default
    preserves the original weighting, including flat censored requests.
    Only original training splits receive positive loss weights.
    """
    if not np.isfinite(legacy_mass) or not 0 < legacy_mass < 1:
        raise ValueError("legacy mass must be finite and strictly between 0 and 1")
    if not np.isfinite(prefill_mass) or not 0 <= prefill_mass < 1:
        raise ValueError("prefill mass must be finite and in [0,1)")
    train = data["splits"] == "train"
    weights = np.zeros(len(train), dtype=np.float64)
    stage_mass = np.array([0.3, 0.4, 0.3])
    ids = data["requests"]
    phases = data.get("phases")
    if prefill_mass and (phases is None or np.asarray(phases).shape != ids.shape):
        raise ValueError("prefill weighting requires one phase per sample")
    if prefill_mass:
        phases = np.asarray(phases)
    cohorts = (
        {"all_requests": train}
        if legacy_ids is None
        else {
            "legacy5000": train & np.isin(ids, sorted(legacy_ids)),
            "new300_and_first60": train & ~np.isin(ids, sorted(legacy_ids)),
        }
    )
    cohort_masses = (1.0,) if legacy_ids is None else (legacy_mass, 1 - legacy_mass)
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
    for (name, mask), mass in zip(cohorts.items(), cohort_masses):
        members = sorted(set(ids[mask]))
        if not members:
            raise ValueError(f"empty training cohort: {name}")
        prefill_only = 0
        for key in members:
            indices = np.asarray(by_id[str(key)])
            request_mass = mass / len(members)
            censor = data["censored"][indices]
            if censor.any() != censor.all():
                raise ValueError("request mixes exact and censored states")
            decode_mass = request_mass
            if prefill_mass:
                prefill = indices[phases[indices] == "PREFILL_COMPLETE"]
                if len(prefill) != 1 or data["generated"][prefill[0]] != 0:
                    raise ValueError(
                        f"request {key} requires one prefill state at generated=0"
                    )
                indices = indices[phases[indices] != "PREFILL_COMPLETE"]
                if not len(indices):
                    weights[prefill] = request_mass
                    prefill_only += 1
                    continue
                if (data["generated"][indices] <= 0).any():
                    raise ValueError("decode states must have positive generated count")
                weights[prefill] = request_mass * prefill_mass
                decode_mass = request_mass * (1 - prefill_mass)
            if censor.all():
                weights[indices] = decode_mass / len(indices)
                continue
            present, counts = np.unique(stages[indices], return_counts=True)
            normalized = stage_mass[present] / stage_mass[present].sum()
            for stage, count, share in zip(present, counts, normalized):
                rows = indices[stages[indices] == stage]
                weights[rows] = decode_mass * share / count
        report[name] = {
            "training_requests": len(members),
            "samples": int(mask.sum()),
            "loss_mass": float(weights[mask].sum()),
        }
        if prefill_mass:
            prefill_rows = mask & (phases == "PREFILL_COMPLETE")
            report[name]["prefill_loss_mass"] = float(weights[prefill_rows].sum())
            report[name]["prefill_only_requests"] = prefill_only
    return weights, {
        "cohorts": report,
        "generation_stages": ["0-25%", "25-75%", "75-100%"],
        "stage_mass": stage_mass.tolist(),
        "prefill_mass_per_request": prefill_mass,
        "prefill_policy": "independent from decode"
        if prefill_mass
        else "included in early stage; no independent reservation",
        "decode_stage_mass_per_request": ((1 - prefill_mass) * stage_mass).tolist()
        if prefill_mass
        else None,
        "prefill_only_rule": "full request mass; no decode states to allocate",
        "censored_rule": "independent prefill; uniform decode; unknown final progress"
        if prefill_mass
        else "flat within request; final progress is unknown",
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
