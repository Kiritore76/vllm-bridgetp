# SPDX-License-Identifier: Apache-2.0
"""Pure numerical-fidelity helpers for Phase 9 D-0 and D-3.

These helpers quantify evidence. They deliberately do not convert an
uncalibrated ULP or KV threshold into a claim that migration is blameless.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean, median

_MANTISSA_BITS = {
    "bfloat16": 7,
    "bf16": 7,
    "float16": 10,
    "fp16": 10,
    "half": 10,
    "float32": 23,
    "fp32": 23,
    "float": 23,
}


def normalize_dtype(dtype: str) -> str:
    key = str(dtype).lower().replace("torch.", "")
    aliases = {"bf16": "bfloat16", "fp16": "float16", "half": "float16"}
    aliases.update({"fp32": "float32", "float": "float32"})
    key = aliases.get(key, key)
    if key not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported floating dtype: {dtype!r}")
    return key


def mantissa_bits(dtype: str) -> int:
    return _MANTISSA_BITS[normalize_dtype(dtype)]


def ulp_at(value: float, dtype: str) -> float:
    """Return normal-number spacing near ``value`` for the recorded dtype."""
    magnitude = abs(float(value))
    if not math.isfinite(magnitude) or magnitude == 0.0:
        raise ValueError("ULP is undefined at zero, NaN, or infinity")
    exponent = math.floor(math.log2(magnitude))
    return 2.0 ** (exponent - mantissa_bits(dtype))


def gap_in_ulps(left: float, right: float, dtype: str) -> float:
    scale = max(abs(float(left)), abs(float(right)))
    if scale == 0.0:
        return 0.0
    return abs(float(left) - float(right)) / ulp_at(scale, dtype)


@dataclass(frozen=True)
class CandidateGap:
    stage: str
    dtype: str
    first_token_id: int
    second_token_id: int
    first_value: float
    second_value: float
    absolute_gap: float
    gap_ulps: float
    descriptive_band: str

    def to_json(self) -> dict:
        return {
            "stage": self.stage,
            "dtype": self.dtype,
            "first_token_id": self.first_token_id,
            "second_token_id": self.second_token_id,
            "first_value": self.first_value,
            "second_value": self.second_value,
            "absolute_gap": self.absolute_gap,
            "gap_ulps": self.gap_ulps,
            "descriptive_band": self.descriptive_band,
        }


def analyze_candidate_gap(
    *,
    stage: str,
    dtype: str,
    first_token_id: int,
    second_token_id: int,
    first_value: float,
    second_value: float,
) -> CandidateGap:
    """Describe a measured candidate gap without assigning causality."""
    normalized = normalize_dtype(dtype)
    ulps = gap_in_ulps(first_value, second_value, normalized)
    if ulps <= 1.0:
        band = "WITHIN_ONE_RECORDED_DTYPE_ULP"
    elif ulps <= 2.0:
        band = "WITHIN_TWO_RECORDED_DTYPE_ULPS"
    else:
        band = "MORE_THAN_TWO_RECORDED_DTYPE_ULPS"
    return CandidateGap(
        stage=stage,
        dtype=normalized,
        first_token_id=int(first_token_id),
        second_token_id=int(second_token_id),
        first_value=float(first_value),
        second_value=float(second_value),
        absolute_gap=abs(float(first_value) - float(second_value)),
        gap_ulps=ulps,
        descriptive_band=band,
    )


def agreement_length(
    left: list[int], right: list[int], budget: int | None = None
) -> int:
    limit = min(len(left), len(right))
    if budget is not None:
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        limit = min(limit, budget)
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sample")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_samples(values: list[int], budget: int | None = None) -> dict:
    if not values:
        raise ValueError("no agreement samples")
    numeric = [int(value) for value in values]
    result = {
        "n": len(numeric),
        "min": min(numeric),
        "p25": _quantile(numeric, 0.25),
        "median": median(numeric),
        "mean": mean(numeric),
        "p75": _quantile(numeric, 0.75),
        "max": max(numeric),
    }
    if budget is not None:
        full = sum(value >= budget for value in numeric)
        result.update(
            {
                "budget": budget,
                "fully_agreeing": full,
                "fully_agreeing_fraction": full / len(numeric),
            }
        )
    return result


def paired_bootstrap_mean_difference(
    migrated: list[int],
    topology: list[int],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 17,
) -> dict:
    """Bootstrap the paired mean of ``migrated - topology``."""
    if len(migrated) != len(topology) or not migrated:
        raise ValueError("paired samples must be nonempty and have equal length")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    differences = [int(left) - int(right) for left, right in zip(migrated, topology)]
    generator = random.Random(seed)
    n = len(differences)
    bootstrap = [
        mean(differences[generator.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimand": "paired_mean_agreement_difference_migrated_minus_topology",
        "estimate": mean(differences),
        "confidence": confidence,
        "ci_low": _quantile(bootstrap, alpha),
        "ci_high": _quantile(bootstrap, 1.0 - alpha),
        "resamples": resamples,
        "seed": seed,
        "paired_differences": differences,
    }
