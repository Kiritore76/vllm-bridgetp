# SPDX-License-Identifier: Apache-2.0
"""Fail-closed deterministic sampling contract for Phase 9 evidence.

Model ``generation_config.json`` files may provide non-neutral sampling
defaults even when a request explicitly sets ``temperature=0``.  Qwen 2.5,
for example, supplies a repetition penalty.  Phase 9 compares tokens and raw
logprobs across TP topologies, so every request participating in that evidence
must explicitly disable all sampling transforms that can change argmax.
"""

from __future__ import annotations

from typing import Any

STRICT_GREEDY_SAMPLING_CONTRACT: dict[str, float | int | bool] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "n": 1,
    "use_beam_search": False,
}

_NEUTRAL_OPTIONAL_FIELDS: dict[str, tuple[Any, ...]] = {
    "logit_bias": (None, {}),
    "allowed_token_ids": (None, []),
    "structured_outputs": (None, {}),
}


def freeze_strict_greedy_sampling(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a request with an explicit, deterministic greedy contract.

    Missing contract fields are filled in.  Explicit conflicting fields or
    non-neutral logits constraints are rejected instead of silently changed.

    Args:
        payload: OpenAI-compatible completion request.

    Returns:
        A shallow copy containing the complete strict-greedy contract.

    Raises:
        ValueError: If the input explicitly requests incompatible sampling.
    """
    errors = strict_greedy_sampling_errors(payload, require_explicit=False)
    if errors:
        raise ValueError("invalid Phase 9 sampling contract: " + "; ".join(errors))
    frozen = dict(payload)
    frozen.update(STRICT_GREEDY_SAMPLING_CONTRACT)
    return frozen


def strict_greedy_sampling_errors(
    payload: dict[str, Any],
    *,
    require_explicit: bool = True,
) -> list[str]:
    """Describe every way a request violates the Phase 9 contract."""
    errors: list[str] = []
    for key, expected in STRICT_GREEDY_SAMPLING_CONTRACT.items():
        if key not in payload:
            if require_explicit:
                errors.append(f"{key} is missing")
            continue
        actual = payload[key]
        if isinstance(expected, float):
            try:
                matches = float(actual) == expected
            except (TypeError, ValueError):
                matches = False
        elif isinstance(expected, bool):
            matches = isinstance(actual, bool) and actual is expected
        else:
            matches = isinstance(actual, int) and not isinstance(actual, bool)
            matches = matches and actual == expected
        if not matches:
            errors.append(f"{key}={actual!r}, expected {expected!r}")

    for key, neutral_values in _NEUTRAL_OPTIONAL_FIELDS.items():
        if key in payload and payload[key] not in neutral_values:
            errors.append(f"{key} must be absent or neutral")
    return errors


def strict_greedy_sampling_detail(payload: dict[str, Any]) -> str:
    """Render a compact inspector detail for one request artifact."""
    errors = strict_greedy_sampling_errors(payload)
    return "strict greedy" if not errors else "; ".join(errors)
