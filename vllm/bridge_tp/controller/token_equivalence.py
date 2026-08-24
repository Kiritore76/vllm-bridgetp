# SPDX-License-Identifier: Apache-2.0
"""Classify greedy counterfactual token divergence without hiding failures.

An exact TP1-control match remains the strongest result.  A mismatch can only
be classified as an equal-logit tie when raw TP1 and TP4 topology probes both
place the two emitted token strings at the maximum logprob.  Missing evidence
is a failure, never an implicit tie.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenObservation:
    """One chosen token and the top-logprob map at that decode position."""

    token_id: int
    token_text: str
    top_logprobs: dict[str, float]


def first_divergence(left: list[int], right: list[int]) -> int | None:
    """Return the first unequal index, including a length-only mismatch."""
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if int(left_id) != int(right_id):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def completion_observations(response: dict[str, Any]) -> list[TokenObservation]:
    """Extract aligned token/logprob records from a non-stream response."""
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError("expected exactly one completion choice")
    choice = choices[0]
    token_ids = choice.get("token_ids") or []
    logprobs = choice.get("logprobs") or {}
    return _aligned_observations(token_ids, logprobs)


def streaming_observations(response: dict[str, Any]) -> list[TokenObservation]:
    """Extract aligned token/logprob records from saved SSE chunks."""
    observations: list[TokenObservation] = []
    for chunk in response.get("chunks") or []:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        token_ids = choice.get("token_ids") or []
        if not token_ids:
            continue
        observations.extend(
            _aligned_observations(token_ids, choice.get("logprobs") or {})
        )
    return observations


def classify_token_equivalence(
    *,
    emitted: list[int],
    control: list[int],
    emitted_rows: list[dict[str, Any]],
    cutover_index: int,
    target_response: dict[str, Any] | None,
    control_response: dict[str, Any] | None,
    token_text_map: dict[str, Any] | None,
    probe_request: dict[str, Any] | None,
    tp1_probe: dict[str, Any] | None,
    tp4_probe: dict[str, Any] | None,
    expected_probe_prompt: list[int] | None,
    tie_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Return an auditable exact/tie/failure classification.

    Only the first mismatch is classified.  Once two equally maximal greedy
    tokens differ, later counterfactual tokens have different contexts and are
    no longer meaningful pairwise comparisons.
    """
    if tie_epsilon < 0 or tie_epsilon > 1e-3:
        raise ValueError("tie_epsilon must be in [0, 1e-3]")
    divergence = first_divergence(emitted, control)
    base: dict[str, Any] = {
        "format_version": 1,
        "exact_token_match": divergence is None,
        "first_divergence_index": divergence,
        "tie_epsilon": tie_epsilon,
    }
    if divergence is None:
        return {
            **base,
            "classification": "EXACT",
            "tie_certified": False,
            "acceptable": True,
            "reason": "unified response is token-identical to TP1 control",
        }
    if divergence >= len(emitted) or divergence >= len(control):
        return _unproven(base, "token sequence lengths differ")
    if divergence < cutover_index:
        return _unproven(base, "divergence occurred before TP4 cutover")
    if divergence >= len(emitted_rows):
        return _unproven(base, "unified emitted-row evidence is incomplete")
    row = emitted_rows[divergence]
    if row.get("origin") != "target" or int(row.get("index", -1)) != divergence:
        return _unproven(base, "first divergence is not a target-origin token")
    required = (
        target_response,
        control_response,
        token_text_map,
        probe_request,
        tp1_probe,
        tp4_probe,
        expected_probe_prompt,
    )
    if any(value is None for value in required):
        return _unproven(base, "raw logprob or topology-probe evidence is missing")

    try:
        tp1_records = completion_observations(tp1_probe or {})
        tp4_records = completion_observations(tp4_probe or {})
    except (KeyError, TypeError, ValueError) as error:
        return _unproven(base, f"invalid logprob evidence: {error}")
    target_ids = [
        int(value) for value in (target_response or {}).get("token_ids") or []
    ]
    control_choices = (control_response or {}).get("choices") or []
    if len(control_choices) != 1:
        return _unproven(base, "control response does not have exactly one choice")
    control_ids = [
        int(value) for value in control_choices[0].get("token_ids") or []
    ]
    target_index = divergence - cutover_index
    if target_index >= len(target_ids) or divergence >= len(control_ids):
        return _unproven(base, "response token evidence does not reach divergence")
    if len(tp1_records) != 1 or len(tp4_records) != 1:
        return _unproven(base, "topology probes must each emit exactly one token")

    target_token_id = target_ids[target_index]
    control_token_id = control_ids[divergence]
    if target_token_id != int(emitted[divergence]):
        return _unproven(base, "target response token does not match unified stream")
    if control_token_id != int(control[divergence]):
        return _unproven(base, "control response token does not match control IDs")
    text_map = token_text_map or {}
    if int(text_map.get("control_token_id", -1)) != control_token_id or int(
        text_map.get("target_token_id", -1)
    ) != target_token_id:
        return _unproven(base, "token-text map IDs do not match the divergence")
    mapped = text_map.get("tokens") or {}
    control_token_text = mapped.get(str(control_token_id))
    target_token_text = mapped.get(str(target_token_id))
    if not isinstance(control_token_text, str) or not isinstance(
        target_token_text, str
    ):
        return _unproven(base, "token-text map is incomplete")
    if target_token_text == control_token_text:
        return _unproven(base, "different token IDs decode to the same probe key")

    request = probe_request or {}
    prompt = request.get("prompt")
    if not isinstance(prompt, list) or [int(x) for x in prompt] != list(
        expected_probe_prompt or []
    ):
        return _unproven(base, "topology probe prompt is not the common prefix")
    if int(request.get("max_tokens", 0)) != 1 or float(
        request.get("temperature", -1)
    ) != 0.0:
        return _unproven(base, "topology probe is not one-token greedy decode")

    tie_details = []
    for name, probe in (("tp1", tp1_records[0]), ("tp4", tp4_records[0])):
        top = probe.top_logprobs
        if not top:
            return _unproven(base, f"{name} probe has no top-logprob map")
        maximum = max(top.values())
        for token_text in (
            control_token_text,
            target_token_text,
        ):
            value = top.get(token_text)
            if value is None or maximum - value > tie_epsilon:
                return _unproven(
                    base,
                    f"{token_text!r} is not tied for maximum in {name} probe",
                )
        tie_details.append(
            {
                "topology": name,
                "chosen_token_id": probe.token_id,
                "chosen_token_text": probe.token_text,
                "maximum_logprob": maximum,
                "control_token_logprob": top[control_token_text],
                "target_token_logprob": top[target_token_text],
            }
        )
    return {
        **base,
        "classification": "TIE_EQUIVALENT_DIVERGENCE",
        "tie_certified": True,
        "acceptable": True,
        "reason": "first mismatch is an equal-maximum greedy tie on TP1 and TP4",
        "control_token_id": control_token_id,
        "control_token_text": control_token_text,
        "target_token_id": target_token_id,
        "target_token_text": target_token_text,
        "probe_details": tie_details,
    }


def _aligned_observations(
    token_ids: list[Any],
    logprobs: dict[str, Any],
) -> list[TokenObservation]:
    tokens = logprobs.get("tokens") or []
    tops = logprobs.get("top_logprobs") or []
    if not (len(token_ids) == len(tokens) == len(tops)):
        raise ValueError("token IDs and logprob arrays are not aligned")
    observations = []
    for token_id, token_text, top in zip(token_ids, tokens, tops):
        if not isinstance(top, dict):
            raise ValueError("top_logprobs entry is not an object")
        observations.append(
            TokenObservation(
                token_id=int(token_id),
                token_text=str(token_text),
                top_logprobs={str(key): float(value) for key, value in top.items()},
            )
        )
    return observations


def _unproven(base: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **base,
        "classification": "UNPROVEN_DIVERGENCE",
        "tie_certified": False,
        "acceptable": False,
        "reason": reason,
    }
