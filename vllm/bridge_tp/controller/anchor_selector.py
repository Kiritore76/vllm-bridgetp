# SPDX-License-Identifier: Apache-2.0
"""Select one migration anchor from a scheduler batch."""

from __future__ import annotations


def select_source_request_id(
    request_ids: list[str],
    configured_prefix: str,
) -> str | None:
    """Select the anchor without inspecting future requests or outputs.

    vLLM appends an internal suffix to the OpenAI ``request_id``.  The pilot
    therefore accepts either an exact match or ``<external-id>-...``.  With no
    configured prefix this deliberately retains the original Phase 6-8 rule:
    publish only when the scheduler batch contains exactly one request.
    """
    if not configured_prefix:
        return request_ids[0] if len(request_ids) == 1 else None
    def matches_prefix(request_id: str) -> bool:
        # The OpenAI completions frontend exposes an external request ID such
        # as ``bridgetp-phase9-...`` to the client but the engine may wrap it
        # as ``cmpl-bridgetp-phase9-...``.  Match the configured external
        # prefix against either representation while retaining the suffix
        # allowance used by other frontends.
        candidates = [request_id]
        if request_id.startswith("cmpl-"):
            candidates.append(request_id.removeprefix("cmpl-"))
        return any(
            candidate == configured_prefix
            or candidate.startswith(configured_prefix + "-")
            for candidate in candidates
        )

    matches = [request_id for request_id in request_ids if matches_prefix(request_id)]
    if len(matches) > 1:
        raise RuntimeError(
            "BRIDGETP_STREAM_SOURCE_REQUEST_ID_PREFIX is ambiguous: "
            f"{configured_prefix!r} matched {matches!r}"
        )
    return matches[0] if matches else None
