# SPDX-License-Identifier: Apache-2.0
"""Commit at the first safe boundary after TP4 history becomes ready.

The dormant target is sized for one immutable output boundary.  The base
candidate includes the delta backlog and admission safety lead; M3 adds no
benefit-based delay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class M3CommitDecision:
    action: str
    candidate_output_tokens: int
    reason: str
    def to_json(self) -> dict[str, object]:
        return asdict(self)


class M3CommitController:
    def plan_candidate(
        self,
        *,
        output_tokens: int,
        base_candidate: int,
        max_output_tokens: int,
    ) -> M3CommitDecision:
        """Keep the earliest safe candidate selected by the data plane."""
        if not output_tokens < base_candidate < max_output_tokens:
            raise ValueError("M3 base candidate is outside the remaining output")
        return M3CommitDecision(
            "COMMIT_EARLIEST", base_candidate,
            "commit at the first safe boundary after history readiness",
        )
