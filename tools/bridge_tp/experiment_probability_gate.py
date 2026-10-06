# SPDX-License-Identifier: Apache-2.0
"""F2 probability crossing, episode assignment, and safety audit."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


def source_load_eligibility(snapshot: dict, minimum_running: int) -> tuple[str, ...]:
    """Select the planned source-load stratum without changing physical safety."""
    if minimum_running == 0:
        return ()
    errors = []
    running = snapshot.get("source_running")
    if running is None or running < minimum_running:
        errors.append("planned source concurrency not active")
    if snapshot.get("source_prefill_pending_kv_tokens") != 0:
        errors.append("source prefill reservation not drained")
    return tuple(errors)


@dataclass
class ProbabilityGate:
    threshold: float
    assignment_seed: str
    start_probability: float = 0.5
    assigned_action: str | None = None
    thresholds: tuple[float, ...] = ()
    crossings: dict[float, dict] = field(default_factory=dict)
    candidate: dict | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1 or not 0 <= self.start_probability <= 1:
            raise ValueError("probability gate values must be in [0,1]")
        if self.assigned_action not in {None, "START", "STAY"}:
            raise ValueError("assignment must be START/STAY or randomized")
        if any(not 0 <= x <= 1 for x in self.thresholds):
            raise ValueError("invalid threshold family")

    def observe(
        self,
        snapshot: dict,
        tick: int,
        eligibility_errors: tuple[str, ...] = (),
    ) -> dict:
        bounds = snapshot.get("p_guard_est_bounds")
        valid = snapshot.get("status") == "VALID" and bounds is not None
        crossed = []
        if valid:
            for theta in sorted(set((self.threshold,) + self.thresholds)):
                if bounds[0] >= theta and theta not in self.crossings:
                    self.crossings[theta] = {"tick": tick, "snapshot": snapshot}
                    crossed.append(theta)
        feasible = snapshot.get("physical_feasible") is True
        first = False
        if (
            valid
            and bounds[0] >= self.threshold
            and feasible
            and not eligibility_errors
            and self.candidate is None
        ):
            first = True
            # Draw after selecting the common feasible candidate; STAY sees
            # exactly the same gate. Complementary paired assignments are
            # pre-registered by the harness with probability 1/2.
            if self.assigned_action is None:
                self.assigned_action = (
                    "START"
                    if random.Random(self.assignment_seed).random()
                    < self.start_probability
                    else "STAY"
                )
            self.candidate = {"tick": tick, "snapshot": snapshot}
        safety = snapshot.get("status") == "PROTECTION_BAND" or (
            snapshot.get("S_bounds_s") is not None and snapshot["S_bounds_s"][0] <= 0
        )
        start = bool(
            self.candidate
            and self.assigned_action == "START"
            and feasible
            and not eligibility_errors
        )
        reason = (
            "CAPACITY_PROTECTION_REQUIRED"
            if safety
            else "LOAD_STRATUM_NOT_READY"
            if eligibility_errors
            else "ASSIGNED_START"
            if start
            else "EPISODE_STAY"
            if self.candidate
            else "PHYSICAL_REFUSAL"
            if valid and bounds[0] >= self.threshold
            else "BELOW_THRESHOLD"
            if valid
            else "INVALID_RISK"
        )
        return {
            "kind": "experiment_probability_gate",
            "format_version": 1,
            "tick": tick,
            "threshold": self.threshold,
            "first_crossings_this_tick": crossed,
            "simultaneous_crossing": len(crossed) > 1,
            "first_feasible_candidate": first,
            "candidate_tick": (self.candidate or {}).get("tick"),
            "assigned_action": self.assigned_action,
            "assignment_probability": (
                self.start_probability
                if self.assigned_action == "START"
                else 1 - self.start_probability
            ),
            "assignment_seed": self.assignment_seed,
            "requested_action": "START_SHADOW" if start else "STAY",
            "actual_action": "PENDING_EXECUTOR" if start else "STAY",
            "reason": reason,
            "safety_override": False,
            "safety_protection_required": safety,
            "physical_rejections": snapshot.get("physical_rejections"),
            "experimental_eligibility_errors": list(eligibility_errors),
            "snapshot": snapshot,
        }
