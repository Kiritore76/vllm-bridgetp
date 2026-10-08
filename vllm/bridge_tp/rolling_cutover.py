# SPDX-License-Identifier: Apache-2.0
"""Source-owned, pre-freeze boundary planning within a fixed KV reservation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MECHANISM = "ROLLING_NO_HISTORY_WAIT_V1"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def resident_progress(run_dir: Path, migration_id: str) -> tuple[bool, int | None]:
    """Use only four exact resident histories and contiguous applied deltas."""
    ends = []
    for rank in range(4):
        initial = read_json(run_dir / "gpu_initial_receipts" / f"tp_rank_{rank}.json")
        if (
            initial.get("migration_id") != migration_id
            or initial.get("status") != "INITIAL_HISTORY_GPU_RESIDENT"
            or initial.get("exact_readback") is not True
        ):
            return False, None
        end = initial.get("end_token")
        if type(end) is not int or end <= 0:
            return False, None
        delta = read_json(run_dir / "gpu_watermarks" / f"tp_rank_{rank}.json")
        if delta:
            if (
                delta.get("migration_id") != migration_id
                or delta.get("exact_readback") is not True
                or delta.get("status") not in {"STREAMING", "TARGET_READY"}
                or type(delta.get("end_token")) is not int
                or delta["end_token"] < end
            ):
                return False, None
            end = delta["end_token"]
        ends.append(end)
    return True, min(ends)


def applied_progress(run_dir: Path, migration_id: str, initial_end: int) -> int | None:
    """Read contiguous all-rank ACK receipts from the original sender path."""
    receipts = []
    for path in (run_dir / "gpu_direct_delta_sender_receipts").glob("*.json"):
        value = read_json(path)
        start, end = value.get("start_token"), value.get("end_token")
        if value.get("migration_id") != migration_id:
            continue
        if (value.get("status") != "APPLIED_ALL_RANKS"
                or type(start) is not int or type(end) is not int
                or not initial_end <= start < end):
            return None
        receipts.append((start, end))
    current = initial_end
    for start, end in sorted(receipts):
        if start != current:
            return None
        current = end
    return current if current > initial_end else None


def transport_lead_estimate(
    *,
    backlog_tokens: int,
    growth_tokens_s: float,
    records: tuple[dict[str, Any], ...],
    inflight: tuple[float, int] | None,
    unix_s: float,
) -> dict[str, Any]:
    """Estimate conservative drain time without waiting for a sender or GPU.

    Completed batch wall times include packing, receiver readiness and ACK.
    In-flight remaining time is an estimate, not a completion guarantee.
    """
    samples = [r for r in records[-8:]
               if r.get("tokens", 0) > 0 and r.get("transfer_ms", 0) > 0]
    if samples:
        service_s = max(r["transfer_ms"] for r in samples) / 1000
        rate = sum(r["tokens"] for r in samples) / sum(
            r["transfer_ms"] / 1000 for r in samples
        )
    else:
        # Bootstrap only until completed batches supply measured throughput.
        service_s, rate = 0.5, 32.0
    active_tokens = inflight[1] if inflight else 0
    elapsed = max(0.0, unix_s - inflight[0]) if inflight else 0.0
    # An overdue batch still needs margin; zero remaining would be optimistic.
    remaining = max(service_s - elapsed, service_s * 0.5) if inflight else 0.0
    queued = max(0, backlog_tokens - active_tokens)
    net_rate = rate - growth_tokens_s
    drain_s = remaining + queued / rate
    if net_rate > 0:
        catchup_s = (queued + remaining * growth_tokens_s) / net_rate
        estimated_s = remaining + catchup_s
    else:
        # Extending indefinitely cannot catch a producer faster than transport.
        estimated_s = drain_s
    horizon_s = min(8.0, 2.0 + 1.5 * estimated_s)
    return {
        "transport_samples": len(samples), "backlog_tokens": backlog_tokens,
        "inflight_tokens": active_tokens, "inflight_elapsed_s": elapsed,
        "inflight_remaining_estimate_s": remaining,
        "transport_tokens_s": rate, "net_catchup_tokens_s": net_rate,
        "estimated_catchup_s": estimated_s, "adaptive_horizon_s": horizon_s,
        "adaptive_horizon_capped": 2.0 + 1.5 * estimated_s > 8.0,
        "adaptive_lead_tokens": math.ceil(growth_tokens_s * horizon_s),
        "estimate_status": ("BOOTSTRAP" if not samples else
                            "CATCHUP" if net_rate > 0 else "DRAIN_AFTER_FREEZE"),
    }


@dataclass
class RollingPlanner:
    reservation_output_tokens: int
    minimum_lead_tokens: int = 128
    lead_seconds: float = 2.0
    maximum_plan_adjustments: int = 2
    boundary: int | None = None
    version: int = 0
    last_output: int | None = None
    last_unix_s: float | None = None
    growth_tokens_s: float = 0.0

    def observe(
        self,
        *,
        output_tokens: int,
        computed_tokens: int,
        resident_end: int | None,
        history_ready: bool,
        delta_applied: bool,
        unix_s: float,
        delta_records: tuple[dict[str, Any], ...] = (),
        delta_inflight: tuple[float, int] | None = None,
        initial_end: int | None = None,
    ) -> dict[str, Any] | None:
        """Plan or move a future boundary; never pause decoding to catch up."""
        if self.last_unix_s is not None and unix_s > self.last_unix_s:
            growth = max(0, output_tokens - self.last_output) / (
                unix_s - self.last_unix_s
            )
            self.growth_tokens_s = max(growth, self.growth_tokens_s * 0.95)
        self.last_output, self.last_unix_s = output_tokens, unix_s
        lag = (
            max(0, computed_tokens - resident_end) if resident_end is not None else None
        )
        ready = (
            history_ready
            and delta_applied
            and lag is not None
            and lag >= 0
            and resident_end <= computed_tokens
        )
        # This is a future cutover lead, not a START token threshold. Include
        # room for continued decoding and multiple applied-delta rounds.
        lead = max(
            self.minimum_lead_tokens,
            math.ceil(self.growth_tokens_s * self.lead_seconds),
        )
        backlog = max(0, computed_tokens - (
            resident_end if resident_end is not None
            else initial_end if initial_end is not None else computed_tokens
        ))
        timing = transport_lead_estimate(
            backlog_tokens=backlog, growth_tokens_s=self.growth_tokens_s,
            records=delta_records, inflight=delta_inflight, unix_s=unix_s,
        )
        lead = max(lead, timing["adaptive_lead_tokens"])
        if delta_records or delta_inflight or initial_end is not None:
            # Keep the base token cushion in addition to backlog drain margin.
            extra_s = min(6.0, 1.5 * timing["estimated_catchup_s"])
            lead = max(lead, self.minimum_lead_tokens + math.ceil(
                self.growth_tokens_s * extra_s
            ))
        reason = None
        if self.boundary is None and history_ready:
            reason = "FIRST_AFTER_HISTORY_RESIDENT"
        elif self.boundary is not None and output_tokens > self.boundary:
            # Only a skipped publication boundary can move an existing plan.
            # ACK lag and new timing estimates never move a published boundary.
            if self.version > self.maximum_plan_adjustments:
                return {"status": "RESERVATION_EXHAUSTED",
                        "version": self.version, "output_tokens": output_tokens,
                        "reason": "PLAN_ADJUSTMENT_LIMIT"}
            reason = "SAFE_PUBLICATION_BOUNDARY_MISSED"
        if output_tokens >= self.reservation_output_tokens - 1 and not ready:
            return {
                "status": "RESERVATION_EXHAUSTED",
                "version": self.version,
                "output_tokens": output_tokens,
                "delta_lag_tokens": lag,
            }
        if reason:
            candidate = min(output_tokens + lead, self.reservation_output_tokens - 1)
            if candidate <= output_tokens:
                return {
                    "status": "RESERVATION_EXHAUSTED",
                    "version": self.version,
                    "output_tokens": output_tokens,
                    "delta_lag_tokens": lag,
                }
            if self.boundary is not None and candidate <= self.boundary:
                return None
            self.boundary = candidate
            self.version += 1
            return {
                "status": "PLANNED",
                "version": self.version,
                "cutover_output_tokens": candidate,
                "reservation_output_tokens": self.reservation_output_tokens,
                "output_tokens": output_tokens,
                "computed_tokens": computed_tokens,
                "resident_end": resident_end,
                "delta_lag_tokens": lag,
                "history_ready": history_ready,
                "first_delta_applied": delta_applied,
                "growth_tokens_s": self.growth_tokens_s,
                "freeze_policy": "FREEZE_AT_BOUNDARY_DRAIN_TAIL",
                "maximum_pre_freeze_delta_lag_tokens": None,
                "requested_lead_tokens": lead,
                "transport_lead_estimate": timing,
                "lead_capped_by_reservation": candidate < output_tokens + lead,
                "plan_adjustments": self.version - 1,
                "maximum_plan_adjustments": self.maximum_plan_adjustments,
                "minimum_lead_tokens": self.minimum_lead_tokens,
                "lead_seconds": self.lead_seconds,
                "published_unix_s": unix_s,
                "reason": reason,
            }
        return None


def finalize_reserved_request(
    request: Any,
    cutover: dict[str, Any],
    *,
    reserved_known_tokens: int,
    total_output_budget: int,
) -> None:
    """Replace reserved placeholders before the scheduler promotes a request."""
    tokens = list(cutover["all_known_token_ids"])
    boundary = cutover["cutover_num_output_tokens"]
    computed = cutover["num_computed_tokens"]
    if (
        type(boundary) is not int
        or type(computed) is not int
        or not 0 < len(tokens) <= reserved_known_tokens
        or computed != len(tokens) - 1
        or len(tokens) != cutover["num_prompt_tokens"] + boundary
        or total_output_budget <= boundary
        or request.num_output_tokens != 0
    ):
        raise ValueError("rolling final token boundary differs from reservation")
    request.prompt_token_ids = tokens
    request._all_token_ids[:] = tokens
    request.num_prompt_tokens = len(tokens)
    request.num_computed_tokens = computed
    request.max_tokens = total_output_budget - boundary
    request.sampling_params.max_tokens = request.max_tokens
    request.block_hashes.clear()
    request.update_block_hashes()


def rolling_evidence_errors(
    run_dir: Path,
    session: dict[str, Any],
    cutover: dict[str, Any],
    freeze_unix_s: float,
) -> list[str]:
    """Validate reservation, plan versions, applied evidence and final remap."""
    errors = []
    try:
        r = read_json(run_dir / "rolling_reservation.json")
        s = read_json(run_dir / "rolling_freeze_selection.json")
        final = read_json(run_dir / "rolling_target_finalized.json")
        plans = [
            json.loads(line)
            for line in (run_dir / "rolling_source_plans.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        identity = session["migration_id"]
        boundary = cutover["cutover_num_output_tokens"]
        reserved = r["reservation_output_tokens"]
        if (
            r.get("mechanism") != MECHANISM
            or r.get("migration_id") != identity
            or r.get("source_request_id") != session["source_request_id"]
            or r["initial_end_token"] != session["num_computed_tokens"]
            or r["num_prompt_tokens"] != session["num_prompt_tokens"]
            or not boundary < reserved < r["source_max_output_tokens"]
        ):
            errors.append("rolling reservation identity or capacity is invalid")
        previous = 0
        previous_s = r["published_unix_s"]
        for version, p in enumerate(plans, 1):
            b = p["cutover_output_tokens"]
            if (
                p.get("mechanism") != MECHANISM
                or p.get("migration_id") != identity
                or p.get("source_request_id") != session["source_request_id"]
                or p.get("status") != "PLANNED"
                or p["version"] != version
                or p["reservation_output_tokens"] != reserved
                or not previous < b < reserved
                or not p["output_tokens"] < b
                or not previous_s <= p["published_unix_s"] <= freeze_unix_s
            ):
                errors.append("rolling plan version chain is invalid")
            previous = b
            previous_s = p["published_unix_s"]
        first = plans[0]
        if (
            first.get("reason")
            not in {
                "FIRST_AFTER_HISTORY_RESIDENT",
                "FIRST_AFTER_HISTORY_AND_DELTA_APPLIED",
            }
            or first.get("history_ready") is not True
        ):
            errors.append("rolling first plan lacks ready history")
        first_s = first["published_unix_s"]
        first_acks = [
            read_json(path)
            for path in (run_dir / "gpu_direct_delta_sender_receipts").glob("*.json")
        ]
        ack_deadline = (
            first_s
            if first.get("reason") == "FIRST_AFTER_HISTORY_AND_DELTA_APPLIED"
            else s["selected_unix_s"]
        )
        if not any(
            p.get("status") == "APPLIED_ALL_RANKS"
            and p.get("migration_id") == identity
            and p.get("start_token") == session["num_computed_tokens"]
            and p.get("end_token", 0) > session["num_computed_tokens"]
            and p.get("completed_unix_s", math.inf) <= ack_deadline
            for p in first_acks
        ):
            errors.append("rolling freeze preceded the first four-rank applied ACK")
        if (
            s.get("mechanism") != MECHANISM
            or s.get("migration_id") != identity
            or s.get("source_request_id") != session["source_request_id"]
            or s.get("history_ready") is not True
            or s.get("first_delta_applied") is not True
            or s.get("version") != len(plans)
            or s.get("cutover_output_tokens") != boundary
            or previous != boundary
            or s.get("computed_tokens") != cutover["num_computed_tokens"]
            or s["delta_lag_tokens"] < 0
            or s["computed_tokens"] - s["resident_end"] != s["delta_lag_tokens"]
            or not first_s <= s["selected_unix_s"] <= freeze_unix_s
        ):
            errors.append("rolling freeze lacks a matching pre-freeze applied proof")
        for rank in range(4):
            h = read_json(run_dir / "gpu_initial_receipts" / f"tp_rank_{rank}.json")
            if (
                h.get("migration_id") != identity
                or h.get("exact_readback") is not True
                or h.get("status") != "INITIAL_HISTORY_GPU_RESIDENT"
                or h.get("resident_completed_unix_s", math.inf) > first_s
            ):
                errors.append(
                    f"rolling rank {rank} history was not ready before planning"
                )
            applied = [
                read_json(path)
                for path in (run_dir / "gpu_delta_receipts" / f"tp_rank_{rank}").glob(
                    "*.json"
                )
            ]
            known_end = max(
                [session["num_computed_tokens"]]
                + [
                    p["end_token"]
                    for p in applied
                    if p.get("exact_readback") is True
                    and p.get("migration_id") == identity
                    and p.get("completed_unix_s", math.inf) <= s["selected_unix_s"]
                ]
            )
            if known_end < s["resident_end"]:
                errors.append(
                    f"rolling rank {rank} pre-freeze delta proof is incomplete"
                )
        if (
            final.get("migration_id") != identity
            or final.get("cutover_output_tokens") != boundary
            or final.get("reserved_known_tokens")
            != session["num_prompt_tokens"] + reserved
            or final.get("num_tokens") != len(cutover["all_known_token_ids"])
            or final.get("num_computed_tokens") != cutover["num_computed_tokens"]
            or final.get("max_tokens") != r["total_output_budget"] - boundary
        ):
            errors.append("rolling target final token/budget remap is invalid")
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        errors.append("rolling cutover evidence is missing or malformed")
    return errors
