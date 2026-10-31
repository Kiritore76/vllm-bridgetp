# SPDX-License-Identifier: Apache-2.0
"""S4: read-only (SHADOW) planner inside a live DRAIN conversion.

Each snapshot turns the controller's ``layout_status.json`` rows of the
converted bundle into ``RemainingRequest`` inputs and asks the bounded prefix
planner what it would do. Nothing is executed: the record is the output.

- Remaining output: the per-request bins the controller publishes (hard cap
  max_output - generated, tightened by a fresh predictor distribution), each
  mass at its upper edge, divided by a conservative speed
  (observed tokens/s x ``speed_margin``).
- Transfer: the measured START-to-source-release budget the controller uses.
- Stages after the clear: the frozen conversion profile (rebuild, release,
  channel retirement, controller apply), valid only on its platform shape.
- Migration exists only toward the surviving pair target TP2 (a TP1 bundle
  merging); assisted split (TP2 -> TP1) is unsupported and reported so.

``MECHANISM_ONLY`` picks the earliest estimated READY; ``NET_UTILITY`` needs
matched utility profiles that do not exist yet, so it must stay UNKNOWN.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tools.bridge_tp.pressure_cpu import load

planner = load("reconfiguration_planner")

ROOT = Path(__file__).resolve().parents[2]
PROFILE = (
    ROOT / "experiments/platforms/qwen3_pressure_observation/conversion_profile.json"
)


def load_profile(path=PROFILE):
    value = json.loads(Path(path).read_text("utf-8"))
    if value.get("schema") != "BRIDGETP_CONVERSION_PROFILE_V1":
        raise ValueError("unknown conversion profile")
    return value


def profile_mismatch(profile, rows, target_mode):
    """Why the profile does not describe this conversion (None if it does)."""
    if target_mode not in profile["stage_upper_s"]["rebuild"]:
        return f"no rebuild profile for {target_mode}"
    if any(
        profile["context_limits"].get("tp1" if r["tp_degree"] == 1 else "tp2")
        != r["context_limit"]
        for r in rows
    ):
        return "context limits differ from the profile"
    gpus = sum(len(r.get("physical_gpu_ids", ())) for r in rows)
    if gpus != profile["bundle_gpus"]:
        return "bundle size differs from the profile"
    return None


def remaining_request(row, *, migratable_target, release_budget_s, speed_margin):
    """One planner input; unknown speed keeps the request's time unknown."""
    speed = row.get("tokens_s")
    bins = row["remaining_bins"]
    if not bins or any(u < 0 or m < 0 for u, m in bins):
        raise ValueError("invalid remaining bins")
    if speed is None or not math.isfinite(speed) or speed <= 0:
        pmf = natural = None
    else:
        lower = speed * speed_margin
        pmf = tuple((u / lower, m) for u, m in bins)
        total = sum(m for _, m in pmf)
        pmf = tuple((t, m / total) for t, m in pmf)  # rounding of published masses
        natural = max(t for t, _ in pmf)
    migratable = bool(
        migratable_target
        and row.get("action_available")
        and row.get("transfer_bytes") is not None
        and release_budget_s is not None
    )
    return planner.RemainingRequest(
        request_id=row["request_id"],
        natural_s=natural,
        duration_pmf=pmf,
        migratable=migratable,
        transfer_s=release_budget_s if migratable else None,
        transfer_bytes=row.get("transfer_bytes") if migratable else None,
    )


class ShadowPlanner:
    """Called by the conversion backend with each fresh controller status."""

    def __init__(self, profile, manifest_rows):
        self.profile = profile
        self.rows = {r["instance_id"]: r for r in manifest_rows}

    def __call__(self, plan, status, now):
        refs = [r["instance_id"] for r in plan["refs"]]
        target_mode = plan["target_bundle_mode"]
        mismatch = profile_mismatch(
            self.profile, [self.rows[i] for i in refs], target_mode
        )
        stages = self.profile["stage_upper_s"]
        after_clear = (
            None
            if mismatch
            else stages["controller_apply"]
            + stages["channel_retire"]
            + stages["stop_to_released"]
            + stages["rebuild"][target_mode]
        )
        target = status.get("migration_target")
        instances = status["instances"]
        merge = target_mode != "TP1"
        migratable_target = bool(
            merge
            and target is not None
            and target not in refs
            and instances.get(target, {}).get("lifecycle") == "ACTIVE"
        )
        rows = [dict(r, instance=i) for i in refs for r in instances[i]["requests"]]
        requests = [
            remaining_request(
                r,
                migratable_target=migratable_target,
                release_budget_s=status.get("release_budget_s"),
                speed_margin=self.profile["speed_margin"],
            )
            for r in rows
        ]
        sessions = any(instances[i]["sessions"] for i in refs) or bool(
            target and instances.get(target, {}).get("sessions")
        )
        busy = status.get("release_budget_s") if sessions else 0
        fresh = 0 <= now - status["unix_s"] <= 2
        drain_only = planner.timeline(requests, (), after_clear)
        record = dict(
            snapshot_unix_s=status["unix_s"],
            evaluated_unix_s=now,
            plan_id=plan["plan_id"],
            target_mode=target_mode,
            profile_mismatch=mismatch,
            migration_target=target if migratable_target else None,
            requests=[
                dict(
                    request_id=q.request_id,
                    instance=r["instance"],
                    remaining_basis=r["remaining_basis"],
                    natural_s=q.natural_s,
                    migratable=q.migratable,
                )
                for q, r in zip(requests, rows)
            ],
            drain_only=dict(
                support_status=drain_only["support_status"],
                estimated_clear_s=drain_only["estimated_clear_time"],
                estimated_ready_s=drain_only["estimated_ready_time"],
                missing_inputs=drain_only["missing_inputs"],
            ),
        )
        if not merge:
            return dict(
                record,
                mechanism=dict(decision="UNSUPPORTED", reason="assisted split"),
                net_utility=dict(decision="UNSUPPORTED", reason="assisted split"),
            )
        results = {}
        for objective in ("MECHANISM_ONLY", "NET_UTILITY"):
            manual = planner.ManualPlan(
                plan_id=plan["plan_id"],
                bundle_id=plan["bundle_id"],
                target_tp=int(target_mode[2:]),
                objective_mode=objective,
                execution_mode="SHADOW",
                deadline=plan["deadline"],
                utility_end=plan["deadline"],
            )
            value = planner.plan_next(
                manual,
                snapshot_id=f"{status['unix_s']}",
                now=now,
                requests=requests,
                rebuild_s=after_clear,
                profile_hashes={"conversion_profile": self.profile["schema"]},
                resources_safe=fresh,
                channel_busy_s=busy,
            )
            results[objective] = {
                k: value.get(k)
                for k in (
                    "support_status",
                    "decision",
                    "reason",
                    "ordered_migration_set",
                    "estimated_clear_time",
                    "estimated_ready_time",
                    "missing_inputs",
                )
            }
        mechanism = results["MECHANISM_ONLY"]
        saving = (
            drain_only["estimated_clear_time"] - mechanism["estimated_clear_time"]
            if drain_only["estimated_clear_time"] is not None
            and mechanism["estimated_clear_time"] is not None
            else None
        )
        return dict(
            record,
            mechanism=dict(mechanism, predicted_clear_saving_s=saving),
            net_utility=results["NET_UTILITY"],
        )
