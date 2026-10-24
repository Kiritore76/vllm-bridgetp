# SPDX-License-Identifier: Apache-2.0
"""Manual, evidence-gated conversion state machine with durable step journal.

The backend owns live processes, native snapshots and existing migration locks.
No predicted completion can advance a destructive step.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DrainEvidence:
    instance_id: str
    instance_generation: int
    sampled_at: float
    running: int | None
    waiting: int | None
    submissions: int | None
    dispatch_leases: int | None
    migration_leases: int | None
    sessions: int | None
    shadows: int | None
    commitments: int | None
    ownership_pending: int | None
    output_tails: int | None
    streams_pending: int | None
    channel_retirable: bool | None

    def drained(self, ref, now, max_age_s=2):
        counts = [
            getattr(self, key)
            for key in (
                "running",
                "waiting",
                "submissions",
                "dispatch_leases",
                "migration_leases",
                "sessions",
                "shadows",
                "commitments",
                "ownership_pending",
                "output_tails",
                "streams_pending",
            )
        ]
        return (
            self.instance_id == ref.instance_id
            and self.instance_generation == ref.instance_generation
            and math.isfinite(self.sampled_at)
            and 0 <= now - self.sampled_at <= max_age_s
            and all(type(v) is int and v == 0 for v in counts)
            and self.channel_retirable is True
        )


class ConversionExecutor:
    """One explicit plan, one bundle, no autonomous layout selection.

    Backend methods are required to be idempotent by plan_id; uncertain effects
    stay tracked and cannot be blindly replayed after a process restart.
    """

    def __init__(self, registry, backend, journal_path):
        self.registry, self.backend = registry, backend
        self.path = Path(journal_path)
        self.plan = None
        if self.path.exists():
            self.plan = json.loads(self.path.read_text(encoding="utf-8"))
            # Restart cannot establish whether a destructive call completed.
            if self.plan["state"] not in {"COMPLETED", "ABORTED", "CREATED"}:
                self.plan["state"] = "RECOVERY_REQUIRED"
                self.save("restart requires backend reconciliation")

    def save(self, reason):
        self.plan["reason"] = reason
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.plan, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def plan_create(
        self,
        plan_id,
        expected_registry_revision,
        bundle_id,
        target_bundle_mode,
        method,
        planner_mode,
        deadline,
        profile_version,
    ):
        if self.plan and self.plan["state"] not in {"ABORTED", "COMPLETED"}:
            raise ValueError("another manual conversion is live")
        if expected_registry_revision != self.registry.registry_revision:
            raise ValueError("registry revision changed")
        refs = [
            r.ref
            for r in self.registry.instances.values()
            if r.bundle_id == bundle_id and r.lifecycle == "ACTIVE"
        ]
        if not refs or target_bundle_mode not in {"TP1", "TP2", "TP4"}:
            raise ValueError("invalid source bundle/target mode")
        if method not in {"DRAIN", "EVAC_ALL_ONCE", "SELECTIVE_ONCE", "SELECTIVE_ROLL"}:
            raise ValueError("invalid method")
        if planner_mode not in {"SHADOW", "ACTIVE"}:
            raise ValueError("explicit planner mode required")
        degrees = [self.registry.get(ref).tp_degree for ref in refs]
        gpus = [gpu for ref in refs for gpu in self.registry.get(ref).physical_gpu_ids]
        target_tp = int(target_bundle_mode[2:])
        if (
            target_tp == 1
            and not (len(refs) == 1 and degrees[0] in (2, 4))
            or target_tp > 1
            and not (degrees == [1] * target_tp and len(gpus) == target_tp)
        ):
            raise ValueError("only TP1 bundle merge/split is supported")
        if method != "DRAIN" and target_tp == 1:
            raise ValueError("assisted split is unsupported")
        if not math.isfinite(deadline):
            raise ValueError("finite deadline required")
        self.plan = dict(
            plan_id=plan_id,
            bundle_id=bundle_id,
            target_bundle_mode=target_bundle_mode,
            method=method,
            planner_mode=planner_mode,
            deadline=deadline,
            profile_version=profile_version,
            refs=[asdict(r) for r in refs],
            state="CREATED",
            destruction_started=False,
            created_revision=expected_registry_revision,
        )
        self.save("explicit manual plan created")
        return self.plan_status()

    def refs(self):
        from .instance_pool import InstanceRef

        return [InstanceRef(**r) for r in self.plan["refs"]]

    def plan_validate(self):
        if self.plan is None:
            raise ValueError("no manual plan")
        for ref in self.refs():
            self.registry.get(ref)
        return self.backend.precheck(self.plan)

    def plan_status(self):
        return json.loads(json.dumps(self.plan))

    def plan_cancel(self):
        if self.plan["destruction_started"]:
            self.plan["state"] = "RECOVERY_REQUIRED"
            self.save("cannot restore retired engines by cancelling")
            return False
        if not self.backend.settle_before_abort(self.plan):
            self.save("abort pending: operations still tracked")
            return False
        for ref in self.refs():
            if self.registry.get(ref).lifecycle == "QUIESCING":
                self.registry.transition(ref, "ACTIVE")
        self.backend.restore_admission(self.plan)
        self.plan["state"] = "ABORTED"
        self.save("healthy engines restored after settlement")
        return True

    def plan_execute(self, now):
        """Advance at most one evidence-gated step; caller drives event cadence."""
        if self.plan is None:
            raise ValueError("no manual plan")
        state = self.plan["state"]
        if state in {"COMPLETED", "ABORTED", "RECOVERY_REQUIRED"}:
            return self.plan_status()
        if self.plan["planner_mode"] == "SHADOW":
            self.save("SHADOW: no admission, engine or migration mutation")
            return self.plan_status()
        if now >= self.plan["deadline"]:
            self.plan_cancel()
            return self.plan_status()
        try:
            if state == "CREATED":
                if not self.plan_validate():
                    self.save("precheck not satisfied")
                    return self.plan_status()
                # Backend closes routing under the SAME dispatch lock used by
                # submit; changing registry lifecycle alone is insufficient.
                self.backend.quiesce(self.plan)
                for ref in self.refs():
                    self.registry.transition(ref, "QUIESCING")
                self.plan["state"] = "DRAINING"
            elif state == "DRAINING":
                if self.backend.kv_emergency(self.plan):
                    self.plan_cancel()
                    return self.plan_status()
                if self.plan["method"] != "DRAIN":
                    self.backend.assisted_step(self.plan, now)
                evidence = self.backend.drain_evidence(self.plan)
                if len(evidence) != len(self.refs()) or not all(
                    row.drained(ref, now) for ref, row in zip(self.refs(), evidence)
                ):
                    self.save("waiting for complete fresh drain evidence")
                    return self.plan_status()
                if not self.backend.fence(self.plan):
                    return self.plan_status()
                self.plan["state"] = "FENCED"
            elif state == "FENCED":
                if not self.backend.retire_channels(self.plan):
                    return self.plan_status()
                # Durable intent BEFORE the first destructive operation.
                self.plan["destruction_started"] = True
                self.plan["state"] = "RECLAIMING"
                self.save("destruction intent; recovery required on uncertainty")
                self.backend.stop_owned_engines(self.plan)
            elif state == "RECLAIMING":
                if not self.backend.resources_released(self.plan):
                    return self.plan_status()
                for ref in self.refs():
                    self.registry.transition(ref, "RETIRED")
                self.backend.start_replacements(self.plan)
                self.plan["state"] = "PREPARING"
            elif state == "PREPARING":
                if not self.backend.warm_and_verify(self.plan):
                    return self.plan_status()
                self.backend.publish_ready(self.plan)
                self.plan["state"] = "COMPLETED"
            self.save("step completed")
        except Exception as error:
            self.plan["state"] = "RECOVERY_REQUIRED"
            self.save(f"operation uncertain: {error!r}")
        return self.plan_status()
