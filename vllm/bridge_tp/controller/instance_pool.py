# SPDX-License-Identifier: Apache-2.0
"""Generation-scoped pool identities, dispatch leases and target commitments.

One coordinator owns this registry. No scheduler capacity is inferred from TP.
"""

from dataclasses import dataclass, field
from threading import RLock

from .tp_geometry import validate_target_tp


@dataclass(frozen=True, order=True)
class InstanceRef:
    instance_id: str
    instance_generation: int


@dataclass
class InstanceRecord:
    ref: InstanceRef
    tp_degree: int
    physical_gpu_ids: tuple[str, ...]
    bundle_id: str
    endpoint: str
    identity: str
    context_limit: int
    lifecycle: str = "PREPARING"
    telemetry_sample_id: int | None = None
    sampled_at: float | None = None


@dataclass
class DispatchLease:
    request_id: str
    ref: InstanceRef
    state: str = "ACQUIRED"


@dataclass
class Commitment:
    session_id: str
    request_id: str
    source: InstanceRef
    target: InstanceRef
    blocks: int
    reserved_sample_id: int
    allocated_sample_id: int | None = None
    state: str = "RESERVED"


@dataclass
class Registry:
    registry_revision: int = 0
    instances: dict[str, InstanceRecord] = field(default_factory=dict)
    leases: dict[str, DispatchLease] = field(default_factory=dict)
    commitments: dict[str, Commitment] = field(default_factory=dict)
    _lock: object = field(default_factory=RLock, repr=False)

    def get(self, ref):
        row = self.instances[ref.instance_id]
        if row.ref != ref:
            raise ValueError("stale instance generation")
        return row

    def register(self, row):
        with self._lock:
            if row.tp_degree != 1:
                validate_target_tp(row.tp_degree)
            if (
                len(row.physical_gpu_ids) != row.tp_degree
                or len(set(row.physical_gpu_ids)) != row.tp_degree
                or row.context_limit <= 0
                or not row.identity
            ):
                raise ValueError("invalid instance geometry/identity")
            old = self.instances.get(row.ref.instance_id)
            if old and (
                old.lifecycle != "RETIRED"
                or row.ref.instance_generation <= old.ref.instance_generation
            ):
                raise ValueError("replacement requires retired older generation")
            for other in self.instances.values():
                if other.lifecycle != "RETIRED" and set(
                    row.physical_gpu_ids
                ).intersection(other.physical_gpu_ids):
                    raise ValueError("GPU is already owned")
            self.instances[row.ref.instance_id] = row
            self.registry_revision += 1

    def transition(self, ref, state):
        allowed = {
            "PREPARING": {"ACTIVE", "FAILED"},
            "ACTIVE": {"QUIESCING", "FAILED"},
            "QUIESCING": {"ACTIVE", "RETIRED", "FAILED"},
            "FAILED": {"RETIRED"},
            "RETIRED": set(),
        }
        with self._lock:
            row = self.get(ref)
            if state == row.lifecycle:
                return
            if state not in allowed[row.lifecycle]:
                raise ValueError("invalid lifecycle transition")
            if state == "RETIRED" and (
                any(
                    x.ref == ref and x.state != "TERMINAL" for x in self.leases.values()
                )
                or any(
                    ref in (x.source, x.target) and x.state != "RELEASED"
                    for x in self.commitments.values()
                )
            ):
                raise ValueError("unsettled operations prevent retirement")
            row.lifecycle = state
            self.registry_revision += 1

    def acquire_dispatch_lease(self, ref, request_id):
        with self._lock:
            if request_id in self.leases:
                raise ValueError("request already has a dispatch owner")
            if self.get(ref).lifecycle != "ACTIVE":
                raise ValueError("new admission closed")
            lease = DispatchLease(request_id, ref)
            self.leases[request_id] = lease
            return lease

    def reconcile_dispatch(self, ref, request_id, state):
        with self._lock:
            self.get(ref)
            lease = self.leases[request_id]
            transitions = {
                "ACQUIRED": {"ACCEPTED", "UNCERTAIN", "NATIVE_VISIBLE", "TERMINAL"},
                "UNCERTAIN": {"ACCEPTED", "NATIVE_VISIBLE", "TERMINAL"},
                "ACCEPTED": {"NATIVE_VISIBLE", "TERMINAL"},
                "NATIVE_VISIBLE": {"TERMINAL"},
                "TERMINAL": set(),
            }
            if lease.ref != ref or (
                state != lease.state and state not in transitions[lease.state]
            ):
                raise ValueError("invalid dispatch reconciliation")
            lease.state = state

    def try_reserve(
        self,
        session_id,
        request_id,
        source,
        target,
        *,
        target_blocks,
        free_blocks,
        sample_id,
        now,
        sampled_at,
        max_age_s=2,
    ):
        """Atomic capacity test and reservation, in native TARGET block units."""
        with self._lock:
            self.get(source)
            row = self.get(target)
            if session_id in self.commitments:
                old = self.commitments[session_id]
                if (old.request_id, old.source, old.target, old.blocks) != (
                    request_id,
                    source,
                    target,
                    target_blocks,
                ):
                    raise ValueError("session reservation identity differs")
                return old
            if (
                row.lifecycle != "ACTIVE"
                or not 0 <= now - sampled_at <= max_age_s
                or type(target_blocks) is not int
                or target_blocks <= 0
                or type(free_blocks) is not int
                or free_blocks < 0
                or type(sample_id) is not int
            ):
                raise ValueError("fresh target capacity evidence required")
            if any(
                x.request_id == request_id and x.state != "RELEASED"
                for x in self.commitments.values()
            ):
                raise ValueError("request already migrating")
            pending = sum(
                x.blocks
                for x in self.commitments.values()
                if x.target == target and x.state in {"RESERVED", "ACKED"}
            )
            if target_blocks + pending > free_blocks:
                return None
            entry = Commitment(
                session_id, request_id, source, target, target_blocks, sample_id
            )
            self.commitments[session_id] = entry
            return entry

    def materialize(self, session_id, target, allocation_sample_id):
        with self._lock:
            self.get(target)
            entry = self.commitments[session_id]
            if entry.target != target or entry.state not in {"RESERVED", "ACKED"}:
                raise ValueError("allocation acknowledgment identity/state differs")
            entry.allocated_sample_id = allocation_sample_id
            entry.state = "ACKED"

    def reconcile_inventory(self, session_id, target, sample_id, request_visible):
        with self._lock:
            self.get(target)
            entry = self.commitments[session_id]
            if entry.target != target:
                raise ValueError("inventory generation differs")
            if (
                entry.state == "ACKED"
                and request_visible
                and sample_id > entry.reserved_sample_id
                and sample_id >= entry.allocated_sample_id
            ):
                entry.state = "INVENTORY"

    def release_after_evidence(self, session_id, target, *, remote_free, fresh_absence):
        with self._lock:
            self.get(target)
            entry = self.commitments[session_id]
            if entry.target != target or not (remote_free and fresh_absence):
                raise ValueError("matching free and inventory absence required")
            entry.state = "RELEASED"


def layouts(target_tp, gpu_ids):
    validate_target_tp(target_tp)
    if len(gpu_ids) != target_tp * 2 or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("exactly two disjoint physical GPU bundles required")
    groups = [tuple(gpu_ids[:target_tp]), tuple(gpu_ids[target_tp:])]
    return {
        label: [
            dict(
                bundle_id=bundle,
                tp_degree=target_tp if merged else 1,
                gpu_ids=list(gpus) if merged else [gpu],
            )
            for bundle, gpus, merged in zip(("A", "B"), groups, flags)
            for gpu in (gpus[:1] if merged else gpus)
        ]
        for label, flags in {
            "L0": (False, False),
            "L1A": (False, True),
            "L1B": (True, False),
            "L2": (True, True),
        }.items()
    }
