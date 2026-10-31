# SPDX-License-Identifier: Apache-2.0
"""Generation-scoped pool identities, dispatch leases and target commitments.

One coordinator owns this registry. No scheduler capacity is inferred from TP.
"""

import itertools
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


LEGACY_SOURCE_POOL = "tp1"
LEGACY_TARGET_POOL = "tp4"  # historical name of the TP-k target pool, any k


def fixed_pair_registry(
    *,
    source_gpu,
    target_gpus,
    source_url,
    target_url,
    identity,
    source_context_limit,
    target_context_limit,
    source_bundle="A",
    target_bundle="B",
    extras=(),
):
    """Registry of the fixed TP1 source / TP-k target pilot.

    Named like the bundle runner (``<bundle>-g<generation>-tp<k>``): the source
    is generation 1, the target 2, routing-only ``extras`` (dicts with bundle,
    tp, gpus, endpoint, context_limit) 3, 4, ... The controller still uses the
    historical pool ids; they are returned as aliases of the pair's ids.

    A single-class layout (L0/L2) has only one side of the pair: pass
    ``source_bundle=None`` or ``target_bundle=None``. The absent side keeps its
    generation number unused and gets no alias.
    """
    registry = Registry()
    target_tp = len(target_gpus)
    if source_bundle is None and target_bundle is None:
        raise ValueError("a layout needs at least one side of the pair")
    rows = [
        InstanceRecord(
            InstanceRef(f"{source_bundle}-g1-tp1", 1),
            1,
            (str(source_gpu),),
            source_bundle,
            source_url,
            identity,
            source_context_limit,
        ),
        InstanceRecord(
            InstanceRef(f"{target_bundle}-g2-tp{target_tp}", 2),
            target_tp,
            tuple(str(gpu) for gpu in target_gpus),
            target_bundle,
            target_url,
            identity,
            target_context_limit,
        ),
    ]
    pair = [
        (pool, row)
        for pool, row, bundle in (
            (LEGACY_SOURCE_POOL, rows[0], source_bundle),
            (LEGACY_TARGET_POOL, rows[1], target_bundle),
        )
        if bundle is not None
    ]
    rows = [row for _, row in pair]
    for generation, extra in enumerate(extras, start=3):
        rows.append(
            InstanceRecord(
                InstanceRef(
                    f"{extra['bundle']}-g{generation}-tp{extra['tp']}", generation
                ),
                extra["tp"],
                tuple(str(gpu) for gpu in extra["gpus"]),
                extra["bundle"],
                extra["endpoint"],
                identity,
                extra["context_limit"],
            )
        )
    for row in rows:
        registry.register(row)
    aliases = {pool: row.ref.instance_id for pool, row in pair}
    return registry, aliases


def registry_manifest(registry, aliases=None):
    """JSON-ready instance list; ``legacy_pool`` names the old pool id, if any."""
    aliases = dict(aliases or {})
    by_instance = {value: key for key, value in aliases.items()}
    if len(by_instance) != len(aliases) or not set(by_instance) <= set(
        registry.instances
    ):
        raise ValueError("aliases must name distinct registered instances")
    return dict(
        schema="BRIDGETP_INSTANCE_MANIFEST_V1",
        registry_revision=registry.registry_revision,
        instances=[
            dict(
                instance_id=row.ref.instance_id,
                instance_generation=row.ref.instance_generation,
                tp_degree=row.tp_degree,
                physical_gpu_ids=list(row.physical_gpu_ids),
                bundle_id=row.bundle_id,
                endpoint=row.endpoint,
                context_limit=row.context_limit,
                lifecycle=row.lifecycle,
                legacy_pool=by_instance.get(row.ref.instance_id),
            )
            for _, row in sorted(registry.instances.items())
        ],
        legacy_pool_aliases=aliases,
    )


def resolve_pool(manifest, pool_id):
    """Instance id for a historical pool id or an instance id."""
    aliases = manifest["legacy_pool_aliases"]
    if pool_id in aliases:
        return aliases[pool_id]
    if any(row["instance_id"] == pool_id for row in manifest["instances"]):
        return pool_id
    raise KeyError(f"unknown pool or instance: {pool_id}")


def layout_label(merged, bundle_ids):
    """``L<k>`` plus the bundles still at TP1 when the layout is mixed.

    With two bundles this is the historical L0/L1A/L1B/L2 naming: L1A keeps
    bundle A at TP1 and merges B.
    """
    count = sum(merged)
    if count in (0, len(merged)):
        return f"L{count}"
    return f"L{count}" + "".join(b for b, m in zip(bundle_ids, merged) if not m)


def layouts(target_tp, gpu_ids):
    """Every TP1/TP-k assignment over consecutive ``target_tp``-GPU bundles.

    Bundles are named A, B, C... in ``gpu_ids`` order; each one is either
    ``target_tp`` TP1 instances or one TP-``target_tp`` instance.
    """
    validate_target_tp(target_tp)
    count = len(gpu_ids) // target_tp
    if (
        count < 1
        or count > 26
        or len(gpu_ids) != count * target_tp
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("disjoint physical GPU bundles of target_tp GPUs required")
    bundle_ids = [chr(ord("A") + i) for i in range(count)]
    groups = [tuple(gpu_ids[i * target_tp : (i + 1) * target_tp]) for i in range(count)]
    result = {}
    for merged in itertools.product((False, True), repeat=count):
        result[layout_label(merged, bundle_ids)] = [
            dict(
                bundle_id=bundle,
                tp_degree=target_tp if flag else 1,
                gpu_ids=list(gpus) if flag else [gpu],
            )
            for bundle, gpus, flag in zip(bundle_ids, groups, merged)
            for gpu in (gpus[:1] if flag else gpus)
        ]
    return result
