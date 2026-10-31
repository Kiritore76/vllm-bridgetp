# SPDX-License-Identifier: Apache-2.0
"""Drive one selected EXISTING source stream through the existing rolling path.

No benefit policy is called. M2, rolling reservation, target construction and
four-rank direct takeover are reused from the validated single-session runner.
Each driver requires a fresh mechanism directory. PressureSessions coordinates
sequential drivers without reusing artifacts or restarting uncertain sessions.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from tools.bridge_tp.pressure_cpu import load


def channel_idle(run_dir, migration_id, request_id):
    """Five matching IDLE receipts, never a socket-close inference."""
    import json

    paths = [run_dir / "gpu_direct_sender.json"] + [
        run_dir / "persistent_channel_receipts" / f"tp_rank_{rank}.json"
        for rank in range(load("tp_geometry").session_target_tp(run_dir))
    ]
    if not all(path.is_file() for path in paths):
        return False
    sender, *targets = [json.loads(path.read_text()) for path in paths]
    return not (
        type(sender.get("channel_session_count")) is not int
        or sender["channel_session_count"] < 1
        or sender.get("communicator_lifecycle") != "PERSISTENT_CHANNEL_IDLE"
        or sender.get("migration_id") != migration_id
        or sender.get("source_request_id") != request_id
        or any(
            row.get("status") != "PERSISTENT_CHANNEL_IDLE"
            or row.get("migration_id") != migration_id
            or row.get("session_request_id") != request_id
            or row.get("tp_rank") != rank
            for rank, row in enumerate(targets)
        )
        or any(
            row.get("channel_create_count") != 1
            or row.get("channel_destroy_count") != 0
            or row.get("channel_session_count") != sender.get("channel_session_count")
            for row in [sender, *targets]
        )
    )


class RollingPressureSession:
    def __init__(
        self,
        *,
        run_dir: Path,
        migration_id: str,
        config,
        coordinator,
        ingress,
        executor,
        audit,
        request_timeout_s=1800.0,
        backend=None,
        persistent_cleanup=False,
    ):
        if backend is None:
            from tools.bridge_tp import run_phase9_controller as old
        else:
            old = backend

        self.old, self.run_dir, self.config = old, run_dir, config
        self.coordinator, self.ingress, self.executor = coordinator, ingress, executor
        self.audit, self.request_timeout_s = audit, request_timeout_s
        self.migration_id = migration_id
        self.persistent_cleanup = persistent_cleanup
        for name in (
            "session_manifest.json",
            "takeover_state.json",
            "rolling_reservation.json",
            "source_kv_release_receipt.json",
        ):
            if (run_dir / name).exists():
                raise ValueError("pressure pilot requires a fresh mechanism directory")
        self.adapter = old.ActionAdapter(
            config.source_url,
            run_dir,
            expected_migration_id=migration_id,
            target_url=config.target_url,
            ready_notification_mode=getattr(
                config, "ready_notification_mode", "FILE_POLL"
            ),
            ready_notification_port=getattr(config, "ready_notification_port", 0),
        )
        self.machine = old.MigrationStateMachine(
            audit.write, allow_shadow_takeover=True
        )
        self.transfer_mode = getattr(config, "transfer_mode", "ADAPTIVE")
        if self.transfer_mode not in {"FULL_SPEED", "ADAPTIVE"}:
            raise ValueError("unknown pressure transfer_mode")
        self.m2 = (
            None
            if self.transfer_mode == "FULL_SPEED"
            else old.M2RateController(
                old.M2RateConfig(
                    low_bytes_s=config.rate.b_min_bytes_s,
                    medium_bytes_s=config.rate.b_start_bytes_s,
                    high_bytes_s=config.rate.b_max_bytes_s,
                )
            )
        )
        self.record = None
        self.stream = None
        self.target_future = None
        self.settled = False
        self.cleanup_requires_shutdown = False
        self.cancel_future = None
        self.cleanup_executor = None
        self.cancel_reason = None
        self.cancel_started_at = None
        self.cancel_policy = load("pressure_cancel").ShadowCancellation()

    def _rate_decision(self, snapshot, *, before_start=False):
        if self.transfer_mode == "FULL_SPEED":
            # RuntimeControl uses zero to mean unlimited, not paused. Do not
            # invoke M2: neither initial arming nor later ticks may downshift.
            return load("manager_m2").M2RateDecision(
                "HOLD", "FULL_SPEED", 0.0, "full speed; rate limiting disabled"
            )
        return self.m2.decide(snapshot, before_start=before_start)

    def reversible(self):
        return (
            self.record is not None
            and self.record.state == self.old.MigrationState.SHADOW
            and not any(
                (self.run_dir / name).exists()
                for name in (
                    "rolling_freeze_selection.json",
                    "request_frozen_receipt.json",
                    "cutover_manifest.json",
                    "request_freeze_control.json",
                )
            )
        )

    def reconsider(self, now, source, target, candidate, cfg, target_inventory=None):
        if self.settled or self.cancel_future is not None:
            return
        if target_inventory is not None:
            target = load("pressure_cancel").without_shadow_demand(
                target,
                target_inventory,
                f"cmpl-bridgetp-phase9-target-{self.run_dir.name}-0-",
            )
        reason = self.cancel_policy.decide(
            now, source, target, candidate, cfg, reversible=self.reversible()
        )
        if reason:
            self.audit.write(
                dict(
                    kind="pressure_cancel_decision",
                    reason=reason,
                    migration_id=self.migration_id,
                    request_id=self.record.request_id,
                    source=vars(source),
                    target=vars(target),
                    remaining=candidate.remaining.summary(cfg.survival_horizon)
                    if candidate is not None and candidate.remaining is not None
                    else None,
                    config_sha256=cfg.sha256,
                )
            )
            self.cancel(reason)

    def _bind_source(self, candidate):
        """Point commit/cleanup at the engine that owns the selected request.

        Those endpoints abort the source request, so they must reach its own
        engine. G2 (10-31) committed a second source's request on the pair
        source, which aborted nothing; the real source kept recomputing it.
        Without an owner or an endpoint map the configured source is kept.
        """
        owner = getattr(candidate, "owner", None)
        urls = getattr(self.ingress, "urls", None)
        if owner is None or urls is None:
            return None
        if owner not in urls:
            raise ValueError(f"selected request owner {owner!r} has no endpoint")
        self.adapter.source_url = urls[owner].rstrip("/")
        self.audit.write(
            {
                "kind": "pressure_session_source",
                "migration_id": self.migration_id,
                "source_instance": owner,
                "source_url": self.adapter.source_url,
            }
        )
        return owner

    def start(self, candidate, snapshot):
        MigrationState, TriggerPath = self.old.MigrationState, self.old.TriggerPath
        RuntimeControl = self.old.RuntimeControl

        if self.record is not None:
            raise ValueError("one session per migration driver")
        stream = self.ingress.for_server_id(candidate.request_id)
        cap = stream.payload["max_tokens"]
        if candidate.max_output_tokens != cap or cap > 8192:
            raise ValueError("selected request stop contract differs from live stream")
        if candidate.generated + 1 >= cap - 1 or stream.future.done():
            raise ValueError("source ended or has no safe output budget")
        now = time.time()
        self.stream = stream
        self.source_instance = self._bind_source(candidate)
        self.record = self.machine.create(
            self.migration_id, candidate.request_id,
            expected_ranks=getattr(self.config, "target_tp_size", 4),
        )
        self.old.atomic_json_dump(stream.payload, self.run_dir / "source_request.json")
        self.old.atomic_json_dump(
            {
                "format_version": 1,
                "source_subrequest_max_tokens": cap,
                "requested_max_output_tokens": cap,
            },
            self.run_dir / "request_budgets.json",
        )
        current = RuntimeControl.load(self.run_dir) or RuntimeControl()
        replace(
            current,
            armed=False,
            migration_id=self.migration_id,
            source_request_id_prefix=candidate.request_id,
            target_request_admitted=False,
            note="pressure selected existing request",
        ).write(self.run_dir)
        rate = self._rate_decision(snapshot, before_start=True)
        self.adapter.arm_shadow(
            candidate.generated + 1,
            rate.rate_bytes_s / 1024**3,
            cutover_output_tokens=cap - 1,
            note="pressure-balanced-v1 selected request",
        )
        self.record.trigger_output_tokens = candidate.generated + 1
        self.record.trigger_path = TriggerPath.PRESSURE_FEEDBACK
        self.record.t_decision = now
        self.machine.transition(
            self.migration_id,
            MigrationState.SHADOW,
            now,
            "pressure-balanced-v1; existing source stream",
        )
        self.audit.write(
            {
                "kind": "pressure_actual_start",
                "request_id": candidate.request_id,
                "migration_id": self.migration_id,
                "rate": rate.to_json(),
                "transfer_mode": self.transfer_mode,
            }
        )

    def tick(self, request_view, snapshot):
        MigrationState = self.old.MigrationState

        if self.record is None or self.settled or self.cleanup_requires_shutdown:
            return
        if self.cancel_future is not None:
            return  # Main loop polls cleanup; never launch/commit a cancelled target.
        if request_view.request_id != self.record.request_id:
            raise ValueError("active request view differs from selected request")
        record, stream = self.record, self.stream
        if record.state == MigrationState.TAKEOVER:
            self._confirm_release()
            return
        if self.target_future is not None and self.target_future.done():
            self.target_future.result()
            raise RuntimeError("target completed before safe takeover")
        if stream.future.done():
            stream.future.result()
            self.cancel("source completed before takeover")
            return
        snapshot = replace(snapshot, state=record.state.value)
        rate = self._rate_decision(snapshot)
        self.adapter.set_rate(rate.rate_bytes_s / 1024**3, note=rate.reason)
        self.audit.write({"kind": "pressure_m2_rate", "decision": rate.to_json()})
        reason = self.old.step_rolling_shadow(
            adapter=self.adapter,
            audit=self.audit,
            record=record,
            request=request_view,
            max_tokens=stream.payload["max_tokens"],
            reserve_tokens=512,
        )
        if reason:
            self.cancel(reason)
            return
        self.target_future = self.old._start_target_if_ready(
            run_dir=self.run_dir,
            source_request=stream.payload,
            requested_max_output_tokens=stream.payload["max_tokens"],
            adapter=self.adapter,
            recorder=stream.recorder,
            executor=self.executor,
            target_url=self.config.target_url,
            request_timeout_s=self.request_timeout_s,
            target_future=self.target_future,
            gpu_resident_shadow=True,
            cutover_output_tokens=record.candidate_cutover_output_tokens,
            rolling_cutover=True,
        )
        if (
            self.target_future is not None
            and (self.run_dir / "cutover_manifest.json").exists()
        ):
            if record.cutover_output_tokens is None:
                cutover = self.old.load_json(self.run_dir / "cutover_manifest.json")
                selection = self.old.load_json(
                    self.run_dir / "rolling_freeze_selection.json"
                )
                boundary = int(cutover["cutover_num_output_tokens"])
                if selection["cutover_output_tokens"] != boundary:
                    raise RuntimeError("rolling freeze differs from manifest")
                record.cutover_output_tokens = boundary
                stream.recorder.set_cutover(boundary, time.time())
                self.machine.transition(
                    record.migration_id,
                    MigrationState.READY_NOT_COMMITTED,
                    time.time(),
                    "rolling boundary frozen with exact live delta",
                )
                self.audit.write(
                    {"kind": "rolling_cutover_selected", "selection": selection}
                )
            self.old.step_shadow_only_takeover(
                self.machine,
                self.adapter,
                self.audit,
                record,
                stream.recorder,
                time.time(),
                False,
            )
        if record.state == MigrationState.TAKEOVER:
            self._confirm_release()
        elif record.state in {MigrationState.ROLLED_BACK, MigrationState.FAILED}:
            # Preserve reservations until cleanup succeeds. A failed/ambiguous
            # commit may require user/server reconciliation; never assume release.
            raise RuntimeError(
                "takeover failed; retained commitment requires reconciliation"
            )

    def _confirm_release(self):
        path = self.run_dir / "source_kv_release_receipt.json"
        if not path.exists():
            return
        release = self.old.load_json(path)
        state = self.adapter.read_takeover_state()
        if (
            release.get("status") != "SOURCE_KV_RELEASED"
            or release.get("request_id") != self.record.request_id
            or release.get("released_unix_ns", 0) / 1e9 < self.record.t_decision
            or not state
            or state.get("state") != "COMMITTED"
            or state.get("migration_id") != self.migration_id
            or state.get("source_request_id") != self.record.request_id
        ):
            raise ValueError("source-release/ownership receipt identity differs")
        self.coordinator.finish(
            self.migration_id,
            self.record.request_id,
            "COMMITTED",
            ownership_confirmed=True,
            source_release_confirmed=True,
        )
        self.audit.write(
            {
                "kind": "pressure_source_release_confirmed",
                "migration_id": self.migration_id,
                "request_id": self.record.request_id,
                "receipt": release,
                "actual_released_blocks": None,
                "count_evidence": "free-return proven; released block count unknown",
            }
        )
        self.settled = True

    def cancel(self, reason):
        if self.record is None or self.settled or self.cleanup_requires_shutdown:
            return
        if self.cancel_future is not None:
            return
        if self.record.state == self.old.MigrationState.TAKEOVER:
            raise ValueError("cannot cancel committed ownership")
        self.cancel_reason = reason
        self.cancel_started_at = time.time()
        self.audit.write(
            dict(
                kind="pressure_cancel_requested",
                migration_id=self.migration_id,
                request_id=self.record.request_id,
                reason=reason,
            )
        )
        # SSE workers can all be occupied by live source/target streams. Cleanup
        # must have its own worker or it could wait until the source ends.
        self.cleanup_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pressure-shadow-cleanup"
        )
        self.cancel_future = self.cleanup_executor.submit(self._cleanup_shadow, reason)

    def _cleanup_shadow(self, reason):
        # Blocking HTTP/drain waits run away from arrival dispatch and observation.
        # Cleanup without a manifest cannot establish that a pending START was
        # never consumed; retain the promise and surface the uncertainty.
        binding = self.adapter.refresh_binding()
        if binding is None:
            binding = self.adapter.wait_for_preparing_binding(timeout_s=30)
        if binding is None:
            if (
                self.stream.future.done()
                and self.stream.future.exception() is None
                and self.target_future is None
            ):
                self.adapter.disarm(
                    note="source ended before binding; owner shutdown required"
                )
                self.stream.recorder.on_rollback(time.time(), reason)
                self.cleanup_requires_shutdown = True
                self.audit.write(
                    {
                        "kind": "pressure_source_eos_pending_owner_cleanup",
                        "request_id": self.record.request_id,
                        "reservation_retained": True,
                    }
                )
                return None
            raise RuntimeError(
                "START binding unresolved; preserve commitment for reconciliation"
            )
        source = self.adapter.cancel(reason, abort_source=False)
        if (
            source.get("state") != "CANCELLED"
            or source.get("source_abort_dispatched") is not False
            or source.get("migration_id") != self.migration_id
            or source.get("source_request_id") != self.record.request_id
        ):
            raise ValueError("cleanup did not preserve source ownership")
        # Also before rolling reservation exists: target must stay alive until
        # history/deltas drain, otherwise a sender can be stranded in NCCL.
        self.adapter.wait_for_source_cleanup()
        target = self.adapter.cancel_shadow_target(reason)
        if target is not None and (
            target.get("status") != "CLEANED"
            or target.get("migration_id") != self.migration_id
        ):
            raise ValueError("target cleanup outcome not confirmed")
        source["cleanup_http_completed_unix_s"] = time.time()
        return source

    def poll_cancel(self, inventory, now, max_age_s=2.0):
        if self.cancel_future is None or self.settled:
            return
        if not self.cancel_future.done():
            if now - self.cancel_started_at > 90:
                raise TimeoutError("shadow cleanup unresolved; commitment retained")
            return
        source = self.cancel_future.result()
        if self.cleanup_executor is not None:
            self.cleanup_executor.shutdown(wait=False)
        if source is None:
            return  # Unbound EOS requires owner shutdown, never channel reuse.
        if now - self.cancel_started_at > 90:
            raise TimeoutError(
                "shadow cleanup receipts incomplete; commitment retained"
            )
        if self.target_future is not None:
            if not self.target_future.done():
                return
            response = self.target_future.result()
            if response.get("token_ids"):
                raise ValueError("cancelled shadow emitted target tokens")
            if response.get("finish_reason") != "abort":
                raise ValueError("cancelled target lacks terminal abort response")
            path = self.run_dir / "target_abort_receipt.json"
            if not path.is_file():
                return
            aborted = self.old.load_json(path)
            expected = f"cmpl-bridgetp-phase9-target-{self.run_dir.name}-0-"
            if (
                aborted.get("status") != "ABORTED"
                or aborted.get("migration_id") != self.migration_id
                or not aborted.get("target_request_id", "").startswith(expected)
                or aborted.get("completed_unix_s", 0) < self.cancel_started_at
            ):
                raise ValueError("cancelled target abort receipt identity differs")
        prefix = f"cmpl-bridgetp-phase9-target-{self.run_dir.name}-0"
        if (
            inventory["pool_id"] != "tp4"
            or not 0 <= now - inventory["sampled_at"] <= max_age_s
            or inventory["sampled_at"]
            <= source.get("cleanup_http_completed_unix_s", source["updated_unix_s"])
            or any(
                row["request_id"].startswith(prefix) for row in inventory["requests"]
            )
        ):
            return
        for name, component in (
            ("source_cleanup_receipt.json", "source_delta_mirror"),
            ("stager_cleanup_receipt.json", "cpu_stager"),
        ):
            path = self.run_dir / name
            if not path.is_file():
                return
            row = self.old.load_json(path)
            if (
                row.get("status") != "CLEANED"
                or row.get("migration_id") != self.migration_id
                or row.get("component") != component
                or row.get("reason") != self.cancel_reason
                or row.get("updated_unix_s", 0) < self.cancel_started_at
            ):
                raise ValueError("shadow cleanup receipt identity differs")
        if self.persistent_cleanup and not channel_idle(
            self.run_dir, self.migration_id, self.record.request_id
        ):
            return
        stager = self.run_dir / "pressure_stager_receipt.json"
        if stager.is_file() and self.old.load_json(stager).get("returncode") != 0:
            raise ValueError("cancelled stager failed")
        # Single-session legacy pilots have no supervisor receipt. Multiple
        # sessions wait for it before rotating in PressureSessions.rotate().
        reason = self.cancel_reason
        self.stream.recorder.on_rollback(time.time(), reason)
        self.machine.transition(
            self.migration_id, self.old.MigrationState.CANCELLED, now, reason
        )
        self.coordinator.finish(
            self.migration_id,
            self.record.request_id,
            "CANCELLED",
            cleanup_confirmed=True,
        )
        self.coordinator.policy.note_cancelled(self.record.request_id, now)
        self.settled = True
        self.audit.write(
            dict(
                kind="pressure_cancel_confirmed",
                migration_id=self.migration_id,
                request_id=self.record.request_id,
                reason=reason,
                source_continues_on_tp1=True,
                target_inventory_sampled_at=inventory["sampled_at"],
            )
        )
