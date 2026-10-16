# SPDX-License-Identifier: Apache-2.0
"""Drive one selected EXISTING source stream through the existing rolling path.

No benefit policy is called. M2, rolling reservation, target construction and
four-rank direct takeover are reused from the validated single-session runner.
Each driver requires a fresh mechanism directory. PressureSessions coordinates
sequential drivers without reusing artifacts or restarting uncertain sessions.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path


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
    ):
        if backend is None:
            from tools.bridge_tp import run_phase9_controller as old
        else:
            old = backend

        self.old, self.run_dir, self.config = old, run_dir, config
        self.coordinator, self.ingress, self.executor = coordinator, ingress, executor
        self.audit, self.request_timeout_s = audit, request_timeout_s
        self.migration_id = migration_id
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
        self.m2 = old.M2RateController(
            old.M2RateConfig(
                low_bytes_s=config.rate.b_min_bytes_s,
                medium_bytes_s=config.rate.b_start_bytes_s,
                high_bytes_s=config.rate.b_max_bytes_s,
            )
        )
        self.record = None
        self.stream = None
        self.target_future = None
        self.settled = False
        self.cleanup_requires_shutdown = False

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
        self.record = self.machine.create(self.migration_id, candidate.request_id)
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
        rate = self.m2.decide(snapshot, before_start=True)
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
            }
        )

    def tick(self, request_view, snapshot):
        MigrationState = self.old.MigrationState

        if self.record is None or self.settled or self.cleanup_requires_shutdown:
            return
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
        rate = self.m2.decide(snapshot)
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
                return
            raise RuntimeError(
                "START binding unresolved; preserve commitment for reconciliation"
            )
        source = self.adapter.cancel(reason, abort_source=False)
        if source.get("source_abort_dispatched") is not False:
            raise ValueError("cleanup did not preserve source ownership")
        target = self.adapter.cancel_shadow_target(reason)
        if target is not None and target.get("status") not in {"CLEANED", "NOT_FOUND"}:
            raise ValueError("target cleanup outcome not confirmed")
        self.stream.recorder.on_rollback(time.time(), reason)
        self.coordinator.finish(
            self.migration_id,
            self.record.request_id,
            "CANCELLED",
            cleanup_confirmed=True,
        )
        self.settled = True
