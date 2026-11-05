# SPDX-License-Identifier: Apache-2.0
"""Serialized migrations on one persistent channel, with isolated receipts."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bridge_tp.pressure_cpu import load  # noqa: E402


def publish_active(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if directory.parent != root / "sessions":
        raise ValueError("active migration must stay inside owned sessions")
    pending = root / "migration_active.next"
    if pending.exists() or pending.is_symlink():
        raise ValueError("unexpected pending migration link")
    pending.symlink_to(directory, target_is_directory=True)
    os.replace(pending, root / "migration_active")


def initialize(root):
    directory = root / "sessions" / "migration-0001"
    directory.mkdir(parents=True, exist_ok=False)
    publish_active(root, directory)
    return directory


class PressureSessions:
    """Keep old response streams alive while the channel serves another request.

    Rotation requires source release, native target progress, all five channel
    IDLE receipts, scheduler finalization and clean stager completion. Uncertain
    cancellation blocks rotation until owner shutdown. No session files are
    overwritten or deleted. Only a confirmed cancelled source may be retried;
    committed requests retain their target ownership.
    """

    def __init__(self, *, maximum_migrations=1, **kwargs):
        from tools.bridge_tp.pressure_session import RollingPressureSession

        self.factory = RollingPressureSession
        self.maximum = maximum_migrations
        self.kwargs = kwargs
        self.root = kwargs["run_dir"]
        self.sessions = []
        self.multiple = maximum_migrations != 1
        self._new()

    def __getattr__(self, key):
        return getattr(self.sessions[-1], key)

    def _new(self):
        index = len(self.sessions) + 1
        kwargs = dict(self.kwargs)
        if self.multiple:
            directory = self.root / "sessions" / f"migration-{index:04d}"
            if index > 1:
                directory.mkdir(exist_ok=False)
                self.old.RuntimeControl(armed=False).write(directory)
                publish_active(self.root, directory)
            kwargs.update(
                run_dir=directory,
                migration_id=f"{kwargs['migration_id']}-{index:04d}",
            )
        self.sessions.append(self.factory(**kwargs))

    @staticmethod
    def _source(session):
        """Source instance of a session; the fixed pair's source when unbound."""
        return getattr(session, "source_instance", None) or "tp1"

    @property
    def ready_for_start(self):
        return self.record is None

    def bind_target(self, pool_id, endpoint, epoch):
        self.sessions[-1].bind_target(pool_id, endpoint, epoch)

    @staticmethod
    def _channel_key(session):
        return (
            PressureSessions._source(session),
            getattr(session, "target_epoch", None),
        )

    def rotate(self):
        if not self.multiple or not self.record or not self.settled:
            return False
        if self.maximum and len(self.sessions) >= self.maximum:
            return False
        if self.cleanup_requires_shutdown:
            return False
        entry = self.coordinator.ledger.entries[self.migration_id]
        cancelled = entry.state == "CANCELLED"
        if not cancelled:
            if self.target_future is None or not self.stream.future.done():
                return False
            self.stream.future.result()
            if entry.state not in {"COMMITTED", "ACCOUNTED", "COMPLETED"}:
                return False
            # A request that ends right after takeover may never be sampled with
            # output on its target (GPU 11-05: 7s left, then no rotation and no
            # more migrations). Completion is stronger evidence than a sample.
            if not entry.service_observed and entry.state != "COMPLETED":
                return False
        from tools.bridge_tp.pressure_session import sender_receipt_paths

        senders = sender_receipt_paths(self.run_dir)
        paths = senders + [
            self.run_dir / "persistent_channel_receipts" / f"tp_rank_{rank}.json"
            for rank in range(load("tp_geometry").session_target_tp(self.run_dir))
        ]
        paths += [
            self.run_dir
            / (
                "stager_cleanup_receipt.json"
                if cancelled
                else "rolling_target_finalized.json"
            ),
            self.run_dir / "pressure_stager_receipt.json",
        ]
        if not all(path.is_file() for path in paths):
            return False
        rows = [json.loads(path.read_text()) for path in paths]
        # Each TP1 source has its own persistent channel (S1-d2): its receipts
        # count that channel's sessions, not every session of the run.
        source = self._channel_key(self.sessions[-1])
        channel_sessions = sum(
            self._channel_key(s) == source
            for s in self.sessions
            if s.record is not None
        )
        sender_rows, rest = rows[: len(senders)], rows[len(senders) :]
        sender = sender_rows[0]
        count = load("tp_geometry").session_target_tp(self.run_dir)
        targets, finalized, stager = rest[:count], rest[count], rest[count + 1]
        if (
            any(
                row.get("communicator_lifecycle") != "PERSISTENT_CHANNEL_IDLE"
                or row.get("migration_id") != self.migration_id
                or row.get("source_request_id") != self.record.request_id
                for row in sender_rows
            )
            or any(
                row.get("channel_create_count") != 1
                or row.get("channel_destroy_count") != 0
                or row.get("channel_session_count") != channel_sessions
                for row in [*sender_rows, *targets]
            )
            or any(
                row.get("status") != "PERSISTENT_CHANNEL_IDLE"
                or row.get("migration_id") != self.migration_id
                or row.get("session_request_id") != self.record.request_id
                or row.get("tp_rank") != rank
                for rank, row in enumerate(targets)
            )
            or finalized.get("migration_id") != self.migration_id
            or stager.get("returncode") != 0
        ):
            raise ValueError("migration rotation receipts disagree")
        # A later layout conversion may close this now-idle communicator.
        # Preserve the verified idle boundary before its live receipts change.
        idle = dict(
            status="VERIFIED_CHANNEL_IDLE",
            migration_id=self.migration_id,
            source_request_id=self.record.request_id,
            target_epoch=getattr(self.sessions[-1], "target_epoch", None),
            verified_unix_s=time.time(),
            sender={
                k: sender.get(k)
                for k in (
                    "communicator_lifecycle",
                    "migration_id",
                    "source_request_id",
                    "channel_create_count",
                    "channel_destroy_count",
                    "channel_session_count",
                )
            },
            ranks=[
                {
                    k: row.get(k)
                    for k in (
                        "status",
                        "migration_id",
                        "session_request_id",
                        "tp_rank",
                        "channel_create_count",
                        "channel_destroy_count",
                        "channel_session_count",
                    )
                }
                for row in targets
            ],
        )
        temporary = self.run_dir / "channel_idle_verified.tmp"
        temporary.write_text(json.dumps(idle), encoding="utf-8")
        os.replace(temporary, self.run_dir / "channel_idle_verified.json")
        old_id = self.migration_id
        self.adapter.close()  # Release the UDP notification port before rebinding.
        self._new()
        # A new service migration needs a fresh persistence period on every
        # source. KV warnings continue to bypass it through decide().
        now = time.time()
        policies = getattr(self.coordinator, "_policies", None) or {}
        for policy in {id(p): p for p in policies.values()}.values() or [
            self.coordinator.policy
        ]:
            policy.note_started(now)
        self.audit.write(
            dict(
                kind="pressure_channel_reused",
                previous_migration_id=old_id,
                previous_outcome="CANCELLED" if cancelled else "COMMITTED",
                migration_id=self.migration_id,
                run_dir=str(self.run_dir),
            )
        )
        return True

    def close(self):
        for session in self.sessions:
            cleanup_executor = getattr(session, "cleanup_executor", None)
            if cleanup_executor is not None:
                cleanup_executor.shutdown(wait=False, cancel_futures=False)
            session.adapter.close()


def verified_idle(root, migration_id, request_id):
    """An IDLE boundary certified by rotate, before a later channel retirement."""
    path = Path(root) / "channel_idle_verified.json"
    if not path.is_file():
        return False
    value = json.loads(path.read_text())
    sender, ranks = value.get("sender", {}), value.get("ranks", [])
    return (
        value.get("status") == "VERIFIED_CHANNEL_IDLE"
        and value.get("migration_id") == sender.get("migration_id") == migration_id
        and value.get("source_request_id")
        == sender.get("source_request_id")
        == request_id
        and sender.get("communicator_lifecycle") == "PERSISTENT_CHANNEL_IDLE"
        and type(sender.get("channel_session_count")) is int
        and sender["channel_session_count"] > 0
        and len(ranks) == 2
        and all(
            row.get("tp_rank") == rank
            and row.get("status") == "PERSISTENT_CHANNEL_IDLE"
            and row.get("migration_id") == migration_id
            and row.get("session_request_id") == request_id
            and row.get("channel_session_count") == sender.get("channel_session_count")
            for rank, row in enumerate(ranks)
        )
        and all(
            row.get("channel_create_count") == 1
            and row.get("channel_destroy_count") == 0
            for row in [sender, *ranks]
        )
    )


def follow_stager(argv, run=None):
    """Reuse the existing CPU stager sequentially, importing torch only once."""
    if run is None:
        from tools.bridge_tp import phase8_stager

        run = phase8_stager.main
    arguments = list(argv)
    index = arguments.index("--run-dir") + 1
    active = Path(arguments[index])
    seen = set()
    while True:
        directory = active.resolve(strict=True)
        if directory not in seen:
            if directory.parent != active.parent / "sessions":
                raise ValueError("stager active link escaped owned sessions")
            seen.add(directory)
            child_args = list(arguments)
            child_args[index] = str(directory)
            failure = None
            try:
                run(child_args)
            except BaseException as error:
                failure = repr(error)
                raise
            finally:
                temporary = directory / "pressure_stager_receipt.tmp"
                temporary.write_text(
                    json.dumps(
                        dict(
                            returncode=0 if failure is None else 1,
                            error=failure,
                        )
                    )
                )
                os.replace(temporary, directory / "pressure_stager_receipt.json")
        time.sleep(0.2)


if __name__ == "__main__":
    follow_stager(sys.argv[1:])
