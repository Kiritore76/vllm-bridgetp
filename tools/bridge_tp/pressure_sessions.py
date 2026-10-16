# SPDX-License-Identifier: Apache-2.0
"""Serialized migrations on one persistent channel, with isolated receipts."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


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

    @property
    def ready_for_start(self):
        return self.record is None

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
            if not entry.service_observed:
                return False
        paths = [self.run_dir / "gpu_direct_sender.json"] + [
            self.run_dir / "persistent_channel_receipts" / f"tp_rank_{rank}.json"
            for rank in range(4)
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
        sender, *rest = rows
        targets, finalized, stager = rest[:4], rest[4], rest[5]
        if (
            sender.get("communicator_lifecycle") != "PERSISTENT_CHANNEL_IDLE"
            or sender.get("migration_id") != self.migration_id
            or sender.get("source_request_id") != self.record.request_id
            or any(
                row.get("channel_create_count") != 1
                or row.get("channel_destroy_count") != 0
                or row.get("channel_session_count") != len(self.sessions)
                for row in [sender, *targets]
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
        old_id = self.migration_id
        self.adapter.close()  # Release the UDP notification port before rebinding.
        self._new()
        # A new service migration needs a fresh persistence period. KV warnings
        # continue to bypass that period through PressurePolicy.decide().
        self.coordinator.policy.note_started(time.time())
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    follow_stager(sys.argv[1:])
