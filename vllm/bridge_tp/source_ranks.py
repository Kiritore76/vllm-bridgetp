# SPDX-License-Identifier: Apache-2.0
"""Which rank of a migration source does what (same-TP migration, user 11-05).

TP1 to TP-k splits one KV into k shards, so one process sends to every target
rank. Between instances of the same TP degree nothing is split: source rank i
already holds exactly the heads of target rank i, so each source rank sends its
own KV to its own target rank. TP1 to TP1 is the one-rank case of the same rule.

The ranks of a source run the same batches, so a token count names the same
step on all of them; that is the only thing they share without communicating.
Rank 0 leads: it publishes the session, plans the boundary and asks for the
freeze. The other ranks follow on files in the session directory, and rank 0
only publishes what depends on them once they have reported.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_TARGET_TP = (1, 2, 4)
RANK_FAILURE = "source_rank_failed.json"
# The ranks finish the same model step within moments of each other, so this
# only runs out when a rank is not coming; the model step waits this long.
JOIN_TIMEOUT_S = 5.0


class PeerFailed(RuntimeError):
    """Another source rank reported that it cannot finish this session."""


@dataclass(frozen=True)
class SourceRole:
    tp_size: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        if self.tp_size < 1 or not 0 <= self.rank < self.tp_size:
            raise ValueError(f"invalid source rank {self.rank}/{self.tp_size}")

    @property
    def leader(self) -> bool:
        return self.rank == 0

    @property
    def paired(self) -> bool:
        """Each rank sends its own KV; False for the single process of a TP1."""
        return self.tp_size > 1

    def check_target(self, target_tp: int) -> None:
        if target_tp not in SUPPORTED_TARGET_TP:
            raise ValueError(f"unsupported target TP {target_tp}")
        if self.paired and target_tp != self.tp_size:
            raise ValueError(
                f"source TP{self.tp_size} can only migrate to the same TP, "
                f"not TP{target_tp}"
            )

    def sender_tp(self, target_tp: int) -> int:
        """Target ranks this process talks to."""
        self.check_target(target_tp)
        return 1 if self.paired else target_tp

    def rank_base(self) -> int:
        """Target rank of this process's first connection."""
        return self.rank if self.paired else 0

    def target_addresses(self, host: str, base_port: int, target_tp: int) -> list[str]:
        ranks = (
            [self.rank] if self.paired else range(self.sender_tp(target_tp))
        )
        self.check_target(target_tp)
        return [f"{host}:{base_port + rank}" for rank in ranks]

    def sender_heads(self, total_heads: int) -> int:
        """KV heads in this process's cache."""
        if total_heads % self.tp_size:
            raise ValueError("KV heads must divide over source TP ranks")
        return total_heads // self.tp_size

    def own_file(self, name: str) -> str:
        """Rank 0 keeps the historical file name; others write beside it."""
        if self.leader:
            return name
        stem, dot, suffix = name.rpartition(".")
        return f"{stem}_rank_{self.rank}{dot}{suffix}"

    def delta_receipt_dir(self, run_dir: Path) -> Path:
        root = Path(run_dir) / "gpu_direct_delta_sender_receipts"
        return root / f"rank_{self.rank}" if self.paired else root

    def peers(self) -> range:
        return range(1, self.tp_size)


def local_role() -> SourceRole:
    """This worker's position in its TP group (TP1 when none is initialised)."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
    except (ImportError, AssertionError, AttributeError):
        return SourceRole()
    return SourceRole(int(group.world_size), int(group.rank_in_group))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def report_failure(run_dir: Path, role: SourceRole, error: str) -> None:
    """Any rank can stop a session; the first report stays the reason."""
    path = Path(run_dir) / RANK_FAILURE
    if path.exists():
        return
    _write(
        path,
        dict(
            format_version=1,
            status="SOURCE_RANK_FAILED",
            rank=role.rank,
            error=error,
            updated_unix_s=time.time(),
        ),
    )


def read_failure(run_dir: Path) -> dict[str, Any] | None:
    return _read(Path(run_dir) / RANK_FAILURE)


def _peer_path(run_dir: Path, stage: str, rank: int) -> Path:
    return Path(run_dir) / "source_rank_reports" / f"{stage}_rank_{rank}.json"


def report_stage(run_dir: Path, role: SourceRole, stage: str, **fields: Any) -> None:
    """A follower's evidence that it has finished ``stage`` of this session."""
    _write(
        _peer_path(run_dir, stage, role.rank),
        dict(format_version=1, rank=role.rank, stage=stage, **fields),
    )


def wait_for_peers(
    run_dir: Path,
    role: SourceRole,
    stage: str,
    *,
    timeout_s: float,
    expected: dict[str, Any] | None = None,
    abandoned: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Rank 0 waits until every follower reported ``stage`` with these fields.

    Fails closed: a follower's failure, a different boundary or the timeout all
    raise, so nothing that depends on the whole KV is published without it.
    """
    deadline = clock() + timeout_s
    while True:
        failure = read_failure(run_dir)
        if failure is not None and stage != "cleaned":
            raise PeerFailed(
                f"source rank {failure.get('rank')}: {failure.get('error')}"
            )
        # A rank that failed has nothing left to clean up.
        ranks = [
            rank
            for rank in role.peers()
            if failure is None or failure.get("rank") != rank
        ]
        reports = [_read(_peer_path(run_dir, stage, rank)) for rank in ranks]
        if all(report is not None for report in reports):
            for report in reports:
                for key, value in (expected or {}).items():
                    if report.get(key) != value:
                        raise PeerFailed(
                            f"source rank {report.get('rank')} {stage} {key} "
                            f"{report.get(key)!r} differs from {value!r}"
                        )
            return reports
        if abandoned():
            raise PeerFailed(f"session abandoned while waiting for {stage}")
        if clock() >= deadline:
            raise TimeoutError(f"source ranks did not report {stage}")
        sleep(0.005)


def trigger_check(role: SourceRole, output_tokens: int, trigger: int) -> str:
    """When a rank starts the session: ``WAIT``, ``START`` or ``MISSED``.

    One process may start at any later token. Several must copy the same
    history, and may only do that at the exact token named in advance: a rank
    that saw the session too late refuses instead of copying a different one.
    """
    if output_tokens < trigger:
        return "WAIT"
    if role.paired and output_tokens != trigger:
        return "MISSED"
    return "START"
