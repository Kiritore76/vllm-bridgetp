# SPDX-License-Identifier: Apache-2.0
"""Unified client-facing response stream (paper Section 3.7).

The client sees one external request ID and one continuous token stream, while
internally a source request and a target request exist. The proxy owns the
seam. Its invariant is checked on every emission:

    emitted global indices are exactly 0, 1, 2, ... with no gap and no repeat.

Two seam policies are implemented, and the difference between them is a result
worth measuring rather than an implementation detail.

``HOLD_BACK`` (default, sampling-agnostic)
    Once the cutover boundary K is set, source tokens with index >= K are
    buffered instead of emitted. On commit the buffer is discarded and the
    target stream continues from K. On rollback or cancel the buffer is
    flushed and the source remains the owner. Client-visible handoff stall is
    then the full Handoff duration.

``GREEDY_FASTPATH`` (greedy sampling only)
    Source tokens keep flowing past K. Because the target resumes from the
    same computed/pending boundary under greedy decoding, its tokens in the
    overlap region [K, high_water) must equal what the client already saw; the
    proxy verifies this and then resumes emission from high_water. Handoff
    stall collapses to roughly one decode step, but the mode is unsound the
    moment sampling is non-deterministic, so it is opt-in and the verification
    is not optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProxyMode(str, Enum):
    HOLD_BACK = "HOLD_BACK"
    GREEDY_FASTPATH = "GREEDY_FASTPATH"


class StreamViolation(RuntimeError):
    """Raised when the no-gap / no-duplicate invariant would be broken."""


@dataclass
class EmittedToken:
    index: int
    token_id: int
    unix_s: float
    origin: str  # "source" or "target"


@dataclass
class ResponseProxy:
    external_request_id: str
    mode: ProxyMode = ProxyMode.HOLD_BACK

    emitted: list[EmittedToken] = field(default_factory=list)
    cutover_index: int | None = None
    committed: bool = False
    finished_reason: str = ""

    _held: list[tuple[int, int, float]] = field(default_factory=list, repr=False)
    _discarded_source: int = field(default=0, repr=False)
    _verified_overlap: int = field(default=0, repr=False)
    _last_source_emit_s: float | None = field(default=None, repr=False)
    _first_target_emit_s: float | None = field(default=None, repr=False)

    # ---- helpers ------------------------------------------------------
    @property
    def high_water(self) -> int:
        """Next global index the client expects."""
        return self.emitted[-1].index + 1 if self.emitted else 0

    def _emit(
        self,
        index: int,
        token_id: int,
        unix_s: float,
        origin: str,
    ) -> EmittedToken:
        if index != self.high_water:
            raise StreamViolation(
                f"{self.external_request_id}: expected index {self.high_water}, "
                f"got {index} from {origin}"
            )
        token = EmittedToken(index, token_id, unix_s, origin)
        self.emitted.append(token)
        if origin == "source":
            self._last_source_emit_s = unix_s
        elif origin == "target" and self._first_target_emit_s is None:
            self._first_target_emit_s = unix_s
        return token

    # ---- source side --------------------------------------------------
    def on_source_token(
        self, index: int, token_id: int, unix_s: float
    ) -> list[EmittedToken]:
        if self.committed:
            # source raced past the abort; its output is no longer authoritative
            self._discarded_source += 1
            return []
        if (
            self.mode is ProxyMode.HOLD_BACK
            and self.cutover_index is not None
            and index >= self.cutover_index
        ):
            self._held.append((index, token_id, unix_s))
            return []
        return [self._emit(index, token_id, unix_s, "source")]

    # ---- seam ---------------------------------------------------------
    def set_cutover(self, cutover_index: int, unix_s: float) -> None:
        if self.cutover_index is not None and self.cutover_index != cutover_index:
            raise StreamViolation(
                f"{self.external_request_id}: cutover already set to "
                f"{self.cutover_index}, cannot move to {cutover_index}"
            )
        if cutover_index < 0:
            raise StreamViolation("cutover index must be non-negative")
        if self.mode is ProxyMode.HOLD_BACK and cutover_index < self.high_water:
            raise StreamViolation(
                f"{self.external_request_id}: cutover {cutover_index} is behind "
                f"already-emitted high water {self.high_water}"
            )
        self.cutover_index = cutover_index

    def on_commit(self, unix_s: float) -> None:
        if self.cutover_index is None:
            raise StreamViolation("cannot commit before a cutover boundary is set")
        self.committed = True
        self._discarded_source += len(self._held)
        self._held.clear()

    def on_rollback(
        self,
        unix_s: float,
        reason: str = "rollback",
    ) -> list[EmittedToken]:
        """Source stays the owner; flush anything held back."""
        if self.committed:
            raise StreamViolation("cannot roll back after commit")
        out = [self._emit(i, t, ts, "source") for i, t, ts in self._held]
        self._held.clear()
        self.cutover_index = None
        self.finished_reason = reason
        return out

    # ---- target side --------------------------------------------------
    def on_target_token(
        self, index: int, token_id: int, unix_s: float
    ) -> list[EmittedToken]:
        """``index`` is the GLOBAL index; the target resumes at the cutover."""
        if not self.committed:
            raise StreamViolation(
                f"{self.external_request_id}: target token {index} arrived before "
                "commit; the target must stay behind the barrier"
            )
        if index < self.high_water:
            # Overlap region. Only legal in the greedy fast path, and only if
            # the target reproduces exactly what the client already received.
            if self.mode is not ProxyMode.GREEDY_FASTPATH:
                raise StreamViolation(
                    f"{self.external_request_id}: duplicate index {index}"
                )
            already = self.emitted[index]
            if already.token_id != token_id:
                raise StreamViolation(
                    f"{self.external_request_id}: greedy fast path diverged at "
                    f"index {index}: client saw {already.token_id}, target "
                    f"produced {token_id}"
                )
            self._verified_overlap += 1
            return []
        return [self._emit(index, token_id, unix_s, "target")]

    # ---- reporting ----------------------------------------------------
    @property
    def handoff_stall_s(self) -> float | None:
        """First target-origin visible token minus last source-origin one."""
        if self._last_source_emit_s is None or self._first_target_emit_s is None:
            return None
        return self._first_target_emit_s - self._last_source_emit_s

    def token_ids(self) -> list[int]:
        return [t.token_id for t in self.emitted]

    def assert_contiguous(self) -> None:
        for position, token in enumerate(self.emitted):
            if token.index != position:
                raise StreamViolation(
                    f"{self.external_request_id}: stream is not contiguous at "
                    f"position {position} (index {token.index})"
                )

    def stats(self) -> dict:
        self.assert_contiguous()
        return {
            "external_request_id": self.external_request_id,
            "mode": self.mode.value,
            "emitted_tokens": len(self.emitted),
            "source_origin_tokens": sum(
                1 for t in self.emitted if t.origin == "source"
            ),
            "target_origin_tokens": sum(
                1 for t in self.emitted if t.origin == "target"
            ),
            "cutover_index": self.cutover_index,
            "committed": self.committed,
            "discarded_source_tokens": self._discarded_source,
            "verified_overlap_tokens": self._verified_overlap,
            "handoff_stall_s": self.handoff_stall_s,
            "finished_reason": self.finished_reason,
            "token_ids": self.token_ids(),
            "emitted": [
                {
                    "index": token.index,
                    "token_id": token.token_id,
                    "unix_s": token.unix_s,
                    "origin": token.origin,
                }
                for token in self.emitted
            ],
        }
