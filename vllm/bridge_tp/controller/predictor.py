# SPDX-License-Identifier: Apache-2.0
"""Survival-conditioned remaining-length predictor.

Design note (paper Section 3.5): the contribution is the control policy, not
the predictor. We therefore use an empirical survival-conditioned CCDF built
from a held-out trace split rather than a learned model. This keeps the risk
estimate auditable and removes "did you tune the predictor" as a reviewer
objection.

Given a request that has already produced ``n`` output tokens, we need

    P(N_remain > x | already produced n)

which, writing L for total output length, equals

    P(L > n + x | L > n) = CCDF(n + x) / CCDF(n).

The table stores, for each progress bucket, the sorted array of *remaining*
lengths observed among requests that survived past that bucket. Querying is a
binary search, so the online cost is microseconds.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SurvivalTable:
    """Empirical conditional survival of remaining output length.

    ``bucket_edges`` is a strictly increasing list of progress checkpoints in
    tokens. ``remaining`` holds, per bucket, the sorted remaining lengths of
    every training request that survived past that checkpoint.
    """

    bucket_edges: tuple[int, ...]
    remaining: tuple[tuple[int, ...], ...]
    source: str = ""
    # Longest total output length observed while building the table. Queries
    # past this point have no training support: the last bucket would otherwise
    # answer them with the statistics of a much shorter request, which reads as
    # confidence where there is none. ``None`` means "unbounded", used only by
    # hand-constructed tables in tests.
    max_observed_length: int | None = None

    def __post_init__(self) -> None:
        if len(self.bucket_edges) != len(self.remaining):
            raise ValueError("bucket_edges and remaining must have equal length")
        if list(self.bucket_edges) != sorted(set(self.bucket_edges)):
            raise ValueError("bucket_edges must be strictly increasing and unique")
        for row in self.remaining:
            if list(row) != sorted(row):
                raise ValueError("each remaining row must be sorted ascending")

    # ---- construction -------------------------------------------------
    @classmethod
    def from_output_lengths(
        cls,
        output_lengths: Iterable[int],
        bucket_edges: Sequence[int] = (
            0,
            32,
            64,
            128,
            192,
            256,
            384,
            512,
            768,
            1024,
            1536,
            2048,
        ),
        source: str = "",
    ) -> SurvivalTable:
        lengths = sorted(int(v) for v in output_lengths if int(v) >= 0)
        if not lengths:
            raise ValueError("no training output lengths supplied")
        rows: list[tuple[int, ...]] = []
        for edge in bucket_edges:
            # requests that survived strictly past `edge`
            survivors = [total - edge for total in lengths if total > edge]
            rows.append(tuple(survivors))
        return cls(
            tuple(int(e) for e in bucket_edges),
            tuple(rows),
            source,
            max_observed_length=lengths[-1],
        )

    # ---- query --------------------------------------------------------
    def _bucket_index(self, produced: int) -> int:
        """Largest bucket whose edge is <= produced."""
        idx = bisect.bisect_right(self.bucket_edges, produced) - 1
        return max(0, idx)

    def in_support(self, produced: int) -> bool:
        """Whether any training request ever reached this progress point."""
        if self.max_observed_length is None:
            return True
        return produced < self.max_observed_length

    def _row(self, produced: int) -> tuple[int, ...]:
        if not self.in_support(produced):
            return ()
        return self.remaining[self._bucket_index(produced)]

    def n_survivors(self, produced: int) -> int:
        return len(self._row(produced))

    def p_remaining_gt(self, produced: int, x: float) -> float:
        """P(N_remain > x | produced).

        Returns 0.0 when there is no support at this progress point. Callers
        that need to distinguish "no remaining work" from "no data" must check
        :meth:`in_support` first; :mod:`policy` does exactly that.
        """
        row = self._row(produced)
        if not row:
            return 0.0
        if x <= 0:
            return 1.0
        if math.isinf(x):
            return 0.0
        # number of entries strictly greater than x
        greater = len(row) - bisect.bisect_right(row, x)
        return greater / len(row)

    def expected_remaining(self, produced: int) -> float:
        row = self._row(produced)
        if not row:
            return 0.0
        return sum(row) / len(row)

    def quantile_remaining(self, produced: int, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError("q must be in (0,1)")
        row = self._row(produced)
        if not row:
            return 0.0
        pos = min(len(row) - 1, int(q * len(row)))
        return float(row[pos])

    def uncertainty(self, produced: int) -> float:
        """Interquartile spread of remaining length, in tokens.

        Reported alongside the decision so a wide-spread escalation can be
        distinguished from a confident one in the audit log.
        """
        row = self._row(produced)
        if len(row) < 4:
            return float("inf") if row else 0.0
        return float(
            row[min(len(row) - 1, int(0.75 * len(row)))]
            - row[min(len(row) - 1, int(0.25 * len(row)))]
        )

    # ---- persistence --------------------------------------------------
    def to_json(self) -> dict:
        return {
            "format_version": 1,
            "source": self.source,
            "bucket_edges": list(self.bucket_edges),
            "remaining": [list(row) for row in self.remaining],
            "max_observed_length": self.max_observed_length,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SurvivalTable:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("format_version") != 1:
            raise ValueError("unsupported survival table format")
        cap = raw.get("max_observed_length")
        return cls(
            tuple(int(e) for e in raw["bucket_edges"]),
            tuple(tuple(int(v) for v in row) for row in raw["remaining"]),
            str(raw.get("source", "")),
            None if cap is None else int(cap),
        )
