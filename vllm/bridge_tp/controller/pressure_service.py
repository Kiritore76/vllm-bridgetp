# SPDX-License-Identifier: Apache-2.0
"""Observed local engine-span lookups, never a net-benefit predictor."""

import math


class LocalServiceLookup:
    def __init__(self, summary, minimum_runs=2, minimum_samples=5):
        if summary.get("schema") != "PRESSURE_EVIDENCE_V1":
            raise ValueError("invalid service evidence")
        self.cells = summary["service_lookup"]
        self.minimum_runs, self.minimum_samples = minimum_runs, minimum_samples

    def estimate(self, pool, nd, kd, prefill=0):
        if nd == kd == prefill == 0:
            return 0.0
        if min(nd, kd, prefill) < 0:
            return None
        phase = (
            "decode"
            if not prefill
            else "prefill_lt256"
            if prefill < 256
            else "prefill_256_767"
            if prefill < 768
            else "prefill_768_1535"
            if prefill < 1536
            else "prefill_ge1536"
        )
        for cell in self.cells:
            if (
                cell["pool"] == pool
                and cell["phase"] == phase
                and cell["decode_requests"] == nd
                and cell["mean_context_bin_1024"] == int((kd / nd if nd else 0) // 1024)
                and cell["prefill_bin_256"] == math.ceil(prefill / 256)
                and cell["runs"] >= self.minimum_runs
                and cell["samples"] >= self.minimum_samples
                and cell["context_sum_range"]["min"]
                <= kd
                <= cell["context_sum_range"]["max"]
                and cell["prefill_tokens_range"]["min"]
                <= prefill
                <= cell["prefill_tokens_range"]["max"]
            ):
                return cell["engine_ms"]["p95"] / 1000
        return None

    def decode_envelope(self, pool, nd, kd):
        """Empirical P95 envelope from observed heavier decode shapes.

        Assumes nondecreasing cost in request count and mean context. This is
        a bounded-domain working estimate, not a statistical/physical upper
        bound. Preserve the observed maximum separately, including stalls.
        """
        if nd == kd == 0:
            return dict(upper_s=0.0, tail_max_s=0.0, cells=[])
        if nd <= 0 or kd < 0:
            return None
        cells = [
            (index, cell)
            for index, cell in enumerate(self.cells)
            if cell["pool"] == pool
            and cell["phase"] == "decode"
            and cell["runs"] >= self.minimum_runs
            and cell["samples"] >= self.minimum_samples
            and cell["decode_requests"] >= nd
            and cell["decode_requests"] > 0
            and cell["context_sum_range"]["min"] / cell["decode_requests"] >= kd / nd
        ]
        if not cells:
            return None
        return dict(
            upper_s=max(cell["engine_ms"]["p95"] for _, cell in cells) / 1000,
            tail_max_s=max(cell["engine_ms"]["max"] for _, cell in cells) / 1000,
            cells=[index for index, _ in cells],
        )

    def deltas(self, source, target, request_id, reference_s=0.1):
        row = next(
            (r for r in source.get("requests", []) if r["request_id"] == request_id),
            None,
        )
        if (
            row is None
            or row["decode_tokens"] != 1
            or row["prefill_tokens"]
            or row["recompute_tokens"]
            or source.get("recompute_tokens")
            or target.get("recompute_tokens")
        ):
            return None, None
        n, k, context = (
            source["decode_requests"],
            source["decode_context_sum"],
            row["computed_before"],
        )
        tn, tk = target["decode_requests"], target["decode_context_sum"]
        v = [
            self.estimate("source", n, k, source["prefill_tokens"]),
            self.estimate("source", n - 1, k - context, source["prefill_tokens"]),
            self.estimate("target", tn, tk, target["prefill_tokens"]),
            self.estimate("target", tn + 1, tk + context, target["prefill_tokens"]),
        ]
        if any(x is None for x in v):
            # Do not infer positive source relief from a sparse lookup. Budget
            # the entire post-move target step, rather than an optimistic delta.
            source_support = self.decode_envelope("source", n, k)
            target_support = self.decode_envelope("target", tn + 1, tk + context)
            if (
                source["prefill_tokens"]
                or target["prefill_tokens"]
                or (source_support is None or target_support is None)
            ):
                return None, None
            return 0.0, target_support["upper_s"] / reference_s
        return max(0, v[0] - v[1]) / reference_s, max(0, v[3] - v[2]) / reference_s

    def admission_cost(self, pool, batch, prompt_tokens, reference_s=0.1):
        return self.admission_details(pool, batch, prompt_tokens, reference_s)["cost"]

    def admission_details(self, pool, batch, prompt_tokens, reference_s=0.1):
        if batch.get("recompute_tokens"):
            return dict(cost=None, basis="recompute unsupported")
        n, k, prefill = (
            batch["decode_requests"],
            batch["decode_context_sum"],
            batch["prefill_tokens"],
        )
        before = self.estimate(pool, n, k, prefill)
        after = self.estimate(pool, n + 1, k + prompt_tokens, prefill)
        if before is None or after is None:
            support = self.decode_envelope(pool, n + 1, k + prompt_tokens)
            if prefill or support is None:
                return dict(cost=None, basis="outside empirical decode support")
            return dict(
                cost=support["upper_s"] / reference_s,
                basis="heavier-observed-shape P95 envelope; not a hard bound",
                **support,
            )
        return dict(
            cost=max(0, after - before) / reference_s,
            basis="exact local P95 difference",
        )
