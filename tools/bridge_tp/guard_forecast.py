# SPDX-License-Identifier: Apache-2.0
"""Finite future prefill allocation; no additional deduction from current free."""
import math


def unallocated_prefill_tokens(requests, manager, block_size, max_model_len):
    """Return future prompt allocation for the supported single KV group."""
    total = 0
    for request in requests:
        prompt = min(request.num_prompt_tokens, max_model_len)
        if request.num_computed_tokens >= prompt:
            continue
        groups = manager.get_blocks(request.request_id).blocks
        if len(groups) != 1:
            return None
        required = math.ceil(prompt / block_size) * block_size
        allocated = len(groups[0]) * block_size
        total += max(0, required - allocated)
    return total


def guard_time(headroom, decode_rate, prefill_rate, unallocated):
    """Solve d*t + min(P, p*t) >= H; None denotes missing evidence."""
    if headroom <= 0:
        return 0.0
    if (not isinstance(decode_rate, (int, float))
            or not math.isfinite(decode_rate) or decode_rate < 0
            or not isinstance(unallocated, int) or unallocated < 0):
        return None
    if unallocated == 0:
        return headroom / decode_rate if decode_rate > 0 else math.inf
    if (not isinstance(prefill_rate, (int, float))
            or not math.isfinite(prefill_rate) or prefill_rate <= 0):
        return None
    combined = headroom / (decode_rate + prefill_rate)
    if combined <= unallocated / prefill_rate:
        return combined
    return ((headroom - unallocated) / decode_rate
            if decode_rate > 0 else math.inf)


def snapshot_guard_time(snapshot, headroom, scale=1.0):
    pending = snapshot.get("source_prefill_pending_kv_tokens")
    unallocated = snapshot.get("source_prefill_unallocated_kv_tokens")
    # A completed prefill needs no future allocation, even if old traces lack
    # block telemetry. Nonzero unfinished prompts require allocation evidence.
    if unallocated is None and pending == 0:
        unallocated = 0
    decode = snapshot.get("source_decode_growth_tokens_s")
    prefill = snapshot.get("source_prefill_growth_tokens_s")
    return guard_time(headroom,
                      decode * scale if isinstance(decode, (int, float)) else None,
                      prefill * scale if isinstance(prefill, (int, float)) else None,
                      unallocated)
