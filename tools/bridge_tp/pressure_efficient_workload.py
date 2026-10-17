# SPDX-License-Identifier: Apache-2.0
"""Targeted operating-region collection; no migration or routing decisions."""

from __future__ import annotations

import json
import time
from collections import Counter

POOLS = ("source", "target")


def collection_selection(plan, last_only=False):
    """Replay the interrupted final job with its original repetition identity."""
    if not last_only:
        return plan["scenes"], list(range(1, plan["repetitions"] + 1))
    selected = [s for s in plan["scenes"] if s["name"] == "region-short-n8-n24"]
    if len(selected) != 1 or plan["repetitions"] != 2:
        raise ValueError("interrupted scene/repetition identity differs")
    return selected, [2]


def validate_plan(plan, inputs):
    from tools.bridge_tp.run_pressure_calibration import payload

    if (
        plan["schema"] != "PRESSURE_EFFICIENT_REGION_V1"
        or plan["migration_enabled"] is not False
        or plan["repetitions"] != 2
        or plan["itl_s"] != 0.1
        or plan["slow_fraction"] != 0.01
        or len(plan["scenes"]) != 5
        or len({s["name"] for s in plan["scenes"]}) != 5
    ):
        raise ValueError("efficient-region plan differs from collection contract")
    schedules = []
    for scene in plan["scenes"]:
        if (
            scene.get("efficient_region") is not True
            or scene["headroom_blocks"] != 125
            or scene["min_common_decode_s"] < 20
            or not 1 <= scene["source_count"] <= 8
            or not 1 <= scene["target_count"] <= 32
        ):
            raise ValueError("invalid collection scope/headroom")
        for pool in POOLS:
            payload(
                inputs,
                scene.get(pool + "_prompt_tokens", scene["prompt_tokens"]),
                scene.get(pool + "_max_tokens", scene["max_tokens"]),
                "validate",
            )
        if "fixed_schedule" not in scene:
            if "injection_prompt_tokens" in scene:
                raise ValueError("injections require a fixed arrival schedule")
            continue
        schedule = scene["fixed_schedule"]
        if (
            len(scene["injection_prompt_tokens"]) != 1
            or scene["injection_max_tokens"] != 1
            or scene["max_inflight_injections"] != 2
            or schedule["interval_s"] <= 0
            or schedule["count"] < 1
            or schedule["duration_s"] != 48
            or schedule["interval_s"] * schedule["count"] != 48
            or schedule["max_dispatch_lag_s"] != 1.0
        ):
            raise ValueError("invalid fixed-arrival contract")
        size = scene["injection_prompt_tokens"][0]
        payload(inputs, size, 1, "validate")
        schedules.append((size, schedule, scene))
    if len(schedules) != 2:
        raise ValueError("two frequency contrasts required")
    a, b = schedules
    if a[0] == b[0] or a[0] * a[1]["count"] != b[0] * b[1]["count"]:
        raise ValueError("frequency pair must offer equal total prompt tokens")
    for key in (
        "source_count",
        "target_count",
        "prompt_tokens",
        "max_tokens",
        "target_max_tokens",
        "settle_s",
    ):
        if a[2][key] != b[2][key]:
            raise ValueError("frequency backgrounds differ")


def drive(work, scene, repetition, root):
    from tools.bridge_tp.run_pressure_calibration import io

    deadline = time.monotonic() + work.timeout_s
    background = []
    for pool in POOLS:
        background.extend(
            work.submit(
                pool,
                scene.get(pool + "_prompt_tokens", scene["prompt_tokens"]),
                scene.get(pool + "_max_tokens", scene["max_tokens"]),
                f"r{repetition}-{pool}-background-{index}",
            )[0]
            for index in range(scene[pool + "_count"])
        )

    def rows():
        return {r["id"]: r for r in work.snapshot()}

    def live_background():
        current = rows()
        if any(current[k]["done"] for k in background):
            raise RuntimeError("background ended before the controlled phase finished")

    def background_ready():
        current = rows()
        return all(current[k]["first_token_unix_s"] is not None for k in background)

    work.wait(background_ready, deadline)
    phase = dict(background_ids=background, injection_arrivals=[])
    # Persist a partial receipt before any scheduled work can fail.
    io.atomic_json_dump(phase, root / "controlled_phase.json")
    if "fixed_schedule" in scene:
        settle_end = time.monotonic() + scene["settle_s"]
        work.phase = "等待背景稳定"
        work.wait(lambda: time.monotonic() >= settle_end, deadline)
        live_background()
        schedule = scene["fixed_schedule"]
        origin_mono, origin_wall = time.monotonic(), time.time()
        phase.update(
            started_unix_s=origin_wall,
            ended_unix_s=origin_wall + schedule["duration_s"],
        )
        work.phase = "固定频率prefill对照"
        inflight = {p: [] for p in POOLS}
        for wave in range(schedule["count"]):
            offset = wave * schedule["interval_s"]
            work.wait(
                lambda due=origin_mono + offset: time.monotonic() >= due, deadline
            )
            live_background()
            for pool in POOLS:
                active = [f for f in inflight[pool] if not f.done()]
                if len(active) >= scene["max_inflight_injections"]:
                    raise RuntimeError("fixed arrivals exceeded test KV envelope")
                label, future = work.submit(
                    pool,
                    scene["injection_prompt_tokens"][0],
                    1,
                    f"r{repetition}-{pool}-prefill-{wave}",
                )
                inflight[pool] = active + [future]
                phase["injection_arrivals"].append(
                    dict(
                        id=label,
                        pool=pool,
                        planned_unix_s=origin_wall + offset,
                    )
                )
            io.atomic_json_dump(phase, root / "controlled_phase.json")
        work.wait(
            lambda: time.monotonic() >= origin_mono + schedule["duration_s"], deadline
        )
        live_background()
        phase["actual_end_check_unix_s"] = time.time()
    work.phase = "等待全部请求完整收尾"
    work.wait(lambda: all(f.done() for f in work.futures), deadline)
    work.phase = "完成"
    io.atomic_json_dump(phase, root / "controlled_phase.json")
    io.atomic_json_dump(
        dict(status="COMPLETED", migration_starts=0, requests=work.snapshot()),
        root / "request_results.json",
    )


def union_duration(intervals):
    total, end = 0.0, float("-inf")
    for a, b in sorted(intervals):
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total


def assess_coverage(root, scene):
    """Check actual common output lifetime and native batches, never nominal N."""
    from tools.bridge_tp.verify_performance_trace import read_journal

    result = json.loads((root / "request_results.json").read_text())
    requests = {r["id"]: r for r in result["requests"]}
    phase = json.loads((root / "controlled_phase.json").read_text())
    receipt = dict(status="COVERED", pools={}, dispatch=[])
    mixed = "fixed_schedule" in scene
    for pool in POOLS:
        background = [
            requests[k] for k in phase["background_ids"] if requests[k]["pool"] == pool
        ]
        start = max(r["first_token_unix_s"] for r in background)
        end = min(r["last_token_unix_s"] for r in background)
        left = phase["started_unix_s"] if mixed else start
        right = phase["ended_unix_s"] if mixed else end
        engine = [
            r
            for r in read_journal(
                root / "controller" / "performance_trace" / pool / "engine.jsonl"
            )
            if r.get("kind") == "engine_batch" and left <= r["scheduled_unix_s"] < right
        ]
        pure = [
            r
            for r in engine
            if not r["prefill_tokens"]
            and not r["recompute_tokens"]
            and r["decode_requests"]
        ]
        matching = [r for r in pure if r["decode_requests"] == scene[pool + "_count"]]
        purity = len(matching) / len(pure) if pure else 0
        span = (
            (
                max(r["scheduled_unix_s"] for r in matching)
                - min(r["scheduled_unix_s"] for r in matching)
            )
            if matching
            else 0
        )
        covered = (
            end - start >= scene["min_common_decode_s"]
            and start <= left
            and end >= right
            and len(matching) >= 20
            and purity >= 0.9
            and span >= scene["min_common_decode_s"] - 2
        )
        prefill = [
            (max(left, r["scheduled_unix_s"]), min(right, r["completed_unix_s"]))
            for r in engine
            if r["prefill_tokens"] and r["completed_unix_s"] > left
        ]
        contexts = Counter(
            int(r["decode_context_sum"] / r["decode_requests"] // 2048)
            for r in matching
        )
        prefill_tokens = sum(r["prefill_tokens"] for r in engine)
        expected_prefill = (
            scene["injection_prompt_tokens"][0] * scene["fixed_schedule"]["count"]
            if mixed
            else 0
        )
        if mixed and (
            prefill_tokens != expected_prefill
            or any(r["completed_unix_s"] > right for r in engine if r["prefill_tokens"])
        ):
            covered = False
        receipt["pools"][pool] = dict(
            covered=covered,
            expected_nd=scene[pool + "_count"],
            common_output_overlap_s=end - start,
            phase_start_unix_s=left,
            phase_end_unix_s=right,
            native_pure_samples=len(pure),
            matching_nd_samples=len(matching),
            matching_nd_fraction=purity,
            matching_sample_span_s=span,
            context_bins_2048=dict(contexts),
            phase_prefill_tokens=prefill_tokens,
            expected_phase_prefill_tokens=expected_prefill,
            measured_prefill_duty=union_duration(prefill) / (right - left)
            if right > left
            else None,
        )
        if not covered:
            receipt["status"] = "COVERAGE_GAP"
    for arrival in phase["injection_arrivals"]:
        request = requests[arrival["id"]]
        lag = request["response"]["request_started_unix_s"] - arrival["planned_unix_s"]
        receipt["dispatch"].append(dict(**arrival, actual_http_lag_s=lag))
        if lag < 0 or lag > scene["fixed_schedule"]["max_dispatch_lag_s"]:
            receipt["status"] = "COVERAGE_GAP"
    receipt["note"] = (
        "Collection coverage only; inspect context/window support and measured duty. "
        "Equal offered prompt tokens do not guarantee equal prefill duty. "
        "Different injection counts require separate background and total goodoutput."
    )
    return receipt
