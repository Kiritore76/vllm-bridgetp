# SPDX-License-Identifier: Apache-2.0
"""Fill only the actual post-migration Nd2 reference gaps, without migration."""


def validate(plan, inputs):
    from tools.bridge_tp.run_pressure_calibration import payload

    expected = dict(
        name="health-nd2",
        efficient_region=True,
        health_nd2=True,
        prompt_tokens=2048,
        max_tokens=4096,
        target_max_tokens=6144,
        source_count=2,
        target_count=2,
        headroom_blocks=125,
        min_common_decode_s=20,
    )
    if (
        plan["schema"] != "PRESSURE_HEALTH_ND2_V1"
        or plan["repetitions"] != 2
        or plan["migration_enabled"] is not False
        or plan["scenes"] != [expected]
        or plan["seed"] != 20261015
        or plan["maximum_gap_s"] != 1
        or plan["itl_s"] != 0.1
        or plan["slow_fraction"] != 0.01
    ):
        raise ValueError("Nd2 supplement differs from frozen two-run plan")
    for output in (4096, 6144):
        request = payload(inputs, 2048, output, "validate")
        if request["seed"] != plan["seed"]:
            raise ValueError("sampling seed differs")


def coverage(root, scene):
    import collections
    import json

    from tools.bridge_tp.verify_performance_trace import read_journal

    result = {}
    origin = json.loads((root / "measurement_window.json").read_text())[
        "started_unix_s"
    ]
    requests = json.loads((root / "request_results.json").read_text())["requests"]
    for pool, expected in (("source", {1, 2}), ("target", {1, 2, 3})):
        streams = [r for r in requests if r["pool"] == pool]
        if len(streams) != 2:
            raise ValueError("Nd2 collection requires exactly two requests per pool")
        start = max(r["first_token_unix_s"] for r in streams)
        end = min(r["last_token_unix_s"] for r in streams)
        engine = [
            r
            for r in read_journal(
                root / "controller/performance_trace" / pool / "engine.jsonl"
            )
            if r.get("kind") == "engine_batch"
        ]
        by_window = collections.defaultdict(list)
        for row in engine:
            if row["scheduled_unix_s"] >= origin:
                by_window[int((row["scheduled_unix_s"] - origin) // 10)].append(row)
        qualified = collections.Counter()
        for index, values in by_window.items():
            if origin + index * 10 < start or origin + (index + 1) * 10 > end:
                continue
            stamps = sorted(
                [origin + index * 10, origin + (index + 1) * 10]
                + [r["scheduled_unix_s"] for r in values]
            )
            if max(b - a for a, b in zip(stamps, stamps[1:])) > 2:
                continue
            if any(r["prefill_tokens"] or r["recompute_tokens"] for r in values):
                continue
            matched = [r for r in values if r["decode_requests"] == 2]
            if len(matched) < 3 or len(matched) / len(values) < 0.8:
                continue
            bins = collections.Counter(
                int(r["decode_context_sum"] / 2 // 2048) for r in matched
            )
            cb, count = bins.most_common(1)[0]
            span = max(r["scheduled_unix_s"] for r in matched) - min(
                r["scheduled_unix_s"] for r in matched
            )
            if count / len(matched) >= 0.8 and span >= 8:
                qualified[cb] += 1
        result[pool] = dict(
            expected_context_bins=sorted(expected),
            qualified_10s_windows=dict(qualified),
            covered=all(qualified[cb] >= 2 for cb in expected),
        )
    return dict(
        status="COVERED"
        if all(v["covered"] for v in result.values())
        else "COVERAGE_GAP",
        pools=result,
        note="require two complete Nd2 windows per bin per run; retain all runs",
    )
