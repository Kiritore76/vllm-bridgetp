# SPDX-License-Identifier: Apache-2.0
"""Audit token-level GoodOutput in one Phase 9 online result archive.

The migrated anchor is counted once from the externally visible proxy stream.
Source/target internal responses are used only for its request boundaries.
No STAY counterfactual or migration benefit is inferred by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path
from typing import Any


NEEDED = (
    "online/contract.json",
    "background/background_summary.json",
    "controller/response_proxy_stats.json",
    "controller/source_response.json",
)


def _read_members(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Read JSON without extracting archive members or following archive links."""
    values: dict[str, Any] = {}
    locations: dict[str, str] = {}
    wanted = (*NEEDED, "controller/target_response.json")
    if path.is_file():
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                for suffix in wanted:
                    if member.name.endswith("/" + suffix):
                        if suffix in values:
                            raise ValueError(f"multiple members match {suffix}")
                        stream = archive.extractfile(member)
                        assert stream is not None
                        values[suffix] = json.load(stream)
                        locations[suffix] = member.name
    elif path.is_dir():
        for suffix in wanted:
            matches = list(path.rglob(suffix))
            if len(matches) > 1:
                raise ValueError(f"multiple files match {suffix}")
            if matches:
                values[suffix] = json.loads(matches[0].read_text(encoding="utf-8"))
                locations[suffix] = str(matches[0])
    else:
        raise ValueError(f"input is not an archive or directory: {path}")
    return values, locations


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percentile(values: list[float], fraction: float) -> float | None:
    """Match the online background runner's interpolated percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _positive_thresholds(contract: dict[str, Any], errors: list[str]) -> dict[str, float]:
    raw = contract.get("slo_thresholds") or {}
    values: dict[str, float] = {}
    for name in ("tpot_ms", "ttft_ms", "e2e_ms", "handoff_ms"):
        number = _number(raw.get(name))
        if number is None or number <= 0:
            errors.append(f"contract missing positive slo_thresholds.{name}")
        else:
            values[name] = number
    return values


def _score_request(
    *,
    request_id: str,
    pool: str,
    status: str,
    expected_tokens: Any,
    started: Any,
    ended: Any,
    times: Any,
    thresholds: dict[str, float],
    v3_ttft_ms: float,
    v3_mean_tpot_ms: float,
    errors: list[str],
    handoff_ms: Any = None,
) -> dict[str, Any]:
    prefix = f"{pool}/{request_id}"
    row: dict[str, Any] = {
        "request_id": request_id, "pool": pool, "status": status,
        "output_tokens": expected_tokens,
        "good_tokens": None, "bad_intervals": None,
        "goodoutput_v3_success": None,
        "goodoutput_v4_success": None,
        "goodoutput_v5_success": None,
        "request_slo_success_p99": None,
        "strict_slo_success": None,
    }
    start, end = _number(started), _number(ended)
    if start is None or end is None or end < start:
        errors.append(f"{prefix}: request start/end timestamps are invalid")
        return row
    if status != "COMPLETED":
        if not isinstance(status, str) or status in ("", "MISSING"):
            errors.append(f"{prefix}: request status is missing")
            return row
        row.update({
            "output_tokens": expected_tokens if isinstance(expected_tokens, int)
            and not isinstance(expected_tokens, bool) and expected_tokens >= 0 else 0,
            "started_unix_s": start, "ended_unix_s": end,
            "good_tokens": 0, "bad_intervals": 0,
            "goodoutput_v3_success": False,
            "goodoutput_v4_success": False,
            "goodoutput_v5_success": False,
            "request_slo_success_p99": False,
            "strict_slo_success": False,
        })
        return row
    if isinstance(expected_tokens, bool) or not isinstance(expected_tokens, int) or expected_tokens < 1:
        errors.append(f"{prefix}: output_tokens is missing or invalid")
        return row
    if not isinstance(times, list) or len(times) != expected_tokens:
        errors.append(f"{prefix}: token timestamps do not match output_tokens")
        return row
    token_times = [_number(value) for value in times]
    if any(value is None for value in token_times):
        errors.append(f"{prefix}: nonfinite token timestamp")
        return row
    assert all(value is not None for value in token_times)
    if token_times[0] < start - 0.001 or token_times[-1] > end + 0.001:
        errors.append(f"{prefix}: token timestamps lie outside request boundaries")
        return row
    if any(b < a for a, b in zip(token_times, token_times[1:])):
        errors.append(f"{prefix}: token timestamps are not monotonic")
        return row
    if len(thresholds) != 4:
        return row
    ttft_ms = (token_times[0] - start) * 1000
    e2e_ms = (end - start) * 1000
    gaps_ms = [(b - a) * 1000 for a, b in zip(token_times, token_times[1:])]
    bad_intervals = sum(gap > thresholds["tpot_ms"] for gap in gaps_ms)
    bad_interval_rate = bad_intervals / len(gaps_ms) if gaps_ms else 0.0
    mean_itl_ms = sum(gaps_ms) / len(gaps_ms) if gaps_ms else None
    p99_itl_ms = _percentile(gaps_ms, 0.99)
    request_valid = ttft_ms <= thresholds["ttft_ms"] and e2e_ms <= thresholds["e2e_ms"]
    if pool == "anchor":
        handoff = _number(handoff_ms)
        if handoff is None or handoff < 0:
            errors.append(f"{prefix}: handoff time is missing or invalid")
            return row
        row["handoff_ms"] = handoff
        handoff_valid = handoff <= thresholds["handoff_ms"]
    else:
        handoff_valid = True
    row.update({
        "started_unix_s": start,
        "ended_unix_s": end,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "ttft_violation": ttft_ms > thresholds["ttft_ms"],
        "e2e_violation": e2e_ms > thresholds["e2e_ms"],
        "max_itl_ms": max(gaps_ms) if gaps_ms else None,
        "mean_itl_ms": mean_itl_ms,
        "p99_itl_ms": p99_itl_ms,
        "bad_intervals": bad_intervals,
        "bad_interval_rate": bad_interval_rate,
        "good_tokens": expected_tokens - bad_intervals if request_valid else 0,
        "goodoutput_v3_success": (
            ttft_ms <= v3_ttft_ms
            and e2e_ms <= thresholds["e2e_ms"]
            and (mean_itl_ms is None or mean_itl_ms <= v3_mean_tpot_ms)
            and handoff_valid
        ),
        "goodoutput_v4_success": (
            e2e_ms <= thresholds["e2e_ms"]
            and (mean_itl_ms is None or mean_itl_ms <= v3_mean_tpot_ms)
            and handoff_valid
        ),
        "goodoutput_v5_success": (
            e2e_ms <= thresholds["e2e_ms"]
            and (mean_itl_ms is None or mean_itl_ms <= v3_mean_tpot_ms)
            and bad_interval_rate <= 0.01
            and (not gaps_ms or max(gaps_ms) <= thresholds["handoff_ms"])
            and handoff_valid
        ),
        "request_slo_success_p99": (
            request_valid and handoff_valid
            and (p99_itl_ms is None or p99_itl_ms <= thresholds["tpot_ms"])
        ),
        "strict_slo_success": request_valid and bad_intervals == 0 and handoff_valid,
    })
    return row


def audit_payload(
    values: dict[str, Any], *,
    v3_ttft_ms: float = 3000.0,
    v3_mean_tpot_ms: float = 50.0,
) -> dict[str, Any]:
    if any(not math.isfinite(value) or value <= 0
           for value in (v3_ttft_ms, v3_mean_tpot_ms)):
        raise ValueError("v3 SLO thresholds must be positive and finite")
    errors: list[str] = []
    missing = [name for name in NEEDED if name not in values]
    if missing:
        return {"format_version": 1, "computable": False,
                "errors": [f"missing {name}" for name in missing]}
    contract = values["online/contract.json"]
    background = values["background/background_summary.json"]
    proxy = values["controller/response_proxy_stats.json"]
    source = values["controller/source_response.json"]
    if not all(isinstance(value, dict) for value in (contract, background, proxy, source)):
        return {"format_version": 1, "computable": False,
                "errors": ["required JSON document is not an object"]}
    thresholds = _positive_thresholds(contract, errors)
    results = background.get("results")
    if not isinstance(results, list):
        results = []
        errors.append("background results list is missing")
    if background.get("jobs") != len(results):
        errors.append("background jobs count differs from result rows")
    if background.get("completed") != sum(
        isinstance(result, dict) and result.get("status") == "COMPLETED"
        for result in results
    ):
        errors.append("background completed count differs from result rows")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            errors.append("background result is not an object")
            continue
        request_id = result.get("job_id")
        pool = result.get("pool")
        if not isinstance(request_id, str) or not request_id or request_id in ids:
            errors.append("background job id is missing or duplicated")
            continue
        ids.add(request_id)
        if pool not in ("source", "target"):
            errors.append(f"{request_id}: unknown pool {pool!r}")
            continue
        rows.append(_score_request(
            request_id=request_id, pool=pool,
            status=result.get("status", "MISSING"),
            expected_tokens=result.get("output_tokens"),
            started=result.get("request_started_unix_s"),
            ended=result.get("request_ended_unix_s"),
            times=result.get("token_times_unix_s"),
            thresholds=thresholds, v3_ttft_ms=v3_ttft_ms,
            v3_mean_tpot_ms=v3_mean_tpot_ms, errors=errors,
        ))
    emitted = proxy.get("emitted")
    anchor_times: list[Any] | None = None
    if isinstance(emitted, list):
        if any(not isinstance(item, dict) or item.get("index") != index
               for index, item in enumerate(emitted)):
            errors.append("anchor visible token indices are not contiguous")
        else:
            anchor_times = [item.get("unix_s") for item in emitted]
    else:
        errors.append("anchor visible emitted list is missing")
    target = values.get("controller/target_response.json")
    if target is not None and not isinstance(target, dict):
        errors.append("target response JSON is not an object")
        target = None
    anchor_end = (
        target.get("completed_unix_s") if target else source.get("completed_unix_s")
    )
    anchor_status = "COMPLETED" if (
        anchor_times and proxy.get("emitted_tokens") == len(anchor_times)
        and (target or source).get("finish_reason") in ("stop", "length")
    ) else "INCOMPLETE"
    rows.append(_score_request(
        request_id=str(proxy.get("external_request_id", "anchor")),
        pool="anchor", status=anchor_status,
        expected_tokens=proxy.get("emitted_tokens"),
        started=source.get("request_started_unix_s"), ended=anchor_end,
        times=anchor_times, thresholds=thresholds,
        v3_ttft_ms=v3_ttft_ms, v3_mean_tpot_ms=v3_mean_tpot_ms,
        errors=errors,
        handoff_ms=(
            0.0
            if (proxy.get("committed") is False
                and proxy.get("target_origin_tokens") == 0
                and target is None
                and ((contract.get("paired_stay") is True)
                     or (proxy.get("source_origin_tokens") == len(
                         source.get("token_ids") or [])
                         and source.get("finish_reason") in ("stop", "length"))))
            else (
                _number(proxy.get("handoff_stall_s")) * 1000
                if _number(proxy.get("handoff_stall_s")) is not None else None
            )
        ),
    ))
    computable = not errors and all(row["good_tokens"] is not None for row in rows)
    by_pool: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] | None = None
    if computable:
        started = min(row["started_unix_s"] for row in rows)
        ended = max(row["ended_unix_s"] for row in rows)
        wall_s = ended - started
        if wall_s <= 0:
            errors.append("shared experiment wall time is not positive")
            computable = False
        else:
            for pool in ("anchor", "source", "target", "system"):
                subset = rows if pool == "system" else [row for row in rows if row["pool"] == pool]
                raw = sum(row["output_tokens"] for row in subset)
                good = sum(row["good_tokens"] for row in subset)
                strict = sum(row["output_tokens"] for row in subset if row["strict_slo_success"])
                request_slo = sum(
                    row["output_tokens"] for row in subset
                    if row["request_slo_success_p99"]
                )
                v3_good = sum(
                    row["output_tokens"] for row in subset
                    if row["goodoutput_v3_success"]
                )
                v4_good = sum(
                    row["output_tokens"] for row in subset
                    if row["goodoutput_v4_success"]
                )
                v5_good = sum(
                    row["output_tokens"] for row in subset
                    if row["goodoutput_v5_success"]
                )
                intervals = sum(max(0, row["output_tokens"] - 1) for row in subset
                                if row["status"] == "COMPLETED")
                bad = sum(row["bad_intervals"] for row in subset)
                strict_requests = sum(bool(row["strict_slo_success"]) for row in subset)
                request_slo_successes = sum(
                    bool(row["request_slo_success_p99"]) for row in subset
                )
                v3_successes = sum(
                    bool(row["goodoutput_v3_success"]) for row in subset
                )
                v4_successes = sum(
                    bool(row["goodoutput_v4_success"]) for row in subset
                )
                v5_successes = sum(
                    bool(row["goodoutput_v5_success"]) for row in subset
                )
                by_pool[pool] = {
                    "requests": len(subset), "output_tokens": raw,
                    "good_tokens": good,
                    "goodoutput_v3_tokens": v3_good,
                    "goodoutput_v3_success_requests": v3_successes,
                    "goodoutput_v3_success_rate": (
                        v3_successes / len(subset) if subset else None
                    ),
                    "goodoutput_v4_tokens": v4_good,
                    "goodoutput_v4_success_requests": v4_successes,
                    "goodoutput_v4_success_rate": (
                        v4_successes / len(subset) if subset else None
                    ),
                    "goodoutput_v5_tokens": v5_good,
                    "goodoutput_v5_success_requests": v5_successes,
                    "goodoutput_v5_success_rate": (
                        v5_successes / len(subset) if subset else None
                    ),
                    "strict_request_good_tokens": strict,
                    "request_slo_p99_good_tokens": request_slo,
                    "request_slo_p99_success_requests": request_slo_successes,
                    "request_slo_p99_success_rate": (
                        request_slo_successes / len(subset) if subset else None
                    ),
                    "strict_success_requests": strict_requests,
                    "strict_success_rate": strict_requests / len(subset) if subset else None,
                    "token_intervals": intervals,
                    "bad_intervals": bad,
                    "bad_interval_rate": bad / intervals if intervals else None,
                    "ttft_violations": sum(bool(row.get("ttft_violation")) for row in subset),
                    "e2e_violations": sum(bool(row.get("e2e_violation")) for row in subset),
                    "max_itl_ms": max(
                        (row["max_itl_ms"] for row in subset
                         if row.get("max_itl_ms") is not None), default=None,
                    ),
                    "goodoutput_tokens_s": good / wall_s,
                    "goodoutput_v3_tokens_s": v3_good / wall_s,
                    "goodoutput_v4_tokens_s": v4_good / wall_s,
                    "goodoutput_v5_tokens_s": v5_good / wall_s,
                    "request_slo_p99_goodput_tokens_s": request_slo / wall_s,
                    "strict_request_goodput_tokens_s": strict / wall_s,
                }
            metrics = {"start_unix_s": started, "end_unix_s": ended,
                       "wall_time_s": wall_s, "by_pool": by_pool}
    return {
        "format_version": 1,
        "metric_definition": "completed_request_ttft_e2e_and_per_token_itl_v1",
        "candidate_metric": "goodoutput_v3_tokens_s",
        "candidate_metric_definition": "completed_request_ttft_mean_tpot_e2e_handoff_v3",
        "candidate_note": (
            "3 s TTFT is a sensitivity threshold; the formal SLO requires "
            "normal-arrival workload and service-objective validation"
        ),
        "primary_metric": "goodoutput_v5_tokens_s",
        "primary_metric_definition": (
            "completed_request_mean_tpot_bad_interval_rate_max_gap_e2e_handoff_without_ttft_v5"
        ),
        "v5_thresholds": {
            "mean_tpot_ms": v3_mean_tpot_ms,
            "max_bad_interval_rate": 0.01,
            "max_visible_interval_ms": thresholds.get("handoff_ms"),
            "e2e_ms": thresholds.get("e2e_ms"),
            "handoff_ms": thresholds.get("handoff_ms"),
        },
        "v4_thresholds": {
            "mean_tpot_ms": v3_mean_tpot_ms,
            "e2e_ms": thresholds.get("e2e_ms"),
            "handoff_ms": thresholds.get("handoff_ms"),
        },
        "v3_thresholds": {
            "ttft_ms": v3_ttft_ms, "mean_tpot_ms": v3_mean_tpot_ms,
            "e2e_ms": thresholds.get("e2e_ms"),
            "handoff_ms": thresholds.get("handoff_ms"),
        },
        "request_slo_definition": "ttft_e2e_p99_itl_handoff_v2",
        "computable": computable,
        "slo_thresholds": thresholds,
        "errors": errors,
        "request_rows": rows,
        "metrics": metrics,
        "benefit_vs_stay": None,
        "benefit_reason": "no matched STAY counterfactual in this archive",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="one online result .tar.gz or extracted run directory")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--v3-ttft-ms", type=float, default=3000.0)
    parser.add_argument("--mean-tpot-ms", "--v3-mean-tpot-ms",
                        dest="mean_tpot_ms", type=float, default=50.0)
    args = parser.parse_args()
    values, locations = _read_members(args.input)
    report = audit_payload(
        values, v3_ttft_ms=args.v3_ttft_ms,
        v3_mean_tpot_ms=args.mean_tpot_ms,
    )
    report["input"] = str(args.input.resolve())
    report["member_locations"] = locations
    if args.input.is_file():
        with args.input.open("rb") as stream:
            report["input_sha256"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "input": report["input"], "computable": report["computable"],
        "errors": report["errors"],
        "system": (report.get("metrics") or {}).get("by_pool", {}).get("system"),
        "out_json": str(args.out_json.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
