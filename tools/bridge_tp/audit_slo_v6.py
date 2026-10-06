# SPDX-License-Identifier: Apache-2.0
"""Score one online run using a versioned length-aware SLO reference.

The result is diagnostic for one run. A capacity or migration benefit claim
requires matched arrival streams and repeated STAY/MIGRATE experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bridge_tp.audit_goodoutput import audit_payload  # noqa: E402


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _prompt_length(request: Any) -> int | None:
    if not isinstance(request, dict):
        return None
    tokens = request.get("prompt_token_ids", request.get("prompt"))
    if isinstance(tokens, list) and tokens and all(
        isinstance(token, int) and not isinstance(token, bool) and token >= 0
        for token in tokens
    ):
        return len(tokens)
    return None


def _reference(config: dict[str, Any]) -> list[tuple[int, float]]:
    if config.get("slo_version") != "v6":
        raise ValueError("reference is not SLO v6")
    anchors = config.get("anchors")
    if not isinstance(anchors, list) or len(anchors) < 2:
        raise ValueError("reference needs at least two anchors")
    result: list[tuple[int, float]] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            raise ValueError("reference anchor is not an object")
        length = anchor.get("prompt_tokens")
        p95 = _finite(anchor.get("reference_p95_ms"))
        if (not isinstance(length, int) or isinstance(length, bool)
                or length <= 0 or p95 is None or p95 <= 0):
            raise ValueError("reference anchor length or P95 is invalid")
        if result and (length <= result[-1][0] or p95 < result[-1][1]):
            raise ValueError("reference anchors are not strictly ordered")
        result.append((length, p95))
    for key in (
        "queue_allowance_ms", "primary_tpot_mean_ms",
        "primary_slow_interval_threshold_ms", "max_visible_interval_ms",
        "max_handoff_ms",
    ):
        number = _finite(config.get(key))
        if number is None or number <= 0:
            raise ValueError(f"reference {key} is invalid")
    rate = _finite(config.get("primary_max_slow_interval_rate"))
    if rate is None or not 0 <= rate <= 1:
        raise ValueError("reference slow interval rate is invalid")
    return result


def ttft_limit_ms(length: int, config: dict[str, Any]) -> float:
    """Interpolate the frozen P95 curve and add the fixed queue allowance."""
    anchors = _reference(config)
    if length <= anchors[0][0]:
        base = anchors[0][1]
    elif length > anchors[-1][0]:
        raise ValueError("prompt length exceeds the frozen SLO range")
    else:
        base = anchors[-1][1]
        for (left_n, left_ms), (right_n, right_ms) in zip(
            anchors, anchors[1:]
        ):
            if length <= right_n:
                fraction = (length - left_n) / (right_n - left_n)
                base = left_ms + fraction * (right_ms - left_ms)
                break
    return base + float(config["queue_allowance_ms"])


def audit_v6_payload(
    values: dict[str, Any], config: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    *, gpu_match_mode: str = "uuid",
) -> dict[str, Any]:
    """Audit complete client streams; missing lengths or times fail closed."""
    _reference(config)
    if gpu_match_mode not in {"uuid", "model"}:
        raise ValueError("GPU match mode must be uuid or model")
    if provenance is None:
        applicability = "UNVERIFIED_GPU_ROSTER"
    else:
        model_matches = (
            isinstance(provenance, dict)
            and provenance.get("model_config_sha256")
            == config.get("model_config_sha256")
        )
        if gpu_match_mode == "model":
            gpu_matches = (
                provenance.get("gpu_models")
                == ["NVIDIA A100-PCIE-40GB"] * 5
            ) if isinstance(provenance, dict) else False
            verified = "VERIFIED_GPU_MODEL_AND_MODEL_CONFIG"
        else:
            gpu_matches = (
                isinstance(provenance, dict)
                and isinstance(config.get("gpu_uuids"), list)
                and provenance.get("gpu_uuids") == config["gpu_uuids"]
            )
            verified = "VERIFIED_GPU_AND_MODEL_CONFIG"
        applicability = (verified if model_matches and gpu_matches
                         else "REFERENCE_INPUT_MISMATCH")
    legacy = audit_payload(values)
    errors = list(legacy["errors"])
    if not legacy["computable"]:
        return {"format_version": 1, "slo_version": "v6", "computable": False,
                "reference_applicability": applicability,
                "errors": errors, "request_rows": [], "metrics": None}
    manifest = values.get("background/background_manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("jobs"), list):
        errors.append("background manifest jobs are missing")
        manifest_jobs: list[Any] = []
    else:
        manifest_jobs = manifest["jobs"]
    jobs: dict[str, dict[str, Any]] = {}
    for job in manifest_jobs:
        if not isinstance(job, dict) or not isinstance(job.get("job_id"), str):
            errors.append("background manifest job id is invalid")
            continue
        job_id = job["job_id"]
        if job_id in jobs:
            errors.append(f"duplicate manifest job id: {job_id}")
        jobs[job_id] = job
    background = values["background/background_summary.json"]
    results = {result["job_id"]: result for result in background["results"]}
    if set(jobs) != set(results):
        errors.append("background manifest and result job ids differ")
    for job_id in set(jobs) & set(results):
        if jobs[job_id].get("pool") != results[job_id].get("pool"):
            errors.append(f"{job_id}: manifest and result pools differ")
    contract = values["online/contract.json"]
    proxy = values["controller/response_proxy_stats.json"]
    lengths: dict[str, int] = {}
    for job_id, job in jobs.items():
        length = _prompt_length(job.get("request"))
        if length is None:
            errors.append(f"{job_id}: original prompt token ids are missing")
        else:
            lengths[job_id] = length
    anchor_length = contract.get("anchor_prompt_tokens")
    anchor_id = str(proxy.get("external_request_id", "anchor"))
    if anchor_id in jobs:
        errors.append("anchor request id collides with background job id")
    if (not isinstance(anchor_length, int) or isinstance(anchor_length, bool)
            or anchor_length <= 0):
        errors.append("anchor original prompt token count is missing")
    else:
        lengths[anchor_id] = anchor_length
    if errors:
        return {"format_version": 1, "slo_version": "v6", "computable": False,
                "reference_applicability": applicability,
                "errors": errors, "request_rows": [], "metrics": None}

    rows: list[dict[str, Any]] = []
    for row in legacy["request_rows"]:
        request_id = row["request_id"]
        pool = row["pool"]
        length = lengths.get(request_id)
        if length is None:
            errors.append(f"{pool}/{request_id}: prompt length is missing")
            continue
        try:
            limit = ttft_limit_ms(length, config)
        except ValueError as exc:
            errors.append(f"{pool}/{request_id}: {exc}")
            continue
        if pool == "anchor":
            emitted = proxy["emitted"]
            times = [item["unix_s"] for item in emitted]
        else:
            times = results[request_id].get("token_times_unix_s")
        status = row["status"]
        if status == "COMPLETED":
            if not isinstance(times, list) or len(times) != row["output_tokens"]:
                errors.append(f"{pool}/{request_id}: token timestamps are missing")
                continue
            gaps = [(right - left) * 1000 for left, right in zip(times, times[1:])]
            mean = sum(gaps) / len(gaps) if gaps else None
            slow = sum(
                gap > config["primary_slow_interval_threshold_ms"] for gap in gaps
            )
            slow_rate = slow / len(gaps) if gaps else 0.0
            longest = max(gaps) if gaps else None
            ttft = row["ttft_ms"]
            handoff = row.get("handoff_ms")
            failures = []
            if ttft > limit:
                failures.append("TTFT")
            if mean is not None and mean > config["primary_tpot_mean_ms"]:
                failures.append("MEAN_TPOT")
            if slow_rate > config["primary_max_slow_interval_rate"]:
                failures.append("SLOW_INTERVAL_RATE")
            if longest is not None and longest > config["max_visible_interval_ms"]:
                failures.append("MAX_VISIBLE_INTERVAL")
            if pool == "anchor" and handoff > config["max_handoff_ms"]:
                failures.append("HANDOFF")
            success = not failures
        else:
            ttft, mean, longest, slow, slow_rate = None, None, None, None, None
            handoff, success = None, False
            failures = ["INCOMPLETE"]
        rows.append({
            "request_id": request_id, "pool": pool, "status": status,
            "prompt_tokens": length, "output_tokens": row["output_tokens"],
            "ttft_ms": ttft, "ttft_limit_ms": limit,
            "mean_tpot_ms": mean, "slow_intervals": slow,
            "slow_interval_rate": slow_rate, "max_visible_interval_ms": longest,
            "handoff_ms": handoff, "slo_success": success,
            "failure_reasons": failures,
        })
    if errors:
        return {"format_version": 1, "slo_version": "v6", "computable": False,
                "reference_applicability": applicability,
                "errors": errors, "request_rows": rows, "metrics": None}
    wall_s = legacy["metrics"]["wall_time_s"]
    successes = sum(row["slo_success"] for row in rows)
    good_tokens = sum(
        row["output_tokens"] for row in rows if row["slo_success"]
    )
    metrics = {
        "requests": len(rows), "completed_requests": sum(
            row["status"] == "COMPLETED" for row in rows
        ),
        "slo_success_requests": successes,
        "slo_attainment": successes / len(rows) if rows else None,
        "raw_output_tokens": sum(row["output_tokens"] for row in rows),
        "good_output_tokens": good_tokens,
        "wall_time_s": wall_s,
        "goodoutput_tokens_s": good_tokens / wall_s,
    }
    return {"format_version": 1, "slo_version": "v6", "computable": True,
            **({"slo_policy_id": config["slo_policy_id"]}
               if "slo_policy_id" in config else {}),
            "reference_applicability": applicability,
            "errors": [], "request_rows": rows, "metrics": metrics,
            "workload_kind": (
                "synthetic_ignore_eos" if any(
                    job["request"].get("ignore_eos") is True
                    for job in manifest_jobs
                ) else "natural_eos_or_unspecified"
            )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--preflight-json", type=Path)
    parser.add_argument("--require-reference-match", action="store_true")
    parser.add_argument("--gpu-match-mode", choices=("uuid", "model"),
                        default="uuid")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    if args.out_json.exists():
        raise FileExistsError(args.out_json)
    reference_bytes = args.reference.read_bytes()
    config = json.loads(reference_bytes)
    run = args.run_root
    paths = {
        "online/contract.json": run.parent / "contract.json",
        "background/background_manifest.json": (
            run / "background/background_manifest.json"
        ),
        "background/background_summary.json": (
            run / "background/background_summary.json"
        ),
        "controller/response_proxy_stats.json": (
            run / "controller/response_proxy_stats.json"
        ),
        "controller/source_response.json": run / "controller/source_response.json",
    }
    target = run / "controller/target_response.json"
    if target.is_file():
        paths["controller/target_response.json"] = target
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"run files are missing: {missing}")
    values = {name: json.loads(path.read_text(encoding="utf-8"))
              for name, path in paths.items()}
    provenance = (
        json.loads(args.preflight_json.read_text(encoding="utf-8"))
        if args.preflight_json else None
    )
    report = audit_v6_payload(
        values, config, provenance, gpu_match_mode=args.gpu_match_mode)
    report["gpu_match_mode"] = args.gpu_match_mode
    report["reference_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
    report["run_root"] = str(run.resolve())
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"computable": report["computable"],
                      "reference_applicability": report["reference_applicability"],
                      "errors": report["errors"], "metrics": report["metrics"]}))
    if not report["computable"]:
        raise RuntimeError("v6 SLO audit is not computable; inspect out-json")
    required = ("VERIFIED_GPU_MODEL_AND_MODEL_CONFIG"
                if args.gpu_match_mode == "model"
                else "VERIFIED_GPU_AND_MODEL_CONFIG")
    if (args.require_reference_match
            and report["reference_applicability"] != required):
        raise RuntimeError("v6 reference does not match this run; inspect out-json")


if __name__ == "__main__":
    main()
