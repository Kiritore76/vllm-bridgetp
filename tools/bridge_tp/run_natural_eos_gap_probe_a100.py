#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe client-visible token gaps under three loads with natural EOS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402
from vllm.bridge_tp.controller.online_io import (  # noqa: E402
    post_streaming_completion,
)
from vllm.bridge_tp.controller.sampling_contract import (  # noqa: E402
    freeze_strict_greedy_sampling,
)

THRESHOLDS_MS = (50, 100, 200, 500, 1000)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def select_requests(path: Path, count: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    selected: list[dict[str, Any]] = []
    for group, quota in (("natural", count * 4 // 5),
                         ("long_form", count - count * 4 // 5)):
        pool = [row for row in rows if row.get("split") == "test"
                and row.get("workload_group") == group]
        pool.sort(key=lambda row: hashlib.sha256(
            ("natural-eos-gap-v1:" + str(row["id"])).encode()
        ).hexdigest())
        if len(pool) < quota:
            raise ValueError(f"not enough test/{group} inputs: {len(pool)}")
        selected.extend(pool[:quota])
    selected.sort(key=lambda row: hashlib.sha256(
        ("natural-eos-arrival-v1:" + str(row["id"])).encode()
    ).hexdigest())
    if len({str(row["id"]) for row in selected}) != count:
        raise ValueError("selected request IDs are not unique")
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    natural = [row for row in rows if row.get("finish_reason") == "stop"]
    gaps = [gap for row in natural for gap in row["gaps_ms"]]
    maximums = [max(row["gaps_ms"]) for row in natural if row["gaps_ms"]]
    events = sorted((
        {"request_id": row["input_id"], "group": row["workload_group"],
         "output_tokens": row["output_tokens"], "gap_index": index + 1,
         "gap_ms": gap, "preceding_token_unix_s": row["token_times_unix_s"][index]}
        for row in natural for index, gap in enumerate(row["gaps_ms"])
    ), key=lambda item: item["gap_ms"], reverse=True)
    groups = {}
    for group in ("natural", "long_form"):
        group_rows = [row for row in rows if row["workload_group"] == group]
        group_gaps = [gap for row in group_rows
                      if row.get("finish_reason") == "stop"
                      for gap in row["gaps_ms"]]
        groups[group] = {
            "arrivals": len(group_rows),
            "natural_eos": sum(row.get("finish_reason") == "stop"
                               for row in group_rows),
            "gaps": len(group_gaps),
            "p99_gap_ms": percentile(group_gaps, .99),
            "max_gap_ms": max(group_gaps) if group_gaps else None,
            "over_100_ms": sum(gap > 100 for gap in group_gaps),
            "over_1000_ms": sum(gap > 1000 for gap in group_gaps),
        }
    return {
        "arrivals": len(rows), "natural_eos": len(natural),
        "length_censored": sum(row.get("finish_reason") == "length" for row in rows),
        "failed": sum(row.get("status") != "COMPLETED" for row in rows),
        "natural_eos_requests_with_gaps": len(maximums),
        "natural_eos_gaps": len(gaps),
        "output_tokens_natural_eos": sum(row["output_tokens"] for row in natural),
        "gap_ms": {"p50": percentile(gaps, .5), "p90": percentile(gaps, .9),
                   "p95": percentile(gaps, .95), "p99": percentile(gaps, .99),
                   "p999": percentile(gaps, .999), "max": max(gaps) if gaps else None},
        "request_max_gap_ms": {
            "p50": percentile(maximums, .5), "p90": percentile(maximums, .9),
            "p95": percentile(maximums, .95), "p99": percentile(maximums, .99),
            "max": max(maximums) if maximums else None,
        },
        "thresholds": {
            str(threshold): {
                "gaps_over": sum(gap > threshold for gap in gaps),
                "gap_fraction_over": (sum(gap > threshold for gap in gaps) / len(gaps)
                                      if gaps else None),
                "requests_with_any_over": sum(value > threshold for value in maximums),
            } for threshold in THRESHOLDS_MS
        },
        "largest_20_gaps": events[:20],
        "by_workload_group": groups,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if os.name == "nt":
        raise RuntimeError("run this probe on the A100 Linux server")
    revision = common.git("rev-parse", "HEAD")
    branch = common.git("branch", "--show-current")
    if (revision != args.expected_revision or branch != args.expected_branch
            or common.git("status", "--porcelain")):
        raise RuntimeError("HEAD, branch or worktree differs; no GPU run")
    hostname = socket.gethostname()
    if hostname != args.expected_hostname:
        raise RuntimeError(f"hostname differs: {hostname}")
    gpu_csv = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    if len(gpu_csv) != 5 or any(
        not line.strip().startswith("NVIDIA A100-PCIE-40GB, GPU-")
        for line in gpu_csv
    ):
        raise RuntimeError(f"expected five A100-PCIE-40GB GPUs: {gpu_csv}")
    uuids = [line.split(",", 1)[1].strip() for line in gpu_csv]
    if len(set(uuids)) != 5:
        raise RuntimeError("duplicate GPU UUIDs")
    if uuids != args.expected_gpu_uuids.split(","):
        raise RuntimeError(f"GPU UUID roster differs: {uuids}")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    if any(line.split(",", 1)[0].strip() in set(uuids[1:5]) for line in active):
        raise RuntimeError("TP4 GPUs already have compute processes")
    expected = ((args.input, args.expected_input_sha256),
                (args.model_path / "config.json", args.expected_model_sha256))
    for path, sha in expected:
        if not path.is_file() or digest(path) != sha:
            raise RuntimeError(f"missing input or SHA-256 mismatch: {path}")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    return {
        "revision": revision, "branch": branch, "hostname": hostname,
        "gpu_inventory": gpu_csv, "gpu_uuids": uuids,
        "input_path": str(args.input.resolve()),
        "input_sha256": args.expected_input_sha256,
        "model_path": str(args.model_path.resolve()),
        "model_config_sha256": args.expected_model_sha256,
        "rates_rps": args.rates, "requests_per_rate": args.requests_per_rate,
        "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
        "temperature": 0, "ignore_eos": False, "port": args.port,
    }


def one_request(
    row: dict[str, Any], url: str, timeout_s: float,
) -> dict[str, Any]:
    times: list[float] = []
    try:
        result = post_streaming_completion(
            url, row["payload"], timeout_s,
            lambda _index, _token, unix_s: times.append(unix_s),
        )
        finish = result["finish_reason"]
        status = "COMPLETED"
        error = None
    except Exception as exc:  # Keep failed arrivals in the denominator.
        result = {}
        finish = None
        status = "FAILED"
        error = f"{type(exc).__name__}: {exc}"
    gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    return {
        "input_id": row["id"], "workload_group": row["workload_group"],
        "prompt_tokens": row["prompt_tokens"], "status": status,
        "finish_reason": finish, "error": error,
        "output_tokens": len(times), "token_times_unix_s": times,
        "gaps_ms": gaps, "ttft_ms": result.get("ttft_ms"),
        "request_started_unix_s": result.get("request_started_unix_s"),
        "completed_unix_s": result.get("completed_unix_s"),
    }


def run(args: argparse.Namespace) -> None:
    if (args.requests_per_rate < 5 or args.requests_per_rate % 5
            or args.max_tokens < 2 or args.max_model_len <= args.max_tokens
            or args.timeout_s <= 0 or any(rate <= 0 for rate in args.rates)
            or len(set(args.rates)) != len(args.rates)):
        raise ValueError("invalid workload, rate, output or timeout setting")
    provenance = preflight(args)
    selected = select_requests(args.input, args.requests_per_rate)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prepared = []
    for row in selected:
        prompt = (row["prompt"] if "prompt" in row else
                  tokenizer.apply_chat_template(
                      row["messages"], tokenize=False, add_generation_prompt=True
                  ))
        tokens = tokenizer.encode(prompt)
        if len(tokens) + args.max_tokens > args.max_model_len:
            raise ValueError(f"request {row['id']} exceeds context: {len(tokens)}")
        payload = freeze_strict_greedy_sampling({
            "model": "bridgetp-model", "prompt": tokens,
            "request_id": f"gap-{row['id']}", "max_tokens": args.max_tokens,
            "ignore_eos": False, "stream": True, "return_token_ids": True,
        })
        prepared.append({"id": row["id"], "workload_group": row["workload_group"],
                         "prompt_tokens": len(tokens), "payload": payload})
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    common.write_json(args.out_dir / "preflight.json", provenance)
    common.write_json(args.out_dir / "selected_requests.json", [
        {key: value for key, value in row.items() if key != "payload"}
        for row in prepared
    ])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
    env["OMP_NUM_THREADS"] = "1"
    server_args = SimpleNamespace(
        python_bin=Path(sys.executable), model_path=args.model_path,
        dtype="bfloat16", max_model_len=args.max_model_len,
        gpu_memory_utilization=.88,
    )
    server = common.start_process(
        "tp4", common.server_command(server_args, 4, args.port), env,
        args.out_dir / "tp4.log",
    )
    url = f"http://127.0.0.1:{args.port}"
    try:
        common.wait_healthy(url, server, 900)
        for rate in args.rates:
            label = str(rate).replace(".", "p")
            print(f"starting {rate} RPS, {len(prepared)} requests", flush=True)
            rows: list[dict[str, Any]] = []
            start = time.monotonic() + 2
            start_unix = time.time() + 2
            with ThreadPoolExecutor(max_workers=len(prepared)) as executor:
                futures = []
                for index, row in enumerate(prepared):
                    sleep_s = start + index / rate - time.monotonic()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    futures.append(executor.submit(
                        one_request, row, url, args.timeout_s,
                    ))
                future_index = {future: index for index, future in enumerate(futures)}
                for future in as_completed(futures):
                    result = future.result()
                    index = future_index[future]
                    result["scheduled_unix_s"] = start_unix + index / rate
                    rows.append(result)
                    print(f"{rate} RPS {len(rows)}/{len(prepared)} "
                          f"{result['finish_reason'] or result['status']}", flush=True)
            order = {row["id"]: index for index, row in enumerate(prepared)}
            rows.sort(key=lambda row: order[row["input_id"]])
            with (args.out_dir / f"requests_{label}rps.jsonl").open(
                "w", encoding="utf-8"
            ) as output:
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
            report = summarize(rows)
            report.update({"rate_rps": rate, "status": "DIAGNOSTIC_NO_SLO"})
            common.write_json(args.out_dir / f"summary_{label}rps.json", report)
            print(json.dumps({k: report[k] for k in (
                "rate_rps", "arrivals", "natural_eos", "length_censored",
                "failed", "natural_eos_gaps", "gap_ms", "thresholds"
            )}), flush=True)
    finally:
        common.stop_processes([server])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--expected-gpu-uuids", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--rates", nargs="+", type=float, default=[.3, .7, 1.15])
    parser.add_argument("--requests-per-rate", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--port", type=int, default=8200)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
