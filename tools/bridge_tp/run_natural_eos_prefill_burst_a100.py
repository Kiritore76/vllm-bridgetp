#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pair natural-EOS decode streams with controlled long-prefill bursts."""

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
from tools.bridge_tp.run_natural_eos_gap_probe_a100 import (  # noqa: E402
    digest, preflight, select_requests, summarize,
)
from vllm.bridge_tp.controller.online_io import (  # noqa: E402
    post_streaming_completion,
)
from vllm.bridge_tp.controller.sampling_contract import (  # noqa: E402
    freeze_strict_greedy_sampling,
)

BURST_INDICES = (15, 30, 45)
BURST_LENGTHS = (2048, 4096)


def one_request(
    row: dict[str, Any], url: str, timeout_s: float,
) -> dict[str, Any]:
    unix_times: list[float] = []
    mono_times: list[float] = []

    def sink(_index: int, _token: int, unix_s: float) -> None:
        unix_times.append(unix_s)
        mono_times.append(time.monotonic())

    try:
        result = post_streaming_completion(url, row["payload"], timeout_s, sink)
        token_ids = result["token_ids"]
        if len(token_ids) != len(unix_times):
            raise RuntimeError("token IDs and callback timestamps differ")
        status = "COMPLETED"
        error = None
    except Exception as exc:
        result = {}
        token_ids = []
        status = "FAILED"
        error = f"{type(exc).__name__}: {exc}"
    return {
        "input_id": row["id"], "role": row["role"],
        "workload_group": row["workload_group"],
        "prompt_tokens": row["prompt_tokens"],
        "status": status, "error": error,
        "finish_reason": result.get("finish_reason"),
        "output_tokens": len(unix_times),
        "token_ids_sha256": (
            hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()
            if status == "COMPLETED" else None
        ),
        "token_times_unix_s": unix_times,
        "token_times_monotonic_s": mono_times,
        "gaps_ms": [(b - a) * 1000 for a, b in zip(mono_times, mono_times[1:])],
        "ttft_ms": result.get("ttft_ms"),
        "request_started_unix_s": result.get("request_started_unix_s"),
        "completed_unix_s": result.get("completed_unix_s"),
    }


def prepare_victims(args: argparse.Namespace) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    selected = select_requests(args.input, args.requests_per_rate)
    prepared = []
    for row in selected:
        prompt = (row["prompt"] if "prompt" in row else
                  tokenizer.apply_chat_template(
                      row["messages"], tokenize=False, add_generation_prompt=True
                  ))
        tokens = tokenizer.encode(prompt)
        if len(tokens) + args.max_tokens > args.max_model_len:
            raise ValueError(f"victim {row['id']} exceeds model context")
        prepared.append({
            "id": row["id"], "role": "victim",
            "workload_group": row["workload_group"],
            "prompt_tokens": len(tokens),
            "payload": freeze_strict_greedy_sampling({
                "model": "bridgetp-model", "prompt": tokens,
                "request_id": f"gap-victim-{row['id']}",
                "max_tokens": args.max_tokens, "ignore_eos": False,
                "stream": True, "return_token_ids": True,
            }),
        })
    return prepared


def prepare_bursts(args: argparse.Namespace) -> list[dict[str, Any]]:
    if (not args.base_manifest.is_file()
            or digest(args.base_manifest) != args.expected_base_sha256):
        raise RuntimeError("frozen base manifest is missing or SHA-256 differs")
    manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs", [])
    if len(jobs) != 4 or any(job.get("pool") != "target" for job in jobs):
        raise ValueError("frozen base manifest must have four target jobs")
    tokens = jobs[0]["request"].get("prompt")
    if not isinstance(tokens, list) or len(tokens) != 2048:
        raise ValueError("frozen base prompt is not 2048 token IDs")
    if any(not isinstance(token, int) or isinstance(token, bool) for token in tokens):
        raise ValueError("base prompt has non-integer token IDs")
    result = []
    for burst_index in BURST_INDICES:
        for length in BURST_LENGTHS:
            prompt = (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]
            if length + args.burst_max_tokens > args.max_model_len:
                raise ValueError("prefill request exceeds model context")
            result.append({
                "id": f"controlled-prefill-{burst_index}-{length}",
                "role": "controlled_prefill", "workload_group": "controlled_prefill",
                "prompt_tokens": length,
                "payload": freeze_strict_greedy_sampling({
                    "model": "bridgetp-model", "prompt": prompt,
                    "request_id": f"gap-prefill-{burst_index}-{length}",
                    "max_tokens": args.burst_max_tokens, "ignore_eos": False,
                    "stream": True, "return_token_ids": True,
                }),
            })
    return result


def schedule(
    victims: list[dict[str, Any]], bursts: list[dict[str, Any]],
    rate: float, with_bursts: bool,
) -> list[tuple[float, dict[str, Any]]]:
    events = [(index / rate, row) for index, row in enumerate(victims)]
    if with_bursts:
        for row in bursts:
            index = int(row["id"].split("-")[2])
            # Two long prefills arrive within 50 ms after the selected victim.
            offset = .05 if row["prompt_tokens"] == 2048 else .10
            events.append((index / rate + offset, row))
    events.sort(key=lambda item: (item[0], item[1]["id"]))
    return events


def run_condition(
    *, args: argparse.Namespace, url: str, rate: float, arm: str,
    victims: list[dict[str, Any]], bursts: list[dict[str, Any]],
) -> dict[str, Any]:
    label = f"{str(rate).replace('.', 'p')}rps_{arm}"
    events = schedule(victims, bursts, rate, arm == "burst")
    start_mono = time.monotonic() + 2
    start_unix = time.time() + 2
    futures = {}
    with ThreadPoolExecutor(max_workers=len(events)) as executor:
        for index, (offset, row) in enumerate(events):
            delay = start_mono + offset - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            future = executor.submit(one_request, row, url, args.timeout_s)
            futures[future] = (index, offset)
        completed = []
        for future in as_completed(futures):
            row = future.result()
            index, offset = futures[future]
            row["schedule_index"] = index
            row["scheduled_unix_s"] = start_unix + offset
            completed.append(row)
            print(f"{label} {len(completed)}/{len(events)} "
                  f"{row['role']} {row['finish_reason'] or row['status']}", flush=True)
    completed.sort(key=lambda row: row["schedule_index"])
    with (args.out_dir / f"requests_{label}.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for row in completed:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    victim_rows = [row for row in completed if row["role"] == "victim"]
    burst_rows = [row for row in completed if row["role"] == "controlled_prefill"]
    report = summarize(victim_rows)
    report.update({
        "status": "DIAGNOSTIC_NO_SLO", "rate_rps": rate, "arm": arm,
        "victim_arrivals": len(victim_rows),
        "controlled_prefill_arrivals": len(burst_rows),
        "controlled_prefill_complete": sum(
            row["status"] == "COMPLETED" for row in burst_rows
        ),
        "controlled_prefill_natural_eos": sum(
            row["finish_reason"] == "stop" for row in burst_rows
        ),
        "controlled_prefill_length_censored": sum(
            row["finish_reason"] == "length" for row in burst_rows
        ),
    })
    common.write_json(args.out_dir / f"summary_{label}.json", report)
    print(json.dumps({key: report[key] for key in (
        "rate_rps", "arm", "victim_arrivals", "natural_eos", "failed",
        "length_censored", "controlled_prefill_complete", "gap_ms",
        "thresholds",
    )}), flush=True)
    return report


def run(args: argparse.Namespace) -> None:
    if (args.requests_per_rate < 50 or args.requests_per_rate % 5
            or args.max_tokens < 2 or args.burst_max_tokens < 1
            or any(rate <= 0 for rate in args.rates)
            or len(set(args.rates)) != 2):
        raise ValueError(
            "expected two positive rates and >=50 requests divisible by five"
        )
    provenance = preflight(args)
    victims = prepare_victims(args)
    bursts = prepare_bursts(args)
    provenance.update({
        "base_manifest_path": str(args.base_manifest.resolve()),
        "base_manifest_sha256": args.expected_base_sha256,
        "prefill_lengths": list(BURST_LENGTHS),
        "burst_after_victim_indices": list(BURST_INDICES),
        "burst_max_tokens": args.burst_max_tokens,
        "experiment_order": "low baseline, high burst, high baseline, low burst",
    })
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    common.write_json(args.out_dir / "preflight.json", provenance)
    common.write_json(args.out_dir / "selected_requests.json", [
        {key: value for key, value in row.items() if key != "payload"}
        for row in victims + bursts
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
        low, high = args.rates
        for rate, arm in ((low, "baseline"), (high, "burst"),
                          (high, "baseline"), (low, "burst")):
            print(f"starting {rate} RPS {arm}", flush=True)
            run_condition(args=args, url=url, rate=rate, arm=arm,
                          victims=victims, bursts=bursts)
    finally:
        common.stop_processes([server])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--expected-gpu-uuids", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--rates", nargs=2, type=float, default=[.7, 1.15])
    parser.add_argument("--requests-per-rate", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--burst-max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--port", type=int, default=8200)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
