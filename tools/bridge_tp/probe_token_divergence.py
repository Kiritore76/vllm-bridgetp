#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a fail-closed TP1/TP4 tie certificate for a Phase 8/9 run."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.bridge_tp.controller.online_io import (  # noqa: E402
    atomic_json_dump,
    load_json,
)
from vllm.bridge_tp.controller.sampling_contract import (  # noqa: E402
    freeze_strict_greedy_sampling,
    strict_greedy_sampling_errors,
)
from vllm.bridge_tp.controller.token_equivalence import (  # noqa: E402
    classify_token_equivalence,
    first_divergence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-url", default="http://127.0.0.1:8001")
    parser.add_argument("--target-url", default="http://127.0.0.1:8200")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="local model/tokenizer path; required only when tokens diverge",
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--tie-epsilon", type=float, default=1e-6)
    return parser.parse_args()


def _post(base_url: str, payload: dict[str, Any], timeout_s: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(
            f"probe request failed at {base_url}: HTTP {error.code}: {body}"
        ) from error
    if not isinstance(value, dict):
        raise TypeError(f"probe response from {base_url} is not an object")
    return value


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    source_request = load_json(run / "source_request.json")
    control_response = load_json(run / "control_response.json")
    target_response = load_json(run / "target_response.json")
    staging = load_json(run / "staging_manifest.json")
    proxy_path = run / "response_proxy_stats.json"
    phase8_path = run / "phase8_result.json"
    is_phase9 = proxy_path.exists()
    if is_phase9:
        source_errors = strict_greedy_sampling_errors(source_request)
        if source_errors:
            raise ValueError(
                "source request does not carry the Phase 9 sampling "
                "contract: " + "; ".join(source_errors)
            )
        stats = load_json(proxy_path)
        emitted = [int(value) for value in stats.get("token_ids") or []]
        emitted_rows = list(stats.get("emitted") or [])
        cutover = int(stats["cutover_index"])
    elif phase8_path.exists():
        result = load_json(phase8_path)
        emitted = [int(value) for value in result.get("assembled_token_ids") or []]
        cutover = int(staging["snapshot_num_output_tokens"])
        emitted_rows = [
            {
                "index": index,
                "token_id": token_id,
                "origin": "source" if index < cutover else "target",
            }
            for index, token_id in enumerate(emitted)
        ]
    else:
        raise FileNotFoundError(
            "run has neither response_proxy_stats.json nor phase8_result.json"
        )
    choices = control_response.get("choices") or []
    if len(choices) != 1:
        raise ValueError("control response must contain exactly one choice")
    control = [int(value) for value in choices[0].get("token_ids") or []]
    divergence = first_divergence(emitted, control)

    probe_request: dict[str, Any] | None = None
    tp1_probe: dict[str, Any] | None = None
    tp4_probe: dict[str, Any] | None = None
    expected_prompt: list[int] | None = None
    token_text_map: dict[str, Any] | None = None
    if divergence is not None and divergence < min(len(emitted), len(control)):
        if args.tokenizer is None:
            raise ValueError("--tokenizer is required for a divergent run")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            local_files_only=True,
        )
        control_token_id = control[divergence]
        target_token_id = emitted[divergence]
        token_text_map = {
            "format_version": 1,
            "tokenizer_path": str(args.tokenizer.resolve()),
            "tokenizer_class": type(tokenizer).__name__,
            "control_token_id": control_token_id,
            "target_token_id": target_token_id,
            "tokens": {
                str(control_token_id): tokenizer.decode(
                    [control_token_id],
                    clean_up_tokenization_spaces=False,
                ),
                str(target_token_id): tokenizer.decode(
                    [target_token_id],
                    clean_up_tokenization_spaces=False,
                ),
            },
        }
        num_prompt = int(staging["num_prompt_tokens"])
        known = [int(value) for value in staging["all_known_token_ids"]]
        expected_prompt = known[:num_prompt] + control[:divergence]
        probe_base = {
            "model": source_request["model"],
            "prompt": expected_prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": False,
            "return_token_ids": True,
            "logprobs": args.logprobs,
        }
        probe_request = (
            freeze_strict_greedy_sampling(probe_base)
            if is_phase9
            else probe_base
        )
        tp1_probe = _post(args.source_url, probe_request, args.timeout_s)
        tp4_probe = _post(args.target_url, probe_request, args.timeout_s)
        atomic_json_dump(probe_request, run / "topology_probe_request.json")
        atomic_json_dump(tp1_probe, run / "topology_probe_tp1.json")
        atomic_json_dump(tp4_probe, run / "topology_probe_tp4.json")
        atomic_json_dump(token_text_map, run / "token_text_map.json")

    classification = classify_token_equivalence(
        emitted=emitted,
        control=control,
        emitted_rows=emitted_rows,
        cutover_index=cutover,
        target_response=target_response,
        control_response=control_response,
        token_text_map=token_text_map,
        probe_request=probe_request,
        tp1_probe=tp1_probe,
        tp4_probe=tp4_probe,
        expected_probe_prompt=expected_prompt,
        tie_epsilon=args.tie_epsilon,
        require_strict_sampling_contract=is_phase9,
    )
    atomic_json_dump(classification, run / "token_equivalence.json")
    print(json.dumps(classification, ensure_ascii=False, indent=2))
    if not classification["acceptable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
