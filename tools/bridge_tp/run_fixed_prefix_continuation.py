#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run strict-greedy continuation from exact token IDs, optionally at batch N."""

from __future__ import annotations

import argparse
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixed-prefix-token-ids", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_tokens(path: Path) -> list[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("fixed_token_ids")
    if not isinstance(raw, list) or not raw:
        raise SystemExit("fixed-prefix file must contain fixed_token_ids")
    return [int(value) for value in raw]


def request_payload(
    *, model: str, prompt: list[int], request_id: str, max_tokens: int
) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "request_id": request_id,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
        "use_beam_search": False,
        "n": 1,
        "stream": False,
        "add_special_tokens": False,
        "return_token_ids": True,
    }


def post_json(url: str, payload: dict, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.load(response)


def response_tokens(response: dict) -> list[int]:
    values = response.get("choices", [{}])[0].get("token_ids")
    if not isinstance(values, list):
        raise SystemExit("server response did not include choices[0].token_ids")
    return [int(value) for value in values]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_tokens <= 0:
        raise SystemExit("batch size and max tokens must be positive")
    prefix = load_tokens(args.fixed_prefix_token_ids)
    barrier = threading.Barrier(args.batch_size)

    def run(index: int) -> tuple[int, dict, dict]:
        suffix = "primary" if index == 0 else f"filler-{index}"
        payload = request_payload(
            model=args.model,
            prompt=prefix,
            request_id=f"{args.request_id}-{suffix}",
            max_tokens=args.max_tokens,
        )
        barrier.wait()
        return index, payload, post_json(args.base_url, payload, args.timeout_s)

    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        results = sorted(executor.map(run, range(args.batch_size)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "base_url": args.base_url,
        "model": args.model,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "fixed_prefix_file": str(args.fixed_prefix_token_ids),
        "requests": [],
    }
    for index, payload, response in results:
        name = "primary" if index == 0 else f"filler_{index}"
        request_path = args.out_dir / f"{name}_request.json"
        response_path = args.out_dir / f"{name}_response.json"
        tokens_path = args.out_dir / f"{name}_tokens.json"
        request_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        response_path.write_text(
            json.dumps(response, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tokens = response_tokens(response)
        tokens_path.write_text(json.dumps(tokens, indent=2) + "\n", encoding="utf-8")
        manifest["requests"].append(
            {
                "role": name,
                "request_id": payload["request_id"],
                "tokens": len(tokens),
                "token_file": str(tokens_path),
            }
        )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"completed batch={args.batch_size}; primary tokens="
        f"{manifest['requests'][0]['tokens']}"
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
