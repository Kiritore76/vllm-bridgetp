# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path

from vllm.bridge_tp.kv_restore import RESTORE_PARAM, load_restore_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the exact OpenAI completion request for Phase 5."
    )
    parser.add_argument("reshard_dir", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    artifact = load_restore_artifact(args.reshard_dir)
    request = {
        "model": args.model,
        "prompt": artifact.all_known_token_ids,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": False,
        "return_token_ids": True,
        "kv_transfer_params": {
            RESTORE_PARAM: artifact.source_request_id,
        },
    }
    rendered = json.dumps(request, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
