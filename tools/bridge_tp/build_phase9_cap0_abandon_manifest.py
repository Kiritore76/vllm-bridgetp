#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a working CAP-0 Safe-abandon workload manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp.build_phase9_cap0_noop_manifest import (  # noqa: E402
    build_manifest as build_noop_manifest,
)
from tools.bridge_tp.build_phase9_cap0_noop_manifest import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-copies", type=int, default=4)
    parser.add_argument("--target-prompt-tokens", type=int, default=7000)
    parser.add_argument("--target-output-tokens", type=int, default=1100)
    parser.add_argument("--target-start-after-s", type=float, default=0.0)
    parser.add_argument("--target-start-interval-s", type=float, default=0.01)
    parser.add_argument("--target-prompt-token-id", type=int, default=100)
    parser.add_argument("--source-copies", type=int, default=4)
    parser.add_argument("--source-output-tokens", type=int, default=4200)
    parser.add_argument("--source-start-after-s", type=float, default=2.0)
    parser.add_argument("--source-start-interval-s", type=float, default=0.2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--model", default="bridgetp-model")
    return parser.parse_args()


def build_manifest(**overrides: Any) -> dict[str, Any]:
    values = {
        "target_copies": 4,
        "target_prompt_tokens": 7000,
        "target_output_tokens": 1100,
        "target_start_after_s": 0.0,
        "target_start_interval_s": 0.01,
        "target_prompt_token_id": 100,
        "source_copies": 4,
        "source_output_tokens": 4200,
        "source_start_after_s": 2.0,
        "source_start_interval_s": 0.2,
        "max_model_len": 8192,
        "model": "bridgetp-model",
    }
    values.update(overrides)
    manifest = build_noop_manifest(**values)
    manifest["scenario"] = "CAP-0 Safe abandon bring-up"
    manifest["status"] = "WORKING_NOT_FROZEN"
    manifest["design_note"] = (
        "Finite TP1 pressure should activate the frozen guard, then finish "
        "during Shadow so headroom recovery cancels before cutover"
    )
    return manifest


def main() -> None:
    args = parse_args()
    values = vars(args).copy()
    out = values.pop("out")
    manifest = build_manifest(**values)
    write_json(out, manifest)
    parameters = manifest["parameters"]
    source_tokens = int(parameters["source_copies"]) * int(
        parameters["source_output_tokens"]
    )
    print(
        f"wrote CAP-0 Safe-abandon working manifest: {out.resolve()} "
        f"({len(manifest['jobs'])} jobs, "
        f"source background output={source_tokens} tokens)"
    )


if __name__ == "__main__":
    main()
