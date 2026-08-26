#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D-0: compare actual vLLM raw/processed logits at one fixed prefix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.bridge_tp.controller.numerics import analyze_candidate_gap  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-capture", type=Path, required=True)
    parser.add_argument("--migrated-capture", type=Path, required=True)
    parser.add_argument(
        "--candidate-token-ids",
        type=int,
        nargs=2,
        required=True,
        metavar=("CONTROL_TOKEN", "MIGRATED_TOKEN"),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_capture(path: Path) -> dict:
    candidate = path / "capture.json" if path.is_dir() else path
    if not candidate.is_file():
        raise SystemExit(f"capture not found: {candidate}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise SystemExit(f"unsupported capture format: {candidate}")
    if set(payload.get("stages", {})) != {"raw", "processed"}:
        raise SystemExit(f"raw and processed stages are required: {candidate}")
    return payload


def token_value(stage: dict, token_id: int, label: str) -> float:
    values = stage.get("candidate_values", {})
    key = str(token_id)
    if key in values:
        return float(values[key])
    top_ids = [int(value) for value in stage.get("top_token_ids", [])]
    if token_id in top_ids:
        return float(stage["top_values"][top_ids.index(token_id)])
    raise SystemExit(
        f"{label} does not include token {token_id}; re-run capture with "
        "BRIDGETP_LOGIT_CAPTURE_CANDIDATE_TOKEN_IDS containing both tokens"
    )


def analyze_side(label: str, capture: dict, token_ids: list[int]) -> dict:
    result: dict[str, object] = {}
    for stage_name in ("raw", "processed"):
        stage = capture["stages"][stage_name]
        first = token_value(stage, token_ids[0], f"{label}/{stage_name}")
        second = token_value(stage, token_ids[1], f"{label}/{stage_name}")
        result[stage_name] = analyze_candidate_gap(
            stage=stage_name,
            dtype=stage["dtype"],
            first_token_id=token_ids[0],
            second_token_id=token_ids[1],
            first_value=first,
            second_value=second,
        ).to_json()
    return result


def main() -> None:
    args = parse_args()
    control = load_capture(args.control_capture)
    migrated = load_capture(args.migrated_capture)
    for field in ("global_output_index", "prefix_token_ids_sha256"):
        if control.get(field) != migrated.get(field):
            raise SystemExit(
                f"capture provenance mismatch for {field}: "
                f"{control.get(field)!r} != {migrated.get(field)!r}"
            )

    token_ids = [int(value) for value in args.candidate_token_ids]
    if int(control.get("sampled_token_id", -1)) != token_ids[0]:
        raise SystemExit(
            "control capture did not sample the declared control candidate token"
        )
    if int(migrated.get("sampled_token_id", -1)) != token_ids[1]:
        raise SystemExit(
            "migrated capture did not sample the declared migrated candidate token"
        )
    payload = {
        "format_version": 1,
        "phase": "BridgeTP D3 Phase 9 D-0",
        "global_output_index": control["global_output_index"],
        "prefix_token_ids_sha256": control["prefix_token_ids_sha256"],
        "candidate_token_ids": token_ids,
        "control_sampled_token_id": control["sampled_token_id"],
        "migrated_sampled_token_id": migrated["sampled_token_id"],
        "control": analyze_side("control", control, token_ids),
        "migrated": analyze_side("migrated", migrated, token_ids),
        "formal_causal_conclusion": None,
        "evidence_boundary": (
            "This report uses actual vLLM tensor values and records ULP distance "
            "at the tensors' stored dtypes. It does not locate a CUDA kernel, "
            "prove that a gap was caused by one rounding operation, or clear the "
            "migration path without D-1 and paired D-3 evidence."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"global index: {payload['global_output_index']}")
    for side in ("control", "migrated"):
        raw = payload[side]["raw"]
        processed = payload[side]["processed"]
        print(
            f"{side}: raw={raw['absolute_gap']:.9g} "
            f"({raw['gap_ulps']:.3f} stored-dtype ULP), "
            f"processed={processed['absolute_gap']:.9g} "
            f"({processed['gap_ulps']:.3f} stored-dtype ULP)"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
