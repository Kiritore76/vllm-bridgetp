#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct a Phase 8 verdict from immutable archived raw evidence.

This tool exists for the Phase 8 commit run whose migration completed but
whose controller used the ordinary-completion token accessor on the normalized
streaming target response.  It never overwrites archived files.  Derived files
carry an explicit ``offline_reconstruction`` marker, and timing values that
were only held in controller-local monotonic variables remain unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _atomic_dump(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_original_manifest(run_dir: Path) -> int:
    manifest_path = run_dir / "SHA256SUMS"
    verified = 0
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(
                f"Malformed SHA256SUMS line {line_number}: {line!r}"
            ) from error
        relative = relative.removeprefix("*").removeprefix("./")
        path = run_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Archived evidence file is missing: {relative}")
        actual = _sha256(path)
        if actual != expected.lower():
            raise ValueError(
                f"Archived evidence hash differs for {relative}: "
                f"{actual} != {expected.lower()}"
            )
        verified += 1
    return verified


def _choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one completion choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise TypeError("Completion choice is not an object")
    return choice


def _ordinary_token_ids(response: dict[str, Any]) -> list[int]:
    values = _choice(response).get("token_ids")
    if not isinstance(values, list):
        raise ValueError("Ordinary completion response has no token_ids")
    return [int(value) for value in values]


def _streaming_token_ids(response: dict[str, Any]) -> list[int]:
    values = response.get("token_ids")
    if not isinstance(values, list):
        raise ValueError("Normalized streaming response has no top-level token_ids")
    return [int(value) for value in values]


def _coverage_exact(
    coverage: list[list[int]], expected_start: int, expected_end: int
) -> bool:
    if not coverage:
        return False
    if int(coverage[0][0]) != expected_start:
        return False
    if int(coverage[-1][1]) != expected_end:
        return False
    return all(
        int(left[1]) == int(right[0])
        for left, right in zip(coverage, coverage[1:])
    )


def _refuse_existing(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite derived evidence without --force: {names}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_paths = [
        run_dir / "phase8_result.offline.json",
        run_dir / "inspection.offline.json",
        run_dir / "OFFLINE_RECONSTRUCTION.json",
        run_dir / "SHA256SUMS.offline",
    ]
    _refuse_existing(output_paths, args.force)
    verified_files = _verify_original_manifest(run_dir)

    session = _load(run_dir / "session_manifest.json")
    cutover = _load(run_dir / "cutover_manifest.json")
    staging = _load(run_dir / "staging_manifest.json")
    takeover = _load(run_dir / "takeover_state.json")
    source_request = _load(run_dir / "source_request.json")
    source_response = _load(run_dir / "source_response.json")
    target_response = _load(run_dir / "target_response.json")
    control_response = _load(run_dir / "control_response.json")

    receiver_root = run_dir / "receiver_receipts"
    receiver_dirs = sorted(path for path in receiver_root.iterdir() if path.is_dir())
    if len(receiver_dirs) != 1:
        raise ValueError("Phase 8 evidence requires exactly one receiver directory")
    deliveries = [
        _load(run_dir / "stage_delivery_receipts" / f"tp_rank_{rank}.json")
        for rank in range(4)
    ]
    receivers = [
        _load(receiver_dirs[0] / f"tp_rank_{rank}.json")
        for rank in range(4)
    ]

    initial_end = int(session["num_computed_tokens"])
    final_end = int(staging["num_computed_tokens"])
    rank_evidence: list[dict[str, Any]] = []
    for rank, (record, delivery, receiver) in enumerate(
        zip(staging["ranks"], deliveries, receivers)
    ):
        coverage = [
            [int(boundary[0]), int(boundary[1])]
            for boundary in record["delta_coverage"]
        ]
        coverage_ok = _coverage_exact(coverage, initial_end, final_end)
        delivery_ok = (
            delivery.get("status") == "READY"
            and receiver.get("status") == "OWNERSHIP_COMMITTED"
            and receiver.get("exact_readback") is True
            and delivery.get("payload_sha256")
            == receiver.get("payload_sha256")
            and int(delivery["payload_bytes"])
            == int(receiver["payload_bytes"])
        )
        rank_evidence.append(
            {
                "tp_rank": rank,
                "delta_ranges": len(coverage),
                "delta_start_token": coverage[0][0] if coverage else None,
                "delta_end_token": coverage[-1][1] if coverage else None,
                "delta_coverage_exact": coverage_ok,
                "delivery_and_gpu_readback_exact": delivery_ok,
                "receiver_status": receiver.get("status"),
                "payload_bytes": delivery.get("payload_bytes"),
                "payload_sha256": delivery.get("payload_sha256"),
            }
        )

    delta_times = [
        float(_load(path)["staged_unix_s"])
        for path in (run_dir / "delta_sender_receipts").glob("*.json")
    ]
    initial_times = [
        float(
            _load(
                run_dir / "initial_stage_receipts" / f"tp_rank_{rank}.json"
            )["completed_unix_s"]
        )
        for rank in range(4)
    ]
    if not delta_times:
        raise ValueError("Phase 8 evidence contains no delta sender receipts")
    first_delta = min(delta_times)
    last_initial = max(initial_times)
    overlap = first_delta < last_initial

    prompt_tokens = int(staging["num_prompt_tokens"])
    cutover_ids = [
        int(token_id)
        for token_id in staging["all_known_token_ids"][prompt_tokens:]
    ]
    target_ids = _streaming_token_ids(target_response)
    source_ids = _ordinary_token_ids(source_response)
    control_ids = _ordinary_token_ids(control_response)
    assembled_ids = cutover_ids + target_ids
    remaining_tokens = int(source_request["max_tokens"]) - int(
        staging["snapshot_num_output_tokens"]
    )
    source_prefix_exact = source_ids[: len(cutover_ids)] == cutover_ids
    continuity_exact = assembled_ids == control_ids
    all_coverage_exact = all(
        record["delta_coverage_exact"] for record in rank_evidence
    )
    all_delivery_exact = all(
        record["delivery_and_gpu_readback_exact"] for record in rank_evidence
    )
    passed = all(
        (
            session.get("phase") == "BridgeTP D3 Phase 8",
            cutover.get("phase") == "BridgeTP D3 Phase 8",
            staging.get("phase") == "BridgeTP D3 Phase 8",
            takeover.get("state") == "COMMITTED",
            takeover.get("source_abort_dispatched") is True,
            _choice(source_response).get("finish_reason") == "abort",
            source_prefix_exact,
            len(target_ids) == remaining_tokens,
            int(staging["new_kv_delta_tokens"]) > 0,
            all_coverage_exact,
            all_delivery_exact,
            overlap,
            continuity_exact,
        )
    )

    reconstruction = {
        "kind": "offline_reconstruction",
        "reason": (
            "original controller used the ordinary-completion token accessor "
            "for a normalized streaming target response"
        ),
        "original_files_modified": False,
        "original_sha256_manifest_verified": True,
        "original_sha256_files_verified": verified_files,
        "source_archive": (
            str(args.source_archive.resolve()) if args.source_archive else None
        ),
        "source_archive_sha256": (
            _sha256(args.source_archive.resolve())
            if args.source_archive
            else None
        ),
        "source_git_revision": (run_dir / "git_revision.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "generated_unix_s": time.time(),
        "unrecoverable_controller_local_metrics": [
            "target_ready_to_commit_response_ms",
            "commit_api_ms",
            "commit_to_target_first_token_ms",
        ],
    }
    result = {
        "status": "PASS" if passed else "FAIL",
        "phase": "BridgeTP D3 Phase 8",
        "mode": "dualwrite_commit",
        "evidence_origin": "offline_reconstruction",
        "migration_id": staging["migration_id"],
        "takeover_state": takeover.get("state"),
        "source_abort_dispatched": takeover.get("source_abort_dispatched"),
        "source_finish_reason": _choice(source_response).get("finish_reason"),
        "old_kv_num_computed_tokens": staging["old_kv_num_computed_tokens"],
        "cutover_num_output_tokens": staging["snapshot_num_output_tokens"],
        "cutover_num_computed_tokens": staging["num_computed_tokens"],
        "new_kv_delta_tokens": staging["new_kv_delta_tokens"],
        "new_kv_delta_batches": staging["new_kv_delta_batches"],
        "all_rank_delta_coverage_exact": all_coverage_exact,
        "all_rank_delivery_exact": all_delivery_exact,
        "rank_evidence": rank_evidence,
        "old_kv_new_kv_overlap_proven": overlap,
        "first_delta_staged_unix_s": first_delta,
        "last_initial_rank_staged_unix_s": last_initial,
        "source_tokens_computed_before_abort": len(source_ids),
        "discarded_source_tokens_after_cutover": max(
            0, len(source_ids) - len(cutover_ids)
        ),
        "source_prefix_matches_cutover": source_prefix_exact,
        "remaining_generation_budget_transferred": remaining_tokens,
        "target_tokens": len(target_ids),
        "assembled_token_ids": assembled_ids,
        "control_token_ids": control_ids,
        "exact_end_to_end_token_continuity": continuity_exact,
        "target_ready_to_commit_response_ms": None,
        "commit_api_ms": None,
        "commit_to_target_first_token_ms": None,
        "reconstruction": reconstruction,
    }
    inspection = {
        "status": result["status"],
        "phase": "BridgeTP D3 Phase 8",
        "mode": "dualwrite_commit",
        "evidence_origin": "offline_reconstruction",
        "migration_id": staging["migration_id"],
        "original_sha256_manifest_verified": True,
        "original_sha256_files_verified": verified_files,
        "old_kv_num_computed_tokens": staging["old_kv_num_computed_tokens"],
        "new_kv_delta_tokens": staging["new_kv_delta_tokens"],
        "cutover_num_computed_tokens": staging["num_computed_tokens"],
        "all_rank_delta_coverage_exact": all_coverage_exact,
        "all_rank_delivery_exact": all_delivery_exact,
        "old_kv_new_kv_overlap_proven": overlap,
        "takeover_state": takeover.get("state"),
        "source_abort_dispatched": takeover.get("source_abort_dispatched"),
        "source_finish_reason": _choice(source_response).get("finish_reason"),
        "source_prefix_matches_cutover": source_prefix_exact,
        "target_tokens": len(target_ids),
        "assembled_tokens": len(assembled_ids),
        "control_tokens": len(control_ids),
        "end_to_end_continuity": continuity_exact,
        "controller_local_timing_recovered": False,
    }

    result_path, inspection_path, provenance_path, sums_path = output_paths
    _atomic_dump(result, result_path)
    _atomic_dump(inspection, inspection_path)
    _atomic_dump(reconstruction, provenance_path)
    sums = "".join(
        f"{_sha256(path)}  {path.name}\n"
        for path in (result_path, inspection_path, provenance_path)
    )
    sums_path.write_text(sums, encoding="utf-8")
    print(json.dumps(inspection, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
