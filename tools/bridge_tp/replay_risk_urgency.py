#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Recompute F1 from existing raw event archives, without future predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bridge_tp.experiment_probability_gate import ProbabilityGate
from tools.bridge_tp.risk_urgency import DecodeRateTracker, build_snapshot


def read_json(archive: tarfile.TarFile, name: str, default=None):
    try:
        stream = archive.extractfile(name)
    except KeyError:
        return default
    return json.load(stream) if stream else default


def replay(path: Path, thresholds: tuple[float, ...]) -> list[dict]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    output = []
    with tarfile.open(path, "r:gz") as archive:
        summary = next(
            (
                read_json(archive, m.name)
                for m in archive
                if m.isfile() and m.name.endswith("/pilot_summary.json")
            ),
            {},
        )
        kind = (
            "PROMPT_AUGMENTED_DIAGNOSTIC"
            if summary.get("source_prompt_tokens")
            else "NATURAL_PROMPTS"
        )
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(
                "/controller/phase9_audit.jsonl"
            ):
                continue
            episode = member.name.removesuffix("/controller/phase9_audit.jsonl")
            predictor_name = episode + "/predictor_events.jsonl"
            try:
                stream = archive.extractfile(predictor_name)
            except KeyError:
                continue
            events = [json.loads(line) for line in stream if line.strip()]
            if not events or events[0].get("kind") != "predictor_header":
                raise ValueError(f"missing header: {predictor_name}")
            header = events[0]
            predictions = {
                (r.get("request_id"), r.get("generated_tokens")): r
                for r in events[1:]
                if r.get("kind") == "predictor_prediction"
            }
            rows = [
                json.loads(line) for line in archive.extractfile(member) if line.strip()
            ]
            telemetry = {r["tick"]: r for r in rows if r.get("kind") == "telemetry"}
            config = next((r["config"] for r in rows if "config" in r), {})
            source = read_json(archive, episode + "/controller/source_request.json", {})
            tracker = DecodeRateTracker()
            gate = ProbabilityGate(
                thresholds[0], episode, assigned_action="STAY", thresholds=thresholds
            )
            m5 = {}
            for row in rows:
                if row.get("kind") == "manager_m5_predictor_shadow":
                    m5 = row
                if row.get("kind") != "manager_m1_start_decision":
                    continue
                snap = dict(row["snapshot"])
                snap["target_prefill_pending_kv_tokens"] = (
                    telemetry.get(row["tick"], {})
                    .get("tp4", {})
                    .get("prefill_pending_kv_tokens")
                )
                prediction = dict(m5)
                raw = predictions.get(
                    (snap.get("request_id"), m5.get("prediction_output_tokens"))
                )
                if raw:
                    age = snap["unix_s"] - raw["captured_unix_ns"] / 1e9
                    lag = snap["generated_tokens"] - raw["generated_tokens"]
                    if (
                        raw.get("checkpoint_sha256") != header["checkpoint_sha256"]
                        or m5.get("output_tokens") != snap["generated_tokens"]
                        or not -1 <= age <= 30
                        or not 0 <= lag <= 40
                    ):
                        prediction["status"] = "STALE"
                    prediction.update(
                        probabilities=raw["probabilities"],
                        category_upper_edges=header["category_upper_edges"],
                        captured_unix_ns=raw["captured_unix_ns"],
                    )
                else:
                    prediction["status"] = "MISSING_RAW_PREDICTION"
                if prediction.get("max_remaining_output_tokens") is None:
                    cap = source.get("max_tokens")
                    prediction["max_remaining_output_tokens"] = (
                        max(0, cap - snap["generated_tokens"])
                        if cap is not None
                        else None
                    )
                observation = build_snapshot(
                    snapshot=snap,
                    prediction=prediction,
                    candidate_rate=tracker.update(
                        snap["unix_s"], snap["generated_tokens"]
                    ),
                    initial_rate=row.get("initial_rate_preview") or {},
                    kv_bytes_per_token=config.get("policy", {}).get(
                        "kv_bytes_per_token", 0
                    ),
                    release_tail_s=row.get("decision", {}).get("source_release_tail_s")
                    or 5.0,
                    block_size=config.get("block_size", 16),
                    model_config_sha256=header.get("model_config_sha256"),
                ).to_json()
                observation.update(
                    archive=str(path.resolve()),
                    archive_sha256=digest,
                    raw_predictor_member=predictor_name,
                    raw_audit_member=member.name,
                    episode=episode,
                    arm=episode.split("/")[-2],
                    provenance_kind=kind,
                    seed=summary.get("seed"),
                    tick=row["tick"],
                    release_tail_provenance="logged_or_frozen_5s_allowance",
                )
                observation["gate_replay"] = gate.observe(
                    observation.copy(), row["tick"]
                )
                output.append(observation)
    return output


def summarize(rows: list[dict], thresholds: tuple[float, ...]) -> dict:
    groups = {}
    for kind in sorted({r["provenance_kind"] for r in rows}):
        subset = [r for r in rows if r["provenance_kind"] == kind]
        valid = [r for r in subset if r["status"] == "VALID"]
        lower = sorted(r["p_guard_est_bounds"][0] for r in valid)
        urgencies = sorted(r["U"] for r in valid if r["U"] is not None)
        groups[kind] = {
            "episodes": len({r["episode"] for r in subset}),
            "ticks": len(subset),
            "valid_ticks": len(valid),
            "status_counts": dict(Counter(r["status"] for r in subset)),
            "p_lower_quantiles_0_25_50_75_100": [
                lower[int((len(lower) - 1) * q)] for q in (0, 0.25, 0.5, 0.75, 1)
            ]
            if lower
            else [],
            "U_min_max": [urgencies[0], urgencies[-1]] if urgencies else [],
            "feasible_ticks": sum(r["physical_feasible"] for r in subset),
            "stay_episodes": len({r["episode"] for r in subset if r["arm"] == "stay"}),
            "p_U_cells": {
                f"p[{pl},{ph})_U[{ul},{uh})": {
                    "ticks": sum(
                        pl <= r["p_guard_est_bounds"][0] < ph
                        and r["U"] is not None
                        and ul <= r["U"] < uh
                        for r in valid
                    ),
                    "episodes": len(
                        {
                            r["episode"]
                            for r in valid
                            if pl <= r["p_guard_est_bounds"][0] < ph
                            and r["U"] is not None
                            and ul <= r["U"] < uh
                        }
                    ),
                }
                for pl, ph in ((0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.01))
                for ul, uh in ((0, 0.1), (0.1, 0.5), (0.5, 1e6))
            },
            "threshold_coverage": {
                str(t): {
                    "episodes_crossing": len(
                        {r["episode"] for r in valid if r["p_guard_est_bounds"][0] >= t}
                    ),
                    "episodes_feasible_crossing": len(
                        {
                            r["episode"]
                            for r in valid
                            if r["p_guard_est_bounds"][0] >= t
                            and r["physical_feasible"]
                        }
                    ),
                }
                for t in thresholds
            },
        }
    candidates = {}
    for row in rows:
        if row["status"] != "VALID" or not row["physical_feasible"]:
            continue
        for theta in thresholds:
            key = f"{row['episode']}:{theta:g}"
            if row["p_guard_est_bounds"][0] >= theta and key not in candidates:
                candidates[key] = {
                    "episode": row["episode"],
                    "theta": theta,
                    "tick": row["tick"],
                    "generated_tokens": row["generated_tokens"],
                    "p_bounds": row["p_guard_est_bounds"],
                    "U": row["U"],
                    "provenance_kind": row["provenance_kind"],
                }
    return {
        "status": "OFFLINE_F1_REPLAY_NOT_CALIBRATED_POOL_OOM_NOT_FITTED_BENEFIT",
        "thresholds": thresholds,
        "groups": groups,
        "independent_samples_are_not_ticks": True,
        "first_feasible_candidates": list(candidates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs="+", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = [r for path in args.archive for r in replay(path, tuple(args.thresholds))]
    with (args.out_dir / "risk_urgency_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = summarize(rows, tuple(args.thresholds))
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": report["status"], "groups": report["groups"]}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
