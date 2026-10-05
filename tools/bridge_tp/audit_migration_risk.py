#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit observed TP1 guard risk without inventing migration counterfactuals.

Each decision tick is an observation, not an independent training example.
The episode key must be used for train/validation splitting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bridge_tp.risk_observation import build_risk_observation


def _json_member(archive: tarfile.TarFile, member: tarfile.TarInfo | None) -> Any:
    if member is None:
        return None
    stream = archive.extractfile(member)
    if stream is None:
        return None
    return json.load(stream)


def _fallback_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read legacy M1 snapshots, pairing only the M5 row in the same tick."""
    observations: list[dict[str, Any]] = []
    last_m5: dict[str, Any] | None = None
    assigned: str | None = None
    for row in rows:
        kind = row.get("kind")
        if kind == "manager_m5_predictor_shadow":
            last_m5 = row
        elif kind == "experiment_m1_timing":
            assigned = row.get("assigned_action")
        elif kind == "manager_m1_start_decision":
            decision = row.get("decision") or {}
            observations.append(build_risk_observation(
                tick=row["tick"], snapshot=row["snapshot"],
                m5_row=last_m5, natural_m1=decision, applied_m1=decision,
                initial_rate=row.get("initial_rate_preview"),
                assigned_action=assigned,
            ))
            last_m5 = None
            assigned = None
    return observations


def _release_endpoint(response: dict[str, Any] | None,
                      receipt: dict[str, Any] | None) -> tuple[float | None, str]:
    if receipt and receipt.get("status") == "SOURCE_KV_RELEASED":
        ns = receipt.get("released_unix_ns")
        if isinstance(ns, int) and ns > 0:
            return ns / 1e9, "EXACT_KV_RELEASE_RECEIPT"
    if response and response.get("finish_reason") == "stop":
        completed = response.get("completed_unix_s")
        if isinstance(completed, (int, float)) and math.isfinite(completed):
            return float(completed), "NATURAL_EOS_RESPONSE_PROXY"
    return None, "CENSORED_NO_KV_RELEASE_EVIDENCE"


def audit_episode(episode: str, rows: list[dict[str, Any]],
                  response: dict[str, Any] | None,
                  receipt: dict[str, Any] | None) -> list[dict[str, Any]]:
    observations = [row for row in rows
                    if row.get("kind") == "manager_risk_observation_shadow"]
    if not observations:
        observations = _fallback_observations(rows)
    telemetry = [row for row in rows if row.get("kind") == "telemetry"]
    release_s, release_evidence = _release_endpoint(response, receipt)
    output: list[dict[str, Any]] = []
    arm = episode.split("/")[-2] if "/" in episode else "UNKNOWN"
    for observation in observations:
        t0 = observation.get("unix_s")
        if not isinstance(t0, (int, float)):
            continue
        future = [row for row in telemetry
                  if isinstance(row.get("unix_s"), (int, float))
                  and row["unix_s"] > t0
                  and (release_s is None or row["unix_s"] <= release_s)]
        hit_s = None
        for row in future:
            signal = row.get("capacity_signal") or {}
            free = signal.get("free_kv_tokens")
            guard = signal.get("guard_free_kv_tokens")
            pending = signal.get("prefill_pending_kv_tokens")
            if all(isinstance(x, int) for x in (free, guard, pending)):
                if free - guard - pending <= 0:
                    hit_s = row["unix_s"]
                    break
        if hit_s is not None:
            status = "OBSERVED_SAMPLED_GUARD_HIT"
        elif release_s is not None and future:
            status = "NO_HIT_IN_SAMPLES"
        else:
            status = "CENSORED"
        horizons: dict[str, str] = {}
        for horizon in (5, 10, 30):
            if hit_s is not None and hit_s - t0 <= horizon:
                horizon_status = "OBSERVED_HIT"
            elif release_s is not None and release_s <= t0 + horizon and future:
                horizon_status = "NO_HIT_BEFORE_RELEASE_IN_SAMPLES"
            elif future and future[-1]["unix_s"] >= t0 + horizon:
                horizon_status = "NO_HIT_IN_WINDOW_SAMPLES"
            else:
                horizon_status = "CENSORED"
            horizons[str(horizon)] = horizon_status
        output.append({
            "episode": episode, "arm": arm,
            "observation": observation,
            "label": {
                "observed_guard_status": status,
                "observed_time_to_guard_s": (
                    hit_s - t0 if hit_s is not None else None),
                "guard_horizon_s": horizons,
                "sample_count_after_decision": len(future),
                "source_release_evidence": release_evidence,
                "observed_time_to_release_s": (
                    max(0.0, release_s - t0)
                    if release_s is not None and release_s >= t0 else None),
                "counterfactual_stay_guard_status": (
                    "OBSERVED_ARM" if arm == "stay" else "UNKNOWN"),
                "true_oom_status": "NOT_ESTABLISHED_BY_GUARD_TELEMETRY",
            },
        })
    return output


def audit_archive(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with tarfile.open(path, "r:gz") as archive:
        members = {item.name: item for item in archive if item.isfile()}
        for name, member in members.items():
            suffix = "/controller/phase9_audit.jsonl"
            if not name.endswith(suffix):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            rows = [json.loads(line) for line in stream if line.strip()]
            prefix = name[:-len(suffix)]
            response = _json_member(
                archive, members.get(prefix + "/controller/source_response.json"))
            receipt = _json_member(
                archive, members.get(
                    prefix + "/controller/source_kv_release_receipt.json"))
            result.extend(audit_episode(prefix, rows, response, receipt))
    return result


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = {item["episode"] for item in items}
    statuses = Counter(item["label"]["observed_guard_status"]
                       for item in items)
    evidence = Counter(item["label"]["source_release_evidence"]
                       for item in items)
    m5 = Counter(item["observation"]["m5_status"] for item in items)
    horizon_status = {
        horizon: dict(Counter(item["label"]["guard_horizon_s"][horizon]
                              for item in items))
        for horizon in ("5", "10", "30")
    }
    hit_episodes = {item["episode"] for item in items if item["label"][
        "observed_guard_status"] == "OBSERVED_SAMPLED_GUARD_HIT"}
    return {
        "format_version": 1, "episodes": len(episodes),
        "decision_ticks": len(items), "observed_guard_hit_episodes": len(hit_episodes),
        "guard_status_ticks": dict(statuses),
        "guard_horizon_status_ticks": horizon_status,
        "release_evidence_ticks": dict(evidence), "m5_status_ticks": dict(m5),
        "calibration_status": (
            "COVERAGE_ONLY_NO_RISK_FIT" if not hit_episodes
            else "EVENTS_PRESENT_FIT_REQUIRES_EPISODE_SPLIT"),
        "warning": (
            "Guard samples are not true OOM; migration does not reveal "
            "STAY counterfactual."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"archive missing: {args.archive}")
    items = audit_archive(args.archive)
    if not items:
        raise ValueError("no M1/risk observations in archive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "risk_observations.jsonl").open("w", encoding="utf-8") as out:
        for item in items:
            out.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    summary = summarize(items)
    (args.out_dir / "risk_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
