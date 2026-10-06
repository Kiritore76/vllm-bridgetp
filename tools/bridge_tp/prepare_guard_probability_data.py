#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare observed STAY guard labels for later probability calibration.

Only successful STAY episodes reveal the guard trajectory under STAY. Decision
ticks from one episode remain grouped; their count is never treated as a count
of independent guard events. This tool does not fit or publish probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bridge_tp.audit_migration_risk import audit_archive

HORIZONS = (5, 10, 30)
POSITIVE = "OBSERVED_HIT"
NEGATIVE = {"NO_HIT_IN_WINDOW_SAMPLES", "NO_HIT_BEFORE_RELEASE_IN_SAMPLES"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_statuses(path: Path) -> dict[str, str]:
    """Read each episode's runner status without extracting archive paths."""
    statuses: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith("/status.json"):
                continue
            if not re.fullmatch(
                r"r\d+_shadow_only", member.name.split("/")[-2]
            ):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            row = json.load(stream)
            statuses[member.name.removesuffix("/status.json")] = row.get(
                "status", "UNKNOWN"
            )
    return statuses


def workload_provenance(path: Path) -> tuple[str, int | None]:
    """Separate prompt augmentation and retain the selected workload seed."""
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith("/pilot_summary.json"):
                stream = archive.extractfile(member)
                if stream is None:
                    break
                summary = json.load(stream)
                if summary.get("source_prompt_tokens") is not None:
                    return "PROMPT_AUGMENTED_DIAGNOSTIC", summary.get("seed")
                return "NATURAL_PROMPTS", summary.get("seed")
    return "UNKNOWN_PROVENANCE", None


def eligible_rows(
    observations: list[dict[str, Any]], statuses: dict[str, str],
    archive_name: str, archive_sha256: str,
    kind: str = "UNKNOWN_PROVENANCE",
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retain observed STAY labels and expose exclusions explicitly."""
    rows: list[dict[str, Any]] = []
    episode_reasons: dict[str, str] = {}
    for item in observations:
        episode = item["episode"]
        arm = item["arm"].lower()
        status = statuses.get(episode, "MISSING_STATUS")
        if arm != "stay":
            episode_reasons[episode] = "INTERVENED_ARM"
            continue
        if status != "PASS":
            episode_reasons[episode] = f"RUNNER_{status}"
            continue
        if item["label"]["counterfactual_stay_guard_status"] != "OBSERVED_ARM":
            raise ValueError(f"STAY label has no observed STAY evidence: {episode}")
        labels = {}
        for horizon in HORIZONS:
            state = item["label"]["guard_horizon_s"][str(horizon)]
            labels[str(horizon)] = (
                1 if state == POSITIVE else 0 if state in NEGATIVE else None
            )
        rows.append({
            "episode_group": f"{archive_sha256}:{episode}",
            "workload_block": (
                f"{seed}:{episode.split('/')[-3]}:{kind}"
                if seed is not None else f"{archive_sha256}:{episode}"
            ),
            "workload_seed": seed,
            "archive": archive_name,
            "archive_sha256": archive_sha256,
            "workload_class": kind,
            "episode": episode,
            "tick": item["observation"]["tick"],
            "unix_s": item["observation"]["unix_s"],
            "features": item["observation"],
            "guard_hit_within_horizon": labels,
            "sampled_label_status": item["label"]["guard_horizon_s"],
            "release_evidence": item["label"]["source_release_evidence"],
        })
    rows.sort(key=lambda row: (row["episode_group"], row["unix_s"], row["tick"]))
    return rows, {
        "excluded_episodes": dict(Counter(episode_reasons.values())),
        "excluded_episode_count": len(episode_reasons),
    }


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count independent episode support and missing predictor features."""
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_group"]].append(row)
        by_block[row["workload_block"]].append(row)
    classes: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for episode, group in by_episode.items():
        classes[group[0]["workload_class"]][episode] = group
    class_summary = {}
    for kind, groups in classes.items():
        class_summary[kind] = {
            "episodes": len(groups),
            "decision_ticks": sum(len(group) for group in groups.values()),
            "guard_hit_episodes": {
                str(horizon): sum(
                    any(row["guard_hit_within_horizon"][str(horizon)] == 1
                        for row in group)
                    for group in groups.values()
                )
                for horizon in HORIZONS
            },
        }
    horizons = {}
    for horizon in HORIZONS:
        key = str(horizon)
        labels = Counter(row["guard_hit_within_horizon"][key] for row in rows)
        hit_episodes = sum(
            any(row["guard_hit_within_horizon"][key] == 1 for row in group)
            for group in by_episode.values()
        )
        no_hit_episodes = sum(
            any(row["guard_hit_within_horizon"][key] == 0 for row in group)
            for group in by_episode.values()
        )
        hit_blocks = sum(
            any(row["guard_hit_within_horizon"][key] == 1 for row in group)
            for group in by_block.values()
        )
        horizons[key] = {
            "positive_ticks": labels[1],
            "negative_ticks": labels[0],
            "censored_ticks": labels[None],
            "episodes_with_positive": hit_episodes,
            "episodes_with_negative": no_hit_episodes,
            "workload_blocks_with_positive": hit_blocks,
            "fit_readiness": (
                "CANDIDATE_FOR_GROUPED_CALIBRATION"
                if hit_blocks >= 10 and no_hit_episodes >= 10
                else "INSUFFICIENT_INDEPENDENT_EVENTS"
            ),
        }
    return {
        "format_version": 1,
        "label_definition": (
            "Sampled TP1 free KV minus guard and pending prefill reaches "
            "zero within horizon, before anchor KV release, under STAY"
        ),
        "probability_status": "NOT_FITTED",
        "episode_count": len(by_episode),
        "workload_block_count": len(by_block),
        "distinct_workload_seeds": sorted({
            row["workload_seed"] for row in rows
            if row["workload_seed"] is not None
        }),
        "decision_ticks": len(rows),
        "point_time_to_guard_present_ticks": sum(
            row["features"].get("point_time_to_guard_s") is not None
            for row in rows
        ),
        "m5_available_ticks": sum(
            row["features"].get("m5_status") == "AVAILABLE" for row in rows
        ),
        "workload_classes": class_summary,
        "horizons_s": horizons,
        "note": (
            "Ticks within an episode are correlated. Guard contact is not OOM. "
            "These observed STAY labels do not prove migration benefit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    sources = []
    seen_digests = set()
    for path in args.archive:
        if not path.is_file():
            parser.error(f"archive missing: {path}")
        digest = sha256(path)
        if digest in seen_digests:
            parser.error(f"duplicate archive contents: {path}")
        seen_digests.add(digest)
        observations = audit_archive(path)
        kind, seed = workload_provenance(path)
        selected, exclusions = eligible_rows(
            observations, episode_statuses(path), path.name, digest, kind, seed
        )
        rows.extend(selected)
        sources.append({
            "path": str(path.resolve()), "sha256": digest,
            "workload_class": kind,
            "workload_seed": seed,
            "observed_episodes": len({r["episode"] for r in observations}),
            "eligible_stay_episodes": len({r["episode"] for r in selected}),
            **exclusions,
        })
    result = coverage(rows)
    result["sources"] = sources
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "guard_calibration_rows.jsonl").open(
        "w", encoding="utf-8"
    ) as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (args.out_dir / "guard_calibration_coverage.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
