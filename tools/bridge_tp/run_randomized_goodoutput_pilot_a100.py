#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect a six-trace randomized natural-EOS migration timing pilot on A100.

The pilot checks the paired collection path and actual M1 start tokens. The
configured minimum output token is an eligibility gate, not an exact action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp.horizon_goodoutput import score_horizon  # noqa: E402
from tools.bridge_tp.run_goodoutput_matrix_a100 import (  # noqa: E402
    execute,
    online_command,
    sha256,
    verify,
    write_json,
)

SEED = 20261004
CASES = (
    ("p00_source1_target2", 1, 2, 1.2, 0.8),
    ("p01_source1_target16", 1, 16, 0.35, 0.8),
    ("p02_source3_target8", 3, 8, 0.65, 0.35),
    ("p03_source3_target24", 3, 24, 0.2, 0.35),
    ("p04_source5_target2", 5, 2, 1.2, 0.15),
    ("p05_source5_target24", 5, 24, 0.2, 0.15),
)
ACTIONS = ("stay", "early128", "late1024")
TIMING_ACTIONS = ("stay", "now", "wait")
RETIRED_P03_ANCHOR_ID = "oasst1:98d36c03-5335-4f5b-8dbd-1a6e5fd9b0b2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "input", "base", "survival", "guard",
                 "checkpoint", "reference", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-host")
    parser.add_argument("--expected-gpu-uuids")
    parser.add_argument("--portable-hardware", action="store_true",
                        help="Record actual AutoDL host/UUIDs; require five idle A100s")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--probability-pilot", action="store_true")
    parser.add_argument("--guard-profile", choices=("legacy8448", "reduced2000"),
                        default="legacy8448",
                        help="Pin the guard value and file SHA for this collection")
    parser.add_argument("--slo-profile", choices=("legacy1pct", "slow2pct"),
                        default="legacy1pct",
                        help="Pin the SLO reference SHA for this collection")
    parser.add_argument("--pre-episode-warmup", action="store_true",
                        help="Warm prompt shapes before the measured workload")
    parser.add_argument("--minimum-initial-source-headroom-tokens", type=int)
    parser.add_argument("--constructed-workload", action="store_true",
                        help="Use marked engineering construction recipes")
    parser.add_argument("--expected-input-sha256",
                        help="Pinned request-file SHA for constructed workloads")
    parser.add_argument("--constructed-source-count", type=int, default=3)
    parser.add_argument("--constructed-target-count", type=int, default=8)
    parser.add_argument("--probability-thresholds", type=float, nargs="+",
                        default=[0.0, 0.01, 0.05, 0.2])
    parser.add_argument("--probability-arms", nargs="+",
                        help="Select preregistered arms for original-code A/B")
    parser.add_argument("--paired-only", action="store_true",
                        help="Collect STAY and early128 without the late EOS arm")
    parser.add_argument("--timing-pilot", action="store_true",
                        help="Compare STAY, NOW, and one M5-refresh WAIT")
    parser.add_argument("--risk-observation-shadow", action="store_true",
                        help="record passive predecision risk features in each arm")
    parser.add_argument("--cross-context-smoke", action="store_true",
                        help="one forced-length migration beyond TP1 context; "
                             "not a natural-EOS or GoodOutput experiment")
    parser.add_argument("--actions", nargs="+", choices=TIMING_ACTIONS,
                        help="Collect only these timing actions")
    parser.add_argument("--cases", nargs="+", choices=[row[0] for row in CASES],
                        help="Run only these cases with the original seed")
    parser.add_argument("--anchor-context-limit", action="store_true",
                        help="Use the largest anchor output allowed by context")
    parser.add_argument(
        "--anchor-total-max-tokens", type=int,
        help="client-visible total output budget after TP1-to-TP4 migration",
    )
    parser.add_argument(
        "--p03-anchor-id",
        help="held-out long-form request ID replacing the p03 anchor",
    )
    parser.add_argument("--background-context-limit", action="store_true",
                        help="Use the largest background output allowed by context")
    parser.add_argument("--background-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--source-background-max-tokens", type=int,
        help="source-only output cap; target jobs retain background setting",
    )
    parser.add_argument(
        "--source-prompt-tokens", type=int,
        help="augment held-out source prompts to this exact length; "
             "preserve natural EOS and record augmentation provenance",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp4-max-model-len", type=int)
    parser.add_argument("--random-arrivals", action="store_true",
                        help="Use seeded exponential interarrival gaps")
    parser.add_argument("--arrival-window-s", type=float,
                        help="Replay each case's held-out requests in repeated "
                             "waves until this fixed arrival window ends")
    parser.add_argument("--arrival-wave-period-s", type=float, default=40.0,
                        help="Spacing between repeated arrival waves")
    parser.add_argument("--max-arrival-lag-s", type=float, default=2.0,
                        help="Invalidate a fixed-window arm if an actual "
                             "background arrival misses its schedule")
    parser.add_argument("--evaluation-horizon-s", type=float, default=180.0)
    parser.add_argument("--window-token-goodoutput", action="store_true",
                        help="Count only window tokens; use frozen full-request "
                             "SLO after drain, including requests ending after H")
    return parser.parse_args()


def select_inputs(path: Path, seed: int = SEED,
                  p03_anchor_id: str | None = None,
                  all_held_out: bool = False,
                  constructed_workload: bool = False) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8")
            .splitlines() if line.strip()]
    tree_splits: dict[str, str] = {}
    content_splits: dict[str, str] = {}
    for row in rows:
        tree_id = str(row.get("source_tree_id") or row["id"])
        split = str(row.get("split"))
        if tree_id in tree_splits and tree_splits[tree_id] != split:
            raise ValueError(f"request tree crosses dataset splits: {tree_id}")
        tree_splits[tree_id] = split
        if all_held_out:
            content = json.dumps(row.get("messages", row.get("prompt")),
                                 sort_keys=True, ensure_ascii=False)
            digest = hashlib.sha256(content.encode()).hexdigest()
            for key in (digest, row.get("original_prompt_sha256")):
                if key is not None:
                    if key in content_splits and content_splits[key] != split:
                        raise ValueError("request content crosses predictor splits")
                    content_splits[key] = split
    if constructed_workload:
        from tools.bridge_tp.build_constructed_probability_workload import SPLITS

        if p03_anchor_id is not None or any(
            row.get("workload_origin") != "constructed"
            or row.get("split") not in SPLITS
            or row.get("controller_split") != row.get("split")
            or not isinstance(row.get("construction"), dict)
            or row["construction"].get("role") not in {"anchor", "background"}
            or row.get("workload_group")
            != "constructed_" + row["construction"].get("role", "")
            for row in rows
        ) or len({row["split"] for row in rows}) != 1:
            raise ValueError("constructed input requires one engineering split")
        test = rows
        anchor_group, background_group = "constructed_anchor", "constructed_background"
    else:
        test = [row for row in rows if row.get("split") == "test"
                and row.get("workload_group") in {"natural", "long_form"}
                and (p03_anchor_id is None
                     or row.get("id") != RETIRED_P03_ANCHOR_ID)]
        anchor_group, background_group = "long_form", "natural"
    rng = random.Random(seed)
    long_form = [row for row in test
                 if row["workload_group"] == anchor_group]
    natural = [row for row in test
               if row["workload_group"] == background_group]
    rng.shuffle(long_form)
    rng.shuffle(natural)
    if len(long_form) < 5 or len(natural) < 1:
        raise ValueError("pilot needs five anchor-role and one background-role request")
    anchors = long_form[:6] if constructed_workload else long_form[:5] + natural[:1]
    if len(anchors) != 6:
        raise ValueError("constructed pilot needs six distinct anchors")
    if p03_anchor_id is not None:
        replacements = [row for row in long_form
                        if str(row["id"]) == p03_anchor_id]
        if len(replacements) != 1:
            raise ValueError("p03 replacement must be one held-out long-form row")
        replacement = replacements[0]
        existing = next((index for index, row in enumerate(anchors)
                         if row["id"] == p03_anchor_id), None)
        if existing is not None and existing != 3:
            anchors[existing], anchors[3] = anchors[3], anchors[existing]
        else:
            anchors[3] = replacement
    selected_ids = {str(row["id"]) for row in anchors}
    if len(selected_ids) != len(anchors):
        raise ValueError("duplicate anchor request IDs")
    others = [row for row in test if str(row["id"]) not in selected_ids]
    rng.shuffle(others)
    required = sum(source + target for _, source, target, *_ in CASES)
    if len(others) < required:
        raise ValueError(f"pilot needs {required} unique background requests")
    selected = anchors + (others if all_held_out else others[:required])
    if len({str(row["id"]) for row in selected}) != len(selected):
        raise ValueError("pilot request IDs are not unique")
    return selected


def augmented_source_prompt(tokenizer: Any, row: dict[str, Any],
                            length: int) -> list[int]:
    """Repeat held-out source content inside one chat prompt to a fixed length."""
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": "<CONTEXT>"}],
        tokenize=False, add_generation_prompt=True)
    if template.count("<CONTEXT>") != 1:
        raise ValueError("chat template did not retain context marker")
    before, after = template.split("<CONTEXT>")
    prefix = tokenizer.encode(
        before + "Read the following context.\n", add_special_tokens=False)
    suffix = tokenizer.encode(
        "\nWrite a detailed synthesis of about 350 words. Explain the main "
        "points and end with a brief conclusion." + after,
        add_special_tokens=False)
    budget = length - len(prefix) - len(suffix)
    if budget < 128:
        raise ValueError("source prompt target leaves too little context room")
    content = "\n".join(str(message.get("content", ""))
                        for message in row["messages"])
    body = tokenizer.encode(content + "\n", add_special_tokens=False)
    if not body:
        raise ValueError(f"empty source content: {row['id']}")
    repeat = (budget + len(body) - 1) // len(body)
    prompt = prefix + (body * repeat)[:budget] + suffix
    if len(prompt) != length:
        raise AssertionError("source prompt augmentation length differs")
    return prompt


def case_definitions(args: argparse.Namespace) -> tuple:
    if getattr(args, "constructed_workload", False):
        source = getattr(args, "constructed_source_count", 3)
        target = getattr(args, "constructed_target_count", 8)
        if not 1 <= source <= 5 or not 0 <= target <= 24 or args.cases:
            raise ValueError("constructed load needs source 1..5, target 0..24")
        name = f"constructed_source{source}_target{target}"
        return ((name, source, target, 0.35, 0.2),)
    return CASES


def validate_constructed_slo_prompts(
    anchor: dict[str, Any], jobs: list[dict[str, Any]], reference: Path,
) -> None:
    """Reject unsupported prompt lengths before paying for GPU episodes."""
    from tools.bridge_tp.audit_slo_v6 import ttft_limit_ms

    config = json.loads(reference.read_text(encoding="utf-8"))
    for request in [anchor] + [job["request"] for job in jobs]:
        ttft_limit_ms(len(request["prompt"]), config)


def build_setup(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    seed = getattr(args, "seed", SEED)
    rows = select_inputs(args.input, seed, getattr(args, "p03_anchor_id", None),
                         getattr(args, "probability_pilot", False),
                         getattr(args, "constructed_workload", False))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def tokens(row: dict[str, Any], pool: str | None = None) -> list[int]:
        if getattr(args, "constructed_workload", False):
            from tools.bridge_tp.build_constructed_probability_workload import (
                constructed_prompt_tokens,
            )

            return constructed_prompt_tokens(tokenizer, row, pool)
        prompt = row.get("prompt")
        if prompt is None:
            prompt = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True)
        value = tokenizer.encode(prompt)
        if not value:
            raise ValueError(f"empty prompt for {row['id']}")
        return value

    setup: dict[str, Any] = {
        "format_version": 1, "seed": seed,
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "manifests": {}, "anchors": {}, "cases": [],
    }
    cases = case_definitions(args)
    cursor = len(CASES)
    rotation_cursor = sum(row[1] + row[2] for row in cases) + len(CASES)
    for index, (name, source_count, target_count, target_spacing,
                source_spacing) in enumerate(cases):
        if (getattr(args, "probability_pilot", False) and args.cases
                and name not in args.cases):
            cursor += source_count + target_count
            continue
        arrival_rng = random.Random(f"{seed}:{name}:arrivals")
        offsets: dict[str, list[float]] = {}
        for pool, count, spacing, initial in (
            ("target", target_count, target_spacing, 0.5),
            ("source", source_count, source_spacing, 0.2),
        ):
            values = []
            elapsed = initial
            for job_index in range(count):
                if job_index:
                    gap = (arrival_rng.expovariate(1 / spacing)
                           if getattr(args, "random_arrivals", False)
                           else spacing)
                    elapsed += gap
                values.append(round(elapsed, 3))
            offsets[pool] = values
        anchor = rows[index]
        anchor_tokens = tokens(anchor)
        source_max_model_len = getattr(args, "max_model_len", 8192)
        anchor_cap = (min(source_max_model_len - len(anchor_tokens),
                          source_max_model_len - 128)
                      if getattr(args, "anchor_context_limit", False)
                      else 4096)
        total_budget = getattr(args, "anchor_total_max_tokens", None)
        if total_budget is None:
            total_budget = anchor_cap
        if (anchor_cap <= 128
                or len(anchor_tokens) + anchor_cap > source_max_model_len):
            raise ValueError(f"anchor exceeds context limit: {anchor['id']}")
        target_max_model_len = (
            getattr(args, "tp4_max_model_len", None) or source_max_model_len
        )
        if (total_budget < anchor_cap
                or len(anchor_tokens) + total_budget > target_max_model_len):
            raise ValueError(f"anchor total budget exceeds TP4: {anchor['id']}")
        anchor_request = {
            "model": "bridgetp-model", "prompt": anchor_tokens,
            "max_tokens": anchor_cap,
            "ignore_eos": getattr(args, "cross_context_smoke", False),
            "temperature": 0.0, "top_p": 1.0, "top_k": 0,
            "min_p": 0.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "repetition_penalty": 1.0,
            "n": 1, "use_beam_search": False, "stream": True,
            "bridgetp_group_id": None,
            "bridgetp_group_longest": False,
        }
        anchor_path = root / "inputs" / f"{name}_anchor.json"
        write_json(anchor_path, anchor_request)
        setup["anchors"][name] = {
            "path": str(anchor_path), "sha256": sha256(anchor_path),
            "prompt_tokens": len(anchor_tokens), "input_id": anchor["id"],
            "max_tokens": anchor_cap,
            "total_max_tokens": total_budget,
            "workload_group": anchor["workload_group"],
            "source_tree_id": anchor.get("source_tree_id", anchor["id"]),
            "controller_split": anchor.get(
                "controller_split", "engineering_pilot_not_final_test"),
            "workload_origin": anchor.get("workload_origin", "held_out_natural"),
            "construction": anchor.get("construction"),
            "messages_sha256": hashlib.sha256(json.dumps(
                anchor.get("messages", anchor.get("prompt")), sort_keys=True,
                ensure_ascii=False).encode()).hexdigest(),
            "prompt_token_ids_sha256": hashlib.sha256(json.dumps(
                anchor_tokens).encode()).hexdigest(),
            "input_augmentation": anchor.get("augmentation"),
        }
        jobs = []
        chosen = rows[cursor:cursor + source_count + target_count]
        if getattr(args, "constructed_workload", False):
            chosen = chosen[source_count:] + chosen[:source_count]
        cursor += len(chosen)
        for pool, subset in (("target", chosen[:target_count]),
                             ("source", chosen[target_count:])):
            for job_index, row in enumerate(subset):
                prompt_tokens = (
                    augmented_source_prompt(tokenizer, row,
                                            args.source_prompt_tokens)
                    if pool == "source"
                    and getattr(args, "source_prompt_tokens", None) is not None
                    else tokens(row, pool)
                )
                pool_context = (source_max_model_len if pool == "source"
                                else target_max_model_len)
                background_cap = (
                    min(pool_context - len(prompt_tokens), pool_context - 128)
                    if getattr(args, "background_context_limit", False)
                    else getattr(args, "background_max_tokens", 2048)
                )
                if (pool == "source" and getattr(
                        args, "source_background_max_tokens", None) is not None):
                    background_cap = args.source_background_max_tokens
                if (background_cap <= 0
                        or len(prompt_tokens) + background_cap > pool_context):
                    raise ValueError(f"background exceeds context: {row['id']}")
                job = {
                    "job_id": f"{pool}_{job_index:03d}", "pool": pool,
                    "start_after_s": offsets[pool][job_index],
                    "request": {
                        "model": "bridgetp-model", "prompt": prompt_tokens,
                        "max_tokens": background_cap,
                        "ignore_eos": False,
                        "temperature": 0.0,
                    },
                    "input_id": row["id"],
                    "source_tree_id": row.get("source_tree_id", row["id"]),
                    "input_augmentation": row.get("augmentation"),
                    "workload_origin": row.get("workload_origin", "held_out_natural"),
                    "controller_split": row.get(
                        "controller_split", "engineering_pilot_not_final_test"),
                    "construction": row.get("construction"),
                    "workload_group": (
                        "augmented_long_context_natural_eos"
                        if pool == "source"
                        and getattr(args, "source_prompt_tokens", None)
                        is not None else row["workload_group"]
                    ),
                }
                if pool == "source":
                    job["start_after_event"] = "ANCHOR_FIRST_OUTPUT"
                jobs.append(job)
        arrival_window_s = getattr(args, "arrival_window_s", None)
        if arrival_window_s is not None:
            period_s = getattr(args, "arrival_wave_period_s", 40.0)
            first_wave = list(jobs)
            jobs = []
            wave = 0
            while True:
                scheduled = []
                for job in first_wave:
                    if wave == 0:
                        planned = job["start_after_s"]
                    else:
                        pool_count = (source_count if job["pool"] == "source"
                                      else target_count)
                        index = int(job["job_id"].rsplit("_", 1)[1])
                        phase = (random.Random(
                            f"{seed}:{name}:{job['job_id']}:{wave}"
                        ).random() if getattr(args, "random_arrivals", False)
                            else 0.5)
                        planned = wave * period_s + (
                            index + phase) * period_s / pool_count
                    if planned < arrival_window_s:
                        scheduled.append((job, planned))
                if not scheduled:
                    break
                for job, planned in scheduled:
                    copy = dict(job)
                    if wave > 0 and getattr(args, "probability_pilot", False):
                        # Rotate distinct held-out content, never duplicate
                        # one tiny prompt set to claim independent coverage.
                        if rotation_cursor >= len(rows):
                            raise ValueError(
                                "held-out pool exhausted; shorten arrival window"
                            )
                        rotated = rows[rotation_cursor]
                        rotation_cursor += 1
                        copy["request"] = dict(job["request"])
                        copy["request"]["prompt"] = tokens(rotated, job["pool"])
                        pool_context = (
                            source_max_model_len
                            if job["pool"] == "source"
                            else target_max_model_len
                        )
                        copy["request"]["max_tokens"] = min(
                            job["request"]["max_tokens"],
                            pool_context - len(copy["request"]["prompt"]),
                        )
                        if copy["request"]["max_tokens"] <= 0:
                            raise ValueError("rotated request exceeds context")
                        copy["input_id"] = rotated["id"]
                        copy["source_tree_id"] = rotated.get(
                            "source_tree_id", rotated["id"]
                        )
                        copy["workload_group"] = rotated["workload_group"]
                        copy["input_augmentation"] = rotated.get("augmentation")
                        copy["workload_origin"] = rotated.get(
                            "workload_origin", "held_out_natural")
                        copy["controller_split"] = rotated.get(
                            "controller_split", "engineering_pilot_not_final_test")
                        copy["construction"] = rotated.get("construction")
                    copy["job_id"] = f"{job['job_id']}_wave{wave:03d}"
                    copy["start_after_s"] = round(planned, 3)
                    copy["wave"] = wave
                    jobs.append(copy)
                wave += 1
            if (not jobs or any(
                    max((job["start_after_s"] for job in jobs
                         if job["pool"] == pool), default=-1)
                    < arrival_window_s - period_s / count
                    for pool, count in (("source", source_count),
                                        ("target", target_count)) if count > 0)):
                raise ValueError("arrival schedule does not cover the window")
            if len(jobs) > 512:
                raise ValueError("arrival schedule exceeds 512 jobs per case")
        manifest = {
            "format_version": 1, "scenario": name,
            "status": "RANDOMIZED_GOODOUTPUT_PILOT",
            "seed": seed, "source_input": str(args.input.resolve()),
            "source_input_sha256": sha256(args.input),
            "anchor_input_id": anchor["id"],
            "source_count": sum(job["pool"] == "source" for job in jobs),
            "target_count": sum(job["pool"] == "target" for job in jobs),
            "source_spacing_s": source_spacing,
            "source_prompt_tokens": getattr(args, "source_prompt_tokens", None),
            "source_background_max_tokens": getattr(
                args, "source_background_max_tokens", None),
            "target_spacing_s": target_spacing,
            "natural_eos_required": True,
            "workload_origin": anchor.get("workload_origin", "held_out_natural"),
            "controller_split": anchor.get(
                "controller_split", "engineering_pilot_not_final_test"),
            "arrival_process": (
                "seeded_first_burst_stratified_replay"
                if arrival_window_s is not None and getattr(
                    args, "random_arrivals", False) else
                "fixed_first_burst_stratified_replay"
                if arrival_window_s is not None else
                "seeded_exponential"
                if getattr(args, "random_arrivals", False) else
                "fixed_spacing"),
            "arrival_window_s": arrival_window_s,
            "arrival_wave_period_s": (
                getattr(args, "arrival_wave_period_s", 40.0)
                if arrival_window_s is not None else None),
            "jobs": jobs,
        }
        if (getattr(args, "probability_pilot", False)
                and name == "p00_source1_target2"):
            jobs = [job for job in jobs if job["pool"] == "source"]
            manifest.update(jobs=jobs, target_count=0,
                            target_idle_override=True)
        if target_count == 0:
            manifest["target_idle_override"] = True
        if getattr(args, "constructed_workload", False):
            validate_constructed_slo_prompts(anchor_request, jobs, args.reference)
        for job in jobs:
            job["prompt_token_ids_sha256"] = hashlib.sha256(json.dumps(
                job["request"]["prompt"]).encode()).hexdigest()
        manifest["distinct_input_ids"] = len({job["input_id"] for job in jobs})
        manifest["distinct_source_trees"] = len({job["source_tree_id"] for job in jobs})
        manifest_path = root / "inputs" / f"{name}.json"
        write_json(manifest_path, manifest)
        setup["manifests"][name] = {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "jobs": len(jobs),
            "source_jobs": sum(job["pool"] == "source" for job in jobs),
            "target_jobs": sum(job["pool"] == "target" for job in jobs),
        }
        setup["cases"].append(name)
    if cursor != sum(row[1] + row[2] for row in cases) + len(CASES):
        raise ValueError("pilot selection was not consumed exactly once")
    return setup


def configure_action(command: list[str], action: str) -> None:
    probability_arm = action.startswith("prob_")
    if action not in ACTIONS + TIMING_ACTIONS and not probability_arm:
        raise ValueError(f"unknown pilot action: {action}")
    for flag in ("--manager-m2-force-initial-high",):
        command.remove(flag)
    for flag in ("--manager-m2-expected-profile",
                 "--manager-m2-min-history-byte-frac"):
        index = command.index(flag)
        del command[index:index + 2]
    if probability_arm:
        _, threshold, assigned = action.split("_")
        if assigned not in {"START", "STAY"} or not 0 <= float(threshold) <= 1:
            raise ValueError("invalid probability arm")
        if "--paired-stay" in command:
            command.remove("--paired-stay")
        command[command.index("--m1-min-output-tokens") + 1] = "0"
        command += ["--minimum-window-samples", "0",
                    "--probability-threshold", threshold,
                    "--probability-assigned-action", assigned]
        return
    threshold = 1024 if action == "late1024" else 128
    replacements = {
        "--m1-min-output-tokens": str(threshold),
        "--trigger-output-tokens": "1000" if threshold == 1024 else "64",
        "--bridge-output-tokens": "1024" if threshold == 1024 else "96",
    }
    for flag, value in replacements.items():
        command[command.index(flag) + 1] = value
    # Natural EOS and earliest-ready can leave no BRIDGE-window tokens.
    # Keep the measured windows but do not require a fixed sample count.
    command += ["--minimum-window-samples", "0"]
    command += ["--cutover-output-tokens",
                "1120" if threshold == 1024 else "320"]
    if action in {"now", "wait"}:
        command += ["--experiment-m1-action", action.upper()]


def action_order(name: str, seed: int = SEED,
                 actions: tuple[str, ...] = ACTIONS) -> list[str]:
    actions = list(actions)
    random.Random(f"{seed}:{name}").shuffle(actions)
    return actions


def observed_action(run: Path, *, paired_stay: bool = False) -> dict[str, Any]:
    audit = run / "controller" / "phase9_audit.jsonl"
    if not audit.is_file():
        return {"audit_available": False}
    rows = []
    for line in audit.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    starts = [row for row in rows
              if row.get("kind") == "manager_m1_start_decision"
              and row.get("decision", {}).get("action") == "START_SHADOW"]
    start_tokens = [row.get("snapshot", {}).get("generated_tokens")
                    for row in starts]
    decisions = [row for row in rows
                 if row.get("kind") == "manager_m1_start_decision"]
    timing = [row for row in rows
              if row.get("kind") == "experiment_m1_timing"]
    latest_m5: dict[str, Any] | None = None
    first_candidate: dict[str, Any] | None = None
    for row in rows:
        if row.get("kind") == "manager_m5_predictor_shadow":
            latest_m5 = row
        if row.get("kind") != "manager_m1_start_decision":
            continue
        snapshot = row.get("snapshot") or {}
        output_tokens = snapshot.get("generated_tokens")
        if not isinstance(output_tokens, int) or output_tokens < 128:
            continue
        decision = row.get("decision") or {}
        matching_m5 = (latest_m5 if latest_m5 is not None
                       and latest_m5.get("output_tokens") == output_tokens
                       else None)
        first_candidate = {
            "output_tokens": output_tokens,
            "target_running": snapshot.get("target_running"),
            "target_waiting": snapshot.get("target_waiting"),
            "target_kv_usage_frac": snapshot.get("target_kv_usage_frac"),
            "source_free_kv_tokens": snapshot.get("source_free_kv_tokens"),
            "source_guard_free_kv_tokens": snapshot.get(
                "source_guard_free_kv_tokens"),
            "source_prefill_pending_kv_tokens": snapshot.get(
                "source_prefill_pending_kv_tokens"),
            "source_time_to_guard_s": decision.get("source_time_to_guard_s"),
            "estimated_preparation_s": decision.get("estimated_preparation_s"),
            "m1_action": decision.get("action"),
            "m1_reason": decision.get("reason"),
            "m5_status": matching_m5.get("status") if matching_m5 else None,
            "p_remaining_gt_512_lower": (
                (matching_m5.get(
                    "p_remaining_gt_long_window_runtime_bounds") or [None])[0]
                if matching_m5 else None),
        }
        break
    acceptance = run / "provenance" / "shadow_online_acceptance.json"
    accepted = (json.loads(acceptance.read_text(encoding="utf-8"))
                if acceptance.is_file() else {})
    return {
        "audit_available": True,
        "start_count": 0 if paired_stay else len(starts),
        "actual_m1_start_output_tokens": [] if paired_stay else start_tokens,
        "m1_recommendation_output_tokens": start_tokens,
        "m1_decision_count": len(decisions),
        "timing_gate_reasons": [row.get("gate_reason") for row in timing
                                if row.get("gate_reason")],
        "timing_first_natural_start_output_tokens": next(
            (row.get("output_tokens") for row in timing
             if (row.get("natural_decision") or {}).get("action")
             == "START_SHADOW"), None),
        "first_candidate_at_or_after_128": first_candidate,
        "last_m1_decision": decisions[-1].get("decision")
        if decisions else None,
        "actual_cutover_output_tokens": accepted.get(
            "earliest_ready_cutover_output_tokens"),
        "handoff_stall_ms": accepted.get("handoff_stall_ms"),
        "acceptance_status": accepted.get("status"),
        "acceptance_errors": accepted.get("errors"),
        "outcome": accepted.get("outcome"),
    }


def is_context_censored(*, runner_rc: int, audit_rc: int,
                        source: dict[str, Any], anchor_cap: int,
                        acceptance_errors: list[str],
                        total_cap: int | None = None,
                        target: dict[str, Any] | None = None,
                        proxy: dict[str, Any] | None = None) -> bool:
    """Recognize a clean run stopped only by an output length cap."""
    source_capped = (
        source.get("finish_reason") == "length"
        and len(source.get("token_ids") or []) == anchor_cap
        and acceptance_errors == ["source did not naturally finish before its cap"]
    )
    target_capped = (
        (target or {}).get("finish_reason") == "length"
        and (proxy or {}).get("emitted_tokens") == (total_cap or anchor_cap)
        and acceptance_errors == [
            "unified response did not naturally finish before cap"]
    )
    return runner_rc != 0 and audit_rc == 0 and (source_capped or target_capped)


def cross_context_smoke_result(*, runner_rc: int, source: dict[str, Any],
                               target: dict[str, Any], proxy: dict[str, Any],
                               source_cap: int, total_cap: int) -> dict[str, Any]:
    """Check a forced-length continuation without treating it as natural EOS."""
    emitted = proxy.get("emitted") or []
    passed = (runner_rc == 0 and proxy.get("committed") is True
              and proxy.get("emitted_tokens") == total_cap
              and total_cap > source_cap
              and target.get("finish_reason") == "length"
              and [row.get("index") for row in emitted]
              == list(range(total_cap))
              and proxy.get("source_origin_tokens", 0) > 0
              and proxy.get("target_origin_tokens", 0) > 0)
    return {
        "status": "PASS" if passed else "FAIL",
        "functional_only": True,
        "runner_rc": runner_rc,
        "fatal_error": not passed,
        "source_output_cap": source_cap,
        "total_output_budget": total_cap,
        "emitted_tokens": proxy.get("emitted_tokens"),
        "source_finish_reason": source.get("finish_reason"),
        "target_finish_reason": target.get("finish_reason"),
        "source_origin_tokens": proxy.get("source_origin_tokens"),
        "target_origin_tokens": proxy.get("target_origin_tokens"),
    }


def probability_artifact_errors(run: Path) -> list[str]:
    """Reject missing or corrupt run evidence before computing any service label."""
    errors = []
    required = [
        "online/contract.json",
        "background/background_manifest.json",
        "background/background_summary.json",
        "controller/response_proxy_stats.json",
        "controller/source_response.json",
        "controller/phase9_audit.jsonl",
    ]
    if (run / "controller/target_response.json").is_file():
        required.append("controller/target_response.json")
    for relative in required:
        path = (
            run.parent / "contract.json"
            if relative == "online/contract.json" else run / relative
        )
        if not path.is_file():
            errors.append(f"missing run artifact: {relative}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
            rows = (
                [json.loads(line) for line in content.splitlines() if line.strip()]
                if path.suffix == ".jsonl" else [json.loads(content)]
            )
            if not rows or any(not isinstance(row, dict) for row in rows):
                raise ValueError("expected nonempty object records")
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"invalid run artifact: {relative}: {error}")
    return errors


def collect_arm(args: argparse.Namespace, root: Path,
                setup: dict[str, Any], name: str,
                action: str) -> dict[str, Any]:
    case_root = root / name
    arm_root = case_root / action
    command_setup = {
        "anchor_path": setup["anchors"][name]["path"],
        "anchor_sha256": setup["anchors"][name]["sha256"],
        "anchor_prompt_tokens": setup["anchors"][name]["prompt_tokens"],
        "manifests": setup["manifests"],
    }
    command = online_command(args, command_setup, name,
                             "stay" if action == "stay" else "migrate",
                             arm_root)
    command.extend(("--max-model-len", str(args.max_model_len)))
    if args.tp4_max_model_len is not None:
        command.extend(("--tp4-max-model-len",
                        str(args.tp4_max_model_len)))
    command[command.index("--anchor-max-tokens") + 1] = str(
        setup["anchors"][name]["max_tokens"])
    if (setup["anchors"][name]["total_max_tokens"]
            > setup["anchors"][name]["max_tokens"]):
        command.extend(("--anchor-total-max-tokens", str(
            setup["anchors"][name]["total_max_tokens"])))
    configure_action(command, action)
    if getattr(args, "probability_pilot", False):
        if getattr(args, "constructed_workload", False):
            command += ["--probability-min-source-running",
                        str(args.constructed_source_count + 1)]
        if (name == "p00_source1_target2"
                or setup["manifests"][name].get("target_jobs") == 0):
            command[command.index("--minimum-ready-target-jobs") + 1] = "0"
        command += ["--probability-assignment-seed", f"{args.seed}:{name}:{action}",
                    "--probability-threshold-family",
                    *(str(x) for x in args.probability_thresholds)]
    if args.risk_observation_shadow:
        command.append("--risk-observation-shadow")
    if args.cross_context_smoke:
        command.append("--cross-context-smoke")
    if getattr(args, "pre_episode_warmup", False):
        command.append("--pre-episode-warmup")
    minimum_headroom = getattr(args, "minimum_initial_source_headroom_tokens", None)
    if minimum_headroom is not None:
        command += ["--minimum-initial-source-headroom-tokens", str(minimum_headroom)]
    write_json(case_root / f"{action}.command.json", command)
    runner_rc = execute(command, case_root / f"{action}.console.log")
    run = arm_root / "r01_shadow_only"
    if args.cross_context_smoke:
        source_path = run / "controller" / "source_response.json"
        target_path = run / "controller" / "target_response.json"
        proxy_path = run / "controller" / "response_proxy_stats.json"
        source = json.loads(source_path.read_text()) if source_path.is_file() else {}
        target = json.loads(target_path.read_text()) if target_path.is_file() else {}
        proxy = json.loads(proxy_path.read_text()) if proxy_path.is_file() else {}
        source_cap = setup["anchors"][name]["max_tokens"]
        total_cap = setup["anchors"][name]["total_max_tokens"]
        result = cross_context_smoke_result(
            runner_rc=runner_rc, source=source, target=target, proxy=proxy,
            source_cap=source_cap, total_cap=total_cap)
        write_json(case_root / f"{action}.result.json", result)
        return result
    if getattr(args, "probability_pilot", False):
        artifact_errors = probability_artifact_errors(run)
        if artifact_errors:
            result = {
                "assigned_action": action.split("_")[2],
                "configured_probability_threshold": float(action.split("_")[1]),
                "configured_eligibility_tokens": None,
                "assignment_probability": 0.5,
                "episode_group_id": f"{args.seed}:{name}",
                "runner_rc": runner_rc,
                "audit_rc": None,
                "audit_skipped_reason": "MISSING_OR_CORRUPT_RUN_EVIDENCE",
                "observed_action": {
                    "audit_available": False,
                    "probability_episode_outcome": "TECHNICAL_FAILURE",
                },
                "fatal_error": True,
                "technical_errors": artifact_errors,
                "horizon_score": {"eligible": False, "errors": artifact_errors},
                "fixed_horizon_eligible": False,
                "fixed_horizon_goodoutput_tokens_s": None,
                "fixed_horizon_exclusions": {"technical_failure": True},
            }
            write_json(case_root / f"{action}.result.json", result)
            return result
    audit_path = case_root / f"{action}.slo_v6.json"
    audit_command = [
        sys.executable, "tools/bridge_tp/audit_slo_v6.py",
        "--run-root", str(run), "--reference", str(args.reference),
        "--preflight-json", str(root / "preflight.json"),
        "--require-reference-match", "--out-json", str(audit_path),
    ]
    if args.portable_hardware:
        audit_command += ["--gpu-match-mode", "model"]
    audit_rc = execute(audit_command,
                       case_root / f"{action}.slo_v6.console.log")
    slo = (json.loads(audit_path.read_text(encoding="utf-8"))
           if audit_path.is_file() else {})
    if (args.max_model_len != 8192
            or (args.tp4_max_model_len or args.max_model_len) != 8192):
        slo["reference_context_note"] = (
            "FROZEN_V6_TTFT_CONTRACT_REUSED_ACROSS_CONTEXT_LENGTHS"
        )
        write_json(audit_path, slo)
    background_path = run / "background" / "background_summary.json"
    background = (json.loads(background_path.read_text(encoding="utf-8"))
                  if background_path.is_file() else {})
    source_path = run / "controller" / "source_response.json"
    source = (json.loads(source_path.read_text(encoding="utf-8"))
              if source_path.is_file() else {})
    target_path = run / "controller" / "target_response.json"
    target = (json.loads(target_path.read_text(encoding="utf-8"))
              if target_path.is_file() else {})
    proxy_path = run / "controller" / "response_proxy_stats.json"
    proxy = (json.loads(proxy_path.read_text(encoding="utf-8"))
             if proxy_path.is_file() else {})
    finish_reasons = {
        reason: sum(row.get("finish_reason") == reason
                    for row in background.get("results", []))
        for reason in {row.get("finish_reason")
                       for row in background.get("results", [])}
    }
    observed = observed_action(run, paired_stay=action == "stay")
    if getattr(args, "probability_pilot", False):
        audit_rows = [
            json.loads(line)
            for line in (run / "controller" / "phase9_audit.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        gates = [
            r for r in audit_rows if r.get("kind") == "experiment_probability_gate"
        ]
        executions = [
            r for r in audit_rows if r.get("kind") == "experiment_probability_execution"
        ]
        observed["probability_candidate"] = next(
            (r for r in gates if r.get("first_feasible_candidate")), None)
        observed["actual_probability_executions"] = executions
        observed["probability_gate_ticks"] = len(gates)
        observed["probability_episode_outcome"] = (
            "STARTED_SHADOW" if executions else
            "ASSIGNED_STAY" if observed["probability_candidate"] else
            "NO_FEASIBLE_CROSSING_BEFORE_EOS" if source.get("finish_reason") == "stop"
            else "NO_START_CENSORED_OR_SERVICE_FAILURE"
        )
        observed["probability_crossings"] = [
            {"tick": r["tick"], "thresholds": r["first_crossings_this_tick"],
             "output_tokens": r["snapshot"]["generated_tokens"]}
            for r in gates if r.get("first_crossings_this_tick")]
        observed["safety_override_ticks"] = [r["tick"] for r in gates
                                             if r.get("safety_override")]
        observed["safety_protection_required_ticks"] = [
            r["tick"] for r in gates if r.get("safety_protection_required")
        ]
        observed["guard_deadline_warning_ticks"] = [
            r["tick"] for r in gates if r.get("guard_deadline_warning")
        ]
        observed["source_guard_policy"] = sorted({
            r.get("source_guard_policy", "LEGACY_GUARD_DEADLINE") for r in gates
        })
    acceptance_errors = observed.get("acceptance_errors") or []
    context_censored = is_context_censored(
        runner_rc=runner_rc, audit_rc=audit_rc, source=source,
        anchor_cap=setup["anchors"][name]["max_tokens"],
        total_cap=setup["anchors"][name]["total_max_tokens"],
        acceptance_errors=acceptance_errors, target=target, proxy=proxy,
    )
    natural_noop = (
        action != "stay" and audit_rc == 0
        and slo.get("computable") is True
        and observed.get("audit_available") is True
        and observed.get("start_count") == 0
        and source.get("finish_reason") == "stop"
    )
    result = {
        "assigned_action": action,
        "configured_eligibility_tokens": (
            1024 if action == "late1024" else 128),
        "runner_rc": runner_rc, "audit_rc": audit_rc,
        "observed_action": observed,
        "anchor_source_finish_reason": source.get("finish_reason"),
        "anchor_target_finish_reason": target.get("finish_reason"),
        "anchor_source_output_tokens": len(source.get("token_ids", [])),
        "natural_noop_needs_review": natural_noop,
        "context_censored": context_censored,
        "fatal_error": bool(audit_rc or (runner_rc and not natural_noop
                                          and not context_censored)),
        "slo_computable": slo.get("computable"),
        "slo_reference_applicability": slo.get("reference_applicability"),
        "slo_metrics": slo.get("metrics"),
        "slo_errors": slo.get("errors"),
        "background_jobs": background.get("jobs"),
        "background_completed": background.get("completed"),
        "background_finish_reasons": finish_reasons,
        "background_latest_planned_arrival_s": background.get(
            "latest_planned_arrival_s"),
        "background_max_schedule_lag_s": background.get(
            "max_schedule_lag_s"),
    }
    if getattr(args, "probability_pilot", False):
        horizon = score_horizon(slo, background, source, target, proxy,
                                args.evaluation_horizon_s,
                                settle_after_h=getattr(
                                    args, "window_token_goodoutput", False))
        # A recorded request failure is a service result. An unrecognized
        # runner rejection remains technical and is retained for diagnosis.
        service_errors = {
            "background jobs did not all complete",
            "source did not naturally finish before its cap",
            "unified response did not naturally finish before cap",
        }
        anchor_service_failure = (
            source.get("failure_kind") == "SERVICE_REQUEST_FAILURE"
            or target.get("failure_kind") == "SERVICE_REQUEST_FAILURE"
        )
        if (
            anchor_service_failure
            and not (run / "controller/session_manifest.json").exists()
        ):
            service_errors.update(
                {
                    "controller did not complete on TP1",
                    "source did not finish its full capped output",
                }
            )
        engineering_errors = [e for e in acceptance_errors if e not in service_errors]
        technical = (
            not horizon["eligible"]
            or audit_rc != 0
            or bool(engineering_errors)
            or bool(runner_rc and not acceptance_errors and not anchor_service_failure)
            or any(
                response.get("failure_kind") == "TECHNICAL_INPUT_FAILURE"
                for response in (source, target)
            )
        )
        result.update(
            {
                "configured_eligibility_tokens": None,
                "configured_probability_threshold": float(action.split("_")[1]),
                "assigned_action": action.split("_")[2],
                "assignment_probability": 0.5,
                "episode_group_id": f"{args.seed}:{name}",
                "horizon_score": horizon,
                "workload_origin": setup["anchors"][name].get("workload_origin"),
                "controller_split": setup["anchors"][name].get("controller_split"),
                "construction": setup["anchors"][name].get("construction"),
                "fatal_error": technical,
                "technical_errors": engineering_errors,
                "anchor_service_failure": anchor_service_failure,
                "fixed_horizon_eligible": horizon["eligible"] and not technical,
                "fixed_horizon_goodoutput_tokens_s": (
                    horizon["goodoutput_tokens_s"] if not technical else None
                ),
                "fixed_horizon_exclusions": {"technical_failure": technical},
            }
        )
    write_json(case_root / f"{action}.result.json", result)
    return result


def fixed_horizon_result(result: dict[str, Any], horizon_s: float) -> None:
    """Score a fully drained natural-EOS arm on one common horizon."""
    if "horizon_score" in result:
        return
    metrics = result.get("slo_metrics") or {}
    wall_s = metrics.get("wall_time_s")
    good_tokens = metrics.get("good_output_tokens")
    request_count = metrics.get("requests")
    background_count = result.get("background_jobs")
    completed = (isinstance(request_count, int) and request_count > 0
                 and request_count == metrics.get("completed_requests")
                 and isinstance(background_count, int)
                 and background_count == result.get("background_completed"))
    finish_reasons = result.get("background_finish_reasons")
    source_reason = result.get("anchor_source_finish_reason")
    target_reason = result.get("anchor_target_finish_reason")
    anchor_natural = (source_reason == "stop" if not target_reason
                      else source_reason == "abort" and target_reason == "stop")
    natural_eos = (anchor_natural
                   and isinstance(finish_reasons, dict)
                   and set(finish_reasons) <= {"stop"}
                   and sum(finish_reasons.values()) == background_count)
    arrival_lag_s = result.get("background_max_schedule_lag_s")
    allowed_lag_s = result.get("max_arrival_lag_s")
    arrivals_on_schedule = (allowed_lag_s is None or
                            isinstance(arrival_lag_s, (int, float))
                            and math.isfinite(arrival_lag_s)
                            and arrival_lag_s <= allowed_lag_s)
    eligible = (
        not result.get("fatal_error", True) and result.get("audit_rc") == 0
        and completed and natural_eos and arrivals_on_schedule
        and isinstance(wall_s, (int, float))
        and math.isfinite(wall_s) and wall_s <= horizon_s
        and isinstance(good_tokens, int) and good_tokens >= 0
    )
    result["fixed_horizon_eligible"] = eligible
    result["fixed_horizon_goodoutput_tokens_s"] = (
        good_tokens / horizon_s if eligible else None)
    result["fixed_horizon_exclusions"] = {
        "technical_or_slo_audit": bool(result.get("fatal_error", True)
                                       or result.get("audit_rc") != 0),
        "not_drained": not completed,
        "not_natural_eos": not natural_eos,
        "arrival_schedule_missed": not arrivals_on_schedule,
        "exceeds_horizon": not isinstance(wall_s, (int, float))
        or not math.isfinite(wall_s) or wall_s > horizon_s,
    }


def pair_results(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep descriptive paired deltas separate from fitted causal effects."""
    pairs = {}
    for name, arms in summary["cases"].items():
        stay = arms.get("stay", {})
        now = arms.get("early128", {})
        arm_eligible = bool(stay.get("fixed_horizon_eligible")
                            and now.get("fixed_horizon_eligible"))
        starts = now.get("observed_action", {}).get(
            "actual_m1_start_output_tokens", [])
        eligible = arm_eligible and bool(starts)
        pairs[name] = {
            "eligible": eligible,
            "assigned_now_actuated": bool(starts),
            "paired_arms_natural_eos": arm_eligible,
            "stay_predecision_state": stay.get("observed_action", {}).get(
                "first_candidate_at_or_after_128"),
            "now_predecision_state": now.get("observed_action", {}).get(
                "first_candidate_at_or_after_128"),
            "stay_goodoutput_tokens_s": stay.get(
                "fixed_horizon_goodoutput_tokens_s"),
            "now_goodoutput_tokens_s": now.get(
                "fixed_horizon_goodoutput_tokens_s"),
            "descriptive_delta_tokens_s": (
                now["fixed_horizon_goodoutput_tokens_s"]
                - stay["fixed_horizon_goodoutput_tokens_s"]
                if eligible else None),
            "stay_raw_output_tokens": (stay.get("slo_metrics") or {}).get(
                "raw_output_tokens"),
            "now_raw_output_tokens": (now.get("slo_metrics") or {}).get(
                "raw_output_tokens"),
        }
    return {
        "status": "EXPLORATORY_PAIRED_OUTCOMES_NOT_FITTED_EFFECTS",
        "seed": summary["seed"],
        "evaluation_horizon_s": summary["evaluation_horizon_s"],
        "pairs": pairs,
    }


def timing_results(summary: dict[str, Any]) -> dict[str, Any]:
    """Report observed timing and descriptive deltas, without fitting policy."""
    cases = {}
    for name, arms in summary["cases"].items():
        stay_rate = arms.get("stay", {}).get(
            "fixed_horizon_goodoutput_tokens_s")
        cases[name] = {}
        for action in TIMING_ACTIONS:
            arm = arms.get(action, {})
            observed = arm.get("observed_action") or {}
            rate = arm.get("fixed_horizon_goodoutput_tokens_s")
            cases[name][action] = {
                "eligible": arm.get("fixed_horizon_eligible", False),
                "context_censored": arm.get("context_censored", False),
                "slo_reference_applicability": arm.get(
                    "slo_reference_applicability"),
                "goodoutput_tokens_s": rate,
                "descriptive_delta_vs_stay_tokens_s": (
                    rate - stay_rate if rate is not None
                    and stay_rate is not None else None),
                "actual_start_output_tokens": observed.get(
                    "actual_m1_start_output_tokens", []),
                "first_natural_candidate_output_tokens": observed.get(
                    "timing_first_natural_start_output_tokens"),
                "timing_gate_reasons": observed.get("timing_gate_reasons", []),
                "handoff_stall_ms": observed.get("handoff_stall_ms"),
                "slo_attainment": (arm.get("slo_metrics") or {}).get(
                    "slo_attainment"),
                "exclusions": arm.get("fixed_horizon_exclusions"),
            }
    return {
        "status": "ENGINEERING_TIMING_PILOT_NOT_FITTED_BENEFIT",
        "seed": summary["seed"],
        "evaluation_horizon_s": summary["evaluation_horizon_s"],
        "cases": cases,
    }


def probability_results(summary: dict[str, Any]) -> dict[str, Any]:
    """One sample per intervention episode; repeated controls share a group."""
    samples = []
    for name, arms in summary["cases"].items():
        for theta in summary["probability_thresholds"]:
            start = arms.get(f"prob_{theta:g}_START", {})
            stay = arms.get(f"prob_{theta:g}_STAY", {})
            candidate = (start.get("observed_action") or {}).get(
                "probability_candidate"
            )
            control = (stay.get("observed_action") or {}).get("probability_candidate")
            start_g = start.get("fixed_horizon_goodoutput_tokens_s")
            stay_g = stay.get("fixed_horizon_goodoutput_tokens_s")
            executed = (start.get("observed_action") or {}).get(
                "actual_probability_executions", []
            )
            valid = (
                candidate is not None
                and control is not None
                and start_g is not None
                and stay_g is not None
                and bool(executed)
                and not candidate.get("safety_override")
                and not control.get("safety_override")
            )
            if any(
                ((arm.get("observed_action") or {}).get("safety_override_ticks")
                 or (arm.get("observed_action") or {}).get(
                     "safety_protection_required_ticks"))
                for arm in (start, stay)
            ):
                valid = False
            samples.append(
                {
                    "episode_group_id": f"{summary['seed']}:{name}",
                    "sample_id": (
                        f"{summary['collection_id']}:{name}:{theta:g}"
                        if summary.get("collection_id") else
                        f"{summary['seed']}:{name}:{theta:g}"
                    ),
                    "threshold": theta,
                    "goodoutput_scoring_policy": summary.get(
                        "goodoutput_scoring_policy", "COMPLETED_BY_H_LEGACY"),
                    "slo_profile": summary.get("slo_profile", "legacy1pct"),
                    "assignment_probability": 0.5,
                    "effect_sample_eligible": valid,
                    "policy_outcome_eligible": start_g is not None
                    and stay_g is not None,
                    "no_start_is_valid_policy_outcome": not executed,
                    "start_candidate": candidate,
                    "stay_candidate": control,
                    "descriptive_delta_tokens_s": (start_g - stay_g if valid else None),
                    "policy_delta_tokens_s": (
                        start_g - stay_g
                        if start_g is not None and stay_g is not None
                        else None
                    ),
                    "predecision_generated_token_difference": (
                        candidate["snapshot"]["generated_tokens"]
                        - control["snapshot"]["generated_tokens"]
                        if candidate and control
                        else None
                    ),
                    "predecision_H_difference": (
                        candidate["snapshot"]["H_tokens"]
                        - control["snapshot"]["H_tokens"]
                        if candidate and control
                        else None
                    ),
                    "start_raw_path": f"{name}/prob_{theta:g}_START",
                    "stay_raw_path": f"{name}/prob_{theta:g}_STAY",
                }
            )
    return {
        "status": "COLLECTOR_PILOT_NOT_FITTED_NET_BENEFIT",
        "samples": samples,
        "independent_grouping": "seed/request_tree/load_block",
    }


def select_probability_arms(
    thresholds: list[float], requested: list[str] | None = None,
) -> tuple[str, ...]:
    """Select engineering arms without changing the preregistered gate family."""
    available = tuple(
        f"prob_{theta:g}_{assignment}"
        for theta in thresholds for assignment in ("START", "STAY")
    )
    if requested is None:
        return available
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("probability arms must be nonempty and unique")
    if any(name not in available for name in requested):
        raise ValueError("probability arm is outside preregistered thresholds")
    return tuple(requested)

def execute_pilot(args: argparse.Namespace) -> None:
    if getattr(args, "constructed_workload", False):
        if (
            not args.probability_pilot
            or not args.expected_input_sha256
            or len(args.expected_input_sha256) != 64
            or any(c not in "0123456789abcdef" for c in args.expected_input_sha256)
            or args.source_prompt_tokens is not None
        ):
            raise ValueError(
                "constructed collection needs probability mode and pinned SHA")
        case_definitions(args)
    elif getattr(args, "expected_input_sha256", None) is not None:
        raise ValueError("input SHA override is restricted to constructed collection")
    if args.probability_pilot:
        if (
            args.paired_only
            or args.timing_pilot
            or args.cross_context_smoke
            or args.actions
            or args.source_prompt_tokens is not None
        ):
            raise ValueError(
                "probability pilot needs its own actions without legacy augmentation"
            )
        if (
            not args.probability_thresholds
            or len(set(args.probability_thresholds)) != len(args.probability_thresholds)
            or len({f"{x:g}" for x in args.probability_thresholds})
            != len(args.probability_thresholds)
            or any(
                not math.isfinite(x) or not 0 <= x <= 1
                for x in args.probability_thresholds
            )
        ):
            raise ValueError("pre-register unique finite thresholds in [0, 1]")
    if not 128 < args.max_model_len <= 32768:
        raise ValueError("source max model length must be in (128, 32768]")
    if (args.tp4_max_model_len is not None
            and not args.max_model_len <= args.tp4_max_model_len <= 32768):
        raise ValueError("TP4 max model length must cover TP1 and be <= 32768")
    if not math.isfinite(args.evaluation_horizon_s) or args.evaluation_horizon_s <= 0:
        raise ValueError("evaluation horizon must be positive and finite")
    if getattr(args, "window_token_goodoutput", False) and (
        not args.probability_pilot
        or args.arrival_window_s != args.evaluation_horizon_s
    ):
        raise ValueError("window token scoring requires probability mode and "
                         "arrival window equal to H")
    if args.arrival_window_s is not None:
        if (not math.isfinite(args.arrival_window_s)
                or not 0 < args.arrival_window_s <= args.evaluation_horizon_s):
            raise ValueError("arrival window must end at or before horizon")
        if (not math.isfinite(args.arrival_wave_period_s)
                or not 0 < args.arrival_wave_period_s
                < args.arrival_window_s):
            raise ValueError("arrival wave period must be shorter than window")
        if (not math.isfinite(args.max_arrival_lag_s)
                or args.max_arrival_lag_s < 0):
            raise ValueError("maximum arrival lag must be non-negative")
    if args.paired_only and args.timing_pilot:
        raise ValueError("choose paired-only or timing-pilot, not both")
    if args.cross_context_smoke and (
            args.paired_only or args.timing_pilot or args.actions
            or args.cases != ["p03_source3_target24"]
            or not args.anchor_context_limit
            or args.tp4_max_model_len is None
            or args.anchor_total_max_tokens is None
            or args.arrival_window_s is not None):
        raise ValueError("cross-context smoke requires only p03, a TP4 "
                         "total budget beyond the TP1 cap, and no arrival waves")
    if args.actions and not args.timing_pilot:
        raise ValueError("--actions requires --timing-pilot")
    if args.actions and len(args.actions) != len(set(args.actions)):
        raise ValueError("timing actions must be unique")
    if (not args.background_context_limit
            and not 1 <= args.background_max_tokens <= 4096):
        raise ValueError("background max tokens must be in [1, 4096]")
    if (args.source_prompt_tokens is not None
            and not 512 <= args.source_prompt_tokens <= 6144):
        raise ValueError("source prompt tokens must be in [512, 6144]")
    if (args.source_background_max_tokens is not None
            and not 1 <= args.source_background_max_tokens <= 4096):
        raise ValueError("source background max tokens must be in [1, 4096]")
    if args.anchor_total_max_tokens is not None and args.anchor_total_max_tokens <= 0:
        raise ValueError("anchor total max tokens must be positive")
    selected_cases = list(args.cases or (row[0] for row in case_definitions(args)))
    if len(selected_cases) != len(set(selected_cases)):
        raise ValueError("case names must be unique")
    actions = (tuple(f"prob_{theta:g}_{assignment}"
                     for theta in args.probability_thresholds
                     for assignment in ("START", "STAY")) if args.probability_pilot else
               ("now",) if args.cross_context_smoke else
               tuple(args.actions) if args.actions else
               TIMING_ACTIONS if args.timing_pilot else
               ("stay", "early128") if args.paired_only else ACTIONS)
    if args.probability_arms is not None:
        if not args.probability_pilot:
            raise ValueError("arm selection requires probability pilot")
        actions = select_probability_arms(
            args.probability_thresholds, args.probability_arms
        )
    preflight = verify(args)
    root = args.out_dir.resolve()
    protocol = {k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items() if k not in {"resume"}}
    setup_path = root / "pilot_setup.json"
    if args.resume:
        if not setup_path.is_file():
            raise ValueError("resume requires an existing pilot_setup.json")
        saved = json.loads((root / "preflight.json").read_text(
            encoding="utf-8"))
        if saved != preflight:
            raise ValueError("resume preflight differs from original run")
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        if args.probability_pilot and json.loads((root / "protocol.json").read_text(
                encoding="utf-8")) != protocol:
            raise ValueError("resume probability protocol differs")
    else:
        if root.exists():
            raise ValueError(f"pilot output directory exists: {root}")
        root.mkdir(parents=True)
        write_json(root / "preflight.json", preflight)
        write_json(root / "protocol.json", protocol)
        setup = build_setup(args, root)
        write_json(setup_path, setup)
    summary: dict[str, Any] = {
        "format_version": 1, "status": "PILOT_IN_PROGRESS",
        "seed": args.seed, "cases": {},
        "slo_profile": getattr(args, "slo_profile", "legacy1pct"),
        "collection_id": hashlib.sha256(json.dumps(
            {"protocol": protocol, "manifests": setup["manifests"]},
            sort_keys=True).encode()).hexdigest(),
        "probability_pilot": args.probability_pilot,
        "probability_thresholds": args.probability_thresholds,
        "actions": actions,
        "evaluation_horizon_s": args.evaluation_horizon_s,
        "goodoutput_scoring_policy": (
            "WINDOW_TOKENS_FULL_REQUEST_SLO_DRAIN_V1"
            if getattr(args, "window_token_goodoutput", False)
            else "COMPLETED_BY_H_LEGACY"
        ),
        "arrival_window_s": args.arrival_window_s,
        "arrival_wave_period_s": (args.arrival_wave_period_s
                                  if args.arrival_window_s else None),
        "max_arrival_lag_s": (args.max_arrival_lag_s
                              if args.arrival_window_s else None),
        "background_max_tokens": (None if args.background_context_limit
                                  else args.background_max_tokens),
        "background_context_limit": args.background_context_limit,
        "source_prompt_tokens": args.source_prompt_tokens,
        "source_background_max_tokens": args.source_background_max_tokens,
        "selected_cases": selected_cases,
        "anchor_context_limit": args.anchor_context_limit,
        "anchor_total_max_tokens": args.anchor_total_max_tokens,
        "p03_anchor_id": args.p03_anchor_id,
        "cross_context_smoke": args.cross_context_smoke,
        "source_max_model_len": args.max_model_len,
        "tp4_max_model_len": args.tp4_max_model_len or args.max_model_len,
    }
    summary_path = root / "pilot_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("seed") != args.seed
                or tuple(summary.get("actions", ())) != tuple(actions)
                or summary.get("evaluation_horizon_s")
                != args.evaluation_horizon_s
                or summary.get("arrival_window_s") != args.arrival_window_s
                or summary.get("arrival_wave_period_s") != (
                    args.arrival_wave_period_s if args.arrival_window_s
                    else None)
                or summary.get("max_arrival_lag_s") != (
                    args.max_arrival_lag_s if args.arrival_window_s
                    else None)
                or summary.get("background_max_tokens")
                != (None if args.background_context_limit
                    else args.background_max_tokens)
                or summary.get("background_context_limit")
                != args.background_context_limit
                or summary.get("source_prompt_tokens")
                != args.source_prompt_tokens
                or summary.get("source_background_max_tokens")
                != args.source_background_max_tokens
                or summary.get("selected_cases") != selected_cases
                or summary.get("anchor_context_limit")
                != args.anchor_context_limit
                or summary.get("anchor_total_max_tokens")
                != args.anchor_total_max_tokens
                or summary.get("p03_anchor_id") != args.p03_anchor_id
                or summary.get("cross_context_smoke")
                != args.cross_context_smoke
                or summary.get("source_max_model_len") != args.max_model_len
                or summary.get("tp4_max_model_len")
                != (args.tp4_max_model_len or args.max_model_len)):
            raise ValueError("resume options differ from original pilot")
    failed = False
    try:
        for name in selected_cases:
            manifest = setup["manifests"][name]
            if sha256(Path(manifest["path"])) != manifest["sha256"]:
                raise ValueError(f"manifest changed: {name}")
            anchor = setup["anchors"][name]
            if sha256(Path(anchor["path"])) != anchor["sha256"]:
                raise ValueError(f"anchor request changed: {name}")
            validation = [
                sys.executable,
                "tools/bridge_tp/run_phase9_capacity_background.py",
                "--manifest", manifest["path"], "--out-dir",
                str(root / name / "manifest-validation"), "--validate-only",
            ]
            if execute(validation, root / name / "manifest-validation.log"):
                failed = True
                break
            outcomes = summary["cases"].setdefault(name, {})
            for action in action_order(name, args.seed, actions):
                if action in outcomes:
                    prior = outcomes[action]
                    if prior.get("fatal_error", True):
                        raise ValueError(
                            f"previous failed arm must be diagnosed: {name}/{action}")
                    continue
                if (root / name / action).exists():
                    raise ValueError(
                        f"unrecorded arm directory must be inspected: "
                        f"{name}/{action}")
                print(f"=== {name} {action} ===", flush=True)
                result = collect_arm(args, root, setup, name, action)
                result["max_arrival_lag_s"] = (
                    args.max_arrival_lag_s if args.arrival_window_s else None)
                outcomes[action] = result
                if not args.cross_context_smoke:
                    fixed_horizon_result(result, args.evaluation_horizon_s)
                write_json(summary_path, summary)
                if result["fatal_error"]:
                    failed = True
                    break
            if failed:
                break
    finally:
        complete = not failed and all(
            len(summary["cases"].get(name, {})) == len(actions)
            for name in selected_cases)
        summary["status"] = (
            "INCOMPLETE_DIAGNOSTIC" if not complete else
            "CROSS_CONTEXT_SMOKE_PASS" if args.cross_context_smoke else
            "PILOT_COLLECTION_COMPLETE")
        write_json(summary_path, summary)
        if args.paired_only:
            write_json(root / "paired_outcomes.json", pair_results(summary))
        if args.timing_pilot:
            write_json(root / "timing_outcomes.json", timing_results(summary))
        if args.probability_pilot:
            write_json(root / "probability_outcomes.json", probability_results(summary))
        archive = root.with_suffix(".tar.gz")
        print(f"packing={root}", flush=True)
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(root, arcname=root.name)
        print(f"archive_to_retrieve={archive}", flush=True)
        print(f"pilot_status={summary['status']}", flush=True)
    if summary["status"] not in {"PILOT_COLLECTION_COMPLETE",
                                "CROSS_CONTEXT_SMOKE_PASS"}:
        raise RuntimeError("pilot incomplete; retrieve the diagnostic archive")


def main() -> None:
    execute_pilot(parse_args())


if __name__ == "__main__":
    main()
