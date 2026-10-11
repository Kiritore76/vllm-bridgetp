"""Capture four pinned public tasks with natural EOS and completion diagnostics."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from predictor_progress import emit
from predictor_targeted_quality import evaluate
from run_predictor_tail_pilot import capture_command, serialized

MAX_MODEL_LEN = 24576
GROUPS = (("longproc", 11000), ("longgenbench", 18432))


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def recipe():
    """Check pinned prompts and separately stored references before collection."""
    directory = Path(__file__).parent
    rows = json.loads(
        (directory / "predictor_targeted4_requests_20261011.json").read_text("utf-8")
    )
    refs = json.loads(
        (directory / "predictor_targeted4_references_20261011.json").read_text("utf-8")
    )
    if len(rows) != 4 or len({row["source_tree_id"] for row in rows}) != 4:
        raise ValueError("four distinct public sources are required")
    if [row["task_kind"] for row in rows] != [
        "html_to_tsv",
        "path_traversal",
        "diary",
        "menu",
    ]:
        raise ValueError("task order differs from preregistered recipe")
    for index, row in enumerate(rows):
        if (
            row["split"] != "pilot"
            or row["prompt_adapted"]
            or [m["role"] for m in row["messages"]] != ["user"]
            or hashlib.sha256(row["messages"][0]["content"].encode()).hexdigest()
            != row["source_prompt_sha256"]
            or digest(refs[row["id"]]) != row["quality_spec_sha256"]
            or row["max_output_tokens"] != GROUPS[index // 2][1]
        ):
            raise ValueError("public source or diagnostic provenance differs")
    return rows, refs


def prepare(directory, tokenizer, config):
    """Validate all four rendered prompts before creating input files."""
    if config.get("max_position_embeddings", 0) < MAX_MODEL_LEN:
        raise ValueError("model configuration cannot support pilot context length")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("existing pilot data must be preserved")
    rows, _ = recipe()
    budgets = []
    for row in rows:
        # Match run_predictor_capture's string rendering and encode exactly.
        # len(apply_chat_template(tokenize=True)) can count BatchEncoding keys.
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True
        )
        count = len(tokenizer.encode(prompt))
        if count + row["max_output_tokens"] + 16 > MAX_MODEL_LEN:
            raise ValueError(f"prompt/output budget exceeded: {row['id']}")
        budgets.append(
            {
                "id": row["id"],
                "prompt_tokens": count,
                "max_output_tokens": row["max_output_tokens"],
            }
        )
    directory.mkdir(parents=True, exist_ok=True)
    payload = serialized(rows)
    (directory / "input_requests.jsonl").write_bytes(payload)
    groups = []
    for index, (name, cap) in enumerate(GROUPS):
        group_dir = directory / name
        group_dir.mkdir()
        group_payload = serialized(rows[2 * index : 2 * index + 2])
        (group_dir / "input_requests.jsonl").write_bytes(group_payload)
        manifest = {
            "group": name,
            "requests": 2,
            "input_sha256": hashlib.sha256(group_payload).hexdigest(),
            "max_model_len": MAX_MODEL_LEN,
            "max_output_tokens": cap,
            "prompt_budgets": budgets[2 * index : 2 * index + 2],
            "ignore_eos": False,
            "min_tokens": 0,
            "training_started": False,
        }
        (group_dir / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        groups.append(manifest)
    manifest = {
        "recipe_version": "targeted4-public-v1-20261011",
        "requests": 4,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "groups": groups,
        "prompt_budgets": budgets,
        "max_model_len": MAX_MODEL_LEN,
        "feature_layer": "decoder:31",
        "capture_interval": 20,
        "training_started": False,
        "formal_holdout_eligible": False,
        "ignore_eos": False,
        "min_tokens": 0,
        "reference_outputs_are_not_local_eos_labels": True,
    }
    (directory / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def command(args, manifest):
    """Reuse the unchanged collector, replacing only this group's output cap."""
    group_args = SimpleNamespace(**vars(args))
    group_args.out_dir = args.out_dir / manifest["group"]
    result = capture_command(group_args, manifest)
    result[result.index("--max-tokens") + 1] = str(manifest["max_output_tokens"])
    return result


def summarize(directory, tokenizer, codes):
    """Retain partial observations and score only the actual saved outputs."""
    rows, refs = recipe()
    inputs = {row["id"]: row for row in rows}
    responses, quality, seen = [], [], set()
    complete = True
    for name, _ in GROUPS:
        capture = directory / name / "capture"
        labels = capture / "labels.jsonl"
        complete &= (
            codes.get(name) == 0
            and labels.is_file()
            and (capture / "summary.json").is_file()
            and (capture / "sample_index.jsonl").is_file()
        )
        if not labels.is_file():
            continue
        with labels.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                label = json.loads(line)
                identity = label["input_id"]
                if identity not in inputs or identity in seen:
                    raise ValueError("unknown or duplicate saved response")
                if inputs[identity]["source"].lower() != name:
                    raise ValueError("saved response belongs to another group")
                ids = label.get("output_token_ids")
                if ids is None or len(ids) != label["output_tokens"]:
                    raise ValueError("saved response lacks complete token IDs")
                natural = label["finish_reason"] == "stop"
                if natural != label["natural_finish"]:
                    raise ValueError("natural-EOS label disagrees with termination")
                seen.add(identity)
                text = tokenizer.decode(ids, skip_special_tokens=True)
                responses.append({**label, "output_text": text})
                quality.append(
                    {
                        "id": identity,
                        "task_kind": inputs[identity]["task_kind"],
                        "output_tokens": len(ids),
                        "natural_finish": natural,
                        **evaluate(
                            inputs[identity]["task_kind"], text, refs[identity], ids
                        ),
                    }
                )
    (directory / "responses.jsonl").write_bytes(serialized(responses))
    (directory / "quality_summary.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    complete &= seen == set(inputs)
    result = {
        "status": "COLLECTION_COMPLETE" if complete else "INCOMPLETE_DIAGNOSTIC",
        "planned_requests": 4,
        "completed_requests": len(seen),
        "group_returncodes": codes,
        "natural_gt8192": sum(
            r["natural_finish"] and r["output_tokens"] > 8192 for r in responses
        ),
        "natural_gt12288": sum(
            r["natural_finish"] and r["output_tokens"] > 12288 for r in responses
        ),
        "censored_requests": sum(not r["natural_finish"] for r in responses),
        "training_started": False,
        "formal_holdout_eligible": False,
        "next_action": "inspect entry completion and repetition before expanding",
    }
    (directory / "pilot_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    emit(
        f"定向验证完成{len(seen)}/4条；自然>8192有{result['natural_gt8192']}条；"
        "条目完成度与重复诊断已保存，本轮未训练"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    rows, _ = recipe()
    if hashlib.sha256(serialized(rows)).hexdigest() != args.expected_recipe_sha256:
        parser.error("preregistered recipe SHA differs")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != args.expected_revision:
        parser.error("HEAD differs from expected revision")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("worktree has uncommitted files")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((args.model / "config.json").read_text("utf-8"))
    manifest = prepare(args.out_dir, tokenizer, config)
    emit("4条实际tokenizer预算通过；先LongProc 2条，再LongGenBench 2条")
    if args.prepare_only:
        return
    codes = {}
    started = time.monotonic()
    for index, group in enumerate(manifest["groups"]):
        emit(f"总进度：已完成{index * 2}/4条；启动{group['group']}，本组2条")
        code = subprocess.call(command(args, group))
        codes[group["group"]] = code
        if code:
            break
    result = summarize(args.out_dir, tokenizer, codes)
    result["collection_wall_time_s"] = time.monotonic() - started
    (args.out_dir / "pilot_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if result["status"] != "COLLECTION_COMPLETE":
        raise RuntimeError("pilot incomplete; retrieve the diagnostic archive")


if __name__ == "__main__":
    main()
