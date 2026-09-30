"""Collect request-level labels and final-layer states for a length predictor.

Input is local JSONL with a unique ``id`` and either ``prompt`` or ``messages``
per row. Feature capture is opt-in and runs on a single GPU. This script does
not train a predictor or change any migration decision.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

_INTERNAL_ID_SUFFIX = re.compile(r"[0-9a-fA-F]{8}")


def label_for_engine_request(
    request_id: str, labels: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """Join worker's randomized ID to the externally returned request ID."""
    if request_id in labels:
        return labels[request_id]
    external_id, separator, suffix = request_id.rpartition("-")
    if separator and _INTERNAL_ID_SUFFIX.fullmatch(suffix):
        return labels.get(external_id)
    return None


def load_requests(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Read a local prompt corpus, retaining request IDs for split hygiene."""
    requests = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = row.get("id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(f"line {line_number}: nonempty id required")
            if request_id in seen:
                raise ValueError(f"line {line_number}: duplicate id {request_id}")
            if ("prompt" in row) == ("messages" in row):
                raise ValueError(
                    f"line {line_number}: provide exactly one of prompt/messages"
                )
            if "prompt" in row and not isinstance(row["prompt"], str):
                raise ValueError(f"line {line_number}: prompt must be text")
            if "messages" in row and not isinstance(row["messages"], list):
                raise ValueError(f"line {line_number}: messages must be a list")
            seen.add(request_id)
            requests.append(row)
            if limit is not None and len(requests) >= limit:
                break
    if not requests:
        raise ValueError("input contains no requests")
    return requests


def audit_capture(
    feature_dir: Path,
    labels: dict[str, dict[str, Any]],
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Check that every feature has one final response and a valid label."""
    sample_count = 0
    natural_samples = 0
    phases: dict[str, int] = {}
    seen: set[tuple[str, int]] = set()
    hidden_size = None
    index_handle = index_path.open("w", encoding="utf-8") if index_path else None
    try:
        for filename, row_number, state, request_id, count, stage in iter_feature_rows(
            feature_dir
        ):
            if hidden_size is None:
                hidden_size = len(state)
            elif hidden_size != len(state):
                raise ValueError("hidden width changed within one collection")
            if not np.isfinite(state).all():
                raise ValueError(f"nonfinite hidden state in {filename}")
            label = label_for_engine_request(request_id, labels)
            if label is None:
                raise ValueError(f"feature has no final response: {request_id}")
            if count > label["output_tokens"] or count < 0:
                raise ValueError(f"invalid generated count for {request_id}")
            if (request_id, count) in seen:
                raise ValueError(f"duplicate capture for {request_id} at {count}")
            seen.add((request_id, count))
            if stage not in ("PREFILL_COMPLETE", "DECODE"):
                raise ValueError(f"unknown capture stage {stage}")
            phases[stage] = phases.get(stage, 0) + 1
            sample_count += 1
            natural_samples += int(label["natural_finish"])
            observed_remaining = label["output_tokens"] - count
            index_row = {
                "request_id": request_id,
                "input_id": label.get("input_id"),
                "split": label.get("split"),
                "lang": label.get("lang"),
                "feature_file": filename,
                "feature_row": row_number,
                "phase": stage,
                "generated_tokens": count,
                "remaining_tokens": (
                    observed_remaining if label["natural_finish"] else None
                ),
                "observed_remaining_lower_bound": observed_remaining,
                "censored": not label["natural_finish"],
            }
            if index_handle is not None:
                index_handle.write(json.dumps(index_row, ensure_ascii=False) + "\n")
    finally:
        if index_handle is not None:
            index_handle.close()
    if not sample_count:
        raise ValueError("no hidden-state samples were captured")
    return {
        "format_version": 1,
        "requests": len(labels),
        "naturally_finished_requests": sum(
            row["natural_finish"] for row in labels.values()
        ),
        "censored_requests": sum(not row["natural_finish"] for row in labels.values()),
        "samples": sample_count,
        "samples_with_exact_remaining_length": natural_samples,
        "hidden_size": hidden_size,
        "phases": phases,
    }


def iter_feature_rows(feature_dir: Path):
    """Yield both the new SQLite format and legacy pilot NPZ features."""
    database_path = feature_dir / "features.sqlite3"
    if database_path.exists():
        with closing(sqlite3.connect(database_path)) as database:
            for sample_id, request_id, count, stage, width, blob in database.execute(
                """SELECT sample_id, request_id, generated_tokens, phase,
                          hidden_size, hidden_fp16 FROM samples ORDER BY sample_id"""
            ):
                if width <= 0 or len(blob) != 2 * width:
                    raise ValueError(f"invalid hidden-state bytes at row {sample_id}")
                yield (
                    database_path.name,
                    sample_id,
                    np.frombuffer(blob, dtype=np.float16),
                    request_id,
                    count,
                    stage,
                )
    for path in sorted(feature_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            states = data["hidden_states"]
            request_ids = data["request_ids"]
            generated = data["generated_tokens"]
            phase = data["phase"]
            if states.ndim != 2 or states.shape[0] != len(request_ids):
                raise ValueError(f"invalid state shape in {path}")
            if not (len(request_ids) == len(generated) == len(phase)):
                raise ValueError(f"metadata shape mismatch in {path}")
            for row_number, (state, request_id, count, stage) in enumerate(
                zip(states, request_ids, generated, phase)
            ):
                yield (
                    path.name,
                    row_number,
                    state,
                    str(request_id),
                    int(count),
                    str(stage),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.interval <= 0 or args.max_tokens <= 0 or args.max_model_len <= 0:
        parser.error("interval, max-tokens, and max-model-len must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between 0 and 1")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("out-dir must be new or empty")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != args.expected_revision:
        parser.error("HEAD differs from expected revision")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("worktree has uncommitted files")
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if digest != args.expected_input_sha256:
        parser.error("input SHA256 differs from expected digest")
    model_config = Path(args.model) / "config.json"
    if not model_config.is_file():
        parser.error(f"model config missing: {model_config}")
    gpu_names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    if len(gpu_names) != args.expected_gpu_count or any(
        name.strip() != args.expected_gpu_name for name in gpu_names
    ):
        parser.error(f"GPU inventory differs: {gpu_names}")
    requests = load_requests(args.input, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features = args.out_dir / "features"
    os.environ["BRIDGETP_PREDICTOR_CAPTURE_DIR"] = str(features.resolve())
    os.environ["BRIDGETP_PREDICTOR_CAPTURE_INTERVAL"] = str(args.interval)
    preflight = {
        "format_version": 1,
        "revision": revision,
        "input_path": str(args.input.resolve()),
        "input_sha256": digest,
        "model_path": str(Path(args.model).resolve()),
        "model_config_sha256": hashlib.sha256(model_config.read_bytes()).hexdigest(),
        "gpu_names": gpu_names,
        "interval": args.interval,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "requests": len(requests),
    }
    (args.out_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preflight, ensure_ascii=False), flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        async_scheduling=False,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        ignore_eos=False,
        temperature=args.temperature,
    )
    labels: dict[str, dict[str, Any]] = {}
    label_path = args.out_dir / "labels.jsonl"
    with label_path.open("w", encoding="utf-8") as handle:
        for request_number, row in enumerate(requests, 1):
            if request_number == 1 or request_number % 10 == 0:
                print(
                    f"capture request {request_number}/{len(requests)} id={row['id']}",
                    flush=True,
                )
            prompt = (
                row["prompt"]
                if "prompt" in row
                else tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=True
                )
            )
            prompt_tokens = len(tokenizer.encode(prompt))
            if prompt_tokens + args.max_tokens > args.max_model_len:
                raise ValueError(
                    f"request {row['id']} needs {prompt_tokens} prompt tokens "
                    f"plus {args.max_tokens} output tokens, exceeding "
                    f"max-model-len {args.max_model_len}"
                )
            result = llm.generate([prompt], sampling, use_tqdm=False)[0]
            completion = result.outputs[0]
            label = {
                "input_id": row["id"],
                "split": row.get("split"),
                "lang": row.get("lang"),
                "source": row.get("source"),
                "source_tree_id": row.get("source_tree_id"),
                "request_id": result.request_id,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_tokens": len(result.prompt_token_ids),
                "output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "natural_finish": completion.finish_reason == "stop",
                "max_tokens": args.max_tokens,
            }
            if result.request_id in labels:
                raise RuntimeError(f"reused engine request id: {result.request_id}")
            labels[result.request_id] = label
            handle.write(json.dumps(label, ensure_ascii=False) + "\n")
            handle.flush()
    summary = audit_capture(features, labels, args.out_dir / "sample_index.jsonl")
    if summary["phases"].get("PREFILL_COMPLETE") != len(labels):
        raise RuntimeError("not every request produced a prefill feature")
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
