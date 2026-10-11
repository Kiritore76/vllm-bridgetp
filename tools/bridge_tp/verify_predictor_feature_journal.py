#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay exact saved online features, including candidate-feature associations."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


def verify(directory, checkpoint, model_config_sha, audit=None):
    """Check hashes, probability agreement and all referenced candidate states."""
    runtime_path = Path(__file__).resolve().parents[2] / (
        "vllm/bridge_tp/controller/distribution_predictor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "predictor_journal_runtime", runtime_path
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    predictor = runtime.DistributionPredictor(
        checkpoint,
        checkpoint_sha256=checkpoint_sha,
        model_config_sha256=model_config_sha,
    )
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    files = sorted(directory.glob("*.pt"))
    if (
        receipt["dropped_batches"]
        or receipt["writer_error"]
        or not files
        or len(files) != receipt["saved_batches"]
    ):
        raise ValueError("feature journal incomplete")
    states = set()
    maximum = 0.0
    for path in files:
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"].get(
            path.name
        ):
            raise ValueError("feature SHA differs")
        batch = torch.load(path, map_location="cpu", weights_only=True)
        if batch["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("checkpoint identity differs")
        if (
            batch.get("model_config_sha256") != model_config_sha
            or batch.get("feature_layer") != predictor.feature_layer
        ):
            raise ValueError("feature layer or base model identity differs")
        if not torch.equal(
            batch["category_upper_edges"], predictor.category_upper_edges
        ):
            raise ValueError("bucket geometry differs")
        actual = predictor.probabilities(
            batch["hidden_fp16"], batch["generated_tokens"]
        )
        error = float((actual - batch["probabilities"]).abs().max())
        maximum = max(maximum, error)
        if not torch.isfinite(actual).all() or error > 1e-5:
            raise ValueError("online/offline predictor probabilities differ")
        for request, position in zip(
            batch["request_ids"], batch["generated_tokens"].tolist()
        ):
            states.add((path.stem, request, int(position)))
    references = 0
    if audit:
        for line in audit.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kind") != "experiment_probability_gate" or not row.get(
                "first_feasible_candidate"
            ):
                continue
            prediction = row["snapshot"]["prediction"]
            key = (
                prediction.get("feature_id"),
                prediction.get("request_id"),
                prediction.get("prediction_output_tokens"),
            )
            if key not in states:
                raise ValueError("candidate cannot be associated with a saved feature")
            references += 1
    return {
        "saved_states": len(states),
        "saved_batches": len(files),
        "candidate_references": references,
        "max_probability_absolute_error": maximum,
        "checkpoint_sha256": checkpoint_sha,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = verify(
        args.directory, args.checkpoint, args.model_config_sha256, args.audit
    )
    (args.directory / "replay_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
