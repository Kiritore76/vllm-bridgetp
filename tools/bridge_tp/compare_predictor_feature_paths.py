"""Compare paired eager auxiliary and offline-hook features at equal tokens."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_predictor_capture import iter_feature_rows, label_for_engine_request


def read_labels(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def compare(live: Path, offline: Path):
    a, b = read_labels(live / "labels.jsonl"), read_labels(offline / "labels.jsonl")
    by_input = {r["input_id"]: r for r in a}
    if set(by_input) != {r["input_id"] for r in b}:
        raise ValueError("paired inputs differ")
    for row in b:
        other = by_input[row["input_id"]]
        if (
            row["prompt_sha256"] != other["prompt_sha256"]
            or row["output_token_ids"] != other["output_token_ids"]
        ):
            raise ValueError(
                "paired token trajectories differ; cannot compare features"
            )
    online_labels = {r["request_id"]: r for r in a}
    offline_labels = {r["request_id"]: r for r in b}
    features = {}
    for file in sorted((live / "probes").glob("probe-*.pt")):
        saved = torch.load(file, map_location="cpu", weights_only=True)
        for rid, count, hidden in zip(
            saved["request_ids"], saved["generated_tokens"], saved["hidden_fp16"]
        ):
            label = label_for_engine_request(rid, online_labels)
            if label is None:
                raise ValueError("unmatched live request")
            key = (label["input_id"], int(count))
            if key in features:
                raise ValueError("duplicate live token position")
            features[key] = hidden.numpy().astype(np.float32)
    errors, matched = [], set()
    for _, _, state, rid, count, _ in iter_feature_rows(offline / "features"):
        label = label_for_engine_request(rid, offline_labels)
        key = (label["input_id"], count)
        if key not in features:
            continue
        difference = float(np.max(np.abs(state.astype(np.float32) - features[key])))
        errors.append(difference)
        matched.add(key)
    if not errors or matched != set(features):
        raise ValueError("unmatched or empty live features")
    maximum = max(errors)
    result = {
        "status": "PASS" if maximum <= 1e-3 else "FEATURE_PATH_MISMATCH",
        "samples": len(errors),
        "maximum_hidden_absolute_difference": maximum,
        "tolerance": 1e-3,
        "scope": "Paired TP1 eager runs, equal full output token trajectories; not CUDA graph or loaded-system validation",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.live, args.offline)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if result["status"] != "PASS":
        raise RuntimeError(
            "feature paths differ; inspect before collecting training data"
        )


if __name__ == "__main__":
    main()
