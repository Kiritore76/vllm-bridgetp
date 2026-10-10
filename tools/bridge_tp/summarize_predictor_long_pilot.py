"""Report actual long-request coverage, separating EOS from output caps."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path):
    labels = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]
    natural = [r["output_tokens"] for r in labels if r["natural_finish"]]
    all_lengths = [r["output_tokens"] for r in labels]
    edges = [0, 2048, 4096, 8192, 12288, 16384]
    histogram = {}
    for low, high in zip(edges, edges[1:]):
        histogram[f"{low + 1}..{high}"] = sum(low < x <= high for x in natural)
    return {
        "requests": len(labels),
        "natural_finish": len(natural),
        "censored_requests": len(labels) - len(natural),
        "natural_length_histogram": histogram,
        "natural_length_quantiles": np.quantile(
            natural, [0, 0.25, 0.5, 0.75, 1]
        ).tolist()
        if natural
        else None,
        "all_output_tokens": sum(all_lengths),
        "per_request": [
            {
                k: r[k]
                for k in [
                    "input_id",
                    "prompt_tokens",
                    "output_tokens",
                    "finish_reason",
                    "natural_finish",
                ]
            }
            for r in labels
        ],
        "status": "DIAGNOSTIC_COMPLETE_REVIEW_BEFORE_FORMAL_CAPTURE",
        "formal_training_started": False,
        "note": "Requested lessons do not determine labels. Pilot is not representative natural-risk calibration.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.labels)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
