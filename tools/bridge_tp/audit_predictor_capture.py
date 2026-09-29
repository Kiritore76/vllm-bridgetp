"""Rebuild the sample index and summary from an existing capture directory."""

import argparse
import json
from pathlib import Path

from run_predictor_capture import audit_capture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    labels_path = run_dir / "labels.jsonl"
    features = run_dir / "features"
    if not labels_path.is_file() or not features.is_dir():
        parser.error("run-dir must contain labels.jsonl and features/")
    labels = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label = json.loads(line)
        request_id = label["request_id"]
        if request_id in labels:
            raise ValueError(f"duplicate response label: {request_id}")
        labels[request_id] = label
    summary = audit_capture(features, labels, run_dir / "sample_index.jsonl")
    if summary["phases"].get("PREFILL_COMPLETE") != len(labels):
        raise RuntimeError("not every response has a prefill feature")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
