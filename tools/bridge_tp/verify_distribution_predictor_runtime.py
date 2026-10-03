"""Replay a frozen runtime predictor against captured features and saved PMFs.

This is a read-only M5-A check. It does not start a model server or issue any
migration action. The capture and training artifacts must be audited first.
"""

import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch

from predictor_distribution import probability_gt_bounds

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "vllm/bridge_tp/controller/distribution_predictor.py"
SPEC = importlib.util.spec_from_file_location("distribution_predictor", RUNTIME)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DistributionPredictor = MODULE.DistributionPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_count <= 0:
        parser.error("sample-count must be positive")
    capture = args.capture_dir.resolve()
    training = args.training_dir.resolve()
    meta = json.loads((capture / "preflight.json").read_text(encoding="utf-8"))
    report = json.loads((training / "report.json").read_text(encoding="utf-8"))
    if report["capture_preflight"] != meta:
        raise ValueError("capture and training preflight differ")
    predictor = DistributionPredictor(
        training / "predictor_distribution.pt",
        checkpoint_sha256=report["checkpoint_sha256"],
        model_config_sha256=meta["model_config_sha256"],
        feature_layer=meta["feature_layer"],
    )
    if predictor.capture_input_sha256 != meta["input_sha256"]:
        raise ValueError("predictor was trained on another capture input")
    with np.load(training / "distribution_predictions.npz") as stored:
        rows = len(stored["request_ids"])
        positions = set(
            np.linspace(0, rows - 1, min(rows, args.sample_count), dtype=int)
        )
        selected = []
        first_groups = set()
        with (capture / "sample_index.jsonl").open(encoding="utf-8") as index:
            for position, line in enumerate(index):
                row = json.loads(line)
                group = (row["split"], row["phase"])
                if position in positions or group not in first_groups:
                    selected.append((position, row))
                    first_groups.add(group)
            if position + 1 != rows:
                raise ValueError("capture index and predictions have different rows")
        hidden = []
        generated = []
        with sqlite3.connect(capture / "features/features.sqlite3") as database:
            for position, row in selected:
                if (
                    stored["request_ids"][position] != row["input_id"]
                    or stored["generated_tokens"][position] != row["generated_tokens"]
                    or stored["phases"][position] != row["phase"]
                    or stored["splits"][position] != row["split"]
                ):
                    raise ValueError(f"prediction/index mismatch at {position}")
                record = database.execute(
                    """SELECT request_id, generated_tokens, phase, hidden_size,
                              hidden_fp16 FROM samples WHERE sample_id=?""",
                    (row["feature_row"],),
                ).fetchone()
                if record is None or record[:4] != (
                    row["request_id"], row["generated_tokens"], row["phase"], 5120
                ):
                    raise ValueError(f"feature/index mismatch at {position}")
                hidden.append(np.frombuffer(record[4], dtype=np.float16).copy())
                generated.append(row["generated_tokens"])
        predicted = []
        for start in range(0, len(selected), 128):
            with torch.no_grad():
                predicted.append(
                    predictor.probabilities(
                        torch.from_numpy(np.stack(hidden[start : start + 128])),
                        torch.tensor(generated[start : start + 128]),
                    ).cpu().numpy()
                )
        actual = np.concatenate(predicted)
        expected = stored["probabilities"][[position for position, _ in selected]]
        max_p_error = float(np.max(np.abs(actual - expected)))
        if max_p_error > 2e-5:
            raise ValueError(f"checkpoint/NPZ probability drift: {max_p_error}")
        edges = stored["category_upper_edges"]
        max_risk_error = 0.0
        for horizon in (0, 32, 128, 257, 512, 1024, 2048, 8192, 9000):
            low, high = predictor.probability_gt_bounds(
                torch.from_numpy(actual), float(horizon)
            )
            expected_low, expected_high = probability_gt_bounds(
                expected, edges, horizon
            )
            max_risk_error = max(
                max_risk_error,
                float(np.max(np.abs(low.numpy() - expected_low))),
                float(np.max(np.abs(high.numpy() - expected_high))),
            )
        if max_risk_error > 2e-5:
            raise ValueError(f"runtime risk bounds drift: {max_risk_error}")
        output = {
            "result": "PASS",
            "feature_layer": predictor.feature_layer,
            "checkpoint_sha256": predictor.checkpoint_sha256,
            "capture_input_sha256": predictor.capture_input_sha256,
            "samples_checked": len(selected),
            "total_samples": rows,
            "max_probability_abs_error": max_p_error,
            "max_risk_bound_abs_error": max_risk_error,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
