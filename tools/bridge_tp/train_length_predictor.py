"""Train an offline two-layer remaining-length predictor from captured states.

Training uses naturally finished requests only. Entire request trees stay in
their preassigned train, validation, or test split. No migration policy is
changed by this script.
"""

import argparse
import hashlib
import json
import math
import sqlite3
import subprocess
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

import numpy as np
from run_predictor_capture import audit_capture


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_examples(run_dir: Path, include_censored: bool = False) -> dict:
    """Verify SQLite samples and preserve censoring for distribution training."""
    labels = read_jsonl(run_dir / "labels.jsonl")
    index = read_jsonl(run_dir / "sample_index.jsonl")
    label_by_id = {}
    tree_splits = {}
    for label in labels:
        request_id = label["request_id"]
        if request_id in label_by_id:
            raise ValueError(f"duplicate response label: {request_id}")
        split = label.get("split")
        tree_id = label.get("source_tree_id")
        if split not in ("train", "validation", "test") or not tree_id:
            raise ValueError("every formal request needs a split and tree ID")
        if tree_id in tree_splits and tree_splits[tree_id] != split:
            raise ValueError(f"tree split leakage: {tree_id}")
        tree_splits[tree_id] = split
        label_by_id[request_id] = label
    audit = audit_capture(run_dir / "features", label_by_id)
    existing = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if audit != existing:
        raise ValueError("capture summary differs from independent audit")
    if len(index) != audit["samples"]:
        raise ValueError("sample index does not cover every captured feature")
    database_path = run_dir / "features" / "features.sqlite3"
    if not database_path.is_file():
        raise ValueError("formal training requires SQLite features")

    states = []
    remaining = []
    generated = []
    splits = []
    requests = []
    phases = []
    languages = []
    censored = []
    seen_rows = set()
    with closing(sqlite3.connect(database_path)) as database:
        for row in index:
            if row["feature_file"] != database_path.name:
                raise ValueError("sample index references another feature format")
            sample_id = row["feature_row"]
            if sample_id in seen_rows:
                raise ValueError(f"duplicate feature row: {sample_id}")
            seen_rows.add(sample_id)
            label = label_by_id.get(row["request_id"])
            if label is None and "-" in row["request_id"]:
                label = label_by_id.get(row["request_id"].rsplit("-", 1)[0])
            if label is None:
                raise ValueError(f"missing response label: {row['request_id']}")
            if row.get("split") != label["split"]:
                raise ValueError("sample split differs from response label")
            if row["censored"] == label["natural_finish"]:
                raise ValueError("sample censoring differs from response label")
            if row["censored"] and not include_censored:
                continue
            result = database.execute(
                """SELECT request_id, generated_tokens, phase, hidden_size,
                          hidden_fp16 FROM samples WHERE sample_id=?""",
                (sample_id,),
            ).fetchone()
            if result is None:
                raise ValueError(f"missing feature row: {sample_id}")
            request_id, count, phase, width, blob = result
            if (
                request_id != row["request_id"]
                or count != row["generated_tokens"]
                or phase != row["phase"]
                or len(blob) != width * 2
            ):
                raise ValueError(f"feature/index mismatch at row {sample_id}")
            observed_remaining = label["output_tokens"] - count
            if not row["censored"] and row["remaining_tokens"] != observed_remaining:
                raise ValueError(f"remaining length mismatch at row {sample_id}")
            if row["censored"] and (
                row["remaining_tokens"] is not None
                or row["observed_remaining_lower_bound"] != observed_remaining
            ):
                raise ValueError(f"censored length mismatch at row {sample_id}")
            states.append(np.frombuffer(blob, dtype=np.float16).copy())
            remaining.append(observed_remaining)
            generated.append(count)
            splits.append(label["split"])
            requests.append(label["input_id"])
            phases.append(phase)
            languages.append(label.get("lang", "unknown"))
            censored.append(row["censored"])
    if not states:
        raise ValueError("no exact-length samples")
    hidden = np.stack(states)
    if not np.isfinite(hidden).all():
        raise ValueError("nonfinite hidden state")
    return {
        "hidden": hidden,
        "remaining": np.asarray(remaining, dtype=np.float32),
        "generated": np.asarray(generated, dtype=np.float32),
        "splits": np.asarray(splits),
        "requests": np.asarray(requests),
        "phases": np.asarray(phases),
        "languages": np.asarray(languages),
        "censored": np.asarray(censored, dtype=bool),
        "audit": audit,
        "labels": labels,
    }


def request_balanced_mae(
    truth: np.ndarray, predicted: np.ndarray, request_ids: np.ndarray
) -> float:
    """Give each response one vote despite its number of decode samples."""
    errors = defaultdict(list)
    for actual, estimate, request_id in zip(truth, predicted, request_ids):
        errors[str(request_id)].append(abs(float(actual) - float(estimate)))
    return float(np.mean([np.mean(values) for values in errors.values()]))


def metrics(truth: np.ndarray, predicted: np.ndarray, request_ids: np.ndarray) -> dict:
    errors = np.abs(truth - predicted)
    return {
        "samples": int(len(truth)),
        "requests": int(len(set(request_ids))),
        "mae_tokens": float(np.mean(errors)),
        "request_balanced_mae_tokens": request_balanced_mae(
            truth, predicted, request_ids
        ),
        "median_absolute_error_tokens": float(np.median(errors)),
        "p90_absolute_error_tokens": float(np.quantile(errors, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-capture-revision", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_width, args.patience) <= 0:
        parser.error("epochs, batch-size, hidden-width, patience must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("out-dir must be new or empty")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != args.expected_revision:
        parser.error("HEAD differs from expected revision")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("worktree has uncommitted files")
    run_dir = args.run_dir.resolve()
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    if (
        preflight["revision"] != args.expected_capture_revision
        or preflight["input_sha256"] != args.expected_input_sha256
    ):
        parser.error("capture revision or input SHA256 differs")
    if preflight["model_config_sha256"] != sha256_file(
        Path(preflight["model_path"]) / "config.json"
    ):
        parser.error("model config differs from capture")
    gpu_names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    if len(gpu_names) != args.expected_gpu_count or any(
        name.strip() != args.expected_gpu_name for name in gpu_names
    ):
        parser.error(f"GPU inventory differs: {gpu_names}")
    data = load_examples(run_dir)
    if data["audit"]["censored_requests"] / data["audit"]["requests"] > 0.2:
        parser.error("more than 20% of requests were capped; inspect capture")
    masks = {
        split: data["splits"] == split for split in ("train", "validation", "test")
    }
    request_counts = {
        split: len(set(data["requests"][mask])) for split, mask in masks.items()
    }
    if (
        request_counts["train"] < 100
        or request_counts["validation"] < 20
        or request_counts["test"] < 20
    ):
        parser.error(f"too few complete requests for formal training: {request_counts}")

    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable for formal training")
    device = torch.device("cuda:0")
    hidden = data["hidden"].astype(np.float32)
    train_hidden = hidden[masks["train"]]
    feature_mean = train_hidden.mean(axis=0)
    feature_std = train_hidden.std(axis=0).clip(min=1e-4)
    del train_hidden
    hidden = (hidden - feature_mean) / feature_std
    position = np.log1p(data["generated"]) / math.log1p(preflight["max_tokens"])
    features = np.column_stack((hidden, position)).astype(np.float32)
    del hidden
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(np.log1p(data["remaining"])).to(device)
    max_log = math.log1p(preflight["max_tokens"])
    model = nn.Sequential(
        nn.Linear(features.shape[1], args.hidden_width),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(args.hidden_width, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    train_indices = np.flatnonzero(masks["train"])
    per_request = Counter(data["requests"][masks["train"]])
    weight = torch.from_numpy(
        np.asarray(
            [
                1.0 / per_request[request_id] if request_id in per_request else 0.0
                for request_id in data["requests"]
            ],
            dtype=np.float32,
        )
    ).to(device)

    def predict(mask: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            output = model(x[torch.from_numpy(np.flatnonzero(mask)).to(device)])
            output = output.squeeze(-1).clamp(min=0, max=max_log)
            return torch.expm1(output).cpu().numpy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_mae = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = np.random.permutation(train_indices)
        losses = []
        for start in range(0, len(shuffled), args.batch_size):
            batch = torch.from_numpy(shuffled[start : start + args.batch_size]).to(
                device
            )
            predicted = model(x[batch]).squeeze(-1)
            individual = torch.nn.functional.smooth_l1_loss(
                predicted, y[batch], reduction="none"
            )
            loss = (individual * weight[batch]).sum() / weight[batch].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = metrics(
            data["remaining"][masks["validation"]],
            predict(masks["validation"]),
            data["requests"][masks["validation"]],
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation": validation,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        current = validation["request_balanced_mae_tokens"]
        if current < best_mae:
            best_mae = current
            best_epoch = epoch
            torch.save(
                {
                    "format_version": 1,
                    "state_dict": model.state_dict(),
                    "input_width": features.shape[1],
                    "hidden_width": args.hidden_width,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "max_tokens": preflight["max_tokens"],
                    "capture_revision": preflight["revision"],
                    "capture_input_sha256": preflight["input_sha256"],
                },
                args.out_dir / "predictor.pt",
            )
        if epoch - best_epoch >= args.patience:
            break

    checkpoint = torch.load(
        args.out_dir / "predictor.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    completed_train = [
        label["output_tokens"]
        for label in data["labels"]
        if label["split"] == "train" and label["natural_finish"]
    ]
    baseline_total = float(np.median(completed_train))
    results = {}
    for split, mask in masks.items():
        truth = data["remaining"][mask]
        request_ids = data["requests"][mask]
        baseline = np.maximum(0, baseline_total - data["generated"][mask])
        results[split] = {
            "model": metrics(truth, predict(mask), request_ids),
            "median_total_baseline": metrics(truth, baseline, request_ids),
        }
    report = {
        "format_version": 1,
        "training_revision": revision,
        "capture_revision": preflight["revision"],
        "capture_input_sha256": preflight["input_sha256"],
        "capture_audit": data["audit"],
        "exact_request_counts": request_counts,
        "censored_requests": data["audit"]["censored_requests"],
        "device": str(device),
        "gpu_names": gpu_names,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "baseline_train_median_total_tokens": baseline_total,
        "results": results,
        "history": history,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"best_epoch": best_epoch, "results": results}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
