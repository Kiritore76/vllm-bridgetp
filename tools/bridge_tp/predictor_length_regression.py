# SPDX-License-Identifier: Apache-2.0
"""Offline scalar remaining-token regression and matched point-estimate metrics."""

import math
from pathlib import Path

import numpy as np
from predictor_distribution import request_weights
from predictor_network import build_mlp
from predictor_progress import emit
from train_predictor_distribution import validation_groups

LIMIT = 8192
SCALE = 1024.0


def targets(data):
    """Keep censored lower bounds distinct from observed natural EOS targets."""
    remaining = np.asarray(data["remaining"], dtype=np.float32)
    if not np.isfinite(remaining).all() or (remaining < 0).any():
        raise ValueError("invalid remaining-token labels")
    bounded = np.minimum(remaining, LIMIT)
    exact = ~data["censored"] | (remaining >= LIMIT)
    return bounded, exact


def regression_loss(predicted, truth, exact):
    """Huber on exact lengths; one-sided Huber on censored lower bounds."""
    import torch
    from torch.nn import functional as f

    return torch.where(
        exact,
        f.smooth_l1_loss(predicted, truth, reduction="none", beta=1.0),
        f.smooth_l1_loss(
            torch.relu(truth - predicted),
            torch.zeros_like(predicted),
            reduction="none",
            beta=1.0,
        ),
    )


def normalization(data):
    """Estimate moments on training states only, without a full float32 copy."""
    indices = np.flatnonzero(data["splits"] == "train")
    if not len(indices):
        raise ValueError("empty training split")
    total = np.zeros(data["hidden"].shape[1], dtype=np.float64)
    squares = total.copy()
    for start in range(0, len(indices), 512):
        block = data["hidden"][indices[start : start + 512]].astype(np.float64)
        if not np.isfinite(block).all():
            raise ValueError("nonfinite training features")
        total += block.sum(0)
        squares += (block * block).sum(0)
    mean = total / len(indices)
    std = np.sqrt(np.maximum(0, squares / len(indices) - mean * mean)).clip(1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


def features(data, indices, checkpoint, device):
    import torch

    hidden = torch.from_numpy(data["hidden"][indices].astype(np.float32)).to(device)
    generated = torch.from_numpy(data["generated"][indices].astype(np.float32)).to(
        device
    )
    mean = checkpoint["feature_mean"].to(device)
    std = checkpoint["feature_std"].to(device)
    position = torch.log1p(generated) / checkpoint["position_log_scale"]
    return torch.cat(((hidden - mean) / std, position[:, None]), dim=1)


def predict(data, checkpoint, model, device, batch_size=256):
    import torch

    model.eval()
    result = np.empty(len(data["remaining"]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(result), batch_size):
            indices = np.arange(start, min(start + batch_size, len(result)))
            result[indices] = (
                model(features(data, indices, checkpoint, device))
                .squeeze(1)
                .cpu()
                .numpy()
                * SCALE
            )
    if not np.isfinite(result).all():
        raise ValueError("nonfinite regression predictions")
    return result


def categorical_points(probabilities, edges):
    """Compare both clipped bucket mean and median; retain open-tail mass."""
    p = np.asarray(probabilities)
    edges = np.asarray(edges)
    if (
        p.ndim != 2
        or p.shape[1] != len(edges) + 1
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(1), 1, atol=1e-5)
        or edges[0] != 0
        or edges[-1] != LIMIT
        or not (np.diff(edges) > 0).all()
    ):
        raise ValueError("invalid bounded categorical distribution")
    midpoints = np.r_[(np.r_[0, edges[:-1] + 1] + edges) / 2, LIMIT]
    median = np.minimum((p.cumsum(1) < 0.5).sum(1), len(edges))
    return p @ midpoints, midpoints[median]


def accuracy(truth, predicted, ids):
    if not len(truth):
        return None
    w = request_weights(ids)
    error = predicted - truth
    absolute = np.abs(error)
    return dict(
        samples=len(truth),
        requests=len(set(ids)),
        request_balanced_mae_tokens=float(w @ absolute),
        request_balanced_rmse_tokens=float(np.sqrt(w @ (error**2))),
        request_balanced_bias_tokens=float(w @ error),
        sample_p90_absolute_error_tokens=float(np.quantile(absolute, 0.9)),
        within_tokens={
            str(n): float(w @ (absolute <= n)) for n in (128, 256, 512, 1024)
        },
        underestimate_over512_fraction=float(w @ (error < -512)),
        overestimate_over512_fraction=float(w @ (error > 512)),
    )


def selection_masks(data, cohorts):
    selected = np.zeros(len(data["remaining"]), dtype=bool)
    calibration = selected.copy()
    groups = {}
    for name, ids in cohorts.items():
        a, b = validation_groups(data, ids)
        if ((a | b) & (selected | calibration)).any():
            raise ValueError("validation cohorts overlap")
        groups[name] = a
        selected |= a
        calibration |= b
    if (selected & calibration).any() or (
        selected & (data["splits"] != "validation")
    ).any():
        raise ValueError("selection leakage")
    return groups, selected, calibration


def fit(
    data,
    meta,
    out_dir: Path,
    weights,
    cohorts,
    *,
    revision,
    device="cuda:0",
    epochs=30,
    patience=6,
    batch_size=256,
    width=256,
    seed=42,
):
    """Train four affine layers, choose by held-out request-balanced MAE."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = data["splits"] == "train"
    bounded, exact = targets(data)
    groups, selection, calibration = selection_masks(data, cohorts)
    if (
        weights.shape != train.shape
        or not np.isfinite(weights).all()
        or (weights[~train] != 0).any()
        or (weights[train] <= 0).any()
        or not np.isclose(weights.sum(), 1)
        or epochs < 1
        or patience < 1
    ):
        raise ValueError("invalid train-only weights or epoch budget")
    mean, std = normalization(data)
    checkpoint = dict(
        format_version=1,
        model_type="remaining_length_regression_mlp4",
        input_width=len(mean) + 1,
        hidden_width=width,
        network_depth=4,
        hidden_widths=[width, 2 * width, width],
        dropout=0.1,
        feature_mean=torch.from_numpy(mean),
        feature_std=torch.from_numpy(std),
        position_log_scale=math.log1p(meta["max_tokens"]),
        output_scale_tokens=SCALE,
        output_clip_tokens=[0, LIMIT],
        target="min(natural remaining output tokens, 8192)",
        model_config_sha256=meta["model_config_sha256"],
        feature_layer=meta["feature_layer"],
        feature_semantics=meta["feature_semantics"],
        training_revision=revision,
        capture_revision=meta["revision"],
        capture_input_sha256=meta["input_sha256"],
        seed=seed,
        initialization="from scratch; no classifier weights inherited",
        loss="Huber beta=1024 tokens; one-sided Huber for censored lower bounds",
    )
    model = build_mlp(len(mean) + 1, width, 1, 0.1, 4).to(device)
    with torch.no_grad():
        model[-1].bias.fill_(float(weights @ bounded / SCALE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    indices = np.flatnonzero(train)
    out_dir.mkdir(parents=True, exist_ok=False)
    path = out_dir / "predictor_regression.pt"
    history, best, best_epoch = [], math.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(indices)
        total_loss = 0.0
        for start in range(0, len(shuffled), batch_size):
            rows = shuffled[start : start + batch_size]
            prediction = model(features(data, rows, checkpoint, device)).squeeze(1)
            truth = torch.from_numpy(bounded[rows] / SCALE).to(device)
            observed = torch.from_numpy(exact[rows]).to(device)
            importance = torch.from_numpy(
                (weights[rows] * train.sum()).astype(np.float32)
            ).to(device)
            loss = (regression_loss(prediction, truth, observed) * importance).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
        model.eval()
        predictions = np.zeros(len(bounded), dtype=np.float32)
        with torch.no_grad():
            select_indices = np.flatnonzero(selection)
            for start in range(0, len(select_indices), batch_size):
                rows = select_indices[start : start + batch_size]
                predictions[rows] = (
                    model(features(data, rows, checkpoint, device))
                    .squeeze(1)
                    .cpu()
                    .numpy()
                    * SCALE
                )
        predictions = predictions.clip(0, LIMIT)
        losses = [
            accuracy(bounded[m], predictions[m], data["requests"][m])[
                "request_balanced_mae_tokens"
            ]
            for m in groups.values()
        ]
        score = float(np.mean(losses))
        if not math.isfinite(score):
            raise ValueError("nonfinite validation score")
        history.append(
            dict(
                epoch=epoch,
                training_loss=total_loss / len(indices),
                selection_mae_tokens=score,
            )
        )
        emit(
            f"四层长度回归 {epoch}/{epochs}轮：验证MAE={score:.2f} token，"
            f"最佳={min(best, score):.2f}"
        )
        if score < best:
            best, best_epoch = score, epoch
            checkpoint["state_dict"] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            checkpoint["best_epoch"] = epoch
            torch.save(checkpoint, path)
        if epoch - best_epoch >= patience:
            break
    saved = torch.load(path, map_location="cpu", weights_only=True)
    reloaded = build_mlp(saved["input_width"], width, 1, saved["dropout"], 4).to(device)
    reloaded.load_state_dict(saved["state_dict"], strict=True)
    prediction = predict(data, saved, reloaded, device, batch_size)
    return prediction, dict(
        best_epoch=best_epoch,
        selection_mae_tokens=best,
        history=history,
        architecture=f"{len(mean) + 1}->{width}->{2 * width}->{width}->1",
        parameter_count=sum(p.numel() for p in reloaded.parameters()),
        selected_requests=sorted(set(data["requests"][selection])),
        reserved_calibration_requests=sorted(set(data["requests"][calibration])),
        optimizer="AdamW lr=0.0003 weight_decay=0.01",
        dropout=0.1,
        normalization="training states only",
        initialization=saved["initialization"],
        loss=saved["loss"],
        raw_outside_bounds_fraction=float(
            ((prediction < 0) | (prediction > LIMIT)).mean()
        ),
    )
