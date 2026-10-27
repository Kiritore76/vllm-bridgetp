"""Train and calibrate a remaining-length probability distribution.

Reuses an existing capture; does not load the LLM or actuate the controller.
Exact responses train category likelihood. Capped responses train a compatible
tail likelihood with their lower bound rounded down to its containing category.
"""

import argparse
import hashlib
import json
import math
import socket
import subprocess
from pathlib import Path

import numpy as np
from predictor_distribution import (
    aggregate_probabilities,
    binary_risk_metrics,
    category_targets,
    default_upper_edges,
    distribution_nll,
    km_conditional_survival,
    nonuniform16k91_upper_edges,
    nonuniform24k100_upper_edges,
    nonuniform84_upper_edges,
    probability_remaining_gt,
    request_weights,
    risk76_upper_edges,
    softmax,
)
from predictor_network import build_mlp, hidden_widths
from predictor_progress import EpochProgress, emit
from train_length_predictor import load_examples, read_jsonl, sha256_file


def finite_support_examples(data: dict, maximum: int = 16384) -> tuple[dict, dict]:
    """Exclude whole censored/out-of-domain requests; preserve the capture audit."""
    excluded = {
        row["input_id"]: (
            "censored" if not row["natural_finish"] else "output_exceeds_support"
        )
        for row in data["labels"]
        if not row["natural_finish"] or row["output_tokens"] > maximum
    }
    keep = ~np.isin(data["requests"], list(excluded))
    if data["censored"][keep].any() or (data["remaining"][keep] > maximum).any():
        raise ValueError("finite support labels disagree with captured states")
    filtered = {
        key: value[keep]
        if isinstance(value, np.ndarray) and value.ndim and len(value) == len(keep)
        else value
        for key, value in data.items()
    }
    filtered["labels"] = [
        row for row in data["labels"] if row["input_id"] not in excluded
    ]
    return filtered, {
        "maximum_tokens": maximum,
        "retained_requests": len(set(filtered["requests"])),
        "excluded_requests": excluded,
        "excluded_samples": int((~keep).sum()),
        "rule": "whole requests; natural EOS only; never relabel a cap as EOS",
    }


def validation_groups(
    data: dict, allowed_ids: set[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Separate model selection and calibration at the request-tree level."""
    validation = {
        label["input_id"]: label["source_tree_id"]
        for label in data["labels"]
        if label["split"] == "validation"
        and label["natural_finish"]
        and (allowed_ids is None or label["input_id"] in allowed_ids)
    }
    ranked = sorted(
        set(validation.values()),
        key=lambda key: hashlib.sha256(
            ("probability-validation-v1:" + key).encode()
        ).digest(),
    )
    if len(ranked) < 2:
        raise ValueError("two independent validation requests are required")
    half = len(ranked) // 2
    selected = [key for key, tree in validation.items() if tree in ranked[:half]]
    calibrated = [key for key, tree in validation.items() if tree in ranked[half:]]
    return (np.isin(data["requests"], selected), np.isin(data["requests"], calibrated))


def fit_distribution(
    data: dict,
    preflight: dict,
    out_dir: Path,
    *,
    revision: str,
    device_name: str = "cuda:0",
    epochs: int = 30,
    batch_size: int = 256,
    hidden_width: int = 256,
    learning_rate: float = 0.001,
    dropout: float = 0.1,
    weight_decay: float = 0.01,
    patience: int = 6,
    seed: int = 42,
    bin_step: int = 32,
    known_test_ids: set[str] | None = None,
    warmstart_checkpoint: Path | None = None,
    warmstart_sha256: str | None = None,
    tail_edges: list[int] | None = None,
    bucket_profile: str = "parent",
    head_warmup_epochs: int = 0,
    head_learning_rate: float = 0.0003,
    training_weights: np.ndarray | None = None,
    weighting_provenance: dict | None = None,
    validation_ids: set[str] | None = None,
    diagnostic_groups: dict[str, np.ndarray] | None = None,
    network_depth: int = 2,
    balanced_validation_ids: dict[str, set[str]] | None = None,
    reference_checkpoint: Path | None = None,
    reference_sha256: str | None = None,
) -> dict:
    """Fit on train, select on one validation half, calibrate on the other."""
    import torch
    from predictor_warmstart import configure_training_phase

    if (
        not math.isfinite(dropout)
        or not 0 <= dropout < 1
        or not math.isfinite(weight_decay)
        or weight_decay < 0
        or bucket_profile
        not in (
            "parent",
            "nonuniform84",
            "risk76",
            "nonuniform24k100",
            "nonuniform16k91",
        )
        or not 0 <= head_warmup_epochs < epochs
        or not math.isfinite(head_learning_rate)
        or head_learning_rate <= 0
        or network_depth not in (2, 4)
    ):
        raise ValueError("dropout must be in [0, 1); weight decay must be nonnegative")
    overflow_category = bucket_profile != "nonuniform16k91"
    support_selection = None
    if not overflow_category:
        if warmstart_checkpoint is not None or reference_checkpoint is not None:
            raise ValueError("finite-support classifiers must train from scratch")
        if training_weights is not None or diagnostic_groups is not None:
            raise ValueError(
                "finite-support filtering requires unweighted raw examples"
            )
        data, support_selection = finite_support_examples(data)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device_name)
    train = data["splits"] == "train"
    selection, calibration = validation_groups(data, validation_ids)
    validation_cohorts = {}
    if balanced_validation_ids is not None:
        if len(balanced_validation_ids) != 2:
            raise ValueError("balanced validation requires exactly two cohorts")
        selection = np.zeros(len(train), dtype=bool)
        calibration = selection.copy()
        for name, allowed in balanced_validation_ids.items():
            selected, calibrated = validation_groups(data, allowed)
            if ((selection | calibration) & (selected | calibrated)).any():
                raise ValueError("validation cohorts overlap")
            validation_cohorts[name] = (selected, calibrated)
            selection |= selected
            calibration |= calibrated
        if (selection & calibration).any():
            raise ValueError("validation selection/calibration overlap")

    def validation_nll(probabilities, mask):
        if not validation_cohorts:
            return distribution_nll(
                probabilities,
                targets[mask],
                data["censored"][mask],
                data["requests"][mask],
            )
        losses = []
        for selected, calibrated in validation_cohorts.values():
            subset = (selected | calibrated)[mask]
            losses.append(
                distribution_nll(
                    probabilities[subset],
                    targets[mask][subset],
                    data["censored"][mask][subset],
                    data["requests"][mask][subset],
                )
            )
        return float(np.mean(losses))

    test = (data["splits"] == "test") & ~data["censored"]
    parent = None
    original_parent = None
    parent_temperature = None
    parent_categories = None
    if (
        bucket_profile in ("risk76", "nonuniform24k100")
        and warmstart_checkpoint is not None
    ):
        raise ValueError(f"{bucket_profile} classifiers must train from scratch")
    if warmstart_checkpoint:
        from predictor_warmstart import expand_tail, load_warmstart, rebuild_output_head

        parent = load_warmstart(
            warmstart_checkpoint, warmstart_sha256, preflight, data["hidden"].shape[1]
        )
        if hidden_width != parent["hidden_width"]:
            raise ValueError("warmstart hidden width differs")
        parent_temperature = parent["temperature"]
        parent_categories = len(parent["category_upper_edges"]) + 1
        original_parent = parent
        if bucket_profile == "nonuniform84":
            if tail_edges or head_warmup_epochs < 1:
                raise ValueError("84 categories need head warmup, without tail-edges")
            parent = rebuild_output_head(parent, nonuniform84_upper_edges(), seed)
        else:
            parent = expand_tail(parent, tail_edges or [])
    elif tail_edges:
        raise ValueError("tail extension requires warmstart")
    elif head_warmup_epochs:
        raise ValueError("head warmup requires warmstart")
    if network_depth == 4 and parent is not None:
        raise ValueError("four-layer trial trains from scratch")
    if reference_checkpoint is not None:
        if original_parent is not None:
            raise ValueError("explicit reference cannot replace a warmstart parent")
        from predictor_warmstart import load_warmstart

        original_parent = load_warmstart(
            reference_checkpoint, reference_sha256, preflight, data["hidden"].shape[1]
        )
    edges = (
        parent["category_upper_edges"].numpy()
        if parent
        else nonuniform84_upper_edges()
        if bucket_profile == "nonuniform84"
        else risk76_upper_edges()
        if bucket_profile == "risk76"
        else nonuniform24k100_upper_edges()
        if bucket_profile == "nonuniform24k100"
        else nonuniform16k91_upper_edges()
        if bucket_profile == "nonuniform16k91"
        else default_upper_edges(preflight["max_tokens"], bin_step)
    )
    categories = len(edges) + int(overflow_category)
    targets = category_targets(data["remaining"], edges)
    weight = np.zeros(len(targets), dtype=np.float64)
    if training_weights is None:
        weight[train] = request_weights(data["requests"][train])
    else:
        weight = np.asarray(training_weights, dtype=np.float64)
        if (
            weight.shape != targets.shape
            or not np.isfinite(weight).all()
            or (weight[train] <= 0).any()
            or (weight[~train] != 0).any()
            or not np.isclose(weight.sum(), 1)
        ):
            raise ValueError("invalid train-only normalized sample weights")
    hidden = data["hidden"].astype(np.float32)
    feature_mean = (
        parent["feature_mean"].numpy() if parent else hidden[train].mean(axis=0)
    )
    feature_std = (
        parent["feature_std"].numpy()
        if parent
        else hidden[train].std(axis=0).clip(min=1e-4)
    )
    position_scale = (
        parent["position_log_scale"] if parent else math.log1p(preflight["max_tokens"])
    )
    features = np.column_stack(
        (
            (hidden - feature_mean) / feature_std,
            np.log1p(data["generated"]) / position_scale,
        )
    ).astype(np.float32)
    del hidden
    x = torch.from_numpy(features).to(device)
    target = torch.from_numpy(targets).to(device)
    censored = torch.from_numpy(data["censored"]).to(device)
    model = build_mlp(
        features.shape[1], hidden_width, categories, dropout, network_depth
    ).to(device)
    exact_train = train & ~data["censored"]
    prior = (
        np.bincount(
            targets[exact_train],
            weights=weight[exact_train] / weight[exact_train].sum(),
            minlength=categories,
        )
        + 0.001
    )
    prior /= prior.sum()
    with torch.no_grad():
        model[-1].bias.copy_(torch.from_numpy(np.log(prior)).to(device))
    if parent:
        model.load_state_dict(parent["state_dict"], strict=True)
        if bucket_profile == "nonuniform84":
            # Train-only priors; parent temperature cannot calibrate a new head.
            with torch.no_grad():
                model[-1].bias.copy_(torch.from_numpy(np.log(prior)).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_indices = np.flatnonzero(train)
    weights = torch.from_numpy((weight * train.sum()).astype(np.float32)).to(device)

    def logits_for(mask, evaluated_model=None):
        evaluated_model = model if evaluated_model is None else evaluated_model
        evaluated_model.eval()
        indices = np.flatnonzero(mask)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch = torch.from_numpy(indices[start : start + batch_size]).to(device)
                outputs.append(evaluated_model(x[batch]).cpu().numpy())
        return np.concatenate(outputs)

    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_nll = float("inf")
    best_epoch = 0
    checkpoint_path = out_dir / "predictor_distribution.pt"
    training_requests = len(set(data["requests"][train]))
    for epoch in range(1, epochs + 1):
        head_only = epoch <= head_warmup_epochs
        configure_training_phase(
            model,
            optimizer,
            head_only=head_only,
            head_learning_rate=head_learning_rate,
            learning_rate=learning_rate,
        )
        emit(
            f"训练阶段：{'冻结第一层，训练新增层' if head_only else '整个小预测器'}，"
            f"学习率{optimizer.param_groups[0]['lr']:g}"
        )
        model.train()
        shuffled = rng.permutation(train_indices)
        epoch_loss = 0.0
        progress = EpochProgress(
            epoch, epochs, len(shuffled), batch_size, training_requests
        )
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            batch = torch.from_numpy(indices).to(device)
            log_p = torch.log_softmax(model(x[batch]), dim=1)
            log_tail = torch.logcumsumexp(log_p.flip(1), dim=1).flip(1)
            exact_log = log_p.gather(1, target[batch, None]).squeeze(1)
            censored_log = log_tail.gather(1, target[batch, None]).squeeze(1)
            loss = (
                -torch.where(censored[batch], censored_log, exact_log) * weights[batch]
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(indices)
            progress.update(start // batch_size + 1, start + len(indices))
        emit(f"续训 第{epoch}/{epochs}轮：训练完成，正在验证")
        validation_loss = validation_nll(softmax(logits_for(selection)), selection)
        history.append(
            {
                "epoch": epoch,
                "train_nll": epoch_loss / len(shuffled),
                "selection_validation_nll": validation_loss,
                "phase": "head_only" if head_only else "joint",
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(json.dumps(history[-1]), flush=True)
        emit(
            f"续训 第{epoch}/{epochs}轮：训练NLL={history[-1]['train_nll']:.4f}，"
            f"验证NLL={validation_loss:.4f}"
        )
        if validation_loss < best_nll:
            best_nll, best_epoch = validation_loss, epoch
            torch.save(
                {
                    "warmstart_provenance": {
                        "parent_sha256": warmstart_sha256,
                        "tail_edges": tail_edges or [],
                        "feature_normalization": "parent"
                        if parent
                        else "new train split",
                        "optimizer": "new AdamW, not optimizer resume",
                        "parent_temperature": parent_temperature,
                        "parent_categories": parent_categories,
                        "network_depth": network_depth,
                        "initialization": "warmstart" if parent else "from scratch",
                        "bucket_profile": bucket_profile,
                        "head_warmup_epochs": head_warmup_epochs,
                        "head_initialization": "seeded weights, train-only prior bias"
                        if bucket_profile in ("nonuniform84", "risk76") or not parent
                        else "parent head",
                        "tail_initialization": "not used for new classifier"
                        if bucket_profile
                        in ("nonuniform84", "risk76", "nonuniform24k100")
                        else "equal split preserving parent calibrated sums",
                    },
                    "format_version": 2,
                    "model_type": "remaining_length_categorical_mlp4"
                    if network_depth == 4
                    else "remaining_length_categorical",
                    "state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "input_width": features.shape[1],
                    "hidden_width": hidden_width,
                    "network_depth": network_depth,
                    "hidden_widths": hidden_widths(hidden_width, network_depth),
                    "activation": "GELU",
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                    "feature_mean": torch.from_numpy(feature_mean),
                    "feature_std": torch.from_numpy(feature_std),
                    "position_log_scale": position_scale,
                    "category_upper_edges": torch.from_numpy(edges),
                    "overflow_category": overflow_category,
                    "support_scope": "unbounded_with_open_tail"
                    if overflow_category
                    else "natural_eos_within_finite_support",
                    "temperature": 1.0,
                    "training_revision": revision,
                    "capture_revision": preflight["revision"],
                    "capture_input_sha256": preflight["input_sha256"],
                    "capture_max_model_len": preflight.get("max_model_len"),
                    "bucket_profile": bucket_profile,
                    "model_config_sha256": preflight["model_config_sha256"],
                    "generation_contract": preflight.get("generation_contract"),
                    "feature_layer": preflight.get("feature_layer", "final"),
                    "feature_semantics": preflight.get(
                        "feature_semantics", "last token after final model norm"
                    ),
                    "query_rule": "P(N>H) upper bound; open tail is not forced to zero"
                    if overflow_category
                    else "conditional on natural EOS within finite support",
                },
                checkpoint_path,
            )
        if epoch > head_warmup_epochs and epoch - best_epoch >= patience:
            break

    emit(f"训练结束，最佳轮次{best_epoch}，正在校准温度并评估")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    calibration_logits = logits_for(calibration)
    candidates = sorted(set(np.linspace(0.5, 3.0, 51).tolist() + [1.0]))
    calibration_curve = [
        {
            "temperature": temperature,
            "nll": validation_nll(
                softmax(calibration_logits, temperature), calibration
            ),
        }
        for temperature in candidates
    ]
    temperature = min(calibration_curve, key=lambda row: row["nll"])["temperature"]
    checkpoint["temperature"] = temperature
    torch.save(checkpoint, checkpoint_path)
    all_logits = logits_for(np.ones(len(features), dtype=bool))
    calibrated_p = softmax(all_logits, temperature)
    raw_p = softmax(all_logits)
    horizons = [
        int(h)
        for h in (
            32,
            64,
            128,
            256,
            512,
            768,
            1024,
            1536,
            2048,
            3072,
            4096,
            8192,
            12288,
            16384,
            18432,
            20480,
            24576,
        )
        if h in edges
    ]
    train_labels = [label for label in data["labels"] if label["split"] == "train"]
    lengths = np.asarray([label["output_tokens"] for label in train_labels])
    cap = np.asarray([not label["natural_finish"] for label in train_labels])

    def risk_report(mask):
        result = {}
        ids = data["requests"][mask]
        for horizon in horizons:
            actual = data["remaining"][mask] > horizon
            raw = probability_remaining_gt(raw_p[mask], edges, horizon)
            probability = probability_remaining_gt(calibrated_p[mask], edges, horizon)
            baseline = km_conditional_survival(
                lengths, cap, data["generated"][mask], horizon
            )
            supported = np.isfinite(baseline)
            result[str(horizon)] = {
                "raw_model": binary_risk_metrics(raw, actual, ids),
                "calibrated_model": binary_risk_metrics(probability, actual, ids),
                "km_survival_baseline": (
                    binary_risk_metrics(
                        baseline[supported], actual[supported], ids[supported]
                    )
                    if supported.any()
                    else None
                ),
                "baseline_unsupported_samples": int((~supported).sum()),
            }
        return result

    results = {}
    groups = {
        "train_exact": exact_train,
        "selection_validation": selection,
        "calibration_validation": calibration,
        "test": test,
    }
    for name, mask in (diagnostic_groups or {}).items():
        if name in groups or np.asarray(mask).shape != test.shape:
            raise ValueError("invalid diagnostic group")
        mask = np.asarray(mask, dtype=bool) & ~data["censored"]
        if mask.any():
            groups[name] = mask
    if known_test_ids:
        known = np.isin(data["requests"], sorted(known_test_ids))
        if (test & known).any():
            groups["test_previous_stage"] = test & known
        if (test & ~known).any():
            groups["test_new_stage"] = test & ~known
    for name, mask in groups.items():
        results[name] = {
            "requests": len(set(data["requests"][mask])),
            "samples": int(mask.sum()),
            "raw_nll": distribution_nll(
                raw_p[mask],
                targets[mask],
                data["censored"][mask],
                data["requests"][mask],
            ),
            "calibrated_nll": distribution_nll(
                calibrated_p[mask],
                targets[mask],
                data["censored"][mask],
                data["requests"][mask],
            ),
            "risk_by_horizon_tokens": risk_report(mask),
        }
        if validation_cohorts and name in (
            "selection_validation",
            "calibration_validation",
        ):
            results[name]["balanced_raw_nll"] = validation_nll(raw_p[mask], mask)
            results[name]["balanced_calibrated_nll"] = validation_nll(
                calibrated_p[mask], mask
            )
    common_comparison = None
    if (
        bucket_profile in ("nonuniform84", "risk76", "nonuniform24k100")
        and original_parent is not None
    ):
        old_edges = original_parent["category_upper_edges"].numpy()
        common_edges = edges[np.isin(edges, old_edges)]
        old_model = build_mlp(
            features.shape[1],
            int(original_parent["hidden_width"]),
            len(old_edges) + 1,
            float(original_parent["dropout"]),
        ).to(device)
        old_model.load_state_dict(original_parent["state_dict"], strict=True)
        old_model.eval()
        reference_logits = []
        ref_mean = original_parent["feature_mean"].to(device)
        ref_std = original_parent["feature_std"].to(device)
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                stop = start + batch_size
                ref_hidden = torch.tensor(
                    data["hidden"][start:stop].astype(np.float32), device=device
                )
                ref_generated = torch.tensor(
                    data["generated"][start:stop], device=device
                )
                ref_features = torch.cat(
                    (
                        (ref_hidden - ref_mean) / ref_std,
                        (
                            torch.log1p(ref_generated)
                            / original_parent["position_log_scale"]
                        ).to(torch.float32)[:, None],
                    ),
                    dim=1,
                )
                reference_logits.append(old_model(ref_features).cpu().numpy())
        old_p = softmax(
            np.concatenate(reference_logits), float(original_parent["temperature"])
        )
        old_common = aggregate_probabilities(old_p, old_edges, common_edges)
        new_common = aggregate_probabilities(calibrated_p, edges, common_edges)
        common_targets = category_targets(data["remaining"], common_edges)
        common_comparison = {
            "category_upper_edges": common_edges.tolist(),
            "categories": len(common_edges) + 1,
            "tail_note": "shared inclusive category edges and an open tail",
            "reference_sha256": reference_sha256 or warmstart_sha256,
            "reference_role": "read-only comparison; not initialization"
            if reference_checkpoint
            else "warmstart parent",
            "results": {},
            "progress_by_split": {},
        }
        total_length = data["generated"] + data["remaining"]
        fraction = np.divide(
            data["generated"],
            total_length,
            out=np.zeros(len(total_length), dtype=np.float64),
            where=total_length > 0,
        )
        stage = np.minimum((fraction * 4).astype(int), 3)
        for name, mask in groups.items():
            common_comparison["results"][name] = {
                "parent_nll": distribution_nll(
                    old_common[mask],
                    common_targets[mask],
                    data["censored"][mask],
                    data["requests"][mask],
                ),
                "new_nll": distribution_nll(
                    new_common[mask],
                    common_targets[mask],
                    data["censored"][mask],
                    data["requests"][mask],
                ),
            }
            stage_results = {}
            for index, label in enumerate(("0-25%", "25-50%", "50-75%", "75-100%")):
                subset = mask & ~data["censored"] & (stage == index)
                if not subset.any():
                    continue
                stage_results[label] = {
                    "requests": len(set(data["requests"][subset])),
                    "samples": int(subset.sum()),
                    "parent_common_nll": distribution_nll(
                        old_common[subset],
                        common_targets[subset],
                        data["censored"][subset],
                        data["requests"][subset],
                    ),
                    "new_common_nll": distribution_nll(
                        new_common[subset],
                        common_targets[subset],
                        data["censored"][subset],
                        data["requests"][subset],
                    ),
                    f"new{len(edges) + 1}_true_bucket_nll": distribution_nll(
                        calibrated_p[subset],
                        targets[subset],
                        data["censored"][subset],
                        data["requests"][subset],
                    ),
                }
                from predictor_training_mix import length_metrics

                stage_results[label]["parent_length"] = length_metrics(
                    old_p[subset],
                    old_edges,
                    data["remaining"][subset],
                    data["requests"][subset],
                )
                stage_results[label]["new_length"] = length_metrics(
                    calibrated_p[subset],
                    edges,
                    data["remaining"][subset],
                    data["requests"][subset],
                )
            common_comparison["progress_by_split"][name] = stage_results
        del old_model, old_p, old_common, new_common
    stratified = {}
    for field in ("phases", "languages"):
        for value in sorted(set(data[field][test])):
            mask = test & (data[field] == value)
            stratified[f"{field}/{value}"] = risk_report(mask)
    validation_stratified = {}
    for name, group in (
        ("selection_validation", selection),
        ("calibration_validation", calibration),
    ):
        validation_stratified[name] = {
            phase: risk_report(group & (data["phases"] == phase))
            for phase in ("PREFILL_COMPLETE", "DECODE")
            if (group & (data["phases"] == phase)).any()
        }
    report = {
        "format_version": 2,
        "objective": "remaining_length_distribution",
        "training_revision": revision,
        "capture_preflight": preflight,
        "capture_audit": data["audit"],
        "best_epoch": best_epoch,
        "seed": seed,
        "training_parameters": {
            "hidden_width": hidden_width,
            "network_depth": network_depth,
            "hidden_widths": hidden_widths(hidden_width, network_depth),
            "bin_step": bin_step,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "bucket_profile": bucket_profile,
            "head_warmup_epochs": head_warmup_epochs,
            "head_learning_rate": head_learning_rate,
        },
        "device": str(device),
        "category_upper_edges": edges.tolist(),
        "overflow_category": overflow_category,
        "open_tail_lower_inclusive": int(edges[-1]) + 1 if overflow_category else None,
        "support_scope": checkpoint["support_scope"],
        "support_selection": support_selection,
        "censored_training_requests": sum(
            not x["natural_finish"] for x in train_labels
        ),
        "censor_rule": "known lower bound coarsened to its containing category"
        if overflow_category
        else "excluded whole capped requests; raw capture retained for diagnostics",
        "selection_validation_request_ids": sorted(set(data["requests"][selection])),
        "calibration_validation_request_ids": sorted(
            set(data["requests"][calibration])
        ),
        "temperature": temperature,
        "calibration_curve": calibration_curve,
        "metric_weighting": "each request has equal total sample weight",
        "training_weighting": weighting_provenance or {"rule": "request equal"},
        "validation_scope": "legacy/new validation cohorts, equal loss mass"
        if validation_cohorts
        else "primary new requests only"
        if validation_ids is not None
        else "all validation trees",
        "validation_cohorts": {
            name: {
                "selection_ids": sorted(set(data["requests"][selected])),
                "calibration_ids": sorted(set(data["requests"][calibrated])),
                "loss_mass": 0.5,
            }
            for name, (selected, calibrated) in validation_cohorts.items()
        },
        "previous_stage_test_request_ids": sorted(known_test_ids or set()),
        "horizon_note": "capacity-exceedance probability, not physical CUDA OOM",
        "results": results,
        "test_stratified": stratified,
        "validation_stratified": validation_stratified,
        "history": history,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "warmstart_provenance": checkpoint["warmstart_provenance"],
        "reference_provenance": {
            "sha256": reference_sha256,
            "role": "read-only evaluation; no parameters or normalization inherited",
        }
        if reference_checkpoint
        else None,
        "common_bucket_comparison": common_comparison,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        out_dir / "distribution_predictions.npz",
        probabilities=calibrated_p.astype(np.float32),
        category_upper_edges=edges,
        temperature=np.array(temperature),
        generated_tokens=data["generated"],
        observed_remaining=data["remaining"],
        censored=data["censored"],
        request_ids=data["requests"],
        splits=data["splits"],
        phases=data["phases"],
        languages=data["languages"],
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "temperature": temperature,
                "checkpoint": str(checkpoint_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    emit(f"续训完成：最佳第{best_epoch}轮，温度{temperature:.2f}，结果保存到{out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    capture_input = parser.add_mutually_exclusive_group(required=True)
    capture_input.add_argument("--run-dir", type=Path)
    capture_input.add_argument("--collection-batch", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-capture-revision", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--network-depth", type=int, choices=(2, 4), default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bin-step", type=int, default=32)
    parser.add_argument("--known-test-input", type=Path)
    replay_input = parser.add_mutually_exclusive_group()
    replay_input.add_argument("--replay-run-dir", type=Path)
    replay_input.add_argument("--replay-collection-batch", type=Path)
    parser.add_argument("--expected-replay-input-sha256")
    parser.add_argument("--expected-replay-capture-revision")
    parser.add_argument("--warmstart-checkpoint", type=Path)
    parser.add_argument("--warmstart-sha256")
    parser.add_argument("--tail-edges", nargs="+", type=int)
    parser.add_argument(
        "--bucket-profile",
        choices=(
            "parent",
            "nonuniform84",
            "risk76",
            "nonuniform24k100",
            "nonuniform16k91",
        ),
        default="parent",
    )
    parser.add_argument("--head-warmup-epochs", type=int, default=0)
    parser.add_argument("--head-learning-rate", type=float, default=0.0003)

    parser.add_argument("--expected-feature-layer", default="final")
    args = parser.parse_args()
    if (
        min(
            args.epochs,
            args.batch_size,
            args.hidden_width,
            args.patience,
            args.bin_step,
            args.expected_gpu_count,
        )
        <= 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.dropout)
        or not 0 <= args.dropout < 1
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
        or not math.isfinite(args.head_learning_rate)
        or args.head_learning_rate <= 0
    ):
        parser.error("training dimensions and learning-rate must be positive")
    if bool(args.warmstart_checkpoint) != bool(args.warmstart_sha256):
        parser.error("warmstart checkpoint and SHA must be provided together")
    if (
        args.bucket_profile in ("risk76", "nonuniform24k100")
        and args.warmstart_checkpoint
    ):
        parser.error(
            f"{args.bucket_profile} classifiers train from scratch, "
            "without parent weights"
        )
    if args.tail_edges and not args.warmstart_checkpoint:
        parser.error("tail edges require warmstart checkpoint")
    if args.bucket_profile == "nonuniform16k91" and args.warmstart_checkpoint:
        parser.error("finite-support classifiers must train from scratch")
    if args.bucket_profile == "nonuniform84" and (
        not args.warmstart_checkpoint or args.tail_edges or args.head_warmup_epochs < 1
    ):
        parser.error(
            "84 categories require warmstart and head warmup, without tail-edges"
        )
    if not 0 <= args.head_warmup_epochs < args.epochs:
        parser.error("head warmup must leave at least one joint-training epoch")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("out-dir must be new or empty")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != args.expected_revision:
        parser.error("HEAD differs from expected revision")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("worktree has uncommitted files")
    data = None
    capture_dir = args.run_dir or args.collection_batch
    if args.collection_batch:
        from predictor_collection_training import load_collection_examples

        data, preflight = load_collection_examples(
            args.collection_batch,
            expected_revision=args.expected_capture_revision,
            expected_recipe_sha256=args.expected_input_sha256,
            expected_feature_layer=args.expected_feature_layer,
        )
    else:
        preflight = json.loads(
            (args.run_dir / "preflight.json").read_text(encoding="utf-8")
        )
        copied_input = args.run_dir / "input_requests.jsonl"
        if (
            not copied_input.is_file()
            or sha256_file(copied_input) != args.expected_input_sha256
        ):
            parser.error("archived input_requests.jsonl missing or SHA256 differs")
    if preflight.get("feature_layer", "final") != args.expected_feature_layer:
        parser.error("capture feature layer differs from expected")
    if (
        preflight["revision"] != args.expected_capture_revision
        or preflight["input_sha256"] != args.expected_input_sha256
    ):
        parser.error("capture revision or input SHA256 differs")
    if (
        sha256_file(Path(preflight["model_path"]) / "config.json")
        != (preflight["model_config_sha256"])
    ):
        parser.error("model config differs from capture")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(names) != args.expected_gpu_count or any(
        x.strip() != args.expected_gpu_name for x in names
    ):
        parser.error(f"GPU inventory differs: {names}")
    if data is None:
        data = load_examples(args.run_dir, include_censored=True)
    replay_meta = None
    known_test_ids = set()
    if args.replay_run_dir or args.replay_collection_batch:
        from predictor_warmstart import merge_captures

        if not args.expected_replay_input_sha256:
            parser.error("replay input SHA256 is required")
        if args.replay_collection_batch:
            from predictor_collection_training import load_collection_examples

            if not args.expected_replay_capture_revision:
                parser.error("replay collection capture revision is required")
            replay_data, replay_meta = load_collection_examples(
                args.replay_collection_batch,
                expected_revision=args.expected_replay_capture_revision,
                expected_recipe_sha256=args.expected_replay_input_sha256,
                expected_feature_layer=args.expected_feature_layer,
                require_complete=False,
            )
        else:
            replay_meta = json.loads(
                (args.replay_run_dir / "preflight.json").read_text()
            )
            if (
                sha256_file(args.replay_run_dir / "input_requests.jsonl")
                != args.expected_replay_input_sha256
                or replay_meta["input_sha256"] != args.expected_replay_input_sha256
            ):
                parser.error("replay input SHA256 differs")
            replay_data = load_examples(args.replay_run_dir, include_censored=True)
        for key in [
            "feature_layer",
            "feature_semantics",
            "model_config_sha256",
            "generation_contract",
        ]:
            if replay_meta.get(key) != preflight.get(key):
                parser.error(f"replay {key} differs")
        known_test_ids.update(replay_data["requests"][replay_data["splits"] == "test"])
        data = merge_captures(data, replay_data)
    if args.known_test_input:
        known_test_ids |= {
            row["id"]
            for row in read_jsonl(args.known_test_input)
            if row.get("split") == "test"
        }
        if not known_test_ids <= set(data["requests"][data["splits"] == "test"]):
            parser.error("previous-stage test IDs must remain in the test split")
    if data["audit"]["censored_requests"] / data["audit"]["requests"] > 0.2:
        parser.error("more than 20% of requests were capped; inspect capture")
    counts = {
        split: len(set(data["requests"][(data["splits"] == split) & ~data["censored"]]))
        for split in ("train", "validation", "test")
    }
    if args.bucket_profile == "nonuniform16k91":
        finite, _ = finite_support_examples(data)
        counts = {
            split: len(set(finite["requests"][finite["splits"] == split]))
            for split in ("train", "validation", "test")
        }
    if counts["train"] < 100 or counts["validation"] < 40 or counts["test"] < 20:
        parser.error(f"too few complete requests: {counts}")
    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable for formal training")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_preflight = {
        "hostname": socket.gethostname(),
        "training_revision": revision,
        "replay_capture": replay_meta,
        "capture_dir": str(capture_dir.resolve()),
        "capture_revision": preflight["revision"],
        "input_sha256": preflight["input_sha256"],
        "model_path": preflight["model_path"],
        "model_config_sha256": preflight["model_config_sha256"],
        "feature_layer": preflight.get("feature_layer", "final"),
        "gpu_names": names,
        "torch_version": str(torch.__version__),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out_dir / "training_preflight.json").write_text(
        json.dumps(training_preflight, indent=2) + "\n", encoding="utf-8"
    )
    fit_distribution(
        data,
        preflight,
        args.out_dir,
        revision=revision,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_width=args.hidden_width,
        network_depth=args.network_depth,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        bin_step=args.bin_step,
        known_test_ids=known_test_ids,
        warmstart_checkpoint=args.warmstart_checkpoint,
        warmstart_sha256=args.warmstart_sha256,
        tail_edges=args.tail_edges,
        bucket_profile=args.bucket_profile,
        head_warmup_epochs=args.head_warmup_epochs,
        head_learning_rate=args.head_learning_rate,
    )


if __name__ == "__main__":
    main()
