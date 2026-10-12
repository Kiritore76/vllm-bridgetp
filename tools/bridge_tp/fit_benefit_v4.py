#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fit F8/F10 on grouped new-profile pairs; final tests require an explicit mode."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

NAMES = [
    "p_lower",
    "U",
    "p_U",
    "source_running",
    "target_running",
    "target_prompt_tokens_s_10s",
    "p_target",
    "U_target",
]
EXTRA = ["target_pending_prefill_tokens", "target_decode_pressure_p95_3s"]
MASK = ["target_decode_pressure_available"]
ALPHAS = (0.1, 1.0, 10.0, 100.0)
MARGINS = (0.0, 5.0, 10.0)
PROFILE_SHA = "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7"


def features(snapshot, width):
    """Build only predecision observable features; missing values remain None."""
    observation = snapshot.get("target_observation") or {}
    source = snapshot.get("source") or {}
    try:
        p = snapshot["p_guard_est_bounds"][0]
        u = snapshot["U"]
        s, t = source["source_running"], source["target_running"]
        arrival = observation["prompt_arrival_tokens_s"]
        values = [p, u, p * u, s, t, arrival, p * t, u * t]
        if width in (10, 11):
            status = observation.get("decode_status")
            available = status in {"VALID", "IDLE"}
            pending = observation.get("prefill_pending_tokens")
            pressure = observation.get("decode_pressure_p95")
            if not available:
                age = observation.get("telemetry_age_s")
                if (width == 10 or status != "MISSING"
                        or observation.get("integrity_errors")
                        or not isinstance(pending, (int, float))
                        or not math.isfinite(pending) or pending < 1024
                        or not isinstance(age, (int, float))
                        or not math.isfinite(age) or not 0 <= age <= 2):
                    return None
                # Encoded placeholder with an explicit availability indicator.
                # Never interpreted as measured zero decode pressure.
                pressure = 0.0
            values += [pending, pressure]
            if width == 11:
                values += [float(available)]
        if observation.get("version") != "TARGET_OBSERVATION_V1":
            return None
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in values
        ):
            return None
        return values
    except (KeyError, TypeError, IndexError):
        return None


def load_pairs(roots, allowed, *, selected_pair_ids=None):
    pairs, seen = [], set()
    for root in roots:
        plan = json.loads((root / "preregistration.json").read_text(encoding="utf-8"))
        if plan.get("diagnostic_only"):
            raise ValueError("diagnostic repair pilot is excluded from model selection")
        if plan["protocol"] != "RISK76_MLP4_TARGET_PHASE_V4":
            raise ValueError("mixed collection protocol")
        if plan["predictor_sha256"] != PROFILE_SHA:
            raise ValueError("mixed predictor identity")
        selected = set(plan["selected_pairs"])
        for row in plan["all_pairs"]:
            if row["pair_id"] not in selected or row["role"] not in allowed:
                continue  # In selection mode, do not read final test outcomes.
            if (
                selected_pair_ids is not None
                and row["pair_id"] not in selected_pair_ids
            ):
                continue  # Explicit subset for offline diagnostics; main uses all.
            path = root / row["pair_id"] / "probability_outcomes.json"
            if not path.exists():
                raise ValueError(f"missing probability outcome: {path}")
            outcome = json.loads(path.read_text(encoding="utf-8"))
            if outcome.get("status") != "COLLECTOR_PILOT_NOT_FITTED_NET_BENEFIT":
                raise ValueError("expected probability collector outcome contract")
            journals = list(
                path.parent.glob("*/*/r01_shadow_only/controller/predictor_features")
            )
            if len(journals) != 2 or any(
                not (directory / "replay_verification.json").is_file()
                for directory in journals
            ):
                raise ValueError("paired predictor replay verification missing")
            for directory in journals:
                replay = json.loads(
                    (directory / "replay_verification.json").read_text(encoding="utf-8")
                )
                error = replay.get("max_probability_absolute_error")
                if (
                    replay.get("checkpoint_sha256") != PROFILE_SHA
                    or not replay.get("saved_states")
                    or not isinstance(error, (float, int))
                    or not math.isfinite(error)
                    or error > 1e-5
                ):
                    raise ValueError("paired predictor replay verification failed")
            for sample in outcome["samples"]:
                key = (row["seed"], row["source_family_group"], sample["threshold"])
                if key in seen:
                    raise ValueError("duplicate pilot/train root or threshold")
                seen.add(key)
                snapshot = (sample.get("start_candidate") or {}).get("snapshot") or {}
                control = (sample.get("stay_candidate") or {}).get("snapshot") or {}
                if any(
                    state
                    and (state.get("prediction") or {}).get("checkpoint_sha256")
                    != PROFILE_SHA
                    for state in (snapshot, control)
                ):
                    raise ValueError("candidate predictor identity differs")
                if sample["effect_sample_eligible"] and (
                    sample.get("descriptive_delta_tokens_s") is None
                    or sample.get("stay_quality_output_tokens_s") is None
                ):
                    raise ValueError("eligible pair lacks a finite quality label")
                pair = {
                    **row,
                    "eligible": sample["effect_sample_eligible"],
                    "delta": sample.get("descriptive_delta_tokens_s"),
                    "stay_q": sample.get("stay_quality_output_tokens_s"),
                    "snapshot": snapshot,
                    "stay_snapshot": control,
                    "matching": {
                        "generated_token_difference": sample.get(
                            "predecision_generated_token_difference"
                        ),
                        "H_difference": sample.get("predecision_H_difference"),
                    },
                }
                pairs.append(pair)
    return pairs


def train(rows, width, alpha):
    valid = [
        row for row in rows if row["eligible"] and features(row["snapshot"], width)
    ]
    if len(valid) < 3:
        raise ValueError("too few supported training pairs")
    x = np.asarray([features(row["snapshot"], width) for row in valid], dtype=float)
    y = np.asarray([row["delta"] for row in valid], dtype=float)
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1
    z = (x - mean) / std
    intercept = float(y.mean())
    coefficients = np.linalg.solve(
        z.T @ z + alpha * np.eye(width), z.T @ (y - intercept)
    )
    return {
        "width": width,
        "alpha": alpha,
        "features": NAMES + (EXTRA if width in (10, 11) else [])
        + (MASK if width == 11 else []),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "support_min": x.min(0).tolist(),
        "support_max": x.max(0).tolist(),
        "training_pairs": len(valid),
    }


def predict(model, row, *, enforce_support=True):
    if model["width"] == 11 and model.get("features") != NAMES + EXTRA + MASK:
        raise ValueError("F11 feature contract differs")
    value = features(row["snapshot"], model["width"])
    if value is None:
        return None
    x = np.asarray(value)
    if enforce_support and (
        np.any(x < model["support_min"]) or np.any(x > model["support_max"])
    ):
        return None
    return float(
        ((x - model["mean"]) / model["std"]) @ model["coefficients"]
        + model["intercept"]
    )


def evaluate(model, rows, margin):
    records = []
    for row in rows:
        if not row["eligible"]:
            continue
        prediction = predict(model, row)
        delta = row["delta"]
        epsilon = max(5.0, 0.02 * row["stay_q"])
        action = int(prediction is not None and prediction > margin)
        records.append(
            {
                "pair_id": row["pair_id"],
                "target_state": row["target_state"],
                "delta": delta,
                "prediction": prediction,
                "action": action,
                "epsilon": epsilon,
                "harmful_start": action and delta < -epsilon,
                "missed_positive": not action and delta > epsilon,
                "harmful_loss": max(-delta, 0) * action,
                "regret": max(0, delta) - action * delta,
            }
        )
    known = [row for row in records if row["prediction"] is not None]
    errors = [row["prediction"] - row["delta"] for row in known]
    return {
        "pairs": len(records),
        "known": len(known),
        "unknown": len(records) - len(known),
        "mae": float(np.mean(np.abs(errors))) if errors else None,
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "harmful_starts": sum(bool(row["harmful_start"]) for row in records),
        "missed_positives": sum(bool(row["missed_positive"]) for row in records),
        "total_harmful_loss": sum(row["harmful_loss"] for row in records),
        "total_regret": sum(row["regret"] for row in records),
        "records": records,
        "by_target_state": {
            state: {
                "pairs": sum(row["target_state"] == state for row in records),
                "unknown": sum(
                    row["target_state"] == state and row["prediction"] is None
                    for row in records
                ),
                "harmful_starts": sum(
                    bool(row["harmful_start"])
                    for row in records
                    if row["target_state"] == state
                ),
                "total_regret": sum(
                    row["regret"] for row in records if row["target_state"] == state
                ),
            }
            for state in ("idle", "decode", "prefill")
        },
    }


def grouped_alpha(rows, width):
    partitions = [
        ("request_family", lambda row: row["source_family_group"]),
        ("load_block", lambda row: row["source_state"] + ":" + row["target_state"]),
    ]
    scores = []
    for alpha in ALPHAS:
        errors = []
        for _, group_key in partitions:
            for group in sorted({group_key(row) for row in rows}):
                fold = [row for row in rows if group_key(row) != group]
                held = [row for row in rows if group_key(row) == group]
                try:
                    model = train(fold, width, alpha)
                except ValueError:
                    continue
                for row in held:
                    value = predict(model, row, enforce_support=False)
                    if row["eligible"] and value is not None:
                        errors.append(abs(value - row["delta"]))
        scores.append(
            {
                "alpha": alpha,
                "mae": float(np.mean(errors)) if errors else None,
                "held_pairs": len(errors),
            }
        )
    available = [score for score in scores if score["mae"] is not None]
    if not available:
        raise ValueError("group validation has no supported pairs")
    return min(available, key=lambda score: score["mae"])["alpha"], scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collections", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--frozen-models",
        type=Path,
        help="Explicit final-test mode; no training or selection",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if args.frozen_models:
        frozen = json.loads(args.frozen_models.read_text(encoding="utf-8"))
        if (
            frozen.get("protocol") != "RISK76_MLP4_TARGET_PHASE_V4"
            or frozen.get("predictor_sha256") != PROFILE_SHA
        ):
            raise ValueError("frozen model protocol or predictor differs")
        pairs = load_pairs(args.collections, {"test"})
        result = {
            name: evaluate(item["model"], pairs, item["margin"])
            for name, item in frozen["models"].items()
        }
        result["frozen_models_sha256"] = hashlib.sha256(
            args.frozen_models.read_bytes()
        ).hexdigest()
    else:
        pairs = load_pairs(args.collections, {"train", "validation"})
        training = [row for row in pairs if row["role"] == "train"]
        validation = [row for row in pairs if row["role"] == "validation"]
        if not validation:
            raise ValueError("selection requires separate validation pairs")
        result = {
            "protocol": "RISK76_MLP4_TARGET_PHASE_V4",
            "models": {},
            "status": "OFFLINE_ONLY_NO_VALIDATED_LCB",
            "predictor_sha256": PROFILE_SHA,
        }
        for width in (8, 11):
            alpha, cv = grouped_alpha(training, width)
            model = train(training, width, alpha)
            choices = [
                {"margin": margin, "evaluation": evaluate(model, validation, margin)}
                for margin in MARGINS
            ]
            chosen = min(
                choices,
                key=lambda item: (
                    item["evaluation"]["total_harmful_loss"],
                    item["evaluation"]["total_regret"],
                ),
            )
            result["models"][f"F{width}"] = {
                "model": model,
                "margin": chosen["margin"],
                "group_cv": cv,
                "validation_choices": choices,
            }
    models = frozen["models"] if args.frozen_models else result["models"]
    evaluation_rows = pairs if args.frozen_models else validation
    common = [
        row
        for row in evaluation_rows
        if row["eligible"]
        and all(predict(item["model"], row) is not None for item in models.values())
    ]
    result["common_supported_evaluation"] = {
        name: evaluate(item["model"], common, item["margin"])
        for name, item in models.items()
    }
    result["engineering_gate"] = {
        "automatic_deployment": False,
        "meaningful_negative_pairs": sum(
            row["eligible"] and row["delta"] < -max(5.0, 0.02 * row["stay_q"])
            for row in evaluation_rows
        ),
        "meaningful_positive_pairs": sum(
            row["eligible"] and row["delta"] > max(5.0, 0.02 * row["stay_q"])
            for row in evaluation_rows
        ),
        "required_positive_and_negative_pairs_each": 2,
        "unsupported_is_unknown_not_correct": True,
    }
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
