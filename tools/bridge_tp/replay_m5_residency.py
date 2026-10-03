#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay TP1 residency decisions from an M5 online result without actuation.

An optional calibration JSON must provide independent preparation and benefit
estimates. Without it, uncertain fields remain null and no speculative START
decision is manufactured from a nominal bandwidth or future run receipts.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _policy_module():
    import importlib.util
    import sys

    path = ROOT / "vllm/bridge_tp/controller/residency_policy.py"
    spec = importlib.util.spec_from_file_location("residency_policy_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


POLICY = _policy_module()
GIB = 1024**3


def _read_result(path: Path, suffix: str) -> bytes:
    if path.is_dir():
        matches = list(path.rglob(suffix))
        if len(matches) != 1:
            raise ValueError(f"expected one {suffix}, found {len(matches)}")
        return matches[0].read_bytes()
    with tarfile.open(path, "r:gz") as archive:
        matches = [m for m in archive if m.isfile() and m.name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected one {suffix}, found {len(matches)}")
        handle = archive.extractfile(matches[0])
        assert handle is not None
        return handle.read()


def _jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def _probability_bounds(
    values: list[float], edges: list[int], horizon: int
) -> tuple[float, float]:
    """Same bin convention as M5: lower bound and conservative upper bound."""
    if horizon < 0:
        return 1.0, 1.0
    index = bisect.bisect_left(edges, horizon)
    if index == len(edges):
        return 0.0, values[-1]
    lower = sum(values[index + 1:])
    upper = lower if horizon == edges[index] else sum(values[index:])
    return lower, upper


def _expected_remaining_lower(
    values: list[float], edges: list[int], cap: int
) -> float:
    lower_edges = [0, *(edge + 1 for edge in edges)]
    return sum(p * min(cap, edge) for p, edge in zip(values, lower_edges))


def _calibration(path: Path | None) -> dict | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    required = ("basis", "effective_rates_gib_s", "tail_upper_s")
    benefit = (
        "tp1_token_time_s", "tp4_token_time_s", "handoff_upper_s",
        "target_penalty_upper_s", "min_useful_tokens",
    )
    if any(key not in value for key in required):
        raise ValueError("calibration is missing required measured fields")
    if not value["basis"] or set(value["effective_rates_gib_s"]) != {
        "LOW", "MEDIUM", "HIGH"
    }:
        raise ValueError("calibration needs a basis and all three rates")
    if any(key in value for key in benefit) and not all(
        key in value for key in benefit
    ):
        raise ValueError("benefit calibration must provide all benefit fields")
    numbers = [*value["effective_rates_gib_s"].values(), value["tail_upper_s"]]
    numbers.extend(value[key] for key in benefit[:-1] if key in value)
    for number in numbers:
        if (
            not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise ValueError("calibration values must be nonnegative finite numbers")
    if any(value["effective_rates_gib_s"][profile] <= 0 for profile in (
        "LOW", "MEDIUM", "HIGH"
    )) or (
        "tp1_token_time_s" in value and
        (value["tp1_token_time_s"] <= 0 or value["tp4_token_time_s"] <= 0)
    ):
        raise ValueError("rates and token times must be positive")
    if "min_useful_tokens" in value and (
        not isinstance(value["min_useful_tokens"], int)
        or value["min_useful_tokens"] < 0
    ):
        raise ValueError("min_useful_tokens must be a nonnegative integer")
    return value


def replay(path: Path, calibration: dict | None = None) -> dict:
    audit_bytes = _read_result(path, "controller/phase9_audit.jsonl")
    audit = _jsonl(audit_bytes)
    predictor = _jsonl(_read_result(path, "predictor_events.jsonl"))
    config = json.loads(_read_result(path, "provenance/controller_config.json"))
    capacity = json.loads(_read_result(path, "provenance/source_kv_capacity.json"))
    provenance = json.loads(_read_result(path, "provenance/inputs.json"))
    source_request = json.loads(_read_result(path, "controller/source_request.json"))
    prompt = source_request.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError("source request prompt tokens are unavailable")
    if config["tp1_total_kv_blocks"] != capacity["measured_blocks"]:
        raise ValueError("controller did not use measured TP1 KV capacity")
    metadata = next(row for row in audit if row.get("kind") == "run_metadata")
    header = predictor[0]
    if header.get("kind") != "predictor_header":
        raise ValueError("missing predictor header")
    checkpoint_sha = header["checkpoint_sha256"]
    if metadata.get("manager_m5_predictor_shadow") is not True:
        raise ValueError("input is not an M5 shadow run")

    predictions = {}
    for row in predictor[1:]:
        if row.get("kind") != "predictor_prediction":
            continue
        if row.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("predictor checkpoint changed in one run")
        predictions[(row["request_id"], row["generated_tokens"])] = row

    cfg = POLICY.ResidencyConfig()
    if calibration is not None:
        cfg = POLICY.ResidencyConfig(
            long_probability_min=float(calibration.get("tau_long", 0.8)),
            minimum_gain_s=float(calibration.get("minimum_gain_s", 0.0)),
            start_margin_s=float(calibration.get("start_margin_s", 2.0)),
        )
    cfg.validate()

    actual_start_tick = next((
        row["tick"] for row in audit
        if row.get("kind") == "manager_m1_start_decision"
        and row.get("decision", {}).get("action") == "START_SHADOW"
    ), None)
    ticks = []
    current: dict = {}

    def flush() -> None:
        if not current:
            return
        telemetry = current["telemetry"]
        signal = telemetry["capacity_signal"]
        m5 = current.get("manager_m5_predictor_shadow", {})
        m1 = current.get("manager_m1_start_decision", {})
        snapshot = m1.get("snapshot", {})
        source = telemetry["tp1"]
        target = telemetry["tp4"]
        # Free-KV decline during a prefill wave is not a sustained decode rate.
        # The separated decode counter is the only valid growth estimate here.
        growth = signal.get("decode_growth_tokens_s")
        source_fresh = (
            signal.get("samples", 0) >= 4
            and signal.get("sampled_unix_s") is not None
            and 0 <= telemetry["unix_s"] - signal["sampled_unix_s"] <= 2.0
            and source.get("preemptions_total", 0) == 0
        )
        context = snapshot.get("current_context_tokens")
        if context is None:
            context = len(prompt) + telemetry["output_tokens"]
        history_bytes = (
            context * config["policy"]["kv_bytes_per_token"]
            if isinstance(context, int) and context > 0 else None
        )
        ready_best = ready_initial = None
        if calibration is not None and history_bytes is not None:
            rates = calibration["effective_rates_gib_s"]
            ready_best = history_bytes / (max(rates.values()) * GIB) + calibration[
                "tail_upper_s"
            ]
            initial_profile = (m1.get("initial_rate_preview") or {}).get("profile")
            if initial_profile in rates:
                ready_initial = (
                    history_bytes / (rates[initial_profile] * GIB)
                    + calibration["tail_upper_s"]
                )

        probability_lower = gain_lower = long_horizon = None
        prediction_status = m5.get("status")
        if (
            calibration is not None and "tp1_token_time_s" in calibration
            and ready_initial is not None
            and prediction_status == "AVAILABLE"
            and m5.get("model_applicable_to_runtime_stop_rule") is True
        ):
            event = predictions.get((
                m5.get("request_id"), m5.get("prediction_output_tokens")
            ))
            if event is None or event["published_unix_ns"] > int(m5["unix_s"] * 1e9):
                prediction_status = "EVENT_NOT_AVAILABLE_AT_TICK"
            else:
                t1 = calibration["tp1_token_time_s"]
                t4 = calibration["tp4_token_time_s"]
                if t1 > t4:
                    n_prepare = math.ceil(ready_initial / t1)
                    n_break_even = math.ceil((
                        calibration["handoff_upper_s"]
                        + calibration["target_penalty_upper_s"]
                    ) / (t1 - t4))
                    long_horizon = (n_prepare + n_break_even
                                    + calibration["min_useful_tokens"])
                    cap = m5.get("max_remaining_output_tokens")
                    if isinstance(cap, int) and cap >= 0:
                        values = event["probabilities"]
                        edges = header["category_upper_edges"]
                        probability_lower = (
                            0.0 if long_horizon >= cap else
                            _probability_bounds(values, edges, long_horizon)[0]
                        )
                        useful_lower = max(0.0, _expected_remaining_lower(
                            values, edges, cap
                        ) - n_prepare)
                        gain_lower = (
                            useful_lower * (t1 - t4)
                            - calibration["handoff_upper_s"]
                            - calibration["target_penalty_upper_s"]
                        )

        required = snapshot.get("target_required_tokens")
        if required is None and context is not None and isinstance(
            m5.get("max_remaining_output_tokens"), int
        ):
            required = context + m5["max_remaining_output_tokens"]
        target_safe = (
            target["kv_usage_frac"] <= 0.85
            and target["num_waiting"] <= 4
            and required is not None
            and target["free_kv_blocks"] * target["block_size"] >= required
        )
        prefill_event = predictions.get((metadata.get("source_request_id"), 0))
        prefill_complete = telemetry["output_tokens"] > 0 or (
            prefill_event is not None
            and prefill_event.get("phase") == "PREFILL_COMPLETE"
            and prefill_event["published_unix_ns"] <= int(telemetry["unix_s"] * 1e9)
        )
        evidence = POLICY.ResidencyEvidence(
            prefill_complete=prefill_complete,
            state=telemetry["state"],
            source_fresh=source_fresh,
            source_free_kv_tokens=signal.get("free_kv_tokens"),
            source_guard_kv_tokens=signal.get("guard_free_kv_tokens"),
            unallocated_prefill_tokens=signal.get("prefill_pending_kv_tokens"),
            safe_pool_growth_tokens_s=growth,
            ready_best_s=ready_best,
            target_safe=target_safe,
            channel_available=snapshot.get("channel_available"),
            long_probability_lower=probability_lower,
            gain_lower_s=gain_lower,
        )
        decision = POLICY.decide_residency(evidence, cfg)
        headroom = decision.headroom_tokens
        guard_time = decision.time_to_guard_s
        capacity_due_if_local = (
            headroom is not None and guard_time is not None
            and ready_best is not None
            and guard_time <= ready_best + cfg.start_margin_s
        )
        ticks.append({
            "tick": telemetry["tick"],
            "unix_s": telemetry["unix_s"],
            "state": telemetry["state"],
            "output_tokens": telemetry["output_tokens"],
            "source_free_kv_tokens": signal.get("free_kv_tokens"),
            "source_guard_kv_tokens": signal.get("guard_free_kv_tokens"),
            "unallocated_prefill_tokens": signal.get("prefill_pending_kv_tokens"),
            "safe_pool_growth_tokens_s": growth,
            "capacity_signal_active": signal.get("active"),
            "ready_best_s": ready_best,
            "ready_initial_s": ready_initial,
            "capacity_due_if_local": capacity_due_if_local,
            "long_horizon_tokens": long_horizon,
            "prediction_status": prediction_status,
            "actual_m1_action": m1.get("decision", {}).get("action"),
            "after_actual_start": (
                actual_start_tick is not None
                and telemetry["tick"] > actual_start_tick
            ),
            "decision": decision.to_json(),
        })

    for row in audit:
        kind = row.get("kind")
        if kind == "telemetry":
            flush()
            current = {"telemetry": row}
        elif current and kind in {
            "manager_m5_predictor_shadow", "manager_m1_start_decision"
        }:
            current[kind] = row
    flush()
    counts = Counter(row["decision"]["action"] for row in ticks)
    return {
        "format_version": 1,
        "mode": "advisory_replay_no_actuation",
        "input": str(path.resolve()),
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "revision": provenance.get("revision"),
        "manifest_sha256": provenance.get("manifest_sha256"),
        "checkpoint_sha256": checkpoint_sha,
        "measured_tp1_kv_blocks": capacity["measured_blocks"],
        "actual_start_tick": actual_start_tick,
        "counterfactual_limit": (
            "ticks after actual Shadow start are observational diagnostics, "
            "not a causal STAY trajectory"
        ),
        "calibration_basis": calibration["basis"] if calibration else None,
        "capacity_estimates_available": calibration is not None,
        "long_benefit_estimates_available": (
            calibration is not None and "tp1_token_time_s" in calibration
        ),
        "action_counts": dict(counts),
        "ticks": ticks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = replay(args.result, _calibration(args.calibration))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(json.dumps({
            "output": str(args.out.resolve()),
            "ticks": len(result["ticks"]),
            "action_counts": result["action_counts"],
        }, ensure_ascii=False))
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
