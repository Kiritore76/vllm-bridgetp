# SPDX-License-Identifier: Apache-2.0
"""Build held-repeat-out prefill references and replay pressure gates on CPU.

Only recorded past SSE events and native inventory enter each replay tick.
GPU/engine completion labels enter reference training, never replay features.
There is no synthetic migration, candidate, benefit or TTFT SLO label.
"""

import argparse
import bisect
import collections
import hashlib
import json
import math
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import prefill_cost_reference as ref  # noqa: E402
from tools.bridge_tp.pressure_cpu import load  # noqa: E402

observation = load("pressure_observation")


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_archive(path):
    scenes = collections.defaultdict(
        lambda: dict(tokens={}, inventories={}, batches=[], observations=[])
    )
    metadata = {}
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = member.name.split("/")[1:]
            if not parts or any(p in {"..", ""} for p in parts):
                raise ValueError("invalid archive member")
            stream = archive.extractfile(member)
            if len(parts) == 1:
                if parts[0] in {
                    "frozen_spec.json",
                    "scene_results.json",
                    "calibration_config.json",
                }:
                    metadata[parts[0]] = json.load(stream)
                continue
            scene, filename = scenes[parts[0]], parts[-1]
            if filename.endswith(".tokens.jsonl"):
                rows = [json.loads(line) for line in stream]
                if [r["index"] for r in rows] != list(range(len(rows))):
                    raise ValueError("token sequence not contiguous")
                scene["tokens"][filename.removesuffix(".tokens.jsonl")] = [
                    r["unix_s"] for r in rows
                ]
            elif filename == "observation_events.jsonl":
                scene["observations"] = [json.loads(line) for line in stream]
            elif filename.endswith("_inventory.jsonl"):
                pool = "tp1" if filename.startswith("source") else "tp4"
                values = []
                for line in stream:
                    row = json.loads(line)
                    if any(
                        row.get(k)
                        for k in (
                            "writer_error",
                            "inventory_error",
                            "dropped",
                            "dropped_before",
                        )
                    ):
                        raise ValueError("inventory evidence incomplete")
                    if row.get("kind") == "evidence_footer":
                        continue
                    if row.get("pool_id") != pool:
                        raise ValueError("inventory pool differs")
                    if values and (
                        row["generation"] <= values[-1]["generation"]
                        or row["sampled_at"] < values[-1]["sampled_at"]
                    ):
                        raise ValueError("inventory time/generation regressed")
                    # pool_reading uses emptiness only; no candidate is fabricated.
                    row["requests"] = [True] if row["requests"] else []
                    batch = row.get("last_scheduled_batch")
                    if batch and not batch.get("error"):
                        batch["requests"] = [
                            r for r in batch["requests"] if r["prefill_tokens"]
                        ]
                    values.append(row)
                scene["inventories"][pool] = values
            elif filename == "engine.jsonl" and "performance_trace" in parts:
                pool = "tp1" if parts[-2] == "source" else "tp4"
                for line in stream:
                    row = json.loads(line)
                    if row.get("kind") == "engine_batch" and row["prefill_tokens"]:
                        row.update(pool=pool, run_id=parts[0])
                        scene["batches"].append(row)
            elif len(parts) == 2 and filename in {
                "request_results.json",
                "measurement_window.json",
                "recipe.json",
            }:
                scene[filename] = json.load(stream)
    selected = {}
    for result in metadata["scene_results.json"]:
        if result["status"] != "COMPLETED":
            raise ValueError("calibration scene did not complete")
        name = result["scene"]
        scene = scenes[name]
        window = scene["measurement_window.json"]
        scene["batches"] = [
            r
            for r in scene["batches"]
            if window["started_unix_s"]
            <= r["scheduled_unix_s"]
            <= window["ended_unix_s"]
        ]
        selected[name] = scene
    return metadata, selected


def cost_validation(model, batches):
    predicted = []
    unsupported = collections.Counter()
    for batch in batches:
        estimate = model.estimate(batch["pool"], batch)
        if estimate is None:
            unsupported[batch["pool"]] += 1
            continue
        predicted.append(
            dict(
                run_id=batch["run_id"],
                pool=batch["pool"],
                features=ref.features(batch),
                predicted_ms=estimate["engine_s"] * 1000,
                observed_ms=batch["completion_latency_ms"],
                absolute_relative_error=abs(
                    estimate["engine_s"] * 1000 / batch["completion_latency_ms"] - 1
                ),
            )
        )
    errors = [r["absolute_relative_error"] for r in predicted]
    return dict(
        validation_batches=len(batches),
        supported=len(predicted),
        unsupported=dict(unsupported),
        relative_error_p50=ref.quantile(errors, 0.5) if errors else None,
        relative_error_p95=ref.quantile(errors, 0.95) if errors else None,
        underprediction_over25pct=sum(
            r["predicted_ms"] < r["observed_ms"] * 0.75 for r in predicted
        ),
        predictions=predicted,
        scope="same-input independent repeat; no unseen-family accuracy claim",
    )


def events(scene):
    result = []
    for request in scene["request_results.json"]["requests"]:
        pool = "tp1" if request["pool"] == "source" else "tp4"
        request_id = request["id"]
        times = scene["tokens"][request_id]
        if len(times) != request["max_tokens"] or request["error"]:
            raise ValueError("incomplete stream")
        result.append((request["submitted_unix_s"], 0, request_id, pool))
        result.extend((timestamp, 1, request_id, pool) for timestamp in times)
        result.append((request["completed_unix_s"], 2, request_id, pool))
    return sorted(result)


def timeline(scene, cfg, model, lag_s):
    """Reconstruct the actual NativeServiceWindow, never finish-time averages."""
    event_rows = events(scene)
    cursor = 0
    service = observation.NativeServiceWindow()
    inv = scene["inventories"]
    times = {pool: [r["sampled_at"] for r in rows] for pool, rows in inv.items()}
    observed = {
        pool: [
            r
            for r in scene["observations"]
            if r.get("pool") == label and "metrics" in r and not r.get("error")
        ]
        for pool, label in (("tp1", "source"), ("tp4", "target"))
    }
    observed_times = {
        pool: [r["captured_unix_s"] for r in rows] for pool, rows in observed.items()
    }
    by_generation = {
        pool: {r["generation"]: r for r in rows} for pool, rows in inv.items()
    }
    window = scene["measurement_window.json"]
    result = []
    if lag_s is None:
        # Evaluate after actual observer cycles, including failed target scrapes.
        # A stalled observer must not manufacture continuous decision ticks.
        tick_times = sorted(
            {
                r["captured_unix_s"]
                for r in scene["observations"]
                if r.get("pool") == "target"
                and window["started_unix_s"]
                <= r["captured_unix_s"]
                <= window["ended_unix_s"]
            }
        )
    else:
        ticks = math.ceil((window["ended_unix_s"] - window["started_unix_s"]) / 0.2)
        tick_times = [
            min(window["ended_unix_s"], window["started_unix_s"] + 0.2 * index)
            for index in range(ticks + 1)
        ]
    for now in tick_times:
        while cursor < len(event_rows) and event_rows[cursor][0] <= now:
            timestamp, kind, request_id, pool = event_rows[cursor]
            if kind == 0:
                service.arrive(request_id, pool, timestamp)
            elif kind == 1:
                service.token(request_id, timestamp)
            else:
                service.finish(request_id)
            cursor += 1
        states = {}
        for pool in ("tp1", "tp4"):
            heartbeat = None
            if lag_s is None:
                pos = bisect.bisect_right(observed_times[pool], now) - 1
                if pos < 0:
                    states[pool] = None
                    continue
                event = observed[pool][pos]
                inventory = by_generation[pool][event["generation"]]
                published_at = event["observed_at"]
                heartbeat = event["metrics"]
            else:
                pos = bisect.bisect_right(times[pool], now - lag_s) - 1
                if pos >= 0:
                    inventory = inv[pool][pos]
                    published_at = inventory["sampled_at"] + lag_s
            if pos < 0:
                states[pool] = None
                continue
            reading = observation.pool_reading(
                inventory,
                published_at,
                service.snapshot(pool, now),
                cfg,
                now,
                heartbeat=heartbeat,
            )
            estimate = None
            batch = inventory.get("last_scheduled_batch")
            if (
                reading.prefill_batch_tokens
                and batch
                and not batch.get("error")
                and 0 <= now - batch["sampled_at"] <= cfg.max_age_s
            ):
                estimate = model.estimate(pool, batch)
            states[pool] = (reading, estimate)
        result.append((now, states))
    return result


def replay(rows, cfg, variant, origin):
    gate = ref.ReliefGate(cfg)
    unknown = collections.Counter()
    zones = {pool: collections.Counter() for pool in ("tp1", "tp4")}
    opportunities = []
    for now, states in rows:
        pressures = {}
        for pool, state in states.items():
            if state is None:
                pressure = ref.p.Pressure(
                    None, None, None, "UNKNOWN", ("no published inventory",)
                )
            else:
                reading, estimate = state
                pressure = (
                    ref.baseline_classify(reading, cfg, now)
                    if variant == "token_budget"
                    else ref.cost_pressure(reading, cfg, now, estimate)
                )
            pressures[pool] = pressure
            zones[pool][pressure.zone] += 1
            unknown.update(f"{pool}:{reason}" for reason in pressure.missing)
        if gate.update(now, pressures["tp1"], pressures["tp4"]):
            opportunities.append(
                dict(
                    at_s=now - origin,
                    persisted_s=now - gate.since,
                    source_decode=pressures["tp1"].decode,
                    source_prefill=pressures["tp1"].prefill,
                    target_decode=pressures["tp4"].decode,
                    target_prefill=pressures["tp4"].prefill,
                )
            )
    return dict(
        variant=variant,
        sustain_s=cfg.sustain_s,
        spare_ratio=cfg.spare_ratio,
        opportunities=opportunities,
        zone_ticks={p: dict(v) for p, v in zones.items()},
        missing_ticks=dict(unknown),
        ticks=len(rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata, scenes = load_archive(args.archive)
    cfg_path = ROOT / "experiments/phase9/pressure_balanced_v2/policy.json"
    cfg_bytes = cfg_path.read_bytes().replace(b"\r\n", b"\n")
    expected = metadata["frozen_spec.json"]["sha256"][
        cfg_path.relative_to(ROOT).as_posix()
    ]
    if hashlib.sha256(cfg_bytes).hexdigest() != expected:
        raise ValueError("policy differs from recorded baseline")
    cfg = ref.p.PressureConfig(**json.loads(cfg_bytes))
    implementation_sha = {
        name: hashlib.sha256(
            (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for name in (
            "tools/bridge_tp/prefill_cost_reference.py",
            "tools/bridge_tp/replay_pressure_calibration.py",
            "vllm/bridge_tp/controller/pressure_policy.py",
            "vllm/bridge_tp/controller/pressure_observation.py",
        )
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    batches = [r for scene in scenes.values() for r in scene["batches"]]
    model = ref.PrefillReference.fit(batches)
    with args.archive.open("rb") as stream:
        archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    write(
        args.out_dir / "prefill_reference.json",
        dict(
            schema="OFFLINE_PREFILL_REFERENCE_V1",
            archive_sha256=archive_sha,
            cells=model.cells,
            minimum_events=model.minimum_events,
            enabled_in_serving=False,
            implementation_sha256=implementation_sha,
            estimator="empirical engine P95; exact shape and bounded context",
        ),
    )
    folds = []
    for held in (1, 2):
        train = [r for r in batches if not r["run_id"].endswith(f"-r{held}")]
        validate = [r for r in batches if r["run_id"].endswith(f"-r{held}")]
        fold_model = ref.PrefillReference.fit(train)
        validation = cost_validation(fold_model, validate)
        replays = []
        for name, scene in scenes.items():
            if not name.endswith(f"-r{held}"):
                continue
            for lag_s in (None,) if scene["observations"] else (0, 0.2):
                rows = timeline(scene, cfg, fold_model, lag_s)
                for sustain_s in (1.0, 1.5, 2.0):
                    for spare in (0.6, 0.8):
                        settings = replace(cfg, sustain_s=sustain_s, spare_ratio=spare)
                        for variant in ("token_budget", "observed_cost"):
                            value = replay(
                                rows,
                                settings,
                                variant,
                                scene["measurement_window.json"]["started_unix_s"],
                            )
                            value.update(scene=name, assumed_inventory_lag_s=lag_s)
                            replays.append(value)
        folds.append(
            dict(
                held_repeat=held,
                training_batches=len(train),
                validation=validation,
                replays=replays,
            )
        )
        print(
            json.dumps(
                {
                    "held_repeat": held,
                    "supported": validation["supported"],
                    "validation_batches": validation["validation_batches"],
                    "relative_error_p95": validation["relative_error_p95"],
                }
            ),
            flush=True,
        )
    write(
        args.out_dir / "replay.json",
        dict(
            archive_sha256=archive_sha,
            policy_sha256=expected,
            folds=folds,
            engine_cost_reference_s=0.1,
            no_future_event_access=True,
            runtime_modified=False,
            implementation_sha256=implementation_sha,
            caveats=[
                "NativeServiceWindow uses recorded callbacks, not final averages",
                "old archives lack publication times and idle metrics heartbeat",
                "old: synthetic .2s ticks; new: actual observer ticks",
                "assumed 0/.2s inventory lag; no synthetic idle heartbeat",
                "pressure gate episodes are not feasible START or benefit estimates",
                "no counterfactual benefit, cooldown fit, or simulated migration",
                "ingress ledger and candidate feasibility are not reconstructed",
                "reference validated across two repeated runs, not new prompt families",
            ],
        ),
    )


if __name__ == "__main__":
    main()
