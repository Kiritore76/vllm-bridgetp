# SPDX-License-Identifier: Apache-2.0
"""Fixed-arrival STAY/migration pairs, including the four-run KV-release pilot."""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import launch_pressure_balanced_pilot as pilot  # noqa: E402
from tools.bridge_tp import pressure_health_screen as health_screen  # noqa: E402
from tools.bridge_tp import pressure_kv_pilot as kv  # noqa: E402
from tools.bridge_tp import pressure_sustained as sustained  # noqa: E402
from tools.bridge_tp import run_pressure_calibration as calibration  # noqa: E402
from tools.bridge_tp.pressure_progress import save_progress  # noqa: E402


def validate(config, inputs):
    if (
        config["schema"] != "PRESSURE_COMPARISON_V1"
        or config["arms"] != ["stay", "migrate"]
        or config["maximum_migrations"] != 1
        or config["policy_overrides"] != {"sustain_s": 1.5}
        or config["itl_s"] != 0.1
        or config["slow_fraction"] != 0.01
        or config["max_dispatch_lag_s"] != 0.5
        or len(config["recipes"]) != 4
    ):
        raise ValueError("comparison differs from the single-change eight-run plan")
    expected = {(kind, rep) for kind in ("pulse", "train") for rep in (1, 2)}
    if {(r["kind"], r["repetition"]) for r in config["recipes"]} != expected:
        raise ValueError("require pulse/train, two repetitions each")
    names = set()
    for recipe in config["recipes"]:
        if recipe["name"] in names:
            raise ValueError("duplicate recipe")
        names.add(recipe["name"])
        jobs = arrivals(inputs, recipe)
        if len(jobs) != (5 if recipe["kind"] == "pulse" else 16):
            raise ValueError("incorrect frozen request count")
        backgrounds = [j for j in jobs if "background" in j["id"]]
        if (
            sum(j["route"] == "tp1" for j in backgrounds) != 3
            or sum(j["route"] == "tp4" for j in backgrounds) != 1
            or any(j["at_s"] >= 1 for j in backgrounds)
            or any(j["at_s"] < 5 for j in jobs if j not in backgrounds)
        ):
            raise ValueError("frozen background/injection phases differ")
        for job in jobs:
            request = job["request"]
            background = "background" in job["id"]
            prompt_size = 2048 if background or recipe["kind"] == "pulse" else 6144
            output_size = (
                (2048 if job["route"] == "tp1" else 4096) if background else 16
            )
            if (
                len(request["prompt"]) != prompt_size
                or request["max_tokens"] != output_size
                or (not background and job["route"] != "tp1")
            ):
                raise ValueError("frozen workload shape differs")
        offsets = [j["at_s"] for j in jobs]
        if offsets != sorted(offsets) or offsets[0] < 0:
            raise ValueError("arrival offsets must be ordered and nonnegative")


def arrivals(inputs, recipe):
    return [
        dict(
            id=row["id"],
            at_s=row["at_s"],
            route=row["route"],
            request=calibration.payload(
                inputs, row["prompt_tokens"], row["max_tokens"], row["id"]
            ),
        )
        for row in recipe["arrivals"]
    ]


def build_plans(spec, config, inputs, root, common, settings):
    policy = pilot.read(spec["policy_config"])
    policy.update(config["policy_overrides"])
    policy_path = root / "comparison_policy.json"
    common.write_json(policy_path, policy)
    local_spec = copy.deepcopy(spec)
    local_spec["policy_config"] = str(policy_path)
    if policy.get("health_service"):
        local_spec["health_reference"] = str(
            Path(spec.get("service_evidence", spec["policy_config"])).parent
            / "health_reference_v2.json"
        )
    plans = []
    for recipe in config["recipes"]:
        # Counterbalance run order while keeping each pair adjacent.
        arms = ["stay", "migrate"] if recipe["repetition"] == 1 else ["migrate", "stay"]
        path = root / (recipe["name"] + "_arrivals.json")
        common.write_json(path, arrivals(inputs, recipe))
        for arm in arms:
            scene = dict(name=recipe["name"] + "-" + arm, arrivals=str(path))
            plan = pilot.build_scene(local_spec, scene, root, common, settings)
            if arm == "stay":
                plan["controller_argv"].append("--no-migration")
            for pool in ("source", "target"):
                plan[pool + "_env"].update(
                    BRIDGETP_PERFORMANCE_TRACE_DIR=str(
                        plan["run_dir"] / "performance_trace" / pool
                    ),
                    BRIDGETP_PERFORMANCE_SAMPLE_EVERY="8",
                    BRIDGETP_PERFORMANCE_FINALIZE_TOKEN=secrets.token_hex(32),
                )
            plans.append((recipe, arm, scene, plan))
    return plans


def finalize(plan, common):
    from tools.bridge_tp.run_atomic_takeover import _post_json
    from tools.bridge_tp.verify_performance_trace import verify_pool

    root, run_dir = plan["run_dir"].parent, plan["run_dir"]
    common.write_json(
        root / "native_empty_receipt.json", calibration.wait_empty(run_dir)
    )
    for pool, ranks in (("source", 1), ("target", 4)):
        pilot.announce(root.name, "收尾性能轨迹", pool=pool)
        result = _post_json(
            plan["config"][pool + "_url"],
            "/bridge_tp/v1/performance_trace/finalize",
            {"token": plan[pool + "_env"]["BRIDGETP_PERFORMANCE_FINALIZE_TOKEN"]},
            60,
        )
        common.write_json(root / f"{pool}_trace_finalization.json", result)
        if result.get("status") != "FINALIZED":
            raise RuntimeError("trace did not finalize")
        common.write_json(
            root / f"{pool}_trace_acceptance.json",
            verify_pool(run_dir / "performance_trace" / pool, ranks),
        )


def check_kv_shape(plan, jobs, common):
    # Only the initial pressure stratum must fit the one-offload shape.
    # Later arrivals are submitted normally; they never get an admission gate.
    jobs = [job for job in jobs if job["at_s"] < 10]
    result = kv.capacity_shape(calibration.wait_empty(plan["run_dir"]), jobs)
    common.write_json(plan["run_dir"].parent / "kv_capacity_shape.json", result)
    if result["status"] != "VALID":
        raise RuntimeError("actual KV capacity does not match the frozen experiment")


def check_health_shape(plan, jobs, common):
    import math

    source, target = calibration.wait_empty(plan["run_dir"])
    source_jobs = [j["request"] for j in jobs if j["at_s"] < 10 and j["route"] == "tp1"]
    target_jobs = [j["request"] for j in jobs if j["at_s"] < 10 and j["route"] == "tp4"]

    def full(req):
        return math.ceil((len(req["prompt"]) + req["max_tokens"]) / 16)

    needed_source = sum(map(full, source_jobs)) + 128 + 125
    needed_target = sum(map(full, target_jobs)) + max(map(full, source_jobs)) + 125
    valid = (
        source["block_size"] == target["block_size"] == 16
        and needed_source <= source["free_blocks"]
        and needed_target <= target["free_blocks"]
    )
    common.write_json(
        plan["run_dir"].parent / "health_capacity_shape.json",
        dict(
            valid=valid,
            source_required_blocks=needed_source,
            target_required_blocks=needed_target,
            source_free_blocks=source["free_blocks"],
            target_free_blocks=target["free_blocks"],
            note="experimental isolation check, not an ingress reservation",
        ),
    )
    if not valid:
        raise RuntimeError("health screen lacks the intended KV headroom")


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def result_metrics(rows, jobs, origin, lag_budget):
    planned = {j["id"]: j for j in jobs}
    if len(rows) != len(planned) or {r["job_id"] for r in rows} != set(planned):
        raise RuntimeError("missing/duplicate request result")
    metrics = []
    for row in rows:
        job = planned[row["job_id"]]
        emitted = row["response"]["emitted"]
        transport = row.get("source_transport")
        if (
            row["source_error"]
            or not transport
            or len(emitted) != job["request"]["max_tokens"]
            or [t["index"] for t in emitted] != list(range(len(emitted)))
            or row["initial_pool"] != job["route"]
        ):
            raise RuntimeError("incomplete output, wrong route or transport failure")
        times = [t["unix_s"] for t in emitted]
        if times != sorted(times):
            raise RuntimeError("visible token times regressed")
        gaps = [b - a for a, b in zip(times, times[1:])]
        started = transport["request_started_unix_s"]
        lag = started - origin - job["at_s"]
        metrics.append(
            dict(
                job_id=row["job_id"],
                pool=row["initial_pool"],
                role="background" if "background" in row["job_id"] else "injection",
                migrated=row["response"]["committed"],
                planned_s=job["at_s"],
                dispatch_lag_s=lag,
                arrival_valid=-0.01 <= lag <= lag_budget,
                output_tokens=len(emitted),
                ttft_s=times[0] - started,
                completion_s=times[-1] - origin,
                e2e_s=times[-1] - started,
                itl_p95_s=percentile(gaps, 0.95),
                max_itl_s=max(gaps, default=0),
                intervals=len(gaps),
                slow_intervals=sum(g > 0.1 for g in gaps),
                slow_fraction=sum(g > 0.1 for g in gaps) / len(gaps) if gaps else 0,
                handoff_stall_s=row["response"]["handoff_stall_s"],
            )
        )
    return metrics


def assess(plan, jobs, arm, config, common):
    run_dir = plan["run_dir"]
    audit = [
        json.loads(line)
        for line in (run_dir / "pressure_audit.jsonl").read_text().splitlines()
    ]
    starts = [r for r in audit if r.get("kind") == "pressure_actual_start"]
    if len(starts) > 1 or (arm == "stay" and starts):
        raise RuntimeError("comparison arm violated migration limit")
    if arm == "stay" and any(r.get("execution") == "RESERVED" for r in audit):
        raise RuntimeError("STAY arm reserved a migration")
    origin = next(
        r["unix_s"] for r in audit if r.get("kind") == "pressure_clock_origin"
    )
    metrics = result_metrics(
        pilot.read(run_dir / "pressure_request_results.json"),
        jobs,
        origin,
        config["max_dispatch_lag_s"],
    )
    common.write_json(run_dir.parent / "comparison_metrics.json", metrics)
    common.write_json(
        run_dir.parent / "measurement_window.json",
        dict(started_unix_s=origin, ended_unix_s=time.time()),
    )
    released = sum(r.get("kind") == "pressure_source_release_confirmed" for r in audit)
    return dict(
        status="COMPLETED",
        actual_starts=len(starts),
        source_releases=released,
        committed_requests=sum(m["migrated"] for m in metrics),
        fixed_arrivals_valid=all(m["arrival_valid"] for m in metrics),
        maximum_dispatch_lag_s=max(m["dispatch_lag_s"] for m in metrics),
        note="STAY/coverage gap is evidence, not proof of migration benefit",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--kv-release", action="store_true")
    modes.add_argument("--sustained", action="store_true")
    modes.add_argument("--health-screen", action="store_true")
    args = parser.parse_args()
    spec = pilot.load_spec(args.spec)
    pilot.announce("comparison", "核对checkout、代码、固定输入及模型SHA")
    pilot.verify_spec(spec)
    name = (
        "sustained.json"
        if args.sustained
        else ("kv_release.json" if args.kv_release else "comparison.json")
    )
    if args.health_screen:
        name = "health_screen.json"
    config = pilot.read(args.spec.parent / name)
    inputs = pilot.read(args.spec.parent / "calibration.json")
    calibration.validate_config(inputs)
    validator = (
        sustained.validate
        if args.sustained
        else (kv.validate if args.kv_release else validate)
    )
    if args.health_screen:
        validator = health_screen.validate
    validator(config, inputs)
    if args.health_screen and config["policy_overrides"] != pilot.read(
        args.spec.parent / "health_service_policy.json"
    ):
        raise ValueError(
            "health screen policy differs from frozen experimental settings"
        )
    kv_mode = args.kv_release or args.sustained
    total = 2 * len(config["recipes"])
    common, settings = pilot.arguments(spec)
    from tools.bridge_tp import phase8_stager
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    with tempfile.TemporaryDirectory(prefix="pressure-compare-preflight-") as temp:
        for _, _, _, plan in build_plans(
            spec, config, inputs, Path(temp), common, settings
        ):
            for pool in ("source", "target"):
                actual = make_arg_parser(FlexibleArgumentParser()).parse_args(
                    plan[pool + "_argv"][3:]
                )
                validate_parsed_serve_args(actual)
            phase8_stager.parse_args(plan["stager_argv"][2:])
            actual = pilot.controller.parse_args(plan["controller_argv"][2:])
            actual.controller_config = Path(spec["controller_config_template"])
            pilot.controller.preflight(actual)
    if args.validate_only:
        pilot.announce(
            "comparison",
            f"{total}场输入/serving/stager/controller参数检查通过；尚未启动GPU",
        )
        return 0
    calibration.check_gpu()
    pilot.announce("comparison", "GPU空闲；执行迁移、账本和对照入口CPU回归")
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            *[
                "tests.bridge_tp." + name
                for name in (
                    "test_pressure_comparison",
                    "test_pressure_kv_pilot",
                    "test_pressure_sustained",
                    "test_pressure_health_service",
                    "test_pressure_pause_diagnostics",
                    "test_pressure_health_screen",
                    "test_pressure_health_reference",
                    "test_pressure_policy",
                    "test_pressure_preflight",
                    "test_pressure_session",
                    "test_pressure_bootstrap",
                    "test_pressure_observation",
                    "test_rolling_cutover",
                    "test_delta_batching",
                    "test_performance_trace",
                )
            ],
        ],
        cwd=ROOT,
        check=True,
    )
    args.out_root.mkdir(parents=True, exist_ok=False)
    (args.out_root / "gpu_inventory.txt").write_text(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv"],
            text=True,
        )
    )
    common.write_json(args.out_root / "frozen_spec.json", spec)
    common.write_json(args.out_root / "comparison_config.json", config)
    plans = build_plans(spec, config, inputs, args.out_root, common, settings)
    results = []
    for recipe, arm, scene, plan in plans:
        pilot.announce(
            scene["name"], "开始对照场景", number=f"{len(results) + 1}/{total}", arm=arm
        )
        try:
            calibration.check_gpu()
            pilot.run_scene(
                spec,
                scene,
                plan,
                common,
                settings,
                diagnose_pauses=args.health_screen,
                before_shutdown=lambda plan: finalize(plan, common),
                before_workload=(
                    lambda plan, arrivals_path=scene["arrivals"]: (
                        check_health_shape if args.health_screen else check_kv_shape
                    )(plan, pilot.read(arrivals_path), common)
                )
                if kv_mode or args.health_screen
                else None,
            )
            result = assess(plan, pilot.read(scene["arrivals"]), arm, config, common)
            if kv_mode or args.health_screen:
                pilot.scene_result(plan["run_dir"], "kv-release")
                audit = [
                    json.loads(line)
                    for line in (plan["run_dir"] / "pressure_audit.jsonl")
                    .read_text()
                    .splitlines()
                ]
                coverage = (
                    health_screen.assess(
                        audit,
                        pilot.read(plan["run_dir"].parent / "comparison_metrics.json"),
                        recipe["kind"],
                        arm,
                    )
                    if args.health_screen
                    else kv.assess(
                        audit,
                        pilot.read(plan["run_dir"].parent / "comparison_metrics.json"),
                        arm,
                        config["minimum_remaining_at_start"],
                    )
                )
                common.write_json(
                    plan["run_dir"].parent / "kv_release_metrics.json", coverage
                )
                result["coverage"] = coverage["coverage"]
                if args.sustained or args.health_screen:
                    score = sustained.score(
                        pilot.read(plan["run_dir"] / "pressure_request_results.json"),
                        config,
                    )
                    score["queue_observations"] = sustained.queue_summary(
                        audit, score["origin_unix_s"], config["horizon_s"]
                    )
                    common.write_json(
                        plan["run_dir"].parent / "sustained_goodoutput.json", score
                    )
                    result.update(
                        G300=score["goodoutput_tokens_s"],
                        old_itl_G300=score["old_itl_G"],
                        load_coverage=score["load_coverage"],
                    )
        except Exception as error:
            result = dict(status="TECHNICAL_FAILURE", error=repr(error))
        results.append(
            dict(scene=scene["name"], pair=recipe["name"], arm=arm, **result)
        )
        common.write_json(args.out_root / "scene_results.json", results)
        save_progress(
            args.out_root / "pilot_progress.json",
            scene=scene["name"],
            stage="场景结束",
            completed_scenes=len(results),
            total_scenes=total,
            **result,
        )
        pilot.announce(scene["name"], "场景结束", **result)
        if result["status"] != "COMPLETED":
            return 1
    pairs = [
        dict(
            pair=recipe["name"],
            fixed_arrivals_valid=all(
                r["fixed_arrivals_valid"]
                for r in results
                if r["pair"] == recipe["name"]
            ),
        )
        for recipe in config["recipes"]
    ]
    if args.sustained or args.health_screen:
        for pair in pairs:
            arms = {r["arm"]: r for r in results if r["pair"] == pair["pair"]}
            stay, move = arms["stay"]["G300"], arms["migrate"]["G300"]
            pair.update(
                stay_G300=stay,
                migrate_G300=move,
                delta_G300=move - stay,
                relative_delta=move / stay - 1 if stay > 0 else None,
                load_coverage={arm: r["load_coverage"] for arm, r in arms.items()},
                migration_coverage=arms["migrate"]["coverage"],
            )
    common.write_json(args.out_root / "pair_acceptance.json", pairs)
    pilot.announce("comparison", f"{total}场完成；检查实际迁移覆盖与迟发后再评估收益")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
