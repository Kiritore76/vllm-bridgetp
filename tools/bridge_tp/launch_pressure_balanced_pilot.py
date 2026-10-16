#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Own four fresh function-pilot server groups; never run paired benefit fitting.

The spec freezes input/code/model/checkpoint SHA. Repository bundles record the
actual clean committed HEAD at runtime, avoiding self-referential manifests.
Linux dependency/rolling checks precede GPU startup.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import run_pressure_balanced_pilot as controller  # noqa: E402
from tools.bridge_tp.pressure_progress import save_progress  # noqa: E402


def announce(scene, stage, **details):
    stamp = time.strftime("%H:%M:%S")
    fields = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[{stamp}] [{scene}] {stage} {fields}", flush=True)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_spec(path):
    spec = read(path)
    if spec.get("identity_mode") != "git-tracked-bundle-v1":
        return spec
    relative = path.resolve().relative_to(ROOT).as_posix()
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    spec["expected_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    spec["bundle_spec_path"] = relative
    spec["bundle_spec_sha256"] = sha(path)

    def resolve(value):
        resolved = (ROOT / value).resolve()
        resolved.relative_to(ROOT)
        return str(resolved)

    for key in ("policy_config", "service_evidence", "controller_config_template"):
        spec[key] = resolve(spec[key])
    if spec.get("health_reference"):
        spec["health_reference"] = resolve(spec["health_reference"])
    for scene in spec["scenarios"]:
        scene["arrivals"] = resolve(scene["arrivals"])
    return spec


def verify_spec(spec):
    if (
        spec["model_config_sha256"]
        != "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e"
        or spec["checkpoint_sha256"]
        != "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7"
    ):
        raise ValueError("frozen Qwen/predictor identities differ")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if head != spec["expected_head"]:
        raise ValueError("checkout HEAD differs from frozen pilot spec")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != spec["expected_branch"]:
        raise ValueError("checkout branch differs from frozen pilot spec")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("pilot requires committed code; preserve dirty work and stop")
    for name, expected in spec["sha256"].items():
        path = ROOT / name if not Path(name).is_absolute() else Path(name)
        if sha(path) != expected:
            raise ValueError(f"pilot file SHA differs: {name}")
    if sha(Path(spec["model_path"]) / "config.json") != spec["model_config_sha256"]:
        raise ValueError("model configuration SHA differs")
    if sha(spec["checkpoint"]) != spec["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA differs")
    if {r["name"] for r in spec["scenarios"]} != {
        "both-light",
        "source-decode",
        "target-prefill",
        "capacity-arrival",
    } or len(spec["scenarios"]) != 4:
        raise ValueError("first pilot requires exactly four function scenarios")


def arguments(spec):
    # Existing transport helper is imported only in the actual server environment.
    from tools.bridge_tp import run_phase9_cap0_calibration as common

    values = copy.deepcopy(spec["server_settings"])
    values.update(model_path=Path(spec["model_path"]), python_bin=Path(sys.executable))
    args = SimpleNamespace(**values)
    if (
        args.dtype != "bfloat16"
        or args.max_model_len != 8192
        or args.tp4_max_model_len != 16384
        or args.tp1_gpu != "0"
        or args.tp4_gpus != "1,2,3,4"
        or not args.persistent_channel
        or not args.preconnect_persistent_channel
        or not args.gpu_direct_history
        or not args.gpu_direct_delta
    ):
        raise ValueError("pilot transport/hardware scope differs from frozen mechanism")
    return common, args


def build_scene(spec, scene, run_root, common, args):
    run_dir = run_root / scene["name"] / "controller"
    config_path = run_root / scene["name"] / "controller_config.json"
    config = read(spec["controller_config_template"])
    config.update(
        source_url=f"http://127.0.0.1:{args.tp1_port}",
        target_url=f"http://127.0.0.1:{args.tp4_port}",
        run_dir=str(run_dir),
    )
    config.update(
        ready_notification_mode="UDP",
        ready_notification_port=args.ready_notification_port,
    )
    inherited = {k: v for k, v in os.environ.items() if k.startswith("BRIDGETP_")}
    try:
        for key in inherited:
            os.environ.pop(key)
        source = common.source_environment(args, scene["name"], run_dir)
    finally:
        os.environ.update(inherited)
    source.update(
        VLLM_DISABLE_COMPILE_CACHE="1",
        BRIDGETP_SHADOW_STRATEGY="S_NEW_OLD",
        BRIDGETP_REQUEST_FREEZE_ENABLED="1",
        BRIDGETP_GPU_DIRECT_HISTORY_PACING="1",
        BRIDGETP_PREDICTOR_LIVE_CHECKPOINT=spec["checkpoint"],
        BRIDGETP_PREDICTOR_LIVE_SHA256=spec["checkpoint_sha256"],
        BRIDGETP_PREDICTOR_LIVE_EVENTS=str(run_dir / "predictor_events.jsonl"),
        BRIDGETP_PRESSURE_INVENTORY_PATH=str(run_dir / "source_inventory.jsonl"),
        BRIDGETP_PRESSURE_POOL="tp1",
        BRIDGETP_ROLLING_MIN_LEAD_TOKENS="128",
    )
    target = os.environ.copy()
    # Never leak the source publisher into TP4 workers.
    for key in list(target):
        if key.startswith("BRIDGETP_"):
            target.pop(key)
    target.update(
        CUDA_VISIBLE_DEVICES=args.tp4_gpus,
        OMP_NUM_THREADS="1",
        BRIDGETP_TAKEOVER_ENABLED="1",
        BRIDGETP_TAKEOVER_RUN_DIR=str(run_dir),
        BRIDGETP_TAKEOVER_MIGRATION_ID=scene["name"],
        BRIDGETP_PRESSURE_INVENTORY_PATH=str(run_dir / "target_inventory.jsonl"),
        BRIDGETP_PRESSURE_POOL="tp4",
    )
    target_argv = common.server_command(args, 4, args.tp4_port) + [
        "--kv-transfer-config",
        common.target_connector(
            run_dir,
            gpu_resident_shadow=True,
            persistent_channel=True,
            ready_sync_mode="STREAM_EVENT",
            ready_notification_mode="UDP",
            ready_notification_port=args.ready_notification_port,
            channel_generation=args.channel_generation,
            preconnect_gpu_direct=True,
            gpu_direct_base_port=args.gpu_direct_base_port,
        ),
    ]
    source_argv = common.server_command(args, 1, args.tp1_port)
    stager_argv = [
        sys.executable,
        str(common.STAGER),
        "--run-dir",
        str(run_dir),
        "--delta-host",
        "127.0.0.1",
        "--delta-base-port",
        str(args.delta_port),
        "--delivery-host",
        "127.0.0.1",
        "--delivery-base-port",
        str(args.delivery_port),
        "--timeout-s",
        str(spec["run_timeout_s"]),
        "--gpu-resident-shadow",
    ]
    argv = [
        sys.executable,
        str(ROOT / "tools/bridge_tp/run_pressure_balanced_pilot.py"),
        "--policy",
        "pressure-balanced-v1",
        "--policy-config",
        spec["policy_config"],
        "--controller-config",
        str(config_path),
        "--service-evidence",
        spec["service_evidence"],
        "--arrivals",
        scene["arrivals"],
        "--run-dir",
        str(run_dir),
        "--source-inventory",
        str(run_dir / "source_inventory.jsonl"),
        "--target-inventory",
        str(run_dir / "target_inventory.jsonl"),
        "--source-context-limit",
        str(args.max_model_len),
        "--target-context-limit",
        str(args.tp4_max_model_len),
        "--predictor-events",
        str(run_dir / "predictor_events.jsonl"),
        "--predictor-sha256",
        spec["checkpoint_sha256"],
        "--migration-id",
        scene["name"],
        "--max-seconds",
        str(spec["run_timeout_s"]),
        "--tick-s",
        "0.2",
        "--no-progress-s",
        str(spec.get("no_progress_s", 60)),
    ]
    if spec.get("health_reference"):
        argv.extend(["--health-reference", spec["health_reference"]])
    return dict(
        run_dir=run_dir,
        config=config,
        config_path=config_path,
        source_env=source,
        target_env=target,
        source_argv=source_argv,
        target_argv=target_argv,
        stager_argv=stager_argv,
        controller_argv=argv,
    )


def scene_result(run_dir, name):
    rows = [
        json.loads(v)
        for v in (run_dir / "pressure_audit.jsonl").read_text().splitlines()
    ]
    starts = [r for r in rows if r.get("kind") == "pressure_actual_start"]
    released = [r for r in rows if r.get("kind") == "pressure_source_release_confirmed"]
    final = next(r for r in reversed(rows) if r.get("kind") == "pressure_final")
    shutdown = read(run_dir.parent / "owned_shutdown_receipt.json")
    pending_shutdown = final.get("shutdown_cleanup_required", False)
    if final["outcome"] not in {"COMPLETED", "COMPLETED_PENDING_OWNER_CLEANUP"}:
        raise RuntimeError("controller did not complete; keep full evidence")
    if len(starts) > 1 or any(
        r["state"] in {"ACTIVE", "COMMITTED", "ACCOUNTED"}
        and not (
            pending_shutdown
            and shutdown["all_owned_groups_ended"]
            and r["kind"] == "migration"
        )
        for r in final["reservations"]
    ):
        raise RuntimeError("duplicate START or outstanding reservation")
    decisions = [r for r in rows if r.get("kind") == "pressure_decision"]
    light = [
        r
        for r in decisions
        if r["decision"]["source"]["zone"] == "SPARE"
        and r["decision"]["target"]["zone"] == "SPARE"
    ]
    if any(
        r["execution"] == "RESERVED" and r["decision"].get("service_path") != "health"
        for r in light
    ):
        raise RuntimeError("actually light pools started migration")
    if name == "both-light":
        covered = bool(light) and not starts
    elif name == "source-decode":
        eligible = [
            r
            for r in decisions
            if r["execution"] == "RESERVED"
            and not r["decision"]["capacity_path"]
            and (r["decision"]["source"]["decode"] or 0) >= 1.2
            and sum(
                c["request"].get("action_available", True) and not c["rejections"]
                for c in r["decision"]["candidates"]
            )
            >= 2
        ]
        covered = bool(eligible and starts and released)
    elif name == "target-prefill":
        observed = [
            r
            for r in decisions
            if r["target_reading"]["pending_prompt_tokens"]
            and (r["decision"]["source"]["decode"] or 0) >= 1.2
        ]
        if any(r["target_reading"]["idle"] for r in observed):
            raise RuntimeError("target prefill was treated as idle")
        covered = bool(observed)
    else:
        capacity = [
            r
            for r in decisions
            if r["execution"] == "RESERVED" and r["decision"]["capacity_path"]
        ]
        begun = next(
            (r["unix_s"] for r in rows if r.get("kind") == "pressure_start_published"),
            None,
        )
        ended = next(
            (r["unix_s"] for r in rows if r.get("kind") == "pressure_settlement"), None
        )
        arrival = (
            begun is not None
            and ended is not None
            and any(
                r.get("kind") == "pressure_dispatch"
                and r["pool"] == "tp4"
                and begun <= r["unix_s"] < ended
                for r in rows
            )
        )
        covered = bool(capacity and released and arrival)
    return dict(
        status="FUNCTION_OBSERVED" if covered else "COVERAGE_GAP",
        actual_starts=len(starts),
        source_releases=len(released),
        note="coverage gap/STAY is not technical failure; inspect actual strata",
    )


def run_scene(
    spec, scene, plan, common, args, *, before_shutdown=None, before_workload=None
):
    from tools.bridge_tp.run_phase9_cap0_noop import warmup_prompt_shapes
    from tools.bridge_tp.run_phase9_controller import RuntimeControl

    root, processes = plan["run_dir"].parent, []
    plan["run_dir"].mkdir(parents=True, exist_ok=False)
    common.write_json(plan["config_path"], plan["config"])
    RuntimeControl(
        armed=False,
        migration_id=scene["name"],
        source_request_id_prefix="pressure-unselected-" + secrets.token_hex(16),
    ).write(plan["run_dir"])
    common.write_json(
        root / "effective_argv.json",
        {k: v for k, v in plan.items() if k.endswith("_argv")},
    )
    common.write_json(
        root / "effective_environment.json",
        {
            name: {
                key: value
                for key, value in plan[name + "_env"].items()
                if key.startswith(("BRIDGETP_", "CUDA_", "NCCL_", "VLLM_", "OMP_"))
                and not key.endswith("_TOKEN")
            }
            for name in ("source", "target")
        },
    )
    try:
        announce(scene["name"], "启动TP4服务，等待就绪")
        target = common.start_process(
            "target TP4", plan["target_argv"], plan["target_env"], root / "target.log"
        )
        processes.append(target)
        common.wait_healthy(
            plan["config"]["target_url"], target, spec["startup_timeout_s"]
        )
        announce(scene["name"], "TP4就绪；启动TP1服务")
        source = common.start_process(
            "source TP1", plan["source_argv"], plan["source_env"], root / "source.log"
        )
        processes.append(source)
        common.wait_healthy(
            plan["config"]["source_url"], source, spec["startup_timeout_s"]
        )
        announce(scene["name"], "TP1就绪；开始warmup")
        requests = [job["request"] for job in read(scene["arrivals"])]
        for pool in ("source", "target"):
            announce(scene["name"], "warmup", pool=pool)
            warmup_prompt_shapes(
                plan["config"][pool + "_url"],
                requests,
                root / (pool + "_warmup.json"),
                spec["run_timeout_s"],
            )
        if before_workload is not None:
            before_workload(plan)
        announce(scene["name"], "warmup完成；启动stager和正式调度器")
        stager = common.start_process(
            "stager", plan["stager_argv"], os.environ.copy(), root / "stager.log"
        )
        processes.append(stager)
        checked = controller.parse_args(
            plan["controller_argv"][2:] + ["--validate-only"]
        )
        controller.preflight(checked)
        (root / "controller_config.sha256").write_text(sha(plan["config_path"]) + "\n")
        worker = common.start_process(
            "pressure controller",
            plan["controller_argv"],
            os.environ.copy(),
            root / "controller.log",
        )
        processes.append(worker)
        deadline = time.monotonic() + spec["run_timeout_s"] + 60
        last_progress_at = -float("inf")
        while worker.process.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError("owned controller deadline reached")
            if time.monotonic() - last_progress_at >= 10:
                last_progress_at = time.monotonic()
                path = plan["run_dir"] / "pressure_progress.json"
                if path.exists():
                    status = read(path)
                    save_progress(
                        root.parent / "pilot_progress.json",
                        scene=scene["name"],
                        **status,
                    )
                    announce(
                        scene["name"],
                        status["stage"],
                        elapsed=status.get("elapsed_s"),
                        submitted=(
                            f"{status.get('submitted', 0)}/{status.get('total', '?')}"
                        ),
                        completed=status.get("completed", 0),
                        pending_dispatch=len(status.get("queued", [])),
                        engine_waiting=status.get("engine_waiting", {}),
                        pools=(
                            f"{status.get('source_zone', '?')}/"
                            f"{status.get('target_zone', '?')}"
                        ),
                        active_tokens=status.get(
                            "active_generated", status.get("visible_tokens", 0)
                        ),
                        migration=status.get("migration", "none"),
                    )
                    if status.get("queue_reasons"):
                        announce(
                            scene["name"], "排队原因", reasons=status["queue_reasons"]
                        )
                    if status.get("health"):
                        announce(scene["name"], "健康参考", pools=status["health"])
                else:
                    announce(scene["name"], "等待controller初始化")
            final_path = plan["run_dir"] / "pressure_request_results.json"
            if final_path.exists() and worker.process.poll() is None:
                # A failing Python process can wait for in-flight HTTP threads.
                # The owner will stop its servers in finally, releasing them.
                tail = (plan["run_dir"] / "pressure_audit.jsonl").read_text()
                if '"outcome": "FAILED_OR_INTERRUPTED"' in tail:
                    status_path = plan["run_dir"] / "pressure_progress.json"
                    status = read(status_path) if status_path.exists() else {}
                    raise RuntimeError(
                        f"controller failed: {status.get('error')}; "
                        f"submitted={status.get('submitted', '?')}"
                    )
            time.sleep(0.2)
        if worker.process.returncode:
            raise subprocess.CalledProcessError(
                worker.process.returncode, plan["controller_argv"]
            )
        if before_shutdown is not None:
            before_shutdown(plan)
    finally:
        announce(scene["name"], "清理本场受管进程，核对退出")
        common.stop_processes(processes)
        ended = not any(common.process_group_alive(process) for process in processes)
        common.write_json(
            root / "owned_shutdown_receipt.json",
            dict(
                all_owned_groups_ended=ended,
                method="owned process-group termination",
                processes=[
                    dict(
                        name=process.name,
                        pid=process.process.pid,
                        returncode=process.process.poll(),
                    )
                    for process in processes
                ],
            ),
        )
        if not ended:
            raise RuntimeError(
                "owned server groups did not terminate; commitments unresolved"
            )
    return scene_result(plan["run_dir"], scene["name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    announce("preflight", "核对HEAD、分支和代码/输入/模型SHA")
    verify_spec(spec)
    common, settings = arguments(spec)
    plans = [
        build_scene(spec, scene, args.out_root, common, settings)
        for scene in spec["scenarios"]
    ]
    from tools.bridge_tp import phase8_stager
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    for plan in plans:
        for key in ("source_argv", "target_argv"):
            actual = make_arg_parser(FlexibleArgumentParser()).parse_args(plan[key][3:])
            validate_parsed_serve_args(actual)
        phase8_stager.parse_args(plan["stager_argv"][2:])
        actual = controller.parse_args(
            plan["controller_argv"][2:] + ["--validate-only"]
        )
        actual.controller_config = Path(spec["controller_config_template"])
        controller.preflight(actual)
    if args.validate_only:
        announce("preflight", "真实serving/stager/controller参数检查通过；尚未启动GPU")
        return 0
    archive_path = args.out_root.parent / (args.out_root.name + ".tar.gz")
    if args.out_root.exists() or archive_path.exists():
        raise FileExistsError("refuse to reuse a pilot result directory")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(names) != 5 or any(
        name.strip() != "NVIDIA A100-PCIE-40GB" for name in names
    ):
        raise ValueError("five A100-PCIE-40GB GPUs required; UUIDs are not matched")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPUs already have compute processes; no server was started")
    announce("preflight", "核对GPU通过；运行rolling和会话CPU回归")
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "tests.bridge_tp.test_rolling_cutover",
            "tests.bridge_tp.test_pressure_session",
            "tests.bridge_tp.test_pressure_bootstrap",
            "tests.bridge_tp.test_pressure_observation",
            "tests.bridge_tp.test_delta_batching",
            "tests.bridge_tp.test_pressure_preflight",
            "tests.bridge_tp.test_pressure_policy",
        ],
        cwd=ROOT,
        check=True,
    )
    args.out_root.mkdir(parents=True)
    (args.out_root / "gpu_inventory.txt").write_text(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv"],
            text=True,
        )
    )
    common.write_json(args.out_root / "frozen_spec.json", spec)
    results, failures, first_error_rc = [], 0, 0
    selected = list(zip(spec["scenarios"], plans))
    if args.smoke_only:
        selected = [
            (scene, plan) for scene, plan in selected if scene["name"] == "both-light"
        ]
    for index, (scene, plan) in enumerate(selected, 1):
        announce(
            "pilot", "开始场景", scenario=f"{index}/{len(selected)} {scene['name']}"
        )
        try:
            result = run_scene(spec, scene, plan, common, settings)
        except Exception as error:
            failures += 1
            real_rc = getattr(error, "returncode", 1)
            first_error_rc = first_error_rc or real_rc
            result = dict(
                status="TECHNICAL_FAILURE", error=repr(error), returncode=real_rc
            )
        results.append(dict(scene=scene["name"], **result))
        announce(
            scene["name"],
            "场景结束",
            status=result["status"],
            failures=failures,
            error=result.get("error", "none"),
        )
        common.write_json(args.out_root / "scene_results.json", results)
        if failures >= 3:
            break
    try:
        announce("pilot", "打包原始证据", scenes=len(results), failures=failures)
        from tools.bridge_tp.result_archive import pack

        packed = pack(args.out_root, archive_path)
        print(json.dumps(packed), flush=True)
    except Exception as error:
        print(f"evidence packaging failed: {error!r}", file=sys.stderr, flush=True)
        if not first_error_rc:
            raise
    return first_error_rc


if __name__ == "__main__":
    raise SystemExit(main())
