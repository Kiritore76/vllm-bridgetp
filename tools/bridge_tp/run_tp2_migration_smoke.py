# SPDX-License-Identifier: Apache-2.0
"""Explicit TP1-to-TP2 execution smoke using the existing rolling mechanism.

This bypasses policy trigger selection ONLY. It does not bypass target native
allocation, exact rank readiness, commit, ownership or source-release gates.
It is not evidence of a natural pressure trigger or migration benefit.
"""

import argparse
import contextlib
import json
import math
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import launch_pressure_balanced_pilot as pilot  # noqa: E402
from tools.bridge_tp.pressure_cpu import load  # noqa: E402
from tools.bridge_tp.pressure_session import (  # noqa: E402
    RollingPressureSession,
    channel_idle,
)
from tools.bridge_tp.run_tp2_calibration import prepare  # noqa: E402


class Audit:
    def __init__(self, path):
        self.path, self.lock = path, threading.RLock()

    def write(self, row):
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(row, recorded_unix_s=time.time())) + "\n")
            stream.flush()


def migration_request(inputs, max_tokens, request_id):
    """Freeze the source contract before serving or constructing its target."""
    from tools.bridge_tp.run_pressure_calibration import payload

    return load("sampling_contract").freeze_strict_greedy_sampling(
        payload(inputs, 2048, max_tokens, request_id)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--history-tokens", type=int, choices=(128, 2048), default=128)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--platform", choices=("a100", "qwen3"), default="a100")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--expected-revision")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--predictor-checkpoint", type=Path)
    parser.add_argument("--natural-probe", action="store_true")
    parser.add_argument("--target-background", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--diagnostic-probe", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("finite positive timeout required")
    if (
        args.predictor_checkpoint
        or args.natural_probe
        or args.target_background
        or args.diagnostic_probe
    ) and (args.platform != "qwen3" or not args.predictor_checkpoint):
        raise ValueError("loaded/live validation requires Qwen3 and its checkpoint")
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    target_tp=2,
                    source_gpu="2",
                    target_gpus="0,1",
                    trigger="explicit mechanism smoke",
                    history_tokens=args.history_tokens,
                    rate=0,
                )
            )
        )
        return 0
    from tools.bridge_tp import run_phase9_controller as mechanism
    from tools.bridge_tp.freeze_pressure_baseline import cloud_identity, freeze
    from tools.bridge_tp.run_pressure_calibration import wait_empty

    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if args.platform == "a100" and (
        len(names) != 4 or any(n.strip() != "NVIDIA A100-PCIE-40GB" for n in names)
    ):
        raise RuntimeError("four A100 PCIe 40GB required")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPUs busy; no unrelated processes are stopped")
    inputs = None
    if args.platform == "qwen3":
        from tools.bridge_tp.prepare_4090d_qwen3 import PROFILE, read, runtime_inventory
        from tools.bridge_tp.qwen3_calibration_contract import (
            migration_spec,
            require_revision,
            verified_inputs,
        )

        require_revision(args.expected_revision)
        runtime_inventory(read(PROFILE))
        inputs = verified_inputs(args.model_path, args.inputs)
        spec = migration_spec(args.model_path)
    else:
        spec = prepare(args.model_path)
        if pilot.sha(args.model_path / "config.json") != spec["model_config_sha256"]:
            raise ValueError("frozen model config differs")
        if pilot.sha(spec["checkpoint"]) != spec["checkpoint_sha256"]:
            raise ValueError("frozen predictor differs")
    common, settings = pilot.arguments(spec)
    args.out_root.mkdir(parents=True, exist_ok=False)
    freeze(args.out_root / "source_identity")
    common.write_json(
        args.out_root / "runtime_identity.json",
        cloud_identity(
            args.model_path, spec["checkpoint"] if args.platform == "a100" else None
        ),
    )
    plan = pilot.build_scene(
        spec,
        dict(name="mechanism", arrivals=spec["scenarios"][0]["arrivals"]),
        args.out_root,
        common,
        settings,
    )
    if args.platform == "qwen3":
        from tools.bridge_tp.qwen3_calibration_contract import adjust_migration_plan

        adjust_migration_plan(plan)
        if args.predictor_checkpoint:
            from tools.bridge_tp.qwen3_online_validation import (
                background_completion,
                configure_predictor,
                latest_inventory,
            )

            configure_predictor(plan, args.predictor_checkpoint, args.model_path)
    common.write_json(
        args.out_root / "effective_plan.json",
        {
            "platform": args.platform,
            "source_argv": plan["source_argv"],
            "target_argv": plan["target_argv"],
            "source_environment": {
                k: v
                for k, v in plan["source_env"].items()
                if k.startswith(("BRIDGETP_", "CUDA_", "VLLM_", "OMP_"))
                and not k.endswith("TOKEN")
            },
            "target_environment": {
                k: v
                for k, v in plan["target_env"].items()
                if k.startswith(("BRIDGETP_", "CUDA_", "VLLM_", "OMP_"))
                and not k.endswith("TOKEN")
            },
        },
    )
    directory = plan["run_dir"]
    directory.mkdir(parents=True)
    if inputs is None:
        inputs = pilot.read(
            ROOT / "experiments/phase9/pressure_balanced_v2/calibration.json"
        )
    request = migration_request(inputs, 6144, f"bridgetp-phase9-{directory.name}")
    warm_requests = {
        role: migration_request(inputs, 16, "warm-" + role)
        for role in ("source", "target")
    }
    config = SimpleNamespace(**plan["config"])
    io = load("online_io")
    mechanism.RuntimeControl(armed=False).write(directory)
    audit = Audit(directory / "pressure_audit.jsonl")
    result, code, processes = {}, 0, []
    pool = ThreadPoolExecutor(max_workers=4 + args.target_background)
    background_futures = []
    driver = None
    emissions = []
    first_idle_observed = None
    last_telemetry = 0.0
    telemetry = Audit(directory / "loaded_telemetry.jsonl")
    monitor = None
    try:
        for role in ("target", "source"):
            process = common.start_process(
                role,
                plan[role + "_argv"],
                plan[role + "_env"],
                args.out_root / (role + ".log"),
            )
            processes.append(process)
            common.wait_healthy(
                plan["config"][role + "_url"], process, spec["startup_timeout_s"]
            )
        if args.diagnostic_probe:
            registry = args.out_root / "diagnostic_processes.json"
            common.write_json(
                registry, [dict(pid=p.process.pid, name=p.name) for p in processes]
            )
            monitor = common.start_process(
                "diagnostic monitor",
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "tools/bridge_tp/monitor_pressure_host.py"),
                    "--output",
                    str(args.out_root / "host_monitor.jsonl"),
                    "--stop-file",
                    str(args.out_root / "monitor.stop"),
                    "--process-file",
                    str(registry),
                    "--gpu-processes",
                    "--gpu-interval-s",
                    "2",
                ],
                plan["source_env"],
                args.out_root / "monitor.log",
            )
        for role in ("source", "target"):
            io.post_streaming_completion(
                plan["config"][role + "_url"],
                warm_requests[role],
                args.timeout_s,
                lambda *values: None,
                directory / (role + "_warm_failure.json"),
                retain_chunks=False,
            )
        if args.natural_probe:
            from tools.bridge_tp.qwen3_online_validation import natural_probe

            natural_probe(config.source_url, directory, args.model_path, io)
        processes.append(
            common.start_process(
                "stager",
                plan["stager_argv"],
                plan["target_env"],
                args.out_root / "stager.log",
            )
        )
        if args.platform == "qwen3":
            from tools.bridge_tp.qwen3_migration_diagnostics import wait_stager_ready

            wait_stager_ready(directory, processes[-1])
        if args.diagnostic_probe:
            from tools.bridge_tp.qwen3_migration_diagnostics import control_batch

            common.write_json(
                registry, [dict(pid=p.process.pid, name=p.name) for p in processes]
            )
            control_batch(
                io,
                config.target_url,
                inputs,
                directory,
                pool,
                args.target_background + 1,
                "before",
            )
        native = wait_empty(directory)
        common.write_json(args.out_root / "native_initial_receipt.json", native)
        config.tp1_total_kv_blocks = native[0]["total_blocks"]
        config.tp4_total_kv_blocks = native[1]["total_blocks"]
        common.write_json(
            args.out_root / "effective_controller_config.json",
            dict(
                mechanism_config=vars(config),
                pressure_policy_executed=False,
                predictor_loaded=args.platform == "a100"
                or bool(args.predictor_checkpoint),
                native_capacity_replaced=True,
            ),
        )
        for index in range(args.target_background):
            background_futures.append(
                pool.submit(
                    background_completion,
                    io,
                    config.target_url,
                    migration_request(inputs, 6144, f"target-background-{index}"),
                    args.timeout_s,
                    directory,
                    index,
                )
            )
        recorder = io.ProxyRecorder(
            request["request_id"],
            io.ProxyMode.HOLD_BACK,
            emission_sink=emissions.append,
        )
        future = pool.submit(
            io.post_streaming_completion,
            config.source_url,
            request,
            args.timeout_s,
            recorder.on_source_token,
            directory / "source_failure.json",
            retain_chunks=False,
        )
        stream = SimpleNamespace(payload=request, future=future, recorder=recorder)
        settled = []
        coordinator = SimpleNamespace(finish=lambda *a, **kw: settled.append((a, kw)))
        driver = RollingPressureSession(
            run_dir=directory,
            migration_id="mechanism",
            config=config,
            coordinator=coordinator,
            ingress=SimpleNamespace(for_server_id=lambda request_id: stream),
            executor=pool,
            audit=audit,
            backend=mechanism,
            request_timeout_s=args.timeout_s,
        )
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline:
            if any(p.process.poll() is not None for p in processes[:2]):
                raise RuntimeError("serving engine exited")
            if args.predictor_checkpoint and time.monotonic() - last_telemetry >= 0.2:
                inventories = {
                    role: latest_inventory(directory / f"{role}_inventory.jsonl")
                    for role in ("source", "target")
                }
                telemetry.write(
                    dict(kind="inventory_observation", inventories=inventories)
                )
                last_telemetry = time.monotonic()
            view = mechanism.read_source_progress(directory, request, time.time())
            if view is None:
                if future.done():
                    future.result()
                    raise RuntimeError("source ended without progress evidence")
                time.sleep(0.05)
                continue
            if driver.record is None:
                if future.done():
                    future.result()
                    raise RuntimeError("source ended before explicit trigger")
                if view.output_tokens >= args.history_tokens:
                    if args.target_background:
                        target = latest_inventory(directory / "target_inventory.jsonl")
                        if any(f.done() for f in background_futures):
                            raise RuntimeError(
                                "background ended before migration START"
                            )
                        if not target or target["running"] < args.target_background:
                            time.sleep(0.05)
                            continue
                        if not 0 <= time.time() - target["sampled_at"] <= 2:
                            raise RuntimeError(
                                "loaded target inventory stale or future"
                            )
                        common.write_json(
                            directory / "target_loaded_at_start.json", target
                        )
                    # Target starts empty. The existing native placeholder
                    # allocator still decides whether the reserved prefix fits.
                    candidate = SimpleNamespace(
                        request_id=view.request_id,
                        generated=view.output_tokens,
                        max_output_tokens=request["max_tokens"],
                    )
                    snapshot = load("manager_m0").RuntimeSnapshot(
                        time.time(),
                        "mechanism",
                        view.request_id,
                        "LOCAL",
                        view.output_tokens,
                    )
                    driver.start(candidate, snapshot)
            else:
                driver.tick(view, snapshot)
                if (
                    args.target_background
                    and driver.settled
                    and not (
                        directory / "target_loaded_at_source_release.json"
                    ).exists()
                ):
                    if any(f.done() for f in background_futures):
                        raise RuntimeError("background did not overlap source release")
                    target = latest_inventory(directory / "target_inventory.jsonl")
                    if not target or not 0 <= time.time() - target["sampled_at"] <= 2:
                        raise RuntimeError(
                            "source-release target inventory unavailable"
                        )
                    if (
                        sum(
                            r["request_id"].startswith("cmpl-target-background-")
                            for r in target["requests"]
                        )
                        < args.target_background
                    ):
                        raise RuntimeError(
                            "background missing from native release inventory"
                        )
                    common.write_json(
                        directory / "target_loaded_at_source_release.json", target
                    )
                if driver.cancel_future is not None or driver.cleanup_requires_shutdown:
                    raise RuntimeError(
                        "migration did not finish; inspect cleanup evidence"
                    )
                if (
                    driver.settled
                    and first_idle_observed is None
                    and channel_idle(directory, "mechanism", driver.record.request_id)
                ):
                    first_idle_observed = time.time()
                if (
                    driver.settled
                    and driver.target_future is not None
                    and driver.target_future.done()
                    and future.done()
                ):
                    source, target = future.result(), driver.target_future.result()
                    break
            time.sleep(0.05)
        else:
            raise TimeoutError("explicit migration smoke deadline")
        if not settled or len(emissions) != request["max_tokens"]:
            raise RuntimeError("source release or full unified output missing")
        if [r["index"] for r in emissions] != list(range(request["max_tokens"])):
            raise RuntimeError("unified response indices are not contiguous")
        if not channel_idle(directory, "mechanism", driver.record.request_id):
            raise RuntimeError("matching source/target IDLE evidence missing")
        for index, background in enumerate(background_futures):
            response = background.result(timeout=args.timeout_s)
            if (
                len(response["token_ids"]) != 6144
                or response["finish_reason"] != "length"
            ):
                raise RuntimeError("background output incomplete")
            common.write_json(directory / f"background-{index}-result.json", response)
        wait_empty(directory, after_generations=[r["generation"] for r in native])
        full_speed = pilot.verify_full_speed(directory)
        if args.diagnostic_probe:
            control_batch(
                io,
                config.target_url,
                inputs,
                directory,
                pool,
                args.target_background + 1,
                "after",
            )
            wait_empty(directory)
        seams = [
            b["unix_s"] - a["unix_s"]
            for a, b in zip(emissions, emissions[1:])
            if a["origin"] == "source" and b["origin"] == "target"
        ]
        if len(seams) != 1:
            raise RuntimeError("exactly one source-to-target client seam required")
        if first_idle_observed is None:
            first_idle_observed = time.time()
        result = dict(
            status="COMPLETED",
            mechanism_only=True,
            natural_policy_trigger=False,
            target_tp=2,
            target_background_requests=args.target_background,
            predictor_loaded=bool(args.predictor_checkpoint) or args.platform == "a100",
            source_finish=source["finish_reason"],
            target_finish=target["finish_reason"],
            full_speed=full_speed,
            source_release_confirmed=True,
            output_tokens=len(emissions),
            visible_handoff_gap_s=seams[0],
            start_to_source_release_and_idle_observed_s=first_idle_observed
            - driver.record.t_decision,
            idle_observation_note=(
                "first polling observation after confirmed source release; "
                "not target output completion"
            ),
            history_trigger_tokens=args.history_tokens,
            observed_generated_at_start=driver.record.trigger_output_tokens - 1,
            source_release_to_channel_idle_checked=True,
            start_to_release_s=(
                mechanism.load_json(directory / "source_kv_release_receipt.json")[
                    "released_unix_ns"
                ]
                / 1e9
                - driver.record.t_decision
            ),
        )
        common.write_json(directory / "unified_tokens.json", emissions)
    except KeyboardInterrupt:
        code, result = 130, dict(status="INTERRUPTED", natural_policy_trigger=False)
    except Exception as error:
        code, result = (
            1,
            dict(
                status="TECHNICAL_FAILURE",
                error=repr(error),
                natural_policy_trigger=False,
            ),
        )
    finally:
        if driver:
            driver.adapter.close()
        common.stop_processes(processes)
        if monitor is not None:
            (args.out_root / "monitor.stop").touch()
            with contextlib.suppress(subprocess.TimeoutExpired):
                monitor.process.wait(timeout=10)
            common.stop_processes([monitor])
        pool.shutdown(wait=True, cancel_futures=False)
        ended = not any(
            common.process_group_alive(p)
            for p in processes + ([monitor] if monitor else [])
        )
        if not ended:
            result["status"], code = "TECHNICAL_FAILURE", 1
        result.update(runner_code=code, all_owned_groups_ended=ended)
        common.write_json(args.out_root / "result.json", result)
        if not args.no_archive:
            import tarfile

            with tarfile.open(str(args.out_root) + ".tar.gz", "w:gz") as archive:
                archive.add(args.out_root, arcname=args.out_root.name)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
