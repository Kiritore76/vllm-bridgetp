# SPDX-License-Identifier: Apache-2.0
"""Explicit TP1-to-TP2 execution smoke using the existing rolling mechanism.

This bypasses policy trigger selection ONLY. It does not bypass target native
allocation, exact rank readiness, commit, ownership or source-release gates.
It is not evidence of a natural pressure trigger or migration benefit.
"""

import argparse
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
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("finite positive timeout required")
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
    from tools.bridge_tp.run_pressure_calibration import payload, wait_empty

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
    config = SimpleNamespace(**plan["config"])
    io = load("online_io")
    mechanism.RuntimeControl(armed=False).write(directory)
    audit = Audit(directory / "pressure_audit.jsonl")
    result, code, processes = {}, 0, []
    pool = ThreadPoolExecutor(max_workers=4)
    driver = None
    emissions = []
    first_idle_observed = None
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
        if inputs is None:
            inputs = pilot.read(
                ROOT / "experiments/phase9/pressure_balanced_v2/calibration.json"
            )
        for role in ("source", "target"):
            io.post_streaming_completion(
                plan["config"][role + "_url"],
                payload(inputs, 2048, 16, "warm-" + role),
                args.timeout_s,
                lambda *values: None,
                directory / (role + "_warm_failure.json"),
                retain_chunks=False,
            )
        processes.append(
            common.start_process(
                "stager",
                plan["stager_argv"],
                plan["target_env"],
                args.out_root / "stager.log",
            )
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
                predictor_loaded=args.platform == "a100",
                native_capacity_replaced=True,
            ),
        )
        request = payload(
            inputs,
            2048,
            6144,
            f"bridgetp-phase9-{directory.name}",
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
        wait_empty(directory, after_generations=[r["generation"] for r in native])
        full_speed = pilot.verify_full_speed(directory)
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
        pool.shutdown(wait=True, cancel_futures=False)
        ended = not any(common.process_group_alive(p) for p in processes)
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
