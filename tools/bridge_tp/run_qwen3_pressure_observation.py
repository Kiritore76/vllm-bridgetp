# SPDX-License-Identifier: Apache-2.0
"""Run the existing A100 pressure policy on a bounded TP1/TP2 stress matrix."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import qwen3_pressure_observation as obs  # noqa: E402
from tools.bridge_tp.prepare_4090d_qwen3 import (  # noqa: E402
    PROFILE,
    read,
    runtime_inventory,
)
from tools.bridge_tp.qwen3_calibration_contract import (  # noqa: E402
    migration_spec,
    require_revision,
    runtime_versions,
    verified_inputs,
)
from tools.bridge_tp.qwen3_online_validation import (  # noqa: E402
    CHECKPOINT_SHA,
    configure_predictor,
)
from tools.bridge_tp.result_archive import pack  # noqa: E402
from tools.bridge_tp.run_qwen3_online_loaded_validation import (  # noqa: E402
    gpu_configuration,
)
from tools.bridge_tp.run_qwen3_parameter_batch import gpu_idle, write  # noqa: E402


def child_arguments(args, plan, directory, jobs_path):
    return [
        "--platform",
        "qwen3",
        "--policy",
        "pressure-balanced-v1",
        "--policy-config",
        str(obs.ASSETS / "policy.json"),
        "--controller-config",
        str(directory / "controller_config.json"),
        "--service-evidence",
        str(obs.ASSETS / "service_evidence.json"),
        "--health-reference",
        str(obs.ASSETS / "health_reference.json"),
        "--arrivals",
        str(jobs_path),
        "--run-dir",
        str(plan["run_dir"]),
        "--source-inventory",
        str(plan["run_dir"] / "source_inventory.jsonl"),
        "--target-inventory",
        str(plan["run_dir"] / "target_inventory.jsonl"),
        "--predictor-events",
        str(plan["run_dir"] / "predictor_events.jsonl"),
        "--predictor-sha256",
        CHECKPOINT_SHA,
        "--migration-id",
        directory.name,
        "--source-context-limit",
        "8192",
        "--target-context-limit",
        "16384",
        "--max-seconds",
        "480",
        "--tick-s",
        ".2",
        "--no-progress-s",
        "90",
        "--no-migration",
        "--maximum-migrations",
        "1",
    ]


def run_scene(args, name, inputs, tokenizer):
    from tools.bridge_tp import launch_pressure_balanced_pilot as pilot
    from tools.bridge_tp import run_pressure_balanced_pilot as policy_runner
    from tools.bridge_tp.run_phase9_cap0_noop import warmup_prompt_shapes
    from tools.bridge_tp.run_pressure_calibration import service_only_plan, wait_empty

    directory = args.output / name
    directory.mkdir()
    spec = migration_spec(args.model)
    common, settings = pilot.arguments(spec)
    settings.max_model_len, settings.tp4_max_model_len = 8192, 16384
    plan = service_only_plan(spec, dict(name=name), args.output, common, settings)
    plan["run_dir"].mkdir()
    configure_predictor(plan, args.checkpoint, args.model)
    if any(
        "--kv-transfer-config" in plan[role + "_argv"] for role in ("source", "target")
    ):
        raise ValueError("observation services must not load a migration connector")
    values = obs.natural_jobs(obs.jobs(name, inputs), tokenizer)
    write(directory / "arrivals.json", values)
    write(directory / "scope.json", read(obs.ASSETS / "scope.json"))
    write(
        directory / "effective_argv.json",
        {r: plan[r + "_argv"] for r in ("source", "target")},
    )
    controller = dict(
        plan["config"],
        block_size=16,
        target_tp_size=2,
        transfer_mode="FULL_SPEED",
        rate={},
        policy=dict(kv_bytes_per_token=147456),
    )
    owned = []
    processes_record = directory / "diagnostic_processes.json"
    try:
        for role in ("target", "source"):
            print(f"[场景] {name} 启动 {role}", flush=True)
            process = common.start_process(
                role,
                plan[role + "_argv"],
                plan[role + "_env"],
                directory / f"{role}.log",
            )
            owned.append(process)
            write(
                processes_record, [dict(name=p.name, pid=p.process.pid) for p in owned]
            )
            common.wait_healthy(controller[role + "_url"], process, 900)
            warmup_prompt_shapes(
                controller[role + "_url"],
                [v["request"] for v in values],
                directory / f"{role}_warmup.json",
                120,
            )
        initial = wait_empty(plan["run_dir"])
        write(directory / "native_initial_receipt.json", initial)
        controller.update(
            tp1_total_kv_blocks=initial[0]["total_blocks"],
            tp4_total_kv_blocks=initial[1]["total_blocks"],
        )
        write(directory / "controller_config.json", controller)
        print(
            f"[场景] {name} 开始原策略分流及迁移旁路，共{len(values)}条请求", flush=True
        )
        code = policy_runner.main(
            child_arguments(args, plan, directory, directory / "arrivals.json")
        )
        if code:
            raise RuntimeError(f"original policy runner returned {code}")
        final = wait_empty(
            plan["run_dir"], after_generations=[r["generation"] for r in initial]
        )
        write(directory / "native_empty_receipt.json", final)
        if any(a["free_blocks"] != b["free_blocks"] for a, b in zip(initial, final)):
            raise ValueError("native KV did not return to initial state")
        summary = obs.summarize(plan["run_dir"])
        if name == "natural_long_pressure" and not summary["available_point_estimates"]:
            raise ValueError(
                "natural traffic produced no consumed finite62 mean estimate"
            )
        events = plan["run_dir"] / "predictor_events.jsonl"
        predictions = [json.loads(line) for line in events.read_text().splitlines()]
        if not any(r.get("kind") == "predictor_prediction" for r in predictions):
            raise ValueError("no real predictor events")
        summary["predictor_events"] = sum(
            r.get("kind") == "predictor_prediction" for r in predictions
        )
        write(directory / "observation_summary.json", summary)
        return summary
    finally:
        common.stop_processes(owned)
        proof = dict(
            all_owned_groups_ended=not any(common.process_group_alive(p) for p in owned)
        )
        write(directory / "owned_shutdown_receipt.json", proof)
        if not proof["all_owned_groups_ended"]:
            raise RuntimeError("owned service cleanup incomplete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "checkpoint", "inputs", "prior-cleanup", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    replay = obs.reactions()
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="CPU_PLAN_ONLY",
                    replay=replay,
                    scenes=obs.scenes(),
                    original_policy_reused=True,
                    routing_execution=True,
                    migration_execution=False,
                    maximum_live_scenes=4,
                    maximum_scene_workload_s=480,
                    profiles=obs.profiles(),
                ),
                indent=2,
            )
        )
        return 0
    require_revision(args.expected_revision)
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        raise ValueError("results must be outside checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    result, ledger, code = dict(status="TECHNICAL_FAILURE"), [], 1
    try:
        previous = read(args.prior_cleanup / "result.json")
        if (
            previous.get("status") != "COMPLETED"
            or previous.get("total_profile_attempts") != 27
        ):
            raise ValueError("requires the completed cleanup validation")
        prior_identity = read(args.prior_cleanup / "identity.json")
        gpu, versions = runtime_inventory(read(PROFILE)), runtime_versions()
        write(
            args.output / "identity.json",
            dict(
                revision=args.expected_revision,
                gpu=gpu,
                versions=versions,
                checkpoint_sha256=CHECKPOINT_SHA,
            ),
        )
        if (
            gpu_configuration(gpu["gpus"])
            != gpu_configuration(prior_identity["gpu"]["gpus"])
            or json.loads(json.dumps(versions)) != prior_identity["versions"]
        ):
            raise ValueError(
                "GPU capabilities/driver/runtime differ; UUID not compared"
            )
        inputs = verified_inputs(args.model, args.inputs)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        write(args.output / "reaction_replay.json", replay)
        write(args.output / "scope.json", read(obs.ASSETS / "scope.json"))
        for name in obs.scenes():
            if not gpu_idle():
                raise RuntimeError("GPUs busy; unrelated processes are not stopped")
            row = dict(name=name, status="STARTED")
            ledger.append(row)
            write(args.output / "attempt_ledger.json", ledger)
            row.update(
                status="COMPLETED", observation=run_scene(args, name, inputs, tokenizer)
            )
            write(args.output / "attempt_ledger.json", ledger)
            if not gpu_idle():
                raise RuntimeError("GPU cleanup incomplete")
        result, code = dict(status="COMPLETED_OBSERVATION_ONLY"), 0
    except KeyboardInterrupt:
        result, code = dict(status="INTERRUPTED"), 130
    except Exception as error:
        result["error"] = repr(error)
        print(f"[错误] {error!r}", flush=True)
    finally:
        if ledger and ledger[-1]["status"] == "STARTED":
            ledger[-1]["status"] = "FAILED"
        result.update(
            runner_code=code,
            scenes=ledger,
            previous_profile_attempts=27,
            total_profile_attempts=27 + len(ledger),
            migration_attempts_added=0,
            observation_scenes_attempted=len(ledger),
            maximum_observation_scenes=4,
            migration_execution=False,
            deployment_ready=False,
            general_guard_blocks=None,
            general_release_bound_s=None,
        )
        write(args.output / "result.json", result)
        archive = Path(str(args.output) + ".tar.gz")
        receipt = pack(args.output, archive)
        Path(str(archive) + ".sha256").write_text(
            receipt["sha256"] + "  " + archive.name + "\n"
        )
        print(f"[结果] {result['status']}；请拿回 {archive}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
