# SPDX-License-Identifier: Apache-2.0
"""Bounded Qwen3/4090D service calibration; no predictor or migration policy."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import run_pressure_calibration as collect  # noqa: E402
from tools.bridge_tp.prepare_4090d_qwen3 import (  # noqa: E402
    PROFILE,
    inspect_model,
    read,
    runtime_inventory,
    sha,
)
from tools.bridge_tp.result_archive import pack  # noqa: E402
from tools.bridge_tp.summarize_tp2_calibration import summarize  # noqa: E402


def matrix():
    """Five shapes twice; reserve two of twelve profile units for migration."""
    scenes = [
        dict(
            name=f"qwen3-nd{target}",
            prompt_tokens=2048,
            max_tokens=6144,
            source_count=source,
            target_count=target,
            headroom_blocks=125,
        )
        for source, target in ((1, 2), (2, 4), (4, 8))
    ]
    # Spend the highest-Nd cell on the previously unmeasured >8K TP2 domain.
    # Low-context Nd8 and low-context source Nd4 remain outside this sparse grid.
    scenes[-1].update(
        name="qwen3-nd8-long",
        prompt_tokens=6144,
        max_tokens=1024,
        target_max_tokens=8192,
    )
    scenes.extend(
        dict(
            name=f"qwen3-prefill-{size}",
            prompt_tokens=2048,
            max_tokens=4096,
            target_max_tokens=6144,
            source_count=1,
            target_count=2,
            headroom_blocks=125,
            injection_prompt_tokens=[size],
            injections_per_size=5,
            injection_pools=["source", "target"],
            settle_s=2,
            recovery_s=1,
        )
        for size in (512, 4096)
    )
    return scenes


def prepare_inputs(tokenizer):
    """Use this model's non-thinking chat template and exact controlled lengths."""
    marker = "BRIDGETP_CALIBRATION_CONTENT"
    template = tokenizer.apply_chat_template(
        [dict(role="user", content=marker)],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    left, right = template.split(marker)
    prefix = tokenizer.encode(left, add_special_tokens=False)
    suffix = tokenizer.encode(right, add_special_tokens=False)
    body = tokenizer.encode(
        "Explain queue scheduling, memory accounting, and backpressure. " * 6000,
        add_special_tokens=False,
    )
    requests = {}
    for length in (2048, 6144):
        count = length - len(prefix) - len(suffix)
        if count <= 0 or count > len(body):
            raise ValueError("tokenizer cannot satisfy the registered input budget")
        requests[str(length)] = dict(
            model="bridgetp-model",
            prompt=prefix + body[:count] + suffix,
            temperature=0.0,
            top_p=1.0,
            stream=True,
        )
    return dict(
        schema="QWEN3_SERVICE_CALIBRATION_V1",
        requests=requests,
        context_limits=dict(source=8192, target=16384),
        seed=20261028,
        thinking_mode="disabled",
        output_contract="fixed length, ignore_eos=True",
        purpose="system service profiling, not natural generation quality",
    )


def validate_inputs(config):
    for scene in matrix():
        for role in ("source", "target"):
            collect.payload(
                config,
                scene["prompt_tokens"],
                scene.get(role + "_max_tokens", scene["max_tokens"]),
                "validate",
                context_limit=config["context_limits"][role],
            )
        for size in scene.get("injection_prompt_tokens", []):
            collect.payload(config, size, 16, "validate")


def accepted_scene(directory):
    """Only reuse fully closed evidence; failed attempts remain in the archive."""
    from tools.bridge_tp.verify_performance_trace import verify_pool

    result = read(directory / "request_results.json")
    if result["status"] != "COMPLETED" or any(
        r["error"] or not r["done"] or r["tokens"] != r["max_tokens"]
        for r in result["requests"]
    ):
        raise ValueError("scene request completion differs")
    if not read(directory / "owned_shutdown_receipt.json")["all_owned_groups_ended"]:
        raise ValueError("scene owned processes did not end")
    for role, ranks in (("source", 1), ("target", 2)):
        verify_pool(directory / "controller/performance_trace" / role, ranks)
    return result


def service_facts(output):
    """Startup/empty native facts are not a bundle rebuild or safe KV guard."""
    facts = []
    for directory in output.iterdir():
        if not (directory / "native_initial_receipt.json").is_file():
            continue
        initial = read(directory / "native_initial_receipt.json")
        lifecycle = {r["pool"]: r for r in read(directory / "lifecycle_events.json")}
        facts.append(
            dict(
                scene=directory.name,
                pools={
                    role: dict(
                        total_blocks=inv["total_blocks"],
                        free_blocks=inv["free_blocks"],
                        block_size=inv["block_size"],
                        context_limit=inv["context_limit"],
                        load_to_health_s=event["healthy_unix_s"]
                        - event["load_started_unix_s"],
                        load_to_warm_ready_s=event["warm_ready_unix_s"]
                        - event["load_started_unix_s"],
                    )
                    for role, inv, event in (
                        (role, inv, lifecycle[role])
                        for role, inv in zip(("source", "target"), initial)
                    )
                },
            )
        )
    return facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--phase", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prior-service-attempts", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.prior_service_attempts <= 10:
        raise ValueError("prior service attempts must be between zero and ten")
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="STATIC_PLAN_ONLY",
                    shapes=matrix(),
                    maximum_profile_units=12,
                    service_profile_units=10,
                    reserved_migration_units=2,
                    source_gpu="2",
                    target_gpus="0,1",
                    reserved_gpu="3",
                    target_tp=2,
                    predictor_loaded=False,
                    migration_enabled=False,
                ),
                indent=2,
            )
        )
        return 0
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if (
        revision != args.expected_revision
        or subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
        ).strip()
    ):
        raise ValueError("requires the expected clean code revision")
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        raise ValueError("results must be outside the source checkout")
    if args.resume:
        old = read(args.output / "preflight.json")
        if (
            old["revision"] != revision
            or old["model"] != str(args.model.resolve())
            or old.get("prior_service_attempts", 0) != args.prior_service_attempts
        ):
            raise ValueError("resume code/model identity differs")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    archive = Path(
        str(args.output)
        + (f".resume_{time.time_ns()}" if args.resume else "")
        + ".tar.gz"
    )
    result, code = dict(status="TECHNICAL_FAILURE"), 1
    try:
        hardware = runtime_inventory(read(PROFILE))
        if [g["index"] for g in hardware["gpus"]] != ["0", "1", "2", "3"]:
            raise ValueError("requires physical indices 0,1,2,3")
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip():
            raise RuntimeError("GPUs busy; no unrelated process is stopped")
        identity = inspect_model(args.model, read(PROFILE))
        expected = read(ROOT / "experiments/platforms/qwen3_8b_verified_assets.json")
        if {k: v["sha256"] for k, v in identity["files"].items()} != expected["sha256"]:
            raise ValueError("model asset SHA differs from the verified baseline")
        import torch

        versions = dict(
            python=sys.version,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            nccl=torch.cuda.nccl.version(),
            backend_environment={
                k: v
                for k, v in os.environ.items()
                if k.startswith(("NCCL_", "UCX_", "GLOO_"))
            },
        )
        if args.resume and (
            old["platform_sha256"] != sha(PROFILE)
            or old["versions"] != json.loads(json.dumps(versions))
            or read(args.output / "hardware.json")["gpus"] != hardware["gpus"]
        ):
            raise ValueError("resume GPU/driver/runtime/platform identity differs")
        from transformers import AutoTokenizer

        config = prepare_inputs(
            AutoTokenizer.from_pretrained(
                args.model,
                local_files_only=True,
            )
        )
        validate_inputs(config)
        raw = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        if args.resume:
            if (
                old["input_sha256"] != digest
                or sha(args.output / "inputs.json") != digest
            ):
                raise ValueError("resume tokenizer/input differs")
        else:
            (args.output / "inputs.json").write_bytes(raw)
        from tools.bridge_tp import run_phase9_cap0_calibration as common

        common.write_json(args.output / "hardware.json", hardware)
        common.write_json(args.output / "model_identity.json", identity)
        common.write_json(
            args.output / "preflight.json",
            dict(
                revision=revision,
                model=str(args.model.resolve()),
                input_sha256=digest,
                platform_sha256=sha(PROFILE),
                full_matrix=matrix(),
                maximum_profile_units=12,
                service_profile_units=10,
                reserved_migration_units=2,
                engine_mode="V1/eager",
                source_gpu="2",
                target_gpus="0,1",
                predictor_loaded=False,
                migration_enabled=False,
                versions=versions,
                capacity_guard_blocks=None,
                experiment_shape_headroom_blocks=125,
                prior_service_attempts=args.prior_service_attempts,
            ),
        )
        settings = SimpleNamespace(
            python_bin=Path(sys.executable),
            model_path=args.model,
            dtype="bfloat16",
            max_model_len=8192,
            tp4_max_model_len=16384,
            gpu_memory_utilization=0.88,
            tp1_gpu="2",
            tp4_gpus="0,1",
            tp1_port=8610,
            tp4_port=8611,
            target_tp_size=2,
        )
        spec = dict(service_only=True, startup_timeout_s=900, run_timeout_s=1200)
        selected = matrix()[:1] if args.phase == "pilot" else matrix()
        common.write_json(
            args.output / "selection.json",
            dict(
                target_tp=2,
                platform="4090d24-qwen3-8b",
                phase=args.phase,
                scenes=selected,
                repetitions=2,
                policy_enabled=False,
                protocol="service_only_no_transport_buffers",
            ),
        )
        runs = []
        ledger_path = args.output / "attempt_ledger.json"
        attempts = read(ledger_path) if ledger_path.exists() else []
        for repetition in (1, 2):
            order = selected if repetition == 1 else list(reversed(selected))
            for scene in order:
                directory = args.output / f"{scene['name']}-r{repetition}"
                if not directory.resolve().is_relative_to(args.output):
                    raise ValueError("scene directory escapes the batch")
                if directory.exists():
                    try:
                        if read(directory / "recipe.json") != scene:
                            raise ValueError("scene recipe differs")
                        accepted_scene(directory)
                        print(f"[进度] 复用完整场次 {directory.name}", flush=True)
                        runs.append(dict(scene=directory.name, reused=True))
                        continue
                    except Exception:
                        if not args.resume:
                            raise
                        preserved = args.output / "incomplete"
                        preserved.mkdir(exist_ok=True)
                        directory.rename(
                            preserved / f"{directory.name}_{time.time_ns()}"
                        )
                print(f"[进度] 开始 {directory.name}", flush=True)
                if len(attempts) + args.prior_service_attempts >= 10:
                    raise RuntimeError("ten service attempts spent, including failures")
                attempts.append(dict(scene=directory.name, status="STARTED"))
                common.write_json(ledger_path, attempts)
                row = collect.run_scene(
                    spec,
                    config,
                    scene,
                    repetition,
                    directory,
                    common,
                    settings,
                    True,
                )
                accepted_scene(directory)
                attempts[-1]["status"] = "COMPLETED"
                common.write_json(ledger_path, attempts)
                runs.append(dict(scene=directory.name, reused=False, **row))
        profile = summarize(args.output)
        profile.update(
            platform="4090d24-qwen3-8b",
            engine_mode="V1/eager",
            deployment_ready=False,
            native_capacity_and_startup=service_facts(args.output),
            capacity_guard_blocks=None,
            capacity_lookahead_s=None,
            predictor_checkpoint_activated=False,
            peak_scope="measured sparse grid only, not maximum service capacity",
        )
        common.write_json(args.output / "service_profile.json", profile)
        result = dict(
            status="COMPLETED",
            phase=args.phase,
            runs=runs,
            service_profile="service_profile.json",
            deployment_ready=False,
            next="review pilot before --phase full --resume"
            if args.phase == "pilot"
            else "review coverage before transport/migration/rebuild calibration",
        )
        code = 0
    except KeyboardInterrupt:
        result.update(status="INTERRUPTED")
        code = 130
    except Exception as error:
        result.update(status="TECHNICAL_FAILURE", error=repr(error))
        print(f"[错误] {error!r}", flush=True)
    finally:
        if (args.output / "result.json").exists():
            (args.output / f"result_previous_{time.time_ns()}.json").write_bytes(
                (args.output / "result.json").read_bytes()
            )
        result.update(runner_code=code, predictor_training_started=False)
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        receipt = pack(args.output, archive)
        Path(str(archive) + ".sha256").write_text(
            receipt["sha256"] + "  " + archive.name + "\n"
        )
        print(json.dumps(receipt), flush=True)
        print(f"[结果] {result['status']}；完整证据包：{archive}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
