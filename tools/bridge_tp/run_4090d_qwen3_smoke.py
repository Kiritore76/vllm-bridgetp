# SPDX-License-Identifier: Apache-2.0
"""Real Qwen3 deployment smoke, no predictor, pressure policy or KV migration."""

import argparse
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

from tools.bridge_tp.prepare_4090d_qwen3 import (  # noqa: E402
    PROFILE,
    inspect_model,
    read,
    runtime_inventory,
    validate_config,
)
from tools.bridge_tp.pressure_cpu import load  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    profile = read(PROFILE)
    geometry = validate_config(read(args.model / "config.json"), profile)
    prompts = read(args.inputs / "smoke_prompts.json")
    if not prompts or any(
        not p["token_ids"] or p["enable_thinking"] is not False
        for p in prompts.values()
    ):
        raise ValueError("new tokenizer/non-thinking smoke inputs required")
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="INPUT_VALIDATED",
                    tp=args.tp,
                    kv_bytes_per_token_tp1=geometry,
                    old_predictor_loaded=False,
                    GPU_execution=False,
                )
            )
        )
        return 0
    from tools.bridge_tp import run_phase9_cap0_calibration as common
    from tools.bridge_tp.run_atomic_takeover import _post_json
    from tools.bridge_tp.run_pressure_calibration import latest_inventory
    from tools.bridge_tp.verify_performance_trace import verify_pool

    hardware = runtime_inventory(profile)
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPU computation exists; no unrelated process is stopped")
    args.output.mkdir(parents=True, exist_ok=False)
    common.write_json(args.output / "hardware.json", hardware)
    common.write_json(
        args.output / "model_identity.json", inspect_model(args.model, profile)
    )
    settings = SimpleNamespace(
        python_bin=Path(sys.executable),
        model_path=args.model,
        dtype="bfloat16",
        max_model_len=profile["service_max_model_len"]["tp1"],
        tp4_max_model_len=profile["service_max_model_len"]["tp2"],
        gpu_memory_utilization=profile["gpu_memory_utilization"],
    )
    command = common.server_command(settings, args.tp, args.port) + [
        "--enforce-eager",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "2048",
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith("BRIDGETP_")}
    token = secrets.token_hex(32)
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(args.tp)),
        OMP_NUM_THREADS="1",
        BRIDGETP_PRESSURE_POOL="tp1" if args.tp == 1 else "tp4",
        BRIDGETP_PRESSURE_INVENTORY_PATH=str(args.output / "inventory.jsonl"),
        BRIDGETP_PERFORMANCE_TRACE_DIR=str(args.output / "performance_trace"),
        BRIDGETP_PERFORMANCE_FINALIZE_TOKEN=token,
        BRIDGETP_PERFORMANCE_SAMPLE_EVERY="8",
    )
    common.write_json(args.output / "argv.json", command)
    common.write_json(
        args.output / "environment.json",
        {
            k: v
            for k, v in env.items()
            if k.startswith(("BRIDGETP_", "CUDA_", "OMP_")) and not k.endswith("TOKEN")
        },
    )
    process, code, result = None, 0, {}
    try:
        print(f"[部署检查] 启动 Qwen3-8B TP{args.tp}", flush=True)
        started = time.time()
        process = common.start_process(
            "deployment", command, env, args.output / "server.log"
        )
        url = f"http://127.0.0.1:{args.port}"
        common.wait_healthy(url, process, 900)
        ready = time.time()
        io = load("online_io")
        responses = []
        for name, prompt in prompts.items():
            if prompt.get("only_tp", args.tp) != args.tp:
                continue
            print(
                f"[部署检查] {name}: 输入 {len(prompt['token_ids'])} tokens", flush=True
            )
            rows = []
            response = io.post_streaming_completion(
                url,
                dict(
                    model="bridgetp-model",
                    prompt=prompt["token_ids"],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=32,
                    ignore_eos=False,
                    stream=True,
                    return_token_ids=True,
                    request_id=f"4090d-qwen3-{name}",
                ),
                180,
                lambda index, token_id, stamp, rows=rows: rows.append(
                    dict(index=index, token_id=token_id, unix_s=stamp)
                ),
                args.output / (name + ".failure.json"),
                retain_chunks=False,
            )
            if not response["token_ids"] or response["finish_reason"] not in {
                "stop",
                "length",
            }:
                raise RuntimeError("no real model output or invalid finish")
            common.write_json(args.output / (name + ".response.json"), response)
            common.write_json(args.output / (name + ".tokens.json"), rows)
            responses.append(
                dict(
                    name=name,
                    tokens=len(response["token_ids"]),
                    finish_reason=response["finish_reason"],
                )
            )
        finalization = _post_json(
            url, "/bridge_tp/v1/performance_trace/finalize", {"token": token}, 60
        )
        if finalization.get("status") != "FINALIZED":
            raise RuntimeError("performance trace did not finalize")
        trace = verify_pool(args.output / "performance_trace", args.tp)
        native = None
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            native = latest_inventory(args.output / "inventory.jsonl")
            if native and not (
                native["requests"] or native["running"] or native["waiting"]
            ):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("native inventory did not end empty")
        result = dict(
            status="COMPLETED",
            tp=args.tp,
            dtype="bfloat16",
            mode="deployment_eager_only",
            max_model_len=profile["service_max_model_len"][f"tp{args.tp}"],
            load_to_health_s=ready - started,
            responses=responses,
            inventory=native,
            trace=trace,
            migration_validated=False,
            predictor_validated=False,
            performance_profile_validated=False,
        )
    except Exception as error:
        result, code = dict(status="TECHNICAL_FAILURE", error=repr(error)), 1
    finally:
        if process:
            common.stop_processes([process])
            if common.process_group_alive(process):
                code, result["status"] = 1, "TECHNICAL_FAILURE"
                result["cleanup_error"] = "owned process group remains"
        result["runner_code"] = code
        common.write_json(args.output / "result.json", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
