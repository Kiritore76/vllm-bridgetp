# SPDX-License-Identifier: Apache-2.0
"""Identity and engine settings shared by bounded Qwen3 calibration stages."""

import os
import sys
from pathlib import Path

from tools.bridge_tp.prepare_4090d_qwen3 import PROFILE, read, sha, validate_config

ROOT = Path(__file__).resolve().parents[2]
INPUT_SHA = "53e4f904db5c9f94f672fa1b6bf3a008d4b2b02c98df5f06a59649529f671aec"


def verified_inputs(model, path):
    assets = read(ROOT / "experiments/platforms/qwen3_8b_verified_assets.json")[
        "sha256"
    ]
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if sha(Path(model) / name) != assets[name]:
            raise ValueError(f"Qwen3 identity differs: {name}")
    validate_config(read(Path(model) / "config.json"), read(PROFILE))
    if path is None or sha(path) != INPUT_SHA:
        raise ValueError("requires the audited Qwen3 service inputs")
    return read(path)


def require_revision(expected):
    import subprocess

    if (
        not expected
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != expected
    ):
        raise ValueError("code revision differs")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("requires a clean checkout")


def engine_environment(environment):
    environment.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        VLLM_USE_V2_MODEL_RUNNER="0",
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )


def engine_arguments():
    return [
        "--enforce-eager",
        "--max-num-seqs",
        "16",
        "--max-num-batched-tokens",
        "2048",
        "--block-size",
        "16",
    ]


def expanded_context_spec(spec, model_config):
    """Opt in to native 32K limits in both pools; preserve frozen profiles."""
    validate_config(model_config, read(PROFILE))
    if (
        model_config.get("max_position_embeddings", 0) < 32768
        or spec["server_settings"].get("target_tp_size") != 2
    ):
        raise ValueError("expanded context requires native Qwen3 TP1/TP2")
    spec["context_profile"] = "QWEN3_NATIVE_32K_BOTH_V1"
    spec["server_settings"].update(max_model_len=32768, tp4_max_model_len=32768)


def migration_spec(model):
    """Reuse transport configuration, never its old model/predictor evidence."""
    from tools.bridge_tp.run_tp2_calibration import prepare

    spec = prepare(model)
    spec["model_config_sha256"] = sha(Path(model) / "config.json")
    spec["checkpoint"] = ""
    spec["checkpoint_sha256"] = ""
    spec["calibration_identity"] = "Qwen3 mechanism-only; no predictor/policy loaded"
    return spec


def adjust_migration_plan(plan):
    for role in ("source", "target"):
        environment = plan[role + "_env"]
        for key in list(environment):
            if key.startswith("BRIDGETP_PREDICTOR"):
                environment.pop(key)
        engine_environment(environment)
        plan[role + "_argv"].extend(engine_arguments())
    # Old capacity/byte constants are replaced with native facts before arming.
    plan["config"]["policy"]["kv_bytes_per_token"] = 147456


def runtime_versions():
    import torch

    try:
        nccl = torch.cuda.nccl.version()
    except Exception:
        nccl = None
    return dict(
        python=sys.version,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        nccl=nccl,
        backend_environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("NCCL_", "UCX_", "GLOO_"))
        },
    )
