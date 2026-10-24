# SPDX-License-Identifier: Apache-2.0
"""TP2 development service collection using the existing instrumented runner.

Six decode runs (Nd=2/4/8 twice) and four prefill runs form a bounded first
batch. Two remaining profile units are reserved for migration/cleanup pilots.
No TP4 target profile is loaded into an online TP2 policy.
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import launch_pressure_balanced_pilot as pilot  # noqa: E402
from tools.bridge_tp import run_pressure_calibration as collect  # noqa: E402


def matrix():
    scenes = [
        dict(
            name=f"tp2-nd{n}",
            prompt_tokens=2048,
            max_tokens=6144,
            source_count=1,
            target_count=n,
            headroom_blocks=125,
        )
        for n in (2, 4, 8)
    ]
    scenes += [
        dict(
            name=f"tp2-prefill-{size}",
            prompt_tokens=2048,
            max_tokens=4096,
            target_max_tokens=6144,
            source_count=1,
            target_count=2,
            headroom_blocks=125,
            injection_prompt_tokens=[size],
            injections_per_size=5,
            injection_pools=["target"],
            settle_s=2,
            recovery_s=1,
        )
        for size in (512, 2048)
    ]
    return scenes


def prepare(model_path, source_gpu="2", target_gpus="0,1"):
    spec = pilot.read(ROOT / "experiments/phase9/pressure_balanced_v2/pilot_spec.json")
    # Keep frozen identities/settings, but do not pass its TP4 service profile to
    # any decision loop. run_scene starts neither a controller nor a stager.
    spec = copy.deepcopy(spec)
    for key in (
        "policy_config",
        "service_evidence",
        "controller_config_template",
        "checkpoint",
    ):
        spec[key] = str(ROOT / spec[key])
    for scene in spec["scenarios"]:
        scene["arrivals"] = str(ROOT / scene["arrivals"])
    spec["model_path"] = str(model_path)
    spec["expected_branch"] = "bridgetp/tp2-reconfiguration"
    spec["server_settings"].update(
        target_tp_size=2,
        tp1_gpu=source_gpu,
        tp4_gpus=target_gpus,
        tp4_max_model_len=16384,
    )
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-gpu", default="2")
    parser.add_argument("--target-gpus", default="0,1")
    parser.add_argument("--scene", choices=[s["name"] for s in matrix()])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    selected = [s for s in matrix() if args.scene is None or s["name"] == args.scene]
    spec = prepare(args.model_path, args.source_gpu, args.target_gpus)
    inputs = pilot.read(
        ROOT / "experiments/phase9/pressure_balanced_v2/calibration.json"
    )
    for scene in selected:
        for role in ("source", "target"):
            collect.payload(
                inputs,
                scene["prompt_tokens"],
                scene.get(role + "_max_tokens", scene["max_tokens"]),
                "validate",
            )
    gpus = args.target_gpus.split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or args.source_gpu in gpus:
        raise ValueError("one source and two distinct target GPUs required")
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="VALIDATED_INPUTS_ONLY",
                    target_tp=2,
                    scenes=selected,
                    repetitions=2,
                    profile_units=len(selected) * 2,
                ),
                indent=2,
            )
        )
        return 0
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(names) != 4 or any(n.strip() != "NVIDIA A100-PCIE-40GB" for n in names):
        raise RuntimeError("this development contract requires four A100 40GB")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPU processes exist; nothing is automatically stopped")
    if pilot.sha(args.model_path / "config.json") != spec["model_config_sha256"]:
        raise ValueError("model config differs from frozen Qwen baseline")
    if pilot.sha(spec["checkpoint"]) != spec["checkpoint_sha256"]:
        raise ValueError("predictor checkpoint differs")
    common, settings = pilot.arguments(spec)
    args.out_root.mkdir(parents=True, exist_ok=False)
    common.write_json(
        args.out_root / "selection.json",
        dict(
            target_tp=2,
            purpose="development profiles",
            objective="NO_POLICY",
            source_gpu=args.source_gpu,
            target_gpus=gpus,
            scenes=selected,
            repetitions=2,
            remaining_profile_budget=2,
        ),
    )
    common.write_json(args.out_root / "effective_spec.json", spec)
    (args.out_root / "git_status.txt").write_bytes(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
    )
    (args.out_root / "git.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
    )
    from tools.bridge_tp.freeze_pressure_baseline import cloud_identity, freeze

    freeze(args.out_root / "source_identity")
    common.write_json(
        args.out_root / "runtime_identity.json",
        cloud_identity(args.model_path, spec["checkpoint"]),
    )
    for filename, command in (
        ("gpu_inventory.txt", ["nvidia-smi", "-L"]),
        ("gpu_topology.txt", ["nvidia-smi", "topo", "-m"]),
    ):
        (args.out_root / filename).write_bytes(subprocess.check_output(command))
    results = []
    code = 0
    try:
        for repetition in (1, 2):
            order = selected if repetition == 1 else list(reversed(selected))
            for scene in order:
                root = args.out_root / f"{scene['name']}-r{repetition}"
                result = collect.run_scene(
                    spec, inputs, scene, repetition, root, common, settings, True
                )
                results.append(
                    dict(scene=scene["name"], repetition=repetition, **result)
                )
                if result["status"] != "COMPLETED":
                    code = 1
                    return code
    except Exception as error:
        results.append(dict(status="TECHNICAL_FAILURE", error=repr(error)))
        code = 1
    finally:
        common.write_json(
            args.out_root / "batch_result.json", dict(runner_code=code, runs=results)
        )
        import tarfile

        with tarfile.open(str(args.out_root) + ".tar.gz", "w:gz") as archive:
            archive.add(args.out_root, arcname=args.out_root.name)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
