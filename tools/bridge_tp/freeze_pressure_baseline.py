# SPDX-License-Identifier: Apache-2.0
"""Record exact local baseline inputs without importing CUDA or changing Git."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path("experiments/phase9/pressure_balanced_v2")


def cloud_identity(model_path, checkpoint_path):
    """Capture installed versions and vetted predictor metadata before serving."""
    import importlib.metadata

    import torch

    model_files = {}
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        path = Path(model_path) / name
        model_files[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    fields = (
        "format_version",
        "model_type",
        "feature_layer",
        "hidden_width",
        "hidden_widths",
        "input_width",
        "network_depth",
        "temperature",
        "position_log_scale",
        "overflow_category",
        "model_config_sha256",
    )
    predictor = {key: weights.get(key) for key in fields}
    predictor["category_upper_edges"] = weights["category_upper_edges"].tolist()
    predictor["sha256"] = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = None
    try:
        nccl = torch.cuda.nccl.version()
    except Exception:
        nccl = None
    return dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nccl=nccl,
        vllm=vllm_version,
        predictor=predictor,
        model_files=model_files,
    )


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def freeze(destination):
    destination.mkdir(parents=True, exist_ok=False)
    paths = (
        git(
            "ls-files",
            "-z",
            "vllm/bridge_tp",
            "tools/bridge_tp",
            "tests/bridge_tp",
            str(BUNDLE),
        )
        .decode()
        .split("\0")
    )
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in paths
        if name and (ROOT / name).is_file()
    }
    sources = {}
    for name in (
        "health_service_policy.json",
        "controller_template.json",
        "pilot_spec.json",
        "health_reference_v2.json",
        "service_evidence.json",
        "comparison.json",
    ):
        source = ROOT / BUNDLE / name
        sources[name] = json.loads(source.read_text(encoding="utf-8"))
        (destination / name).write_bytes(source.read_bytes())
    # Preserve byte-exact patches and identities, including pre-existing work.
    (destination / "tracked.patch").write_bytes(git("diff", "HEAD", "--binary"))
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    identities = {}
    for name in untracked.decode().split("\0"):
        if name and (ROOT / name).is_file():
            identities[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            source = ROOT / name
            if (
                name.startswith(
                    ("tools/bridge_tp/", "vllm/bridge_tp/", "tests/bridge_tp/")
                )
                and source.suffix in {".py", ".json", ".md"}
                and source.stat().st_size < 2 * 1024 * 1024
            ):
                copy = destination / "untracked_source" / name
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_bytes(source.read_bytes())
    receipt = dict(
        recorded_at=datetime.now(timezone.utc).isoformat(),
        repository=str(ROOT),
        head=git("rev-parse", "HEAD").decode().strip(),
        branch=git("branch", "--show-current").decode().strip(),
        status=git("status", "--porcelain").decode(),
        file_sha256=hashes,
        untracked_sha256=identities,
        configuration_sources=sources,
        cloud_unverified=[
            "GPU UUID/topology",
            "model/tokenizer actual files",
            "torch/CUDA/NCCL",
            "compile/CUDA Graph actual settings",
            "engine arguments and allocated KV blocks",
        ],
    )
    (destination / "baseline.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.output)
    print(
        json.dumps(
            {
                "head": result["head"],
                "files": len(result["file_sha256"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )
