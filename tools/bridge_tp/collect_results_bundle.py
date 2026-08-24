#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package one experiment's runs into a single archive to hand back for analysis.

Collects exactly the artifacts needed to re-derive every claim offline, and
nothing else. KV tensors and model weights are never included -- the bundle is
evidence, not data.

    python tools/bridge_tp/collect_results_bundle.py \
        --experiment E-1 \
        --runs runs/e1_seed0 runs/e1_seed1 runs/e1_seed2 \
        --calibration calibration/ \
        --config experiments/phase9/configs/e1_correctness.json \
        --out bundles/

What lands in the bundle, per run:

    phase9_audit.jsonl          every decision, transition, and rate change
    takeover_state.json         the Phase 7 terminal state
    response_proxy_stats.json   client-visible stream statistics
    source_progress.json        the computed/pending boundary the policy saw
    session_manifest.json       session binding
    staging_manifest.json       Phase 8 staging record, when present
    sender_receipts/            four-rank send evidence
    stage_delivery_receipts/    four-rank staged delivery evidence (Phase 8)
    receiver_receipts/          four-rank exact-readback evidence
    control_tokens.json         the clean-TP1 control the run is compared to
    inspect.json                the pass/fail report for this run

Plus, once for the whole bundle: the config used, the calibration inputs, an
environment capture, and a manifest with SHA256 of every collected file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Per-run files copied verbatim when present.
RUN_FILES = (
    "phase9_audit.jsonl",
    "takeover_state.json",
    "response_proxy_stats.json",
    "unified_response.jsonl",
    "source_progress.json",
    "source_request.json",
    "source_response.json",
    "target_request.json",
    "target_response.json",
    "session_manifest.json",
    "staging_manifest.json",
    "cutover_manifest.json",
    "runtime_control.json",
    "runtime_control_honored",
    "control_tokens.json",
    "cleanup_request.json",
    "inspect.json",
)
RUN_DIRS = (
    "sender_receipts",
    "initial_stage_receipts",
    "delta_sender_receipts",
    "stage_delivery_receipts",
    "receiver_receipts",
)
# Anything matching these is refused outright: the bundle must stay reviewable.
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".npy", ".npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", required=True, help="plan ID, e.g. E-1 / E-3 / C-2"
    )
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="output of summarize_phase9_runs.py, if produced",
    )
    parser.add_argument(
        "--note", default="", help="anything the artifacts cannot say for themselves"
    )
    parser.add_argument("--out", type=Path, default=Path("bundles"))
    parser.add_argument("--max-mb", type=float, default=200.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_environment() -> dict:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(run(["git", "status", "--porcelain"])),
        "nvidia_smi": run(["nvidia-smi"]),
        "bridgetp_env": {
            k: v for k, v in os.environ.items() if k.startswith("BRIDGETP_")
        },
    }
    try:
        import torch  # type: ignore

        env["torch"] = torch.__version__
        env["cuda"] = getattr(torch.version, "cuda", None)
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception:  # noqa: BLE001 - torch is optional for offline packaging
        env["torch"] = None
    try:
        import vllm  # type: ignore

        env["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001
        env["vllm"] = None
    return env


def copy_tree(src: Path, dst: Path, collected: list[Path]) -> None:
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        collected.append(target)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / f"bridgetp_{args.experiment.replace('/', '_')}"
        root.mkdir(parents=True)
        collected: list[Path] = []
        run_report: list[dict] = []

        for run in args.runs:
            if not run.is_dir():
                print(f"  WARNING: {run} is not a directory, skipped")
                continue
            dst = root / "runs" / run.name
            dst.mkdir(parents=True, exist_ok=True)
            present, missing = [], []
            for name in RUN_FILES:
                source = run / name
                if source.is_file():
                    shutil.copy2(source, dst / name)
                    collected.append(dst / name)
                    present.append(name)
                else:
                    missing.append(name)
            for name in RUN_DIRS:
                source = run / name
                if source.is_dir():
                    copy_tree(source, dst / name, collected)
                    present.append(name + "/")
                else:
                    missing.append(name + "/")
            run_report.append({"run": str(run), "present": present, "missing": missing})
            print(f"  {run.name}: {len(present)} artifacts")

        if args.config and args.config.is_file():
            shutil.copy2(args.config, root / "config.json")
            collected.append(root / "config.json")
        if args.summary and args.summary.is_file():
            shutil.copy2(args.summary, root / "summary.json")
            collected.append(root / "summary.json")
        if args.calibration and args.calibration.is_dir():
            copy_tree(args.calibration, root / "calibration", collected)

        manifest = {
            "format_version": 1,
            "experiment": args.experiment,
            "note": args.note,
            "runs": run_report,
            "environment": capture_environment(),
            "files": {
                str(p.relative_to(root)): {
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
                for p in sorted(collected)
            },
        }
        (root / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        archive = args.out / f"{root.name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(root, arcname=root.name)

    size_mb = archive.stat().st_size / 1024 / 1024
    print(f"\nwrote {archive}  ({size_mb:.1f} MB)")
    print(f"  runs: {len(run_report)}   files: {len(manifest['files'])}")
    if size_mb > args.max_mb:
        print(
            f"  WARNING: bundle exceeds {args.max_mb:.0f} MB. The audit log is the\n"
            f"  usual cause; raise the controller tick_s or trim telemetry records\n"
            f"  rather than deleting decisions."
        )
    if not manifest["environment"]["git_commit"]:
        print("  WARNING: no git commit captured; run this from inside the repo")
    if manifest["environment"]["git_dirty"]:
        print("  WARNING: working tree is dirty; the commit alone will not reproduce")


if __name__ == "__main__":
    main()
