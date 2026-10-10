"""Verify live predictions against identical stored features on CPU."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(directory: Path, checkpoint: Path, expected_sha: str, tolerance=1e-5):
    import torch
    from vllm.bridge_tp.controller.distribution_predictor import DistributionPredictor

    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("checkpoint SHA differs")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    predictor = DistributionPredictor(
        checkpoint,
        checkpoint_sha256=expected_sha,
        model_config_sha256=ckpt["model_config_sha256"],
        device="cpu",
    )
    rows = []
    for file in sorted(directory.rglob("probe-*.pt")):
        saved = torch.load(file, map_location="cpu", weights_only=True)
        if saved["checkpoint_sha256"] != expected_sha:
            raise ValueError("probe checkpoint differs")
        if not torch.isfinite(saved["hidden_fp16"]).all():
            raise ValueError("nonfinite feature")
        replay = predictor.probabilities(
            saved["hidden_fp16"], saved["generated_tokens"]
        )
        online = saved["probabilities"]
        if (
            online.shape != replay.shape
            or not online.numel()
            or not torch.isfinite(online).all()
            or not torch.isfinite(replay).all()
        ):
            raise ValueError("invalid live probability geometry or values")
        difference = float((replay - online).abs().max())
        if difference > tolerance:
            raise ValueError(
                f"CPU/live probability difference {difference} > {tolerance}"
            )
        rows.append(
            {
                "file": str(file),
                "samples": len(online),
                "max_probability_difference": difference,
            }
        )
    if not rows:
        raise ValueError("no live feature probes")
    return {
        "status": "PASS",
        "checkpoint_sha256": expected_sha,
        "scope": "Same auxiliary feature CPU/live inference; does not prove auxiliary/offline-hook feature equivalence",
        "files": rows,
        "samples": sum(r["samples"] for r in rows),
        "tolerance": tolerance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.probe_dir, args.checkpoint, args.expected_sha256)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
