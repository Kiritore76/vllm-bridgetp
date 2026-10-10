"""Initialize continued training without changing feature normalization."""

from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path
import numpy as np


def load_warmstart(path: Path, expected_sha: str, preflight: dict, width: int):
    """Validate immutable parent weights and representation metadata."""
    import torch

    if (
        not expected_sha
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha
    ):
        raise ValueError("warmstart SHA256 differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format_version") != 2
        or checkpoint.get("model_type") != "remaining_length_categorical"
    ):
        raise ValueError("unsupported warmstart checkpoint")
    for key in ["model_config_sha256", "feature_layer"]:
        if checkpoint.get(key) != preflight.get(key):
            raise ValueError(f"warmstart {key} differs")
    if checkpoint["input_width"] != width + 1 or not checkpoint.get(
        "overflow_category"
    ):
        raise ValueError("warmstart feature width or open tail differs")
    if checkpoint.get("feature_semantics") != preflight.get("feature_semantics"):
        raise ValueError("warmstart feature semantics differs")
    mean, std = checkpoint["feature_mean"], checkpoint["feature_std"]
    edges = checkpoint["category_upper_edges"]
    scale = float(checkpoint["position_log_scale"])
    if (
        mean.shape != (width,)
        or std.shape != mean.shape
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or not (std > 0).all()
        or not math.isfinite(scale)
        or scale <= 0
        or edges.ndim != 1
        or edges.numel() < 2
        or edges[0] != 0
        or not (edges[1:] > edges[:-1]).all()
    ):
        raise ValueError("invalid parent normalization or bucket geometry")
    return checkpoint


def expand_tail(checkpoint: dict, tail_edges: list[int]):
    """Split the open tail, preserving old calibrated probability sums."""
    import torch

    old = checkpoint["category_upper_edges"].cpu().numpy()
    if tail_edges and (
        any(x <= old[-1] for x in tail_edges)
        or any(a >= b for a, b in zip(tail_edges, tail_edges[1:]))
    ):
        raise ValueError("new tail edges must strictly increase beyond old edges")
    result = copy.deepcopy(checkpoint)
    edges = np.concatenate([old, np.asarray(tail_edges, dtype=np.int64)])
    state = result["state_dict"]
    w, b = state["3.weight"], state["3.bias"]
    if len(b) != len(old) + 1:
        raise ValueError("parent head shape differs")
    if tail_edges:
        pieces = len(tail_edges) + 1
        temperature = float(checkpoint["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("bad parent temperature")
        state["3.weight"] = torch.cat([w[:-1], w[-1:].repeat(pieces, 1)], dim=0)
        state["3.bias"] = torch.cat(
            [b[:-1], b[-1:].repeat(pieces) - temperature * math.log(pieces)], dim=0
        )
    result["category_upper_edges"] = torch.from_numpy(edges)
    return result


def merge_captures(primary: dict, replay: dict) -> dict:
    """Mix requests equally while keeping their original tree splits."""
    previous_trees = set()
    request_ids = set()
    for data in [primary, replay]:
        current_trees = {}
        for label in data["labels"]:
            tree = label["source_tree_id"]
            if tree in previous_trees:
                raise ValueError("capture request trees overlap")
            if tree in current_trees and current_trees[tree] != label["split"]:
                raise ValueError("request tree split leakage")
            current_trees[tree] = label["split"]
            if label["input_id"] in request_ids:
                raise ValueError("capture input IDs overlap")
            request_ids.add(label["input_id"])
        previous_trees.update(current_trees)
    result = {}
    for key in [
        "hidden",
        "remaining",
        "generated",
        "splits",
        "requests",
        "phases",
        "languages",
        "censored",
    ]:
        result[key] = np.concatenate([primary[key], replay[key]], axis=0)
    result["labels"] = primary["labels"] + replay["labels"]
    result["audit"] = {
        "requests": len(request_ids),
        "samples": len(result["remaining"]),
        "censored_requests": sum(
            not label["natural_finish"] for label in result["labels"]
        ),
        "capture_audits": [primary["audit"], replay["audit"]],
    }
    return result
