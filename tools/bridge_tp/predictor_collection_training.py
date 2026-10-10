"""Read audited collection shards without moving their source-tree splits."""

import hashlib
import json
from pathlib import Path

import numpy as np
from train_length_predictor import load_examples, read_jsonl, sha256_file


def load_collection_examples(
    batch: Path,
    *,
    expected_revision: str,
    expected_recipe_sha256: str,
    expected_feature_layer: str,
    require_complete: bool = True,
) -> tuple[dict, dict]:
    """Validate a collection and concatenate its completed, distinct requests.

    Args:
        batch: Original or retrieved collection directory.
        expected_revision: Immutable capture-code revision, not training HEAD.
        expected_recipe_sha256: Preregistered recipe SHA from the launcher.
        expected_feature_layer: Representation required by the parent model.
        require_complete: Require all planned requests for the primary batch;
            a successful partial collection can be used as historical replay.

    Returns:
        Audited examples and combined capture preflight with shard provenance.

    Raises:
        ValueError: A digest, identity, representation or tree split differs.
    """
    batch = Path(batch)
    inputs = batch / "inputs"
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (batch / "collection_summary.json").read_text(encoding="utf-8")
    )
    if (
        sha256_file(inputs / "recipe.jsonl") != expected_recipe_sha256
        or manifest["recipe_sha256"] != expected_recipe_sha256
    ):
        raise ValueError("collection recipe SHA differs")
    if manifest["rows"] != sum(s["rows"] for s in manifest["shards"]):
        raise ValueError("collection manifest request count differs")
    completed = summary["completed_shards"]
    by_name = {s["name"]: s for s in manifest["shards"]}
    if (
        not completed
        or len(completed) != len(set(completed))
        or not set(completed) <= set(by_name)
    ):
        raise ValueError("invalid completed shard list")
    if require_complete and set(completed) != set(by_name):
        raise ValueError("primary collection is not complete")
    parts = []
    preflights = []
    seen_ids = set()
    tree_splits = {}
    recipe_by_id = {r["id"]: r for r in read_jsonl(inputs / "recipe.jsonl")}
    if len(recipe_by_id) != manifest["rows"]:
        raise ValueError("duplicate or missing recipe request IDs")
    for name in completed:
        shard = by_name[name]
        source = inputs / shard["input"]
        if (
            not source.resolve().is_relative_to(inputs.resolve())
            or sha256_file(source) != shard["sha256"]
        ):
            raise ValueError("collection shard input SHA or path differs")
        out = batch / "captures" / name
        if not out.resolve().is_relative_to((batch / "captures").resolve()):
            raise ValueError("collection shard path escapes capture directory")
        meta = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
        if (
            meta["revision"] != expected_revision
            or meta["input_sha256"] != shard["sha256"]
            or meta.get("feature_layer") != expected_feature_layer
        ):
            raise ValueError("collection shard capture identity differs")
        if preflights:
            for key in ("model_config_sha256", "feature_layer", "feature_semantics"):
                if meta.get(key) != preflights[0].get(key):
                    raise ValueError(f"incompatible collection {key}")
        copied = out / "input_requests.jsonl"
        if copied.exists() and sha256_file(copied) != shard["sha256"]:
            raise ValueError("redundant capture input copy differs")
        source_by_id = {r["id"]: r for r in read_jsonl(source)}
        data = load_examples(out, include_censored=True)
        actual_ids = {r["input_id"] for r in data["labels"]}
        if (
            len(source_by_id) != shard["rows"]
            or len(data["labels"]) != shard["rows"]
            or actual_ids != set(source_by_id)
            or seen_ids & actual_ids
        ):
            raise ValueError("collection request identities differ or repeat")
        for label in data["labels"]:
            request = source_by_id[label["input_id"]]
            recipe = recipe_by_id.get(label["input_id"])
            if recipe is None:
                raise ValueError("capture request is absent from recipe")
            for key in ("split", "source_tree_id"):
                if label[key] != request[key] or request[key] != recipe[key]:
                    raise ValueError(f"collection request {key} differs")
            if (
                hashlib.sha256(request["prompt"].encode()).hexdigest()
                != label["prompt_sha256"]
                or request["planned_prompt_tokens"] != label["prompt_tokens"]
            ):
                raise ValueError("captured prompt differs from archived input")
            tree = label["source_tree_id"]
            if tree_splits.setdefault(tree, label["split"]) != label["split"]:
                raise ValueError("source tree crosses collection shards/splits")
        seen_ids.update(actual_ids)
        parts.append(data)
        preflights.append(meta)
    labels = [label for part in parts for label in part["labels"]]
    natural = sum(label["natural_finish"] for label in labels)
    if (
        summary["requests"] != len(labels)
        or summary["natural_eos"] != natural
        or summary["censored"] != len(labels) - natural
    ):
        raise ValueError("collection summary differs from audited shards")
    result = {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in (
            "hidden",
            "remaining",
            "generated",
            "splits",
            "requests",
            "phases",
            "languages",
            "censored",
        )
    }
    result["labels"] = labels
    result["audit"] = {
        "requests": len(labels),
        "censored_requests": len(labels) - natural,
        "samples": len(result["remaining"]),
        "capture_audits": [part["audit"] for part in parts],
    }
    preflight = dict(preflights[0])
    preflight["input_sha256"] = expected_recipe_sha256
    preflight["max_tokens"] = max(p["max_tokens"] for p in preflights)
    preflight["collection_provenance"] = {
        "batch": str(batch.resolve()),
        "manifest_sha256": sha256_file(inputs / "manifest.json"),
        "recipe_sha256": expected_recipe_sha256,
        "completed_shards": completed,
        "input_sha256_kind": "preregistered collection recipe",
        "capture_input_sha256_by_shard": {
            name: by_name[name]["sha256"] for name in completed
        },
    }
    return result, preflight
