"""Audit three saved datasets and train the new 84-class predictor on one GPU."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from predictor_collection_training import load_collection_examples
from predictor_progress import emit
from predictor_training_mix import capture_coverage, mixed_training_weights
from predictor_warmstart import merge_captures
from train_length_predictor import load_examples, read_jsonl, sha256_file
from train_predictor_distribution import fit_distribution

NEW_REVISION = "a5e307cffbba302062962372c010878f0c88a747"
NEW_RECIPE = "52d3930183a1e308353267d406ce86e0986696f25631f76a495314fa40ba8341"
FIRST60_REVISION = "ff5ad1b40568f01962063f484f37e75a86b07542"
FIRST60_RECIPE = "2fb6094e74028ca0ab81381fd44ea45aeb2c8a2db4fc52b344a03d4046a6ab28"
LEGACY_REVISION = "2762813167b5e762edd2156e4dc08831aadcf0c3"
PARENT_SHA = "7d55bec981884ce50aa986693e0e2f687b7f1f4e2ae6f609a833ec50228f506f"
CONFIG_SHA = "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e"
LEGACY_DIGESTS = {
    "input_requests.jsonl": (
        "75cae22e298548b54b6164b3df9adc9ebdc61ced3a85eecfd7df9e96ea7a1be3"
    ),
    "labels.jsonl": "4e1ba33f5b18d1c460f158d0a6c587d3bdeb14614bd249430bafac9373e7ca53",
    "sample_index.jsonl": (
        "c40fd2a817c701a1566fb5cd886132d757d2f9d0dff92a86db61c52956228350"
    ),
    "features/features.sqlite3": (
        "e726adddeb8614117d6b4982eedad3a9e5d6dc217a930058f02c2dcd35bd5d91"
    ),
}


def resolve_collection(results: Path, explicit: str | None, recipe: str, count: int):
    """Select a unique completed local collection; never guess among batches."""
    candidates = [Path(explicit)] if explicit else sorted(results.glob("predictor-*"))
    matches = []
    for path in candidates:
        summary_path = path / "collection_summary.json"
        manifest_path = path / "inputs/manifest.json"
        if not summary_path.is_file() or not manifest_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if summary.get("requests") == count and manifest.get("recipe_sha256") == recipe:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"expected one {count}-request collection, found {matches}; "
            "set BRIDGETP_PREDICTOR_LONG300_BATCH / BRIDGETP_PREDICTOR_FIRST60_BATCH"
        )
    return matches[0]


def verify_legacy_files(path: Path) -> None:
    """Bind historical features and labels to the previously audited archive."""
    for name, expected in LEGACY_DIGESTS.items():
        if sha256_file(path / name) != expected:
            raise ValueError(f"legacy SHA differs: {name}")


def resolve_legacy(results: Path, out: Path, explicit: str | None) -> Path:
    """Reuse saved capture, or safely extract an uploaded immutable archive."""
    roots = (
        [Path(explicit)]
        if explicit
        else [
            results / "length_predictor/oasst1-long5000-v1-decoder31/capture-5000",
            results / "capture-5000",
            results / "capture-5000.tar.gz",
        ]
    )
    matches = [p for p in roots if p.exists()]
    if not matches:
        raise ValueError(
            "旧5000条数据不在这台服务器。请将本地 capture-5000.tar.gz 上传到 "
            f"{results / 'capture-5000.tar.gz'}，"
            "或设置 BRIDGETP_PREDICTOR_LEGACY_CAPTURE"
        )
    # Prefer an existing audited directory; an archive copy may coexist.
    source = next((p for p in matches if p.is_dir()), matches[0])
    if source.is_dir():
        verify_legacy_files(source)
        return source
    destination = out / "legacy-input"
    destination.mkdir()
    allowed = set(LEGACY_DIGESTS) | {
        "preflight.json",
        "summary.json",
        "input_manifest.json",
        "length_coverage.json",
    }
    seen = set()
    emit("正在解包旧5000条特征，并核对文件SHA")
    with tarfile.open(source, "r|gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            prefix = "capture-5000/"
            name = member.name.removeprefix(prefix)
            if (
                not member.name.startswith(prefix)
                or name not in allowed
                or name in seen
                or not member.isfile()
            ):
                raise ValueError(f"unexpected legacy archive member: {member.name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as original, target.open("xb") as saved:
                shutil.copyfileobj(original, saved)
            seen.add(name)
    if seen != allowed:
        raise ValueError("legacy archive is incomplete")
    verify_legacy_files(destination)
    return destination


def validate_legacy(data: dict, meta: dict, inputs: list[dict]) -> None:
    """Verify original 4000/500/500 identities without moving held-out rows."""
    if (
        meta["revision"] != LEGACY_REVISION
        or meta["input_sha256"] != LEGACY_DIGESTS["input_requests.jsonl"]
        or len(data["labels"]) != 5000
    ):
        raise ValueError("legacy capture revision or request count differs")
    expected = {r["id"]: (r["split"], r["source_tree_id"]) for r in inputs}
    actual = {r["input_id"]: (r["split"], r["source_tree_id"]) for r in data["labels"]}
    if actual != expected or len(expected) != 5000:
        raise ValueError("legacy input IDs or tree splits differ")
    counts = {
        split: sum(r[0] == split for r in actual.values())
        for split in ("train", "validation", "test")
    }
    if counts != {"train": 4000, "validation": 500, "test": 500}:
        raise ValueError("legacy original split differs")


def write_result_summary(out: Path, report: dict) -> None:
    """Save reviewable paired metrics; completion does not authorize deployment."""
    comparison = report["common_bucket_comparison"]
    lines = [
        "# 84桶混合续训结果", "",
        "训练完成，新权重仍为离线候选，尚未替换控制器。", "",
        f"最佳轮次：{report['best_epoch']}；温度：{report['temperature']:.2f}。",
        "训练权重：旧4000条25%，新300条及首60条的训练请求75%。",
        "自然EOS请求的早/中/后期权重30%/40%/30%；截断请求保留下界似然。",
        "只用新300条验证树选权重及校准；旧验证/测试与首60条仅作诊断。", "",
        "## 公共76桶的同样本对比", "",
        "NLL越低越好。下面是相同桶划分，不直接比较旧260类与新84类NLL。", "",
        "| 数据 | 父模型NLL | 新模型NLL | 新减旧 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in comparison["results"].items():
        old, new = row["parent_nll"], row["new_nll"]
        lines.append(f"| {name} | {old:.4f} | {new:.4f} | {new - old:+.4f} |")
    for name in (
        "selection_validation", "test_new_stage", "legacy5000_test_diagnostic"
    ):
        rows = comparison["progress_by_split"].get(name, {})
        lines += ["", f"## {name} 按实际生成进度", "",
                  "| 进度 | 请求数 | 父公共NLL | 新公共NLL | 新90%覆盖 |",
                  "|---|---:|---:|---:|---:|"]
        for stage, row in rows.items():
            lines.append(
                f"| {stage} | {row['requests']} | {row['parent_common_nll']:.4f} | "
                f"{row['new_common_nll']:.4f} | "
                f"{row['new_length']['interval90_coverage']:.1%} |"
            )
    lines += ["", "完整真实桶概率、区间宽度、开放尾占比及长度误差见train/report.json。",
              "开放尾区间不裁剪成16384，有限宽度和误差只在有限区间上统计。",
              "查看覆盖率时须同时查看宽度，覆盖提升不等于长度预测更集中。", ""]
    (out / "TRAINING_RESULT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        revision != args.expected_revision
        or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    ):
        parser.error("training HEAD differs or checkout is dirty")
    if sha256_file(args.checkpoint) != PARENT_SHA:
        parser.error("parent checkpoint SHA differs")
    if sha256_file(args.model / "config.json") != CONFIG_SHA:
        parser.error("model config SHA differs")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total", "--format=csv"],
        text=True,
    )
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(names) != args.expected_gpu_count or any(
        name.strip() != "NVIDIA A100-PCIE-40GB" for name in names
    ):
        parser.error("physical GPU model/count differs")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("select exactly one visible CUDA device for training")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "train").exists():
        parser.error("training output must be new")
    emit("正在定位新增300条、已有60条和旧5000条")
    primary_path = resolve_collection(
        args.results,
        os.environ.get("BRIDGETP_PREDICTOR_LONG300_BATCH"),
        NEW_RECIPE,
        300,
    )
    replay_path = resolve_collection(
        args.results,
        os.environ.get("BRIDGETP_PREDICTOR_FIRST60_BATCH"),
        FIRST60_RECIPE,
        60,
    )
    legacy_path = resolve_legacy(
        args.results, args.out_dir, os.environ.get("BRIDGETP_PREDICTOR_LEGACY_CAPTURE")
    )
    parts, metas, sources = [], [], {}
    for name, path, capture_revision, recipe, complete in (
        ("new300", primary_path, NEW_REVISION, NEW_RECIPE, True),
        ("first60", replay_path, FIRST60_REVISION, FIRST60_RECIPE, False),
    ):
        emit(f"正在审计并读取{name}的真实标签和hidden特征")
        data, meta = load_collection_examples(
            path,
            expected_revision=capture_revision,
            expected_recipe_sha256=recipe,
            expected_feature_layer="decoder:31",
            require_complete=complete,
        )
        parts.append(data)
        metas.append(meta)
        sources[name] = {
            "path": str(path.resolve()),
            "coverage": capture_coverage(data),
            "provenance": meta["collection_provenance"],
        }
    emit("正在审计并读取旧5000条，仅原4000条训练请求参与梯度")
    old_meta = json.loads((legacy_path / "preflight.json").read_text())
    old_inputs = read_jsonl(legacy_path / "input_requests.jsonl")
    old = load_examples(legacy_path, include_censored=True)
    validate_legacy(old, old_meta, old_inputs)
    parts.append(old)
    metas.append(old_meta)
    sources["legacy5000"] = {
        "path": str(legacy_path.resolve()),
        "coverage": capture_coverage(old),
        "sha256": LEGACY_DIGESTS,
    }
    for meta in metas:
        for key in ("model_config_sha256", "feature_layer", "feature_semantics"):
            if meta.get(key) != metas[0].get(key):
                raise ValueError(f"incompatible replay {key}")
    data = merge_captures(merge_captures(parts[0], parts[1]), parts[2])
    legacy_ids = set(old["requests"])
    validation_ids = set(parts[0]["requests"])
    weights, weighting = mixed_training_weights(data, legacy_ids)
    groups = {}
    for name, part in zip(("new300", "first60", "legacy5000"), parts):
        ids = set(part["requests"])
        for split in ("validation", "test"):
            groups[f"{name}_{split}_diagnostic"] = np.isin(
                data["requests"], sorted(ids)
            ) & (data["splits"] == split)
    known = set(parts[1]["requests"][parts[1]["splits"] == "test"])
    known |= set(old["requests"][old["splits"] == "test"])
    provenance = {
        "training_revision": revision,
        "hostname": socket.gethostname(),
        "gpu_inventory_uuid_record_only": inventory,
        "sources": sources,
        "weighting": weighting,
        "parent_sha256": PARENT_SHA,
        "validation_policy": "new300 validation trees only for selection/calibration",
        "test_policy": "all held-out requests remain held out; legacy diagnostic only",
        "qwen_training": False,
        "deployment": "offline candidate only",
    }
    (args.out_dir / "training_preflight.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    emit(f"数据审计完成：{weighting['cohorts']}；共{len(data['remaining'])}个状态")
    report = fit_distribution(
        data,
        metas[0],
        args.out_dir / "train",
        revision=revision,
        epochs=5,
        batch_size=256,
        learning_rate=5e-5,
        dropout=0.1,
        weight_decay=0.01,
        patience=2,
        seed=42,
        warmstart_checkpoint=args.checkpoint,
        warmstart_sha256=PARENT_SHA,
        bucket_profile="nonuniform84",
        head_warmup_epochs=1,
        head_learning_rate=3e-4,
        training_weights=weights,
        weighting_provenance=weighting,
        validation_ids=validation_ids,
        diagnostic_groups=groups,
        known_test_ids=known,
    )
    if sha256_file(args.checkpoint) != PARENT_SHA:
        raise ValueError("immutable parent changed during training")
    write_result_summary(args.out_dir, report)
    (args.out_dir / "status.json").write_text(
        json.dumps(
            {
                "completed": True,
                "best_epoch": report["best_epoch"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "deployed": False,
        "model_status": "CANDIDATE_REQUIRES_REVIEW",
            },
            indent=2,
        )
        + "\n"
    )
    emit("84桶续训完成，请取回完整训练包；新权重尚未部署")


if __name__ == "__main__":
    main()
