# SPDX-License-Identifier: Apache-2.0
"""Reuse audited saved captures for a four-layer scalar/classifier comparison."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from predictor_collection_training import load_collection_examples
from predictor_distribution import risk76_upper_edges
from predictor_length_regression import (
    LIMIT,
    accuracy,
    categorical_points,
    features,
    fit,
    targets,
)
from predictor_network import build_mlp
from predictor_progress import emit
from predictor_training_mix import capture_coverage, mixed_training_weights
from predictor_warmstart import merge_captures
from run_predictor_nonuniform84_training import (
    FIRST60_RECIPE,
    FIRST60_REVISION,
    LEGACY_DIGESTS,
    NEW_RECIPE,
    NEW_REVISION,
    resolve_collection,
    validate_legacy,
    verify_legacy_files,
)
from train_length_predictor import load_examples, read_jsonl, sha256_file

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "bridgetp/pressure-balanced-controller"
REFERENCE_SHA = "dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7"
REFERENCE_DIR = "predictor-risk76-mlp2-mlp4-v1-20261011T093049Z-1353/train-mlp4-50pct"


def write(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_data(results, out):
    """Use the original three captures and unchanged source-tree splits."""
    old_path = Path(
        os.environ.get(
            "BRIDGETP_PREDICTOR_LEGACY_CAPTURE",
            results / "length_predictor/oasst1-long5000-v1-decoder31/capture-5000",
        )
    )
    if not old_path.is_dir():
        raise FileNotFoundError(
            f"旧特征目录不存在：{old_path}；可设置 BRIDGETP_PREDICTOR_LEGACY_CAPTURE"
        )
    parts, metas, cohorts, sources = [], [], {}, {}
    for name, env, recipe, revision, count, complete in (
        (
            "new300",
            "BRIDGETP_PREDICTOR_LONG300_BATCH",
            NEW_RECIPE,
            NEW_REVISION,
            300,
            True,
        ),
        (
            "first60",
            "BRIDGETP_PREDICTOR_FIRST60_BATCH",
            FIRST60_RECIPE,
            FIRST60_REVISION,
            60,
            False,
        ),
    ):
        path = resolve_collection(results, os.environ.get(env), recipe, count)
        emit(f"审计保存的{name}特征：{path}")
        data, meta = load_collection_examples(
            path,
            expected_revision=revision,
            expected_recipe_sha256=recipe,
            expected_feature_layer="decoder:31",
            require_complete=complete,
        )
        hashes = {}
        for shard in meta["collection_provenance"]["completed_shards"]:
            for relative in (
                "features/features.sqlite3",
                "labels.jsonl",
                "sample_index.jsonl",
            ):
                source = path / "captures" / shard / relative
                hashes[f"captures/{shard}/{relative}"] = sha256_file(source)
        parts.append(data)
        metas.append(meta)
        cohorts[name] = set(data["requests"])
        sources[name] = dict(
            path=str(path.resolve()),
            coverage=capture_coverage(data),
            sha256=hashes,
            provenance=meta["collection_provenance"],
        )
    emit(f"审计旧5000条特征：{old_path}")
    verify_legacy_files(old_path)
    old = load_examples(old_path, include_censored=True)
    old_meta = json.loads((old_path / "preflight.json").read_text(encoding="utf-8"))
    validate_legacy(old, old_meta, read_jsonl(old_path / "input_requests.jsonl"))
    parts.append(old)
    metas.append(old_meta)
    cohorts["legacy5000"] = set(old["requests"])
    sources["legacy5000"] = dict(
        path=str(old_path.resolve()),
        coverage=capture_coverage(old),
        sha256=LEGACY_DIGESTS,
    )
    for meta in metas:
        for key in ("model_config_sha256", "feature_layer", "feature_semantics"):
            if meta[key] != metas[0][key]:
                raise ValueError(f"incompatible feature metadata: {key}")
    data = merge_captures(merge_captures(parts[0], parts[1]), parts[2])
    weights, weighting = mixed_training_weights(
        data, cohorts["legacy5000"], legacy_mass=0.5, prefill_mass=0.2
    )
    write(
        out / "data_audit.json",
        dict(
            sources=sources,
            audit=data["audit"],
            weighting=weighting,
            split_policy="original request/tree splits; no random state split",
            censored_rule=(
                "lower-bound Huber; exact bounded target only if lower bound >=8192"
            ),
        ),
    )
    return data, metas[0], weights, cohorts, weighting


def load_reference(path, expected_sha, meta, feature_width):
    import torch

    if sha256_file(path) != expected_sha:
        raise ValueError("frozen classifier SHA differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved.get("model_type") != "remaining_length_categorical_mlp4"
        or saved.get("network_depth") != 4
        or saved.get("input_width") != feature_width + 1
        or saved.get("hidden_width") != 256
        or not np.array_equal(
            saved["category_upper_edges"].numpy(), risk76_upper_edges()
        )
    ):
        raise ValueError("require the current frozen four-layer risk76 checkpoint")
    for key in ("model_config_sha256", "feature_layer", "feature_semantics"):
        if saved.get(key) != meta[key]:
            raise ValueError(f"classifier representation differs: {key}")
    for name in ("feature_mean", "feature_std"):
        value = saved[name]
        if value.shape != (feature_width,) or not torch.isfinite(value).all():
            raise ValueError("invalid classifier normalization")
    if (
        not (saved["feature_std"] > 0).all()
        or not np.isfinite(saved["temperature"])
        or saved["temperature"] <= 0
    ):
        raise ValueError("invalid classifier scale or temperature")
    return saved


def baseline_predictions(data, checkpoint, device, batch_size=256):
    import torch

    model = build_mlp(
        checkpoint["input_width"],
        checkpoint["hidden_width"],
        76,
        checkpoint["dropout"],
        4,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    mean = np.empty(len(data["remaining"]), dtype=np.float32)
    median = mean.copy()
    with torch.no_grad():
        for start in range(0, len(mean), batch_size):
            rows = np.arange(start, min(start + batch_size, len(mean)))
            logits = (
                model(features(data, rows, checkpoint, device))
                / checkpoint["temperature"]
            )
            p = torch.softmax(logits, dim=1).cpu().numpy()
            mean[rows], median[rows] = categorical_points(
                p, checkpoint["category_upper_edges"].numpy()
            )
    return mean, median


def comparisons(data, cohorts, predictions):
    """Every model uses exactly the same exact-EOS evaluation states."""
    bounded, _ = targets(data)
    total = data["generated"] + data["remaining"]
    fraction = np.divide(
        data["generated"], total, out=np.zeros(len(total)), where=total > 0
    )
    decode = data["phases"] != "PREFILL_COMPLETE"
    phases = {
        "all": np.ones(len(total), dtype=bool),
        "prefill": ~decode,
        "decode_0_25": decode & (fraction < 0.25),
        "decode_25_50": decode & (fraction >= 0.25) & (fraction < 0.5),
        "decode_50_75": decode & (fraction >= 0.5) & (fraction < 0.75),
        "decode_75_100": decode & (fraction >= 0.75),
    }
    lengths = {
        "all_bounded": total >= 0,
        "output_0_1024": total <= 1024,
        "output_1025_2048": (total > 1024) & (total <= 2048),
        "output_2049_4096": (total > 2048) & (total <= 4096),
        "output_4097_8192": (total > 4096) & (total <= 8192),
        "output_1025_8192": (total > 1024) & (total <= 8192),
        "output_over8192_bounded_task_only": total > 8192,
    }
    result = {}
    for cohort, ids in cohorts.items():
        for split in ("validation", "test"):
            base = (
                np.isin(data["requests"], sorted(ids))
                & (data["splits"] == split)
                & ~data["censored"]
            )
            for length, length_mask in lengths.items():
                for stage, phase in phases.items():
                    mask = base & length_mask & phase
                    if mask.any():
                        key = f"{cohort}/{split}/{length}/{stage}"
                        result[key] = {
                            name: accuracy(
                                bounded[mask], values[mask], data["requests"][mask]
                            )
                            for name, values in predictions.items()
                        }
    return result


def report_markdown(report):
    lines = [
        "# 四层 MLP：剩余长度回归与现有分类版对照",
        "",
        "本轮只训练小预测器，复用原保存特征；Qwen 和在线控制器未修改。",
        "",
        "结构：5121 → 256 → 512 → 256 → 1。随机初始化，直接回归剩余 token。",
        "主任务：min(自然剩余输出,8192)，不是保证逐 token 精确命中。",
        "训练沿用旧/新增各50%、prefill20%、decode阶段30/40/30；截断样本只用下界。",
        "仅原验证selection半集按旧/新各50%的请求等权MAE选轮次；其余验证半集保留，测试只评估。",
        "分类版用原温度与归一化，分别比较桶均值和中位桶；开放尾质量保留在8192。",
        "",
        "| 测试范围 | 回归MAE | 分类桶均值MAE | 分类中位桶MAE |",
        "|---|---:|---:|---:|",
    ]
    for cohort in ("new300", "legacy5000"):
        for length in ("all_bounded", "output_1025_8192"):
            for stage in (
                "all",
                "prefill",
                "decode_0_25",
                "decode_25_50",
                "decode_50_75",
                "decode_75_100",
            ):
                key = f"{cohort}/test/{length}/{stage}"
                row = report["comparison"].get(key)
                if row:
                    values = [
                        row[name]["request_balanced_mae_tokens"]
                        for name in (
                            "regression",
                            "classifier_mean",
                            "classifier_median",
                        )
                    ]
                    lines.append(
                        f"| {cohort}/{length}/{stage} | "
                        + " | ".join(f"{v:.1f}" for v in values)
                        + " |"
                    )
    lines += [
        "",
        (
            "完整长度分组、偏差、误差≤128/256/512/1024比例、逐请求预测及训练记录"
            "见 report.json / predictions.npz。"
        ),
        "已有测试曾被查看过，本轮是开发对照，不是新的盲测；相同数据重复不能证明泛化。",
        "点预测不提供校准概率；取得结果后再决定是否替换，不自动撤销现有生存概率安全检查。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if (
        revision != args.expected_revision
        or branch != BRANCH
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    ):
        parser.error("checkout revision/branch differs or there is uncommitted work")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if (
        not 0 <= args.gpu_index < len(names)
        or names[args.gpu_index].strip() != "NVIDIA A100-PCIE-40GB"
    ):
        parser.error("selected GPU must be A100-PCIE-40GB")
    if subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={args.gpu_index}",
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip():
        parser.error("selected GPU is busy; stop here without terminating processes")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ["OMP_NUM_THREADS"] = "4"
    import torch

    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    reference = (
        args.reference_checkpoint
        or args.results / REFERENCE_DIR / "predictor_distribution.pt"
    )
    if sha256_file(reference) != REFERENCE_SHA:
        raise ValueError("reference classifier differs")
    write(
        args.out_dir / "run_identity.json",
        dict(
            revision=revision,
            branch=branch,
            reference=str(reference),
            reference_sha256=REFERENCE_SHA,
            selected_gpu=args.gpu_index,
            gpu_inventory=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total",
                    "--format=csv",
                ],
                text=True,
            ),
            torch_version=str(torch.__version__),
            device="cuda:0",
            deployment="offline only",
        ),
    )
    emit("运行回归损失、划分隔离及checkpoint重载CPU自检")
    with (args.out_dir / "unit_tests.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                os.sys.executable,
                "-B",
                "-m",
                "unittest",
                "tests.bridge_tp.test_predictor_length_regression",
                "tests.bridge_tp.test_predictor_training_mix",
                "tests.bridge_tp.test_predictor_collection_training",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    emit("CPU自检通过；开始读取保存的数据，详细记录见 unit_tests.log")
    data, meta, weights, cohorts, weighting = load_data(args.results, args.out_dir)
    reference_model = load_reference(
        reference, REFERENCE_SHA, meta, data["hidden"].shape[1]
    )
    emit(
        f"数据审计完成：{len(data['labels'])}条请求，{len(data['remaining'])}个状态；开始四层回归训练"
    )
    raw_prediction, training = fit(
        data,
        meta,
        args.out_dir / "train",
        weights,
        {"legacy5000": cohorts["legacy5000"], "new300": cohorts["new300"]},
        revision=revision,
    )
    emit("最佳checkpoint已严格重载；在相同状态上计算现有分类版均值/中位数")
    mean, median = baseline_predictions(data, reference_model, "cuda:0")
    predictions = dict(
        regression=raw_prediction.clip(0, LIMIT),
        classifier_mean=mean,
        classifier_median=median,
    )
    bounded, exact = targets(data)
    report = dict(
        status="COMPLETED_OFFLINE_COMPARISON",
        deployment="unchanged",
        target="min(natural remaining tokens,8192); output>8192 is not finely modeled",
        reference_sha256=REFERENCE_SHA,
        regression_sha256=sha256_file(args.out_dir / "train/predictor_regression.pt"),
        training=training,
        weighting=weighting,
        comparison=comparisons(data, cohorts, predictions),
        censored=dict(
            samples=int(data["censored"].sum()),
            bounded_target_known_samples=int((data["censored"] & exact).sum()),
        ),
        caveat=(
            "previously inspected test sets; development comparison, "
            "not a fresh blind test"
        ),
    )
    np.savez_compressed(
        args.out_dir / "predictions.npz",
        request_ids=data["requests"],
        splits=data["splits"],
        phases=data["phases"],
        generated=data["generated"],
        raw_remaining_or_lower_bound=data["remaining"],
        bounded_target=bounded,
        censored=data["censored"],
        regression_raw=raw_prediction,
        **predictions,
    )
    write(args.out_dir / "report.json", report)
    (args.out_dir / "TRAINING_RESULT_CN.md").write_text(
        report_markdown(report), encoding="utf-8"
    )
    if sha256_file(reference) != REFERENCE_SHA:
        raise ValueError("read-only classifier changed during training")
    emit("训练及同样本对照完成；在线预测器保持原样，准备打包")


if __name__ == "__main__":
    main()
